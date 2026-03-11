"""
ar_glow.py — AR GLOW: GRU/LSTM context + conditional GLOW flow

Architecture:
    RNN:    window(B,W,S) → h(B,rnn_hidden)
    Flow:   n_steps blocks, each:
              ActNorm:     data-driven scale+shift (no conditioning)
              InvLinearLU: invertible 1×1 linear via LU (no conditioning)
              CondCoupling: affine coupling conditioned on h
    Loss:   exact NLL
    Sample: z ~ N(0,I) → reverse blocks conditioned on h
"""

import math
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel
from models.ar_real_nvp import _make_rnn, _extract_h
from models.conditioning import ConditioningBlock, DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS


class _ActNorm(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        self.loc   = nn.Parameter(torch.zeros(n))
        self.scale = nn.Parameter(torch.ones(n))
        self.initialized = False

    def _init(self, x):
        with torch.no_grad():
            self.loc.data   = -x.mean(0)
            self.scale.data = 1.0 / (x.std(0) + 1e-6)
        self.initialized = True

    def forward(self, x):
        if not self.initialized:
            self._init(x)
        y = (x + self.loc) * self.scale
        log_det = self.scale.abs().log().sum().expand(x.shape[0])
        return y, log_det

    def inverse(self, y):
        return y / self.scale - self.loc


class _InvLinearLU(nn.Module):
    def __init__(self, n: int):
        super().__init__()
        W = torch.linalg.qr(torch.randn(n, n))[0]
        P, L, U = torch.linalg.lu(W)
        self.register_buffer('P', P)
        self.L = nn.Parameter(L)
        self.U = nn.Parameter(U)
        self.n = n
        self._w_inv_cache = None  # cached during eval; cleared on train()

    def train(self, mode: bool = True):
        if mode:
            self._w_inv_cache = None  # invalidate cache when re-entering training
        return super().train(mode)

    def _W(self):
        L = torch.tril(self.L, diagonal=-1) + torch.eye(self.n, device=self.L.device)
        return self.P @ L @ torch.triu(self.U)

    def forward(self, x):
        W = self._W()
        log_det = torch.diagonal(self.U).abs().log().sum().expand(x.shape[0])
        return x @ W.T, log_det

    def inverse(self, y):
        if self._w_inv_cache is None:
            self._w_inv_cache = torch.linalg.inv(self._W())
        return y @ self._w_inv_cache.T


class _CondAffineCoupling(nn.Module):
    """GLOW coupling conditioned on h. Splits: x1 = x[:n_half], x2 = x[n_half:]."""

    def __init__(self, input_size: int, rnn_hidden: int, cond_dim: int, hidden: int = 128):
        super().__init__()
        n_half = input_size // 2
        n_free = input_size - n_half
        self.n_half = n_half
        self.n_free = n_free
        self.net = nn.Sequential(
            nn.Linear(n_half + rnn_hidden + cond_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_free * 2),
        )

    def forward(self, x, h):
        x1, x2 = x[:, :self.n_half], x[:, self.n_half:]
        st = self.net(torch.cat([x1, h], dim=-1))
        log_s, t = st[:, :self.n_free], st[:, self.n_free:]
        log_s = torch.tanh(log_s)
        return torch.cat([x1, x2 * torch.exp(log_s) + t], dim=-1), log_s.sum(dim=-1)

    def inverse(self, y, h):
        y1, y2 = y[:, :self.n_half], y[:, self.n_half:]
        st = self.net(torch.cat([y1, h], dim=-1))
        log_s, t = st[:, :self.n_free], st[:, self.n_free:]
        log_s = torch.tanh(log_s)
        return torch.cat([y1, (y2 - t) * torch.exp(-log_s)], dim=-1)


class ARGlow(BaseModel):
    def __init__(self, input_size=90, window_size=30, rnn_hidden=128,
                 n_steps=8, hidden_size=128, rnn_type='gru', **kwargs):
        super().__init__()
        self.n_stations  = input_size
        self.window_size = window_size
        self.rnn_type    = rnn_type
        self.cond_block  = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        self.cond_dim    = self.cond_block.total_dim
        self.rnn         = _make_rnn(rnn_type, input_size, rnn_hidden)
        self.act_norms   = nn.ModuleList([_ActNorm(input_size)           for _ in range(n_steps)])
        self.inv_linears = nn.ModuleList([_InvLinearLU(input_size)       for _ in range(n_steps)])
        self.couplings   = nn.ModuleList([_CondAffineCoupling(input_size, rnn_hidden, self.cond_dim, hidden_size)
                                          for _ in range(n_steps)])

    def _encode_window(self, window):
        return _extract_h(self.rnn(window), self.rnn_type)

    def _cond_embed(self, cond: dict | None, batch_size: int, device: torch.device) -> Tensor:
        if cond is None:
            return torch.zeros(batch_size, self.cond_dim, device=device)
        return self.cond_block(cond)

    def _make_day_cond(self, doy: int, batch_size: int, device: torch.device) -> dict:
        """Build conditioning dict for a given day-of-year (1–366)."""
        angle = 2.0 * math.pi * doy / 365.25
        month_idx = int((doy - 1) * 12 / 365) % 12
        return {
            'month':   torch.full((batch_size,), month_idx, dtype=torch.long,  device=device),
            'day_sin': torch.full((batch_size,), math.sin(angle),               device=device),
            'day_cos': torch.full((batch_size,), math.cos(angle),               device=device),
        }

    def _fwd(self, x, h_cond):
        z, ld = x, torch.zeros(x.shape[0], device=x.device)
        for an, il, cp in zip(self.act_norms, self.inv_linears, self.couplings):
            z, d = an(z);    ld = ld + d
            z, d = il(z);    ld = ld + d
            z, d = cp(z, h_cond); ld = ld + d
        return z, ld

    def _inv(self, z, h_cond):
        x = z
        for an, il, cp in zip(reversed(self.act_norms), reversed(self.inv_linears), reversed(self.couplings)):
            x = cp.inverse(x, h_cond)
            x = il.inverse(x)
            x = an.inverse(x)
        return x

    def log_prob(self, x, h_cond):
        z, ld = self._fwd(x, h_cond)
        return -0.5 * (z**2 + np.log(2 * np.pi)).sum(-1) + ld

    def loss(self, x, beta=1.0):
        if len(x) == 3:
            window, target, cond = x
        else:
            window, target = x
            cond = None
        h   = self._encode_window(window)
        cond_emb = self._cond_embed(cond, target.shape[0], target.device)
        h_cond = torch.cat([h, cond_emb], dim=-1)
        nll = -self.log_prob(target, h_cond).mean()
        return {'total': nll, 'nll': nll}

    def _flow_sample(self, h_cond, n):
        z = torch.randn(n, self.n_stations, device=h_cond.device)
        return self._inv(z, h_cond).clamp(min=0.0)

    @torch.no_grad()
    def sample(self, n, steps=None, method=None, start_doy: int = 1):
        device = next(self.parameters()).device
        window = torch.zeros(1, self.window_size, self.n_stations, device=device)
        for i in range(self.window_size):
            doy = (start_doy - self.window_size + i - 1) % 365 + 1
            cond = self._make_day_cond(doy, 1, device)
            h = self._encode_window(window)
            cond_emb = self._cond_embed(cond, 1, device)
            h_cond = torch.cat([h, cond_emb], dim=-1)
            y = self._flow_sample(h_cond, 1)
            window = torch.cat([window[:, 1:], y.unsqueeze(1)], dim=1)
        samples = []
        for i in range(n):
            doy = (start_doy + i - 1) % 365 + 1
            cond = self._make_day_cond(doy, 1, device)
            h = self._encode_window(window)
            cond_emb = self._cond_embed(cond, 1, device)
            h_cond = torch.cat([h, cond_emb], dim=-1)
            y = self._flow_sample(h_cond, 1)
            window = torch.cat([window[:, 1:], y.unsqueeze(1)], dim=1)
            samples.append(y)
        return torch.cat(samples, dim=0)

    @torch.no_grad()
    def sample_rollout(self, seed_window, n_days, n_scenarios=10, start_doy: int = 1):
        device = next(self.parameters()).device
        window = seed_window.to(device).unsqueeze(0).expand(n_scenarios, -1, -1).clone()
        days = []
        for i in range(n_days):
            doy = (start_doy + i - 1) % 365 + 1
            cond = self._make_day_cond(doy, n_scenarios, device)
            h = self._encode_window(window)
            cond_emb = self._cond_embed(cond, n_scenarios, device)
            h_cond = torch.cat([h, cond_emb], dim=-1)
            y = self._flow_sample(h_cond, n_scenarios)
            window = torch.cat([window[:, 1:], y.unsqueeze(1)], dim=1)
            days.append(y)
        return torch.stack(days, dim=1)
