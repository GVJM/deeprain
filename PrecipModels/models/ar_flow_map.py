"""
ar_flow_map.py — AR FlowMap: GRU/LSTM context + ODE solution operator

Reference: arXiv 2505.18825 (Boffi, Albergo, Vanden-Eijnden — NeurIPS 2025)

Network: Phi_theta(z_s, s_emb, t_emb, h) → z_t   [position, not velocity]
         Input: [z_s(S) || s_emb(T) || t_emb(T) || h(H)]

Training on OT paths:
    z_0 ~ N(0,I), x = target
    s, t ~ U(0,1)  (independently, no ordering constraint)
    z_s = (1-s)*z_0 + s*x   [point on OT path at time s]
    z_t = (1-t)*z_0 + t*x   [point on OT path at time t]
    Loss: MSE(Phi_theta(z_s, s_emb, t_emb, h), z_t)

    For OT straight paths, the exact flow map IS z_t — so this is the ground truth.
    Training on all (s,t) pairs teaches the model to jump between any two path times.

Sampling: z_0 ~ N(0,I) [pure noise, t=0 on OT scale].
          Euler steps from s=0 to t=1 via self.flow_map.
          n_steps=1 gives single-step prediction;
          n_steps>1 gives multi-step ODE (better diversity).
"""

import math
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel
from models.flow_match import SinusoidalEmbedding
from models.ar_real_nvp import _make_rnn, _extract_h
from models.conditioning import ConditioningBlock, DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS


class _FlowMapMLP(nn.Module):
    """Phi_theta: [z_s(S) || s_emb(T) || t_emb(T) || h(H)] → z_t(S)"""
    def __init__(self, data_dim, t_embed_dim, rnn_hidden, cond_dim, hidden=256, n_layers=4):
        super().__init__()
        in_dim = data_dim + 2 * t_embed_dim + rnn_hidden + cond_dim
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_s, s_emb, t_emb, h):
        return self.net(torch.cat([z_s, s_emb, t_emb, h], dim=-1))


class ARFlowMap(BaseModel):
    def __init__(self, input_size=90, window_size=30, rnn_hidden=128,
                 hidden_size=256, n_layers=4, t_embed_dim=64,
                 rnn_type='gru', n_steps: int = 1, **kwargs):
        super().__init__()
        self.n_stations  = input_size
        self.window_size = window_size
        self.rnn_type    = rnn_type
        self.n_steps     = n_steps
        self.cond_block  = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        self.cond_dim    = self.cond_block.total_dim
        self.rnn         = _make_rnn(rnn_type, input_size, rnn_hidden)
        self.t_embed     = SinusoidalEmbedding(t_embed_dim)
        self.flow_map    = _FlowMapMLP(input_size, t_embed_dim, rnn_hidden, self.cond_dim, hidden_size, n_layers)

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

    def loss(self, x, beta=1.0):
        """MSE regression: Phi_theta(z_s, s, t, h) → z_t on OT paths."""
        if len(x) == 3:
            window, target, cond = x
        else:
            window, target = x
            cond = None
        B = target.shape[0]
        device = target.device

        h    = self._encode_window(window)
        cond_emb = self._cond_embed(cond, B, device)
        h_cond = torch.cat([h, cond_emb], dim=-1)
        z_0  = torch.randn_like(target)
        s    = torch.rand(B, device=device)
        t    = torch.rand(B, device=device)
        z_s  = (1 - s.unsqueeze(-1)) * z_0 + s.unsqueeze(-1) * target
        z_t  = (1 - t.unsqueeze(-1)) * z_0 + t.unsqueeze(-1) * target
        s_emb = self.t_embed(s)
        t_emb = self.t_embed(t)

        z_t_pred = self.flow_map(z_s, s_emb, t_emb, h_cond)
        fm_loss  = F.mse_loss(z_t_pred, z_t)
        return {'total': fm_loss, 'fm_loss': fm_loss}

    def _generate_sample(self, h_cond, n):
        """n_steps Euler steps from noise(t=0) to data(t=1).
        Correct OT direction: z_0 ~ N(0,I) at t=0 (noise side),
        iteratively mapped to t=1 (data side) via self.flow_map.
        """
        device = h_cond.device
        z = torch.randn(n, self.n_stations, device=device)
        ds = 1.0 / self.n_steps
        for i in range(self.n_steps):
            s_val = i * ds
            t_val = s_val + ds
            s_emb = self.t_embed(torch.full((n,), s_val, device=device))
            t_emb = self.t_embed(torch.full((n,), t_val, device=device))
            z = self.flow_map(z, s_emb, t_emb, h_cond)
        return z.clamp(min=0.0)

    @torch.no_grad()
    def sample(self, n, steps=None, method=None, start_doy: int = 1):
        _n_steps_orig = self.n_steps
        if steps is not None:
            self.n_steps = steps
        device = next(self.parameters()).device
        window = torch.zeros(1, self.window_size, self.n_stations, device=device)
        for i in range(self.window_size):
            doy = (start_doy - self.window_size + i - 1) % 365 + 1
            cond = self._make_day_cond(doy, 1, device)
            h = self._encode_window(window)
            cond_emb = self._cond_embed(cond, 1, device)
            h_cond = torch.cat([h, cond_emb], dim=-1)
            y = self._generate_sample(h_cond, 1)
            window = torch.cat([window[:, 1:], y.unsqueeze(1)], dim=1)
        samples = []
        log_every = max(1, n // 4)
        for i in range(n):
            if i > 0 and i % log_every == 0:
                print(f"  [ar_flow_map] sampling step {i}/{n}...", flush=True)
            doy = (start_doy + i - 1) % 365 + 1
            cond = self._make_day_cond(doy, 1, device)
            h = self._encode_window(window)
            cond_emb = self._cond_embed(cond, 1, device)
            h_cond = torch.cat([h, cond_emb], dim=-1)
            y = self._generate_sample(h_cond, 1)
            window = torch.cat([window[:, 1:], y.unsqueeze(1)], dim=1)
            samples.append(y)
        self.n_steps = _n_steps_orig
        return torch.cat(samples, dim=0)

    @torch.no_grad()
    def sample_rollout(self, seed_window, n_days, n_scenarios=10, start_doy: int = 1, steps=None):
        _n_steps_orig = self.n_steps
        if steps is not None:
            self.n_steps = steps
        device = next(self.parameters()).device
        window = seed_window.to(device).unsqueeze(0).expand(n_scenarios, -1, -1).clone()
        days = []
        for i in range(n_days):
            doy = (start_doy + i - 1) % 365 + 1
            cond = self._make_day_cond(doy, n_scenarios, device)
            h = self._encode_window(window)
            cond_emb = self._cond_embed(cond, n_scenarios, device)
            h_cond = torch.cat([h, cond_emb], dim=-1)
            y = self._generate_sample(h_cond, n_scenarios)
            window = torch.cat([window[:, 1:], y.unsqueeze(1)], dim=1)
            days.append(y)
        self.n_steps = _n_steps_orig
        return torch.stack(days, dim=1)
