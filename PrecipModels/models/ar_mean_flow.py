"""
ar_mean_flow.py — AR MeanFlow: GRU/LSTM context + mean flow velocity

Reference: arXiv 2505.13447 (Geng et al., May 2025)

Network: u_theta(z_t, r_emb, t_emb, h) → average velocity u
         Input: [z_t(S) || r_emb(T) || t_emb(T) || h(H)]  (two time embeddings)

Training:
    OT path: z_t = (1-t)*z_0 + t*x,  v_cond = x - z_0   [analytical]

    75% FM batches (r = t):
        loss = MSE(u_theta(z_t, t_emb, t_emb, h), v_cond)

    25% MeanFlow batches (r ~ U(0, t)):
        correction = sg[JVP_z(u, v_cond) + finite_diff(∂u/∂t)]
        u_target = v_cond - (t - r) * correction
        loss = MSE(u_theta(z_t, r_emb, t_emb, h), u_target)

1-step sampling:
    z_hat = z_1 - u_theta(z_1, emb(0), emb(1), h)
"""

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.func import jvp as func_jvp

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel
from models.flow_match import SinusoidalEmbedding
from models.ar_real_nvp import _make_rnn, _extract_h


class _MeanFlowMLP(nn.Module):
    """u_theta: [z_t(S) || r_emb(T) || t_emb(T) || h(H)] → velocity(S)"""
    def __init__(self, data_dim, t_embed_dim, rnn_hidden, hidden=256, n_layers=4):
        super().__init__()
        in_dim = data_dim + 2 * t_embed_dim + rnn_hidden
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t, r_emb, t_emb, h):
        return self.net(torch.cat([z_t, r_emb, t_emb, h], dim=-1))


class ARMeanFlow(BaseModel):
    def __init__(self, input_size=90, window_size=30, rnn_hidden=128,
                 hidden_size=256, n_layers=4, t_embed_dim=64,
                 mf_ratio=0.25, jvp_eps=0.01, rnn_type='gru', **kwargs):
        super().__init__()
        self.n_stations  = input_size
        self.window_size = window_size
        self.rnn_type    = rnn_type
        self.mf_ratio    = mf_ratio
        self.jvp_eps     = jvp_eps
        self.rnn         = _make_rnn(rnn_type, input_size, rnn_hidden)
        self.t_embed     = SinusoidalEmbedding(t_embed_dim)
        self.velocity    = _MeanFlowMLP(input_size, t_embed_dim, rnn_hidden, hidden_size, n_layers)

    def _encode_window(self, window):
        return _extract_h(self.rnn(window), self.rnn_type)

    def loss(self, x, beta=1.0):
        window, target = x
        B = target.shape[0]
        device = target.device

        h      = self._encode_window(window)
        z_0    = torch.randn_like(target)
        t      = torch.rand(B, device=device)
        z_t    = (1 - t.unsqueeze(-1)) * z_0 + t.unsqueeze(-1) * target
        v_cond = target - z_0
        t_emb  = self.t_embed(t)

        is_mf  = torch.rand(B, device=device) < self.mf_ratio
        fm_idx = (~is_mf).nonzero(as_tuple=True)[0]
        mf_idx = is_mf.nonzero(as_tuple=True)[0]

        # ── FM batches (r = t): standard flow matching ────────────────────────
        loss_fm = torch.tensor(0.0, device=device)
        if fm_idx.numel() > 0:
            u = self.velocity(z_t[fm_idx], t_emb[fm_idx], t_emb[fm_idx], h[fm_idx])
            loss_fm = F.mse_loss(u, v_cond[fm_idx])

        # ── MeanFlow batches (r < t): MeanFlow Identity ───────────────────────
        loss_mf = torch.tensor(0.0, device=device)
        if mf_idx.numel() > 0:
            t_mf      = t[mf_idx]
            r_mf      = torch.rand(mf_idx.numel(), device=device) * t_mf
            r_mf_emb  = self.t_embed(r_mf)
            t_mf_emb  = t_emb[mf_idx]
            z_t_mf    = z_t[mf_idx]
            v_cond_mf = v_cond[mf_idx]
            h_mf      = h[mf_idx]

            # JVP: directional derivative of u w.r.t. z_t in direction v_cond
            def u_fn(z):
                return self.velocity(z, r_mf_emb, t_mf_emb, h_mf)

            u_val, jvp_z = func_jvp(u_fn, (z_t_mf,), (v_cond_mf.detach(),))

            # Finite difference for ∂u/∂t
            t_plus     = (t_mf + self.jvp_eps).clamp(max=1.0)
            t_plus_emb = self.t_embed(t_plus)
            with torch.no_grad():
                u_t_plus = self.velocity(z_t_mf, r_mf_emb, t_plus_emb, h_mf)
            du_dt = (u_t_plus - u_val.detach()) / self.jvp_eps

            correction = (jvp_z + du_dt).detach()
            u_target   = v_cond_mf - (t_mf - r_mf).unsqueeze(-1) * correction
            loss_mf    = F.mse_loss(u_val, u_target.detach())

        total = (1 - self.mf_ratio) * loss_fm + self.mf_ratio * loss_mf
        return {'total': total, 'fm_loss': loss_fm, 'mf_loss': loss_mf}

    def _generate_sample(self, h, n):
        """1-step: z_hat = z_1 - u_theta(z_1, r=0, t=1, h)"""
        device = h.device
        z_1   = torch.randn(n, self.n_stations, device=device)
        r_emb = self.t_embed(torch.zeros(n, device=device))
        t_emb = self.t_embed(torch.ones(n, device=device))
        return (z_1 - self.velocity(z_1, r_emb, t_emb, h)).clamp(min=0.0)

    @torch.no_grad()
    def sample(self, n, steps=None, method=None):
        device = next(self.parameters()).device
        window = torch.zeros(1, self.window_size, self.n_stations, device=device)
        for _ in range(self.window_size):
            h = self._encode_window(window)
            y = self._generate_sample(h, 1)
            window = torch.cat([window[:, 1:], y.unsqueeze(1)], dim=1)
        samples = []
        for _ in range(n):
            h = self._encode_window(window)
            y = self._generate_sample(h, 1)
            window = torch.cat([window[:, 1:], y.unsqueeze(1)], dim=1)
            samples.append(y)
        return torch.cat(samples, dim=0)

    @torch.no_grad()
    def sample_rollout(self, seed_window, n_days, n_scenarios=10):
        device = next(self.parameters()).device
        window = seed_window.to(device).unsqueeze(0).expand(n_scenarios, -1, -1).clone()
        days = []
        for _ in range(n_days):
            h = self._encode_window(window)
            y = self._generate_sample(h, n_scenarios)
            window = torch.cat([window[:, 1:], y.unsqueeze(1)], dim=1)
            days.append(y)
        return torch.stack(days, dim=1)
