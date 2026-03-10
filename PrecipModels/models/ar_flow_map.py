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

1-step sampling:
    h = GRU(window)
    z_1 ~ N(0,I)  [pure noise, t=1 on OT scale]
    z_hat = Phi_theta(z_1, emb(1), emb(0), h)  [jump to data manifold at t=0]
    y = clamp(z_hat, min=0)
"""

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel
from models.flow_match import SinusoidalEmbedding
from models.ar_real_nvp import _make_rnn, _extract_h


class _FlowMapMLP(nn.Module):
    """Phi_theta: [z_s(S) || s_emb(T) || t_emb(T) || h(H)] → z_t(S)"""
    def __init__(self, data_dim, t_embed_dim, rnn_hidden, hidden=256, n_layers=4):
        super().__init__()
        in_dim = data_dim + 2 * t_embed_dim + rnn_hidden
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
                 rnn_type='gru', **kwargs):
        super().__init__()
        self.n_stations  = input_size
        self.window_size = window_size
        self.rnn_type    = rnn_type
        self.rnn         = _make_rnn(rnn_type, input_size, rnn_hidden)
        self.t_embed     = SinusoidalEmbedding(t_embed_dim)
        self.flow_map    = _FlowMapMLP(input_size, t_embed_dim, rnn_hidden, hidden_size, n_layers)

    def _encode_window(self, window):
        return _extract_h(self.rnn(window), self.rnn_type)

    def loss(self, x, beta=1.0):
        """MSE regression: Phi_theta(z_s, s, t, h) → z_t on OT paths."""
        window, target = x
        B = target.shape[0]
        device = target.device

        h    = self._encode_window(window)
        z_0  = torch.randn_like(target)
        s    = torch.rand(B, device=device)
        t    = torch.rand(B, device=device)
        z_s  = (1 - s.unsqueeze(-1)) * z_0 + s.unsqueeze(-1) * target
        z_t  = (1 - t.unsqueeze(-1)) * z_0 + t.unsqueeze(-1) * target
        s_emb = self.t_embed(s)
        t_emb = self.t_embed(t)

        z_t_pred = self.flow_map(z_s, s_emb, t_emb, h)
        fm_loss  = F.mse_loss(z_t_pred, z_t)
        return {'total': fm_loss, 'fm_loss': fm_loss}

    def _generate_sample(self, h, n):
        """1-step: Phi(z_1, s=1, t=0, h) → data space."""
        device = h.device
        z_1   = torch.randn(n, self.n_stations, device=device)
        s_emb = self.t_embed(torch.ones(n, device=device))
        t_emb = self.t_embed(torch.zeros(n, device=device))
        return self.flow_map(z_1, s_emb, t_emb, h).clamp(min=0.0)

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
