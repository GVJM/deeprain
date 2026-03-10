"""
ar_real_nvp.py — AR RealNVP: GRU/LSTM context + conditional normalizing flow

Architecture:
    RNN:    window(B,W,S) → h(B,rnn_hidden)
    Flow:   n_coupling _CondCouplingLayer instances
            Each: split x via checkerboard mask
                  (s, t) = MLP(cat(x_fixed, h))   [conditioned on h]
                  x_free = x_free * exp(tanh(s)) + t
    Loss:   exact NLL = -log_prob(x | h)
    Sample: z ~ N(0,I) → inverse coupling chain conditioned on h

rnn_type: 'gru' (default) or 'lstm'
"""

import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel


def _make_rnn(rnn_type: str, input_size: int, hidden_size: int):
    cls = {'gru': nn.GRU, 'lstm': nn.LSTM}[rnn_type]
    return cls(input_size=input_size, hidden_size=hidden_size,
               num_layers=2, batch_first=True, dropout=0.1)


def _extract_h(rnn_output, rnn_type: str) -> Tensor:
    _, h_n = rnn_output
    if rnn_type == 'lstm':
        h_n = h_n[0]   # LSTM returns (h_n, c_n)
    return h_n[-1]      # last layer: (B, hidden)


class _CondCouplingLayer(nn.Module):
    """
    Affine coupling layer conditioned on GRU/LSTM hidden state h.
    x_fixed = x[mask], x_free = x[~mask]
    (s, t) = MLP(cat(x_fixed, h))
    y_free = x_free * exp(tanh(s)) + t
    log_det = sum(tanh(s))
    Invertibility preserved: h only conditions s,t computation, not x_fixed.
    """

    def __init__(self, input_size: int, mask: Tensor, rnn_hidden: int, hidden: int = 128):
        super().__init__()
        self.register_buffer('mask', mask.float())
        n_fixed = int(mask.sum().item())
        n_free  = input_size - n_fixed
        self.n_fixed = n_fixed
        self.n_free  = n_free
        self.net = nn.Sequential(
            nn.Linear(n_fixed + rnn_hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_free * 2),
        )

    def forward(self, x: Tensor, h: Tensor):
        x_fixed = x[:, self.mask.bool()]
        x_free  = x[:, ~self.mask.bool()]
        st = self.net(torch.cat([x_fixed, h], dim=-1))
        s, t = st[:, :self.n_free], st[:, self.n_free:]
        s = torch.tanh(s)
        y_free  = x_free * torch.exp(s) + t
        log_det = s.sum(dim=-1)
        y = torch.empty_like(x)
        y[:, self.mask.bool()]  = x_fixed
        y[:, ~self.mask.bool()] = y_free
        return y, log_det

    def inverse(self, y: Tensor, h: Tensor):
        x_fixed = y[:, self.mask.bool()]
        y_free  = y[:, ~self.mask.bool()]
        st = self.net(torch.cat([x_fixed, h], dim=-1))
        s, t = st[:, :self.n_free], st[:, self.n_free:]
        s = torch.tanh(s)
        x_free = (y_free - t) * torch.exp(-s)
        x = torch.empty_like(y)
        x[:, self.mask.bool()]  = x_fixed
        x[:, ~self.mask.bool()] = x_free
        return x


class ARRealNVP(BaseModel):
    def __init__(self, input_size=90, window_size=30, rnn_hidden=128,
                 n_coupling=8, hidden_size=256, rnn_type='gru', **kwargs):
        super().__init__()
        self.n_stations  = input_size
        self.window_size = window_size
        self.rnn_type    = rnn_type
        self.rnn = _make_rnn(rnn_type, input_size, rnn_hidden)
        self.layers = nn.ModuleList()
        for i in range(n_coupling):
            mask = torch.zeros(input_size, dtype=torch.bool)
            mask[::2] = True if i % 2 == 0 else False
            mask[1::2] = True if i % 2 == 1 else False
            if mask.all() or (~mask).all():
                mask = torch.zeros(input_size, dtype=torch.bool)
                mask[:input_size // 2] = True
            self.layers.append(_CondCouplingLayer(input_size, mask, rnn_hidden, hidden_size))
        self.log_scale = nn.Parameter(torch.zeros(input_size))

    def _encode_window(self, window: Tensor) -> Tensor:
        return _extract_h(self.rnn(window), self.rnn_type)

    def log_prob(self, x: Tensor, h: Tensor) -> Tensor:
        z = x * torch.exp(self.log_scale)
        log_det = self.log_scale.sum().expand(x.shape[0])
        for layer in self.layers:
            z, ld = layer(z, h)
            log_det = log_det + ld
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det

    def loss(self, x, beta: float = 1.0) -> dict:
        window, target = x
        h   = self._encode_window(window)
        nll = -self.log_prob(target, h).mean()
        return {'total': nll, 'nll': nll}

    def _flow_sample(self, h: Tensor, n: int) -> Tensor:
        z = torch.randn(n, self.n_stations, device=h.device)
        z = z * torch.exp(-self.log_scale)
        for layer in reversed(self.layers):
            z = layer.inverse(z, h)
        return z.clamp(min=0.0)

    @torch.no_grad()
    def sample(self, n: int, steps=None, method=None) -> Tensor:
        device = next(self.parameters()).device
        window = torch.zeros(1, self.window_size, self.n_stations, device=device)
        for _ in range(self.window_size):
            h = self._encode_window(window)
            y = self._flow_sample(h, 1)
            window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)
        samples = []
        for _ in range(n):
            h = self._encode_window(window)
            y = self._flow_sample(h, 1)
            window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)
            samples.append(y)
        return torch.cat(samples, dim=0)

    @torch.no_grad()
    def sample_rollout(self, seed_window: Tensor, n_days: int, n_scenarios: int = 10) -> Tensor:
        device = next(self.parameters()).device
        window = seed_window.to(device).unsqueeze(0).expand(n_scenarios, -1, -1).clone()
        days = []
        for _ in range(n_days):
            h = self._encode_window(window)
            y = self._flow_sample(h, n_scenarios)
            window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)
            days.append(y)
        return torch.stack(days, dim=1)
