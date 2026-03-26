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

Sampling: h = GRU(window_{t-W:t-1})  [context from history]
          z_0 ~ N(0,I) [pure noise, t=0 on OT scale]
          z_1 = FlowMap(z_0, s=0, t=1, h)  [n_steps=1: single jump]
          or: z_1 = Euler(z_0→1 via self.flow_map, h, n_steps)  [multi-step ODE]
          n_steps=1: single-step prediction (may collapse diversity)
          n_steps>1: multi-step ODE (better diversity, used by ar_flow_map_ms)

Variants:
  ar_flow_map       — base position predictor
  ar_flow_map_lstm  — LSTM RNN context
  ar_flow_map_ms    — multi-step refinement (n_steps=10)
  ar_flow_map_sd    — LSD auxiliary loss (derivative matching)
  ar_flow_map_res   — residual parameterization: output = z_s + (t-s)*MLP(...)

Residual variant (ar_flow_map_res):
  Standard position prediction collapses to output ≈ 0 (conditional mean) on sparse
  precipitation data because MSE is minimized by predicting the mean. The residual fix
  forces output = z_s + (t-s)*delta, so even if delta→0 the output tracks z_s.
  This is structurally equivalent to a single Euler step of a velocity model.
  use_residual=True must be saved to config.json for correct checkpoint loading.
"""

import math
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
                 rnn_type='gru', n_steps: int = 1,
                 lsd_weight: float = 0.0,
                 ayf_weight: float = 0.0,
                 ayf_delta_t: float = 0.05,
                 use_residual: bool = False,
                 **kwargs):
        super().__init__()
        self.n_stations   = input_size
        self.window_size  = window_size
        self.rnn_type     = rnn_type
        self.n_steps      = n_steps
        self.lsd_weight   = lsd_weight
        self.ayf_weight   = ayf_weight
        self.ayf_delta_t  = ayf_delta_t
        self.use_residual = use_residual
        self.cond_block  = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        self.cond_dim    = self.cond_block.total_dim
        self.rnn         = _make_rnn(rnn_type, input_size, rnn_hidden)
        self.t_embed     = SinusoidalEmbedding(t_embed_dim)
        self.flow_map    = _FlowMapMLP(input_size, t_embed_dim, rnn_hidden, self.cond_dim, hidden_size, n_layers)
        object.__setattr__(self, '_teacher', None)

    def set_teacher(self, teacher_model):
        teacher_model.eval()
        teacher_model.requires_grad_(False)
        object.__setattr__(self, '_teacher', teacher_model)

    def _encode_window(self, window):
        return _extract_h(self.rnn(window), self.rnn_type)

    def _cond_embed(self, cond: dict | None, batch_size: int, device: torch.device) -> Tensor:
        if cond is None:
            return torch.zeros(batch_size, self.cond_dim, device=device)
        return self.cond_block(cond)

    def _make_day_cond(self, day: int, batch_size: int, device: torch.device) -> dict:
        """Build conditioning dict for a given day-of-year (1-366)."""
        angle = 2.0 * math.pi * day / 365.25
        month_idx = int((day - 1) * 12 / 365) % 12
        return {
            'month':   torch.full((batch_size,), month_idx, dtype=torch.long,  device=device),
            'day_sin': torch.full((batch_size,), math.sin(angle),               device=device),
            'day_cos': torch.full((batch_size,), math.cos(angle),               device=device),
        }

    def _phi(self, z_s, s_emb, t_emb, h_cond, dt=None):
        """Compute flow map output, with optional residual parameterization.

        Standard: output = MLP(z_s, s_emb, t_emb, h_cond)  [predicts z_t directly]
        Residual:  output = z_s + dt * MLP(...)              [predicts displacement delta]

        The residual form prevents stochastic collapse: even if MLP→0, output tracks z_s.
        dt: scalar float (in _generate_sample) or (B,) tensor (in loss).
        """
        out = self.flow_map(z_s, s_emb, t_emb, h_cond)
        if self.use_residual:
            if isinstance(dt, float):
                return z_s + dt * out
            return z_s + dt.unsqueeze(-1) * out
        return out

    def loss(self, x, beta=1.0, **kwargs):
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

        z_t_pred = self._phi(z_s, s_emb, t_emb, h_cond, dt=(t - s))
        fm_loss  = F.mse_loss(z_t_pred, z_t)

        lsd_loss = torch.tensor(0.0, device=device)
        if self.lsd_weight > 0.0:
            def phi_wrt_t(t_val):
                t_e = self.t_embed(t_val)
                if self.use_residual:
                    delta = self.flow_map(z_s, s_emb, t_e, h_cond)
                    return z_s + (t_val - s).unsqueeze(-1) * delta
                return self.flow_map(z_s, s_emb, t_e, h_cond)
            _, dphi_dt = func_jvp(phi_wrt_t, (t,), (torch.ones_like(t),))
            lsd_loss = F.mse_loss(dphi_dt, (target - z_0).detach())

        ayf_loss = torch.tensor(0.0, device=device)
        if self.ayf_weight > 0.0 and self._teacher is not None:
            with torch.no_grad():
                h_t = self._teacher._encode_window(window)
                cond_emb_t = self._teacher._cond_embed(cond, B, device)
                h_cond_t = torch.cat([h_t, cond_emb_t], dim=-1)
                v_teacher = self._teacher.velocity(z_t, self._teacher.t_embed(t), h_cond_t)
                ayf_dt = self.ayf_delta_t
                t_back = (t - ayf_dt).clamp(min=0.0)
            # Student's backward prediction outside no_grad to allow gradient flow
            z_t_back = self._phi(z_s, s_emb, self.t_embed(t_back), h_cond, dt=(t_back - s))
            actual_dt = (t - t_back).clamp(min=1e-6)
            v_student = (z_t.detach() - z_t_back) / actual_dt.unsqueeze(-1)
            ayf_loss = F.mse_loss(v_student, v_teacher.detach())

        total = fm_loss + self.lsd_weight * lsd_loss + self.ayf_weight * ayf_loss
        return {'total': total, 'fm_loss': fm_loss, 'lsd_loss': lsd_loss, 'ayf_loss': ayf_loss}

    def _generate_sample(self, h_cond, n):
        """Refinement loop: n_steps passes that always feed in-distribution OT-path inputs.

        A position predictor Phi(z_s, s, t, h)->z_t is trained on clean OT-path points
        z_s = (1-s)*z_0 + s*x.  Chaining outputs directly (sequential Euler) passes
        the model's own imperfect z_{t+dt} as input to the next step — an out-of-
        distribution input that compounds errors across all 10 steps.

        Fix: keep z_0 fixed and build each probe as z_s = (1-s)*z_0 + s*z_1_estimate.
        All inputs remain on an OT path; each step refines z_1 from a better vantage point.
        n_steps=1 reduces to the original single-step FlowMap(z_0, 0, 1, h).
        """
        device = h_cond.device
        z_0 = torch.randn(n, self.n_stations, device=device)
        s0_emb = self.t_embed(torch.zeros(n, device=device))
        t1_emb = self.t_embed(torch.ones(n, device=device))
        # Initial prediction: direct jump from t=0 to t=1 (dt = 1.0 - 0.0 = 1.0)
        z_1 = self._phi(z_0, s0_emb, t1_emb, h_cond, dt=1.0)
        # Refinement: probe from intermediate OT-path points built with current z_1 estimate
        for i in range(1, self.n_steps):
            s_val = i / self.n_steps
            z_s = (1.0 - s_val) * z_0 + s_val * z_1   # always on OT path ✓
            s_emb = self.t_embed(torch.full((n,), s_val, device=device))
            z_1 = self._phi(z_s, s_emb, t1_emb, h_cond, dt=(1.0 - s_val))
        return z_1.clamp(min=0.0)

    @torch.no_grad()
    def sample(self, n, steps=None, method=None, start_day: int = 1):
        _n_steps_orig = self.n_steps
        if steps is not None:
            self.n_steps = steps
        device = next(self.parameters()).device
        window = torch.zeros(1, self.window_size, self.n_stations, device=device)
        for i in range(self.window_size):
            day = (start_day - self.window_size + i - 1) % 365 + 1
            cond = self._make_day_cond(day, 1, device)
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
            day = (start_day + i - 1) % 365 + 1
            cond = self._make_day_cond(day, 1, device)
            h = self._encode_window(window)
            cond_emb = self._cond_embed(cond, 1, device)
            h_cond = torch.cat([h, cond_emb], dim=-1)
            y = self._generate_sample(h_cond, 1)
            window = torch.cat([window[:, 1:], y.unsqueeze(1)], dim=1)
            samples.append(y)
        self.n_steps = _n_steps_orig
        return torch.cat(samples, dim=0)

    @torch.no_grad()
    def sample_rollout(self, seed_window, n_days, n_scenarios=10, start_day: int = 1, steps=None):
        _n_steps_orig = self.n_steps
        if steps is not None:
            self.n_steps = steps
        device = next(self.parameters()).device
        window = seed_window.to(device).unsqueeze(0).expand(n_scenarios, -1, -1).clone()
        days = []
        for i in range(n_days):
            day = (start_day + i - 1) % 365 + 1
            cond = self._make_day_cond(day, n_scenarios, device)
            h = self._encode_window(window)
            cond_emb = self._cond_embed(cond, n_scenarios, device)
            h_cond = torch.cat([h, cond_emb], dim=-1)
            y = self._generate_sample(h_cond, n_scenarios)
            window = torch.cat([window[:, 1:], y.unsqueeze(1)], dim=1)
            days.append(y)
        self.n_steps = _n_steps_orig
        return torch.stack(days, dim=1)
