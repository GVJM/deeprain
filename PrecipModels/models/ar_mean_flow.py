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


class _MeanFlowMLP(nn.Module):
    """u_theta: [z_t(S) || r_emb(T) || t_emb(T) || h(H)] → velocity(S)"""
    def __init__(self, data_dim, t_embed_dim, rnn_hidden, cond_dim, hidden=256, n_layers=4):
        super().__init__()
        in_dim = data_dim + 2 * t_embed_dim + rnn_hidden + cond_dim
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
                 mf_ratio=0.25, jvp_eps=0.01, rnn_type='gru',
                 tangent_warmup_steps: int = 0,
                 improved_interval_sampling: bool = False,
                 mu_sad: float = 0.0,
                 sigma_sad: float = 1.0,
                 **kwargs):
        super().__init__()
        self.n_stations  = input_size
        self.window_size = window_size
        self.rnn_type    = rnn_type
        self.mf_ratio    = mf_ratio
        self.jvp_eps     = jvp_eps
        self.tangent_warmup_steps       = tangent_warmup_steps
        self.improved_interval_sampling = improved_interval_sampling
        self.mu_sad                     = mu_sad
        self.sigma_sad                  = sigma_sad
        self.cond_block  = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        self.cond_dim    = self.cond_block.total_dim
        self.rnn         = _make_rnn(rnn_type, input_size, rnn_hidden)
        self.t_embed     = SinusoidalEmbedding(t_embed_dim)
        self.vel_net     = _MeanFlowMLP(input_size, t_embed_dim, rnn_hidden, self.cond_dim, hidden_size, n_layers)
        # Backward-compat: remap old checkpoints saved with key prefix "velocity."
        self._register_load_state_dict_pre_hook(self._remap_old_velocity_keys)

    # ── Backward-compat key remap ────────────────────────────────────────────

    @staticmethod
    def _remap_old_velocity_keys(state_dict, prefix, *args, **kwargs):
        """Remap old checkpoint keys 'velocity.*' → 'vel_net.*'."""
        for k in list(state_dict.keys()):
            if k.startswith(prefix + 'velocity.'):
                state_dict[k.replace(prefix + 'velocity.', prefix + 'vel_net.')] = state_dict.pop(k)

    # ── Public velocity interface ────────────────────────────────────────────

    def velocity(self, z_t: Tensor, t_emb: Tensor, h_cond: Tensor) -> Tensor:
        """
        Compute instantaneous velocity at (z_t, t_emb) given context h_cond.
        Uses r_emb = t_emb (FM-mode evaluation: r = t).

        z_t:    (B, S)
        t_emb:  (B, T)
        h_cond: (B, rnn_hidden + cond_dim)
        → velocity: (B, S)
        """
        return self.vel_net(z_t, t_emb, t_emb, h_cond)

    def _encode_window(self, window):
        return _extract_h(self.rnn(window), self.rnn_type)

    def _cond_embed(self, cond: dict | None, batch_size: int, device: torch.device) -> Tensor:
        if cond is None:
            return torch.zeros(batch_size, self.cond_dim, device=device)
        return self.cond_block(cond)

    def _make_day_cond(self, day: int, batch_size: int, device: torch.device) -> dict:
        """Build conditioning dict for a given day-of-year (1–366)."""
        angle = 2.0 * math.pi * day / 365.25
        month_idx = int((day - 1) * 12 / 365) % 12
        return {
            'month':   torch.full((batch_size,), month_idx, dtype=torch.long,  device=device),
            'day_sin': torch.full((batch_size,), math.sin(angle),               device=device),
            'day_cos': torch.full((batch_size,), math.cos(angle),               device=device),
        }

    def loss(self, x, beta=1.0, effective_mf_ratio=None, training_step: int = 0):
        mf_r = effective_mf_ratio if effective_mf_ratio is not None else self.mf_ratio
        if len(x) == 3:
            window, target, cond = x
        else:
            window, target = x
            cond = None
        B = target.shape[0]
        device = target.device

        h      = self._encode_window(window)
        cond_emb = self._cond_embed(cond, B, device)
        h_cond = torch.cat([h, cond_emb], dim=-1)
        z_0    = torch.randn_like(target)
        t      = torch.rand(B, device=device)
        z_t    = (1 - t.unsqueeze(-1)) * z_0 + t.unsqueeze(-1) * target
        v_cond = target - z_0
        t_emb  = self.t_embed(t)

        is_mf  = torch.rand(B, device=device) < mf_r
        fm_idx = (~is_mf).nonzero(as_tuple=True)[0]
        mf_idx = is_mf.nonzero(as_tuple=True)[0]

        # ── FM batches (r = t): standard flow matching ────────────────────────
        loss_fm = torch.tensor(0.0, device=device)
        if fm_idx.numel() > 0:
            u = self.vel_net(z_t[fm_idx], t_emb[fm_idx], t_emb[fm_idx], h_cond[fm_idx])
            loss_fm = F.mse_loss(u, v_cond[fm_idx])

        # ── MeanFlow batches (r < t): MeanFlow Identity ───────────────────────
        loss_mf = torch.tensor(0.0, device=device)
        if mf_idx.numel() > 0:
            t_mf      = t[mf_idx]
            if self.improved_interval_sampling:
                tau  = torch.randn(mf_idx.numel(), device=device)
                d    = torch.sigmoid(tau * self.sigma_sad + self.mu_sad)
                r_mf = (t_mf - d).clamp(min=0.0)
            else:
                r_mf = torch.rand(mf_idx.numel(), device=device) * t_mf
            r_mf_emb  = self.t_embed(r_mf)
            t_mf_emb  = t_emb[mf_idx]
            z_t_mf    = z_t[mf_idx]
            v_cond_mf = v_cond[mf_idx]
            h_mf      = h_cond[mf_idx]

            # JVP: directional derivative of u w.r.t. z_t in direction v_cond
            def u_fn(z):
                return self.vel_net(z, r_mf_emb, t_mf_emb, h_mf)

            u_val, jvp_z = func_jvp(u_fn, (z_t_mf,), (v_cond_mf.detach(),))

            # Finite difference for ∂u/∂t
            t_plus     = (t_mf + self.jvp_eps).clamp(max=1.0)
            t_plus_emb = self.t_embed(t_plus)
            with torch.no_grad():
                u_t_plus = self.vel_net(z_t_mf, r_mf_emb, t_plus_emb, h_mf)
            du_dt = (u_t_plus - u_val.detach()) / self.jvp_eps

            correction = (jvp_z + du_dt).detach()
            correction = correction / (correction.norm(dim=-1, keepdim=True) + 0.1)
            r_scale    = min(1.0, training_step / self.tangent_warmup_steps) if self.tangent_warmup_steps > 0 else 1.0
            u_target   = v_cond_mf - r_scale * (t_mf - r_mf).unsqueeze(-1) * correction
            loss_mf    = F.mse_loss(u_val, u_target.detach())

        total = (1 - mf_r) * loss_fm + mf_r * loss_mf
        return {'total': total, 'fm_loss': loss_fm, 'mf_loss': loss_mf}

    def _generate_sample(self, h_cond, n):
        """1-step: z_hat = z_1 - u_theta(z_1, r=0, t=1, h)"""
        device = h_cond.device
        z_1   = torch.randn(n, self.n_stations, device=device)
        r_emb = self.t_embed(torch.zeros(n, device=device))
        t_emb = self.t_embed(torch.ones(n, device=device))
        return (z_1 - self.vel_net(z_1, r_emb, t_emb, h_cond)).clamp(min=0.0)

    @torch.no_grad()
    def sample(self, n, steps=None, method=None, start_day: int = 1):
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
                print(f"  [ar_mean_flow] sampling step {i}/{n}...", flush=True)
            day = (start_day + i - 1) % 365 + 1
            cond = self._make_day_cond(day, 1, device)
            h = self._encode_window(window)
            cond_emb = self._cond_embed(cond, 1, device)
            h_cond = torch.cat([h, cond_emb], dim=-1)
            y = self._generate_sample(h_cond, 1)
            window = torch.cat([window[:, 1:], y.unsqueeze(1)], dim=1)
            samples.append(y)
        return torch.cat(samples, dim=0)

    @torch.no_grad()
    def sample_rollout(self, seed_window, n_days, n_scenarios=10, start_day: int = 1):
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
        return torch.stack(days, dim=1)
