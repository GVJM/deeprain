"""
ar_ddpm.py — AR Diffusion Model: GRU/LSTM context + continuous VP-SDE with v-prediction

Architecture: GRU context encoder + MLP denoiser with v-prediction parameterization.

Continuous VP-SDE (Song et al. 2021 "Score-Based Generative Modeling through SDEs"):
    β(t) = β_min + (β_max - β_min) * t        [linear noise schedule, t ∈ [0,1]]
    ᾱ(t) = exp(-(β_min*t + 0.5*(β_max-β_min)*t²))  [cumulative signal retention]
    forward: x_t = √ᾱ(t)*x_0 + √(1-ᾱ(t))*ε   [noising process]

V-prediction parameterization (Salimans & Ho 2022):
    v_target = √ᾱ(t)*ε - √(1-ᾱ(t))*x_0
    loss = MSE(v_θ(x_t, t_emb, h), v_target)

    Recovery (orthogonal transform, exact):
        x_0_pred = √ᾱ(t)*x_t - √(1-ᾱ(t))*v_pred
        ε_pred   = √(1-ᾱ(t))*x_t + √ᾱ(t)*v_pred

Sampling (DDIM deterministic, η=0, n_sample_steps ≤ total):
    x_1 ~ N(0,I), reverse from t=1 → t=0:
    x_{t-dt} = √ᾱ(t-dt)*x_0_pred + √(1-ᾱ(t-dt))*ε_pred

velocity() interface (AYF distillation):
    Returns v-prediction output directly — compatible with ar_flow_map AYF distillation
    infrastructure which calls teacher.velocity(z_t, t_emb, h_cond).

Variants: ar_ddpm, ar_ddpm_lstm
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


# ──────────────────────────────────────────────────────────────────────────────
# Noise schedule (continuous VP-SDE, linear β schedule)
# ──────────────────────────────────────────────────────────────────────────────

_VP_BETA_MIN = 0.1
_VP_BETA_MAX = 20.0


def _vp_alpha_cumprod(t: Tensor, beta_min: float = _VP_BETA_MIN,
                      beta_max: float = _VP_BETA_MAX) -> Tensor:
    """ᾱ(t) = exp(-(β_min*t + 0.5*(β_max-β_min)*t²))  for t ∈ [0,1]."""
    log_alpha = -(beta_min * t + 0.5 * (beta_max - beta_min) * t ** 2)
    return torch.exp(log_alpha)


# ──────────────────────────────────────────────────────────────────────────────
# V-prediction MLP
# ──────────────────────────────────────────────────────────────────────────────

class _VPredMLP(nn.Module):
    """v-prediction denoiser: [x_t(S) || t_emb(T) || h(H)] → v_theta(S)"""

    def __init__(self, data_dim, t_embed_dim, rnn_hidden, cond_dim,
                 hidden=256, n_layers=4):
        super().__init__()
        in_dim = data_dim + t_embed_dim + rnn_hidden + cond_dim
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: Tensor, t_emb: Tensor, h: Tensor) -> Tensor:
        return self.net(torch.cat([x_t, t_emb, h], dim=-1))


# ──────────────────────────────────────────────────────────────────────────────
# AR Diffusion Model
# ──────────────────────────────────────────────────────────────────────────────

class ARDiffusion(BaseModel):
    """
    Autoregressive VP-SDE diffusion teacher with v-prediction.

    Sample quality is controlled by n_sample_steps (inference) and by
    hidden_size/n_layers. As a teacher, use hidden_size=256, n_layers=6,
    n_sample_steps=100 for best quality.
    """

    def __init__(self, input_size=90, window_size=30, rnn_hidden=128,
                 hidden_size=256, n_layers=4, t_embed_dim=64,
                 rnn_type='gru', n_sample_steps=50,
                 beta_min: float = _VP_BETA_MIN,
                 beta_max: float = _VP_BETA_MAX,
                 **kwargs):
        super().__init__()
        self.n_stations    = input_size
        self.window_size   = window_size
        self.rnn_type      = rnn_type
        self.n_sample_steps = n_sample_steps
        self.beta_min      = beta_min
        self.beta_max      = beta_max
        self.cond_block    = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        self.cond_dim      = self.cond_block.total_dim
        self.rnn           = _make_rnn(rnn_type, input_size, rnn_hidden)
        self.t_embed       = SinusoidalEmbedding(t_embed_dim)
        self.denoiser      = _VPredMLP(
            input_size, t_embed_dim, rnn_hidden, self.cond_dim, hidden_size, n_layers
        )

    # ── Context encoding ─────────────────────────────────────────────────────

    def _encode_window(self, window: Tensor) -> Tensor:
        return _extract_h(self.rnn(window), self.rnn_type)

    def _cond_embed(self, cond: dict | None, batch_size: int,
                    device: torch.device) -> Tensor:
        if cond is None:
            return torch.zeros(batch_size, self.cond_dim, device=device)
        return self.cond_block(cond)

    def _make_day_cond(self, day: int, batch_size: int,
                       device: torch.device) -> dict:
        angle = 2.0 * math.pi * day / 365.25
        month_idx = int((day - 1) * 12 / 365) % 12
        return {
            'month':   torch.full((batch_size,), month_idx, dtype=torch.long, device=device),
            'day_sin': torch.full((batch_size,), math.sin(angle), device=device),
            'day_cos': torch.full((batch_size,), math.cos(angle), device=device),
        }

    # ── Noise schedule helpers ────────────────────────────────────────────────

    def _ac(self, t: Tensor) -> Tensor:
        """Compute ᾱ(t) for a batch of t values."""
        return _vp_alpha_cumprod(t, self.beta_min, self.beta_max)

    # ── velocity() interface (AYF distillation) ───────────────────────────────

    def velocity(self, z_t: Tensor, t_emb: Tensor, h_cond: Tensor) -> Tensor:
        """
        V-prediction output — serves as velocity signal for AYF distillation.

        Compatible with the teacher.velocity(z_t, t_emb, h_cond) interface
        expected by ar_flow_map's AYF loss. The v-prediction output is a
        learned combination of the score and the data, providing meaningful
        directional information at any (x_t, t) point.
        """
        return self.denoiser(z_t, t_emb, h_cond)

    # ── Training loss ─────────────────────────────────────────────────────────

    def loss(self, x, beta=1.0, **kwargs):
        """VP-SDE v-prediction loss on temporal (window, target) pairs."""
        if len(x) == 3:
            window, target, cond = x
        else:
            window, target = x
            cond = None
        B = target.shape[0]
        device = target.device

        h        = self._encode_window(window)
        cond_emb = self._cond_embed(cond, B, device)
        h_cond   = torch.cat([h, cond_emb], dim=-1)

        # Sample continuous time and compute noise schedule values
        t_float = torch.rand(B, device=device)
        ac      = self._ac(t_float)                        # (B,)
        sqrt_ac     = ac.sqrt().unsqueeze(-1)              # (B, 1)
        sqrt_1mAc   = (1.0 - ac).clamp(min=1e-8).sqrt().unsqueeze(-1)

        # Forward noising: x_t = √ᾱ * x_0 + √(1-ᾱ) * ε
        noise  = torch.randn_like(target)
        x_t    = sqrt_ac * target + sqrt_1mAc * noise

        # V-prediction target: v = √ᾱ * ε - √(1-ᾱ) * x_0
        v_target = sqrt_ac * noise - sqrt_1mAc * target

        # Denoiser forward pass
        t_emb  = self.t_embed(t_float)
        v_pred = self.denoiser(x_t, t_emb, h_cond)

        diff_loss = F.mse_loss(v_pred, v_target)
        return {'total': diff_loss, 'diff_loss': diff_loss}

    # ── Sampling ──────────────────────────────────────────────────────────────

    def _generate_sample(self, h_cond: Tensor, n: int) -> Tensor:
        """
        DDIM reverse diffusion (deterministic, η=0) from t=1 to t=0.

        Recovers x_0 from Gaussian noise using n_sample_steps steps.
        Larger n_sample_steps → higher quality, slower sampling.
        """
        device = h_cond.device
        x = torch.randn(n, self.n_stations, device=device)  # start from pure noise

        # Evenly-spaced timesteps from t=1 to 0 (inclusive of t=0)
        t_steps = torch.linspace(1.0, 0.0, self.n_sample_steps + 1,
                                  device=device)

        for i in range(self.n_sample_steps):
            t_cur  = t_steps[i].expand(n)    # (n,)
            t_next = t_steps[i + 1].expand(n)

            ac_cur      = self._ac(t_cur)
            sqrt_ac_c   = ac_cur.sqrt().unsqueeze(-1)       # (n, 1)
            sqrt_1mAc_c = (1.0 - ac_cur).clamp(min=1e-8).sqrt().unsqueeze(-1)

            ac_next     = self._ac(t_next)
            sqrt_ac_n   = ac_next.sqrt().unsqueeze(-1)
            sqrt_1mAc_n = (1.0 - ac_next).clamp(min=1e-8).sqrt().unsqueeze(-1)

            # V-prediction
            t_emb  = self.t_embed(t_cur)
            v_pred = self.denoiser(x, t_emb, h_cond)

            # Recover x_0 and ε predictions (orthogonal transform, exact)
            x0_pred  = sqrt_ac_c * x - sqrt_1mAc_c * v_pred
            eps_pred = sqrt_1mAc_c * x + sqrt_ac_c * v_pred

            # DDIM deterministic step (η=0)
            x = sqrt_ac_n * x0_pred + sqrt_1mAc_n * eps_pred

        return x.clamp(min=0.0)

    @torch.no_grad()
    def sample(self, n, steps=None, method=None, start_day: int = 1):
        _n_steps_orig = self.n_sample_steps
        if steps is not None:
            self.n_sample_steps = steps
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
                print(f"  [ar_ddpm] sampling step {i}/{n}...", flush=True)
            day = (start_day + i - 1) % 365 + 1
            cond = self._make_day_cond(day, 1, device)
            h = self._encode_window(window)
            cond_emb = self._cond_embed(cond, 1, device)
            h_cond = torch.cat([h, cond_emb], dim=-1)
            y = self._generate_sample(h_cond, 1)
            window = torch.cat([window[:, 1:], y.unsqueeze(1)], dim=1)
            samples.append(y)
        self.n_sample_steps = _n_steps_orig
        return torch.cat(samples, dim=0)

    @torch.no_grad()
    def sample_rollout(self, seed_window, n_days, n_scenarios=10,
                       start_day: int = 1, steps=None):
        _n_steps_orig = self.n_sample_steps
        if steps is not None:
            self.n_sample_steps = steps
        device = next(self.parameters()).device
        window = seed_window.to(device).unsqueeze(0).expand(
            n_scenarios, -1, -1).clone()
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
        self.n_sample_steps = _n_steps_orig
        return torch.stack(days, dim=1)
