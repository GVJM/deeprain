"""
ar_latent_fm.py — AR Latent Flow Matching: GRU context + VAE latent + latent flow matching

Architecture:
    GRU:      window(B,W,S) → h(B,gru_hidden)          [shared context encoder]

    VAE head (trains encoder/decoder, provides latent target for flow):
      Encoder: [target(S) || h(H)] → (mu, logvar) → z(latent_size)
      Decoder: [z(L) || h(H)] → y_hat(S)  [ReLU — non-negative]
      Loss:    MSE(y_hat, target) + beta * KL(mu, logvar)

    Flow head (trains velocity net in latent space):
      z_target = z.detach()       ← decouple flow from VAE gradients
      z_0 ~ N(0, I_L), t ~ U(0,1)
      z_t = (1-t)*z_0 + t*z_target
      v_θ([z_t(L) || t_emb(T) || h(H)]) → velocity(L)
      Loss: MSE(v_θ, z_target - z_0)

    Total loss: mse_recon + beta*kl + fm_loss

Sampling (rollout, same pattern as ARFlowMatch):
    h = GRU(window)
    z_0 ~ N(0, I_L)
    z_1 = Euler(v_θ(·, ·, h), z_0)    ← flow in latent space
    y = ReLU(decoder([z_1, h]))
    window ← shift + append y

Key advantage over ARVAE:
    - Flow matching avoids posterior collapse (no mode-dropping via KL)
    - Flow samples the full latent posterior, not a single z per step
Key advantage over ARFlowMatch:
    - Operates in compressed latent space → smoother distribution to match
    - VAE decoder provides structured non-negative output
"""

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel
from models.flow_match import SinusoidalEmbedding
from models.conditioning import ConditioningBlock, DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS


class _LatentVelocityMLP(nn.Module):
    """
    Velocity network operating in VAE latent space.

    Input: concatenation of [z_t (L), t_embed (T), h (H)]
    Output: velocity in latent space (L,)
    """

    def __init__(self, latent_size: int, t_embed_dim: int, gru_hidden: int, cond_dim: int,
                 hidden: int = 256, n_layers: int = 4, dropout: float = 0.0):
        super().__init__()
        in_dim = latent_size + t_embed_dim + gru_hidden + cond_dim
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        if dropout > 0.0:
            layers.append(nn.Dropout(dropout))
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden, latent_size))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: Tensor, t_emb: Tensor, h: Tensor) -> Tensor:
        """
        z_t:   (B, L) — latent point at time t
        t_emb: (B, T) — sinusoidal time embedding
        h:     (B, H) — GRU context
        → velocity: (B, L)
        """
        return self.net(torch.cat([z_t, t_emb, h], dim=-1))


class ARLatentFM(BaseModel):
    """
    AR Latent Flow Matching: joint end-to-end training of GRU + VAE + latent flow.

    Interface with train.py:
        loss(x, beta) accepts x as tuple (window, target) —
        train_neural_model_temporal passes TemporalDataset pairs.

    Interface with evaluate_model:
        sample(n) performs rollout of n steps from zero window.

    Primary generation method:
        sample_rollout(seed_window, n_days, n_scenarios)
        → Tensor (n_scenarios, n_days, n_stations)
    """

    def __init__(
        self,
        input_size: int = 90,
        window_size: int = 30,
        gru_hidden: int = 128,
        latent_size: int = 32,
        hidden_size: int = 256,
        n_layers: int = 4,
        t_embed_dim: int = 64,
        n_sample_steps: int = 50,
        free_bits: float = 0.5,
        dropout: float = 0.0,
        **kwargs,
    ):
        """
        Args:
            input_size:     number of stations (S)
            window_size:    historical window length (W)
            gru_hidden:     GRU hidden dimension (H)
            latent_size:    VAE latent dimension (L)
            hidden_size:    hidden dim of encoder, decoder, velocity MLP
            n_layers:       depth of velocity MLP
            t_embed_dim:    sinusoidal time embedding dimension (T)
            n_sample_steps: Euler integration steps during generation
            free_bits:      minimum KL per latent dim (prevents collapse)
        """
        super().__init__()
        self.n_stations     = input_size
        self.window_size    = window_size
        self.gru_hidden     = gru_hidden
        self.latent_size    = latent_size
        self.n_sample_steps = n_sample_steps
        self.free_bits      = free_bits
        self.cond_block     = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        self.cond_dim       = self.cond_block.total_dim

        # ── GRU: compresses (W, S) → h (gru_hidden,) ─────────────────────────
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # ── VAE Encoder: [target(S) || h(H)] → (mu, logvar) ──────────────────
        enc_in = input_size + gru_hidden + self.cond_dim
        _enc_layers = [nn.Linear(enc_in, hidden_size), nn.LeakyReLU(0.1)]
        if dropout > 0.0:
            _enc_layers.append(nn.Dropout(dropout))
        _enc_layers += [nn.Linear(hidden_size, hidden_size // 2), nn.LeakyReLU(0.1)]
        if dropout > 0.0:
            _enc_layers.append(nn.Dropout(dropout))
        self.encoder = nn.Sequential(*_enc_layers)
        self.fc_mu     = nn.Linear(hidden_size // 2, latent_size)
        self.fc_logvar = nn.Linear(hidden_size // 2, latent_size)

        # ── VAE Decoder: [z(L) || h(H)] → y_hat(S) ───────────────────────────
        dec_in = latent_size + gru_hidden + self.cond_dim
        _dec_layers = [nn.Linear(dec_in, hidden_size // 2), nn.LeakyReLU(0.1)]
        if dropout > 0.0:
            _dec_layers.append(nn.Dropout(dropout))
        _dec_layers += [nn.Linear(hidden_size // 2, hidden_size), nn.LeakyReLU(0.1)]
        if dropout > 0.0:
            _dec_layers.append(nn.Dropout(dropout))
        _dec_layers += [nn.Linear(hidden_size, input_size), nn.ReLU()]
        self.decoder = nn.Sequential(*_dec_layers)

        # ── Flow head: latent velocity network ───────────────────────────────
        self.t_embed  = SinusoidalEmbedding(t_embed_dim)
        self.velocity = _LatentVelocityMLP(
            latent_size=latent_size,
            t_embed_dim=t_embed_dim,
            gru_hidden=gru_hidden,
            cond_dim=self.cond_dim,
            hidden=hidden_size,
            n_layers=n_layers,
            dropout=dropout,
        )

    # ── Internal components ──────────────────────────────────────────────────

    def _encode_window(self, window: Tensor) -> Tensor:
        """window: (B, W, S) → h: (B, gru_hidden)"""
        _, h_n = self.gru(window)   # h_n: (num_layers, B, gru_hidden)
        return h_n[-1]              # last layer: (B, gru_hidden)

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

    def _encode_latent(self, target: Tensor, h: Tensor, cond_emb: Tensor):
        """
        target: (B, S), h: (B, H)
        → mu, logvar: each (B, L)
        """
        feat = self.encoder(torch.cat([target, h, cond_emb], dim=-1))
        return self.fc_mu(feat), torch.clamp(self.fc_logvar(feat), -10, 10)

    def _decode(self, z: Tensor, h: Tensor, cond_emb: Tensor) -> Tensor:
        """z: (B, L), h: (B, H) → y_hat: (B, S)"""
        return self.decoder(torch.cat([z, h, cond_emb], dim=-1))

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Reparameterization trick: z = mu + eps * std, eps ~ N(0,I)"""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    def _flow_sample(self, h: Tensor, cond_emb: Tensor, n: int) -> Tensor:
        """
        Euler integration from t=0 to t=1 in latent space, conditioned on h.

        h: (n, gru_hidden)
        → z_1: (n, latent_size) — decoded via VAE decoder afterwards
        """
        device = h.device
        z = torch.randn(n, self.latent_size, device=device)
        dt = 1.0 / self.n_sample_steps

        for i in range(self.n_sample_steps):
            t_val    = i * dt
            t_tensor = torch.full((n,), t_val, device=device)
            t_emb    = self.t_embed(t_tensor)
            v        = self.velocity(z, t_emb, torch.cat([h, cond_emb], dim=-1))
            z        = z + v * dt

        return z

    # ── BaseModel interface ──────────────────────────────────────────────────

    def loss(self, x, beta: float = 1.0, **kwargs) -> dict:
        """
        Joint VAE + latent flow matching loss.

        Args:
            x:    tuple (window, target)
                  window: (B, W, S) — historical context
                  target: (B, S)    — current day
            beta: KL weight (annealing via train.py)

        Returns:
            {'total': ..., 'mse_recon': ..., 'kl': ..., 'fm_loss': ...}
        """
        if len(x) == 3:
            window, target, cond = x
        else:
            window, target = x
            cond = None
        B = target.shape[0]

        # ── VAE forward ──────────────────────────────────────────────────────
        h              = self._encode_window(window)          # (B, H)
        cond_emb       = self._cond_embed(cond, B, target.device)
        h_cond         = torch.cat([h, cond_emb], dim=-1)
        mu, logvar     = self._encode_latent(target, h, cond_emb)       # (B, L)
        z              = self._reparameterize(mu, logvar)     # (B, L)
        y_hat          = self._decode(z, h, cond_emb)         # (B, S)

        mse_recon  = F.mse_loss(y_hat, target)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())   # (B, L)
        kl         = torch.clamp(kl_per_dim, min=self.free_bits).mean()

        # ── Flow matching in latent space ─────────────────────────────────────
        # Detach z so flow gradients don't propagate into VAE encoder
        z_target = z.detach()
        z_0      = torch.randn_like(z_target)

        t        = torch.rand(B, device=target.device)
        t_exp    = t.unsqueeze(-1)

        z_t      = (1 - t_exp) * z_0 + t_exp * z_target   # OT straight line
        target_v = z_target - z_0                           # constant velocity

        t_emb    = self.t_embed(t)
        v_pred   = self.velocity(z_t, t_emb, h_cond)   # h shared: GRU trained by both VAE and flow
        fm_loss  = F.mse_loss(v_pred, target_v)

        total = mse_recon + beta * kl + fm_loss
        return {"total": total, "mse_recon": mse_recon, "kl": kl, "fm_loss": fm_loss}

    @torch.no_grad()
    def sample(self, n: int, steps: int | None = None, method=None,
               start_day: int = 1) -> Tensor:
        """
        Generates n samples via autoregressive rollout from a zero window.

        Compatible with evaluate_model (returns Tensor (n, n_stations)).
        Includes warmup of window_size steps before collecting samples.
        """
        if steps is not None:
            self.n_sample_steps = steps

        device = next(self.parameters()).device
        window = torch.zeros(1, self.window_size, self.n_stations, device=device)

        # Warmup: let GRU state converge
        for i in range(self.window_size):
            day = (start_day - self.window_size + i - 1) % 365 + 1
            cond = self._make_day_cond(day, 1, device)
            h = self._encode_window(window)
            cond_emb = self._cond_embed(cond, 1, device)
            z = self._flow_sample(h, cond_emb, 1)
            y = self._decode(z, h, cond_emb)
            window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)

        samples = []
        log_every = max(1, n // 4)
        for i in range(n):
            if i > 0 and i % log_every == 0:
                print(f"  [ar_latent_fm] sampling step {i}/{n}...", flush=True)
            day = (start_day + i - 1) % 365 + 1
            cond = self._make_day_cond(day, 1, device)
            h = self._encode_window(window)
            cond_emb = self._cond_embed(cond, 1, device)
            z = self._flow_sample(h, cond_emb, 1)
            y = self._decode(z, h, cond_emb)
            window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)
            samples.append(y)

        return torch.cat(samples, dim=0)  # (n, n_stations)

    @torch.no_grad()
    def sample_rollout(
        self,
        seed_window: Tensor,
        n_days: int,
        n_scenarios: int = 10,
        start_day: int = 1,
    ) -> Tensor:
        """
        Generates multiple scenarios via autoregressive rollout.

        All scenarios share the same seed_window but diverge via
        independent latent flow samples at each step.

        Args:
            seed_window: (W, S) — initial historical window (normalized)
            n_days:      number of days to generate
            n_scenarios: number of parallel scenarios
            start_day:   day-of-year (1–366) of the first generated day

        Returns:
            Tensor (n_scenarios, n_days, n_stations)
        """
        device = next(self.parameters()).device

        # Replicate seed for all scenarios
        window = (
            seed_window.to(device)
            .unsqueeze(0)
            .expand(n_scenarios, -1, -1)
            .clone()
        )   # (n_scenarios, W, n_stations)

        days = []
        for i in range(n_days):
            day = (start_day + i - 1) % 365 + 1
            cond = self._make_day_cond(day, n_scenarios, device)
            h = self._encode_window(window)          # (n_sc, H)
            cond_emb = self._cond_embed(cond, n_scenarios, device)
            z = self._flow_sample(h, cond_emb, n_scenarios)    # (n_sc, L)
            y = self._decode(z, h, cond_emb)                   # (n_sc, S)
            window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)
            days.append(y)

        return torch.stack(days, dim=1)  # (n_scenarios, n_days, n_stations)
