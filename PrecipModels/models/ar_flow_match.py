"""
ar_flow_match.py — AR Flow Matching: GRU context + conditional flow matching

Architecture:
    GRU:  window(B,W,S) → h(B,gru_hidden)   [identical to ARVAE]
    Flow: v_θ(z_t, t, h) integrates z_0~N(0,I) → z_1 ≈ y

    Velocity network input: [z_t (S) || t_embed (T) || h (gru_hidden)]

Loss (conditional OT flow matching):
    z_0 ~ N(0,I),  t ~ U(0,1)
    z_t = (1 - t) * z_0 + t * y      (straight-line interpolation)
    target = y - z_0                   (constant velocity)
    MSE(v_θ(z_t, t, h), target)
    — No KL divergence → no posterior collapse.

Sampling (Euler integration conditioned on h):
    h = GRU(window)
    z = z_0 ~ N(0, I)
    for t in {0, dt, 2dt, ..., 1-dt}:
        z += v_θ(z, t, h) * dt
    y_hat = clip(z, 0, ∞)             (physical non-negativity)

Rollout (same pattern as ARVAE):
    h_t = GRU(window_{t-W:t-1})
    y(t) = flow_sample | h_t
    window ← shift + append y(t)

Advantages over ARVAE:
    - No KL collapse (VAE's main failure mode)
    - Better coverage: flow matching explores the full output manifold
    - Reutiliza SinusoidalEmbedding, TemporalDataset, train_neural_model_temporal
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


class _CondVelocityMLP(nn.Module):
    """
    Rede de velocidade condicional para AR Flow Matching.

    Input: concatenação de [z_t (S), t_embed (T), h (H)] = S + T + H
    Output: velocidade estimada (S,)
    """

    def __init__(self, data_dim: int, t_embed_dim: int, gru_hidden: int,
                 hidden: int = 256, n_layers: int = 4):
        super().__init__()
        in_dim = data_dim + t_embed_dim + gru_hidden
        layers = [nn.Linear(in_dim, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        layers.append(nn.Linear(hidden, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, z_t: Tensor, t_emb: Tensor, h: Tensor) -> Tensor:
        """
        z_t:   (B, S)
        t_emb: (B, T)
        h:     (B, H)
        → velocidade: (B, S)
        """
        inp = torch.cat([z_t, t_emb, h], dim=-1)
        return self.net(inp)


class ARFlowMatch(BaseModel):
    """
    AR Flow Matching: contexto GRU + flow matching condicional.

    Interface com train.py:
        loss(x, beta) aceita x como tupla (window, target) —
        o train_neural_model_temporal passa pares do TemporalDataset.
        beta é ignorado (sem KL).

    Interface com evaluate_model:
        sample(n) faz rollout de n passos a partir de janela zero.

    Geração principal:
        sample_rollout(seed_window, n_days, n_scenarios)
        → Tensor (n_scenarios, n_days, n_stations)
    """

    def __init__(
        self,
        input_size: int = 90,
        window_size: int = 30,
        gru_hidden: int = 128,
        hidden_size: int = 256,
        n_layers: int = 4,
        t_embed_dim: int = 64,
        n_sample_steps: int = 50,
        **kwargs,
    ):
        """
        Args:
            input_size:     número de estações (S)
            window_size:    tamanho da janela histórica (W)
            gru_hidden:     dimensão oculta do GRU
            hidden_size:    dimensão das camadas ocultas do velocity MLP
            n_layers:       número de camadas ocultas no velocity MLP
            t_embed_dim:    dimensão do embedding sinusoidal de tempo
            n_sample_steps: passos de integração Euler na geração
        """
        super().__init__()
        self.n_stations     = input_size
        self.window_size    = window_size
        self.gru_hidden     = gru_hidden
        self.n_sample_steps = n_sample_steps

        # ── GRU: comprime (W, S) → h (gru_hidden,) ──────────────────────────
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # ── Embedding de tempo + velocity MLP condicional ─────────────────────
        self.t_embed  = SinusoidalEmbedding(t_embed_dim)
        self.velocity = _CondVelocityMLP(
            data_dim=input_size,
            t_embed_dim=t_embed_dim,
            gru_hidden=gru_hidden,
            hidden=hidden_size,
            n_layers=n_layers,
        )

    # ── Componentes internos ─────────────────────────────────────────────────

    def _encode_window(self, window: Tensor) -> Tensor:
        """
        window: (B, W, S) → h: (B, gru_hidden)
        Retorna hidden state da última camada do GRU no último timestep.
        """
        _, h_n = self.gru(window)   # h_n: (num_layers, B, gru_hidden)
        return h_n[-1]              # (B, gru_hidden)

    def _flow_sample(self, h: Tensor, n: int) -> Tensor:
        """
        Integração Euler de t=0 a t=1 condicionada em h.

        h: (n, gru_hidden) — pode ser batched com n cenários
        → y_hat: (n, n_stations)
        """
        device = h.device
        z = torch.randn(n, self.n_stations, device=device)
        dt = 1.0 / self.n_sample_steps

        for i in range(self.n_sample_steps):
            t_val = i * dt
            t_tensor = torch.full((n,), t_val, device=device)
            t_emb = self.t_embed(t_tensor)
            v = self.velocity(z, t_emb, h)
            z = z + v * dt

        return z.clamp(min=0.0)  # precipitação não-negativa

    # ── Interface BaseModel ──────────────────────────────────────────────────

    def loss(self, x, beta: float = 1.0) -> dict:
        """
        Conditional OT Flow Matching loss.

        Args:
            x:    tupla (window, target)
                  window: (B, W, S) — contexto histórico
                  target: (B, S)    — dia alvo
            beta: ignorado (sem KL — não é um VAE)

        Returns:
            {'total': ..., 'fm_loss': ...}
        """
        window, target = x
        B = target.shape[0]

        h   = self._encode_window(window)   # (B, gru_hidden)
        z_0 = torch.randn_like(target)
        z_1 = target

        t     = torch.rand(B, device=target.device)
        t_exp = t.unsqueeze(-1)

        # Trajetória reta OT
        z_t    = (1 - t_exp) * z_0 + t_exp * z_1
        target_v = z_1 - z_0   # velocidade constante (alvo)

        t_emb  = self.t_embed(t)
        v_pred = self.velocity(z_t, t_emb, h)

        fm_loss = F.mse_loss(v_pred, target_v)
        return {'total': fm_loss, 'fm_loss': fm_loss}

    @torch.no_grad()
    def sample(self, n: int, steps: int | None = None, method: str | None = None) -> Tensor:
        """
        Gera n amostras via rollout autorregressivo a partir de janela zero.

        Compatível com evaluate_model (retorna Tensor (n, n_stations)).
        """
        if steps is not None:
            self.n_sample_steps = steps

        device = next(self.parameters()).device
        window = torch.zeros(1, self.window_size, self.n_stations, device=device)

        # Warmup: deixa GRU convergir
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

        return torch.cat(samples, dim=0)   # (n, n_stations)

    @torch.no_grad()
    def sample_rollout(
        self,
        seed_window: Tensor,
        n_days: int,
        n_scenarios: int = 10,
    ) -> Tensor:
        """
        Gera múltiplos cenários via rollout autorregressivo.

        Args:
            seed_window: (W, S) — janela histórica inicial (normalizada)
            n_days:      número de dias a gerar
            n_scenarios: número de cenários paralelos

        Returns:
            Tensor (n_scenarios, n_days, n_stations)
        """
        device = next(self.parameters()).device

        # Replica seed para todos os cenários
        window = (
            seed_window.to(device)
            .unsqueeze(0)
            .expand(n_scenarios, -1, -1)
            .clone()
        )   # (n_scenarios, W, n_stations)

        days = []
        for _ in range(n_days):
            h = self._encode_window(window)                   # (n_sc, gru_hidden)
            y = self._flow_sample(h, n_scenarios)             # (n_sc, n_stations)
            window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)
            days.append(y)

        return torch.stack(days, dim=1)   # (n_scenarios, n_days, n_stations)
