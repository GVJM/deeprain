"""
hurdle_ar_vae.py — Hurdle AR VAE: zero-inflation explícita + contexto GRU

Combina a separação ocorrência/quantidade do HurdleVAE com o contexto
autorregressivo do ARVAE.

Arquitetura:
    GRU:   window(B,W,S) → h(B,gru_hidden)   [idêntico ao ARVAE]

    Cabeça 1 — Ocorrência condicionada no histórico:
        sigmoid(MLP(h)) → p_rain (B, S)
        Loss: BCE(p_rain, y > 0)

    Cabeça 2 — Quantidade VAE condicionada em (y_wet, h):
        Encoder: [y_wet (S), h (H)] → (mu_z, logvar_z) (latent_size,)
        Decoder: [z (latent_size), h (H)] → y_hat (S,)
        Loss: MSE(y_hat_wet, y_wet) + beta * KL

Rollout autorregressivo (geração):
    h_t = GRU(window_{t-W:t-1})
    occ(t) = Bernoulli(sigmoid(MLP(h_t)))
    if occ(t).any():
        z ~ N(0, I)
        amt(t) = ReLU(Decoder([z, h_t]))
    y(t) = occ(t) * amt(t)
    window ← shift + append y(t)

Vantagens sobre ARVAE:
    - ~60-70% dias secos: modelar ocorrência explicitamente melhora wet_day_freq_error
    - ARVAE usa ReLU para não-negatividade, ignorando zero-inflation
    - Contexto GRU torna tanto p_rain quanto qty condicionais no histórico
"""

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel


class _OccurrenceMLP(nn.Module):
    """
    MLP que prediz probabilidade de chuva dado contexto GRU.

    h (gru_hidden,) → p_rain (n_stations,) ∈ (0,1)
    """

    def __init__(self, gru_hidden: int, n_stations: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(gru_hidden, hidden),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden, hidden // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden // 2, n_stations),
        )

    def forward(self, h: Tensor) -> Tensor:
        """h: (B, gru_hidden) → logits: (B, n_stations)"""
        return self.net(h)


class HurdleARVAE(BaseModel):
    """
    Hurdle AR VAE: zero-inflation + contexto GRU + VAE de quantidade.

    Interface com train.py:
        loss(x, beta) aceita x como tupla (window, target) —
        o train_neural_model_temporal passa pares do TemporalDataset.

    Interface com evaluate_model:
        sample(n) faz rollout de n passos a partir de janela zero.

    Método principal de geração:
        sample_rollout(seed_window, n_days, n_scenarios)
        → Tensor (n_scenarios, n_days, n_stations)
    """

    def __init__(
        self,
        input_size: int = 90,
        window_size: int = 30,
        gru_hidden: int = 128,
        latent_size: int = 64,
        hidden_size: int = 256,
        **kwargs,
    ):
        """
        Args:
            input_size:  número de estações (S)
            window_size: tamanho da janela histórica (W)
            gru_hidden:  dimensão oculta do GRU
            latent_size: dimensão do espaço latente z (quantidade)
            hidden_size: dimensão das camadas ocultas
        """
        super().__init__()
        self.n_stations  = input_size
        self.window_size = window_size
        self.gru_hidden  = gru_hidden
        self.latent_size = latent_size

        # ── GRU: comprime (W, S) → h (gru_hidden,) ──────────────────────────
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # ── Cabeça 1: ocorrência condicionada em h ────────────────────────────
        self.occ_mlp = _OccurrenceMLP(gru_hidden, input_size, hidden_size // 2)

        # ── Cabeça 2: VAE de quantidade condicionado em (y_wet, h) ───────────
        enc_in = input_size + gru_hidden    # [y_wet || h]
        self.amt_encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1),
        )
        self.fc_mu     = nn.Linear(hidden_size // 2, latent_size)
        self.fc_logvar = nn.Linear(hidden_size // 2, latent_size)

        dec_in = latent_size + gru_hidden   # [z || h]
        self.amt_decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, input_size),
            nn.ReLU(),  # quantidades não-negativas
        )

    # ── Componentes internos ─────────────────────────────────────────────────

    def _encode_window(self, window: Tensor) -> Tensor:
        """window: (B, W, S) → h: (B, gru_hidden)"""
        _, h_n = self.gru(window)
        return h_n[-1]

    def _encode_amount(self, y_wet: Tensor, h: Tensor):
        """
        y_wet: (B, S) — target mascarado (zeros nas estações secas)
        h:     (B, gru_hidden)
        → mu, logvar: (B, latent_size)
        """
        inp  = torch.cat([y_wet, h], dim=-1)
        feat = self.amt_encoder(inp)
        mu     = self.fc_mu(feat)
        logvar = self.fc_logvar(feat).clamp(-10, 10)
        return mu, logvar

    def _decode_amount(self, z: Tensor, h: Tensor) -> Tensor:
        """
        z: (B, latent_size), h: (B, gru_hidden)
        → y_hat: (B, S) — não-negativo via ReLU
        """
        return self.amt_decoder(torch.cat([z, h], dim=-1))

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    # ── Interface BaseModel ──────────────────────────────────────────────────

    def loss(self, x, beta: float = 1.0) -> dict:
        """
        Loss combinada: BCE (ocorrência) + MSE_wet + beta * KL (quantidade)

        Args:
            x:    tupla (window, target)
                  window: (B, W, S) — contexto histórico
                  target: (B, S)    — dia alvo
            beta: peso KL (annealing via train.py)

        Returns:
            {'total': ..., 'bce': ..., 'mse_wet': ..., 'kl': ...}
        """
        window, target = x
        h = self._encode_window(window)   # (B, gru_hidden)

        # --- Ocorrência ---
        occ_target = (target > 0).float()
        occ_logits = self.occ_mlp(h)
        bce = F.binary_cross_entropy_with_logits(occ_logits, occ_target, reduction='mean')

        # --- Quantidade (VAE condicionado em h + y_wet) ---
        y_wet  = target * occ_target        # mascarar dias secos
        mu, logvar = self._encode_amount(y_wet, h)
        z      = self._reparameterize(mu, logvar)
        y_hat  = self._decode_amount(z, h)

        # MSE apenas nas estações úmidas
        wet_mask = occ_target.bool()
        if wet_mask.any():
            mse_wet = F.mse_loss(y_hat[wet_mask], target[wet_mask])
        else:
            mse_wet = torch.tensor(0.0, device=target.device)

        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = bce + mse_wet + beta * kl
        return {'total': total, 'bce': bce, 'mse_wet': mse_wet, 'kl': kl}

    @torch.no_grad()
    def sample(self, n: int, steps=None, method=None) -> Tensor:
        """
        Gera n amostras via rollout autorregressivo a partir de janela zero.

        Compatível com evaluate_model (retorna Tensor (n, n_stations)).
        """
        device = next(self.parameters()).device
        window = torch.zeros(1, self.window_size, self.n_stations, device=device)

        # Warmup
        for _ in range(self.window_size):
            h        = self._encode_window(window)
            p_rain   = torch.sigmoid(self.occ_mlp(h))
            occ      = torch.bernoulli(p_rain)
            z        = torch.randn(1, self.latent_size, device=device)
            amt      = self._decode_amount(z, h)
            y        = occ * amt
            window   = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)

        samples = []
        for _ in range(n):
            h        = self._encode_window(window)
            p_rain   = torch.sigmoid(self.occ_mlp(h))
            occ      = torch.bernoulli(p_rain)
            z        = torch.randn(1, self.latent_size, device=device)
            amt      = self._decode_amount(z, h)
            y        = occ * amt
            window   = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)
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

        window = (
            seed_window.to(device)
            .unsqueeze(0)
            .expand(n_scenarios, -1, -1)
            .clone()
        )   # (n_scenarios, W, n_stations)

        days = []
        for _ in range(n_days):
            h      = self._encode_window(window)                       # (n_sc, gru_hidden)
            p_rain = torch.sigmoid(self.occ_mlp(h))                   # (n_sc, S)
            occ    = torch.bernoulli(p_rain)                           # (n_sc, S)
            z      = torch.randn(n_scenarios, self.latent_size, device=device)
            amt    = self._decode_amount(z, h)                         # (n_sc, S)
            y      = occ * amt                                         # (n_sc, S)
            window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)
            days.append(y)

        return torch.stack(days, dim=1)   # (n_scenarios, n_days, n_stations)
