"""
ar_vae.py — VAE Autorregressivo (ARVAE) para séries temporais de precipitação.

Arquitetura:
    GRU comprime janela histórica [y(t-W), ..., y(t-1)] → contexto h
    Encoder: [y_t, h] → (mu_z, logvar_z)
    Decoder: [z, h]   → y_hat(t)

Treino (teacher forcing com TemporalDataset):
    window [y(t-W),...,y(t-1)] → GRU → h
    z ~ q(z | y_t, h)
    y_hat = Decoder(z, h)
    Loss = MSE(y_t, y_hat) + beta * KL(q || N(0,I))

Rollout autorregressivo (geração):
    h_t = GRU(window_{t-W:t-1})
    z ~ N(0, I)
    y(t) = Decoder(z, h_t)
    window ← shift + append y(t)
    → repete por N dias

Por que VAE sobre MLP puro:
    - z ~ N(0,I) garante diversidade entre cenários dado o mesmo contexto
    - KL evita posterior collapse (todos os z convergem para o mesmo ponto)
    - Espaço latente captura variabilidade residual não explicada pelo histórico

Uso:
    model = ARVAE(input_size=90, window_size=30, gru_hidden=128, latent_size=64)
    # Treino (via train.py):
    loss = model.loss((window_batch, target_batch), beta=0.5)
    # Geração:
    scenarios = model.sample_rollout(seed_window, n_days=365, n_scenarios=50)
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
from models.conditioning import ConditioningBlock, DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS


class ARVAE(BaseModel):
    """
    VAE Autorregressivo para séries temporais de precipitação.

    Interface com train.py:
        loss(x, beta) aceita x como tupla (window, target) —
        o train_neural_model_temporal passa pares do TemporalDataset.

    Interface com evaluate_model:
        sample(n) faz rollout de n passos a partir de janela zero —
        retorna Tensor (n, n_stations) compatível com métricas i.i.d.

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
        occ_weight: float = 0.0,    # new: default 0 = disabled (backward compatible)
        **kwargs,
    ):
        """
        Args:
            input_size:  número de estações (S)
            window_size: tamanho da janela histórica (W)
            gru_hidden:  dimensão oculta do GRU
            latent_size: dimensão do espaço latente z
            hidden_size: dimensão das camadas ocultas do encoder/decoder
            occ_weight:  weight for occurrence BCE loss (0 = disabled)
        """
        super().__init__()
        self.n_stations  = input_size
        self.window_size = window_size
        self.gru_hidden  = gru_hidden
        self.latent_size = latent_size
        self.occ_weight  = occ_weight
        if occ_weight > 0:
            self.threshold = nn.Parameter(torch.zeros(input_size))
        self.cond_block  = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        self.cond_dim    = self.cond_block.total_dim

        # ── GRU: comprime (W, S) → h (gru_hidden,) ──────────────────────────
        # batch_first=True: entrada (B, W, S), saída (B, W, gru_hidden)
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=gru_hidden,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )

        # ── Encoder: [y_t (S), h (gru_hidden)] → (mu, logvar) ───────────────
        enc_in = input_size + gru_hidden + self.cond_dim
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1),
        )
        self.fc_mu     = nn.Linear(hidden_size // 2, latent_size)
        self.fc_logvar = nn.Linear(hidden_size // 2, latent_size)

        # ── Decoder: [z (latent_size), h (gru_hidden)] → y_hat (S,) ─────────
        dec_in = latent_size + gru_hidden + self.cond_dim
        self.decoder = nn.Sequential(
            nn.Linear(dec_in, hidden_size // 2),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size // 2, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, input_size),
            nn.ReLU(),  # precipitação não-negativa
        )

    # ── Componentes internos ─────────────────────────────────────────────────

    def _encode_window(self, window: Tensor) -> Tensor:
        """
        window: (B, W, S) → h: (B, gru_hidden)
        Retorna o hidden state da última camada do GRU no último timestep.
        """
        _, h_n = self.gru(window)   # h_n: (num_layers, B, gru_hidden)
        return h_n[-1]              # última camada: (B, gru_hidden)

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

    def encode(self, x: Tensor, h: Tensor, cond_emb: Tensor):
        """
        x: (B, S), h: (B, gru_hidden)
        → mu, logvar: cada (B, latent_size)
        """
        inp  = torch.cat([x, h, cond_emb], dim=-1)
        feat = self.encoder(inp)
        return self.fc_mu(feat), torch.clamp(self.fc_logvar(feat), -10, 10)

    def decode(self, z: Tensor, h: Tensor, cond_emb: Tensor) -> Tensor:
        """
        z: (B, latent_size), h: (B, gru_hidden)
        → x_hat: (B, S)  — valores não-negativos via ReLU final
        """
        inp = torch.cat([z, h, cond_emb], dim=-1)
        return self.decoder(inp)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Truque de reparametrização: z = mu + eps * std, eps ~ N(0,I)"""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    # ── Interface BaseModel ──────────────────────────────────────────────────

    def loss(self, x, beta: float = 1.0) -> dict:
        """
        Calcula VAE loss condicional: MSE(y_t, y_hat) + beta * KL(q||N(0,I)) [+ occ_weight * BCE]

        Args:
            x:    tupla (window, target, [cond])
                  window: (B, W, S) — contexto histórico
                  target: (B, S)    — dia alvo
            beta: peso KL (annealing via train.py)

        Returns:
            {'total': ..., 'mse': ..., 'kl': ..., 'occ': ...}
            occ is 0.0 when occ_weight=0 (disabled).
        """
        if len(x) == 3:
            window, target, cond = x
        else:
            window, target = x
            cond = None
        h              = self._encode_window(window)    # (B, gru_hidden)
        cond_emb       = self._cond_embed(cond, target.shape[0], target.device)
        mu, logvar     = self.encode(target, h, cond_emb)
        z              = self.reparameterize(mu, logvar)
        x_hat          = self.decode(z, h, cond_emb)

        mse   = F.mse_loss(x_hat, target)
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        occ_loss = torch.tensor(0.0, device=target.device)
        if self.occ_weight > 0:
            pred_logit = x_hat - self.threshold.abs()
            true_occ   = (target > 0).float()
            occ_loss   = F.binary_cross_entropy_with_logits(pred_logit, true_occ)
        total = mse + beta * kl + self.occ_weight * occ_loss
        return {"total": total, "mse": mse, "kl": kl, "occ": occ_loss}

    @torch.no_grad()
    def sample(self, n: int, steps=None, method=None, start_day: int = 1) -> Tensor:
        """
        Gera n amostras via rollout autorregressivo a partir de janela zero.

        Compatível com evaluate_model (retorna Tensor (n, n_stations)).
        Inclui warmup de 30 passos para sair do estado zero antes de coletar.

        Args:
            n:         número de amostras a retornar
            start_day: dia do ano (1–366) do primeiro passo de coleta

        Returns:
            Tensor (n, n_stations) — espaço normalizado
        """
        device = next(self.parameters()).device
        window = torch.zeros(1, self.window_size, self.n_stations, device=device)

        # Warmup: deixa o estado interno do GRU convergir
        for i in range(self.window_size):
            day = (start_day - self.window_size + i - 1) % 365 + 1
            cond = self._make_day_cond(day, 1, device)
            h = self._encode_window(window)
            z = torch.randn(1, self.latent_size, device=device)
            cond_emb = self._cond_embed(cond, 1, device)
            y = self.decode(z, h, cond_emb)
            if self.occ_weight > 0:
                y = y * (y > self.threshold.abs().detach())
            window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)

        samples = []
        log_every = max(1, n // 4)
        for i in range(n):
            if i > 0 and i % log_every == 0:
                print(f"  [ar_vae] sampling step {i}/{n}...", flush=True)
            day = (start_day + i - 1) % 365 + 1
            cond = self._make_day_cond(day, 1, device)
            h = self._encode_window(window)
            z = torch.randn(1, self.latent_size, device=device)
            cond_emb = self._cond_embed(cond, 1, device)
            y = self.decode(z, h, cond_emb)
            if self.occ_weight > 0:
                y = y * (y > self.threshold.abs().detach())
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
        Gera múltiplos cenários via rollout autorregressivo.

        Todos os cenários partem da mesma seed_window mas divergem via
        z ~ N(0, I) amostrado independentemente a cada passo.

        Args:
            seed_window: (W, S) tensor — janela histórica inicial (normalizada)
            n_days:      número de dias a gerar
            n_scenarios: número de cenários paralelos
            start_day:   dia do ano (1–366) do primeiro dia gerado

        Returns:
            Tensor (n_scenarios, n_days, n_stations)
        """
        device = next(self.parameters()).device

        # Replica seed para todos os cenários + clone para divergência
        # shape: (n_scenarios, W, n_stations)
        window = (
            seed_window.to(device)
            .unsqueeze(0)
            .expand(n_scenarios, -1, -1)
            .clone()
        )

        days = []
        for i in range(n_days):
            day = (start_day + i - 1) % 365 + 1
            cond = self._make_day_cond(day, n_scenarios, device)
            h = self._encode_window(window)                         # (n_sc, gru_hidden)
            z = torch.randn(n_scenarios, self.latent_size, device=device)
            cond_emb = self._cond_embed(cond, n_scenarios, device)
            y = self.decode(z, h, cond_emb)                         # (n_sc, n_stations)
            if self.occ_weight > 0:
                y = y * (y > self.threshold.abs().detach())
            window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)
            days.append(y)

        return torch.stack(days, dim=1)  # (n_scenarios, n_days, n_stations)
