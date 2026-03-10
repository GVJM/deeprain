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

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel


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
        **kwargs,
    ):
        """
        Args:
            input_size:  número de estações (S)
            window_size: tamanho da janela histórica (W)
            gru_hidden:  dimensão oculta do GRU
            latent_size: dimensão do espaço latente z
            hidden_size: dimensão das camadas ocultas do encoder/decoder
        """
        super().__init__()
        self.n_stations  = input_size
        self.window_size = window_size
        self.gru_hidden  = gru_hidden
        self.latent_size = latent_size

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
        enc_in = input_size + gru_hidden
        self.encoder = nn.Sequential(
            nn.Linear(enc_in, hidden_size),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.LeakyReLU(0.1),
        )
        self.fc_mu     = nn.Linear(hidden_size // 2, latent_size)
        self.fc_logvar = nn.Linear(hidden_size // 2, latent_size)

        # ── Decoder: [z (latent_size), h (gru_hidden)] → y_hat (S,) ─────────
        dec_in = latent_size + gru_hidden
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

    def encode(self, x: Tensor, h: Tensor):
        """
        x: (B, S), h: (B, gru_hidden)
        → mu, logvar: cada (B, latent_size)
        """
        inp  = torch.cat([x, h], dim=-1)
        feat = self.encoder(inp)
        return self.fc_mu(feat), torch.clamp(self.fc_logvar(feat), -10, 10)

    def decode(self, z: Tensor, h: Tensor) -> Tensor:
        """
        z: (B, latent_size), h: (B, gru_hidden)
        → x_hat: (B, S)  — valores não-negativos via ReLU final
        """
        inp = torch.cat([z, h], dim=-1)
        return self.decoder(inp)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Truque de reparametrização: z = mu + eps * std, eps ~ N(0,I)"""
        std = torch.exp(0.5 * logvar)
        return mu + torch.randn_like(std) * std

    # ── Interface BaseModel ──────────────────────────────────────────────────

    def loss(self, x, beta: float = 1.0) -> dict:
        """
        Calcula VAE loss condicional: MSE(y_t, y_hat) + beta * KL(q||N(0,I))

        Args:
            x:    tupla (window, target)
                  window: (B, W, S) — contexto histórico
                  target: (B, S)    — dia alvo
            beta: peso KL (annealing via train.py)

        Returns:
            {'total': ..., 'mse': ..., 'kl': ...}
        """
        window, target = x
        h              = self._encode_window(window)    # (B, gru_hidden)
        mu, logvar     = self.encode(target, h)
        z              = self.reparameterize(mu, logvar)
        x_hat          = self.decode(z, h)

        mse   = F.mse_loss(x_hat, target)
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = mse + beta * kl
        return {"total": total, "mse": mse, "kl": kl}

    @torch.no_grad()
    def sample(self, n: int, steps=None, method=None) -> Tensor:
        """
        Gera n amostras via rollout autorregressivo a partir de janela zero.

        Compatível com evaluate_model (retorna Tensor (n, n_stations)).
        Inclui warmup de 30 passos para sair do estado zero antes de coletar.

        Args:
            n: número de amostras a retornar

        Returns:
            Tensor (n, n_stations) — espaço normalizado
        """
        device = next(self.parameters()).device
        window = torch.zeros(1, self.window_size, self.n_stations, device=device)

        # Warmup: deixa o estado interno do GRU convergir
        for _ in range(self.window_size):
            h = self._encode_window(window)
            z = torch.randn(1, self.latent_size, device=device)
            y = self.decode(z, h)
            window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)

        samples = []
        for _ in range(n):
            h = self._encode_window(window)
            z = torch.randn(1, self.latent_size, device=device)
            y = self.decode(z, h)
            window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)
            samples.append(y)

        return torch.cat(samples, dim=0)  # (n, n_stations)

    @torch.no_grad()
    def sample_rollout(
        self,
        seed_window: Tensor,
        n_days: int,
        n_scenarios: int = 10,
    ) -> Tensor:
        """
        Gera múltiplos cenários via rollout autorregressivo.

        Todos os cenários partem da mesma seed_window mas divergem via
        z ~ N(0, I) amostrado independentemente a cada passo.

        Args:
            seed_window: (W, S) tensor — janela histórica inicial (normalizada)
            n_days:      número de dias a gerar
            n_scenarios: número de cenários paralelos

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
        for _ in range(n_days):
            h = self._encode_window(window)                         # (n_sc, gru_hidden)
            z = torch.randn(n_scenarios, self.latent_size, device=device)
            y = self.decode(z, h)                                   # (n_sc, n_stations)
            window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)
            days.append(y)

        return torch.stack(days, dim=1)  # (n_scenarios, n_days, n_stations)
