"""
ldm.py — Latent Diffusion Model (LDM) para precipitação.

Treinamento em dois estágios:
  Stage 1 — VAE: aprende codificação latente compacta (padrão: 2000 épocas, latent=128)
  Stage 2 — DDPM no latente: aprende a distribuição latente com um denoiser MLP (padrão: 300 épocas)

Hipótese testada:
    "Um modelo de difusão no espaço latente de um VAE supera hurdle_simple em
     Wasserstein e correlação espacial?"

Resultados em VAE_Tests/best_v2_ldm (não comparados diretamente ao PrecipModels):
    Wasserstein = 1.287  vs  hurdle_simple = 1.603 → potencialmente melhor

Arquitetura:
    Encoder: input → latent*4 → latent*2 → (mu, logvar)   [LeakyReLU + ReLU output]
    Decoder: latent → latent*2 → latent*4 → input
    Denoiser: (z_noisy ‖ t_embed) → hidden*num_layers → latent  [DDPM / DDIM]
    Time embedding: sinusoidal → MLP(dim, dim*4, dim)

Uso com train.py:
    python train.py --model ldm                   # 2000 épocas VAE + 300 épocas DDPM
    python train.py --model ldm --max_epochs 500  # rápido para debug
"""

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel


# ──────────────────────────────────────────────────────────
# BLOCOS AUXILIARES
# ──────────────────────────────────────────────────────────

class _SinusoidalTimeEmbed(nn.Module):
    """Embedding sinusoidal para t contínuo ∈ [0,1] → vetor de dimensão `dim`."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)         # (B, 1)
        args = t * freqs                # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class _LatentDenoiser(nn.Module):
    """
    MLP denoiser para DDPM no espaço latente.

    Recebe (z_noisy ‖ t_embed) e prevê o ruído adicionado.
    """

    def __init__(
        self,
        latent_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        time_embed_dim: int = 64,
    ):
        super().__init__()
        self.time_embed = _SinusoidalTimeEmbed(time_embed_dim)

        in_dim = latent_size + time_embed_dim
        layers: list[nn.Module] = [nn.Linear(in_dim, hidden_size), nn.SiLU()]
        for _ in range(num_layers - 1):
            layers += [nn.Linear(hidden_size, hidden_size), nn.SiLU()]
        layers.append(nn.Linear(hidden_size, latent_size))

        self.net = nn.Sequential(*layers)

    def forward(self, z: Tensor, t_float: Tensor) -> Tensor:
        """
        z:       (B, latent) — latente ruidoso
        t_float: (B,) — tempo em [0,1]
        Returns: (B, latent) — ruído predito
        """
        t_emb = self.time_embed(t_float)    # (B, time_embed_dim)
        x = torch.cat([z, t_emb], dim=-1)  # (B, latent + time_embed_dim)
        return self.net(x)


# ──────────────────────────────────────────────────────────
# MODELO PRINCIPAL
# ──────────────────────────────────────────────────────────

class LDMModel(BaseModel):
    """
    Latent Diffusion Model: VAE encoder/decoder + DDPM no espaço latente.

    Treinamento em dois estágios (gerenciado pelo train.py):
      Stage 'vae'  — treina encoder/decoder com MSE + KL (como vae.py)
      Stage 'ldm'  — congela VAE, treina denoiser com DDPM loss nos latentes

    O método set_stage() congela/descongela os parâmetros adequados.
    A amostragem sempre usa DDPM reverse → decoder VAE.
    """

    def __init__(
        self,
        input_size: int = 15,
        latent_size: int = 128,
        output_activation: str = "none",
        ldm_timesteps: int = 100,
        ldm_hidden_size: int = 128,
        ldm_num_layers: int = 3,
        ldm_time_embed_dim: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.ldm_timesteps = ldm_timesteps
        self._training_stage = "vae"

        # ── VAE ──────────────────────────────────────────
        self.encoder = nn.Sequential(
            nn.Linear(input_size, latent_size * 4),
            nn.LeakyReLU(),
            nn.Linear(latent_size * 4, latent_size * 2),
            nn.LeakyReLU(),
            nn.Linear(latent_size * 2, latent_size),
        )
        self.fc_mu = nn.Linear(latent_size, latent_size)
        self.fc_logvar = nn.Linear(latent_size, latent_size)

        decoder_layers: list[nn.Module] = [
            nn.Linear(latent_size, latent_size * 2),
            nn.LeakyReLU(),
            nn.Linear(latent_size * 2, latent_size * 4),
            nn.LeakyReLU(),
            nn.Linear(latent_size * 4, input_size),
        ]
        if output_activation == "relu":
            decoder_layers.append(nn.ReLU())
        elif output_activation == "softplus":
            decoder_layers.append(nn.Softplus())
        self.decoder = nn.Sequential(*decoder_layers)

        self._vae_criterion = nn.MSELoss(reduction="mean")

        # ── DDPM denoiser ─────────────────────────────────
        self.denoiser = _LatentDenoiser(
            latent_size, ldm_hidden_size, ldm_num_layers, ldm_time_embed_dim
        )

        # Cosine noise schedule (buffers → salvo no state_dict)
        betas = self._cosine_beta_schedule(ldm_timesteps)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat(
            [torch.ones(1, dtype=alphas_cumprod.dtype), alphas_cumprod[:-1]], dim=0
        )
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.register_buffer("_betas", betas)
        self.register_buffer("_alphas", alphas)
        self.register_buffer("_alphas_cumprod", alphas_cumprod)
        self.register_buffer("_alphas_cumprod_prev", alphas_cumprod_prev)
        self.register_buffer("_sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        self.register_buffer("_sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        self.register_buffer("_posterior_variance", posterior_variance.clamp(min=1e-20))

        # Inicia com VAE stage (denoiser congelado)
        self.set_stage("vae")

    # ── Noise schedule ─────────────────────────────────────

    @staticmethod
    def _cosine_beta_schedule(timesteps: int, s: float = 0.008) -> Tensor:
        """Cosine schedule (Nichol & Dhariwal 2021) — suave em ambas as extremidades."""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * math.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1.0 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        # Evita passo reverso explosivo (beta muito próximo de 1).
        return torch.clamp(betas, 1e-4, 0.02)

    # ── Stage management ───────────────────────────────────

    def set_stage(self, stage: str):
        """
        Alterna entre estágio 'vae' e 'ldm'.
        - 'vae': treina encoder/decoder, congela denoiser
        - 'ldm': congela VAE, treina apenas denoiser
        """
        assert stage in ("vae", "ldm"), f"stage deve ser 'vae' ou 'ldm', recebeu '{stage}'"
        self._training_stage = stage

        vae_params = (
            list(self.encoder.parameters())
            + list(self.fc_mu.parameters())
            + list(self.fc_logvar.parameters())
            + list(self.decoder.parameters())
        )
        ldm_params = list(self.denoiser.parameters())

        if stage == "vae":
            for p in vae_params:
                p.requires_grad_(True)
            for p in ldm_params:
                p.requires_grad_(False)
        else:
            for p in vae_params:
                p.requires_grad_(False)
            for p in ldm_params:
                p.requires_grad_(True)

        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[LDM] Estágio: '{stage}' | parâmetros treináveis: {n_trainable:,}")

    # ── VAE internals ──────────────────────────────────────

    def _encode(self, x: Tensor):
        """x → (mu, logvar)"""
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = torch.clamp(self.fc_logvar(h), -10, 10)
        return mu, logvar

    def _reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    # ── Loss ───────────────────────────────────────────────

    def loss(self, x: Tensor, beta: float = 1.0) -> dict:
        """
        Estágio 'vae': MSE + KL (igual a vae.py)
        Estágio 'ldm': DDPM loss (predição de ruído no latente)
        """
        if self._training_stage == "vae":
            return self._vae_loss(x, beta)
        return self._ddpm_loss(x)

    def _vae_loss(self, x: Tensor, beta: float) -> dict:
        mu, logvar = self._encode(x)
        z = self._reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        recons = self._vae_criterion(x_hat, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recons + beta * kl
        return {"total": total, "recons": recons, "kl": kl}

    def _ddpm_loss(self, x: Tensor) -> dict:
        """
        DDPM loss: para cada amostra, codifica ao latente determinístico (mu),
        adiciona ruído num timestep aleatório e treina o denoiser para prevê-lo.
        """
        with torch.no_grad():
            mu, _ = self._encode(x)
        z0 = mu  # latente determinístico (B, latent)

        B = z0.shape[0]
        device = z0.device

        t_int = torch.randint(0, self.ldm_timesteps, (B,), device=device)
        t_float = t_int.float() / max(self.ldm_timesteps - 1, 1)

        noise = torch.randn_like(z0)
        sqrt_a = self._sqrt_alphas_cumprod[t_int].unsqueeze(1)       # (B, 1)
        sqrt_1a = self._sqrt_one_minus_alphas_cumprod[t_int].unsqueeze(1)
        z_noisy = sqrt_a * z0 + sqrt_1a * noise

        noise_pred = self.denoiser(z_noisy, t_float)
        diff_loss = F.mse_loss(noise_pred, noise)
        return {"total": diff_loss, "diffusion": diff_loss}

    def count_parameters(self) -> int:
        """Retorna total de parâmetros do modelo inteiro (VAE + denoiser)."""
        return sum(p.numel() for p in self.parameters())

    # ── Sampling ───────────────────────────────────────────

    @torch.no_grad()
    def sample(self, n: int, steps: int | None = None, method: str | None = None) -> Tensor:
        """
        Geração:
          1. z_T ~ N(0, I)  (ruído puro)
          2. Reverse DDPM/DDIM: T → 0 (denoising iterativo)
          3. Decode z_0 → x com VAE decoder

        Args:
            steps:  número de passos de denoising (None = ldm_timesteps completo).
                    Quando steps < ldm_timesteps, usa subsampling DDIM-style
                    (índices igualmente espaçados no schedule original).
            method: 'ddpm' (padrão, estocástico) ou 'ddim' (determinístico, sem ruído)

        Returns: (n, input_size) — espaço normalizado
        """
        num_steps = steps if steps is not None else self.ldm_timesteps
        solver = method or 'ddpm'

        device = next(self.parameters()).device
        z = torch.randn(n, self.latent_size, device=device)

        # Subconjunto de índices via interpolação linear no schedule original
        indices = np.round(
            np.linspace(0, self.ldm_timesteps - 1, num_steps)
        ).astype(int)

        for t_idx in reversed(indices):
            t_float = torch.full(
                (n,), t_idx / max(self.ldm_timesteps - 1, 1), device=device
            )

            noise_pred = self.denoiser(z, t_float)

            beta_t = self._betas[t_idx]
            alpha_t = self._alphas[t_idx]
            alpha_bar_t = self._alphas_cumprod[t_idx]
            alpha_bar_prev_t = self._alphas_cumprod_prev[t_idx]
            sqrt_one_minus = self._sqrt_one_minus_alphas_cumprod[t_idx]

            # Predição de x0 a partir de x_t e eps_theta(x_t, t)
            x0_pred = (z - sqrt_one_minus * noise_pred) / alpha_bar_t.sqrt()

            if solver == 'ddim':
                # DDIM determinístico (eta=0)
                z = alpha_bar_prev_t.sqrt() * x0_pred + (1.0 - alpha_bar_prev_t).sqrt() * noise_pred
            else:
                # Média posterior q(x_{t-1} | x_t, x0_pred)
                coef_x0 = (alpha_bar_prev_t.sqrt() * beta_t) / (1.0 - alpha_bar_t)
                coef_xt = (alpha_t.sqrt() * (1.0 - alpha_bar_prev_t)) / (1.0 - alpha_bar_t)
                mean = coef_x0 * x0_pred + coef_xt * z

                if t_idx > 0:
                    z = mean + self._posterior_variance[t_idx].sqrt() * torch.randn_like(z)
                else:
                    z = mean

        return self.decoder(z)
