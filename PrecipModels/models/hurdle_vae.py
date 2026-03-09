"""
hurdle_vae.py — Hurdle VAE: duas sub-redes VAE (ocorrência + quantidade)

Mesma lógica do Hurdle Simples mas com VAE em cada parte — captura distribuições
arbitrárias e correlações implicitamente via espaço latente.

Pergunta respondida:
    "Separar ocorrência de quantidade ajuda o VAE?"

Arquitetura:
    OccurrenceVAE (latent_size=32):
        Input: máscara binária o ∈ {0,1}^S
        Decoder: Sigmoid → probabilidades por estação
        Loss: BCE(p_hat, o) + beta * KL

    AmountVAE (latent_size=64):
        Input: [amount_masked ‖ occ_mask] → shape (2S,) (condicionado na ocorrência)
        Decoder: ReLU → quantidades
        Loss: MSE apenas nas estações úmidas (evita distorção de gradientes)

Geração (.sample(n)):
    1. z_o ~ N(0,I) → OccurrenceVAE.decode → p_rain → Bernoulli → occ_sim
    2. z_a ~ N(0,I) → AmountVAE.decode([zeros, occ_sim]) → a_raw
    3. retorna ReLU(a_raw) * occ_sim
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel


class _MiniVAE(nn.Module):
    """Sub-VAE genérico com encoder 3-camadas e decoder 3-camadas."""

    def __init__(self, input_size: int, latent_size: int, output_activation: str = "none",
                 output_size: int = None):
        super().__init__()
        out_size = output_size if output_size is not None else input_size
        h = max(latent_size * 2, input_size)

        self.encoder = nn.Sequential(
            nn.Linear(input_size, h),
            nn.LeakyReLU(),
            nn.Linear(h, h),
            nn.LeakyReLU(),
        )
        self.fc_mu = nn.Linear(h, latent_size)
        self.fc_logvar = nn.Linear(h, latent_size)

        dec_layers = [
            nn.Linear(latent_size, h),
            nn.LeakyReLU(),
            nn.Linear(h, h),
            nn.LeakyReLU(),
            nn.Linear(h, out_size),
        ]
        if output_activation == "sigmoid":
            dec_layers.append(nn.Sigmoid())
        elif output_activation == "relu":
            dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: Tensor):
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 10)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar.clamp(-10, 10))
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

    def decode(self, z: Tensor) -> Tensor:
        return self.decoder(z)


class HurdleVAE(BaseModel):
    """
    Dois VAEs especializados: um para ocorrência binária, um para quantidade.
    """

    def __init__(
        self,
        input_size: int = 15,
        latent_occ: int = 32,
        latent_amt: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_occ = latent_occ
        self.latent_amt = latent_amt

        # OccurrenceVAE: entrada binária (S,) → saída probabilidades (S,)
        self.occ_vae = _MiniVAE(input_size, latent_occ, output_activation="none")

        # AmountVAE: entrada [amount | occ_mask] (2S,) → saída quantidades (S,)
        self.amt_vae = _MiniVAE(input_size * 2, latent_amt, output_activation="relu",
                                output_size=input_size)

    def loss(self, x: Tensor, beta: float = 1.0) -> dict:
        """
        Loss combinada: BCE_occ + beta*KL_occ + MSE_wet + beta*KL_amt

        Args:
            x: (B, S) — dados normalizados
        """
        B, S = x.shape

        # --- Ocorrência ---
        occ_target = (x > 0).float()
        p_hat, mu_o, logvar_o = self.occ_vae(occ_target)
        bce = F.binary_cross_entropy_with_logits(p_hat, occ_target, reduction='mean')
        kl_occ = -0.5 * torch.mean(1 + logvar_o - mu_o.pow(2) - logvar_o.exp())

        # --- Quantidade (condicionado na ocorrência) ---
        x_masked = x * occ_target  # zera os dias secos
        amt_input = torch.cat([x_masked, occ_target], dim=1)  # (B, 2S)
        x_hat_amt, mu_a, logvar_a = self.amt_vae(amt_input)
        kl_amt = -0.5 * torch.mean(1 + logvar_a - mu_a.pow(2) - logvar_a.exp())

        # MSE apenas nas estações úmidas por dia
        wet_mask = occ_target.bool()
        if wet_mask.any():
            mse_wet = F.mse_loss(x_hat_amt[wet_mask], x[wet_mask], reduction='mean')
        else:
            mse_wet = torch.tensor(0.0, device=x.device)

        total = bce + mse_wet + beta * (kl_occ + kl_amt)
        return {
            'total': total,
            'bce': bce,
            'mse_wet': mse_wet,
            'kl_occ': kl_occ,
            'kl_amt': kl_amt,
        }

    def sample(self, n: int, steps: int | None = None, method: str | None = None) -> Tensor:
        """
        Geração em dois passos:
            1. Ocorrência: z_o ~ N(0,I) → OccurrenceVAE → Bernoulli
            2. Quantidade: z_a ~ N(0,I) → AmountVAE([zeros, occ]) → ReLU
        """
        device = next(self.parameters()).device

        with torch.no_grad():
            # Passo 1: ocorrência
            z_o = torch.randn(n, self.latent_occ, device=device)
            p_rain = torch.sigmoid(self.occ_vae.decode(z_o))  # (n, S) — probabilidades
            occ_sim = torch.bernoulli(p_rain)   # (n, S) — binário

            # Passo 2: quantidade
            zeros = torch.zeros(n, self.input_size, device=device)
            amt_input = torch.cat([zeros, occ_sim], dim=1)  # (n, 2S)
            z_a = torch.randn(n, self.latent_amt, device=device)
            a_raw = self.amt_vae.decode(z_a)    # (n, S)

            # Combina: apenas estações com ocorrência recebem quantidade
            precip = F.relu(a_raw) * occ_sim
        return precip
