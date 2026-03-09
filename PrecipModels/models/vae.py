"""
vae.py — VAE limpo com a melhor configuração dos experimentos em VAE_Tests/

Referência direta ao BaseModel com interface .sample(n) e .loss(x, beta).

Arquitetura:
    Encoder: input → latent*4 → latent*2 → (mu, logvar)  [LeakyReLU]
    Decoder: latent → latent*2 → latent*4 → input         [LeakyReLU + ReLU]

Defaults da melhor configuração encontrada em VAE_Tests/run.py:
    latent_size=128, output_activation='relu', kl_warmup=25, lr=0.005

Pergunta respondida:
    "Um espaço latente contínuo melhora sobre distribuições paramétricas?"
"""

import torch
import torch.nn as nn
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel


class VAEModel(BaseModel):
    """
    Variational Autoencoder para dados de precipitação.

    O KL annealing (aquecimento gradual de beta 0→1) é controlado externamente
    pelo train.py via loss(x, beta=beta).
    """

    def __init__(
        self,
        input_size: int = 15,
        latent_size: int = 128,
        output_activation: str = "relu",
        **kwargs,
    ):
        """
        Args:
            input_size: número de estações (colunas dos dados)
            latent_size: dimensão do espaço latente
            output_activation: 'relu' (não-negativo, recomendado) ou 'none'
        """
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_activation = output_activation

        # --- Encoder ---
        self.encoder = nn.Sequential(
            nn.Linear(input_size, latent_size * 4),
            nn.LeakyReLU(),
            nn.Linear(latent_size * 4, latent_size * 2),
            nn.LeakyReLU(),
            nn.Linear(latent_size * 2, latent_size),
        )

        # Cabeças mu e logvar
        self.fc_mu = nn.Linear(latent_size, latent_size)
        self.fc_logvar = nn.Linear(latent_size, latent_size)

        # --- Decoder ---
        decoder_layers = [
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

        self.criterion = nn.MSELoss(reduction='mean')

    def encode(self, x: Tensor):
        """x → (mu, logvar)"""
        h = self.encoder(x)
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), -10, 10)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Truque de reparametrização: z = mu + eps * std, eps ~ N(0,I)"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x: Tensor):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(z)
        return x_hat, mu, logvar

    def loss(self, x: Tensor, beta: float = 1.0) -> dict:
        """
        Loss VAE: MSE (reconstrução) + beta * KL

        Args:
            x: batch (B, input_size)
            beta: peso do KL — aumentado gradualmente pelo train.py (KL annealing)

        Returns:
            {'total': ..., 'recons': ..., 'kl': ...}
        """
        x_hat, mu, logvar = self.forward(x)
        recons = self.criterion(x_hat, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recons + beta * kl
        return {'total': total, 'recons': recons, 'kl': kl}

    def sample(self, n: int, steps: int | None = None, method: str | None = None) -> Tensor:
        """
        Gera n amostras: z ~ N(0, I) → decoder

        Returns:
            Tensor (n, input_size) no espaço normalizado
        """
        device = next(self.parameters()).device
        z = torch.randn(n, self.latent_size, device=device)
        with torch.no_grad():
            return self.decoder(z)
