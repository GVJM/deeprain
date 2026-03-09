"""
vae_mc.py — VAE com Condicionamento Mensal (nn.Embedding)

Variante condicionada de vae.py usando ConditioningBlock.
O embedding mensal é concatenado à entrada do encoder e à entrada do decoder,
permitindo que o modelo aprenda representações latentes condicionais por mês.

Interface compatível com BaseModel:
    loss(x, beta=1.0, cond={"month": LongTensor})
    sample(n, cond={"month": LongTensor})
    set_cond_distribution({"month": np.ndarray(N,)})

Arquitetura:
    Encoder: (input_size + E) → latent*4 → latent*2 → (mu, logvar)
    Decoder: (latent_size + E) → latent*2 → latent*4 → input
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel
from models.conditioning import ConditioningBlock, DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS


class VAEModelMC(BaseModel):
    """
    Variational Autoencoder com condicionamento mensal.

    O KL annealing (beta 0→1) é controlado externamente por train.py.
    O ConditioningBlock é genérico: adicionar ENSO ou outros condicionadores
    requer apenas atualizar DEFAULT_CATEGORICALS em conditioning.py.
    """

    def __init__(
        self,
        input_size: int = 15,
        latent_size: int = 128,
        output_activation: str = "relu",
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size
        self.output_activation = output_activation

        self.cond_block = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        E = self.cond_block.total_dim  # 6 para month-only

        # --- Encoder (input_size + E → latent) ---
        self.encoder = nn.Sequential(
            nn.Linear(input_size + E, latent_size * 4),
            nn.LeakyReLU(),
            nn.Linear(latent_size * 4, latent_size * 2),
            nn.LeakyReLU(),
            nn.Linear(latent_size * 2, latent_size),
        )
        self.fc_mu = nn.Linear(latent_size, latent_size)
        self.fc_logvar = nn.Linear(latent_size, latent_size)

        # --- Decoder (latent_size + E → input_size) ---
        decoder_layers = [
            nn.Linear(latent_size + E, latent_size * 2),
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

        # Distribuição empírica dos condicionadores
        self._cond_probs: dict[str, np.ndarray] = {}

    def set_cond_distribution(self, cond_arrays: dict[str, np.ndarray]):
        """
        Armazena probabilidades empíricas para uso em sample() sem cond explícito.

        Args:
            cond_arrays: dict[str, ndarray(N,)] — arrays de condicionamento do treino
        """
        self._cond_probs = {}
        for name, n_classes, _ in self.cond_block.categoricals:
            arr = cond_arrays[name].astype(int)
            counts = np.bincount(arr, minlength=n_classes).astype(float)
            self._cond_probs[name] = counts / counts.sum()
        self._continuous_data = {
            name: cond_arrays[name].astype(np.float32)
            for name, _ in self.cond_block.continuous
            if name in cond_arrays
        }

    def encode(self, x: Tensor, c_emb: Tensor):
        """(x, c_emb) → (mu, logvar)"""
        h = self.encoder(torch.cat([x, c_emb], dim=1))
        return self.fc_mu(h), torch.clamp(self.fc_logvar(h), -10, 10)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """Truque de reparametrização: z = mu + eps * std"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def loss(self, x: Tensor, beta: float = 1.0, cond: dict = None) -> dict:
        """
        Loss VAE: MSE (reconstrução) + beta * KL

        Args:
            x: batch (B, input_size)
            beta: peso do KL — aumentado gradualmente pelo train.py
            cond: dict[str, LongTensor(B,)] ou None (amostra da distribuição empírica)
        """
        if cond is None:
            cond = self.cond_block.sample_cond(x.shape[0], self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond = {k: v.to(x.device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)  # (B, E)

        mu, logvar = self.encode(x, c_emb)
        z = self.reparameterize(mu, logvar)
        x_hat = self.decoder(torch.cat([z, c_emb], dim=1))

        recons = self.criterion(x_hat, x)
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recons + beta * kl
        return {'total': total, 'recons': recons, 'kl': kl}

    def sample(self, n: int, cond: dict = None, steps=None, method=None) -> Tensor:
        """
        Gera n amostras: z ~ N(0, I) → decoder(z, c_emb)

        Args:
            cond: dict[str, LongTensor(n,)] ou None (usa distribuição empírica)

        Returns:
            Tensor (n, input_size) no espaço normalizado
        """
        device = next(self.parameters()).device

        if cond is None:
            cond = self.cond_block.sample_cond(n, self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond = {k: v.to(device) for k, v in cond.items()}

        z = torch.randn(n, self.latent_size, device=device)
        with torch.no_grad():
            c_emb = self.cond_block(cond)
            return self.decoder(torch.cat([z, c_emb], dim=1))
