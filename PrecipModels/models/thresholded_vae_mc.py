"""
thresholded_vae_mc.py — VAE with Threshold Transform for Zero-Inflated Precipitation

Uses the thresholding trick:
  Training:  y_target = where(x > 0, log1p(x), -1.0)
  Decoding:  expm1(relu(y_pred))  — ReLU maps sentinel <= 0 → 0 (dry), positive → mm/day

No separate occurrence model needed. Single-stage training.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel
from models.conditioning import ConditioningBlock, DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS


class ThresholdedVAEMc(BaseModel):
    """
    VAE with threshold transform for zero-inflated precipitation.

    Encoder/decoder operate in log-space with sentinel -1.0 for dry days.
    No output activation on decoder (raw log-space output).
    Single training stage.
    """

    def __init__(
        self,
        input_size: int = 15,
        latent_size: int = 128,
        n_layers: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.latent_size = latent_size

        self.cond_block = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        E = self.cond_block.total_dim

        h_dim = max(latent_size * 2, input_size * 2)

        # Encoder: (input_size + E) → h_dim → (mu, logvar)
        enc_layers = [nn.Linear(input_size + E, h_dim), nn.LeakyReLU()]
        for _ in range(n_layers - 1):
            enc_layers += [nn.Linear(h_dim, h_dim), nn.LeakyReLU()]
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(h_dim, latent_size)
        self.fc_logvar = nn.Linear(h_dim, latent_size)
        self.fc_logvar.bias.data.fill_(-2.0)

        # Decoder: (latent_size + E) → h_dim → input_size (no activation)
        dec_layers = [nn.Linear(latent_size + E, h_dim), nn.LeakyReLU()]
        for _ in range(n_layers - 1):
            dec_layers += [nn.Linear(h_dim, h_dim), nn.LeakyReLU()]
        dec_layers.append(nn.Linear(h_dim, input_size))
        self.decoder = nn.Sequential(*dec_layers)

        self._cond_probs: dict[str, np.ndarray] = {}

    def set_cond_distribution(self, cond_arrays: dict[str, np.ndarray]):
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

    def encode(self, y: Tensor, c_emb: Tensor):
        h = self.encoder(torch.cat([y, c_emb], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 2)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + (0.5 * logvar).exp() * torch.randn_like(mu)

    def loss(self, x: Tensor, beta: float = 1.0, cond: dict = None) -> dict:
        if cond is None:
            cond = self.cond_block.sample_cond(x.shape[0], self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond = {k: v.to(x.device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)

        # Threshold transform: wet → log1p(x), dry → -1.0
        y_target = torch.where(x > 0, torch.log1p(x), torch.tensor(-1.0, device=x.device))

        mu, logvar = self.encode(y_target, c_emb)
        z = self.reparameterize(mu, logvar)
        y_pred = self.decoder(torch.cat([z, c_emb], dim=-1))

        recon = F.mse_loss(y_pred, y_target, reduction='mean')
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon + beta * kl
        return {'total': total, 'recon': recon, 'kl': kl}

    def sample(self, n: int, cond: dict = None, steps=None, method=None) -> Tensor:
        device = next(self.parameters()).device

        if cond is None:
            cond = self.cond_block.sample_cond(n, self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond = {k: v.to(device) for k, v in cond.items()}

        z = torch.randn(n, self.latent_size, device=device)
        with torch.no_grad():
            c_emb = self.cond_block(cond)
            y_pred = self.decoder(torch.cat([z, c_emb], dim=-1))

        # ReLU maps sentinel <= 0 → 0 (dry); expm1 reverts log1p
        return torch.expm1(F.relu(y_pred))
