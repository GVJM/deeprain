"""
real_nvp.py — Normalizing Flow com camadas de acoplamento afim (RealNVP)

Única arquitetura com likelihood EXATO — sem ELBO, sem aproximação.

Pergunta respondida:
    "Um fluxo normalizante (likelihood exato) supera o VAE?"

Arquitetura:
    n_coupling=8 camadas de acoplamento afim alternadas (máscara checkerboard):
        x2 = x2 * exp(tanh(s)) + t    onde (s,t) = MLP(x1)
        (tanh(s) evita overflow no exp)

    Loss: NLL exato = -log_prob(x)
        log_prob(x) = log p(z) + sum(log|det J_i|)
                    = -0.5||z||² - n/2 log(2π) + sum(log_scales)

    Sampling: z ~ N(0,I) → inverso das camadas em ordem reversa
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import numpy as np

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel


class _CouplingLayer(nn.Module):
    """
    Camada de acoplamento afim (RealNVP).

    Divide x em duas partes via máscara binária:
        x1 = x[mask],  x2 = x[~mask]
        (s, t) = MLP(x1)
        y2 = x2 * exp(tanh(s)) + t
        y1 = x1 (não modificado)
    """

    def __init__(self, input_size: int, mask: Tensor, hidden: int = 128):
        super().__init__()
        self.register_buffer('mask', mask.float())
        n_active = int(mask.sum().item())

        # MLP que processa a metade não-transformada
        self.net = nn.Sequential(
            nn.Linear(n_active, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, (input_size - n_active) * 2),  # s e t
        )
        self.n_masked = n_active
        self.n_free = input_size - n_active

    def forward(self, x: Tensor):
        """
        Passagem direta: x → y (com log-det jacobiano)
        Returns: y, log_det_J
        """
        x1 = x[:, self.mask.bool()]   # parte fixa
        x2 = x[:, ~self.mask.bool()]  # parte transformada

        st = self.net(x1)
        s, t = st[:, :self.n_free], st[:, self.n_free:]
        s = torch.tanh(s)  # bounded: evita exp overflow

        y2 = x2 * torch.exp(s) + t
        log_det = s.sum(dim=-1)  # log|det J| = sum(s) por batch

        y = torch.empty_like(x)
        y[:, self.mask.bool()] = x1
        y[:, ~self.mask.bool()] = y2
        return y, log_det

    def inverse(self, y: Tensor):
        """Passagem inversa: y → x"""
        x1 = y[:, self.mask.bool()]
        y2 = y[:, ~self.mask.bool()]

        st = self.net(x1)
        s, t = st[:, :self.n_free], st[:, self.n_free:]
        s = torch.tanh(s)

        x2 = (y2 - t) * torch.exp(-s)

        x = torch.empty_like(y)
        x[:, self.mask.bool()] = x1
        x[:, ~self.mask.bool()] = x2
        return x


class RealNVP(BaseModel):
    """
    RealNVP: n_coupling camadas de acoplamento afim alternadas.

    A likelihood é EXATA (não é um lower bound como no VAE).
    """

    def __init__(
        self,
        input_size: int = 15,
        n_coupling: int = 12,
        hidden_size: int = 256,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_coupling = n_coupling

        self.layers = nn.ModuleList()
        for i in range(n_coupling):
            # Máscara alternada: pares vs ímpares
            mask = torch.zeros(input_size, dtype=torch.bool)
            if i % 2 == 0:
                mask[::2] = True   # posições pares fixas
            else:
                mask[1::2] = True  # posições ímpares fixas
            # Garante que a máscara não está toda True ou toda False
            if mask.all() or (~mask).all():
                mask = torch.zeros(input_size, dtype=torch.bool)
                mask[:input_size // 2] = True
            self.layers.append(_CouplingLayer(input_size, mask, hidden_size))

        # Normalização de ativação (batch norm nos reais fluxos)
        self.log_scale = nn.Parameter(torch.zeros(input_size))

    def log_prob(self, x: Tensor) -> Tensor:
        """
        Calcula log-probabilidade exata de x.

        log p(x) = log p(z) + sum_i log|det J_i| + sum(log_scale)
        """
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)

        # Escala aprendida (equivale a batch norm simplificado)
        z = z * torch.exp(self.log_scale)
        log_det_total += self.log_scale.sum()

        for layer in self.layers:
            z, log_det = layer(z)
            log_det_total += log_det

        # log p(z) sob N(0,I)
        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det_total

    def loss(self, x: Tensor, beta: float = 1.0) -> dict:
        """
        Loss: NLL exata = -mean(log_prob(x))
        beta é ignorado (não há KL aqui).
        """
        log_p = self.log_prob(x)
        nll = -log_p.mean()
        return {'total': nll, 'nll': nll}

    def sample(self, n: int, steps: int | None = None, method: str | None = None) -> Tensor:
        """
        Gera n amostras: z ~ N(0,I) → inverso das camadas em ordem reversa.
        """
        device = next(self.parameters()).device
        z = torch.randn(n, self.input_size, device=device)

        with torch.no_grad():
            # Inverte as camadas em ordem reversa
            for layer in reversed(self.layers):
                z = layer.inverse(z)

            # Inverte a escala aprendida
            z = z * torch.exp(-self.log_scale)

        return z
