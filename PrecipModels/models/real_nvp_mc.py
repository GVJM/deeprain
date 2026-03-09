"""
real_nvp_mc.py — RealNVP com Condicionamento Mensal (nn.Embedding)

Variante condicionada de real_nvp.py usando ConditioningBlock.
O embedding mensal é concatenado à entrada de cada camada de acoplamento,
permitindo que o fluxo aprenda distribuições condicionais por mês.

Interface compatível com BaseModel:
    loss(x, beta=1.0, cond={"month": LongTensor})
    sample(n, cond={"month": LongTensor})
    set_cond_distribution({"month": np.ndarray(N,)})

Arquitetura:
    n_coupling camadas de acoplamento afim alternadas (máscara checkerboard)
    Cada camada: MLP(cat([x1, c_emb])) → (s, t) para transformar x2
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


class _CouplingLayerCond(nn.Module):
    """
    Camada de acoplamento afim condicionada (RealNVP + embedding mensal).

    A parte fixa x1 é concatenada com c_emb antes do MLP:
        (s, t) = MLP(cat([x1, c_emb]))
        y2 = x2 * exp(tanh(s)) + t
    """

    def __init__(self, input_size: int, embed_dim: int, mask: Tensor, hidden: int = 256):
        super().__init__()
        self.register_buffer('mask', mask.float())
        n_active = int(mask.sum().item())
        n_free = input_size - n_active

        self.net = nn.Sequential(
            nn.Linear(n_active + embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_free * 2),  # s e t concatenados
        )
        self.n_free = n_free

    def forward(self, x: Tensor, c_emb: Tensor):
        """x: (B, S), c_emb: (B, E) → (y, log_det)"""
        x1 = x[:, self.mask.bool()]
        x2 = x[:, ~self.mask.bool()]

        st = self.net(torch.cat([x1, c_emb], dim=-1))
        s, t = st[:, :self.n_free], st[:, self.n_free:]
        s = torch.tanh(s)

        y2 = x2 * torch.exp(s) + t
        log_det = s.sum(dim=-1)

        y = torch.empty_like(x)
        y[:, self.mask.bool()] = x1
        y[:, ~self.mask.bool()] = y2
        return y, log_det

    def inverse(self, y: Tensor, c_emb: Tensor):
        """y: (B, S), c_emb: (B, E) → x"""
        x1 = y[:, self.mask.bool()]
        y2 = y[:, ~self.mask.bool()]

        st = self.net(torch.cat([x1, c_emb], dim=-1))
        s, t = st[:, :self.n_free], st[:, self.n_free:]
        s = torch.tanh(s)

        x2 = (y2 - t) * torch.exp(-s)

        x = torch.empty_like(y)
        x[:, self.mask.bool()] = x1
        x[:, ~self.mask.bool()] = x2
        return x


class RealNVPMC(BaseModel):
    """
    RealNVP com condicionamento mensal via nn.Embedding.

    O ConditioningBlock é genérico: adicionar ENSO ou outros condicionadores
    requer apenas atualizar DEFAULT_CATEGORICALS em conditioning.py.
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

        self.cond_block = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        E = self.cond_block.total_dim  # 6 para month-only

        self.layers = nn.ModuleList()
        for i in range(n_coupling):
            mask = torch.zeros(input_size, dtype=torch.bool)
            if i % 2 == 0:
                mask[::2] = True
            else:
                mask[1::2] = True
            if mask.all() or (~mask).all():
                mask = torch.zeros(input_size, dtype=torch.bool)
                mask[:input_size // 2] = True
            self.layers.append(_CouplingLayerCond(input_size, E, mask, hidden_size))

        # Escala aprendida (equivale a batch norm simplificado)
        self.log_scale = nn.Parameter(torch.zeros(input_size))

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

    def log_prob(self, x: Tensor, c_emb: Tensor) -> Tensor:
        """
        Log-probabilidade exata condicionada.

        log p(x|c) = log p(z) + sum_i log|det J_i| + sum(log_scale)
        """
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)

        z = z * torch.exp(self.log_scale)
        log_det_total += self.log_scale.sum()

        for layer in self.layers:
            z, log_det = layer(z, c_emb)
            log_det_total += log_det

        log_pz = -0.5 * (z ** 2 + np.log(2 * np.pi)).sum(dim=-1)
        return log_pz + log_det_total

    def loss(self, x: Tensor, beta: float = 1.0, cond: dict = None) -> dict:
        """
        NLL exata condicionada = -mean(log_prob(x|c)).
        beta é ignorado (sem KL).

        Args:
            cond: dict[str, LongTensor(B,)] ou None (amostra da distribuição empírica)
        """
        if cond is None:
            cond = self.cond_block.sample_cond(x.shape[0], self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond = {k: v.to(x.device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)

        log_p = self.log_prob(x, c_emb)
        nll = -log_p.mean()
        return {'total': nll, 'nll': nll}

    def sample(self, n: int, cond: dict = None, steps=None, method=None) -> Tensor:
        """
        Gera n amostras condicionadas: z ~ N(0,I) → inverso das camadas.

        Args:
            cond: dict[str, LongTensor(n,)] ou None (usa distribuição empírica)
        """
        device = next(self.parameters()).device

        if cond is None:
            cond = self.cond_block.sample_cond(n, self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond = {k: v.to(device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)

        z = torch.randn(n, self.input_size, device=device)

        with torch.no_grad():
            for layer in reversed(self.layers):
                z = layer.inverse(z, c_emb)
            z = z * torch.exp(-self.log_scale)

        return z
