"""
glow_mc.py — GLOW com Condicionamento Mensal (nn.Embedding)

Variante condicionada de glow.py usando ConditioningBlock.
O embedding mensal é concatenado à entrada de cada camada de acoplamento afim.
ActNorm e InvertibleLinearLU não recebem condicionamento (são permutações).

Interface compatível com BaseModel:
    loss(x, beta=1.0, cond={"month": LongTensor})
    sample(n, cond={"month": LongTensor})
    set_cond_distribution({"month": np.ndarray(N,)})

Arquitetura:
    n_layers blocos de: ActNorm → InvertibleLinearLU → AffineCouplingCond
    Apenas AffineCouplingCond recebe c_emb: MLP(cat([x1, c_emb])) → (s, t)
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor
import scipy.linalg

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel
from models.conditioning import ConditioningBlock, DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS
from models.glow import ActNorm, InvertibleLinearLU


class _AffineCouplingCond(nn.Module):
    """
    Acoplamento afim GLOW condicionado.

    MLP recebe cat([x1, c_emb]) onde x1 é a primeira metade de x:
        (s, t) = MLP(cat([x[:, :n_half], c_emb]))
        y2 = x2 * exp(tanh(s)) + t
    """

    def __init__(self, input_size: int, embed_dim: int, hidden: int = 128):
        super().__init__()
        self.n_half = input_size // 2
        n_out = input_size - self.n_half

        self.net = nn.Sequential(
            nn.Linear(self.n_half + embed_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, n_out * 2),
        )

    def forward(self, x: Tensor, c_emb: Tensor):
        """x: (B, S), c_emb: (B, E) → (y, log_det)"""
        x1, x2 = x[:, :self.n_half], x[:, self.n_half:]
        st = self.net(torch.cat([x1, c_emb], dim=-1))
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s)
        y2 = x2 * torch.exp(s) + t
        log_det = s.sum(dim=-1)
        return torch.cat([x1, y2], dim=-1), log_det

    def inverse(self, y: Tensor, c_emb: Tensor):
        """y: (B, S), c_emb: (B, E) → x"""
        y1, y2 = y[:, :self.n_half], y[:, self.n_half:]
        st = self.net(torch.cat([y1, c_emb], dim=-1))
        s, t = st.chunk(2, dim=-1)
        s = torch.tanh(s)
        x2 = (y2 - t) * torch.exp(-s)
        return torch.cat([y1, x2], dim=-1)


class GlowMC(BaseModel):
    """
    GLOW com condicionamento mensal via nn.Embedding.

    O ConditioningBlock é genérico: adicionar ENSO ou outros condicionadores
    requer apenas atualizar DEFAULT_CATEGORICALS em conditioning.py.
    """

    def __init__(
        self,
        input_size: int = 15,
        n_layers: int = 8,
        hidden_size: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size

        self.cond_block = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        E = self.cond_block.total_dim  # 6 para month-only

        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            self.layers.append(ActNorm(input_size))
            self.layers.append(InvertibleLinearLU(input_size))
            self.layers.append(_AffineCouplingCond(input_size, E, hidden_size))

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

        Camadas de acoplamento recebem c_emb; ActNorm e InvertibleLinearLU não.
        """
        z = x
        log_det_total = torch.zeros(x.shape[0], device=x.device)

        for layer in self.layers:
            if isinstance(layer, _AffineCouplingCond):
                z, log_det = layer(z, c_emb)
            else:
                z, log_det = layer(z)
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
                if isinstance(layer, _AffineCouplingCond):
                    z = layer.inverse(z, c_emb)
                else:
                    z = layer.inverse(z)

        return z
