"""
flow_match_mc.py — Flow Matching com Condicionamento Mensal (nn.Embedding)

Variante condicionada de flow_match.py usando ConditioningBlock.
O embedding mensal é concatenado à entrada da rede de velocidade em cada passo
da integração ODE, condicionando a trajetória ao mês do ano.

Interface compatível com BaseModel:
    loss(x, beta=1.0, cond={"month": LongTensor})
    sample(n, cond={"month": LongTensor})
    set_cond_distribution({"month": np.ndarray(N,)})

Matemática:
    Trajetória reta (Optimal Transport path):
        z_t = (1 - t) * z_0 + t * z_1
        target = z_1 - z_0  (velocidade constante)

    Loss: MSE(v_θ(z_t, t_emb, c_emb), z_1 - z_0)

    Sampling (integração Euler t=0→1):
        z_0 ~ N(0, I)
        z_{t+dt} = z_t + v_θ(z_t, t_emb, c_emb) * dt

Arquitetura:
    SinusoidalEmbedding: t ∈ [0,1] → R^t_embed_dim
    VelocityMLP: cat([x_t, t_emb, c_emb]) → velocidade
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
from models.conditioning import ConditioningBlock, DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS


class SinusoidalEmbedding(nn.Module):
    """
    Embedding sinusoidal para tempo contínuo t ∈ [0, 1].
    Copiado de flow_match.py para evitar importação do módulo com código no nível do módulo.
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
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
            t = t.unsqueeze(-1)
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class _VelocityMLPCond(nn.Module):
    """
    Rede de velocidade condicionada: estima v(z_t, t, c).

    Input: cat([x_t, t_emb, c_emb]) de dimensão D + T + E
    Output: velocidade no espaço de dados (dim D)
    """

    def __init__(self, data_dim: int, t_embed_dim: int, cond_dim: int, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        in_dim = data_dim + t_embed_dim + cond_dim
        layers = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.ReLU()]
        layers.append(nn.Linear(hidden, data_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x_t: Tensor, t_embed: Tensor, c_emb: Tensor) -> Tensor:
        """x_t: (B, D), t_embed: (B, T), c_emb: (B, E) → velocidade: (B, D)"""
        inp = torch.cat([x_t, t_embed, c_emb], dim=-1)
        return self.net(inp)


class FlowMatchingMC(BaseModel):
    """
    Flow Matching condicionado por mês via nn.Embedding.

    O ConditioningBlock é genérico: adicionar ENSO ou outros condicionadores
    requer apenas atualizar DEFAULT_CATEGORICALS em conditioning.py.
    """

    def __init__(
        self,
        input_size: int = 15,
        t_embed_dim: int = 64,
        hidden_size: int = 256,
        n_layers: int = 4,
        n_sample_steps: int = 50,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.n_sample_steps = n_sample_steps

        self.cond_block = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        E = self.cond_block.total_dim  # 6 para month-only

        self.t_embed = SinusoidalEmbedding(t_embed_dim)
        self.velocity = _VelocityMLPCond(input_size, t_embed_dim, E, hidden_size, n_layers)

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

    def loss(self, x: Tensor, beta: float = 1.0, cond: dict = None) -> dict:
        """
        Loss Flow Matching condicionada: MSE(v_θ(z_t, t, c), z_1 - z_0).
        beta é ignorado (sem KL).

        Args:
            cond: dict[str, LongTensor(B,)] ou None (amostra da distribuição empírica)
        """
        if cond is None:
            cond = self.cond_block.sample_cond(x.shape[0], self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond = {k: v.to(x.device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)

        B = x.shape[0]
        z_0 = torch.randn_like(x)
        z_1 = x

        t = torch.rand(B, device=x.device)
        t_exp = t.unsqueeze(-1)

        z_t = (1 - t_exp) * z_0 + t_exp * z_1
        target = z_1 - z_0

        t_emb = self.t_embed(t)
        v_pred = self.velocity(z_t, t_emb, c_emb)

        fm_loss = F.mse_loss(v_pred, target)
        return {'total': fm_loss, 'fm_loss': fm_loss}

    @torch.no_grad()
    def sample(self, n: int, cond: dict = None, steps=None, method=None) -> Tensor:
        """
        Integração ODE condicionada de t=0 até t=1.

        Args:
            cond:   dict[str, LongTensor(n,)] ou None (usa distribuição empírica)
            steps:  número de passos (None = usa n_sample_steps do modelo)
            method: 'euler' (padrão) ou 'heun' (2ª ordem)
        """
        num_steps = steps if steps is not None else self.n_sample_steps
        solver = method or 'euler'

        device = next(self.parameters()).device

        if cond is None:
            cond = self.cond_block.sample_cond(n, self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond = {k: v.to(device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)

        z = torch.randn(n, self.input_size, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i * dt
            t_tensor = torch.full((n,), t_val, device=device)
            t_emb = self.t_embed(t_tensor)
            v = self.velocity(z, t_emb, c_emb)

            if solver == 'heun' and i < num_steps - 1:
                z_tmp = z + v * dt
                t_next = torch.full((n,), (i + 1) * dt, device=device)
                t_next_emb = self.t_embed(t_next)
                v_next = self.velocity(z_tmp, t_next_emb, c_emb)
                v = (v + v_next) / 2.0

            z = z + v * dt

        return z
