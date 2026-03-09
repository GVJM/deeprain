"""
flow_match_mc.py — Flow Matching com Condicionamento Mensal (FiLM + Residual)

Variante condicionada de flow_match.py usando ConditioningBlock.
O embedding mensal e o tempo são concatenados e usados para gerar
parâmetros de escala e translação (FiLM) em cada camada residual,
condicionando a trajetória ao mês do ano de forma muito mais eficiente.
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
    """Embedding sinusoidal para tempo contínuo t ∈ [0, 1]."""
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


class _FiLMCondLayer(nn.Module):
    """
    Camada intermediária com FiLM e Conexão Residual.
    Recebe as features e um vetor de condicionamento (Tempo + Mês).
    """
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        
        # Projeta o vetor de contexto em Escala e Translação
        self.film_proj = nn.Linear(cond_dim, hidden_dim * 2)
        
        # TRUQUE DE ESTABILIDADE: Inicializa a projeção de contexto com zeros.
        # Assim, scale e shift começam em 0, e h_film começa idêntico a h.
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(self, x: Tensor, cond_emb: Tensor) -> Tensor:
        # 1. Transformação não-linear base
        h = F.silu(self.linear(x))
        
        # 2. Gera parâmetros FiLM a partir do contexto (Tempo + Mês)
        film_params = self.film_proj(cond_emb)
        scale, shift = film_params.chunk(2, dim=-1)
        
        # 3. Modulação (FiLM)
        h_film = h * (1.0 + scale) + shift
        
        # 4. Conexão Residual (O segredo para redes profundas!)
        return x + h_film


class _VelocityMLPCond(nn.Module):
    """
    Rede de velocidade condicionada: estima v(z_t, t, c).
    Usa um backbone residual onde o tempo e o mês modulam cada bloco.
    """
    def __init__(self, data_dim: int, t_embed_dim: int, c_embed_dim: int, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        
        # Projeção inicial APENAS dos dados
        self.input_proj = nn.Linear(data_dim, hidden)
        
        # O vetor de contexto será a união do Tempo e da Classe (Mês)
        cond_dim = t_embed_dim + c_embed_dim
        
        self.hidden_layers = nn.ModuleList([
            _FiLMCondLayer(hidden, cond_dim) for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(hidden, data_dim)
        
        # TRUQUE DE ESTABILIDADE: Inicializa a última camada em zero.
        # O fluxo começará prevendo v = 0 (não faz nada ao ruído), estabilizando a loss inicial.
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_t: Tensor, t_embed: Tensor, c_emb: Tensor) -> Tensor:
        # Junta tempo e informações mensais num único vetor de contexto
        cond_emb = torch.cat([t_embed, c_emb], dim=-1)
        
        # Projeta os dados para a dimensão oculta
        h = F.silu(self.input_proj(x_t))
        
        # Passa pelos blocos residuais, injetando o contexto em TODOS eles
        for layer in self.hidden_layers:
            h = layer(h, cond_emb)
            
        return self.output_proj(h)


class FlowMatchingFilmMC(BaseModel):
    """
    Flow Matching condicionado por mês via nn.Embedding e FiLM.
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
        E = self.cond_block.total_dim  # Dimensão total dos condicionadores

        self.t_embed = SinusoidalEmbedding(t_embed_dim)
        self.velocity = _VelocityMLPCond(input_size, t_embed_dim, E, hidden_size, n_layers)

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

    def loss(self, x: Tensor, beta: float = 1.0, cond: dict = None) -> dict:
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