"""
flow_match.py — Flow Matching Simplificado (MLP puro + embedding sinusoidal)

Versão didática do LatentFlowMatching/test.py — sem Transformer, sem EMA,
sem contexto temporal. Apenas MLP + SinusoidalEmbedding.

Pergunta respondida:
    "Flow Matching (trajetórias retas) supera o fluxo normalizante?"

Matemática:
    Trajetória reta (Optimal Transport path):
        z_t = (1 - t) * z_0 + t * z_1
        target = z_1 - z_0  (velocidade constante!)

    Loss: MSE(v_θ(z_t, t), z_1 - z_0)

    Sampling (integração Euler t=0→1):
        z_0 ~ N(0, I)
        para t = 0, dt, 2*dt, ..., 1-dt:
            z_{t+dt} = z_t + v_θ(z_t, t) * dt

Arquitetura:
    SinusoidalEmbedding: t ∈ [0,1] → R^(2*dim)
    VelocityMLP: [x_t ‖ t_embed] → Linear + ReLU × n_layers → velocidade
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel


class SinusoidalEmbedding(nn.Module):
    """
    Embedding sinusoidal para tempo contínuo t ∈ [0, 1].

    Usa frequências logaritmicamente espaçadas (como transformers de posição).
    dim frequências: dim//2 senos + dim//2 cossenos = dim total.
    """

    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        # Projeção MLP após embedding bruto
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        """
        t: (B,) escalar em [0, 1]
        returns: (B, dim)
        """
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0)
            * torch.arange(half, device=t.device, dtype=torch.float32)
            / half
        )
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (B, 1)
        args = t * freqs  # (B, half)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, dim)
        return self.mlp(emb)

class _FiLMLayer(nn.Module):
    """
    Camada intermediária com FiLM e Conexão Residual.
    """
    def __init__(self, hidden_dim: int, t_embed_dim: int):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, hidden_dim)
        self.film_proj = nn.Linear(t_embed_dim, hidden_dim * 2)
        
        # TRUQUE DE ESTABILIDADE: Inicializa a projeção de tempo com zeros.
        # Assim, scale e shift começam em 0, e h_film começa idêntico a h.
        nn.init.zeros_(self.film_proj.weight)
        nn.init.zeros_(self.film_proj.bias)

    def forward(self, x: Tensor, t_embed: Tensor) -> Tensor:
        # 1. Linear + Ativação
        h = F.silu(self.linear(x))  # SiLU (Swish) costuma ser melhor que ReLU em fluxos
        
        # 2. Gera parâmetros FiLM
        film_params = self.film_proj(t_embed)
        scale, shift = film_params.chunk(2, dim=-1)
        
        # 3. Aplica FiLM
        h_film = h * (1.0 + scale) + shift
        
        # 4. CONEXÃO RESIDUAL (Crucial para o loss continuar caindo)
        return x + h_film

class _VelocityMLP(nn.Module):
    """Rede de velocidade aprimorada com ResNet blocks e FiLM."""
    def __init__(self, data_dim: int, t_embed_dim: int = 64, hidden: int = 256, n_layers: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(data_dim, hidden)
        
        self.hidden_layers = nn.ModuleList([
            _FiLMLayer(hidden, t_embed_dim) for _ in range(n_layers)
        ])
        
        self.output_proj = nn.Linear(hidden, data_dim)
        # Inicializa a última camada perto de zero para velocidades iniciais calmas
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, x_t: Tensor, t_embed: Tensor) -> Tensor:
        h = self.input_proj(x_t)
        
        for layer in self.hidden_layers:
            h = layer(h, t_embed)
            
        return self.output_proj(h)


class FlowMatchingModelFilm(BaseModel):
    """
    Flow Matching com trajetórias retas (Optimal Transport Conditional Flow Matching).

    Não precisa de espaço latente — opera diretamente no espaço de dados.
    Mais simples de treinar que fluxos normalizantes (sem Jacobiano).
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

        self.t_embed = SinusoidalEmbedding(t_embed_dim)
        self.velocity = _VelocityMLP(input_size, t_embed_dim, hidden_size, n_layers)

    def loss(self, x: Tensor, beta: float = 1.0) -> dict:
        """
        Loss Flow Matching: MSE entre velocidade prevista e alvo (z_1 - z_0).

        beta é ignorado (não há KL aqui — não é um VAE).

        Processo:
            z_0 ~ N(0, I)
            t ~ Uniform(0, 1)
            z_t = (1 - t) * z_0 + t * z_1
            target = z_1 - z_0  (velocidade constante para trajetória reta)
            loss = ||v_θ(z_t, t) - target||²
        """
        B = x.shape[0]

        z_0 = torch.randn_like(x)
        z_1 = x

        t = torch.rand(B, device=x.device)
        t_exp = t.unsqueeze(-1)  # (B, 1) para broadcasting

        # Trajetória reta
        z_t = (1 - t_exp) * z_0 + t_exp * z_1
        target = z_1 - z_0  # velocidade constante

        # Predição
        t_emb = self.t_embed(t)
        v_pred = self.velocity(z_t, t_emb)

        fm_loss = F.mse_loss(v_pred, target)
        return {'total': fm_loss, 'fm_loss': fm_loss}

    @torch.no_grad()
    def sample(self, n: int, steps: int | None = None, method: str | None = None) -> Tensor:
        """
        Integração ODE de t=0 até t=1.

        Args:
            steps:  número de passos (None = usa n_sample_steps do modelo)
            method: 'euler' (padrão) ou 'heun' (2ª ordem, predictor-corrector)

        z_0 ~ N(0, I)
        Euler:  z_{t+dt} = z_t + v_θ(z_t, t) * dt
        Heun:   predictor-corrector de 2ª ordem
        """
        num_steps = steps if steps is not None else self.n_sample_steps
        solver = method or 'euler'

        device = next(self.parameters()).device
        z = torch.randn(n, self.input_size, device=device)
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t_val = i * dt
            t_tensor = torch.full((n,), t_val, device=device)
            t_emb = self.t_embed(t_tensor)
            v = self.velocity(z, t_emb)

            if solver == 'heun' and i < num_steps - 1:
                # Predictor (Euler) → velocidade no ponto futuro
                z_tmp = z + v * dt
                t_next = torch.full((n,), (i + 1) * dt, device=device)
                t_next_emb = self.t_embed(t_next)
                v_next = self.velocity(z_tmp, t_next_emb)
                # Corrector: média das velocidades (Heun / Runge-Kutta 2ª ordem)
                v = (v + v_next) / 2.0

            z = z + v * dt

        return z

    @torch.no_grad()
    def sample_heun(self, n: int) -> Tensor:
        """Alias de compatibilidade: chama sample(n, method='heun')."""
        return self.sample(n, method='heun')

    # @torch.no_grad()
    # def sample(self, n: int) -> Tensor:
    #     """
    #     Integração com o Método de Heun (2ª ordem) de t=0 até t=1.
    #     Metade dos passos necessários em comparação ao Euler para a mesma precisão.
    #     """
    #     device = next(self.parameters()).device
    #     z = torch.randn(n, self.input_size, device=device)
    #     dt = 1.0 / self.n_sample_steps

    #     for i in range(self.n_sample_steps):
    #         t_val = i * dt
    #         t_next_val = (i + 1) * dt
            
    #         t_tensor = torch.full((n,), t_val, device=device)
    #         t_emb = self.t_embed(t_tensor)
            
    #         # 1. Calcula a velocidade atual v(t)
    #         v_current = self.velocity(z, t_emb)
            
    #         # 2. Faz um "Euler step" temporário para prever onde estaremos no futuro
    #         z_tmp = z + v_current * dt
            
    #         # Se for o último passo, não precisamos fazer a correção de Heun
    #         if i == self.n_sample_steps - 1:
    #             z = z_tmp
    #             break
                
    #         # 3. Calcula a velocidade no futuro v(t+dt)
    #         t_next_tensor = torch.full((n,), t_next_val, device=device)
    #         t_next_emb = self.t_embed(t_next_tensor)
    #         v_next = self.velocity(z_tmp, t_next_emb)
            
    #         # 4. Tira a média das duas velocidades (Correção de Heun)
    #         v_avg = (v_current + v_next) / 2.0
            
    #         # 5. Dá o passo definitivo usando a velocidade corrigida
    #         z = z + v_avg * dt

    #     return z
