"""
latent_flow_mc.py — Latent Flow Matching com condicionamento mensal (sem hurdle).

Arquitetura em dois estágios:
  Estágio 1 (VAE):
    - Encoder: x → (mu, logvar) condicionado em c_emb via concatenação
    - Decoder: z + c_emb → x_hat
    - Loss: MSE_recon + beta * KL

  Estágio 2 (Flow Matching no espaço latente):
    - LatentFlowNet: v_θ(z_t, t, c_emb) com FiLM conditioning
    - Caminho OT linear: z_t = (1-t)*z0 + t*z1, alvo u_t = z1 - z0
    - z1 = mu do encoder (sem ruído) → targets mais limpos
    - VAE congelado durante o treino do fluxo

Uso no train.py:
    # Estágio 1
    model.set_stage("vae")
    # loop: model.loss(x, beta=beta, cond=cond)

    # Estágio 2
    model.set_stage("flow")
    # loop: model.loss(x, beta=1.0, cond=cond)

    # Amostragem
    samples = model.sample(n, cond=cond, steps=100, method='euler')

Registro em train.py:
    MODEL_DEFAULTS["latent_flow_mc"] = {
        "normalization_mode": "standardize",
        "max_epochs": 500,       # estágio VAE
        "latent_size": 64,
        "lr": 0.001,
        "batch_size": 128,
        "kl_warmup": 200,
        "flow_epochs": 500,
        "flow_lr": 0.0002,
    }
    _MC_MODELS.add("latent_flow_mc")
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


# ══════════════════════════════════════════════════════════════════════════════
# AUXILIARES
# ══════════════════════════════════════════════════════════════════════════════

class _SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half  = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


class _FiLM(nn.Module):
    """h' = gamma(c) * h + beta(c). Inicializado como identidade."""
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta  = nn.Linear(cond_dim, hidden_dim)
        nn.init.ones_(self.gamma.weight);  nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, h: Tensor, cond: Tensor) -> Tensor:
        return self.gamma(cond) * h + self.beta(cond)


class _FlowResBlock(nn.Module):
    """ResBlock com FiLM: LayerNorm → Linear → SiLU → Linear → FiLM → residual."""
    def __init__(self, hidden_dim: int, cond_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm    = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.linear2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.film    = _FiLM(hidden_dim, cond_dim)
        self.drop    = nn.Dropout(dropout)

    def forward(self, h: Tensor, cond: Tensor) -> Tensor:
        residual = h
        h = self.norm(h)
        h = F.silu(self.linear1(h))
        h = self.drop(self.linear2(h))
        h = self.film(h, cond)
        return h + residual


# ══════════════════════════════════════════════════════════════════════════════
# VAE CONDICIONAL
# ══════════════════════════════════════════════════════════════════════════════

class _CondVAE(nn.Module):
    """
    VAE com condicionamento por concatenação.
    Encoder: [x | c] → (mu, logvar)
    Decoder: [z | c] → x_hat
    """
    def __init__(
        self,
        input_size: int,
        cond_size: int,
        latent_size: int,
        n_layers: int = 3,
    ):
        super().__init__()
        self.latent_size = latent_size
        h = max(latent_size * 4, 128)

        # Encoder
        enc = [nn.Linear(input_size + cond_size, h), nn.SiLU()]
        for _ in range(n_layers - 1):
            enc += [nn.Linear(h, h), nn.SiLU()]
        self.encoder    = nn.Sequential(*enc)
        self.fc_mu      = nn.Linear(h, latent_size)
        self.fc_logvar  = nn.Linear(h, latent_size)
        self.fc_logvar.bias.data.fill_(-2.0)   # evita posterior collapse inicial

        # Decoder
        dec = [nn.Linear(latent_size + cond_size, h), nn.SiLU()]
        for _ in range(n_layers - 1):
            dec += [nn.Linear(h, h), nn.SiLU()]
        dec.append(nn.Linear(h, input_size))
        self.decoder = nn.Sequential(*dec)

    def encode(self, x: Tensor, c: Tensor):
        h = self.encoder(torch.cat([x, c], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 2)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + (0.5 * logvar).exp() * torch.randn_like(mu)

    def decode(self, z: Tensor, c: Tensor) -> Tensor:
        return self.decoder(torch.cat([z, c], dim=-1))

    def forward(self, x: Tensor, c: Tensor):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar, z


# ══════════════════════════════════════════════════════════════════════════════
# REDE DE VELOCIDADE LATENTE
# ══════════════════════════════════════════════════════════════════════════════

class _LatentFlowNet(nn.Module):
    """
    v_θ(z_t, t, c) → velocidade no espaço latente.
    FiLM em cada ResBlock garante que o condicionamento mensal
    modula todas as camadas, não apenas a entrada.
    """
    def __init__(
        self,
        latent_dim: int,
        cond_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 6,
        t_embed_dim: int = 64,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.time_emb    = _SinusoidalTimeEmb(t_embed_dim)
        self.input_proj  = nn.Linear(latent_dim + t_embed_dim, hidden_dim)
        self.blocks      = nn.ModuleList([
            _FlowResBlock(hidden_dim, cond_dim, dropout) for _ in range(n_layers)
        ])
        self.norm        = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        # Zero-init: velocidade inicial = 0, treino mais estável
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        t_emb = self.time_emb(t)
        h = self.input_proj(torch.cat([z_t, t_emb], dim=-1))
        for block in self.blocks:
            h = block(h, cond)
        return self.output_proj(self.norm(h))


# ══════════════════════════════════════════════════════════════════════════════
# MODELO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class LatentFlowMc(BaseModel):
    """
    Latent Flow Matching com condicionamento mensal. Sem hurdle — modela
    a distribuição conjunta de precipitação direta no espaço latente.

    Parâmetros:
        input_size:     número de estações (S)
        latent_size:    dimensão do espaço latente (= dimensão do fluxo)
        hidden_size:    largura das camadas ocultas do FlowNet
        n_layers:       profundidade do FlowNet (ResBlocks com FiLM)
        t_embed_dim:    dimensão do embedding sinusoidal de tempo
        n_sample_steps: passos de integração Euler/Heun na amostragem
        vae_layers:     profundidade do VAE (padrão: 3)
    """

    def __init__(
        self,
        input_size: int = 15,
        latent_size: int = 64,
        hidden_size: int = 256,
        n_layers: int = 6,
        t_embed_dim: int = 64,
        n_sample_steps: int = 100,
        vae_layers: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.input_size     = input_size
        self.latent_size    = latent_size
        self.n_sample_steps = n_sample_steps
        self._stage         = "vae"

        # Condicionamento mensal
        self.cond_block = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        cond_dim = self.cond_block.total_dim   # 6 para month-only

        # VAE condicional
        self.vae = _CondVAE(
            input_size=input_size,
            cond_size=cond_dim,
            latent_size=latent_size,
            n_layers=vae_layers,
        )

        # Rede de velocidade latente
        self.flow_net = _LatentFlowNet(
            latent_dim=latent_size,
            cond_dim=cond_dim,
            hidden_dim=hidden_size,
            n_layers=n_layers,
            t_embed_dim=t_embed_dim,
        )

        # Distribuição empírica dos condicionadores
        self._cond_probs: dict[str, np.ndarray] = {}

        print(
            f"[LatentFlowMc] S={input_size} | latent={latent_size} | "
            f"flow: hidden={hidden_size} layers={n_layers} | "
            f"params={self.count_parameters():,}"
        )

    # ── Estágio ────────────────────────────────────────────────────────────────

    def set_stage(self, stage: str):
        """
        'vae'  → treina só o VAE (flow congelado)
        'flow' → treina só o flow (VAE congelado)
        """
        assert stage in ("vae", "flow")
        self._stage = stage
        for p in self.vae.parameters():
            p.requires_grad_(stage == "vae")
        for p in self.flow_net.parameters():
            p.requires_grad_(stage == "flow")
        print(f"[LatentFlowMc] Estágio: {stage.upper()} | "
              f"parâmetros ativos: {self.count_parameters():,}")

    # ── Distribuição empírica ──────────────────────────────────────────────────

    def set_cond_distribution(self, cond_arrays: dict[str, np.ndarray]):
        self._cond_probs = {}
        for name, n_classes, _ in self.cond_block.categoricals:
            arr    = cond_arrays[name].astype(int)
            counts = np.bincount(arr, minlength=n_classes).astype(float)
            self._cond_probs[name] = counts / counts.sum()
        self._continuous_data = {
            name: cond_arrays[name].astype(np.float32)
            for name, _ in self.cond_block.continuous
            if name in cond_arrays
        }

    # ── Loss ───────────────────────────────────────────────────────────────────

    def loss(self, x: Tensor, beta: float = 1.0, cond: dict = None) -> dict:
        if cond is None:
            cond = self.cond_block.sample_cond(x.shape[0], self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond  = {k: v.to(x.device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)   # (B, cond_dim)

        if self._stage == "vae":
            return self._vae_loss(x, c_emb, beta)
        else:
            return self._flow_loss(x, c_emb)

    def _vae_loss(self, x: Tensor, c_emb: Tensor, beta: float) -> dict:
        x_hat, mu, logvar, _ = self.vae(x, c_emb)
        recon = F.mse_loss(x_hat, x)
        kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        total = recon + beta * kl
        return {'total': total, 'recon': recon, 'kl': kl}

    def _flow_loss(self, x: Tensor, c_emb: Tensor) -> dict:
        """
        Flow Matching loss no espaço latente.
        z1 = mu do encoder (sem ruído) → targets limpos para o fluxo.
        Caminho OT: z_t = (1-t)*z0 + t*z1, alvo u_t = z1 - z0.
        """
        with torch.no_grad():
            mu_z1, logvar_z1 = self.vae.encode(x, c_emb)
            z1 = mu_z1                          # (B, latent_size)
            # z1 = self.vae.reparameterize(mu_z1, logvar_z1)
        z0 = torch.randn_like(z1)
        t  = torch.rand(z1.shape[0], device=z1.device)

        t_  = t[:, None]
        z_t = (1 - t_) * z0 + t_ * z1          # interpolação linear
        u_t = z1 - z0                            # velocidade alvo (constante no OT path)

        v_pred = self.flow_net(z_t, t, c_emb)
        fm     = F.mse_loss(v_pred, u_t)
        return {'total': fm, 'fm': fm}

    # ── Amostragem ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(
        self,
        n: int,
        cond: dict = None,
        steps: int | None = None,
        method: str | None = None,
    ) -> Tensor:
        """
        Integra a ODE no espaço latente e decodifica.

        Args:
            n:      número de amostras
            steps:  passos de integração (padrão: self.n_sample_steps)
            method: 'euler' (padrão) ou 'heun' (mais preciso, 2x NFE)
            cond:   dict de condicionamento; se None, sorteia da distribuição empírica
        """
        device  = next(self.parameters()).device
        n_steps = steps  if steps  is not None else self.n_sample_steps
        solver  = method if method is not None else 'euler'

        if cond is None:
            cond = self.cond_block.sample_cond(n, self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond  = {k: v.to(device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)           # (n, cond_dim)

        z  = torch.randn(n, self.latent_size, device=device)
        dt = 1.0 / n_steps

        if solver == 'euler':
            for i in range(n_steps):
                t_vec = z.new_full((n,), i / n_steps)
                z = z + dt * self.flow_net(z, t_vec, c_emb)

        elif solver == 'heun':
            for i in range(n_steps):
                t_vec  = z.new_full((n,), i / n_steps)
                t_vec2 = z.new_full((n,), (i + 1) / n_steps)
                v1      = self.flow_net(z, t_vec, c_emb)
                z_tilde = z + dt * v1
                v2      = self.flow_net(z_tilde, t_vec2, c_emb)
                z = z + dt * 0.5 * (v1 + v2)

        else:
            raise ValueError(f"method deve ser 'euler' ou 'heun', recebeu '{solver}'")

        return self.vae.decode(z, c_emb)