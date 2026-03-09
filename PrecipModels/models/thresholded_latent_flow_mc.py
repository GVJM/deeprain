import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys
import os
# Assumes base_model.py and models.conditioning are in the path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel
from models.conditioning import ConditioningBlock, DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS

# ══════════════════════════════════════════════════════════════════════════════
# BLOCOS AUXILIARES (Tempo, FiLM, ResBlock - Mantidos e Otimizados)
# ══════════════════════════════════════════════════════════════════════════════

class _SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None] * freqs[None]
        return torch.cat([args.sin(), args.cos()], dim=-1)

class _FiLM(nn.Module):
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta  = nn.Linear(cond_dim, hidden_dim)
        nn.init.ones_(self.gamma.weight); nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight); nn.init.zeros_(self.beta.bias)

    def forward(self, h: Tensor, cond: Tensor) -> Tensor:
        return self.gamma(cond) * h + self.beta(cond)

class _FlowResBlock(nn.Module):
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
# REDES PRINCIPAIS: FlowNet e Continuous VAE
# ══════════════════════════════════════════════════════════════════════════════

class _LatentFlowNet(nn.Module):
    """Rede de velocidade v_θ(z_t, t, c). Nenhuma máscara espacial necessária."""
    def __init__(self, latent_dim: int, cond_dim: int, hidden_dim: int = 256, n_layers: int = 6, t_embed_dim: int = 64, dropout: float = 0.0):
        super().__init__()
        self.time_emb   = _SinusoidalTimeEmb(t_embed_dim)
        self.input_proj = nn.Linear(latent_dim + t_embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([_FlowResBlock(hidden_dim, cond_dim, dropout) for _ in range(n_layers)])
        self.norm        = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        nn.init.zeros_(self.output_proj.weight); nn.init.zeros_(self.output_proj.bias)

    def forward(self, z_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        t_emb = self.time_emb(t)
        h = self.input_proj(torch.cat([z_t, t_emb], dim=-1))
        for block in self.blocks:
            h = block(h, cond)
        return self.output_proj(self.norm(h))

class _ContinuousVAE(nn.Module):
    """VAE unificado. Codifica campos contínuos de precipitação limiarizada."""
    def __init__(self, input_size: int, cond_size: int, latent_size: int, n_layers: int = 3):
        super().__init__()
        self.latent_size = latent_size
        h_dim = max(latent_size * 2, input_size * 2)

        # Encoder
        enc_layers = [nn.Linear(input_size + cond_size, h_dim), nn.LeakyReLU()]
        for _ in range(n_layers - 1):
            enc_layers += [nn.Linear(h_dim, h_dim), nn.LeakyReLU()]
        self.encoder = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(h_dim, latent_size)
        self.fc_logvar = nn.Linear(h_dim, latent_size)
        self.fc_logvar.bias.data.fill_(-2.0)

        # Decoder
        dec_layers = [nn.Linear(latent_size + cond_size, h_dim), nn.LeakyReLU()]
        for _ in range(n_layers - 1):
            dec_layers += [nn.Linear(h_dim, h_dim), nn.LeakyReLU()]
        dec_layers.append(nn.Linear(h_dim, input_size))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: Tensor, c: Tensor):
        h = self.encoder(torch.cat([x, c], dim=-1))
        return self.fc_mu(h), self.fc_logvar(h).clamp(-10, 2)

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        return mu + (0.5 * logvar).exp() * torch.randn_like(mu)

    def decode(self, z: Tensor, c: Tensor) -> Tensor:
        return self.decoder(torch.cat([z, c], dim=-1))

# ══════════════════════════════════════════════════════════════════════════════
# MODELO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class ThresholdedLatentFlowMc(BaseModel):
    """
    Abordagem Thresholded Continuous Latent Flow.
    Input padrão escalado para 100 estações.
    """
    def __init__(
        self,
        input_size: int = 100,
        latent_size: int = 128,  # Aumentado ligeiramente para 100 estações
        hidden_size: int = 384, # Aumentado para lidar com maior dimensionalidade espacial
        n_layers: int = 6,
        t_embed_dim: int = 64,
        n_sample_steps: int = 100,
        vae_layers: int = 3,
        **kwargs,
    ):
        super().__init__()
        self.input_size     = input_size
        self.latent_dim     = latent_size
        self.n_sample_steps = n_sample_steps
        self._stage         = "vae" 

        self.cond_block = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        cond_dim = self.cond_block.total_dim 

        # VAE Único
        self.vae = _ContinuousVAE(
            input_size=input_size,
            cond_size=cond_dim,
            latent_size=latent_size,
            n_layers=vae_layers,
        )

        # Latent FlowNet
        self.flow_net = _LatentFlowNet(
            latent_dim=latent_size,
            cond_dim=cond_dim,
            hidden_dim=hidden_size,
            n_layers=n_layers,
            t_embed_dim=t_embed_dim,
        )

        self._cond_probs: dict[str, np.ndarray] = {}
        
        print(f"[ThresholdedLatentFlowMc] S={input_size} | latent_dim={latent_size} | params={self.count_parameters():,}")

    def set_stage(self, stage: str):
        assert stage in ("vae", "flow")
        self._stage = stage
        for p in self.vae.parameters():
            p.requires_grad_(stage == "vae")
        for p in self.flow_net.parameters():
            p.requires_grad_(stage == "flow")
        print(f"[ThresholdedLatentFlowMc] Estágio: {stage.upper()}")

    def set_cond_distribution(self, cond_arrays: dict[str, np.ndarray]):
        self._cond_probs = {}
        for name, n_classes, _ in self.cond_block.categoricals:
            arr = cond_arrays[name].astype(int)
            counts = np.bincount(arr, minlength=n_classes).astype(float)
            self._cond_probs[name] = counts / counts.sum()
        self._continuous_data = {
            name: cond_arrays[name].astype(np.float32) for name, _ in self.cond_block.continuous if name in cond_arrays
        }

    def loss(self, x: Tensor, beta: float = 1.0, cond: dict = None) -> dict:
        if cond is None:
            cond = self.cond_block.sample_cond(x.shape[0], self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond = {k: v.to(x.device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)

        # Cria o target contínuo: log1p(x) se chover, -1.0 se seco
        y_target = torch.where(x > 0, torch.log1p(x), torch.tensor(-1.0, device=x.device))

        if self._stage == "vae":
            mu, logvar = self.vae.encode(y_target, c_emb)
            z = self.vae.reparameterize(mu, logvar)
            y_pred = self.vae.decode(z, c_emb)
            
            recon_loss = F.mse_loss(y_pred, y_target, reduction='mean')
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            
            total = recon_loss + beta * kl_loss
            return {'total': total, 'recon': recon_loss, 'kl': kl_loss}
            
        else: # Estágio Flow
            with torch.no_grad():
                z1, _ = self.vae.encode(y_target, c_emb) 
            
            z0 = torch.randn_like(z1)
            t  = torch.rand(z1.shape[0], device=z1.device)
            t_ = t[:, None]
            
            z_t = (1 - t_) * z0 + t_ * z1
            u_t = z1 - z0
            v_pred = self.flow_net(z_t, t, c_emb)

            fm_loss = F.mse_loss(v_pred, u_t)
            return {'total': fm_loss, 'fm': fm_loss}

    @torch.no_grad()
    def sample(self, n: int, cond: dict = None, steps: int | None = None, method: str | None = None) -> Tensor:
        device = next(self.parameters()).device
        if cond is None:
            cond = self.cond_block.sample_cond(n, self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond  = {k: v.to(device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)

        n_steps = steps if steps is not None else self.n_sample_steps
        solver  = method if method is not None else 'euler'

        # 1. Integra ODE no espaço latente
        z  = torch.randn(n, self.latent_dim, device=device)
        dt = 1.0 / n_steps

        if solver == 'euler':
            for i in range(n_steps):
                t_vec = z.new_full((n,), i / n_steps)
                z = z + dt * self.flow_net(z, t_vec, c_emb)
        elif solver == 'heun':
            for i in range(n_steps):
                t_vec  = z.new_full((n,), i / n_steps)
                t_vec2 = z.new_full((n,), (i + 1) / n_steps)
                v1     = self.flow_net(z, t_vec, c_emb)
                z_tilde = z + dt * v1
                v2     = self.flow_net(z_tilde, t_vec2, c_emb)
                z = z + dt * 0.5 * (v1 + v2)
        else:
            raise ValueError(f"Método '{solver}' inválido.")

        # 2. Decodifica e aplica o Limiar (ReLU)
        y_pred = self.vae.decode(z, c_emb)
        
        # ReLU converte tudo <= 0 em exatamente 0 (dias secos)
        # expm1 reverte a transformação log1p original
        x_amt = torch.expm1(F.relu(y_pred))

        return x_amt