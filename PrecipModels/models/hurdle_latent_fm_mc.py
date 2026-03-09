"""
hurdle_latent_flow_mc.py — Latent Flow Matching com hurdle model e condicionamento mensal.

Arquitetura em dois estágios:
  Estágio 1 (VAE, treinado via loss() com beta-annealing):
    - OccVAE:  x_bin  → z_occ → logits de ocorrência  (BCE + KL)
    - AmtVAE:  x_log  → z_amt → quantidades log1p      (MSE_wet + KL)

  Estágio 2 (Flow Matching, treinado via loss() com stage='flow'):
    - LatentFlowNet condicional: mapeia N(0,I) → z_amt usando FiLM com embedding mensal
    - ODE linear (OT path): z_t = (1-t)*z0 + t*z1, alvo u_t = z1 - z0

Uso:
    # Estágio 1 — VAE
    model.set_stage("vae")
    # loop: loss_dict = model.loss(x, beta=beta); loss_dict['total'].backward()

    # Estágio 2 — Flow
    model.set_stage("flow")
    # loop: loss_dict = model.loss(x, beta=1.0, cond=cond); loss_dict['total'].backward()

    # Amostragem
    samples = model.sample(n, cond=cond, steps=100, method='euler')

Integração com train.py:
    - Registrar em MODEL_DEFAULTS como "hurdle_latent_flow_mc" (is_mc=True)
    - Treino VAE: train_neural_model_mc(..., kl_warmup=200)
    - Treino Flow: train_neural_model_mc(..., kl_warmup=0, stage='flow')
    - set_cond_distribution() chamado antes do treino
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
# BLOCO AUXILIAR: Embedding sinusoidal de tempo
# ══════════════════════════════════════════════════════════════════════════════

class _SinusoidalTimeEmb(nn.Module):
    """
    Embedding sinusoidal para o tempo contínuo t ∈ [0, 1].
    Mesmo estilo do LDM/DDPM — frequências log-espaçadas.
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: Tensor) -> Tensor:
        # t: (B,) → (B, dim)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device) / half
        )
        args = t[:, None] * freqs[None]          # (B, half)
        return torch.cat([args.sin(), args.cos()], dim=-1)  # (B, dim)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO AUXILIAR: FiLM (Feature-wise Linear Modulation)
# ══════════════════════════════════════════════════════════════════════════════

class _FiLM(nn.Module):
    """
    Modula h com a condição c: h' = gamma(c) * h + beta(c).
    Inicializado como identidade (gamma=1, beta=0) para training estável.
    """
    def __init__(self, hidden_dim: int, cond_dim: int):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, hidden_dim)
        self.beta  = nn.Linear(cond_dim, hidden_dim)
        nn.init.ones_(self.gamma.weight);  nn.init.zeros_(self.gamma.bias)
        nn.init.zeros_(self.beta.weight);  nn.init.zeros_(self.beta.bias)

    def forward(self, h: Tensor, cond: Tensor) -> Tensor:
        return self.gamma(cond) * h + self.beta(cond)


# ══════════════════════════════════════════════════════════════════════════════
# BLOCO AUXILIAR: ResBlock com FiLM para a rede de fluxo
# ══════════════════════════════════════════════════════════════════════════════

class _FlowResBlock(nn.Module):
    """
    Residual block: LayerNorm → Linear → SiLU → Linear → FiLM → residual.
    A condição (tempo + mês) modula via FiLM em cada camada.
    """
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
# REDE DE FLUXO LATENTE
# ══════════════════════════════════════════════════════════════════════════════

class _LatentFlowNet(nn.Module):
    """
    Rede de velocidade v_θ(z_t, t, c) no espaço latente.

    Entrada:  z_t (B, latent_dim) + t_emb (B, t_embed_dim)
    Condição: cond_emb (B, cond_dim) — embedding mensal via FiLM em cada bloco
    Saída:    v (B, latent_dim) — velocidade estimada

    Output proj inicializado em zero → velocidade inicial = 0 (training estável).
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
        self.time_emb   = _SinusoidalTimeEmb(t_embed_dim)
        self.input_proj = nn.Linear(latent_dim + t_embed_dim, hidden_dim)
        self.blocks = nn.ModuleList([
            _FlowResBlock(hidden_dim, cond_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm        = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        # Zero-init: predição de velocidade começa estável
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)

    def forward(self, z_t: Tensor, t: Tensor, cond: Tensor) -> Tensor:
        # z_t: (B, latent_dim), t: (B,), cond: (B, cond_dim)
        t_emb = self.time_emb(t)                              # (B, t_embed_dim)
        h = self.input_proj(torch.cat([z_t, t_emb], dim=-1)) # (B, hidden_dim)
        for block in self.blocks:
            h = block(h, cond)
        return self.output_proj(self.norm(h))                 # (B, latent_dim)


# ══════════════════════════════════════════════════════════════════════════════
# VAE CONDICIONAL INTERNO (replicado de hurdle_vae_cond_mc, leve refactoring)
# ══════════════════════════════════════════════════════════════════════════════

class _CondVAE(nn.Module):
    """
    CVAE simples: encoder e decoder com condicionamento por concatenação.
    Idêntico ao _ConditionalMiniVAE do hurdle_vae_cond_mc mas com:
      - LayerNorm + residual para profundidades maiores
      - logvar clampado na fonte
    """
    def __init__(
        self,
        input_size: int,
        cond_size: int,
        latent_size: int,
        output_activation: str = "none",
        n_layers: int = 2,
    ):
        super().__init__()
        self.latent_size = latent_size
        enc_in = input_size + cond_size
        h = max(latent_size * 2, enc_in)

        enc_layers = [nn.Linear(enc_in, h), nn.LeakyReLU()]
        for _ in range(n_layers - 1):
            enc_layers += [nn.Linear(h, h), nn.LeakyReLU()]
        self.encoder   = nn.Sequential(*enc_layers)
        self.fc_mu     = nn.Linear(h, latent_size)
        self.fc_logvar = nn.Linear(h, latent_size)
        self.fc_logvar.bias.data.fill_(-2.0)  # evita posterior collapse inicial

        dec_in = latent_size + cond_size
        dec_layers = [nn.Linear(dec_in, h), nn.LeakyReLU()]
        for _ in range(n_layers - 1):
            dec_layers += [nn.Linear(h, h), nn.LeakyReLU()]
        dec_layers.append(nn.Linear(h, input_size))
        if output_activation == "relu":
            dec_layers.append(nn.ReLU())
        self.decoder = nn.Sequential(*dec_layers)

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
# MODELO PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════════

class HurdleLatentFlowMc(BaseModel):
    """
    Hurdle model com Latent Flow Matching e condicionamento mensal.

    Estágio 1 — VAE (set_stage("vae")):
        OccVAE:  x_bin  condicionado em c_emb            → BCE + KL_occ
        AmtVAE:  x_log  condicionado em [occ_mask, c_emb] → MSE_wet + KL_amt

    Estágio 2 — Flow Matching (set_stage("flow")):
        Aprende v_θ(z_t, t, c_emb) para transportar N(0,I) → z_amt
        usando o caminho OT linear: z_t = (1-t)*z0 + t*z1, u_t = z1-z0
        O VAE é congelado neste estágio.

    Amostragem:
        1. Sorteia z0 ~ N(0,I), integra ODE com Euler/Heun → z_amt_pred
        2. Decodifica z_amt_pred via AmtVAE.decode → x_log → expm1 → x_amt
        3. OccVAE: z_occ ~ N(0,I) → decode → sigmoid → Bernoulli → occ_mask
        4. Retorna x_amt * occ_mask

    Parâmetros:
        input_size:    número de estações (S)
        latent_occ:    dimensão latente do OccVAE
        latent_amt:    dimensão latente do AmtVAE  (= dimensão do espaço de fluxo)
        hidden_size:   largura das camadas ocultas do FlowNet
        n_layers:      profundidade do FlowNet (ResBlocks com FiLM)
        t_embed_dim:   dimensão do embedding sinusoidal de tempo
        n_sample_steps: passos de integração Euler/Heun na amostragem
        vae_layers:    profundidade dos VAEs (padrão: 2 camadas)
    """

    def __init__(
        self,
        input_size: int = 15,
        latent_occ: int = 32,
        latent_amt: int = 64,
        hidden_size: int = 256,
        n_layers: int = 6,
        t_embed_dim: int = 64,
        n_sample_steps: int = 100,
        vae_layers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.input_size     = input_size
        self.latent_occ     = latent_occ
        self.latent_amt     = latent_amt
        self.n_sample_steps = n_sample_steps
        self._stage         = "vae"  # "vae" ou "flow"

        # ── Condicionamento mensal ─────────────────────────────────────────────
        self.cond_block = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        cond_dim = self.cond_block.total_dim   # 6 para month-only

        # ── Estágio 1: OccVAE ─────────────────────────────────────────────────
        # Entrada: x_bin (S,) | Condição: c_emb (cond_dim,)
        self.occ_vae = _CondVAE(
            input_size=input_size,
            cond_size=cond_dim,
            latent_size=latent_occ,
            output_activation="none",
            n_layers=vae_layers,
        )

        # ── Estágio 1: AmtVAE ─────────────────────────────────────────────────
        # Entrada: x_log (S,) | Condição: [occ_mask (S,), c_emb (cond_dim,)]
        self.amt_vae = _CondVAE(
            input_size=input_size,
            cond_size=input_size + cond_dim,
            latent_size=latent_amt,
            output_activation="relu",
            n_layers=vae_layers,
        )

        # ── Estágio 2: LatentFlowNet ──────────────────────────────────────────
        # Opera no espaço latente do AmtVAE: R^{latent_amt}
        # Condicionado em c_emb via FiLM
        self.flow_net = _LatentFlowNet(
            latent_dim=latent_amt,
            cond_dim=cond_dim,
            hidden_dim=hidden_size,
            n_layers=n_layers,
            t_embed_dim=t_embed_dim,
        )

        # ── Distribuição empírica dos condicionadores (para sample sem cond) ──
        self._cond_probs: dict[str, np.ndarray] = {}

        print(
            f"[HurdleLatentFlowMc] S={input_size} | "
            f"latent_occ={latent_occ} | latent_amt={latent_amt} | "
            f"flow: hidden={hidden_size} layers={n_layers} | "
            f"params={self.count_parameters():,}"
        )

    # ── API de estágio ─────────────────────────────────────────────────────────

    def set_stage(self, stage: str):
        """
        Alterna entre "vae" e "flow".
        No estágio "flow", o VAE é congelado (requires_grad=False).
        """
        assert stage in ("vae", "flow"), f"stage deve ser 'vae' ou 'flow', recebeu '{stage}'"
        self._stage = stage
        vae_requires_grad = (stage == "vae")
        for p in self.occ_vae.parameters():
            p.requires_grad_(vae_requires_grad)
        for p in self.amt_vae.parameters():
            p.requires_grad_(vae_requires_grad)
        for p in self.flow_net.parameters():
            p.requires_grad_(stage == "flow")
        print(f"[HurdleLatentFlowMc] Estágio: {stage.upper()} | "
              f"parâmetros ativos: {self.count_parameters():,}")

    # ── Distribuição empírica ──────────────────────────────────────────────────

    def set_cond_distribution(self, cond_arrays: dict[str, np.ndarray]):
        """Armazena probabilidades empíricas para uso em sample() sem cond explícito."""
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

    # ── Loss ───────────────────────────────────────────────────────────────────

    def loss(self, x: Tensor, beta: float = 1.0, cond: dict = None) -> dict:
        """
        Estágio VAE  → BCE_occ + KL_occ + MSE_wet + KL_amt
        Estágio Flow → MSE(v_pred, u_t) no espaço latente do AmtVAE

        beta: peso do KL (apenas estágio VAE; ignorado no Flow)
        cond: dict[str, LongTensor (B,)] com chaves dos condicionadores
        """
        if cond is None:
            cond = self.cond_block.sample_cond(x.shape[0], self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond = {k: v.to(x.device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)    # (B, cond_dim)

        if self._stage == "vae":
            return self._vae_loss(x, c_emb, beta)
        else:
            return self._flow_loss(x, c_emb)

    def _vae_loss(self, x: Tensor, c_emb: Tensor, beta: float) -> dict:
        """Loss combinada dos dois VAEs."""
        # ── Ocorrência ────────────────────────────────────────────────────────
        occ_target = (x > 0).float()
        p_hat, mu_o, logvar_o, _ = self.occ_vae(occ_target, c_emb)
        bce    = F.binary_cross_entropy_with_logits(p_hat, occ_target, reduction='mean')
        kl_occ = -0.5 * torch.mean(1 + logvar_o - mu_o.pow(2) - logvar_o.exp())

        # ── Quantidade ────────────────────────────────────────────────────────
        x_log  = torch.log1p(x * occ_target)       # log1p(0)=0 nos dias secos
        amt_cond = torch.cat([occ_target, c_emb], dim=-1)
        x_hat_log, mu_a, logvar_a, _ = self.amt_vae(x_log, amt_cond)
        kl_amt = -0.5 * torch.mean(1 + logvar_a - mu_a.pow(2) - logvar_a.exp())

        wet_mask = occ_target.bool()
        if wet_mask.any():
            mse_wet = F.mse_loss(x_hat_log[wet_mask], x_log[wet_mask], reduction='mean')
        else:
            mse_wet = x.new_zeros(1).squeeze()

        total = 40 * bce + mse_wet + beta * (100 * kl_occ + kl_amt)
        return {
            'total':   total,
            'bce':     bce,
            'mse_wet': mse_wet,
            'kl_occ':  kl_occ,
            'kl_amt':  kl_amt,
        }

    def _flow_loss(self, x: Tensor, c_emb: Tensor) -> dict:
        """
        Flow Matching loss no espaço latente do AmtVAE.

        Caminho OT linear:
            z_t  = (1-t)*z0 + t*z1
            u_t  = z1 - z0          ← alvo constante ao longo da trajetória

        Usamos a média do encoder (encode_mean) como z1 para targets mais limpos.
        """
        occ_target = (x > 0).float()
        x_log      = torch.log1p(x * occ_target)
        amt_cond   = torch.cat([occ_target, c_emb], dim=-1)

        with torch.no_grad():
            # z1 = média do encoder (sem ruído) → target mais limpo para o fluxo
            mu_z1, _ = self.amt_vae.encode(x_log, amt_cond)  # (B, latent_amt)
            z1 = mu_z1

        z0 = torch.randn_like(z1)                             # ruído N(0,I)
        t  = torch.rand(z1.shape[0], device=z1.device)        # t ~ U(0,1)

        # Interpolação linear (OT path)
        t_ = t[:, None]
        z_t = (1 - t_) * z0 + t_ * z1

        # Alvo de velocidade (constante no caminho OT)
        u_t = z1 - z0

        # Velocidade predita
        v_pred = self.flow_net(z_t, t, c_emb)

        fm_loss = F.mse_loss(v_pred, u_t)
        return {'total': fm_loss, 'fm': fm_loss}

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
        Geração em dois passos:
            1. OccVAE: z_occ ~ N(0,I) → decode → sigmoid → Bernoulli → occ_mask
            2. Flow:   z0    ~ N(0,I) → ODE → z_amt → AmtVAE.decode → expm1 → x_amt
            3. Retorna x_amt * occ_mask

        Args:
            n:      número de amostras
            steps:  passos de integração da ODE (padrão: self.n_sample_steps)
            method: 'euler' (padrão) ou 'heun' (mais preciso, 2x NFE)
            cond:   dict de condicionamento; se None, sorteia da distribuição empírica
        """
        device = next(self.parameters()).device

        if cond is None:
            cond = self.cond_block.sample_cond(n, self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond   = {k: v.to(device) for k, v in cond.items()}
        c_emb  = self.cond_block(cond)                      # (n, cond_dim)

        n_steps = steps if steps is not None else self.n_sample_steps
        solver  = method if method is not None else 'euler'

        # ── Passo 1: ocorrência ───────────────────────────────────────────────
        z_occ    = torch.randn(n, self.latent_occ, device=device)
        p_rain   = torch.sigmoid(self.occ_vae.decode(z_occ, c_emb))
        occ_mask = torch.bernoulli(p_rain)                   # (n, S) — binário

        # ── Passo 2: integra ODE no espaço latente do AmtVAE ─────────────────
        z  = torch.randn(n, self.latent_amt, device=device)
        dt = 1.0 / n_steps

        if solver == 'euler':
            for i in range(n_steps):
                t_val = i / n_steps
                t_vec = z.new_full((n,), t_val)
                z = z + dt * self.flow_net(z, t_vec, c_emb)

        elif solver == 'heun':
            # Heun (RK2 explícito) — 2x NFE, curvas mais suaves
            for i in range(n_steps):
                t_val  = i / n_steps
                t_next = (i + 1) / n_steps
                t_vec  = z.new_full((n,), t_val)
                t_vec2 = z.new_full((n,), t_next)
                v1     = self.flow_net(z, t_vec, c_emb)
                z_tilde = z + dt * v1
                v2     = self.flow_net(z_tilde, t_vec2, c_emb)
                z = z + dt * 0.5 * (v1 + v2)

        else:
            raise ValueError(f"method deve ser 'euler' ou 'heun', recebeu '{solver}'")

        # ── Passo 3: decodifica z_amt → precipitação ──────────────────────────
        amt_cond  = torch.cat([occ_mask, c_emb], dim=-1)
        x_log_hat = self.amt_vae.decode(z, amt_cond)        # (n, S) espaço log1p
        x_amt     = torch.expm1(F.relu(x_log_hat))          # inverte log1p, garante ≥ 0

        return x_amt * occ_mask