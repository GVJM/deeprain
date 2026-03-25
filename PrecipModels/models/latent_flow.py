"""
latent_flow.py — Conditional Flow Matching com Transformer para precipitação.

Wraps a arquitetura de LatentFlowMatching/test.py no interface BaseModel,
tornando-a comparável aos demais modelos no PrecipModels/compare.py.

Diferenças em relação ao flow_match.py (modelo simples):
    - Contexto temporal: janela de 30 dias de precipitação passada
    - Condicionamento sazonal: mês/dia-do-ano como embedding cíclico
    - Arquitetura Transformer (WindowEncoder com TransformerEncoder)
    - Blocos AdaLN com conditioning de tempo + contexto + sazonalidade
    - EMA para estabilização do treinamento

Pergunta respondida:
    "Um flow matching com contexto temporal Transformer supera o flow_match simples
     (0.724 composite) que não tem consciência temporal?"

Estratégia de normalização:
    O modelo usa sua própria transformação interna (log1p + padronização por estação).
    Recebe dados em escala "scale_only" de train.py e converte internamente.
    sample() retorna valores no espaço scale_only normalizado — compatível com
    evaluate_model(..., samples_are_normalized=True) usando mu=0, std=std.

    Como scale_only: x_norm = x_raw / std
    Internamente:    x_internal = log1p(x_raw) / station_std_log
    sample() devolve: x_raw / std_scale = x_internal_inverse / std_scale

Amostragem sem contexto real:
    Para evaluate_model (sem série temporal disponível), sorteia janelas aleatórias
    do conjunto de treino armazenado como contexto.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel


# =============================================================================
# 1. Configuração
# =============================================================================

@dataclass
class _FlowConfig:
    n_stations: int = 15
    window_size: int = 30

    # Transformação interna
    log_transform: bool = True
    clip_max: float = 300.0

    # Rede de velocidade
    hidden_dim: int = 128
    n_layers: int = 4
    n_heads: int = 4
    dropout: float = 0.1

    # Conditioning
    use_temporal_cond: bool = True
    use_window_cond: bool = True
    temporal_cond_dim: int = 32
    window_cond_dim: int = 64

    # Flow matching
    sigma_min: float = 1e-4
    time_sampling: str = "logit_normal"
    logit_normal_mean: float = 0.0
    logit_normal_std: float = 1.0

    # Treinamento (não usados no loss; usados externamente)
    ema_decay: float = 0.999


# =============================================================================
# 2. Transformação de precipitação (interna ao modelo)
# =============================================================================

class _PrecipTransform:
    """log1p + padronização por estação (fit nos dados de treino brutos)."""

    def __init__(self, config: _FlowConfig):
        self.config = config
        self.station_mean: Optional[np.ndarray] = None
        self.station_std: Optional[np.ndarray] = None

    def fit(self, data: np.ndarray):
        clipped = np.clip(data, 0, self.config.clip_max)
        transformed = np.log1p(clipped) if self.config.log_transform else clipped
        self.station_mean = np.nanmean(transformed, axis=0)
        self.station_std = np.nanstd(transformed, axis=0)
        self.station_std = np.maximum(self.station_std, 1e-6)
        return self

    def forward(self, data: np.ndarray) -> np.ndarray:
        clipped = np.clip(data, 0, self.config.clip_max)
        x = np.log1p(clipped) if self.config.log_transform else clipped
        return (x - self.station_mean) / self.station_std

    def inverse(self, x: np.ndarray) -> np.ndarray:
        denorm = x * self.station_std + self.station_mean
        if self.config.log_transform:
            return np.expm1(np.clip(denorm, -10, 10))
        return np.clip(denorm, 0, None)

    def forward_tensor(self, data: Tensor) -> Tensor:
        mean = torch.tensor(self.station_mean, dtype=data.dtype, device=data.device)
        std = torch.tensor(self.station_std, dtype=data.dtype, device=data.device)
        clipped = data.clamp(0, self.config.clip_max)
        x = torch.log1p(clipped) if self.config.log_transform else clipped
        return (x - mean) / std

    def inverse_tensor(self, x: Tensor) -> Tensor:
        mean = torch.tensor(self.station_mean, dtype=x.dtype, device=x.device)
        std = torch.tensor(self.station_std, dtype=x.dtype, device=x.device)
        denorm = x * std + mean
        if self.config.log_transform:
            return torch.expm1(denorm.clamp(-10, 10))
        return denorm.clamp(min=0)


# =============================================================================
# 3. Rede de velocidade (Transformer + AdaLN)
# =============================================================================

class _SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.SiLU(), nn.Linear(dim * 4, dim))

    def forward(self, t: Tensor) -> Tensor:
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        args = t * freqs
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return self.mlp(emb)


class _WindowEncoder(nn.Module):
    """Transformer encoder da janela histórica → vetor de contexto."""

    def __init__(self, n_stations: int, window_size: int, hidden_dim: int, output_dim: int, n_heads: int = 4):
        super().__init__()
        self.input_proj = nn.Linear(n_stations, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, window_size, hidden_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.output_proj = nn.Linear(hidden_dim, output_dim)

    def forward(self, window: Tensor) -> Tensor:
        """window: (B, W, S) → context: (B, output_dim)"""
        h = self.input_proj(window) + self.pos_embed
        h = self.transformer(h)
        h = h.mean(dim=1)
        return self.output_proj(h)


class _AdaLNBlock(nn.Module):
    """MLP block com Adaptive Layer Norm conditioning."""

    def __init__(self, dim: int, cond_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False)
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.dropout = nn.Dropout(dropout)
        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 3 * dim))
        nn.init.zeros_(self.adaLN[-1].weight)
        nn.init.zeros_(self.adaLN[-1].bias)

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        scale, shift, gate = self.adaLN(cond).chunk(3, dim=-1)
        h = self.norm1(x) * (1 + scale) + shift
        h = self.linear1(h)
        h = F.gelu(h)
        h = self.dropout(h)
        h = self.linear2(h)
        return x + gate * h


class _VelocityNet(nn.Module):
    """
    v_θ(z_t, t, window, temporal) — prediz velocidade do flow matching.
    """

    def __init__(self, config: _FlowConfig):
        super().__init__()
        self.config = config
        H = config.hidden_dim
        data_dim = config.n_stations

        self.time_embed = _SinusoidalTimeEmbed(H)
        cond_dim = H

        if config.use_window_cond:
            self.window_encoder = _WindowEncoder(
                config.n_stations, config.window_size,
                max(H // 2, config.n_heads),   # hidden_dim divisível por n_heads
                config.window_cond_dim,
                n_heads=config.n_heads,
            )
            self.window_proj = nn.Linear(config.window_cond_dim, H)

        if config.use_temporal_cond:
            self.temporal_proj = nn.Sequential(
                nn.Linear(4, config.temporal_cond_dim),
                nn.SiLU(),
                nn.Linear(config.temporal_cond_dim, H),
            )

        self.input_proj = nn.Linear(data_dim, H)
        self.blocks = nn.ModuleList([_AdaLNBlock(H, cond_dim, config.dropout) for _ in range(config.n_layers)])

        self.final_norm = nn.LayerNorm(H, elementwise_affine=False)
        self.final_adaLN = nn.Sequential(nn.SiLU(), nn.Linear(H, 2 * H))
        self.output_proj = nn.Linear(H, data_dim)
        nn.init.zeros_(self.output_proj.weight)
        nn.init.zeros_(self.output_proj.bias)
        nn.init.zeros_(self.final_adaLN[-1].weight)
        nn.init.zeros_(self.final_adaLN[-1].bias)

    def forward(
        self,
        z_t: Tensor,
        t: Tensor,
        window: Optional[Tensor] = None,
        temporal: Optional[Tensor] = None,
    ) -> Tensor:
        cond = self.time_embed(t)

        if window is not None and self.config.use_window_cond:
            cond = cond + self.window_proj(self.window_encoder(window))

        if temporal is not None and self.config.use_temporal_cond:
            cond = cond + self.temporal_proj(temporal)

        h = self.input_proj(z_t)
        for block in self.blocks:
            h = block(h, cond)

        scale, shift = self.final_adaLN(cond).chunk(2, dim=-1)
        h = self.final_norm(h) * (1 + scale) + shift
        return self.output_proj(h)


# =============================================================================
# 4. EMA
# =============================================================================

class _EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.clone().detach() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.shadow[name].lerp_(p.data, 1 - self.decay)

    def apply(self, model: nn.Module):
        self.backup = {}
        for name, p in model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        for name, p in model.named_parameters():
            if name in self.backup:
                p.data.copy_(self.backup[name])
        self.backup = {}


# =============================================================================
# 5. Modelo BaseModel wrapper
# =============================================================================

class LatentFlowWrapper(BaseModel):
    """
    Conditional Flow Matching Transformer para precipitação diária.

    Compatível com o framework PrecipModels (BaseModel interface).

    Antes do treino, chamar:
        model.fit_flow(train_raw, std_scale)

    O train.py trata este modelo com normalization_mode="scale_only"
    e chama model.fit_flow(train_raw, std=std).

    sample() retorna valores no espaço scale_only normalizado (x_raw / std_scale),
    compatível com evaluate_model(..., samples_are_normalized=True).
    """

    def __init__(
        self,
        input_size: int = 15,
        window_size: int = 30,
        hidden_dim: int = 128,
        n_layers: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.window_size = window_size

        self.config = _FlowConfig(
            n_stations=input_size,
            window_size=window_size,
            hidden_dim=hidden_dim,
            n_layers=n_layers,
            n_heads=4,
        )
        self.velocity_net = _VelocityNet(self.config)

        # EMA (não-parâmetro, gerenciado externamente durante treino)
        self._ema: Optional[_EMA] = None

        # Dados de treino (preenchidos por fit_flow)
        self._train_raw: Optional[np.ndarray] = None    # mm/dia
        self._train_internal: Optional[np.ndarray] = None  # log1p + padronizado
        self._std_scale: Optional[np.ndarray] = None    # std da normalização scale_only
        self._transform: Optional[_PrecipTransform] = None
        self._n_train: int = 0
        # Janelas e encodings sazonais pré-construídos (evita list comprehension por batch)
        self._windows_cache: Optional[np.ndarray] = None    # (N-W, W, S) float32
        self._temporal_cache: Optional[np.ndarray] = None   # (N-W, 4) float32

    # ── Inicialização com dados (chamada por train.py) ─────────────────────────

    def fit_flow(self, train_raw: np.ndarray, std_scale: np.ndarray):
        """
        Armazena dados brutos de treino, ajusta transformação interna e
        pré-constrói todas as janelas para amostragem eficiente por batch.

        Args:
            train_raw:  (N, S) dados brutos em mm/dia
            std_scale:  (1, S) std usado para scale_only normalization (de compute_norm_params)
        """
        self._std_scale = std_scale.copy()
        self._transform = _PrecipTransform(self.config)
        self._transform.fit(train_raw)
        self._train_raw = train_raw.copy()
        self._train_internal = self._transform.forward(train_raw)
        self._n_train = train_raw.shape[0]

        # Pré-constrói janelas e encodings sazonais (evita list comprehension por batch)
        N = self._n_train
        W = self.window_size
        if N > W:
            n_windows = N - W
            # Janelas: (n_windows, W, S)
            self._windows_cache = np.stack(
                [self._train_internal[i: i + W] for i in range(n_windows)]
            ).astype(np.float32)
            # Encodings sazonais: índice i+W representa o dia de destino
            indices = np.arange(W, N)
            day = (indices % 365) / 365.25
            month = ((indices % 365) // 30) / 11.0
            self._temporal_cache = np.column_stack([
                np.sin(2 * np.pi * day),
                np.cos(2 * np.pi * day),
                np.sin(2 * np.pi * month),
                np.cos(2 * np.pi * month),
            ]).astype(np.float32)
        else:
            self._windows_cache = None
            self._temporal_cache = None

        # Inicializa EMA após saber os parâmetros do modelo
        self._ema = _EMA(self.velocity_net, self.config.ema_decay)
        print(f"[LatentFlow] Dados armazenados: {train_raw.shape} | "
              f"janelas pré-construídas: {0 if self._windows_cache is None else len(self._windows_cache)} | "
              f"EMA decay={self.config.ema_decay}")

    # ── Janelas aleatórias ─────────────────────────────────────────────────────

    def _sample_windows(self, B: int, device: torch.device):
        """
        Amostra B janelas do conjunto de treino em espaço interno.

        Usa indexação numpy sobre arrays pré-construídos em fit_flow — sem
        list comprehension por batch nem loop Python para encodings sazonais.
        """
        if self._windows_cache is None:
            return (
                torch.zeros(B, self.window_size, self.input_size, device=device),
                torch.zeros(B, 4, device=device),
            )

        idx = np.random.randint(0, len(self._windows_cache), size=B)
        windows = torch.tensor(self._windows_cache[idx], dtype=torch.float32, device=device)
        windows = torch.nan_to_num(windows, nan=0.0)
        temporal = torch.tensor(self._temporal_cache[idx], dtype=torch.float32, device=device)
        return windows, temporal

    # ── Flow matching utilities ────────────────────────────────────────────────

    def _sample_time(self, B: int, device: torch.device) -> Tensor:
        t = torch.sigmoid(
            self.config.logit_normal_mean
            + self.config.logit_normal_std * torch.randn(B, device=device)
        )
        return t.clamp(self.config.sigma_min, 1.0 - self.config.sigma_min)

    # ── Loss ───────────────────────────────────────────────────────────────────

    def loss(self, x_norm: Tensor, beta: float = 1.0) -> dict:
        """
        Flow matching loss condicional.

        x_norm é escala scale_only: x_norm = x_raw / std_scale.
        Internamente converte para x_raw e aplica transformação log1p.

        Args:
            x_norm: (B, S) — dados em escala scale_only normalization
        """
        device = x_norm.device
        B = x_norm.shape[0]

        if self._transform is None or self._std_scale is None:
            # Fallback: se fit_flow não foi chamado, usa unconditional flow
            z_1 = x_norm
            windows = torch.zeros(B, self.window_size, self.input_size, device=device)
            temporal = torch.zeros(B, 4, device=device)
        else:
            # Converte para raw e aplica transformação interna
            std_t = torch.tensor(self._std_scale, dtype=x_norm.dtype, device=device)
            x_raw = x_norm * std_t                         # (B, S) mm/dia
            z_1 = self._transform.forward_tensor(x_raw)   # (B, S) log1p-normalizado
            windows, temporal = self._sample_windows(B, device)

        z_0 = torch.randn_like(z_1)
        t = self._sample_time(B, device)
        t_expand = t.unsqueeze(-1)
        z_t = (1 - t_expand) * z_0 + t_expand * z_1
        target = z_1 - z_0

        v_pred = self.velocity_net(z_t, t, window=windows, temporal=temporal)
        fm_loss = F.mse_loss(v_pred, target)

        # Atualiza EMA se disponível
        if self._ema is not None and self.training:
            self._ema.update(self.velocity_net)

        return {"total": fm_loss, "flow": fm_loss}

    # ── Amostragem (ODE integration) ──────────────────────────────────────────

    @torch.no_grad()
    def sample(self, n: int, steps: int | None = None, method: str | None = None) -> Tensor:
        """
        Geração via ODE:
          1. z_0 ~ N(0, I)
          2. Integra ODE com contexto aleatório do histórico
          3. Inversa da transformação interna → mm/dia
          4. Divide por std_scale → escala scale_only (compatível com evaluate_model)

        Args:
            steps:  número de passos de integração (None = 30)
            method: 'midpoint' (padrão, 2ª ordem) ou 'euler' (1ª ordem)

        Returns: (n, input_size) — espaço scale_only normalizado
        """
        num_steps = steps if steps is not None else 30
        solver = method or 'midpoint'

        device = next(self.parameters()).device
        data_dim = self.input_size

        z = torch.randn(n, data_dim, device=device)
        dt = 1.0 / num_steps

        windows, temporal = self._sample_windows(n, device)

        if solver == 'euler':
            for i in range(num_steps):
                t = torch.full((n,), i * dt, device=device)
                v = self.velocity_net(z, t, window=windows, temporal=temporal)
                z = z + v * dt
        else:  # midpoint (padrão)
            for i in range(num_steps):
                t = torch.full((n,), i * dt, device=device)
                t_mid = torch.full((n,), (i + 0.5) * dt, device=device)
                v = self.velocity_net(z, t, window=windows, temporal=temporal)
                z_mid = z + v * (dt / 2)
                v_mid = self.velocity_net(z_mid, t_mid, window=windows, temporal=temporal)
                z = z + v_mid * dt

        # Inversão da transformação interna
        if self._transform is not None and self._std_scale is not None:
            z_np = z.cpu().numpy()
            x_raw = self._transform.inverse(z_np)               # mm/dia
            x_raw = np.clip(x_raw, 0, None)
            x_norm = x_raw / self._std_scale                    # scale_only
            return torch.tensor(x_norm, dtype=torch.float32)
        else:
            return z.clamp(min=0)
