"""
hurdle_simple_mc.py — Hurdle Simples com Condicionamento Mensal (nn.Embedding)

Variante condicionada de hurdle_simple.py usando ConditioningBlock.
O embedding mensal é concatenado à entrada das duas MLPs, permitindo que o modelo
aprenda distribuições específicas para cada mês (sazonalidade).

Interface compatível com BaseModel:
    loss(x, beta=1.0, cond={"month": LongTensor})
    sample(n, cond={"month": LongTensor})
    set_cond_distribution({"month": np.ndarray(N,)})

Sem cond explícito, o modelo amostra da distribuição empírica armazenada —
mantendo compatibilidade com evaluate_model() e compare.py.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from scipy import stats
from scipy.linalg import cholesky

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel
from models.conditioning import ConditioningBlock, DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS


class _OccurrenceMLP(nn.Module):
    """MLP de ocorrência condicionado: (x + embedding) → p_rain."""

    def __init__(self, input_size: int, embed_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size + embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size),
        )

    def forward(self, x: Tensor, c_emb: Tensor) -> Tensor:
        """x: (B, S), c_emb: (B, E) → p_rain: (B, S)"""
        return torch.sigmoid(self.net(torch.cat([x, c_emb], dim=1)))


class _AmountMLP(nn.Module):
    """MLP de quantidade condicionado: (occ_mask + embedding) → (mu, sigma)."""

    def __init__(self, input_size: int, embed_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size + embed_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size * 2),
        )
        self.input_size = input_size

    def forward(self, occ_mask: Tensor, c_emb: Tensor):
        """
        occ_mask: (B, S), c_emb: (B, E)
        Returns: mu_log (B, S), sigma_log (B, S)
        """
        out = self.net(torch.cat([occ_mask, c_emb], dim=1))
        mu = out[:, :self.input_size]
        sigma = F.softplus(out[:, self.input_size:]) + 1e-4
        return mu, sigma


class HurdleSimpleMC(BaseModel):
    """
    Hurdle Simples com condicionamento mensal via nn.Embedding.

    O ConditioningBlock é genérico: adicionar ENSO, estação do ano ou outros
    condicionadores requer apenas atualizar DEFAULT_CATEGORICALS em conditioning.py
    e passar o array correspondente em cond_arrays.
    """

    def __init__(
        self,
        input_size: int = 15,
        hidden_occ: int = 32,
        hidden_amt: int = 64,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size

        self.cond_block = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        E = self.cond_block.total_dim  # 6 para month-only

        self.occ_mlp = _OccurrenceMLP(input_size, E, hidden_occ)
        self.amt_mlp = _AmountMLP(input_size, E, hidden_amt)

        # Matrizes de correlação da cópula (estimadas em fit_copulas)
        self.register_buffer('C_occ', torch.eye(input_size))
        self.register_buffer('C_amt', torch.eye(input_size))
        self._chol_occ = np.eye(input_size)
        self._chol_amt = np.eye(input_size)
        self._copulas_fitted = False

        # Distribuição empírica dos condicionadores (set_cond_distribution)
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

    def fit_copulas(self, data_raw: np.ndarray):
        """
        Estima as matrizes de correlação das cópulas a partir dos dados brutos.
        Idêntico ao hurdle_simple.py.

        Args:
            data_raw: (N, S) — precipitação bruta em mm/dia
        """
        N, S = data_raw.shape

        # --- Cópula de ocorrência ---
        occ = (data_raw > 0).astype(float)
        z_occ = np.zeros_like(occ)
        for i in range(S):
            ranks = stats.rankdata(occ[:, i]) / (N + 1)
            ranks = np.clip(ranks, 1e-6, 1 - 1e-6)
            z_occ[:, i] = stats.norm.ppf(ranks)
        C_occ = np.corrcoef(z_occ, rowvar=False)
        C_occ = (C_occ + C_occ.T) / 2
        np.fill_diagonal(C_occ, 1.0)
        C_occ += np.eye(S) * 1e-6

        # --- Cópula de quantidade ---
        log_precip = np.where(data_raw > 0, np.log(data_raw + 1e-8), np.nan)
        for i in range(S):
            col = log_precip[:, i]
            col_nonan = col[~np.isnan(col)]
            if len(col_nonan) > 0:
                fill = col_nonan.mean()
                log_precip[:, i] = np.where(np.isnan(col), fill, col)
        z_amt = np.zeros_like(log_precip)
        for i in range(S):
            ranks = stats.rankdata(log_precip[:, i]) / (N + 1)
            ranks = np.clip(ranks, 1e-6, 1 - 1e-6)
            z_amt[:, i] = stats.norm.ppf(ranks)
        C_amt = np.corrcoef(z_amt, rowvar=False)
        C_amt = (C_amt + C_amt.T) / 2
        np.fill_diagonal(C_amt, 1.0)
        C_amt += np.eye(S) * 1e-6

        self.C_occ = torch.FloatTensor(C_occ)
        self.C_amt = torch.FloatTensor(C_amt)
        try:
            self._chol_occ = cholesky(C_occ, lower=True)
            self._chol_amt = cholesky(C_amt, lower=True)
        except Exception:
            self._chol_occ = np.eye(S)
            self._chol_amt = np.eye(S)
        self._copulas_fitted = True
        print("[HurdleSimpleMC] Cópulas ajustadas.")

    def loss(self, x: Tensor, beta: float = 1.0, cond: dict = None) -> dict:
        """
        Loss total = BCE (ocorrência) + NLL Log-Normal (quantidade)

        Args:
            x: (B, S) — dados normalizados
            cond: dict[str, LongTensor(B,)] ou None (amostra da distribuição empírica)
        """
        if cond is None:
            cond = self.cond_block.sample_cond(x.shape[0], self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond = {k: v.to(x.device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)  # (B, E)

        occ_target = (x > 0).float()

        # --- BCE (ocorrência) ---
        dummy_input = torch.zeros_like(x)
        p_rain = self.occ_mlp(dummy_input, c_emb)
        p_rain = p_rain.clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(p_rain, occ_target, reduction='mean')

        # --- NLL Log-Normal (quantidade, apenas dias chuvosos) ---
        mu, sigma = self.amt_mlp(occ_target, c_emb)

        wet = x > 0
        if wet.any():
            log_x = torch.log(x[wet].clamp(min=1e-8))
            log_x_hat = mu[wet]
            sigma_hat = sigma[wet]
            nll = 0.5 * torch.mean(
                ((log_x - log_x_hat) / sigma_hat) ** 2
                + 2 * torch.log(sigma_hat)
                + torch.log(x[wet].clamp(min=1e-8))
            )
        else:
            nll = torch.tensor(0.0, device=x.device)

        total = bce + nll
        return {'total': total, 'bce': bce, 'nll': nll}

    def sample(self, n: int, cond: dict = None, steps=None, method=None) -> Tensor:
        """
        Gera n amostras usando cópulas gaussianas para correlação espacial.

        Args:
            cond: dict[str, LongTensor(n,)] ou None (usa distribuição empírica)
        """
        S = self.input_size
        device = next(self.parameters()).device

        if cond is None:
            cond = self.cond_block.sample_cond(n, self._cond_probs, continuous_data=getattr(self, '_continuous_data', None))
        cond = {k: v.to(device) for k, v in cond.items()}
        c_emb = self.cond_block(cond)  # (n, E)

        # --- Ocorrência correlacionada ---
        with torch.no_grad():
            dummy = torch.zeros(n, S, device=device)
            p_rain = self.occ_mlp(dummy, c_emb).cpu().numpy()  # (n, S)

        z_occ = np.random.randn(n, S) @ self._chol_occ.T
        u_occ = stats.norm.cdf(z_occ)
        occur = (u_occ < p_rain).astype(float)  # (n, S)

        # --- Quantidade correlacionada ---
        occ_tensor = torch.FloatTensor(occur).to(device)
        with torch.no_grad():
            mu, sigma = self.amt_mlp(occ_tensor, c_emb)
        mu = mu.cpu().numpy()
        sigma = sigma.cpu().numpy()

        z_amt = np.random.randn(n, S) @ self._chol_amt.T
        u_amt = stats.norm.cdf(z_amt)
        u_amt = np.clip(u_amt, 1e-8, 1 - 1e-8)
        log_a = mu + sigma * stats.norm.ppf(u_amt)
        amount = np.exp(log_a)

        precip = occur * amount
        return torch.FloatTensor(precip)
