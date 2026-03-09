"""
hurdle_simple.py — Hurdle Simples: MLP + Log-Normal + Cópula Gaussiana

Sem espaço latente. Duas MLPs aprendem parâmetros de distribuições conhecidas.
Correlação espacial modelada por cópulas gaussianas separadas (ocorrência e quantidade).

Pergunta respondida:
    "Uma rede neural simples com distribuições paramétricas bate a cópula?"

Arquitetura:
    Part 1 — OccurrenceMLP (BCE loss):
        Input: nenhum (parâmetros por estação diretamente)
        Output: Sigmoid → p_rain[i] por estação
        Correlação: Cópula Gaussiana C_occ = corr(I(precip > 0))

    Part 2 — AmountMLP (NLL Log-Normal, apenas em dias chuvosos):
        Input: occ_mask (shape S) — máscara de ocorrência
        Output: (mu_log[i], softplus(sigma_log[i])) por estação
        Correlação: Cópula Gaussiana C_amt = corr(log(precip[wet]))

Geração (.sample(n)):
    1. p_rain = sigmoid(occ_params)
       z_occ ~ MVN(0, C_occ) → occur[i] = (Phi(z_occ[i]) < p_rain[i])
    2. mu, sigma = AmountMLP(occ_mean_vector)
       z_amt ~ MVN(0, C_amt) → u[i] = Phi(z_amt[i])
       log_a = mu[i] + sigma[i] * norm.ppf(u[i])
       amount[i] = exp(log_a)
    3. precip = occur * amount
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


class _OccurrenceMLP(nn.Module):
    """MLP simples para probabilidade de ocorrência por estação."""

    def __init__(self, input_size: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size),
        )

    def forward(self, x: Tensor) -> Tensor:
        """x: (B, S) → p_rain: (B, S) em [0, 1]"""
        return torch.sigmoid(self.net(x))


class _AmountMLP(nn.Module):
    """MLP para parâmetros Log-Normal (mu_log, sigma_log) por estação."""

    def __init__(self, input_size: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_size * 2),  # mu e logvar por estação
        )
        self.input_size = input_size

    def forward(self, occ_mask: Tensor):
        """
        occ_mask: (B, S) — máscara de ocorrência
        Returns: mu_log (B, S), sigma_log (B, S)
        """
        out = self.net(occ_mask)
        mu = out[:, :self.input_size]
        sigma = F.softplus(out[:, self.input_size:]) + 1e-4
        return mu, sigma


class HurdleSimple(BaseModel):
    """
    Modelo Hurdle Simples: MLP aprendem parâmetros paramétricos + cópula para correlação.

    Mais interpretável que o VAE: cada componente tem significado físico direto.
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

        self.occ_mlp = _OccurrenceMLP(input_size, hidden_occ)
        self.amt_mlp = _AmountMLP(input_size, hidden_amt)

        # Matrizes de correlação da cópula (estimadas em fit_copulas)
        self.register_buffer('C_occ', torch.eye(input_size))
        self.register_buffer('C_amt', torch.eye(input_size))
        self._chol_occ = np.eye(input_size)
        self._chol_amt = np.eye(input_size)
        self._copulas_fitted = False

    def fit_copulas(self, data_raw: np.ndarray):
        """
        Estima as matrizes de correlação das cópulas a partir dos dados brutos.
        Deve ser chamado uma vez após load_data().

        Args:
            data_raw: (N, S) — precipitação bruta em mm/dia
        """
        N, S = data_raw.shape

        # --- Cópula de ocorrência: normal-score de I(precip > 0) ---
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

        # --- Cópula de quantidade: normal-score de log(precip) nos dias chuvosos ---
        # Usa média da precipitação em dias secos para construir série completa
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
        print("[HurdleSimple] Cópulas ajustadas.")

    def loss(self, x: Tensor, beta: float = 1.0) -> dict:
        """
        Loss total = BCE (ocorrência) + NLL Log-Normal (quantidade)

        Args:
            x: (B, S) — dados normalizados (scale_only recomendado)
        """
        # Máscara de ocorrência (dias chuvosos)
        occ_target = (x > 0).float()

        # --- Part 1: BCE ---
        dummy_input = torch.zeros_like(x)
        p_rain = self.occ_mlp(dummy_input)
        p_rain = p_rain.clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(p_rain, occ_target, reduction='mean')

        # --- Part 2: NLL Log-Normal nos dias chuvosos ---
        mu, sigma = self.amt_mlp(occ_target)

        wet = x > 0
        if wet.any():
            log_x = torch.log(x[wet].clamp(min=1e-8))
            log_x_hat = mu[wet]
            sigma_hat = sigma[wet]
            nll = 0.5 * torch.mean(
                ((log_x - log_x_hat) / sigma_hat) ** 2
                + 2 * torch.log(sigma_hat)
                + torch.log(x[wet].clamp(min=1e-8))  # jacobiano do log
            )
        else:
            nll = torch.tensor(0.0, device=x.device)

        total = bce + nll
        return {'total': total, 'bce': bce, 'nll': nll}

    def sample(self, n: int, steps: int | None = None, method: str | None = None) -> Tensor:
        """
        Gera n amostras usando cópulas gaussianas para correlação espacial.
        """
        S = self.input_size
        device = next(self.parameters()).device

        # --- Part 1: Ocorrência correlacionada ---
        with torch.no_grad():
            dummy = torch.zeros(1, S, device=device)
            p_rain = self.occ_mlp(dummy).squeeze(0).cpu().numpy()  # (S,)

        z_occ = np.random.randn(n, S) @ self._chol_occ.T
        u_occ = stats.norm.cdf(z_occ)
        occur = (u_occ < p_rain[None, :]).astype(float)  # (n, S)

        # --- Part 2: Quantidade correlacionada ---
        occ_mean = torch.FloatTensor(occur).to(device)
        with torch.no_grad():
            mu, sigma = self.amt_mlp(occ_mean)
        mu = mu.cpu().numpy()
        sigma = sigma.cpu().numpy()

        z_amt = np.random.randn(n, S) @ self._chol_amt.T
        u_amt = stats.norm.cdf(z_amt)
        u_amt = np.clip(u_amt, 1e-8, 1 - 1e-8)
        log_a = mu + sigma * stats.norm.ppf(u_amt)
        amount = np.exp(log_a)

        precip = occur * amount
        return torch.FloatTensor(precip)
