"""
copula.py — Cópula Gaussiana (baseline estatístico clássico)

Este é o estado-da-arte em hidrologia estocástica: sem redes neurais, sem gradientes.
O ajuste é puramente analítico sobre os dados históricos.

Pergunta respondida:
    "O baseline estatístico clássico da hidrologia consegue bons cenários?"

Matemática:
    Fitting (etapa única):
        - Por estação: ajusta distribuição marginal mista
            P(X=0) = proporção de dias secos
            X | X>0 ~ Log-Normal(mu_log, sigma_log)
        - Estrutura de dependência: matriz de correlação empírica das
          variáveis transformadas via normal-score (rank-based)

    Geração (.sample(n)):
        z ~ MVN(0, C_empirica)       # amostra gaussiana correlacionada
        u[i] = Phi(z[i])             # transforma em uniformes marginais
        precip[i] = InvCDF_misto(u[i])  # aplica CDF inversa mista
"""

import numpy as np
import torch
from scipy import stats
from scipy.linalg import cholesky

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from base_model import BaseModel


class GaussianCopula(BaseModel):
    """
    Cópula Gaussiana com marginais mistas (massa em zero + Log-Normal).

    Não usa redes neurais — geração via NumPy puro.
    count_parameters() retorna 0.
    loss() retorna {'total': tensor(0.0)} pois o ajuste é analítico.
    """

    def __init__(self, input_size: int = 15, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.fitted = False

        # Parâmetros estimados por fit()
        self.p_dry = None       # (S,) — probabilidade de dia seco por estação
        self.mu_log = None      # (S,) — média do log nos dias chuvosos
        self.sigma_log = None   # (S,) — std do log nos dias chuvosos
        self.corr_matrix = None # (S, S) — matriz de correlação normal-score
        self.chol = None        # (S, S) — decomposição de Cholesky de corr_matrix

    def fit(self, data: np.ndarray, **kwargs):
        """
        Ajusta a cópula aos dados históricos.

        Args:
            data: np.ndarray (N, S) — precipitação em mm/dia (brutos, sem NaN)
        """
        N, S = data.shape
        self.input_size = S

        # --- Ajuste das marginais ---
        p_dry = np.zeros(S)
        mu_log = np.zeros(S)
        sigma_log = np.ones(S)

        for i in range(S):
            x = data[:, i]
            dry = x == 0
            p_dry[i] = dry.mean()
            wet = x[~dry]
            if len(wet) > 2:
                mu_log[i] = np.mean(np.log(wet + 1e-8))
                sigma_log[i] = np.std(np.log(wet + 1e-8))
                sigma_log[i] = max(sigma_log[i], 1e-6)

        self.p_dry = p_dry
        self.mu_log = mu_log
        self.sigma_log = sigma_log

        # --- Transformada normal-score para capturar dependência ---
        # Normal-score: ranqueia cada coluna e transforma para quantis N(0,1)
        z_scores = np.zeros_like(data, dtype=float)
        for i in range(S):
            x = data[:, i]
            ranks = stats.rankdata(x) / (N + 1)   # uniformes em (0,1)
            ranks = np.clip(ranks, 1e-6, 1 - 1e-6)
            z_scores[:, i] = stats.norm.ppf(ranks)

        # Matriz de correlação empírica dos normal-scores
        self.corr_matrix = np.corrcoef(z_scores, rowvar=False)
        # Substitui NaN/inf (estações de variância zero) por 0 (independentes)
        self.corr_matrix = np.nan_to_num(self.corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
        np.fill_diagonal(self.corr_matrix, 1.0)
        # Garante simetria e positividade
        self.corr_matrix = (self.corr_matrix + self.corr_matrix.T) / 2
        np.fill_diagonal(self.corr_matrix, 1.0)

        # Decomposição de Cholesky para amostragem eficiente
        try:
            self.chol = cholesky(self.corr_matrix, lower=True)
        except Exception:
            # Fallback: regularizar se não for PD
            self.corr_matrix += np.eye(S) * 1e-6
            self.chol = cholesky(self.corr_matrix, lower=True)

        self.fitted = True
        print(f"[GaussianCopula] Ajustado. p_dry médio: {p_dry.mean():.3f}, "
              f"mu_log médio: {mu_log.mean():.3f}")

    def _inverse_marginal(self, u: np.ndarray) -> np.ndarray:
        """
        Aplica CDF inversa mista (massa em zero + Log-Normal) para cada estação.

        u: (n, S) — uniformes marginais em [0, 1]
        returns: (n, S) — precipitação em mm/dia
        """
        n, S = u.shape
        out = np.zeros((n, S))
        for i in range(S):
            p0 = self.p_dry[i]
            # Dias secos: u <= p0 → precipitação = 0
            wet_mask = u[:, i] > p0
            if wet_mask.any():
                # Escala u para o intervalo da parte contínua
                u_wet = (u[wet_mask, i] - p0) / (1.0 - p0 + 1e-12)
                u_wet = np.clip(u_wet, 1e-8, 1 - 1e-8)
                # Inversa da Log-Normal
                log_x = stats.norm.ppf(u_wet) * self.sigma_log[i] + self.mu_log[i]
                out[wet_mask, i] = np.exp(log_x)
        return out

    def sample(self, n: int, steps: int | None = None, method: str | None = None) -> torch.Tensor:
        """
        Gera n amostras via cópula gaussiana.

        Returns:
            Tensor (n, S) — precipitação em mm/dia (espaço bruto)
        """
        if not self.fitted:
            raise RuntimeError("Chame fit() antes de sample().")

        S = self.input_size

        # 1. Amostrar gaussiana correlacionada
        z_iid = np.random.randn(n, S)
        z_corr = z_iid @ self.chol.T   # (n, S) — correlacionado

        # 2. Transformar em uniformes via CDF normal
        u = stats.norm.cdf(z_corr)     # (n, S) em [0, 1]

        # 3. Aplicar CDF inversa mista
        precip = self._inverse_marginal(u)
        return torch.FloatTensor(precip)

    def loss(self, x: torch.Tensor, beta: float = 1.0) -> dict:
        """Cópula não usa gradiente — retorna zero."""
        zero = torch.tensor(0.0, requires_grad=False)
        return {'total': zero}

    def count_parameters(self) -> int:
        return 0
