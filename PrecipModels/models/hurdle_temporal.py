"""
hurdle_temporal.py — Hurdle com contexto temporal via GRU

Extensão do hurdle_simple que condiciona ocorrência e quantidade em um vetor
de contexto derivado de uma janela de 30 dias de precipitação passada.

Pergunta respondida:
    "Adicionar contexto temporal GRU ao modelo vencedor (hurdle_simple) melhora
     a distribuição gerada? O autocorrelação temporal seria capturada melhor?"

Arquitetura:
    Contexto:   window (B, W, S) → GRU → h_T → Linear → context_vec (B, context_dim)

    Ocorrência: [context_vec] → MLP → sigmoid → p_rain (B, S)
                Correlação espacial: Cópula Gaussiana (mesmo que hurdle_simple)

    Quantidade: [context_vec] → MLP → (mu_log, sigma_log) por estação
                Correlação espacial: Cópula Gaussiana (mesmo que hurdle_simple)

Estratégia de treinamento:
    O DataLoader padrão fornece amostras i.i.d. (sem ordenação temporal).
    Para usar contexto temporal durante o treinamento, o modelo armazena os dados
    de treino normalizados (via fit_temporal) e, para cada batch, amostra
    janelas aleatórias do histórico como contexto.

    Isso approxima E[loss | contexto ~ P(contexto)], não a loss condicional exata,
    mas permite que o GRU aprenda a extrair estatísticas climáticas relevantes.

Amostragem (.sample(n)):
    Para cada amostra, sorteia uma janela aleatória do treino como contexto,
    depois aplica o mesmo mecanismo de cópula do hurdle_simple.
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


# ──────────────────────────────────────────────────────────
# COMPONENTES
# ──────────────────────────────────────────────────────────

class _GRUContextEncoder(nn.Module):
    """
    Codifica janela histórica de precipitação em vetor de contexto.

    window: (B, W, S) → context: (B, context_dim)
    """

    def __init__(self, n_stations: int, window_size: int, hidden_dim: int, context_dim: int):
        super().__init__()
        self.gru = nn.GRU(
            input_size=n_stations,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=0.1,
        )
        self.proj = nn.Linear(hidden_dim, context_dim)

    def forward(self, window: Tensor) -> Tensor:
        """window: (B, W, S) → context: (B, context_dim)"""
        _, h_n = self.gru(window)   # h_n: (2, B, hidden_dim)
        h_last = h_n[-1]             # (B, hidden_dim) — última camada GRU
        return self.proj(h_last)     # (B, context_dim)


class _OccurrenceMLPCond(nn.Module):
    """MLP condicionada em contexto temporal para probabilidade de chuva."""

    def __init__(self, n_stations: int, context_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_stations),
        )

    def forward(self, context: Tensor) -> Tensor:
        """context: (B, context_dim) → p_rain: (B, S) ∈ [0,1]"""
        return torch.sigmoid(self.net(context))


class _AmountMLPCond(nn.Module):
    """MLP condicionada em contexto temporal para parâmetros Log-Normal."""

    def __init__(self, n_stations: int, context_dim: int, hidden: int = 128):
        super().__init__()
        self.n_stations = n_stations
        self.net = nn.Sequential(
            nn.Linear(context_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_stations * 2),
        )

    def forward(self, context: Tensor):
        """context: (B, context_dim) → mu_log, sigma_log: (B, S)"""
        out = self.net(context)
        mu = out[:, :self.n_stations]
        sigma = F.softplus(out[:, self.n_stations:]) + 1e-4
        return mu, sigma


# ──────────────────────────────────────────────────────────
# MODELO PRINCIPAL
# ──────────────────────────────────────────────────────────

class HurdleTemporal(BaseModel):
    """
    Hurdle model com contexto temporal GRU + cópula gaussiana espacial.

    API compatível com BaseModel.  Requer duas chamadas antes do treino:
        model.fit_copulas(train_raw)       — matriz de correlação espacial
        model.fit_temporal(train_norm)     — armazena dados normalizados para amostragem de janelas
    """

    def __init__(
        self,
        input_size: int = 15,
        window_size: int = 30,
        gru_hidden: int = 64,
        context_dim: int = 32,
        hidden_occ: int = 64,
        hidden_amt: int = 128,
        **kwargs,
    ):
        super().__init__()
        self.input_size = input_size
        self.window_size = window_size
        self.context_dim = context_dim

        # Módulos neurais
        self.context_encoder = _GRUContextEncoder(input_size, window_size, gru_hidden, context_dim)
        self.occ_mlp = _OccurrenceMLPCond(input_size, context_dim, hidden_occ)
        self.amt_mlp = _AmountMLPCond(input_size, context_dim, hidden_amt)

        # Buffers das cópulas (estimadas em fit_copulas, escala de raw)
        self.register_buffer("C_occ", torch.eye(input_size))
        self.register_buffer("C_amt", torch.eye(input_size))
        self._chol_occ = np.eye(input_size)
        self._chol_amt = np.eye(input_size)
        self._copulas_fitted = False

        # Buffer para dados normalizados de treino (preenchido em fit_temporal)
        self._train_norm: torch.Tensor | None = None
        self._n_train: int = 0
        self._windows_np: np.ndarray | None = None  # janelas pré-construídas: (N-W, W, S)

    # ── Pré-ajuste (chamado por train.py antes do loop de gradiente) ──────────

    def fit_copulas(self, data_raw: np.ndarray):
        """
        Estima matrizes de correlação das cópulas a partir dos dados brutos.
        Idêntico ao hurdle_simple.fit_copulas().
        """
        N, S = data_raw.shape

        # Cópula de ocorrência
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

        # Cópula de quantidade
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
        print("[HurdleTemporal] Cópulas ajustadas.")

    def fit_temporal(self, data_norm: np.ndarray):
        """
        Armazena dados de treino normalizados para amostragem de janelas durante loss/sample.

        Pré-constrói todas as janelas válidas como array numpy para amostragem
        eficiente por indexação (sem list comprehension por batch).

        Args:
            data_norm: (N, S) — dados normalizados (saída de normalize_with_params)
        """
        N = data_norm.shape[0]
        W = self.window_size
        self._n_train = N
        self._train_norm = torch.FloatTensor(data_norm)  # mantido por compatibilidade

        if N > W:
            # Pré-constrói (N-W, W, S): janela[j] = data_norm[j : j+W]
            self._windows_np = np.stack(
                [data_norm[i: i + W] for i in range(N - W)]
            ).astype(np.float32)
        else:
            self._windows_np = None
        print(f"[HurdleTemporal] Dados temporais armazenados: {data_norm.shape} | "
              f"janelas pré-construídas: {0 if self._windows_np is None else len(self._windows_np)}")

    # ── Janelas aleatórias ─────────────────────────────────────────────────────

    def _sample_random_windows(self, B: int, device: torch.device) -> Tensor:
        """
        Amostra B janelas aleatórias do histórico de treino.
        Retorna tensor (B, W, S).

        Usa indexação numpy sobre array pré-construído (fit_temporal) — sem
        list comprehension por batch, eliminando o gargalo de desempenho.
        """
        if self._windows_np is None:
            return torch.zeros(B, self.window_size, self.input_size, device=device)

        idx = np.random.randint(0, len(self._windows_np), size=B)
        return torch.tensor(self._windows_np[idx], dtype=torch.float32, device=device)

    # ── Loss ───────────────────────────────────────────────────────────────────

    def loss(self, x: Tensor, beta: float = 1.0) -> dict:
        """
        Loss total = BCE (ocorrência condicionada) + NLL Log-Normal (quantidade condicionada).

        Contexto: janelas aleatórias amostradas do histórico de treino.

        Args:
            x: (B, S) — dados normalizados (scale_only recomendado)
        """
        B, S = x.shape
        device = x.device

        # Contexto temporal
        windows = self._sample_random_windows(B, device)   # (B, W, S)
        context = self.context_encoder(windows)             # (B, context_dim)

        # --- Ocorrência ---
        occ_target = (x > 0).float()
        p_rain = self.occ_mlp(context).clamp(1e-6, 1.0 - 1e-6)
        bce = F.binary_cross_entropy(p_rain, occ_target, reduction="mean")

        # --- Quantidade (dias chuvosos) ---
        mu, sigma = self.amt_mlp(context)

        wet = x > 0
        if wet.any():
            log_x = torch.log(x[wet].clamp(min=1e-8))
            nll = 0.5 * torch.mean(
                ((log_x - mu[wet]) / sigma[wet]) ** 2
                + 2 * torch.log(sigma[wet])
                + torch.log(x[wet].clamp(min=1e-8))   # Jacobiano log
            )
        else:
            nll = torch.tensor(0.0, device=device)

        total = bce + nll
        return {"total": total, "bce": bce, "nll": nll}

    # ── Amostragem ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def sample(self, n: int, steps: int | None = None, method: str | None = None) -> Tensor:
        """
        Gera n amostras usando:
          1. Janela aleatória do histórico → contexto GRU
          2. Cópula gaussiana para correlação espacial de ocorrência
          3. Cópula gaussiana para correlação espacial de quantidade
        """
        S = self.input_size
        device = next(self.parameters()).device

        # Contexto
        windows = self._sample_random_windows(n, device)   # (n, W, S)
        context = self.context_encoder(windows)             # (n, context_dim)

        # --- Ocorrência correlacionada ---
        p_rain = self.occ_mlp(context).cpu().numpy()       # (n, S)

        z_occ = np.random.randn(n, S) @ self._chol_occ.T
        u_occ = stats.norm.cdf(z_occ)
        occur = (u_occ < p_rain).astype(float)              # (n, S)

        # --- Quantidade correlacionada ---
        mu, sigma = self.amt_mlp(context)
        mu_np = mu.cpu().numpy()
        sigma_np = sigma.cpu().numpy()

        z_amt = np.random.randn(n, S) @ self._chol_amt.T
        u_amt = np.clip(stats.norm.cdf(z_amt), 1e-8, 1 - 1e-8)
        log_a = mu_np + sigma_np * stats.norm.ppf(u_amt)
        amount = np.exp(log_a)

        precip = occur * amount
        return torch.FloatTensor(precip)
