"""
datasets.py — Datasets temporais para modelos autorregressivos.

O TemporalDataset retorna pares (window, target) sequenciais para treino
de modelos que modelam P(y_t | y_{t-k:t-1}).

Uso:
    from datasets import TemporalDataset
    dataset = TemporalDataset(train_norm, window_size=30)
    loader  = DataLoader(dataset, batch_size=128, shuffle=True)
    for window, target in loader:
        # window: (B, W, S), target: (B, S)
        loss = model.loss((window, target), beta=beta)
"""

import numpy as np
import torch
from torch.utils.data import Dataset


class TemporalDataset(Dataset):
    """
    Dataset que retorna pares (window, target) ordenados temporalmente.

    Cada par (window_i, target_i) representa:
        window_i : dados dos dias [i, i+W)  — contexto histórico
        target_i : dados do dia i+W         — dia a prever

    Embaralhar no DataLoader é seguro: cada par é uma instância de treino
    independente para o objetivo L(y_t | window_t).

    Args:
        data_norm:   np.ndarray (N, S) — dados normalizados, já limpos de NaN.
        window_size: int W             — número de dias na janela histórica.
    """

    def __init__(self, data_norm: np.ndarray, window_size: int):
        self.data = torch.FloatTensor(data_norm)
        self.W = window_size

    def __len__(self) -> int:
        return len(self.data) - self.W

    def __getitem__(self, i: int):
        """
        Returns:
            window: FloatTensor (W, S) — contexto histórico
            target: FloatTensor (S,)   — dia alvo
        """
        return self.data[i : i + self.W], self.data[i + self.W]
