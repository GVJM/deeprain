"""
conditioning.py — Bloco genérico de condicionamento por embeddings categóricos.

Projetado para extensibilidade: adicionar um novo condicionador (ENSO, estação do ano,
elevação, etc.) requer apenas atualizar DEFAULT_CATEGORICALS e o dict de dados.
Os modelos condicionados (_mc) não precisam ser modificados.

Uso típico:
    block = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
    emb = block({"month": month_tensor})    # (B, total_dim)

    # Para adicionar ENSO no futuro:
    DEFAULT_CATEGORICALS = [("month", 12, 6), ("enso", 3, 4)]
    # data_utils retorna {"month": ..., "enso": ...}
    # ConditioningBlock auto-redimensiona sem modificar os modelos
"""

import numpy as np
import torch
import torch.nn as nn
from torch import Tensor


# Configuração padrão dos condicionadores categóricos.
# Formato: lista de (nome, n_classes, embed_dim)
# Para adicionar ENSO: [("month", 12, 6), ("enso", 3, 4)]
DEFAULT_CATEGORICALS = [("month", 12, 6)]

DEFAULT_CONTINUOUS = [("day_sin", 1), ("day_cos", 1)]

class ConditioningBlock(nn.Module):
    """
    Concatena embeddings de condicionadores categóricos e features contínuas.

    Args:
        categoricals: lista de (nome, n_classes, embed_dim)
            Ex: [("month", 12, 6)]
        continuous: lista de (nome, input_dim)
            Ex: [("day_sin", 1), ("day_cos", 1)]

    Uso:
        block = ConditioningBlock(DEFAULT_CATEGORICALS, DEFAULT_CONTINUOUS)
        emb = block({
            "month": month_tensor,           # (B,) LongTensor
            "day_sin": day_sin_tensor,       # (B, 1) FloatTensor
            "day_cos": day_cos_tensor        # (B, 1) FloatTensor
        })    # → (B, total_dim)
        # total_dim = 6 (month embed) + 1 (day_sin) + 1 (day_cos) = 8
    """

    def __init__(
            self, 
            categoricals: list[tuple[str, int, int]] = None,
            continuous: list[tuple[str,int]] = None,
    ):
        super().__init__()
        self.categoricals = categoricals or []
        self.continuous = continuous or []

        self.embeddings = nn.ModuleDict({
            name: nn.Embedding(n_classes, embed_dim)
            for name, n_classes, embed_dim in categoricals
        })

        cat_dim = sum(ed for _, _, ed in self.categoricals)
        cont_dim = sum(idim for _, idim in self.continuous)

        self.total_dim = cat_dim + cont_dim

    def forward(self, cond: dict) -> Tensor:
        """
        cond: dict[str, LongTensor (B,)] → (B, total_dim)
        """
        parts = []

        if not cond:
            raise ValueError("cond não pode ser vazio")

        ref_tensor = next(iter(cond.values()))
        batch_size = ref_tensor.shape[0]
        ref_device = ref_tensor.device
        
        for name, _, _ in self.categoricals:
            if name not in cond:
                raise KeyError(
                    f"Condição categórica ausente: '{name}'. "
                    f"Disponíveis: {list(cond.keys())}"
                )
            parts.append(self.embeddings[name](cond[name]))

        for name, input_dim in self.continuous:
            feat = cond.get(name)

            # Fallback defensivo: em inferência/eval, pode faltar contínua
            # (ex.: checkpoints antigos sem _continuous_data).
            if feat is None:
                feat = torch.zeros(batch_size, input_dim, device=ref_device)

            if feat.dim() == 1:
                feat = feat.unsqueeze(1)
            
            parts.append(feat)

        return torch.cat(parts, dim=1)

    def sample_cond(self, n: int, probs: dict[str, np.ndarray], continuous_data: dict[str, np.ndarray] = None) -> dict:
        """
        Sorteia n condições.

        Args:
            n: número de amostras
            probs: dict[str, ndarray] — probabilidades empíricas para categóricas
            continuous_data: dict[str, ndarray] — dados contínuos para shuffle (ex: {"day_sin": array, "day_cos": array})

        Returns:
            dict com tensores prontos para forward()
        """
        cond = {
            name: torch.LongTensor(
                np.random.choice(n_classes, size=n, p=probs[name])
            )
            for name, n_classes, _ in self.categoricals
        }

        for name, _ in self.continuous:
            values = None
            if continuous_data is not None and name in continuous_data:
                values = continuous_data[name]

            if values is not None and len(values) > 0:
                cond[name] = torch.FloatTensor(np.random.choice(values, size=n))
            else:
                cond[name] = torch.zeros(n, dtype=torch.float32)

        return cond
