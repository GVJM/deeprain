"""
base_model.py — Interface abstrata comum para todos os modelos generativos.

Todos os modelos herdam de BaseModel e implementam:
  - sample(n)  → Tensor (n, n_stations)  [geração]
  - loss(x, beta) → dict com chave 'total'  [treino]
  - count_parameters() → int

O train.py é completamente agnóstico ao modelo:
    loss_dict = model.loss(batch, beta=beta)
    loss_dict['total'].backward()
"""

from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch import Tensor


class BaseModel(ABC, nn.Module):
    """
    Classe base para todos os modelos de precipitação.

    Subclasses devem implementar sample() e loss().
    Modelos puramente estatísticos (Cópula) retornam loss com total=0 e
    override fit() para ajuste analítico.
    """

    def __init__(self):
        super().__init__()

    @abstractmethod
    def sample(self, n: int, steps: int | None = None, method: str | None = None) -> Tensor:
        """
        Gera n amostras sintéticas.

        Args:
            n: número de amostras
            steps: número de passos de integração/denoising (None = padrão do modelo)
            method: solver/método de amostragem (None = padrão do modelo)
                    Valores válidos dependem do modelo:
                      flow_match/flow_match_film: 'euler', 'heun'
                      latent_flow: 'midpoint', 'euler'
                      ldm: 'ddpm', 'ddim'

        Returns:
            Tensor de shape (n, n_stations) no espaço original (mm/dia)
            ou no espaço normalizado, dependendo do modelo.
        """
        ...

    @abstractmethod
    def loss(self, x: Tensor, beta: float = 1.0) -> dict:
        """
        Calcula a loss para um batch x.

        Args:
            x: Tensor (batch_size, n_stations) — dados normalizados
            beta: peso do termo de regularização (KL para VAEs)

        Returns:
            dict com ao menos a chave 'total' (scalar Tensor).
            Outras chaves opcionais para logging: 'recons', 'kl', 'bce', etc.
        """
        ...

    def count_parameters(self) -> int:
        """Conta parâmetros treináveis. Retorna 0 para modelos puramente estatísticos."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def fit(self, data: "np.ndarray", **kwargs):  # noqa: F821
        """
        Ajuste analítico/estatístico (sobrescrever em modelos não-neurais).
        Modelos neurais são treinados via train.py com gradiente.
        """
        pass
