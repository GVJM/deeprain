"""
hurdle_temporal_v2.py — HurdleTemporal com otimizadores separados por componente.

Subclasse do HurdleTemporal original. Adiciona apenas get_optimizer_groups()
para ativar o protocolo multi-otimizador em train_v2.py.

Motivação: BCE ∈ [0, 0.693] vs NLL Log-Normal ∈ [2, 10], com o agravante
de que context_encoder (GRU) é compartilhado por ambas as cabeças.

Estratégia para o context_encoder:
    - occ_mlp e amt_mlp recebem otimizadores independentes.
    - context_encoder usa combined=True: seu otimizador é stepped SOMENTE após
      os dois backward passes (bce e nll), acumulando gradientes de ambas as
      cabeças antes de atualizar. lr_scale=0.5 compensa a dupla acumulação.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.hurdle_temporal import HurdleTemporal


class HurdleTemporalV2(HurdleTemporal):
    """
    HurdleTemporal com otimizadores separados: occ_mlp (BCE), amt_mlp (NLL),
    e context_encoder (GRU compartilhado, stepped após acumulação).
    """

    def get_optimizer_groups(self) -> list:
        """
        Retorna grupos de otimização para o protocolo multi-otimizador.

        O context_encoder (GRU) tem combined=True: não é stepped após cada
        backward individual; seus gradientes acumulam de ambos os passes e o
        otimizador é stepped uma única vez ao final do batch.

        Returns:
            Lista de dicts com campos: name, params, loss_key, loss_fn, lr_scale, combined.
        """
        return [
            {
                'name':     'occ',
                'params':   list(self.occ_mlp.parameters()),
                'loss_key': 'bce',
                'loss_fn':  None,
                'lr_scale': 1.0,
                'combined': False,
            },
            {
                'name':     'amt',
                'params':   list(self.amt_mlp.parameters()),
                'loss_key': 'nll',
                'loss_fn':  None,
                'lr_scale': 1.0,
                'combined': False,
            },
            {
                'name':     'ctx',
                'params':   list(self.context_encoder.parameters()),
                'loss_key': None,
                'loss_fn':  None,
                'lr_scale': 0.5,
                'combined': True,
            },
        ]
