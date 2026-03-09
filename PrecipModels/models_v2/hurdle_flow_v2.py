"""
hurdle_flow_v2.py — HurdleFlow com otimizadores separados por componente.

Subclasse do HurdleFlow original. Adiciona apenas get_optimizer_groups()
para ativar o protocolo multi-otimizador em train_v2.py.

Motivação: este é o caso mais crítico — Flow NLL ∈ [20, 50] vs BCE ∈ [0, 0.693].
Com um único otimizador, o gradiente da rede de ocorrência (occ_mlp) é
completamente eclipsado pelo gradiente da flow (~100× maior). Otimizadores
separados permitem que cada componente receba sinal de gradiente adequado.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.hurdle_flow import HurdleFlow


class HurdleFlowV2(HurdleFlow):
    """
    HurdleFlow com otimizadores separados: um para occ_mlp (BCE),
    outro para layers + log_scale (Flow NLL).
    """

    def get_optimizer_groups(self) -> list:
        """
        Retorna grupos de otimização para o protocolo multi-otimizador.

        Returns:
            Lista de dicts com campos: name, params, loss_key, loss_fn, lr_scale, combined.
        """
        flow_params = list(self.layers.parameters()) + [self.log_scale]
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
                'name':     'flow',
                'params':   flow_params,
                'loss_key': 'nll',
                'loss_fn':  None,
                'lr_scale': 1.0,
                'combined': False,
            },
        ]
