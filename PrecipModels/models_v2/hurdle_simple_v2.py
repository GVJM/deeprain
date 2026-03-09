"""
hurdle_simple_v2.py — HurdleSimple com otimizadores separados por componente.

Subclasse do HurdleSimple original. Adiciona apenas get_optimizer_groups()
para ativar o protocolo multi-otimizador em train_v2.py.

Motivação: BCE ∈ [0, 0.693] vs NLL Log-Normal ∈ [2, 10]. Com um único
otimizador, o sinal de gradiente da rede de ocorrência (occ_mlp) é
dominado pela magnitude da NLL. Optimizadores separados eliminam esse
desbalanceamento.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.hurdle_simple import HurdleSimple


class HurdleSimpleV2(HurdleSimple):
    """
    HurdleSimple com otimizadores separados: um para occ_mlp (BCE),
    outro para amt_mlp (NLL Log-Normal).
    """

    def get_optimizer_groups(self) -> list:
        """
        Retorna grupos de otimização para o protocolo multi-otimizador.

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
        ]
