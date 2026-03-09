"""
hurdle_vae_v2.py — HurdleVAE com otimizadores separados por VAE.

Subclasse do HurdleVAE original. Adiciona apenas get_optimizer_groups()
para ativar o protocolo multi-otimizador em train_v2.py.

Motivação: BCE ∈ [0, 0.693] vs MSE_wet ∈ [0.1, 5]. Além do desbalanceamento
de magnitude, cada sub-VAE tem seu próprio KL que precisa ser annealed em
conjunto com sua loss de reconstrução. Separar os otimizadores garante que
cada ELBO parcial seja otimizado de forma coerente.

Estratégia: loss_fn para cada grupo calcula o ELBO parcial correspondente,
incluindo beta para KL annealing.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.hurdle_vae import HurdleVAE


class HurdleVAEV2(HurdleVAE):
    """
    HurdleVAE com otimizadores separados: um para occ_vae (BCE + beta*KL_occ),
    outro para amt_vae (MSE_wet + beta*KL_amt).

    Usa loss_fn (callable) porque cada grupo precisa do beta para o ELBO parcial.
    """

    def get_optimizer_groups(self) -> list:
        """
        Retorna grupos de otimização para o protocolo multi-otimizador.

        Cada grupo usa loss_fn em vez de loss_key porque a loss é uma expressão
        composta (ELBO parcial = reconstrução + beta * KL).

        Returns:
            Lista de dicts com campos: name, params, loss_key, loss_fn, lr_scale, combined.
        """
        return [
            {
                'name':     'occ_vae',
                'params':   list(self.occ_vae.parameters()),
                'loss_key': None,
                'loss_fn':  lambda ld, beta: ld['bce'] + beta * ld['kl_occ'],
                'lr_scale': 1.0,
                'combined': False,
            },
            {
                'name':     'amt_vae',
                'params':   list(self.amt_vae.parameters()),
                'loss_key': None,
                'loss_fn':  lambda ld, beta: ld['mse_wet'] + beta * ld['kl_amt'],
                'lr_scale': 1.0,
                'combined': False,
            },
        ]
