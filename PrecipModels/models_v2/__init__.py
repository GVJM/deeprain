"""
models_v2/ — Variantes com otimizadores separados por componente.

Cada modelo implementa get_optimizer_groups() para ativar o protocolo
multi-otimizador em train_v2.py. Sem alterações nos modelos originais.
"""

from models_v2.hurdle_simple_v2 import HurdleSimpleV2
from models_v2.hurdle_flow_v2 import HurdleFlowV2
from models_v2.hurdle_temporal_v2 import HurdleTemporalV2
from models_v2.hurdle_vae_v2 import HurdleVAEV2

V2_MODEL_REGISTRY = {
    "hurdle_simple_v2":   HurdleSimpleV2,
    "hurdle_flow_v2":     HurdleFlowV2,
    "hurdle_temporal_v2": HurdleTemporalV2,
    "hurdle_vae_v2":      HurdleVAEV2,
}
