"""
models/ — Os 9 modelos generativos para comparação sistemática.

Importação rápida:
    from models import get_model

    model = get_model("vae", input_size=15)
    model = get_model("ldm", input_size=15, latent_size=128)
    model = get_model("hurdle_temporal", input_size=15)
    model = get_model("latent_flow", input_size=15)
"""

from models.copula import GaussianCopula
from models.vae import VAEModel
from models.hurdle_simple import HurdleSimple
from models.hurdle_vae import HurdleVAE
from models.hurdle_vae_cond import HurdleVAECond
from models.hurdle_vae_cond_mc import HurdleVAECondMc
from models.hurdle_vae_cond_nll import HurdleVAECondNll
from models.real_nvp import RealNVP
from models.hurdle_flow import HurdleFlow
from models.flow_match import FlowMatchingModel
from models.flow_match_film import FlowMatchingModelFilm
from models.ldm import LDMModel
from models.hurdle_temporal import HurdleTemporal
from models.latent_flow import LatentFlowWrapper
from models.hurdle_simple_mc import HurdleSimpleMC
from models.vae_mc import VAEModelMC
from models.glow import GLOW
from models.real_nvp_mc import RealNVPMC
from models.glow_mc import GlowMC
from models.flow_match_mc import FlowMatchingMC
from models.flow_match_film_mc import FlowMatchingFilmMC
from models.latent_fm_mc import LatentFlowMc
from models.hurdle_latent_fm_mc import HurdleLatentFlowMc
from models.thresholded_latent_flow_mc import ThresholdedLatentFlowMc
from models.thresholded_vae_mc import ThresholdedVAEMc
from models.thresholded_real_nvp_mc import ThresholdedRealNVPMc
from models.thresholded_glow_mc import ThresholdedGlowMc
from models.ar_vae import ARVAE
from models.ar_flow_match import ARFlowMatch
from models.ar_latent_fm import ARLatentFM
from models.ar_real_nvp import ARRealNVP
from models.ar_glow import ARGlow
from models.ar_mean_flow import ARMeanFlow
from models.ar_flow_map import ARFlowMap

_MODEL_REGISTRY = {
    "copula": GaussianCopula,
    "vae": VAEModel,
    "hurdle_simple": HurdleSimple,
    "hurdle_vae": HurdleVAE,
    "hurdle_vae_cond": HurdleVAECond,
    "hurdle_vae_cond_mc": HurdleVAECondMc,
    "hurdle_vae_cond_nll": HurdleVAECondNll,
    "real_nvp": RealNVP,
    "hurdle_flow": HurdleFlow,
    "flow_match": FlowMatchingModel,
    "flow_match_film": FlowMatchingModelFilm,
    # Novos modelos (Plano de expansão)
    "ldm": LDMModel,
    # "hurdle_temporal": HurdleTemporal,
    "latent_flow": LatentFlowWrapper,
    # Modelos condicionados (_mc = monthly conditioning via nn.Embedding)
    "hurdle_simple_mc": HurdleSimpleMC,
    "vae_mc": VAEModelMC,
    "glow": GLOW,
    "real_nvp_mc": RealNVPMC,
    "glow_mc": GlowMC,
    "flow_match_mc": FlowMatchingMC,
    "flow_match_film_mc": FlowMatchingFilmMC,
    "latent_fm_mc":        LatentFlowMc,
    "hurdle_latent_fm_mc": HurdleLatentFlowMc,
    "thresholded_latent_fm_mc": ThresholdedLatentFlowMc,
    "thresholded_vae_mc":       ThresholdedVAEMc,
    "thresholded_real_nvp_mc":  ThresholdedRealNVPMc,
    "thresholded_glow_mc":      ThresholdedGlowMc,
    "ar_vae":            ARVAE,
    "ar_vae_v2":         ARVAE,
    "ar_flow_match":     ARFlowMatch,
    "ar_latent_fm":      ARLatentFM,
    "ar_real_nvp":       ARRealNVP,
    "ar_real_nvp_lstm":  ARRealNVP,
    "ar_glow":           ARGlow,
    "ar_glow_lstm":      ARGlow,
    "ar_mean_flow":      ARMeanFlow,
    "ar_mean_flow_lstm": ARMeanFlow,
    "ar_mean_flow_v2":   ARMeanFlow,
    "ar_flow_map":       ARFlowMap,
    "ar_flow_map_lstm":  ARFlowMap,
    "ar_flow_map_ms":    ARFlowMap,
    "ar_flow_map_sd":    ARFlowMap,
    "ar_mean_flow_ayfm": ARMeanFlow,
}

MODEL_NAMES = list(_MODEL_REGISTRY.keys())


def get_model(name: str, **kwargs):
    """
    Instancia um modelo pelo nome.

    Args:
        name: um de MODEL_NAMES
        **kwargs: passados ao construtor do modelo (e.g. input_size=15, latent_size=128)

    Returns:
        Instância do modelo (BaseModel)
    """
    if name not in _MODEL_REGISTRY:
        for model_name in MODEL_NAMES:
            if name.startswith(model_name):
                return _MODEL_REGISTRY[model_name](**kwargs)
        raise ValueError(f"Modelo '{name}' desconhecido. Opções: {MODEL_NAMES}")
    return _MODEL_REGISTRY[name](**kwargs)
