"""ar/loader.py — AR model discovery and checkpoint loading."""
import json
from pathlib import Path
import torch
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models import get_model


AR_FAMILIES = [
    "ar_vae",
    "ar_flow_match",
    "ar_flow_map",
    "ar_flow_map_res",
    "ar_mean_flow",
    "ar_latent_fm",
    "ar_real_nvp",
    "ar_glow",
    "ar_ddpm",
]

OUTLIER_STATION = "st_362"
# Model families excluded from Tier 2 scenario charts (kept in Tier 1 tables + per-family dirs)
# OUTLIER_MODELS: frozenset = frozenset({"ar_flow_map_ms"})
OUTLIER_MODELS: frozenset = frozenset({})


def _is_ar_dir(d: Path) -> bool:
    """Return True if directory belongs to an AR model family.

    Accepts both classic 'ar_*' names and prefixed names like 'dp24_ar_*'
    by reading the 'model' field from config.json when available.
    """
    if d.name.startswith("ar_"):
        return True
    cfg_path = d / "config.json"
    if cfg_path.exists():
        try:
            with open(cfg_path) as f:
                cfg = json.load(f)
            model = cfg.get("model", "")
            return any(model == fam or model.startswith(fam) for fam in AR_FAMILIES)
        except Exception:
            pass
    return False


def discover_ar_models(output_dir: str, selected_families: list = None) -> list:
    """Scan output dirs that have metrics.json and an AR model config."""
    base = Path(output_dir)
    variants = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and _is_ar_dir(d) and (d / "metrics.json").exists():
            if selected_families:
                for fam in selected_families:
                    if d.name.startswith(fam) or get_family(d.name, output_dir).startswith(fam):
                        variants.append(d.name)
                        break
            else:
                variants.append(d.name)
    return variants


def discover_ar_models_with_checkpoints(output_dir: str) -> list:
    """Subset of AR models that also have model.pt."""
    base = Path(output_dir)
    variants = []
    for d in sorted(base.iterdir()):
        if (d.is_dir() and _is_ar_dir(d)
                and (d / "metrics.json").exists()
                and (d / "model.pt").exists()):
            variants.append(d.name)
    return variants


def get_family(variant_name: str, output_dir: str) -> str:
    """Read config.json 'model' field; fall back to longest prefix match."""
    cfg_path = Path(output_dir) / variant_name / "config.json"
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
        model_field = cfg.get("model", "")
        if model_field:
            return model_field
    # Fallback: longest matching family prefix
    for fam in sorted(AR_FAMILIES, key=len, reverse=True):
        if variant_name.startswith(fam):
            return fam
    return variant_name


def load_all_metrics(variants: list, output_dir: str) -> dict:
    all_m = {}
    for v in variants:
        path = Path(output_dir) / v / "metrics.json"
        if path.exists():
            with open(path) as f:
                all_m[v] = json.load(f)
        else:
            print(f"[warn] metrics.json missing: {path}")
    return all_m


def _load_config(variant: str, output_dir: str) -> dict:
    path = Path(output_dir) / variant / "config.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_ar_model(variant: str, output_dir: str, input_size: int, device: torch.device):
    """
    Load a trained AR model checkpoint.
    Handles all AR families including LSTM variants.
    """
    model_dir = Path(output_dir) / variant
    cfg = _load_config(variant, output_dir)
    model_class = cfg.get("model", variant)

    # Prefer best-val checkpoint (written when val_ratio > 0); fall back to final model.pt
    ckpt_path = model_dir / "model_best_val.pt"
    if not ckpt_path.exists():
        ckpt_path = model_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found in: {model_dir}")

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]

    # Infer input_size from checkpoint — input_size is not saved in config.json
    # weight_ih_l0 shape is [hidden*gates, input_size] for all RNN types (GRU/LSTM/RNN)
    for key, tensor in state.items():
        if key.endswith("weight_ih_l0"):
            input_size = tensor.shape[1]
            break

    model_kwargs = dict(input_size=input_size)

    # Common architectural params
    for key in (
        "latent_size", "hidden_size", "n_layers", "n_coupling",
        "gru_hidden", "window_size", "t_embed_dim", "n_sample_steps",
        "rnn_hidden", "n_steps", "tangent_warmup_steps",
    ):
        val = cfg.get(key)
        if val is not None:
            model_kwargs[key] = int(val)

    # rnn_type must be passed as string, not int
    rnn_type = cfg.get("rnn_type")
    if rnn_type:
        model_kwargs["rnn_type"] = rnn_type

    # Float params that condition model architecture
    for key in ("occ_weight", "jvp_eps", "mf_ratio",
                "lsd_weight", "ayf_weight", "ayf_delta_t",
                "mu_sad", "sigma_sad", "dropout"):
        val = cfg.get(key)
        if val is not None:
            model_kwargs[key] = float(val)

    # Bool params
    for key in ("improved_interval_sampling", "use_residual"):
        val = cfg.get(key)
        if val is not None:
            model_kwargs[key] = bool(val)

    model = get_model(model_class, **model_kwargs)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model
