"""ar/loader.py — AR model discovery and checkpoint loading."""
import json
import os
from pathlib import Path
import torch
import sys
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


def _resolve_model_dir(variant: str, output_dir, variant_dirs=None) -> Path:
    """Return the directory for *variant*.

    Looks up *variant_dirs* first (populated by recursive discovery); falls
    back to the flat ``Path(output_dir) / variant`` for backward compat.
    """
    if variant_dirs is not None and variant in variant_dirs:
        return variant_dirs[variant]
    return Path(output_dir) / variant


# Directory names that are never AR model dirs and should not be recursed into.
_SKIP_DIR_NAMES = frozenset({
    "comparison_ar", "hyperparameter_sensitivity", "stations",
    "families", "scenarios", "__pycache__",
})


def _discover_recursive(base: Path) -> "dict[str, Path]":
    """Walk *base* recursively and return ``{variant_name: absolute_dir}``
    for every AR model directory found at any depth.

    An AR model directory is any directory that passes ``_is_ar_dir()``.
    Once found, its own subdirectories are NOT descended into, so
    ``scenarios/``, family-detail dirs, etc. are never mistaken for models.

    Deduplication: if the same folder name appears more than once, the
    lexicographically smaller absolute path wins and a warning is printed.
    """
    result: "dict[str, Path]" = {}

    for root, dirs, _files in os.walk(str(base)):
        root_path = Path(root)

        # Determine which subdirs to recurse into vs. claim as AR dirs
        keep_recursing = []
        for d_name in sorted(dirs):
            if d_name.startswith(".") or d_name in _SKIP_DIR_NAMES:
                continue
            d_path = root_path / d_name
            if _is_ar_dir(d_path):
                # Found an AR model dir — record it, do not recurse inside
                if d_name in result:
                    existing = result[d_name]
                    keep = min(existing, d_path, key=lambda p: str(p))
                    dropped = d_path if keep == existing else existing
                    print(f"[discover] duplicate variant '{d_name}': "
                          f"keeping {keep}  (dropping {dropped})")
                    result[d_name] = keep
                else:
                    result[d_name] = d_path
            else:
                keep_recursing.append(d_name)

        # Mutate dirs in-place so os.walk only descends into non-AR dirs
        dirs[:] = keep_recursing

    return result


def discover_ar_models(
    output_dir: str,
    selected_families: list = None,
) -> "dict[str, Path]":
    """Recursively scan *output_dir* for AR models that have ``metrics.json``.

    Returns ``{variant_name: absolute_path}`` (sorted by name).
    Applies *selected_families* filter when provided.
    """
    all_dirs = _discover_recursive(Path(output_dir))
    result: "dict[str, Path]" = {}
    for name, path in sorted(all_dirs.items()):
        if not (path / "metrics.json").exists():
            continue
        if selected_families:
            fam = get_family(name, variant_dirs={name: path})
            if not any(fam.startswith(f) or name.startswith(f)
                       for f in selected_families):
                continue
        result[name] = path
    return result


def discover_ar_models_with_checkpoints(output_dir: str) -> "dict[str, Path]":
    """Subset of AR models (recursive) that also have ``model.pt``."""
    all_dirs = _discover_recursive(Path(output_dir))
    return {
        name: path
        for name, path in sorted(all_dirs.items())
        if (path / "metrics.json").exists() and (path / "model.pt").exists()
    }


def get_family(variant_name: str, output_dir: str = None,
               variant_dirs: "dict[str, Path]" = None) -> str:
    """Read config.json 'model' field; fall back to longest prefix match."""
    cfg_path = _resolve_model_dir(variant_name, output_dir, variant_dirs) / "config.json"
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


def load_all_metrics(variants, output_dir: str = None,
                     variant_dirs: "dict[str, Path]" = None) -> dict:
    """Load metrics.json for each variant.

    *variants* may be a ``list[str]`` or a ``dict[str, Path]`` (as returned
    by ``discover_ar_models``).  In the dict case, *output_dir* is ignored.
    """
    if isinstance(variants, dict):
        variant_dirs = variants
        variants = list(variants.keys())
    all_m = {}
    for v in variants:
        path = _resolve_model_dir(v, output_dir, variant_dirs) / "metrics.json"
        if path.exists():
            with open(path) as f:
                all_m[v] = json.load(f)
        else:
            print(f"[warn] metrics.json missing: {path}")
    return all_m


def _load_config(variant: str, output_dir: str = None,
                 variant_dirs: "dict[str, Path]" = None) -> dict:
    path = _resolve_model_dir(variant, output_dir, variant_dirs) / "config.json"
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def load_ar_model(variant: str, output_dir: str = None, input_size: int = 0,
                  device: torch.device = None,
                  variant_dirs: "dict[str, Path]" = None):
    """
    Load a trained AR model checkpoint.
    Handles all AR families including LSTM variants.
    """
    model_dir = _resolve_model_dir(variant, output_dir, variant_dirs)
    cfg = _load_config(variant, output_dir, variant_dirs)
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
