"""
evaluate.py — Standalone model evaluator.

Loads a trained checkpoint from a model directory and writes metrics.json.
Used by distributed workers to decouple evaluation from GPU training time.

Usage (from PrecipModels/):
    python evaluate.py --model_dir outputs_sabesp/ar_vae
    python evaluate.py --model_dir outputs/vae --n_samples 1000

IMPORTANT: Float arch params (occ_weight, jvp_eps, mf_ratio) must be loaded
as float, not int — see CLAUDE.md "compare_ar.py rollout loader" and commit b64f0a1.
Any new float arch params added to AR models must be added to _FLOAT_PARAMS below.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path

import torch

from data_utils import load_data, load_data_with_cond
from metrics import evaluate_model
from models import get_model
from train import (
    _TEMPORAL_MODELS,
    _MC_MODELS,
    temporal_holdout_split,
    temporal_holdout_split_with_cond,
    compute_norm_params,
    normalize_with_params,
)

# Float params: must be cast to float(), not int(), when loading from config.json
_FLOAT_PARAMS = ("occ_weight", "jvp_eps", "mf_ratio")

# Int architectural params
_INT_PARAMS = (
    "latent_size", "hidden_size", "n_layers", "n_coupling",
    "gru_hidden", "window_size", "t_embed_dim", "n_sample_steps",
    "rnn_hidden", "n_steps", "latent_occ", "latent_amt",
    "hidden_occ", "hidden_amt", "context_dim", "hidden_dim",
)


def evaluate_from_dir(model_dir: Path, n_samples_override: int = None) -> dict:
    """Load a trained model from model_dir and write metrics.json."""
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())

    model_name = cfg["model"]
    data_path = cfg["data_path"]
    holdout_ratio = cfg.get("holdout_ratio", 0.0)
    norm_mode = cfg.get("normalization_mode", "scale_only")
    is_mc = model_name in _MC_MODELS
    is_temporal = model_name in _TEMPORAL_MODELS

    # Load data — must use same split as training to get correct mu/std
    if is_mc or is_temporal:
        _, data_raw_full, _, _, station_names, cond_arrays_full = load_data_with_cond(
            data_path=data_path, normalization_mode="scale_only"
        )
        train_raw, eval_raw, _, _ = temporal_holdout_split_with_cond(
            data_raw_full, cond_arrays_full, holdout_ratio
        )
    else:
        _, data_raw_full, _, _, station_names = load_data(
            data_path=data_path, normalization_mode="scale_only"
        )
        train_raw, eval_raw = temporal_holdout_split(data_raw_full, holdout_ratio)

    mu, std = compute_norm_params(train_raw, norm_mode)
    input_size = data_raw_full.shape[1]

    # Reconstruct model — mirrors compare_ar.py load_ar_model() logic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {"input_size": input_size}
    for key in _INT_PARAMS:
        val = cfg.get(key)
        if val is not None:
            model_kwargs[key] = int(val)
    rnn_type = cfg.get("rnn_type")
    if rnn_type:
        model_kwargs["rnn_type"] = rnn_type
    for key in _FLOAT_PARAMS:
        val = cfg.get(key)
        if val is not None:
            model_kwargs[key] = float(val)   # MUST be float, not int

    model = get_model(model_name, **model_kwargs)
    ckpt_path = model_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"model.pt not found in {model_dir}. "
            f"Copula models do not support deferred evaluation."
        )
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Evaluate — same caps as train.py for AR models
    n_samples = n_samples_override or cfg.get("n_samples", 5000)
    if is_temporal:
        eval_n_samples = min(n_samples, 200)
        timing_n_samples = 100
        timing_n_trials = 1
        print(f"[evaluate] AR model — using {eval_n_samples} samples, 1 timing trial")
    else:
        eval_n_samples = n_samples
        timing_n_samples = 1000
        timing_n_trials = 5

    print(f"[evaluate] Evaluating {model_name} from {model_dir} ...")
    metrics = evaluate_model(
        model, eval_raw, mu, std,
        n_samples=eval_n_samples,
        station_names=station_names,
        timing_n_samples=timing_n_samples,
        timing_n_trials=timing_n_trials,
    )

    out_path = model_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[evaluate] Metrics saved to {out_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model and write metrics.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_dir", required=True,
                        help="Path to trained model directory (contains config.json + model.pt)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Override n_samples from config.json")
    args = parser.parse_args()
    evaluate_from_dir(Path(args.model_dir), args.n_samples)


if __name__ == "__main__":
    main()
