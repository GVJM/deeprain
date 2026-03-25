"""ar/rollout.py — AR model rollout execution and Tier 2 metric computation."""
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from data_utils import load_data
from ar.loader import load_ar_model
import metrics as M


TIER2_METRICS = [
    ("multi_lag_acf_rmse",       "ACF RMSE",          True),
    ("transition_prob_error",    "Trans. Prob Err",   True),
    ("max_cdd_error",            "Max CDD Err",       True),
    ("annual_max_error",         "Ann. Max Err",      True),
    ("monthly_mean_error",       "Monthly Mean Err",  True),
    ("monthly_var_error",        "Monthly Var Err",   True),
    ("inter_scenario_cv",        "Scenario CV",       False),
    ("rx5day_wasserstein",       "Rx5day W1",         True),
    ("rx3day_wasserstein",       "Rx3day W1",         True),
    ("rx10day_wasserstein",      "Rx10day W1",        True),
    ("rx30day_wasserstein",      "Rx30day W1",        True),
    ("seasonal_wet_error",       "Wet Season Err",    True),
    ("seasonal_dry_error",       "Dry Season Err",    True),
]

# Subset of Tier 2 metrics used in combined Tier1+Tier2 ranking score
COMBINED_TIER2_METRICS = [
    ("multi_lag_acf_rmse",    "ACF RMSE",        True),
    ("transition_prob_error", "Trans. Prob Err",  True),
    ("max_cdd_error",         "Max CDD Err",      True),
    ("rx5day_wasserstein",    "Rx5day W1",        True),
    ("seasonal_wet_error",    "Wet Season Err",   True),
    ("seasonal_dry_error",    "Dry Season Err",   True),
]


def recompute_tier1_metrics(
    variants: list,
    output_dir: str,
    data_path: str,
    n_samples: int,
    device: torch.device,
) -> None:
    """Re-evaluate Tier 1 metrics with more samples; overwrites metrics.json in-place.

    Reconstructs the exact train/test split from evaluation_protocol in metrics.json
    so normalization params (mu, std) match those used during training.
    """
    from metrics import evaluate_model as _evaluate_model

    print(f"[compare_ar] Recomputing Tier 1 metrics ({n_samples} samples) "
          f"for {len(variants)} variants...")

    # Load raw data only — discard load_data()'s mu/std (computed from full dataset).
    # We recompute mu/std from train_raw to match training behaviour.
    _, data_raw, _, _, station_names = load_data(
        data_path=data_path,
        normalization_mode="scale_only",
        missing_strategy="impute_station_median",
    )
    input_size = data_raw.shape[1]

    for variant in variants:
        metrics_path = os.path.join(output_dir, variant, "metrics.json")
        config_path  = os.path.join(output_dir, variant, "config.json")
        ckpt_path    = os.path.join(output_dir, variant, "model.pt")
        if not os.path.exists(ckpt_path):
            print(f"  [{variant}] No checkpoint — skipping")
            continue

        # Read existing metrics to preserve training metadata and get split sizes
        old_metrics = {}
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                old_metrics = json.load(f)

        proto      = old_metrics.get("evaluation_protocol") or {}
        train_size = proto.get("train_size")
        test_size  = proto.get("test_size")

        if train_size and test_size:
            train_raw = data_raw[:train_size]
            eval_raw  = data_raw[-test_size:]
        else:
            print(f"  [{variant}] No evaluation_protocol in metrics.json — using full data")
            train_raw = data_raw
            eval_raw  = data_raw

        # Recompute mu/std from train split — mirrors compute_norm_params() in train.py
        norm_mode = "scale_only"
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            norm_mode = cfg.get("normalization_mode", "scale_only")
        std = np.clip(np.std(train_raw, axis=0, keepdims=True), 1e-8, None)
        mu  = (np.zeros_like(std) if norm_mode == "scale_only"
               else np.mean(train_raw, axis=0, keepdims=True))

        try:
            model = load_ar_model(variant, output_dir, input_size=input_size, device=device)
        except Exception as e:
            print(f"  [{variant}] Load error: {e} — skipping")
            continue

        print(f"  [{variant}] Evaluating ({n_samples} samples, "
              f"train={train_raw.shape[0]} test={eval_raw.shape[0]})...")
        try:
            new_metrics = _evaluate_model(
                model, eval_raw, mu, std,
                n_samples=n_samples,
                station_names=station_names,
                timing_n_samples=100,
                timing_n_trials=1,
            )
        except Exception as e:
            print(f"  [{variant}] Evaluation error: {e} — skipping")
            continue

        # Restore training-only keys that evaluate_model does not produce
        for key in ("final_epoch", "final_train_loss", "best_train_loss",
                    "best_val_loss", "training_ms_per_epoch", "evaluation_protocol"):
            if key in old_metrics:
                new_metrics[key] = old_metrics[key]

        with open(metrics_path, "w") as f:
            json.dump(new_metrics, f, indent=2)
        print(f"  [{variant}] metrics.json updated.")


def run_rollout(
    variant: str,
    output_dir: str,
    data_norm: np.ndarray,
    std: np.ndarray,
    n_days: int,
    n_scenarios: int,
    device: torch.device,
    force: bool = False,
) -> np.ndarray:
    """
    Generate scenarios for one model. Returns (n_sc, n_days, S) in mm/dia.
    Caches to outputs/<variant>/scenarios/scenarios.npy.
    """
    cache_path = Path(output_dir) / variant / "scenarios" / "scenarios.npy"

    if not force and cache_path.exists():
        cached = np.load(str(cache_path))
        if cached.shape[0] >= n_scenarios and cached.shape[1] >= n_days:
            print(f"  [rollout] {variant}: using cached {cached.shape}")
            return cached[:n_scenarios, :n_days, :]
        print(f"  [rollout] {variant}: cache shape {cached.shape} too small, re-running")

    try:
        model = load_ar_model(variant, output_dir, input_size=data_norm.shape[1], device=device)
    except Exception as e:
        print(f"  [rollout] {variant}: load failed — {e}")
        return None

    W = getattr(model, "window_size", 7)
    seed_window = torch.FloatTensor(data_norm[-W:]).to(device)  # (W, S)

    print(f"  [rollout] {variant}: generating {n_scenarios} x {n_days} days ...")
    try:
        with torch.no_grad():
            sc_norm = model.sample_rollout(
                seed_window=seed_window,
                n_days=n_days,
                n_scenarios=n_scenarios,
            )  # (n_sc, n_days, S) normalized
    except Exception as e:
        print(f"  [rollout] {variant}: rollout failed — {e}")
        return None

    sc_np = sc_norm.cpu().numpy() if isinstance(sc_norm, torch.Tensor) else sc_norm
    sc_mm = np.clip(sc_np * std[0], 0, None)  # denormalize, clip to 0

    # Cache
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(cache_path), sc_mm)
    print(f"  [rollout] {variant}: saved {sc_mm.shape} -> {cache_path}")

    # Free GPU memory
    del model
    torch.cuda.empty_cache() if device.type == "cuda" else None

    return sc_mm


def compute_tier2_metrics(
    scenarios_mm: np.ndarray,
    data_raw: np.ndarray,
    obs_months: np.ndarray,
) -> dict:
    """Compute all Tier 2 temporal metrics for one model's scenarios."""
    import time
    result = {}
    n_sc, T, S = scenarios_mm.shape
    _t0 = time.perf_counter()

    def _timed(label, fn):
        t = time.perf_counter()
        val = fn()
        print(f"    [{label}] {time.perf_counter()-t:.2f}s", flush=True)
        return val

    result["multi_lag_acf_rmse"] = _timed(
        "acf_rmse", lambda: M.multi_lag_autocorr_rmse(scenarios_mm, data_raw))

    tp = _timed("transition", lambda: M.transition_probability_error(scenarios_mm, data_raw))
    result["transition_prob_error"] = tp["mean"]
    result["transition_probs"] = {k: tp[k] for k in ("p_ww", "p_wd", "p_dw", "p_dd")}

    result["max_cdd_error"]      = _timed("max_cdd",   lambda: M.max_consecutive_dry_days_error(scenarios_mm, data_raw))
    result["annual_max_error"]   = _timed("annual_max", lambda: M.annual_max_daily_error(scenarios_mm, data_raw))
    result["monthly_mean_error"] = _timed("monthly_mean", lambda: M.monthly_mean_error(scenarios_mm, data_raw, obs_months))
    result["monthly_var_error"]  = _timed("monthly_var",  lambda: M.monthly_variance_error(scenarios_mm, data_raw, obs_months))
    result["inter_scenario_cv"]  = _timed("inter_cv",  lambda: M.inter_scenario_cv(scenarios_mm))
    result["rx5day_wasserstein"]  = _timed("rx5",  lambda: M.rx5day_distribution_error(scenarios_mm, data_raw, window=5))
    result["rx3day_wasserstein"]  = _timed("rx3",  lambda: M.rx5day_distribution_error(scenarios_mm, data_raw, window=3))
    result["rx10day_wasserstein"] = _timed("rx10", lambda: M.rx5day_distribution_error(scenarios_mm, data_raw, window=10))
    result["rx30day_wasserstein"] = _timed("rx30", lambda: M.rx5day_distribution_error(scenarios_mm, data_raw, window=30))

    seasonal = _timed("seasonal", lambda: M.seasonal_accumulation_error(scenarios_mm, data_raw, obs_months))
    result["seasonal_wet_error"] = seasonal["wet_season_error"]
    result["seasonal_dry_error"] = seasonal["dry_season_error"]

    print(f"    [tier2_total] {time.perf_counter()-_t0:.2f}s  (n_sc={n_sc}, T={T}, S={S})", flush=True)
    return result
