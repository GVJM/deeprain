"""scoring.py — Composite score utilities shared by compare.py and compare_ar.py."""
import numpy as np


# Quality metrics for composite score (key, label, lower_is_better)
QUALITY_METRICS = [
    ("mean_wasserstein",        "Wasserstein",       True),
    ("corr_rmse",               "Corr RMSE",         True),
    ("wet_day_freq_error_mean", "Wet Freq Err",      True),
    ("extreme_q90_mean",        "Q90 Err",           True),
    ("extreme_q95_mean",        "Q95 Err",           True),
    ("extreme_q99_mean",        "Q99 Err",           True),
    ("energy_score",            "Energy Score",      True),
    ("wet_spell_length_error",  "Wet Spell Err",     True),
    ("dry_spell_length_error",  "Dry Spell Err",     True),
    ("lag1_autocorr_error",     "Lag1 ACF Err",      True),
]


def _metric_array(all_metrics: dict, variants: list, key: str) -> np.ndarray:
    vals = []
    for v in variants:
        val = all_metrics.get(v, {}).get(key, np.nan)
        try:
            vals.append(float(val))
        except (TypeError, ValueError):
            vals.append(float("nan"))
    return np.array(vals, dtype=float)


def compute_composite(all_metrics: dict, metric_defs=None) -> tuple:
    """
    Composite score via min-max normalization. Returns (scores, normalized).
    0 = best, 1 = worst.
    """
    if metric_defs is None:
        metric_defs = QUALITY_METRICS
    variants = list(all_metrics.keys())
    normalized = {v: {} for v in variants}

    for key, _, lower_is_better in metric_defs:
        vals = _metric_array(all_metrics, variants, key)
        valid = ~np.isnan(vals)
        if valid.sum() < 2:
            continue
        mn, mx = vals[valid].min(), vals[valid].max()
        rng = mx - mn + 1e-12
        norm = (vals - mn) / rng
        if not lower_is_better:
            norm = 1.0 - norm
        for i, v in enumerate(variants):
            normalized[v][key] = float(norm[i]) if valid[i] else float("nan")

    scores = {}
    for v in variants:
        vals = [x for x in normalized[v].values() if not np.isnan(x)]
        scores[v] = float(np.mean(vals)) if vals else float("nan")

    return scores, normalized
