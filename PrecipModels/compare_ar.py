"""
compare_ar.py — Comparação sistemática de todos os modelos AR treinados.

Dois níveis de análise:
  Tier 1: métricas pré-computadas (metrics.json) — todos os 42 variantes
  Tier 2: rollout de cenários — modelos com checkpoint (model.pt)

Uso:
    python compare_ar.py --skip_rollouts          # Tier 1 apenas (<1 min)
    python compare_ar.py --only_tier2             # Tier 2 apenas
    python compare_ar.py --models ar_vae ar_flow_match --n_days 30 --n_scenarios 5
    python compare_ar.py --n_days 365 --n_scenarios 50  # execução completa
    python compare_ar.py --skip_rollouts --skip_station_analysis  # exact Tier 1 (no per-station)

Saídas em outputs/comparison_ar/:
    ar_comparison_report.txt
    ar_family_summary.txt
    ar_hyperparameter_sensitivity.txt
    ar_per_station_report.txt            ← per-station text report (Tier 1)
    composite_scores_ar.json
    tier2_temporal_metrics.json
    composite_bar.png, radar_by_family.png, pareto_*.png,
    heatmap_models_x_metrics.png, family_grouped_bars.png,
    training_loss_overlay.png, hyperparameter_sensitivity/*.png,
    station_wasserstein_heatmap.png      ← models × stations heatmap, colorbar clipped at 95th pct (Tier 1)
    station_rank_wasserstein_heatmap.png ← rank-based heatmap (1=best) — no outlier scale bias (Tier 1)
    station_wetfreq_heatmap.png          ← models × stations heatmap (Tier 1)
    station_score_distribution.png       ← per-station box plots (Tier 1)
    station_wins_per_model.png           ← best-model-per-station win count (Tier 1)
    stations/st_*.png                    ← per-station model comparison charts (Tier 1)
    autocorr_multilag.png                ← family-aggregated (Tier 2)
    autocorr_multilag_detail.png         ← all individual models (Tier 2)
    spell_length_comparison.png          ← family-aggregated (Tier 2)
    spell_length_detail.png              ← all individual models (Tier 2)
    monthly_mean_precip.png              ← family-aggregated (Tier 2)
    monthly_precip_detail.png            ← all individual models (Tier 2)
    monthly_variance.png, spread_envelopes.png, transition_probabilities.png, return_period.png,
    station_autocorr_scatter.png         ← per-station lag-1 ACF scatter (Tier 2)
    station_wetfreq_scatter.png          ← per-station wet freq scatter (Tier 2)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm

import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_data, SABESP_DATA_PATH
from models import get_model
import metrics as M


# ─────────────────────────────────────────────────────────────────────────────
# Architecture family mapping
# ─────────────────────────────────────────────────────────────────────────────

AR_FAMILIES = [
    "ar_vae",
    "ar_flow_match",
    "ar_flow_map",
    "ar_mean_flow",
    "ar_latent_fm",
    "ar_real_nvp",
    "ar_glow",
]

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

TIER2_METRICS = [
    ("multi_lag_acf_rmse",       "ACF RMSE",          True),
    ("transition_prob_error",    "Trans. Prob Err",   True),
    ("max_cdd_error",            "Max CDD Err",       True),
    ("annual_max_error",         "Ann. Max Err",      True),
    ("monthly_mean_error",       "Monthly Mean Err",  True),
    ("monthly_var_error",        "Monthly Var Err",   True),
    ("inter_scenario_cv",        "Scenario CV",       False),
]


# ─────────────────────────────────────────────────────────────────────────────
# Discovery
# ─────────────────────────────────────────────────────────────────────────────

def discover_ar_models(output_dir: str) -> list[str]:
    """Scan outputs/ar_* dirs that have metrics.json."""
    base = Path(output_dir)
    variants = []
    for d in sorted(base.iterdir()):
        if d.is_dir() and d.name.startswith("ar_") and (d / "metrics.json").exists():
            variants.append(d.name)
    return variants


def discover_ar_models_with_checkpoints(output_dir: str) -> list[str]:
    """Subset of AR models that also have model.pt."""
    base = Path(output_dir)
    variants = []
    for d in sorted(base.iterdir()):
        if (d.is_dir() and d.name.startswith("ar_")
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


# ─────────────────────────────────────────────────────────────────────────────
# Metrics loading & composite score
# ─────────────────────────────────────────────────────────────────────────────

def load_all_metrics(variants: list[str], output_dir: str) -> dict:
    all_m = {}
    for v in variants:
        path = Path(output_dir) / v / "metrics.json"
        if path.exists():
            with open(path) as f:
                all_m[v] = json.load(f)
        else:
            print(f"[warn] metrics.json missing: {path}")
    return all_m


def _metric_array(all_metrics: dict, variants: list[str], key: str) -> np.ndarray:
    vals = []
    for v in variants:
        val = all_metrics.get(v, {}).get(key, np.nan)
        try:
            vals.append(float(val))
        except (TypeError, ValueError):
            vals.append(float("nan"))
    return np.array(vals, dtype=float)


def compute_composite(all_metrics: dict, metric_defs=QUALITY_METRICS) -> tuple[dict, dict]:
    """
    Composite score via min-max normalization. Returns (scores, normalized).
    0 = best, 1 = worst.
    """
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


# ─────────────────────────────────────────────────────────────────────────────
# Model loading (reuse compare.py pattern, extended for AR models)
# ─────────────────────────────────────────────────────────────────────────────

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

    model_kwargs = dict(input_size=input_size)

    # Common architectural params
    for key in (
        "latent_size", "hidden_size", "n_layers", "n_coupling",
        "gru_hidden", "window_size", "t_embed_dim", "n_sample_steps",
        "rnn_hidden", "n_steps",
    ):
        val = cfg.get(key)
        if val is not None:
            model_kwargs[key] = int(val)

    # rnn_type must be passed as string, not int
    rnn_type = cfg.get("rnn_type")
    if rnn_type:
        model_kwargs["rnn_type"] = rnn_type

    # Float params that condition model architecture
    for key in ("occ_weight", "jvp_eps", "mf_ratio"):
        val = cfg.get(key)
        if val is not None:
            model_kwargs[key] = float(val)

    model = get_model(model_class, **model_kwargs)

    ckpt_path = model_dir / "model.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"model.pt not found: {ckpt_path}")

    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2: Scenario rollout
# ─────────────────────────────────────────────────────────────────────────────

def run_rollout(
    variant: str,
    output_dir: str,
    data_norm: np.ndarray,
    std: np.ndarray,
    n_days: int,
    n_scenarios: int,
    device: torch.device,
    force: bool = False,
) -> np.ndarray | None:
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
    result = {}

    result["multi_lag_acf_rmse"] = M.multi_lag_autocorr_rmse(scenarios_mm, data_raw)

    tp = M.transition_probability_error(scenarios_mm, data_raw)
    result["transition_prob_error"] = tp["mean"]
    result["transition_probs"] = {k: tp[k] for k in ("p_ww", "p_wd", "p_dw", "p_dd")}

    result["max_cdd_error"] = M.max_consecutive_dry_days_error(scenarios_mm, data_raw)
    result["annual_max_error"] = M.annual_max_daily_error(scenarios_mm, data_raw)
    result["monthly_mean_error"] = M.monthly_mean_error(scenarios_mm, data_raw, obs_months)
    result["monthly_var_error"] = M.monthly_variance_error(scenarios_mm, data_raw, obs_months)
    result["inter_scenario_cv"] = M.inter_scenario_cv(scenarios_mm)

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tier 1 Charts
# ─────────────────────────────────────────────────────────────────────────────

def _family_colors(families: list[str]) -> dict[str, str]:
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(AR_FAMILIES))
    return {fam: matplotlib.colors.to_hex(cmap(i)) for i, fam in enumerate(AR_FAMILIES)}


def plot_composite_bar(
    scores: dict,
    families: dict[str, str],
    out_dir: str,
):
    variants = sorted(scores.keys(), key=lambda v: scores.get(v, 1.0))
    vals = [scores.get(v, float("nan")) for v in variants]
    fam_colors = _family_colors(list(set(families.values())))
    colors = [fam_colors.get(families.get(v, ""), "#95a5a6") for v in variants]

    fig, ax = plt.subplots(figsize=(max(12, len(variants) * 0.5), 6))
    bars = ax.bar(range(len(variants)), vals, color=colors)
    if vals and not np.isnan(vals[0]):
        bars[0].set_edgecolor("gold"); bars[0].set_linewidth(2)

    ax.set_xticks(range(len(variants)))
    ax.set_xticklabels(variants, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Composite Score (0=best)")
    ax.set_title("AR Models — Composite Score Ranking")
    ax.grid(axis="y", alpha=0.3)

    # Legend by family
    handles = [mpatches.Patch(color=c, label=f) for f, c in fam_colors.items()]
    ax.legend(handles=handles, fontsize=8, loc="upper left")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "composite_bar.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def plot_radar_by_family(
    all_metrics: dict,
    families: dict[str, str],
    out_dir: str,
):
    # Group variants by family
    fam_variants: dict[str, list] = {}
    for v, fam in families.items():
        fam_variants.setdefault(fam, []).append(v)

    # For each family compute mean normalized metrics
    fam_scores_dict = {}
    for fam, variants in fam_variants.items():
        fam_metrics = {v: all_metrics[v] for v in variants if v in all_metrics}
        if not fam_metrics:
            continue
        _, norm = compute_composite(fam_metrics)
        # Average normalized values per metric over variants in family
        avg = {}
        for key, _, _ in QUALITY_METRICS:
            vals = [norm[v].get(key, float("nan")) for v in variants if v in norm]
            vals = [x for x in vals if not np.isnan(x)]
            avg[key] = float(np.mean(vals)) if vals else 0.5
        fam_scores_dict[fam] = avg

    if not fam_scores_dict:
        return

    metric_keys = [k for k, _, _ in QUALITY_METRICS]
    metric_labels = [lbl for _, lbl, _ in QUALITY_METRICS]
    N = len(metric_keys)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fam_colors = _family_colors(list(fam_scores_dict.keys()))
    fig, ax = plt.subplots(figsize=(9, 9), subplot_kw=dict(polar=True))

    for fam, avg in fam_scores_dict.items():
        vals = [avg.get(k, 0.5) for k in metric_keys] + [avg.get(metric_keys[0], 0.5)]
        color = fam_colors.get(fam, "steelblue")
        ax.plot(angles, vals, "o-", lw=2, color=color, label=fam)
        ax.fill(angles, vals, alpha=0.1, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("AR Families — Radar Chart (lower = better)", fontsize=12)
    ax.legend(loc="upper right", bbox_to_anchor=(1.4, 1.1), fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "radar_by_family.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def plot_pareto(
    scores: dict,
    all_metrics: dict,
    families: dict[str, str],
    out_dir: str,
    x_key: str,
    x_label: str,
    filename: str,
):
    variants = [v for v in scores if not np.isnan(scores[v])
                and not np.isnan(float(all_metrics.get(v, {}).get(x_key, float("nan"))))]
    if len(variants) < 2:
        return

    fam_colors = _family_colors(list(set(families.values())))
    xs = np.array([float(all_metrics[v][x_key]) for v in variants])
    ys = np.array([scores[v] for v in variants])

    # Pareto front: minimize both x and y
    pareto = []
    for i, (xi, yi) in enumerate(zip(xs, ys)):
        dominated = any(xs[j] <= xi and ys[j] <= yi and (xs[j] < xi or ys[j] < yi)
                        for j in range(len(variants)) if j != i)
        if not dominated:
            pareto.append(i)

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, v in enumerate(variants):
        color = fam_colors.get(families.get(v, ""), "#95a5a6")
        marker = "D" if i in pareto else "o"
        size = 100 if i in pareto else 40
        ax.scatter(xs[i], ys[i], color=color, marker=marker, s=size,
                   zorder=3 if i in pareto else 2, alpha=0.85)
        if i in pareto:
            ax.annotate(v, (xs[i], ys[i]), textcoords="offset points",
                        xytext=(4, 4), fontsize=6.5)

    # Pareto front line
    if pareto:
        pf = sorted([(xs[i], ys[i]) for i in pareto], key=lambda t: t[0])
        px, py = zip(*pf)
        ax.step(px, py, where="post", color="gray", lw=1.2, linestyle="--", alpha=0.7)

    ax.set_xlabel(x_label); ax.set_ylabel("Composite Score (lower=better)")
    ax.set_title(f"Pareto Front — Quality vs {x_label}")
    ax.grid(alpha=0.3)

    handles = [mpatches.Patch(color=c, label=f) for f, c in fam_colors.items()]
    handles.append(plt.Line2D([0], [0], marker="D", color="gray", markersize=8,
                              linestyle="None", label="Pareto front"))
    ax.legend(handles=handles, fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def plot_heatmap(
    all_metrics: dict,
    normalized: dict,
    out_dir: str,
):
    variants = sorted(normalized.keys(), key=lambda v: sum(
        normalized[v].get(k, 0.5) for k, _, _ in QUALITY_METRICS) / len(QUALITY_METRICS))
    metric_keys = [k for k, _, _ in QUALITY_METRICS]
    metric_labels = [lbl for _, lbl, _ in QUALITY_METRICS]

    mat = np.array([[normalized[v].get(k, 0.5) for k in metric_keys] for v in variants])

    fig, ax = plt.subplots(figsize=(max(8, len(metric_keys) * 1.1), max(8, len(variants) * 0.35)))
    im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(range(len(metric_keys))); ax.set_xticklabels(metric_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(variants))); ax.set_yticklabels(variants, fontsize=7)
    ax.set_title("AR Models × Metrics — Normalized Score (green=best)")
    plt.colorbar(im, ax=ax, shrink=0.6)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "heatmap_models_x_metrics.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def plot_family_grouped_bars(
    all_metrics: dict,
    families: dict[str, str],
    out_dir: str,
):
    fam_list = [f for f in AR_FAMILIES if any(families.get(v) == f for v in all_metrics)]
    if not fam_list:
        return

    fam_colors = _family_colors(fam_list)
    metric_keys = [k for k, _, _ in QUALITY_METRICS]
    metric_labels = [lbl for _, lbl, _ in QUALITY_METRICS]
    n_m = len(metric_keys)
    n_f = len(fam_list)
    x = np.arange(n_m)
    width = 0.8 / n_f

    fig, ax = plt.subplots(figsize=(max(14, n_m * 1.5), 6))
    for fi, fam in enumerate(fam_list):
        variants = [v for v in all_metrics if families.get(v) == fam]
        means = []
        for k in metric_keys:
            vals = [float(all_metrics[v].get(k, float("nan"))) for v in variants]
            vals = [v for v in vals if not np.isnan(v)]
            means.append(float(np.mean(vals)) if vals else float("nan"))
        offset = (fi - n_f / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=fam, color=fam_colors[fam], alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(metric_labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Mean metric value"); ax.set_title("AR Families — Grouped by Metric")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "family_grouped_bars.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def plot_hyperparameter_sensitivity(
    all_metrics: dict,
    families: dict[str, str],
    scores: dict,
    out_dir: str,
):
    hp_dir = os.path.join(out_dir, "hyperparameter_sensitivity")
    os.makedirs(hp_dir, exist_ok=True)

    hp_keys = ["hidden_size", "n_layers", "n_coupling", "rnn_hidden", "n_steps"]

    for fam in AR_FAMILIES:
        variants = [v for v in all_metrics if families.get(v) == fam]
        if len(variants) < 3:
            continue

        # Collect config params for each variant
        rows = []
        for v in variants:
            cfg_path = Path(args_global.output_dir) / v / "config.json"
            cfg = {}
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = json.load(f)
            row = {"variant": v, "composite": scores.get(v, float("nan"))}
            for hp in hp_keys:
                val = cfg.get(hp)
                row[hp] = int(val) if val is not None else None
            rows.append(row)

        # Find which HP varies within this family
        varying_hps = [hp for hp in hp_keys
                       if len(set(r[hp] for r in rows if r[hp] is not None)) > 1]
        if not varying_hps:
            continue

        fig, axes = plt.subplots(1, len(varying_hps),
                                 figsize=(5 * len(varying_hps), 4), squeeze=False)

        for ax, hp in zip(axes[0], varying_hps):
            hp_vals = [(r[hp], r["composite"]) for r in rows if r[hp] is not None and not np.isnan(r["composite"])]
            if len(hp_vals) < 2:
                ax.set_visible(False)
                continue
            x_vals, y_vals = zip(*sorted(hp_vals))
            ax.plot(x_vals, y_vals, "o-", color="steelblue")
            for xv, yv, r in [(r[hp], r["composite"], r) for r in rows
                               if r[hp] is not None and not np.isnan(r["composite"])]:
                ax.annotate(r["variant"].replace(fam + "_", ""), (xv, yv),
                            fontsize=6, textcoords="offset points", xytext=(3, 3))
            ax.set_xlabel(hp); ax.set_ylabel("Composite Score (lower=better)")
            ax.set_title(f"{fam}: {hp}")
            ax.grid(alpha=0.3)

        fig.suptitle(f"{fam} — Hyperparameter Sensitivity", fontsize=12)
        plt.tight_layout()
        out_path = os.path.join(hp_dir, f"{fam}_sensitivity.png")
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        plt.close()
        print(f"  [plot] {out_path}")


def plot_training_loss_overlay(variants: list[str], output_dir: str, out_dir: str):
    fam_colors = _family_colors(AR_FAMILIES)
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = False

    for variant in variants:
        history_path = Path(output_dir) / variant / "training_history.json"
        if not history_path.exists():
            continue
        with open(history_path) as f:
            hist = json.load(f)
        # history is either a list of per-epoch dicts or a dict with 'train_loss'/'loss'
        if isinstance(hist, list):
            # Try common loss keys in per-epoch records
            for loss_key in ("total", "train_loss", "loss", "nll"):
                if hist and loss_key in hist[0]:
                    losses = [rec[loss_key] for rec in hist if loss_key in rec]
                    break
            else:
                losses = []
        else:
            losses = hist.get("train_loss") or hist.get("loss") or []
        if not losses:
            continue
        cfg_path = Path(output_dir) / variant / "config.json"
        fam = variant
        if cfg_path.exists():
            with open(cfg_path) as f:
                fam = json.load(f).get("model", variant)
        color = fam_colors.get(fam, "gray")
        ax.plot(losses, lw=0.8, alpha=0.6, color=color, label=variant)
        plotted = True

    if not plotted:
        plt.close()
        return

    ax.set_xlabel("Epoch"); ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss Curves — AR Models")
    ax.grid(alpha=0.3)
    # Only show legend for family representatives (deduplicated)
    handles, labels = ax.get_legend_handles_labels()
    # Show at most 15 labels to keep it readable
    if len(labels) > 15:
        step = max(1, len(labels) // 15)
        handles = handles[::step]; labels = labels[::step]
    ax.legend(handles, labels, fontsize=7, ncol=2)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "training_loss_overlay.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Tier 2 Charts
# ─────────────────────────────────────────────────────────────────────────────

def _group_scenarios_by_family(scenarios_by_model: dict, families: dict) -> dict:
    """Returns {family: [sc_mm_array, ...]} grouping scenario arrays by family."""
    by_family: dict = {}
    for variant, sc_mm in scenarios_by_model.items():
        fam = families.get(variant, variant)
        by_family.setdefault(fam, []).append(sc_mm)
    return by_family


def _mean_autocorr_nd(data: np.ndarray, max_lag: int = 30) -> np.ndarray:
    """data: (T, S) → mean ACF (max_lag,)"""
    return np.stack([M._autocorr_1d(data[:, s], max_lag)
                     for s in range(data.shape[1])]).mean(axis=0)


def _spell_dist(data: np.ndarray, threshold: float = 0.1, max_len: int = 30) -> dict:
    """data: (T, S) → {'wet': (max_len,), 'dry': (max_len,)} frequency"""
    wet_c = np.zeros(max_len); dry_c = np.zeros(max_len)
    for s in range(data.shape[1]):
        wet = data[:, s] > threshold
        count = 1
        for i in range(1, len(wet)):
            if wet[i] == wet[i - 1]:
                count += 1
            else:
                arr = wet_c if wet[i - 1] else dry_c
                if 1 <= count <= max_len:
                    arr[count - 1] += 1
                count = 1
        arr = wet_c if wet[-1] else dry_c
        if 1 <= count <= max_len:
            arr[count - 1] += 1
    wet_c /= max(wet_c.sum(), 1); dry_c /= max(dry_c.sum(), 1)
    return {"wet": wet_c, "dry": dry_c}


def _monthly_means(data: np.ndarray, months: np.ndarray) -> np.ndarray:
    """data: (T, S), months: (T,) → (12,) mean over stations"""
    return np.array([data[months == m].mean() if (months == m).any() else 0.0
                     for m in range(12)])


def plot_autocorr_multilag(
    scenarios_by_model: dict,
    data_raw: np.ndarray,
    out_dir: str,
    families: dict | None = None,
    max_lag: int = 30,
):
    lags = np.arange(1, max_lag + 1)
    obs_acf = _mean_autocorr_nd(data_raw, max_lag)
    fam_colors = _family_colors(AR_FAMILIES)

    def _get_fam(variant):
        if families is not None:
            return families.get(variant, variant)
        cfg_path = Path(args_global.output_dir) / variant / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return json.load(f).get("model", variant)
        return variant

    # ── Detail chart (all individual models) ──
    fig, ax = plt.subplots(figsize=(11, 5))
    for variant, sc_mm in scenarios_by_model.items():
        sc_acfs = np.stack([_mean_autocorr_nd(sc_mm[i], max_lag)
                            for i in range(sc_mm.shape[0])])
        p5, p50, p95 = np.percentile(sc_acfs, [5, 50, 95], axis=0)
        color = fam_colors.get(_get_fam(variant), "gray")
        ax.fill_between(lags, p5, p95, alpha=0.08, color=color)
        ax.plot(lags, p50, lw=1.2, alpha=0.75, color=color, label=variant)
    ax.plot(lags, obs_acf, "k--", lw=2.5, label="Observed")
    ax.axhline(0, color="gray", lw=0.7, linestyle=":")
    ax.set_xlabel("Lag (days)"); ax.set_ylabel("Autocorrelation")
    ax.set_title("Multi-lag Autocorrelation — All Models (detail)")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout()
    detail_path = os.path.join(out_dir, "autocorr_multilag_detail.png")
    plt.savefig(detail_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {detail_path}")

    # ── Family-aggregated chart (default) ──
    fam_map = {v: _get_fam(v) for v in scenarios_by_model}
    by_family = _group_scenarios_by_family(scenarios_by_model, fam_map)
    fig, ax = plt.subplots(figsize=(11, 5))
    for fam, sc_list in by_family.items():
        color = fam_colors.get(fam, "gray")
        all_sc = np.concatenate(sc_list, axis=0)  # (total_sc, n_days, S)
        sc_acfs = np.stack([_mean_autocorr_nd(all_sc[i], max_lag)
                            for i in range(all_sc.shape[0])])
        p_min = sc_acfs.min(axis=0)
        p_max = sc_acfs.max(axis=0)
        p_med = np.median(sc_acfs, axis=0)
        ax.fill_between(lags, p_min, p_max, alpha=0.15, color=color)
        ax.plot(lags, p_med, lw=2.0, color=color, label=fam)
    ax.plot(lags, obs_acf, "k--", lw=2.5, label="Observed")
    ax.axhline(0, color="gray", lw=0.7, linestyle=":")
    ax.set_xlabel("Lag (days)"); ax.set_ylabel("Autocorrelation")
    ax.set_title("Multi-lag Autocorrelation — Family Aggregated (band=min–max across variants)")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "autocorr_multilag.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def plot_spell_length_comparison(
    scenarios_by_model: dict,
    data_raw: np.ndarray,
    out_dir: str,
    families: dict | None = None,
    max_len: int = 20,
):
    obs_sp = _spell_dist(data_raw, max_len=max_len)
    lens = np.arange(1, max_len + 1)
    fam_colors = _family_colors(AR_FAMILIES)

    def _get_fam(variant):
        if families is not None:
            return families.get(variant, variant)
        cfg_path = Path(args_global.output_dir) / variant / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return json.load(f).get("model", variant)
        return variant

    # ── Detail chart (all individual models) ──
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, spell_type, title in [(ax1, "wet", "Wet Spells"), (ax2, "dry", "Dry Spells")]:
        ax.bar(lens, obs_sp[spell_type], width=0.4, color="black", alpha=0.5, label="Observed")
        for variant, sc_mm in scenarios_by_model.items():
            sc_dists = np.stack([_spell_dist(sc_mm[i], max_len=max_len)[spell_type]
                                 for i in range(sc_mm.shape[0])])
            p50 = np.median(sc_dists, axis=0)
            color = fam_colors.get(_get_fam(variant), "gray")
            ax.plot(lens, p50, lw=1.2, alpha=0.75, color=color, label=variant)
        ax.set_xlabel("Length (days)"); ax.set_ylabel("Relative frequency")
        ax.set_title(title + " (detail)"); ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Spell Length Distributions — All Models", fontsize=12)
    plt.tight_layout()
    detail_path = os.path.join(out_dir, "spell_length_detail.png")
    plt.savefig(detail_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {detail_path}")

    # ── Family-aggregated chart (default) ──
    fam_map = {v: _get_fam(v) for v in scenarios_by_model}
    by_family = _group_scenarios_by_family(scenarios_by_model, fam_map)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax, spell_type, title in [(ax1, "wet", "Wet Spells"), (ax2, "dry", "Dry Spells")]:
        ax.bar(lens, obs_sp[spell_type], width=0.4, color="black", alpha=0.5, label="Observed")
        for fam, sc_list in by_family.items():
            color = fam_colors.get(fam, "gray")
            all_sc = np.concatenate(sc_list, axis=0)
            sc_dists = np.stack([_spell_dist(all_sc[i], max_len=max_len)[spell_type]
                                 for i in range(all_sc.shape[0])])
            p_min = sc_dists.min(axis=0)
            p_max = sc_dists.max(axis=0)
            p_med = np.median(sc_dists, axis=0)
            ax.fill_between(lens, p_min, p_max, alpha=0.15, color=color)
            ax.plot(lens, p_med, lw=2.0, color=color, label=fam)
        ax.set_xlabel("Length (days)"); ax.set_ylabel("Relative frequency")
        ax.set_title(title); ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Spell Length Distributions — Family Aggregated (band=min–max across variants)", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "spell_length_comparison.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def plot_monthly_precip(
    scenarios_by_model: dict,
    data_raw: np.ndarray,
    obs_months: np.ndarray,
    out_dir: str,
    families: dict | None = None,
):
    obs_mm = _monthly_means(data_raw, obs_months)
    obs_var = np.array([data_raw[obs_months == m].var() if (obs_months == m).any()
                        else 0.0 for m in range(12)])
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    x = np.arange(12)
    fam_colors = _family_colors(AR_FAMILIES)

    def _get_fam(variant):
        if families is not None:
            return families.get(variant, variant)
        cfg_path = Path(args_global.output_dir) / variant / "config.json"
        if cfg_path.exists():
            with open(cfg_path) as f:
                return json.load(f).get("model", variant)
        return variant

    def _sc_monthly_mean(sc_mm):
        T = sc_mm.shape[1]
        sc_months = np.array([t % 12 for t in range(T)])
        return np.array([sc_mm[:, sc_months == m, :].mean() if (sc_months == m).any()
                         else 0.0 for m in range(12)])

    def _sc_monthly_var(sc_mm):
        T = sc_mm.shape[1]
        sc_months = np.array([t % 12 for t in range(T)])
        return np.array([sc_mm[:, sc_months == m, :].var() if (sc_months == m).any()
                         else 0.0 for m in range(12)])

    # ── Detail chart (all individual models) ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    ax1.bar(x, obs_mm, width=0.35, color="black", alpha=0.6, label="Observed")
    ax2.bar(x, obs_var, width=0.35, color="black", alpha=0.6, label="Observed")
    for variant, sc_mm in scenarios_by_model.items():
        color = fam_colors.get(_get_fam(variant), "gray")
        ax1.plot(x, _sc_monthly_mean(sc_mm), "o--", lw=1, alpha=0.7, color=color, label=variant)
        ax2.plot(x, _sc_monthly_var(sc_mm), "o--", lw=1, alpha=0.7, color=color, label=variant)
    ax1.set_ylabel("Mean (mm/day)"); ax1.set_title("Monthly Mean Precipitation (detail)")
    ax1.legend(fontsize=7, ncol=2); ax1.grid(axis="y", alpha=0.3)
    ax2.set_ylabel("Variance (mm/day)²"); ax2.set_title("Monthly Variance (detail)")
    ax2.legend(fontsize=7, ncol=2); ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks(x); ax2.set_xticklabels(month_labels)
    plt.tight_layout()
    detail_path = os.path.join(out_dir, "monthly_precip_detail.png")
    plt.savefig(detail_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {detail_path}")

    # ── Family-aggregated chart (default) ──
    fam_map = {v: _get_fam(v) for v in scenarios_by_model}
    by_family = _group_scenarios_by_family(scenarios_by_model, fam_map)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    ax1.bar(x, obs_mm, width=0.35, color="black", alpha=0.6, label="Observed")
    ax2.bar(x, obs_var, width=0.35, color="black", alpha=0.6, label="Observed")
    for fam, sc_list in by_family.items():
        color = fam_colors.get(fam, "gray")
        all_sc = np.concatenate(sc_list, axis=0)
        means = np.stack([_sc_monthly_mean(all_sc[i:i+1]) for i in range(all_sc.shape[0])])
        vars_ = np.stack([_sc_monthly_var(all_sc[i:i+1]) for i in range(all_sc.shape[0])])
        ax1.fill_between(x, means.min(axis=0), means.max(axis=0), alpha=0.15, color=color)
        ax1.plot(x, np.median(means, axis=0), "o-", lw=2.0, color=color, label=fam)
        ax2.fill_between(x, vars_.min(axis=0), vars_.max(axis=0), alpha=0.15, color=color)
        ax2.plot(x, np.median(vars_, axis=0), "o-", lw=2.0, color=color, label=fam)
    ax1.set_ylabel("Mean (mm/day)"); ax1.set_title("Monthly Mean Precipitation — Family Aggregated")
    ax1.legend(fontsize=9); ax1.grid(axis="y", alpha=0.3)
    ax2.set_ylabel("Variance (mm/day)²"); ax2.set_title("Monthly Variance — Family Aggregated")
    ax2.legend(fontsize=9); ax2.grid(axis="y", alpha=0.3)
    ax2.set_xticks(x); ax2.set_xticklabels(month_labels)
    plt.tight_layout()
    out_path1 = os.path.join(out_dir, "monthly_mean_precip.png")
    plt.savefig(out_path1, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path1}")

    # Variance standalone (family-aggregated)
    fig2, ax = plt.subplots(figsize=(11, 4))
    ax.bar(x, obs_var, width=0.35, color="black", alpha=0.6, label="Observed")
    for fam, sc_list in by_family.items():
        color = fam_colors.get(fam, "gray")
        all_sc = np.concatenate(sc_list, axis=0)
        vars_ = np.stack([_sc_monthly_var(all_sc[i:i+1]) for i in range(all_sc.shape[0])])
        ax.fill_between(x, vars_.min(axis=0), vars_.max(axis=0), alpha=0.15, color=color)
        ax.plot(x, np.median(vars_, axis=0), "o-", lw=2.0, color=color, label=fam)
    ax.set_ylabel("Variance (mm/day)²"); ax.set_title("Monthly Precipitation Variance — Family Aggregated")
    ax.legend(fontsize=9); ax.grid(axis="y", alpha=0.3)
    ax.set_xticks(x); ax.set_xticklabels(month_labels)
    plt.tight_layout()
    out_path2 = os.path.join(out_dir, "monthly_variance.png")
    plt.savefig(out_path2, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path2}")


def plot_spread_envelopes(
    scenarios_by_model: dict,
    data_raw: np.ndarray,
    out_dir: str,
    top_n: int = 5,
):
    if not scenarios_by_model:
        return
    # Use first top_n models
    models_shown = list(scenarios_by_model.keys())[:top_n]
    fam_colors = _family_colors(AR_FAMILIES)

    n_cols = min(len(models_shown), 3)
    n_rows = (len(models_shown) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows), squeeze=False)

    for idx, variant in enumerate(models_shown):
        sc_mm = scenarios_by_model[variant]
        ax = axes[idx // n_cols][idx % n_cols]

        sc_daily = sc_mm.mean(axis=2)  # (n_sc, T) mean over stations
        T = sc_daily.shape[1]
        days = np.arange(T)
        p5, p25, p50, p75, p95 = np.percentile(sc_daily, [5, 25, 50, 75, 95], axis=0)

        cfg_path = Path(args_global.output_dir) / variant / "config.json"
        fam = variant
        if cfg_path.exists():
            with open(cfg_path) as f:
                fam = json.load(f).get("model", variant)
        color = fam_colors.get(fam, "steelblue")

        ax.fill_between(days, p5, p95, alpha=0.15, color=color, label="5–95%")
        ax.fill_between(days, p25, p75, alpha=0.30, color=color, label="25–75%")
        ax.plot(days, p50, color=color, lw=1.2, label="Median")

        # Observed (last T days)
        if data_raw.shape[0] >= T:
            obs_daily = data_raw[-T:].mean(axis=1)
            ax.plot(days, obs_daily, color="black", lw=0.7, alpha=0.5, label="Observed")

        ax.set_title(variant, fontsize=9); ax.grid(alpha=0.3)
        ax.set_ylabel("mm/day (mean stations)"); ax.legend(fontsize=7)

    for i in range(len(models_shown), n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    fig.suptitle("Scenario Spread Envelopes — Mean over Stations", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "spread_envelopes.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def plot_transition_probs(
    scenarios_by_model: dict,
    data_raw: np.ndarray,
    out_dir: str,
    threshold: float = 0.1,
):
    def _trans_probs(data: np.ndarray):
        wet = data > threshold
        S = data.shape[1]
        p_ww = p_wd = p_dw = p_dd = 0.0
        for i in range(S):
            w = wet[:, i].astype(float)
            n_w = w[:-1].sum(); n_d = (1 - w[:-1]).sum()
            p_ww += (w[:-1] * w[1:]).sum() / max(n_w, 1)
            p_wd += (w[:-1] * (1 - w[1:])).sum() / max(n_w, 1)
            p_dw += ((1 - w[:-1]) * w[1:]).sum() / max(n_d, 1)
            p_dd += ((1 - w[:-1]) * (1 - w[1:])).sum() / max(n_d, 1)
        return np.array([p_ww, p_wd, p_dw, p_dd]) / S

    labels = ["P(W|W)", "P(D|W)", "P(W|D)", "P(D|D)"]
    obs_tp = _trans_probs(data_raw)

    variants = list(scenarios_by_model.keys())
    n = len(variants)
    x = np.arange(4)
    width = 0.8 / (n + 1)
    fam_colors = _family_colors(AR_FAMILIES)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - (n / 2) * width, obs_tp, width, color="black", alpha=0.7, label="Observed")
    for fi, variant in enumerate(variants):
        sc_mm = scenarios_by_model[variant]
        sc_tp = np.mean([_trans_probs(sc_mm[i]) for i in range(sc_mm.shape[0])], axis=0)
        cfg_path = Path(args_global.output_dir) / variant / "config.json"
        fam = variant
        if cfg_path.exists():
            with open(cfg_path) as f:
                fam = json.load(f).get("model", variant)
        color = fam_colors.get(fam, "gray")
        offset = (fi + 1 - n / 2) * width
        ax.bar(x + offset, sc_tp, width, color=color, alpha=0.75, label=variant)

    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylabel("Probability"); ax.set_title("Wet/Dry Transition Probabilities")
    ax.legend(fontsize=7, ncol=2); ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "transition_probabilities.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def plot_return_period(
    scenarios_by_model: dict,
    data_raw: np.ndarray,
    out_dir: str,
):
    """Annual max daily precipitation empirical CDF (return period proxy)."""
    fam_colors = _family_colors(AR_FAMILIES)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Observed: compute annual maxima per station, take mean over stations
    obs_ann_max = data_raw.max(axis=0)  # station maxima over all time
    obs_sorted = np.sort(obs_ann_max)
    p_obs = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1)
    ax.plot(obs_sorted, p_obs, "k--", lw=2.5, label="Observed (station max)")

    for variant, sc_mm in scenarios_by_model.items():
        # Per scenario: max over time, mean over stations → distribution over scenarios
        sc_ann_max = sc_mm.max(axis=1).mean(axis=1)  # (n_sc,)
        sc_sorted = np.sort(sc_ann_max)
        p_sc = np.arange(1, len(sc_sorted) + 1) / (len(sc_sorted) + 1)
        cfg_path = Path(args_global.output_dir) / variant / "config.json"
        fam = variant
        if cfg_path.exists():
            with open(cfg_path) as f:
                fam = json.load(f).get("model", variant)
        color = fam_colors.get(fam, "gray")
        ax.plot(sc_sorted, p_sc, lw=1.5, alpha=0.75, color=color, label=variant)

    ax.set_xlabel("Annual Max Daily Precip (mm/day)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title("Return Period — Annual Max Daily Precipitation")
    ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "return_period.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Text Reports
# ─────────────────────────────────────────────────────────────────────────────

def write_comparison_report(
    all_metrics: dict,
    scores: dict,
    families: dict[str, str],
    out_dir: str,
) -> str:
    variants = sorted(scores.keys(), key=lambda v: scores.get(v, 1.0))
    metric_keys = [k for k, _, _ in QUALITY_METRICS]
    metric_labels = [lbl for _, lbl, _ in QUALITY_METRICS]

    col_w = 28
    metric_w = 12

    lines = ["AR Models Comparison Report", "=" * 120, ""]
    header = f"{'Variant':<{col_w}} {'Family':<20} {'Composite':>{metric_w}}"
    for lbl in metric_labels:
        header += f" {lbl[:metric_w]:>{metric_w}}"
    lines.append(header)
    lines.append("-" * len(header))

    for v in variants:
        fam = families.get(v, "?")
        comp = scores.get(v, float("nan"))
        row = f"{v:<{col_w}} {fam:<20} {comp:>{metric_w}.4f}"
        for k in metric_keys:
            val = all_metrics.get(v, {}).get(k, float("nan"))
            try:
                row += f" {float(val):>{metric_w}.4f}"
            except (TypeError, ValueError):
                row += f" {'N/A':>{metric_w}}"
        lines.append(row)

    report = "\n".join(lines)
    out_path = os.path.join(out_dir, "ar_comparison_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"  [report] {out_path}")
    return report


def write_family_summary(
    all_metrics: dict,
    scores: dict,
    families: dict[str, str],
    out_dir: str,
):
    fam_data: dict[str, list] = {}
    for v, s in scores.items():
        fam = families.get(v, "unknown")
        fam_data.setdefault(fam, []).append((v, s))

    lines = ["AR Family Summary", "=" * 80, ""]
    lines.append(f"{'Family':<25} {'Count':>6} {'Best':>10} {'Mean':>10} {'Worst':>10}  Best Variant")
    lines.append("-" * 80)

    for fam in AR_FAMILIES:
        entries = fam_data.get(fam, [])
        if not entries:
            continue
        valid = [(v, s) for v, s in entries if not np.isnan(s)]
        if not valid:
            continue
        scores_arr = [s for _, s in valid]
        best_v, best_s = min(valid, key=lambda x: x[1])
        lines.append(f"{fam:<25} {len(valid):>6} {best_s:>10.4f} {np.mean(scores_arr):>10.4f}"
                     f" {max(scores_arr):>10.4f}  {best_v}")

    out_path = os.path.join(out_dir, "ar_family_summary.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [report] {out_path}")


def write_hyperparameter_sensitivity_report(
    all_metrics: dict,
    scores: dict,
    families: dict[str, str],
    out_dir: str,
):
    hp_keys = ["hidden_size", "n_layers", "n_coupling", "rnn_hidden", "n_steps",
               "latent_size", "gru_hidden"]
    lines = ["AR Hyperparameter Sensitivity Report", "=" * 100, ""]

    for fam in AR_FAMILIES:
        variants = sorted([v for v in all_metrics if families.get(v) == fam])
        if not variants:
            continue
        lines.append(f"\n{'─'*70}")
        lines.append(f"Family: {fam}  ({len(variants)} variants)")
        lines.append(f"{'-'*70}")
        lines.append(f"  {'Variant':<35} {'Composite':>10}" + "".join(f" {hp:>14}" for hp in hp_keys))
        lines.append(f"  {'-'*35} {'-'*10}" + "".join(f" {'-'*14}" for _ in hp_keys))

        for v in sorted(variants, key=lambda x: scores.get(x, 1.0)):
            cfg_path = Path(args_global.output_dir) / v / "config.json"
            cfg = {}
            if cfg_path.exists():
                with open(cfg_path) as f:
                    cfg = json.load(f)
            comp = scores.get(v, float("nan"))
            row = f"  {v:<35} {comp:>10.4f}"
            for hp in hp_keys:
                val = cfg.get(hp)
                row += f" {str(val) if val is not None else 'N/A':>14}"
            lines.append(row)

    out_path = os.path.join(out_dir, "ar_hyperparameter_sensitivity.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [report] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Per-Station Analysis (Tier 1 — from metrics.json)
# ─────────────────────────────────────────────────────────────────────────────

OUTLIER_STATION = "st_362"


def _load_station_names(data_path: str) -> list:
    """Load data just to retrieve station names (fast, ~1 sec)."""
    _, _, _, _, names = load_data(
        data_path=data_path,
        normalization_mode="scale_only",
        missing_strategy="impute_station_median",
    )
    return names


def plot_station_heatmap_ar(
    all_metrics: dict,
    scores: dict,
    metric_key: str,
    title: str,
    out_dir: str,
    filename: str,
    outlier_station: str = OUTLIER_STATION,
    clip_95th: bool = False,
):
    """Heatmap: models (rows, best→worst) × stations (cols, alphabetical)."""
    variants = sorted(
        [v for v in all_metrics if isinstance(all_metrics[v].get(metric_key), dict)],
        key=lambda v: scores.get(v, 1.0),
    )
    if not variants:
        print(f"  [warn] No {metric_key} data for heatmap")
        return

    all_stations = set()
    for v in variants:
        all_stations.update(all_metrics[v][metric_key].keys())
    stations = sorted(all_stations)

    n_models = len(variants)
    n_stations = len(stations)

    mat = np.full((n_models, n_stations), np.nan)
    for i, v in enumerate(variants):
        per_st = all_metrics[v][metric_key]
        for j, st in enumerate(stations):
            val = per_st.get(st)
            if val is not None:
                try:
                    mat[i, j] = float(val)
                except (TypeError, ValueError):
                    pass

    figw = max(16, n_stations * 0.22)
    figh = max(6, n_models * 0.35)
    fig, ax = plt.subplots(figsize=(figw, figh))
    if clip_95th and not np.all(np.isnan(mat)):
        vmax_clipped = np.nanpercentile(mat, 95)
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=vmax_clipped)
        cb = plt.colorbar(im, ax=ax, shrink=0.5)
        cb.set_label("(clipped at 95th pct)")
    else:
        im = ax.imshow(mat, aspect="auto", cmap="RdYlGn_r")
        plt.colorbar(im, ax=ax, shrink=0.5)
    ax.set_xticks(range(n_stations))
    ax.set_xticklabels(stations, rotation=90, fontsize=5)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(variants, fontsize=7)
    ax.set_title(f"{title} — Models × Stations (green=best, red=worst)")

    if outlier_station in stations:
        idx = stations.index(outlier_station)
        ax.axvline(x=idx - 0.5, color="white", lw=2.0, linestyle="--", alpha=0.9)
        ax.axvline(x=idx + 0.5, color="white", lw=2.0, linestyle="--", alpha=0.9)
        ax.text(idx, -1.5, "↑outlier", ha="center", fontsize=6, color="red",
                clip_on=False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, filename)
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def plot_station_score_distribution(
    all_metrics: dict,
    scores: dict,
    out_dir: str,
    outlier_station: str = OUTLIER_STATION,
):
    """Horizontal box plots of per-station Wasserstein, sorted by median (excl. outlier)."""
    key = "wasserstein_per_station"
    variants = [v for v in all_metrics if isinstance(all_metrics[v].get(key), dict)]
    if not variants:
        return

    model_data = {}
    outlier_vals = {}
    for v in variants:
        per_st = all_metrics[v][key]
        vals = []
        for st, val in per_st.items():
            if st == outlier_station:
                try:
                    outlier_vals[v] = float(val)
                except (TypeError, ValueError):
                    pass
                continue
            try:
                vals.append(float(val))
            except (TypeError, ValueError):
                pass
        if vals:
            model_data[v] = np.array(vals)

    if not model_data:
        return

    variants_sorted = sorted(model_data.keys(), key=lambda v: np.median(model_data[v]))
    n = len(variants_sorted)

    fig, ax = plt.subplots(figsize=(10, max(6, n * 0.45)))
    data_list = [model_data[v] for v in variants_sorted]

    ax.boxplot(
        data_list, positions=list(range(n)), vert=False, patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.55),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
        whiskerprops=dict(alpha=0.6),
        capprops=dict(alpha=0.6),
    )

    # Outlier station diamond
    legend_added = False
    for i, v in enumerate(variants_sorted):
        if v in outlier_vals:
            label = f"Outlier ({outlier_station})" if not legend_added else ""
            ax.scatter(outlier_vals[v], i, marker="D", color="crimson", s=45,
                       zorder=5, label=label)
            legend_added = True

    ax.set_yticks(range(n))
    ax.set_yticklabels(
        [f"{v}  σ={np.std(model_data[v]):.3f}" for v in variants_sorted],
        fontsize=7,
    )
    ax.set_xlabel("Wasserstein distance (mm/day)")
    ax.set_title(f"Per-Station Score Distribution (sorted by median, excl. {outlier_station})")
    ax.grid(axis="x", alpha=0.3)
    if legend_added:
        ax.legend(fontsize=8)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "station_score_distribution.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def plot_best_model_per_station(
    all_metrics: dict,
    scores: dict,
    families: dict,
    out_dir: str,
    outlier_station: str = OUTLIER_STATION,
):
    """Bar chart: number of stations where each model has the lowest Wasserstein."""
    key = "wasserstein_per_station"
    variants = [v for v in all_metrics if isinstance(all_metrics[v].get(key), dict)]
    if not variants:
        return

    all_stations = set()
    for v in variants:
        all_stations.update(all_metrics[v][key].keys())
    stations_ranked = sorted(all_stations - {outlier_station})

    wins = {v: 0 for v in variants}
    for st in stations_ranked:
        best_v, best_val = None, float("inf")
        for v in variants:
            val = all_metrics[v][key].get(st)
            if val is not None:
                try:
                    fval = float(val)
                    if fval < best_val:
                        best_val, best_v = fval, v
                except (TypeError, ValueError):
                    pass
        if best_v is not None:
            wins[best_v] += 1

    sorted_variants = sorted(wins.keys(), key=lambda v: wins[v], reverse=True)
    win_counts = [wins[v] for v in sorted_variants]

    fam_colors = _family_colors(list(set(families.values())))
    colors = [fam_colors.get(families.get(v, ""), "#95a5a6") for v in sorted_variants]

    fig, ax = plt.subplots(figsize=(max(10, len(sorted_variants) * 0.6), 5))
    bars = ax.bar(range(len(sorted_variants)), win_counts, color=colors)
    for bar, count in zip(bars, win_counts):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, count + 0.15,
                    str(count), ha="center", va="bottom", fontsize=7)

    ax.set_xticks(range(len(sorted_variants)))
    ax.set_xticklabels(sorted_variants, rotation=60, ha="right", fontsize=7)
    ax.set_ylabel("Stations where model wins (lowest Wasserstein)")
    ax.set_title(
        f"Best Model Per Station — Win Count "
        f"(N={len(stations_ranked)} stations, excl. {outlier_station})\n"
        f"Total wins should sum to {len(stations_ranked)}"
    )
    ax.grid(axis="y", alpha=0.3)

    handles = [mpatches.Patch(color=c, label=f) for f, c in fam_colors.items()]
    ax.legend(handles=handles, fontsize=8, loc="upper right")

    plt.tight_layout()
    out_path = os.path.join(out_dir, "station_wins_per_model.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def write_per_station_report(
    all_metrics: dict,
    scores: dict,
    out_dir: str,
    outlier_station: str = OUTLIER_STATION,
):
    """Write ar_per_station_report.txt with per-model station statistics."""
    key = "wasserstein_per_station"
    variants = sorted(
        [v for v in all_metrics if isinstance(all_metrics[v].get(key), dict)],
        key=lambda v: scores.get(v, 1.0),
    )
    if not variants:
        return

    w = 12
    lines = [
        "AR Per-Station Analysis Report",
        "=" * 110,
        "",
        f"Note: {outlier_station} is a known outlier "
        f"(Wasserstein ~50-100× higher than other stations).",
        "",
        f"{'Variant':<35} {'Composite':>{w}} {'Mean W':>{w}} {'Median W':>{w}}"
        f" {'Std W':>{w}} {'Max W':>{w}} {'Max Station':>15}",
        "-" * 110,
    ]

    for v in variants:
        per_st = all_metrics[v][key]
        vals_dict = {}
        for st, val in per_st.items():
            try:
                vals_dict[st] = float(val)
            except (TypeError, ValueError):
                pass
        if not vals_dict:
            continue

        arr = np.array(list(vals_dict.values()))
        max_st = max(vals_dict, key=vals_dict.get)
        comp = scores.get(v, float("nan"))
        lines.append(
            f"{v:<35} {comp:>{w}.4f} {arr.mean():>{w}.4f} {np.median(arr):>{w}.4f}"
            f" {arr.std():>{w}.4f} {arr.max():>{w}.4f} {max_st:>15}"
        )

    lines += [
        "",
        f"* Mean/Median/Std/Max computed over all stations including {outlier_station}.",
        "* Sort by composite score (ascending = better).",
    ]

    out_path = os.path.join(out_dir, "ar_per_station_report.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"  [report] {out_path}")


def plot_station_rank_heatmap(
    all_metrics: dict,
    scores: dict,
    families: dict,
    out_dir: str,
    outlier_station: str = OUTLIER_STATION,
):
    """Heatmap: rank per station (1=best model, N=worst). Removes outlier scale bias."""
    key = "wasserstein_per_station"
    variants = sorted(
        [v for v in all_metrics if isinstance(all_metrics[v].get(key), dict)],
        key=lambda v: scores.get(v, 1.0),
    )
    if not variants:
        print("  [warn] No wasserstein_per_station data for rank heatmap")
        return

    all_stations = set()
    for v in variants:
        all_stations.update(all_metrics[v][key].keys())
    stations = sorted(all_stations)
    n_models = len(variants)
    n_stations = len(stations)

    mat = np.full((n_models, n_stations), np.nan)
    for i, v in enumerate(variants):
        per_st = all_metrics[v][key]
        for j, st in enumerate(stations):
            val = per_st.get(st)
            if val is not None:
                try:
                    mat[i, j] = float(val)
                except (TypeError, ValueError):
                    pass

    # Convert raw values to per-station ranks (rank 1 = lowest/best Wasserstein)
    rank_mat = np.full_like(mat, np.nan)
    for j in range(n_stations):
        col = mat[:, j]
        valid_mask = ~np.isnan(col)
        if valid_mask.sum() > 0:
            order = np.argsort(col[valid_mask])
            ranks = np.empty_like(order)
            ranks[order] = np.arange(1, valid_mask.sum() + 1)
            rank_mat[valid_mask, j] = ranks

    figw = max(16, n_stations * 0.22)
    figh = max(6, n_models * 0.35)
    fig, ax = plt.subplots(figsize=(figw, figh))
    im = ax.imshow(rank_mat, aspect="auto", cmap="RdYlGn_r", vmin=1, vmax=n_models)
    ax.set_xticks(range(n_stations))
    ax.set_xticklabels(stations, rotation=90, fontsize=5)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(variants, fontsize=7)
    ax.set_title("Wasserstein Rank per Station — Models × Stations (1=best, green; N=worst, red)")
    cb = plt.colorbar(im, ax=ax, shrink=0.5)
    cb.set_label("Rank (1=best)")

    if outlier_station in stations:
        idx = stations.index(outlier_station)
        ax.axvline(x=idx - 0.5, color="white", lw=2.0, linestyle="--", alpha=0.9)
        ax.axvline(x=idx + 0.5, color="white", lw=2.0, linestyle="--", alpha=0.9)
        ax.text(idx, -1.5, "↑outlier", ha="center", fontsize=6, color="red", clip_on=False)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "station_rank_wasserstein_heatmap.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def plot_per_station_detail(
    all_metrics: dict,
    scores: dict,
    families: dict,
    out_dir: str,
    outlier_station: str = OUTLIER_STATION,
):
    """Per-station PNG showing Wasserstein + wet-freq-error sorted bar charts per model."""
    key_w = "wasserstein_per_station"
    key_wf = "wet_day_freq_error_per_station"

    variants_w = [v for v in all_metrics if isinstance(all_metrics[v].get(key_w), dict)]
    if not variants_w:
        return

    all_stations: set = set()
    for v in variants_w:
        all_stations.update(all_metrics[v][key_w].keys())
    stations = sorted(all_stations)
    n_stations = len(stations)

    # Compute median Wasserstein per station for difficulty ranking
    station_medians = {}
    for st in stations:
        vals = []
        for v in variants_w:
            val = all_metrics[v][key_w].get(st)
            if val is not None:
                try:
                    vals.append(float(val))
                except (TypeError, ValueError):
                    pass
        station_medians[st] = float(np.median(vals)) if vals else float("inf")

    stations_by_diff = sorted(stations, key=lambda s: station_medians[s])
    station_rank = {st: i + 1 for i, st in enumerate(stations_by_diff)}

    fam_colors = _family_colors(AR_FAMILIES)
    stations_dir = os.path.join(out_dir, "stations")
    os.makedirs(stations_dir, exist_ok=True)

    print(f"  [plot] Generating per-station detail charts ({n_stations} stations)...")
    for st in stations:
        # Wasserstein per model
        w_vals = {}
        for v in variants_w:
            val = all_metrics[v][key_w].get(st)
            if val is not None:
                try:
                    w_vals[v] = float(val)
                except (TypeError, ValueError):
                    pass

        # Wet freq error per model
        wf_vals = {}
        for v in all_metrics:
            wf_data = all_metrics[v].get(key_wf)
            if isinstance(wf_data, dict):
                val = wf_data.get(st)
                if val is not None:
                    try:
                        wf_vals[v] = float(val)
                    except (TypeError, ValueError):
                        pass

        if not w_vals:
            continue

        rank = station_rank.get(st, 0)
        n_third = max(1, n_stations // 3)
        difficulty = "hardest" if rank >= n_stations - n_third else ("easy" if rank <= n_third else "medium")
        is_outlier = st == outlier_station

        n_rows = max(len(w_vals), len(wf_vals) if wf_vals else 1)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(5, n_rows * 0.28 + 2)))

        # Left: Wasserstein sorted ascending
        sorted_w = sorted(w_vals.items(), key=lambda x: x[1])
        model_names_w = [v for v, _ in sorted_w]
        vals_w = [val for _, val in sorted_w]
        colors_w = [fam_colors.get(families.get(v, v), "#95a5a6") for v in model_names_w]
        ax1.barh(range(len(model_names_w)), vals_w, color=colors_w, alpha=0.8)
        ax1.set_yticks(range(len(model_names_w)))
        ax1.set_yticklabels(model_names_w, fontsize=7)
        ax1.set_xlabel("Wasserstein distance (mm/day)")
        ax1.set_title("Wasserstein (sorted ascending)")
        if vals_w:
            ax1.text(vals_w[0] * 1.02, 0, " best", va="center", fontsize=7, color="green")
        ax1.grid(axis="x", alpha=0.3)

        # Right: Wet freq error sorted by abs value
        if wf_vals:
            sorted_wf = sorted(wf_vals.items(), key=lambda x: abs(x[1]))
            model_names_wf = [v for v, _ in sorted_wf]
            vals_wf = [val for _, val in sorted_wf]
            colors_wf = [fam_colors.get(families.get(v, v), "#95a5a6") for v in model_names_wf]
            ax2.barh(range(len(model_names_wf)), vals_wf, color=colors_wf, alpha=0.8)
            ax2.set_yticks(range(len(model_names_wf)))
            ax2.set_yticklabels(model_names_wf, fontsize=7)
            ax2.axvline(0, color="black", lw=0.8)
        else:
            ax2.text(0.5, 0.5, "no wet freq data", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=9, color="gray")
        ax2.set_xlabel("Wet day freq error")
        ax2.set_title("Wet Freq Error (sorted by |error|)")
        ax2.grid(axis="x", alpha=0.3)

        outlier_note = " (known outlier station)" if is_outlier else ""
        fig.suptitle(
            f"{st}{outlier_note} — Per-Station Model Comparison\n"
            f"Station rank: {rank}/{n_stations} ({difficulty})",
            fontsize=10,
        )
        plt.tight_layout()
        out_path = os.path.join(stations_dir, f"{st}.png")
        plt.savefig(out_path, dpi=100, bbox_inches="tight")
        plt.close()

    print(f"  [plot] {n_stations} station charts → {stations_dir}/")


# ─────────────────────────────────────────────────────────────────────────────
# Per-Station Analysis (Tier 2 — from rollout scenarios)
# ─────────────────────────────────────────────────────────────────────────────

def plot_per_station_lag1_scatter(
    scenarios_by_model: dict,
    data_raw: np.ndarray,
    station_names: list,
    families: dict,
    out_dir: str,
):
    """Scatter: observed vs model median lag-1 ACF per station."""
    S = data_raw.shape[1]
    if station_names is None or len(station_names) != S:
        station_names = [f"st_{i}" for i in range(S)]

    obs_acf = np.array([M._autocorr_1d(data_raw[:, s], 1)[0] for s in range(S)])

    fam_colors = _family_colors(AR_FAMILIES)
    n_models = len(scenarios_by_model)
    if n_models == 0:
        return

    n_cols = min(n_models, 3)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, (variant, sc_mm) in enumerate(scenarios_by_model.items()):
        ax = axes[idx // n_cols][idx % n_cols]
        # Median lag-1 ACF across scenarios per station
        sc_acf = np.array([
            np.median([M._autocorr_1d(sc_mm[i, :, s], 1)[0]
                       for i in range(sc_mm.shape[0])])
            for s in range(S)
        ])
        fam = families.get(variant, variant)
        color = fam_colors.get(fam, "steelblue")
        ax.scatter(obs_acf, sc_acf, alpha=0.5, s=18, color=color)
        lo = min(obs_acf.min(), sc_acf.min()) - 0.02
        hi = max(obs_acf.max(), sc_acf.max()) + 0.02
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Observed lag-1 ACF"); ax.set_ylabel("Model lag-1 ACF")
        ax.set_title(variant, fontsize=8); ax.grid(alpha=0.3)

    for i in range(n_models, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    fig.suptitle("Per-Station Lag-1 Autocorrelation: Observed vs Model (identity=perfect)", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "station_autocorr_scatter.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def plot_per_station_wetfreq_scatter(
    scenarios_by_model: dict,
    data_raw: np.ndarray,
    station_names: list,
    families: dict,
    out_dir: str,
    threshold: float = 0.1,
):
    """Scatter: observed vs model median P(rain > threshold) per station."""
    S = data_raw.shape[1]
    if station_names is None or len(station_names) != S:
        station_names = [f"st_{i}" for i in range(S)]

    obs_wf = (data_raw > threshold).mean(axis=0)  # (S,)

    fam_colors = _family_colors(AR_FAMILIES)
    n_models = len(scenarios_by_model)
    if n_models == 0:
        return

    n_cols = min(n_models, 3)
    n_rows = (n_models + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    for idx, (variant, sc_mm) in enumerate(scenarios_by_model.items()):
        ax = axes[idx // n_cols][idx % n_cols]
        sc_wf = np.median(
            [(sc_mm[i] > threshold).mean(axis=0) for i in range(sc_mm.shape[0])],
            axis=0,
        )  # (S,)
        fam = families.get(variant, variant)
        color = fam_colors.get(fam, "steelblue")
        ax.scatter(obs_wf, sc_wf, alpha=0.5, s=18, color=color)
        hi = max(obs_wf.max(), sc_wf.max()) + 0.02
        ax.plot([0, hi], [0, hi], "k--", lw=1, alpha=0.5)
        ax.set_xlim(0, hi); ax.set_ylim(0, hi)
        ax.set_xlabel("Observed P(rain)"); ax.set_ylabel("Model P(rain)")
        ax.set_title(variant, fontsize=8); ax.grid(alpha=0.3)

    for i in range(n_models, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    fig.suptitle("Per-Station Wet Day Frequency: Observed vs Model (identity=perfect)", fontsize=11)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "station_wetfreq_scatter.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

args_global = None  # set in main() for access in plot helpers


def main():
    global args_global

    parser = argparse.ArgumentParser(
        description="Compare all trained AR models (Tier 1 + Tier 2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--n_days",        type=int,   default=365)
    parser.add_argument("--n_scenarios",   type=int,   default=50)
    parser.add_argument("--models",        nargs="+",  default=None,
                        help="Specific variants; default: auto-discover ar_*")
    parser.add_argument("--skip_rollouts", action="store_true",
                        help="Tier 1 only (no scenario rollouts)")
    parser.add_argument("--only_tier2",    action="store_true",
                        help="Tier 2 only (skip Tier 1 charts)")
    parser.add_argument("--force_rollouts", action="store_true",
                        help="Re-run rollouts even if cached")
    parser.add_argument("--skip_station_analysis", action="store_true",
                        help="Skip per-station analysis (Tier 1)")
    parser.add_argument("--skip_station_detail", action="store_true",
                        help="Skip per-station individual charts (stations/ dir)")
    parser.add_argument("--data_path",    type=str,   default=SABESP_DATA_PATH)
    parser.add_argument("--output_dir",   type=str,   default="./outputs")

    args = parser.parse_args()
    args_global = args

    out_dir = os.path.join(args.output_dir, "comparison_ar")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "hyperparameter_sensitivity"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compare_ar] Device: {device}")

    # ── Discover models ──
    if args.models:
        tier1_variants = args.models
    else:
        tier1_variants = discover_ar_models(args.output_dir)
    print(f"[compare_ar] Tier 1 variants: {len(tier1_variants)}")

    if not tier1_variants:
        print("[compare_ar] No AR models found. Run training first.")
        return

    # ── Load all metrics ──
    all_metrics = load_all_metrics(tier1_variants, args.output_dir)
    if not all_metrics:
        print("[compare_ar] No metrics.json found.")
        return

    # ── Family assignment ──
    families = {v: get_family(v, args.output_dir) for v in all_metrics}

    # ── Composite score ──
    scores, normalized = compute_composite(all_metrics)

    # Save composite scores
    with open(os.path.join(out_dir, "composite_scores_ar.json"), "w") as f:
        json.dump({"composite_scores": scores, "normalized": normalized}, f, indent=2)

    # ── Tier 1 Reports ──
    if not args.only_tier2:
        print("\n[compare_ar] === Tier 1: Analysis from metrics.json ===")

        report = write_comparison_report(all_metrics, scores, families, out_dir)
        print("\n" + report[:3000])  # print first 3000 chars

        write_family_summary(all_metrics, scores, families, out_dir)
        write_hyperparameter_sensitivity_report(all_metrics, scores, families, out_dir)

        print("\n[compare_ar] Generating Tier 1 charts...")
        plot_composite_bar(scores, families, out_dir)
        plot_radar_by_family(all_metrics, families, out_dir)
        plot_pareto(scores, all_metrics, families, out_dir,
                    x_key="sampling_time_ms", x_label="Sampling Time (ms)",
                    filename="pareto_quality_vs_time.png")
        plot_pareto(scores, all_metrics, families, out_dir,
                    x_key="n_parameters", x_label="Parameter Count",
                    filename="pareto_quality_vs_params.png")
        plot_heatmap(all_metrics, normalized, out_dir)
        plot_family_grouped_bars(all_metrics, families, out_dir)
        plot_hyperparameter_sensitivity(all_metrics, families, scores, out_dir)
        plot_training_loss_overlay(tier1_variants, args.output_dir, out_dir)

        # ── Per-Station Analysis (Tier 1) ──
        if not args.skip_station_analysis:
            print("\n[compare_ar] === Per-Station Analysis ===")
            plot_station_heatmap_ar(
                all_metrics, scores, "wasserstein_per_station",
                "Wasserstein per Station", out_dir, "station_wasserstein_heatmap.png",
                clip_95th=True,
            )
            plot_station_heatmap_ar(
                all_metrics, scores, "wet_day_freq_error_per_station",
                "Wet Day Freq Error per Station", out_dir, "station_wetfreq_heatmap.png",
            )
            plot_station_score_distribution(all_metrics, scores, out_dir)
            plot_best_model_per_station(all_metrics, scores, families, out_dir)
            write_per_station_report(all_metrics, scores, out_dir)
            plot_station_rank_heatmap(all_metrics, scores, families, out_dir)
            if not args.skip_station_detail:
                plot_per_station_detail(all_metrics, scores, families, out_dir)

    # ── Tier 2: Scenario Rollouts ──
    if args.skip_rollouts:
        print("\n[compare_ar] --skip_rollouts set; Tier 2 skipped.")
        print(f"\n[compare_ar] Done. Output: {out_dir}")
        return

    # Discover models with checkpoints
    if args.models:
        ckpt_variants = [v for v in args.models
                         if (Path(args.output_dir) / v / "model.pt").exists()]
    else:
        ckpt_variants = discover_ar_models_with_checkpoints(args.output_dir)
    print(f"\n[compare_ar] === Tier 2: Scenario rollouts for {len(ckpt_variants)} models ===")

    # Load data once
    print("[compare_ar] Loading SABESP data...")
    data_norm, data_raw, mu, std, station_names = load_data(
        data_path=args.data_path,
        normalization_mode="scale_only",
        missing_strategy="impute_station_median",
    )
    print(f"[compare_ar] Data: {data_raw.shape}")

    # Get observation months (for monthly metrics)
    try:
        import pandas as pd
        from data_utils import load_sabesp_daily_precip
        df = load_sabesp_daily_precip()
        obs_months = (df.index.month.values - 1).astype(int)
        obs_months = obs_months[-data_raw.shape[0]:]
    except Exception as e:
        print(f"[warn] Could not extract obs months: {e}; using uniform months")
        obs_months = np.arange(data_raw.shape[0]) % 12

    # Run rollouts
    scenarios_by_model = {}
    tier2_metrics = {}

    for variant in ckpt_variants:
        print(f"\n[compare_ar] Processing: {variant}")
        sc_mm = run_rollout(
            variant=variant,
            output_dir=args.output_dir,
            data_norm=data_norm,
            std=std,
            n_days=args.n_days,
            n_scenarios=args.n_scenarios,
            device=device,
            force=args.force_rollouts,
        )
        if sc_mm is None:
            continue
        scenarios_by_model[variant] = sc_mm

        t2 = compute_tier2_metrics(sc_mm, data_raw, obs_months)
        tier2_metrics[variant] = t2
        print(f"  ACF RMSE={t2['multi_lag_acf_rmse']:.4f} | "
              f"Trans Err={t2['transition_prob_error']:.4f} | "
              f"CV={t2['inter_scenario_cv']:.4f}")

    # Save Tier 2 metrics
    with open(os.path.join(out_dir, "tier2_temporal_metrics.json"), "w") as f:
        json.dump(tier2_metrics, f, indent=2)
    print(f"\n  [report] Tier 2 metrics saved.")

    if not scenarios_by_model:
        print("[compare_ar] No scenarios generated.")
    else:
        print(f"\n[compare_ar] Generating Tier 2 charts ({len(scenarios_by_model)} models)...")
        plot_autocorr_multilag(scenarios_by_model, data_raw, out_dir, families=families)
        plot_spell_length_comparison(scenarios_by_model, data_raw, out_dir, families=families)
        plot_monthly_precip(scenarios_by_model, data_raw, obs_months, out_dir, families=families)
        plot_spread_envelopes(scenarios_by_model, data_raw, out_dir)
        plot_transition_probs(scenarios_by_model, data_raw, out_dir)
        plot_return_period(scenarios_by_model, data_raw, out_dir)

        if not args.skip_station_analysis:
            print("\n[compare_ar] === Tier 2 Per-Station Scatter Plots ===")
            plot_per_station_lag1_scatter(
                scenarios_by_model, data_raw, station_names, families, out_dir,
            )
            plot_per_station_wetfreq_scatter(
                scenarios_by_model, data_raw, station_names, families, out_dir,
            )

    print(f"\n[compare_ar] Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
