"""ar/plots.py — Tier 1 and Tier 2 plotting functions for AR model comparison."""
import os
import json
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.cm as cm
import matplotlib.colors as mcolors

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ar.loader import AR_FAMILIES
from ar.scoring import compute_combined_score
from scoring import QUALITY_METRICS, compute_composite, _metric_array
import metrics as M


def _family_colors(families: list) -> dict:
    """Map each family/variant name to a hex color.

    Known AR_FAMILIES keep their stable tab10 index so colors are consistent
    across all top-level plots. Unknown names (e.g. individual variant names
    used in per-family detail dirs) get extra slots from tab10/tab20.
    """
    ordered = [f for f in AR_FAMILIES if f in families]
    ordered += [f for f in families if f not in AR_FAMILIES]
    n = max(len(ordered), 1)
    cmap_name = "tab10" if n <= 10 else "tab20"
    cmap = matplotlib.colormaps.get_cmap(cmap_name).resampled(n)
    return {fam: matplotlib.colors.to_hex(cmap(i)) for i, fam in enumerate(ordered)}


def plot_composite_bar(
    scores: dict,
    families: dict,
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
    families: dict,
    out_dir: str,
):
    # Group variants by family
    fam_variants: dict = {}
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
    families: dict,
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
    families: dict,
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
    families: dict,
    scores: dict,
    out_dir: str,
    output_dir: str,
    variant_dirs: dict = None,
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
            from ar.loader import _resolve_model_dir
            cfg_path = _resolve_model_dir(v, output_dir, variant_dirs) / "config.json"
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


def plot_training_loss_overlay(variants: list, output_dir: str, out_dir: str,
                               variant_dirs: dict = None):
    fam_colors = _family_colors(AR_FAMILIES)
    fig, ax = plt.subplots(figsize=(12, 6))
    plotted = False

    from ar.loader import _resolve_model_dir
    for variant in variants:
        vdir = _resolve_model_dir(variant, output_dir, variant_dirs)
        history_path = vdir / "training_history.json"
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
        cfg_path = vdir / "config.json"
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
    T = data.shape[0]
    for s in range(data.shape[1]):
        wet = (data[:, s] > threshold).astype(np.uint8)
        # Vectorized RLE: sentinel values at both ends guarantee run boundaries
        padded = np.empty(T + 2, dtype=np.uint8)
        padded[0] = 1 - wet[0]; padded[1:-1] = wet; padded[-1] = 1 - wet[-1]
        trans = np.flatnonzero(np.diff(padded))  # transition positions
        lengths = np.diff(trans)                 # run lengths
        labels = wet[trans[:-1]]                 # 1=wet, 0=dry for each run
        valid = lengths <= max_len
        np.add.at(wet_c, lengths[valid & (labels == 1)] - 1, 1)
        np.add.at(dry_c, lengths[valid & (labels == 0)] - 1, 1)
    wet_c /= max(wet_c.sum(), 1); dry_c /= max(dry_c.sum(), 1)
    return {"wet": wet_c, "dry": dry_c}


def _batch_autocorr(data: np.ndarray, max_lag: int = 30) -> np.ndarray:
    """data: (n_sc, T, S) → (n_sc, max_lag) mean ACF over stations, vectorized."""
    n_sc, T, S = data.shape
    mu = data.mean(axis=1, keepdims=True)       # (n_sc, 1, S)
    centered = data - mu                         # (n_sc, T, S)
    var = np.maximum((centered ** 2).mean(axis=1), 1e-10)  # (n_sc, S)
    result = np.zeros((n_sc, max_lag))
    for lag in range(1, max_lag + 1):
        prod = centered[:, :T - lag, :] * centered[:, lag:, :]  # (n_sc, T-lag, S)
        acf_per_station = prod.mean(axis=1) / var                # (n_sc, S)
        result[:, lag - 1] = acf_per_station.mean(axis=1)        # (n_sc,)
    return result


def _monthly_means(data: np.ndarray, months: np.ndarray) -> np.ndarray:
    """data: (T, S), months: (T,) → (12,) mean over stations"""
    return np.array([data[months == m].mean() if (months == m).any() else 0.0
                     for m in range(12)])


def plot_autocorr_multilag(
    scenarios_by_model: dict,
    data_raw: np.ndarray,
    out_dir: str,
    families: dict = None,
    max_lag: int = 30,
):
    lags = np.arange(1, max_lag + 1)
    obs_acf = _mean_autocorr_nd(data_raw, max_lag)
    _fams = list(dict.fromkeys((families or {}).values())) or AR_FAMILIES
    fam_colors = _family_colors(_fams)

    def _get_fam(variant):
        if families is not None:
            return families.get(variant, variant)
        return variant

    # ── Detail chart (all individual models) ──
    fig, ax = plt.subplots(figsize=(11, 5))
    for variant, sc_mm in scenarios_by_model.items():
        sc_acfs = _batch_autocorr(sc_mm, max_lag)
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
        sc_acfs = _batch_autocorr(all_sc, max_lag)
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
    families: dict = None,
    max_len: int = 20,
):
    obs_sp = _spell_dist(data_raw, max_len=max_len)
    lens = np.arange(1, max_len + 1)
    _fams = list(dict.fromkeys((families or {}).values())) or AR_FAMILIES
    fam_colors = _family_colors(_fams)

    def _get_fam(variant):
        if families is not None:
            return families.get(variant, variant)
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
    families: dict = None,
    start_month: int = 0,
):
    obs_mm = _monthly_means(data_raw, obs_months)
    obs_var = np.array([data_raw[obs_months == m].var() if (obs_months == m).any()
                        else 0.0 for m in range(12)])
    month_labels = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    x = np.arange(12)
    _fams = list(dict.fromkeys((families or {}).values())) or AR_FAMILIES
    fam_colors = _family_colors(_fams)

    def _get_fam(variant):
        if families is not None:
            return families.get(variant, variant)
        return variant

    _MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    def _sc_month_indices(T: int) -> np.ndarray:
        """Assign calendar month (0–11) to each of T scenario days starting at start_month."""
        months, m, d = [], start_month, 0
        for _ in range(T):
            months.append(m)
            d += 1
            if d >= _MONTH_DAYS[m % 12]:
                m = (m + 1) % 12
                d = 0
        return np.array(months)

    def _sc_monthly_mean(sc_mm, sc_months):
        """sc_mm: (n_sc, T, S), sc_months: (T,) → (12,) global mean per month."""
        return np.array([sc_mm[:, sc_months == m, :].mean() if (sc_months == m).any()
                         else 0.0 for m in range(12)])

    def _sc_monthly_var(sc_mm, sc_months):
        """sc_mm: (n_sc, T, S), sc_months: (T,) → (12,) global var per month."""
        return np.array([sc_mm[:, sc_months == m, :].var() if (sc_months == m).any()
                         else 0.0 for m in range(12)])

    def _sc_monthly_means_per_sc(all_sc, sc_months):
        """all_sc: (n_sc, T, S) → (n_sc, 12) mean per scenario per month."""
        return np.array([
            all_sc[:, sc_months == m, :].mean(axis=(1, 2)) if (sc_months == m).any()
            else np.zeros(all_sc.shape[0])
            for m in range(12)
        ]).T  # (n_sc, 12)

    def _sc_monthly_vars_per_sc(all_sc, sc_months):
        """all_sc: (n_sc, T, S) → (n_sc, 12) var per scenario per month."""
        return np.array([
            all_sc[:, sc_months == m, :].var(axis=(1, 2)) if (sc_months == m).any()
            else np.zeros(all_sc.shape[0])
            for m in range(12)
        ]).T  # (n_sc, 12)

    # ── Detail chart (all individual models) ──
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    ax1.bar(x, obs_mm, width=0.35, color="black", alpha=0.6, label="Observed")
    ax2.bar(x, obs_var, width=0.35, color="black", alpha=0.6, label="Observed")
    for variant, sc_mm in scenarios_by_model.items():
        sc_months_v = _sc_month_indices(sc_mm.shape[1])
        color = fam_colors.get(_get_fam(variant), "gray")
        ax1.plot(x, _sc_monthly_mean(sc_mm, sc_months_v), "o--", lw=1, alpha=0.7, color=color, label=variant)
        ax2.plot(x, _sc_monthly_var(sc_mm, sc_months_v), "o--", lw=1, alpha=0.7, color=color, label=variant)
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
        sc_months_f = _sc_month_indices(all_sc.shape[1])   # compute once per family
        means = _sc_monthly_means_per_sc(all_sc, sc_months_f)   # (n_sc, 12)
        vars_ = _sc_monthly_vars_per_sc(all_sc, sc_months_f)    # (n_sc, 12)
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
        sc_months_f = _sc_month_indices(all_sc.shape[1])
        vars_ = _sc_monthly_vars_per_sc(all_sc, sc_months_f)    # (n_sc, 12)
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
    output_dir: str = None,
    variant_dirs: dict = None,
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

        fam = variant
        if output_dir is not None or variant_dirs is not None:
            from ar.loader import _resolve_model_dir
            cfg_path = _resolve_model_dir(variant, output_dir, variant_dirs) / "config.json"
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
    output_dir: str = None,
    variant_dirs: dict = None,
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
        fam = variant
        if output_dir is not None or variant_dirs is not None:
            from ar.loader import _resolve_model_dir
            cfg_path = _resolve_model_dir(variant, output_dir, variant_dirs) / "config.json"
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
    families: dict = None,
):
    """Annual max daily precipitation empirical CDF (return period proxy)."""
    _fams = list(dict.fromkeys((families or {}).values())) or AR_FAMILIES
    fam_colors = _family_colors(_fams)

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
        if families is not None:
            fam = families.get(variant, variant)
        else:
            fam = variant
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


def plot_rx5day_distribution(
    scenarios_by_model: dict,
    data_raw: np.ndarray,
    out_dir: str,
    families: dict = None,
    window: int = 5,
):
    """
    CDF empírica do acumulado máximo em `window` dias consecutivos (Rxnday).

    Comparar a distribuição do Rx5day observado vs. cenários mostra se o modelo
    reproduz a frequência e magnitude de eventos extremos multi-dia (cheias).
    """
    _fams = list(dict.fromkeys((families or {}).values())) or AR_FAMILIES
    fam_colors = _family_colors(_fams)

    def _rx_pooled(data: np.ndarray) -> np.ndarray:
        """Todas as somas rolantes de `window` dias para todas as estações."""
        parts = []
        for s in range(data.shape[-1]):
            parts.append(np.convolve(data[:, s], np.ones(window), mode="valid"))
        return np.concatenate(parts)

    # Observed distribution (pool over stations)
    obs_rx = _rx_pooled(data_raw)

    # ── family-aggregated plot ──
    fig, ax = plt.subplots(figsize=(10, 5))

    obs_sorted = np.sort(obs_rx)
    p_obs = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1)
    ax.plot(obs_sorted, p_obs, "k-", lw=2.5, label="Observed", zorder=10)

    _MAX_CDF = 3_000
    seen_fams = set()
    for variant, sc_mm in scenarios_by_model.items():
        # Vectorized rolling sum via cumsum across all scenarios and stations
        cs_sc = np.cumsum(sc_mm, axis=1)  # (n_sc, T, S)
        sc_rx = (cs_sc[:, window:, :] - cs_sc[:, :-window, :]).ravel()
        sc_sorted = np.sort(sc_rx)
        p_sc = np.arange(1, len(sc_sorted) + 1) / (len(sc_sorted) + 1)
        idx = np.round(np.linspace(0, len(sc_sorted) - 1, min(_MAX_CDF, len(sc_sorted)))).astype(int)

        fam = (families or {}).get(variant, variant)
        color = fam_colors.get(fam, "gray")
        label = fam if fam not in seen_fams else None
        seen_fams.add(fam)
        ax.plot(sc_sorted[idx], p_sc[idx], lw=1.2, alpha=0.7, color=color, label=label)

    ax.set_xlabel(f"Rx{window}day — Acumulado {window} dias (mm)")
    ax.set_ylabel("CDF Empírica")
    ax.set_title(f"Distribuição do Rx{window}day — Cenários vs. Observado")
    handles = [mpatches.Patch(color="k", label="Observed")]
    handles += [mpatches.Patch(color=fam_colors.get(f, "gray"), label=f) for f in AR_FAMILIES]
    ax.legend(handles=handles, fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"rx{window}day_distribution.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")

    # ── detail plot (all individual models) ──
    if len(scenarios_by_model) <= 20:
        fig, ax = plt.subplots(figsize=(11, 5))
        ax.plot(obs_sorted, p_obs, "k-", lw=2.5, label="Observed", zorder=10)

        for variant, sc_mm in scenarios_by_model.items():
            cs_sc = np.cumsum(sc_mm, axis=1)
            sc_rx = (cs_sc[:, window:, :] - cs_sc[:, :-window, :]).ravel()
            sc_sorted = np.sort(sc_rx)
            p_sc = np.arange(1, len(sc_sorted) + 1) / (len(sc_sorted) + 1)
            idx = np.round(np.linspace(0, len(sc_sorted) - 1, min(_MAX_CDF, len(sc_sorted)))).astype(int)
            fam = (families or {}).get(variant, variant)
            color = fam_colors.get(fam, "gray")
            ax.plot(sc_sorted[idx], p_sc[idx], lw=1, alpha=0.8, color=color, label=variant)

        ax.set_xlabel(f"Rx{window}day — Acumulado {window} dias (mm)")
        ax.set_ylabel("CDF Empírica")
        ax.set_title(f"Rx{window}day — Todos os Modelos (detalhe)")
        ax.legend(fontsize=6, ncol=3, loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        out_path2 = os.path.join(out_dir, f"rx{window}day_distribution_detail.png")
        plt.savefig(out_path2, dpi=130, bbox_inches="tight")
        plt.close()
        print(f"  [plot] {out_path2}")


def plot_rxnday_multi(
    scenarios_by_model: dict,
    data_raw: np.ndarray,
    out_dir: str,
    families: dict = None,
    windows: tuple = (3, 5, 10, 30),
):
    """
    Grid 2x2 de CDFs empíricas para acumulados Rx3, Rx5, Rx10, Rx30.

    Permite comparar como cada modelo escala com a janela de acumulação —
    um bom modelo deve apresentar erro consistente em todas as janelas.
    """
    _fams = list(dict.fromkeys((families or {}).values())) or AR_FAMILIES
    fam_colors = _family_colors(_fams)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    # Precompute observed cumsum once for all windows
    cs_obs = np.cumsum(data_raw, axis=0)  # (T, S)

    for ax, window in zip(axes, windows):
        # Observed distribution — vectorized rolling sum via cumsum
        obs_rx = (cs_obs[window:, :] - cs_obs[:-window, :]).ravel()
        obs_sorted = np.sort(obs_rx)
        p_obs = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1)
        ax.plot(obs_sorted, p_obs, "k-", lw=2, label="Observed", zorder=10)

        _MAX_CDF = 3_000
        seen_fams = set()
        for variant, sc_mm in scenarios_by_model.items():
            cs_sc = np.cumsum(sc_mm, axis=1)                               # (n_sc, T, S)
            sc_rx = (cs_sc[:, window:, :] - cs_sc[:, :-window, :]).ravel()
            sc_sorted = np.sort(sc_rx)
            p_sc = np.arange(1, len(sc_sorted) + 1) / (len(sc_sorted) + 1)
            idx = np.round(np.linspace(0, len(sc_sorted) - 1, min(_MAX_CDF, len(sc_sorted)))).astype(int)
            fam = (families or {}).get(variant, variant)
            color = fam_colors.get(fam, "gray")
            label = fam if fam not in seen_fams else None
            seen_fams.add(fam)
            ax.plot(sc_sorted[idx], p_sc[idx], lw=1.2, alpha=0.7, color=color, label=label)

        ax.set_title(f"Rx{window}day (acumulado {window} dias)")
        ax.set_xlabel("mm")
        ax.set_ylabel("CDF")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=6, ncol=2, loc="lower right")

    fig.suptitle("Distribuição Rxnday — Cenários vs. Observado", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "rxnday_multi_cdf.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def plot_seasonal_accumulation(
    scenarios_by_model: dict,
    data_raw: np.ndarray,
    obs_months: np.ndarray,
    out_dir: str,
    families: dict = None,
    wet_months: tuple = (10, 11, 0, 1, 2, 3),
    start_month: int = 0,
):
    """
    CDF dos totais mensais da estação chuvosa (Nov–Abr) e seca (Mai–Out).

    Avalia se o modelo reproduz corretamente o volume sazonal acumulado —
    crítico para gestão de reservatórios.
    """
    dry_months = tuple(m for m in range(12) if m not in wet_months)
    _fams = list(dict.fromkeys((families or {}).values())) or AR_FAMILIES
    fam_colors = _family_colors(_fams)

    def _monthly_totals_1d(data2d: np.ndarray, months_mask: np.ndarray, season: tuple) -> np.ndarray:
        """Per-year-month totals pooled across all stations."""
        parts = []
        for m in season:
            indices = np.where(months_mask == m)[0]
            if len(indices) == 0:
                continue
            breaks = np.where(np.diff(indices) > 1)[0] + 1
            for run in np.split(indices, breaks):
                parts.extend(data2d[run].sum(axis=0).tolist())
        return np.array(parts) if parts else np.array([])

    obs_wet = _monthly_totals_1d(data_raw, obs_months, wet_months)
    obs_dry = _monthly_totals_1d(data_raw, obs_months, dry_months)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = ["Estação Chuvosa (Nov–Abr)", "Estação Seca (Mai–Out)"]
    obs_data = [obs_wet, obs_dry]

    for ax, title, obs_rx in zip(axes, titles, obs_data):
        if len(obs_rx) == 0:
            ax.set_title(title + " (sem dados)")
            continue
        obs_sorted = np.sort(obs_rx)
        p_obs = np.arange(1, len(obs_sorted) + 1) / (len(obs_sorted) + 1)
        ax.plot(obs_sorted, p_obs, "k-", lw=2.5, label="Observed", zorder=10)

    n_sc_months = None
    seen_fams = [set(), set()]
    for variant, sc_mm in scenarios_by_model.items():
        T = sc_mm.shape[1]
        if n_sc_months is None:
            MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
            _ml, _mm, _md = [], start_month, 0
            for _tt in range(T):
                _ml.append(_mm)
                _md += 1
                if _md >= MONTH_DAYS[_mm % 12]:
                    _mm = (_mm + 1) % 12
                    _md = 0
            n_sc_months = np.array(_ml)
        fam = (families or {}).get(variant, variant)
        color = fam_colors.get(fam, "gray")

        for ax_idx, (ax, season) in enumerate(zip(axes, [wet_months, dry_months])):
            sc_parts = []
            for i in range(sc_mm.shape[0]):
                sc_parts.append(_monthly_totals_1d(sc_mm[i], n_sc_months, season))
            sc_rx = np.concatenate([p for p in sc_parts if len(p) > 0])
            if len(sc_rx) == 0:
                continue
            sc_sorted = np.sort(sc_rx)
            p_sc = np.arange(1, len(sc_sorted) + 1) / (len(sc_sorted) + 1)
            label = fam if fam not in seen_fams[ax_idx] else None
            seen_fams[ax_idx].add(fam)
            ax.plot(sc_sorted, p_sc, lw=1.2, alpha=0.7, color=color, label=label)

    for ax, title in zip(axes, titles):
        ax.set_title(title)
        ax.set_xlabel("Total mensal (mm)")
        ax.set_ylabel("CDF")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7)

    fig.suptitle("Acumulado Sazonal — Cenários vs. Observado", fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "seasonal_accumulation.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def plot_exceedance_frequency(
    scenarios_by_model: dict,
    data_raw: np.ndarray,
    out_dir: str,
    families: dict = None,
    windows: tuple = (3, 5, 10, 30),
):
    """
    Curva de frequência de excedância P(Rxnday > x) para múltiplas janelas.

    Leitura direta de risco: para um acumulado de X mm em N dias, qual a
    probabilidade de ocorrência? Modelos que subestimam a cauda aparecem
    claramente abaixo da curva observada no eixo log-y.
    """
    _fams = list(dict.fromkeys((families or {}).values())) or AR_FAMILIES
    fam_colors = _family_colors(_fams)
    linestyles = ["-", "--", "-.", ":"]

    fig, ax = plt.subplots(figsize=(11, 6))

    # Precompute observed cumsum once for all windows
    cs_obs = np.cumsum(data_raw, axis=0)  # (T, S)

    # Determine x-axis range from observed data across all windows
    all_obs_rx = np.concatenate([
        (cs_obs[w:, :] - cs_obs[:-w, :]).ravel() for w in windows
    ])
    x_max = np.percentile(all_obs_rx, 99.5)
    thresholds = np.linspace(0, x_max, 300)

    # Observed curves (one per window, all black, different linestyle)
    for window, ls in zip(windows, linestyles):
        obs_rx = (cs_obs[window:, :] - cs_obs[:-window, :]).ravel()
        obs_rx_sorted = np.sort(obs_rx)
        n_obs = len(obs_rx_sorted)
        exceedance = np.clip(
            (n_obs - np.searchsorted(obs_rx_sorted, thresholds)) / max(n_obs, 1),
            1e-6, 1,
        )
        ax.plot(thresholds, exceedance, color="black", lw=2, ls=ls,
                label=f"Obs Rx{window}day", zorder=10)

    # Model curves — one line per (family × window)
    seen = set()
    for variant, sc_mm in scenarios_by_model.items():
        fam = (families or {}).get(variant, variant)
        color = fam_colors.get(fam, "gray")
        cs_sc = np.cumsum(sc_mm, axis=1)  # (n_sc, T, S) — precomputed once per variant
        for window, ls in zip(windows, linestyles):
            sc_rx = (cs_sc[:, window:, :] - cs_sc[:, :-window, :]).ravel()
            sc_rx_sorted = np.sort(sc_rx)
            n_sc_rx = len(sc_rx_sorted)
            exceedance = np.clip(
                (n_sc_rx - np.searchsorted(sc_rx_sorted, thresholds)) / max(n_sc_rx, 1),
                1e-6, 1,
            )
            key = (fam, window)
            label = f"{fam} Rx{window}day" if key not in seen else None
            seen.add(key)
            ax.plot(thresholds, exceedance, color=color, lw=1, ls=ls, alpha=0.6, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("Limiar de acumulado (mm)")
    ax.set_ylabel("P(Rxnday > limiar)")
    ax.set_title("Frequência de Excedância — Múltiplas Janelas de Acumulação")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=6, ncol=3, loc="upper right")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "exceedance_frequency.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  [plot] {out_path}")


def plot_family_detail_dirs(
    scenarios_by_model: dict,
    data_raw: np.ndarray,
    obs_months: np.ndarray,
    all_metrics: dict,
    tier2_metrics: dict,
    families: dict,
    out_dir: str,
    sc_start_month: int = 0,
    top_n: int = None,
    max_per_chart: int = 0,
):
    """
    Generate per-family detail subdirs: comparison_ar/families/<family>/.

    Each subdir contains the full Tier 2 plot suite for variants of that family.

    Args:
        top_n: if set, only show top-N variants per family ranked by combined score.
        max_per_chart: if >0, split variants into groups of this size. Each group
            gets its own part_01/, part_02/, ... subdir within the family dir.
            When only one group, no subdirs are created (identical to original behavior).
    """
    from collections import defaultdict
    import time as _time
    by_family: dict = defaultdict(dict)

    for v, sc in scenarios_by_model.items():
        by_family[families.get(v, v)][v] = sc

    for fam, fam_scenarios in sorted(by_family.items()):
        if not fam_scenarios:
            continue
        fam_dir = os.path.join(out_dir, "families", fam)
        os.makedirs(fam_dir, exist_ok=True)

        # Composite bar ranked by combined score (family only) — always uses all variants
        fam_metrics = {v: all_metrics[v] for v in fam_scenarios if v in all_metrics}
        fam_combined = {}
        if fam_metrics:
            fam_t2 = {v: tier2_metrics.get(v, {}) for v in fam_scenarios}
            fam_combined, _ = compute_combined_score(fam_metrics, fam_t2)
            fam_families_map = {v: fam for v in fam_scenarios}
            plot_composite_bar(fam_combined, fam_families_map, fam_dir)

        # Sort by score, optionally limit to top-N
        sorted_variants = sorted(fam_scenarios.keys(), key=lambda v: fam_combined.get(v, 1.0))
        if top_n is not None:
            sorted_variants = sorted_variants[:top_n]

        print(f"\n  [family] {fam} ({len(sorted_variants)}/{len(fam_scenarios)} variants) → {fam_dir}")

        # Split into chunks if max_per_chart is set
        if max_per_chart > 0 and len(sorted_variants) > max_per_chart:
            chunks = [sorted_variants[i:i + max_per_chart]
                      for i in range(0, len(sorted_variants), max_per_chart)]
        else:
            chunks = [sorted_variants]

        def _tplot(label, fn):
            t = _time.perf_counter()
            fn()
            print(f"    [{label}] {_time.perf_counter()-t:.2f}s", flush=True)

        _t_fam = _time.perf_counter()
        for ci, chunk in enumerate(chunks):
            chunk_dir = os.path.join(fam_dir, f"part_{ci+1:02d}") if len(chunks) > 1 else fam_dir
            if len(chunks) > 1:
                os.makedirs(chunk_dir, exist_ok=True)
                print(f"    [part {ci+1}/{len(chunks)}] {len(chunk)} variants → {chunk_dir}")
            chunk_sc = {v: fam_scenarios[v] for v in chunk}
            fam_fam_map = {v: v for v in chunk_sc}
            _tplot("autocorr",   lambda d=chunk_dir, s=chunk_sc, m=fam_fam_map: plot_autocorr_multilag(s, data_raw, d, families=m))
            _tplot("spell",      lambda d=chunk_dir, s=chunk_sc, m=fam_fam_map: plot_spell_length_comparison(s, data_raw, d, families=m))
            _tplot("monthly",    lambda d=chunk_dir, s=chunk_sc, m=fam_fam_map: plot_monthly_precip(s, data_raw, obs_months, d, families=m, start_month=sc_start_month))
            _tplot("return_per", lambda d=chunk_dir, s=chunk_sc, m=fam_fam_map: plot_return_period(s, data_raw, d, families=m))
            _tplot("rxnday",     lambda d=chunk_dir, s=chunk_sc, m=fam_fam_map: plot_rxnday_multi(s, data_raw, d, families=m))
            _tplot("seasonal",   lambda d=chunk_dir, s=chunk_sc, m=fam_fam_map: plot_seasonal_accumulation(s, data_raw, obs_months, d, families=m))
            _tplot("exceedance", lambda d=chunk_dir, s=chunk_sc, m=fam_fam_map: plot_exceedance_frequency(s, data_raw, d, families=m))
        print(f"  [family_plots_total] {_time.perf_counter()-_t_fam:.2f}s", flush=True)
