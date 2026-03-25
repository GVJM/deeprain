"""ar/station.py — Per-station analysis functions for AR model comparison."""
import os
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ar.loader import OUTLIER_STATION, AR_FAMILIES
from ar.plots import _family_colors
from scoring import QUALITY_METRICS, _metric_array
import metrics as M


def _load_station_names(data_path: str) -> list:
    """Load data just to retrieve station names (fast, ~1 sec)."""
    from data_utils import load_data
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
