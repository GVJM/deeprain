"""ar/reports.py — Text report generation for AR model comparison."""
import csv
import os
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ar.loader import AR_FAMILIES
from ar.rollout import TIER2_METRICS
from scoring import QUALITY_METRICS, _metric_array


def write_comparison_report(
    all_metrics: dict,
    scores: dict,
    families: dict,
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
    families: dict,
    out_dir: str,
):
    fam_data: dict = {}
    for v, s in scores.items():
        fam = families.get(v, "unknown")
        fam_data.setdefault(fam, []).append((v, s))

    lines = ["AR Family Summary", "=" * 80, ""]
    lines.append(f"{'Family':<25} {'Count':>6} {'Best':>10} {'Mean':>10} {'Worst':>10}  Best Variant")
    lines.append("-" * 80)

    for fam in fam_data:
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
    families: dict,
    out_dir: str,
    output_dir: str,
):
    hp_keys = ["hidden_size", "n_layers", "n_coupling", "rnn_hidden", "n_steps",
               "latent_size", "gru_hidden"]
    lines = ["AR Hyperparameter Sensitivity Report", "=" * 100, ""]

    for fam in set(families.values()):
        variants = sorted([v for v in all_metrics if families.get(v) == fam])
        if not variants:
            continue
        lines.append(f"\n{'─'*70}")
        lines.append(f"Family: {fam}  ({len(variants)} variants)")
        lines.append(f"{'-'*70}")
        lines.append(f"  {'Variant':<35} {'Composite':>10}" + "".join(f" {hp:>14}" for hp in hp_keys))
        lines.append(f"  {'-'*35} {'-'*10}" + "".join(f" {'-'*14}" for _ in hp_keys))

        for v in sorted(variants, key=lambda x: scores.get(x, 1.0)):
            cfg_path = Path(output_dir) / v / "config.json"
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


def write_metrics_csv(
    all_metrics: dict,
    tier2_metrics: dict,
    scores: dict,
    families: dict,
    out_dir: str,
):
    """Export all Tier1 + Tier2 metrics to CSV sorted by composite score."""
    tier1_keys = [k for k, _, _ in QUALITY_METRICS]
    tier2_keys = [k for k, _, _ in TIER2_METRICS]

    variants = sorted(scores.keys(), key=lambda v: scores.get(v, 1.0))
    fieldnames = ["variant", "family", "composite_score"] + tier1_keys + tier2_keys
    rows = []
    for v in variants:
        m = all_metrics.get(v, {})
        t2 = (tier2_metrics or {}).get(v, {})
        row = {"variant": v, "family": families.get(v, "?"),
               "composite_score": scores.get(v, "")}
        for k in tier1_keys:
            row[k] = m.get(k, "")
        for k in tier2_keys:
            row[k] = t2.get(k, "")
        rows.append(row)

    out_path = os.path.join(out_dir, "metrics_comparison.csv")
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"  [report] {out_path}")
