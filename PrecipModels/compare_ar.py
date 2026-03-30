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
    station_wasserstein_heatmap.png      ← models x stations heatmap, colorbar clipped at 95th pct (Tier 1)
    station_rank_wasserstein_heatmap.png ← rank-based heatmap (1=best) — no outlier scale bias (Tier 1)
    station_wetfreq_heatmap.png          ← models x stations heatmap (Tier 1)
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
    rx5day_distribution.png                  ← CDF Rx5day cenários vs observado (Tier 2)
    rx5day_distribution_detail.png           ← detalhe por modelo individual (Tier 2)
    rxnday_multi_cdf.png                     ← grid 2x2 CDFs Rx3/5/10/30day (Tier 2)
    seasonal_accumulation.png                ← totais mensais estação chuvosa vs seca (Tier 2)
    exceedance_frequency.png                 ← P(Rxnday > x) multi-janela, escala log (Tier 2)
    combined_scores_ar.json                  ← combined Tier1+Tier2 scores
    families/
        <family>/composite_bar.png           ← ranking de todas as variantes da família
        <family>/acf_multilag.png            ← ACF de todas as variantes da família
        <family>/rxnday_multi_cdf.png        ← CDFs Rxnday de todas as variantes
        <family>/seasonal_accumulation.png
        <family>/exceedance_frequency.png
        <family>/...                         ← suite Tier 2 completa por família
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
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from data_utils import load_data, SABESP_DATA_PATH
from ar import (
    AR_FAMILIES, OUTLIER_STATION, OUTLIER_MODELS,
    TIER2_METRICS, COMBINED_TIER2_METRICS,
    _resolve_model_dir,
    discover_ar_models, discover_ar_models_with_checkpoints, get_family,
    load_all_metrics, _load_config, load_ar_model,
    recompute_tier1_metrics, run_rollout, compute_tier2_metrics,
    compute_combined_score, select_top_n_per_family,
    _family_colors, plot_composite_bar, plot_radar_by_family, plot_pareto,
    plot_heatmap, plot_family_grouped_bars, plot_hyperparameter_sensitivity,
    plot_training_loss_overlay, _group_scenarios_by_family, _mean_autocorr_nd,
    _spell_dist, _monthly_means, plot_autocorr_multilag, plot_spell_length_comparison,
    plot_monthly_precip, plot_spread_envelopes, plot_transition_probs, plot_return_period,
    plot_rx5day_distribution, plot_rxnday_multi, plot_seasonal_accumulation,
    plot_exceedance_frequency, plot_family_detail_dirs,
    _load_station_names, plot_station_heatmap_ar, plot_station_score_distribution,
    plot_best_model_per_station, write_per_station_report,
    plot_station_rank_heatmap, plot_per_station_detail,
    plot_per_station_lag1_scatter, plot_per_station_wetfreq_scatter,
    write_comparison_report, write_family_summary,
    write_hyperparameter_sensitivity_report, write_metrics_csv,
)
from scoring import QUALITY_METRICS, _metric_array, compute_composite


args_global = None  # kept for backward compatibility (e.g. tests importing _rollout_worker)


def _rollout_worker(args_tuple) -> tuple:
    """
    Worker function for ProcessPoolExecutor parallel rollouts.

    Handles the full per-model pipeline: load → rollout → metrics → save cache.
    Returns (variant, tier2_dict) or (variant, None) on failure.

    IMPORTANT: Must not call any function that reads the module-level `args_global`.
    On Windows spawn, each worker process imports the module fresh and args_global
    is None. All required data must be received via args_tuple.
    """
    (variant, model_dir_str, data_norm, std, data_raw, obs_months,
     n_days, n_scenarios, device_str, force) = args_tuple

    from pathlib import Path as _Path
    device = torch.device(device_str)
    _vdirs = {variant: _Path(model_dir_str)}
    try:
        sc_mm = run_rollout(
            variant=variant,
            output_dir=model_dir_str,
            data_norm=data_norm,
            std=std,
            n_days=n_days,
            n_scenarios=n_scenarios,
            device=device,
            force=force,
            variant_dirs=_vdirs,
        )
        if sc_mm is None:
            return variant, None
        t2 = compute_tier2_metrics(sc_mm, data_raw, obs_months)
        return variant, t2
    except Exception:
        import traceback
        print(f"  [worker] {variant}: failed\n{traceback.format_exc()}", flush=True)
        return variant, None


def main():
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
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Number of parallel rollout workers (default: 1 = sequential). "
                             "On a single GPU, workers share VRAM — start with --n_workers 2 "
                             "and reduce if OOM. On CPU-only, can match physical core count.")
    parser.add_argument("--skip_station_analysis", action="store_true",
                        help="Skip per-station analysis (Tier 1)")
    parser.add_argument("--skip_station_detail", action="store_true",
                        help="Skip per-station individual charts (stations/ dir)")
    parser.add_argument("--data_path",    type=str,   default=SABESP_DATA_PATH)
    parser.add_argument("--output_dir",   type=str,   default="./outputs")
    parser.add_argument("--top_n_per_family", type=int, default=None,
                        help="Keep only top N variants per family in top-level plots, "
                             "ranked by combined Tier1+Tier2 score. Also used as default "
                             "for --top_n_family_detail if that flag is not set.")
    parser.add_argument("--recompute_tier1", action="store_true",
                        help="Re-evaluate Tier 1 metrics with --n_samples and overwrite "
                             "metrics.json in-place before generating charts")
    parser.add_argument("--n_samples", type=int, default=1000,
                        help="Sample count for --recompute_tier1 (default: 1000)")
    parser.add_argument("--comparison_folder", type=str, default="comparison_ar",
                        help="Subfolder in output_dir for this comparison run; allows multiple comparisons to coexist")
    parser.add_argument("--families", nargs="+", default=None,
                        help="Optional filter to only include certain model families (e.g. 'ar_vae')")
    parser.add_argument("--max_series_per_chart", type=int, default=0,
                        help="Max variants per detail chart in family dirs; 0 = no limit. "
                             "When exceeded, splits into part_01/, part_02/, ... subdirs.")
    parser.add_argument("--top_n_family_detail", type=int, default=None,
                        help="Limit family detail dirs to top-N variants by score. "
                             "Defaults to --top_n_per_family if not set.")

    args = parser.parse_args()

    if args.n_workers < 1:
        parser.error("--n_workers must be >= 1")

    out_dir = os.path.join(args.output_dir, args.comparison_folder)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "hyperparameter_sensitivity"), exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[compare_ar] Device: {device}")

    # ── Discover models (recursive) ──
    all_variant_dirs = discover_ar_models(args.output_dir, args.families)
    if args.models:
        missing = [m for m in args.models if m not in all_variant_dirs]
        if missing:
            print(f"[compare_ar] [warn] --models not found under {args.output_dir}: {missing}")
        variant_dirs = {m: all_variant_dirs[m] for m in args.models if m in all_variant_dirs}
    else:
        variant_dirs = all_variant_dirs
    tier1_variants = list(variant_dirs.keys())
    print(f"[compare_ar] Tier 1 variants: {len(tier1_variants)}")

    if not tier1_variants:
        print("[compare_ar] No AR models found. Run training first.")
        return

    # ── Optional Tier 1 re-evaluation ──
    if args.recompute_tier1:
        recompute_tier1_metrics(
            tier1_variants,
            output_dir=args.output_dir,
            data_path=args.data_path,
            n_samples=args.n_samples,
            device=device,
            variant_dirs=variant_dirs,
        )

    # ── Load all metrics ──
    all_metrics = load_all_metrics(tier1_variants, args.output_dir, variant_dirs=variant_dirs)
    if not all_metrics:
        print("[compare_ar] No metrics.json found.")
        return

    # ── Family assignment ──
    families = {v: get_family(v, args.output_dir, variant_dirs=variant_dirs)
                for v in all_metrics}

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
        print(f"\n Faimilies: {set(families.values())}")
        write_family_summary(all_metrics, scores, families, out_dir)
        write_hyperparameter_sensitivity_report(all_metrics, scores, families, out_dir,
                                                output_dir=args.output_dir,
                                                variant_dirs=variant_dirs)
        write_metrics_csv(all_metrics, {}, scores, families, out_dir)

        # Filter Tier 1 variant-level plots if --top_n_per_family set
        # (uses Tier 1 composite score — Tier 2 not yet available here)
        if args.top_n_per_family is not None:
            t1_plot_variants = select_top_n_per_family(scores, families, args.top_n_per_family)
            t1_plot_metrics = {v: all_metrics[v] for v in t1_plot_variants if v in all_metrics}
            t1_plot_normalized = {v: normalized[v] for v in t1_plot_variants if v in normalized}
            t1_plot_scores = {v: scores[v] for v in t1_plot_variants if v in scores}
            print(f"[compare_ar] Tier 1 plots: {len(t1_plot_variants)} variants "
                  f"(top {args.top_n_per_family} per family by Tier 1 score)")
        else:
            t1_plot_variants = list(all_metrics.keys())
            t1_plot_metrics = all_metrics
            t1_plot_normalized = normalized
            t1_plot_scores = scores

        print("\n[compare_ar] Generating Tier 1 charts...")
        plot_composite_bar(t1_plot_scores, families, out_dir)
        plot_radar_by_family(t1_plot_metrics, families, out_dir)
        plot_pareto(t1_plot_scores, t1_plot_metrics, families, out_dir,
                    x_key="sampling_time_ms", x_label="Sampling Time (ms)",
                    filename="pareto_quality_vs_time.png")
        plot_pareto(t1_plot_scores, t1_plot_metrics, families, out_dir,
                    x_key="n_parameters", x_label="Parameter Count",
                    filename="pareto_quality_vs_params.png")
        plot_heatmap(t1_plot_metrics, t1_plot_normalized, out_dir)
        # plot_family_grouped_bars(t1_plot_metrics, families, out_dir)
        plot_hyperparameter_sensitivity(t1_plot_metrics, families, t1_plot_scores, out_dir,
                                        output_dir=args.output_dir,
                                        variant_dirs=variant_dirs)
        # plot_training_loss_overlay(t1_plot_variants, args.output_dir, out_dir)

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

    # Discover models with checkpoints (reuse already-discovered variant_dirs)
    all_ckpt_dirs = discover_ar_models_with_checkpoints(args.output_dir)
    if args.models:
        ckpt_variant_dirs = {m: all_ckpt_dirs[m] for m in args.models if m in all_ckpt_dirs}
    else:
        ckpt_variant_dirs = all_ckpt_dirs
    ckpt_variants = list(ckpt_variant_dirs.keys())
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

    # Month that scenario rollouts start on (day after last observation)
    sc_start_month = int((obs_months[-1] + 1) % 12)

    # Run rollouts
    scenarios_by_model = {}
    tier2_metrics = {}

    if args.n_workers == 1:
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
                variant_dirs=ckpt_variant_dirs,
            )
            if sc_mm is None:
                continue
            scenarios_by_model[variant] = sc_mm

            t2 = compute_tier2_metrics(sc_mm, data_raw, obs_months)
            tier2_metrics[variant] = t2
            print(f"  ACF RMSE={t2['multi_lag_acf_rmse']:.4f} | "
                  f"Trans Err={t2['transition_prob_error']:.4f} | "
                  f"CV={t2['inter_scenario_cv']:.4f}")
    else:
        print(f"[compare_ar] Parallel rollouts: {args.n_workers} workers, "
              f"{len(ckpt_variants)} models")
        worker_args = [
            (v, str(ckpt_variant_dirs[v]), data_norm, std, data_raw, obs_months,
             args.n_days, args.n_scenarios, str(device), args.force_rollouts)
            for v in ckpt_variants
        ]
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futures = {pool.submit(_rollout_worker, a): a[0] for a in worker_args}
            for fut in as_completed(futures):
                try:
                    variant, t2 = fut.result()
                except Exception as exc:
                    variant = futures[fut]
                    print(f"  [done] {variant}: worker process died — {exc}", flush=True)
                    continue
                if t2 is None:
                    print(f"  [done] {variant}: failed or skipped")
                    continue
                # Save metrics — do this before np.load so t2 is preserved even on cache failure
                tier2_metrics[variant] = t2
                # Load sc_mm from cache (run_rollout saves it; cache-hit path reuses existing file)
                cache_path = (ckpt_variant_dirs[variant] / "scenarios" / "scenarios.npy")
                try:
                    sc_mm = np.load(str(cache_path))[:args.n_scenarios, :args.n_days, :]
                except Exception as exc:
                    print(f"  [done] {variant}: cache load failed — {exc}", flush=True)
                    continue
                scenarios_by_model[variant] = sc_mm
                print(f"  [done] {variant}: ACF={t2['multi_lag_acf_rmse']:.4f} | "
                      f"Trans Err={t2['transition_prob_error']:.4f} | "
                      f"CV={t2['inter_scenario_cv']:.4f}")

        # Re-sort to original ckpt_variants order for deterministic plot output
        scenarios_by_model = {v: scenarios_by_model[v] for v in ckpt_variants
                              if v in scenarios_by_model}
        tier2_metrics = {v: tier2_metrics[v] for v in ckpt_variants
                        if v in tier2_metrics}

    # Save Tier 2 metrics
    with open(os.path.join(out_dir, "tier2_temporal_metrics.json"), "w") as f:
        json.dump(tier2_metrics, f, indent=2)
    print(f"\n  [report] Tier 2 metrics saved.")

    if not scenarios_by_model:
        print("[compare_ar] No scenarios generated.")
    else:
        # ── Combined Tier1+Tier2 score ──
        combined_scores, _ = compute_combined_score(all_metrics, tier2_metrics)
        with open(os.path.join(out_dir, "combined_scores_ar.json"), "w") as f:
            json.dump({"combined_scores": combined_scores}, f, indent=2)
        print(f"\n  [report] Combined scores saved.")

        # ── Filter top-level plots ──
        if args.top_n_per_family is not None:
            plot_variants = select_top_n_per_family(combined_scores, families, args.top_n_per_family)
            plot_scenarios = {v: scenarios_by_model[v] for v in plot_variants
                              if v in scenarios_by_model}
            print(f"\n[compare_ar] --top_n_per_family={args.top_n_per_family}: "
                  f"{len(plot_scenarios)} variants in top-level plots "
                  f"(from {len(scenarios_by_model)} total with checkpoints)")
        else:
            plot_scenarios = scenarios_by_model

        # ── Exclude known outlier model families from top-level scenario plots ──
        if OUTLIER_MODELS:
            excluded = [v for v in plot_scenarios if any(o in v for o in OUTLIER_MODELS)]
            if excluded:
                print(f"[compare_ar] Excluding outlier models from scenario plots: {excluded}")
            plot_scenarios = {v: sc for v, sc in plot_scenarios.items()
                              if not any(o in v for o in OUTLIER_MODELS)}


        # ── Top-level Tier 2 charts (filtered) ──
        print(f"\n[compare_ar] Generating Tier 2 charts ({len(plot_scenarios)} models)...")
        plot_autocorr_multilag(plot_scenarios, data_raw, out_dir, families=families)
        plot_spell_length_comparison(plot_scenarios, data_raw, out_dir, families=families)
        plot_monthly_precip(plot_scenarios, data_raw, obs_months, out_dir, families=families,
                            start_month=sc_start_month)
        plot_spread_envelopes(plot_scenarios, data_raw, out_dir, output_dir=args.output_dir,
                              variant_dirs=variant_dirs)
        plot_transition_probs(plot_scenarios, data_raw, out_dir, output_dir=args.output_dir,
                              variant_dirs=variant_dirs)
        plot_return_period(plot_scenarios, data_raw, out_dir, families=families)
        plot_rx5day_distribution(plot_scenarios, data_raw, out_dir, families=families)
        plot_rxnday_multi(plot_scenarios, data_raw, out_dir, families=families)
        plot_seasonal_accumulation(plot_scenarios, data_raw, obs_months, out_dir, families=families)
        plot_exceedance_frequency(plot_scenarios, data_raw, out_dir, families=families)

        # ── Overwrite CSV with full Tier1+Tier2 data ──
        write_metrics_csv(all_metrics, tier2_metrics, combined_scores, families, out_dir)

        # ── Per-family detail dirs ──
        family_detail_top_n = (args.top_n_family_detail if args.top_n_family_detail is not None
                               else args.top_n_per_family)
        print("\n[compare_ar] Generating per-family detail dirs...")
        plot_family_detail_dirs(
            scenarios_by_model, data_raw, obs_months,
            all_metrics, tier2_metrics, families, out_dir,
            sc_start_month=sc_start_month,
            top_n=family_detail_top_n,
            max_per_chart=args.max_series_per_chart,
        )

        if not args.skip_station_analysis:
            print("\n[compare_ar] === Tier 2 Per-Station Scatter Plots ===")
            plot_per_station_lag1_scatter(
                plot_scenarios, data_raw, station_names, families, out_dir,
            )
            plot_per_station_wetfreq_scatter(
                plot_scenarios, data_raw, station_names, families, out_dir,
            )

    print(f"\n[compare_ar] Done. Output: {out_dir}")


if __name__ == "__main__":
    main()
