"""
eval_nsteps_sweep.py — Re-evaluate ARFlowMap checkpoints at multiple n_steps values.

Copies config.json + model.pt from a source checkpoint into per-n_steps eval dirs,
then runs evaluate_from_dir() for each combination. Prints a comparison table.

Usage (from PrecipModels/):
    python eval_nsteps_sweep.py --output_dir outputs_v5_res --n_steps 1 3 5 10 20

    # Specify models explicitly:
    python eval_nsteps_sweep.py --output_dir outputs_v5_res --n_steps 1 5 10 20 \\
        --models ar_flow_map_res_h128_l4 ar_flow_map_res_h256_l4 ar_flow_map_res_ms_h256_l4

    # Skip models that already have results:
    python eval_nsteps_sweep.py --output_dir outputs_v5_res --n_steps 1 5 10 20 --skip_existing
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import shutil
import time
from pathlib import Path

from evaluate import evaluate_from_dir

# Models that support n_steps (ARFlowMap family only)
_FLOWMAP_MODELS = {"ar_flow_map", "ar_flow_map_res", "ar_flow_map_ms",
                   "ar_flow_map_sd", "ar_flow_map_lstm"}

# Default models to sweep if --models not specified
_DEFAULT_MODELS = [
    "ar_flow_map_res_h128_l4",
    "ar_flow_map_res_h256_l4",
    "ar_flow_map_res_h256_l6",
    "ar_flow_map_res_h128_l4_lsd01",
    "ar_flow_map_res_h256_l4_lsd01",
    "ar_flow_map_res_ms_h128_l4",
    "ar_flow_map_res_ms_h256_l4",
]


def run_sweep(output_dir: Path, models: list, n_steps_list: list,
              n_samples: int, skip_existing: bool):
    results = {}  # (model, n_steps) -> metrics dict

    total = len(models) * len(n_steps_list)
    done = 0
    t0 = time.time()

    for model_name in models:
        src_dir = output_dir / model_name
        if not src_dir.exists():
            print(f"[sweep] SKIP {model_name} — directory not found in {output_dir}")
            continue

        src_cfg_path = src_dir / "config.json"
        src_ckpt_path = src_dir / "model.pt"
        if not src_cfg_path.exists() or not src_ckpt_path.exists():
            print(f"[sweep] SKIP {model_name} — config.json or model.pt missing")
            continue

        cfg = json.loads(src_cfg_path.read_text())
        if cfg.get("model") not in _FLOWMAP_MODELS:
            print(f"[sweep] SKIP {model_name} — model '{cfg.get('model')}' does not support n_steps")
            continue

        for ns in n_steps_list:
            done += 1
            eval_dir = output_dir / f"{model_name}_s{ns}"

            if skip_existing and (eval_dir / "metrics.json").exists():
                print(f"[sweep] [{done}/{total}] {model_name} n_steps={ns} — cached, loading")
                results[(model_name, ns)] = json.loads((eval_dir / "metrics.json").read_text())
                continue

            # Prepare eval directory
            eval_dir.mkdir(exist_ok=True)

            # Symlink model.pt if possible, otherwise copy
            target_ckpt = eval_dir / "model.pt"
            if not target_ckpt.exists():
                try:
                    target_ckpt.symlink_to(src_ckpt_path.resolve())
                except (OSError, NotImplementedError):
                    shutil.copy2(src_ckpt_path, target_ckpt)

            # Write patched config.json with n_steps override
            patched_cfg = dict(cfg)
            patched_cfg["n_steps"] = ns
            patched_cfg["variant_name"] = f"{model_name}_s{ns}"
            (eval_dir / "config.json").write_text(json.dumps(patched_cfg, indent=2))

            t_start = time.time()
            print(f"\n[sweep] [{done}/{total}] {model_name}  n_steps={ns}")
            metrics = evaluate_from_dir(eval_dir, n_samples_override=n_samples)
            elapsed = time.time() - t_start
            results[(model_name, ns)] = metrics
            print(f"[sweep]   done in {elapsed:.0f}s  Wass={metrics['mean_wasserstein']:.4f}  "
                  f"Cov80={metrics.get('coverage_80', float('nan')):.4f}")

    return results


def print_table(results: dict, models: list, n_steps_list: list):
    metrics_to_show = [
        ("mean_wasserstein",       "Wass"),
        ("coverage_80",            "Cov80%"),
        ("wet_spell_length_error", "WetSpell"),
        ("lag1_autocorr_error",    "Lag1Err"),
        ("energy_score",           "EnergyScore"),
    ]

    # Header
    col_w = 10
    model_w = 42
    header = f"{'Model':<{model_w}}  {'Steps':>5}"
    for _, label in metrics_to_show:
        header += f"  {label:>{col_w}}"
    sep = "-" * len(header)
    print("\n" + sep)
    print(header)
    print(sep)

    for model_name in models:
        for ns in n_steps_list:
            key = (model_name, ns)
            if key not in results:
                continue
            m = results[key]
            row = f"{model_name:<{model_w}}  {ns:>5}"
            for metric_key, _ in metrics_to_show:
                val = m.get(metric_key, float("nan"))
                row += f"  {val:>{col_w}.4f}"
            print(row)
        print(sep)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep n_steps at inference for existing ARFlowMap checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--output_dir", default="outputs_v5_res",
                        help="Directory containing trained model subdirectories")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Model subdirectory names to sweep (default: all ar_flow_map_res* in output_dir)")
    parser.add_argument("--n_steps", type=int, nargs="+", default=[1, 3, 5, 10, 20],
                        help="n_steps values to evaluate")
    parser.add_argument("--n_samples", type=int, default=200,
                        help="Number of samples per evaluation")
    parser.add_argument("--skip_existing", action="store_true",
                        help="Skip evaluations where metrics.json already exists")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    if not output_dir.exists():
        raise SystemExit(f"output_dir not found: {output_dir}")

    # Auto-discover models if not specified
    if args.models is None:
        models = sorted([
            d.name for d in output_dir.iterdir()
            if d.is_dir()
            and (d / "model.pt").exists()
            and (d / "config.json").exists()
            and "ar_flow_map_res" in d.name
            and not any(f"_s{ns}" in d.name for ns in range(100))  # skip eval dirs
        ])
        if not models:
            raise SystemExit(f"No ar_flow_map_res* checkpoints found in {output_dir}")
        print(f"[sweep] Auto-discovered {len(models)} models: {models}")
    else:
        models = args.models

    print(f"[sweep] Models: {len(models)}  n_steps: {args.n_steps}  "
          f"total evals: {len(models) * len(args.n_steps)}")

    results = run_sweep(output_dir, models, args.n_steps, args.n_samples, args.skip_existing)

    print_table(results, models, args.n_steps)


if __name__ == "__main__":
    main()
