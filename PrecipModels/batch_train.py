"""
PrecipModels/batch_train.py — Batch training queue runner.

Usage (from PrecipModels/):
    python batch_train.py                        # train all pending
    python batch_train.py --status               # print table, exit
    python batch_train.py --force                # re-run done runs too
    python batch_train.py --only ar_vae_small    # specific variants
    python batch_train.py --data_path ../dados_sabesp/dayprecip.dat
    python batch_train.py --device cuda
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse, json, sys
from argparse import Namespace
from pathlib import Path

QUEUE_FILE  = Path(__file__).parent / "TRAINING_QUEUE.json"
OUTPUTS_DIR = Path(__file__).parent / "outputs"

# All fields train_model(args) reads; None → falls back to MODEL_DEFAULTS / ARCH_DEFAULTS
_NS_DEFAULTS = dict(
    # Training scalars (None → MODEL_DEFAULTS)
    max_epochs=None, lr=None, batch_size=None, kl_warmup=None,
    latent_size=None, latent_occ=None, latent_amt=None,
    normalization_mode=None,
    # Infrastructure (match argparse defaults)
    device="auto", output_dir=str(OUTPUTS_DIR),
    n_samples=5000, holdout_ratio=0.2,
    resume=False, optimize=False, num_threads=None,
    # Architecture (None → ARCH_DEFAULTS)
    hidden_size=None, n_layers=None, n_coupling=None,
    hidden_occ=None, hidden_amt=None,
    gru_hidden=None, context_dim=None, window_size=None,
    hidden_dim=None, t_embed_dim=None, n_sample_steps=None,
    rnn_hidden=None, rnn_type=None, n_steps=None, mf_ratio=None,
)

def get_status(variant_name):
    out = OUTPUTS_DIR / variant_name
    if (out / "metrics.json").exists():  return "done"
    if (out / "config.json").exists():   return "partial"
    return "pending"

def build_namespace(entry, data_path_override, device_override):
    kw = dict(_NS_DEFAULTS)
    for k, v in entry.items():
        if k == "variant_name": continue
        kw[k] = v
    kw["name"] = entry["variant_name"]
    if data_path_override: kw["data_path"] = data_path_override
    if device_override:    kw["device"]    = device_override
    if "data_path" not in kw or kw.get("data_path") is None:
        kw["data_path"] = "../dados_sabesp/dayprecip.dat"
    return Namespace(**kw)

def print_table(queue):
    statuses = {e["variant_name"]: get_status(e["variant_name"]) for e in queue}
    icons = {"done": "[DONE]", "partial": "[PARTIAL]", "pending": "[PENDING]"}
    counts = {s: sum(1 for v in statuses.values() if v == s) for s in icons}
    print(f"\n{'='*65}")
    print(f"  QUEUE: {len(queue)} runs  |  Done:{counts['done']}  Partial:{counts['partial']}  Pending:{counts['pending']}")
    print(f"{'='*65}")
    for e in queue:
        vn = e["variant_name"]
        print(f"  {icons[statuses[vn]]:<10} {vn:<35} ({e['model']})")
    print()

def main():
    # Must run from PrecipModels/
    if not Path("MODEL_DEFAULTS.json").exists():
        print("ERROR: Run from PrecipModels/ directory.")
        sys.exit(1)

    p = argparse.ArgumentParser()
    p.add_argument("--status", action="store_true")
    p.add_argument("--force",  action="store_true")
    p.add_argument("--only",   nargs="+", metavar="VARIANT")
    p.add_argument("--data_path", default=None)
    p.add_argument("--device",    default=None)
    args = p.parse_args()

    with open(QUEUE_FILE) as f:
        queue = json.load(f)

    # Validate uniqueness
    names = [e["variant_name"] for e in queue]
    if len(names) != len(set(names)):
        dupes = {n for n in names if names.count(n) > 1}
        print(f"ERROR: duplicate variant_names: {dupes}"); sys.exit(1)

    if args.only:
        queue = [e for e in queue if e["variant_name"] in args.only]

    print_table(queue)
    if args.status:
        return

    to_run = [(e, get_status(e["variant_name"])) for e in queue
              if args.force or get_status(e["variant_name"]) != "done"]

    if not to_run:
        print("Nothing to train. Use --force to re-run completed runs.")
        return

    sys.path.insert(0, str(Path(__file__).parent))
    from train import train_model

    for i, (entry, status) in enumerate(to_run):
        vn = entry["variant_name"]
        print(f"\n{'#'*65}")
        print(f"# [{i+1}/{len(to_run)}] {vn}  status={status}")
        print(f"{'#'*65}\n")

        ns = build_namespace(entry, args.data_path, args.device)
        if status == "partial":
            ns.resume = True
            print(f"[batch_train] Partial run detected — auto-resuming {vn}")

        try:
            train_model(ns)
        except KeyboardInterrupt:
            print(f"\n[batch_train] Interrupted at '{vn}'. Stopping.")
            break
        except Exception as e:
            print(f"\n[batch_train] ERROR in '{vn}': {e}")
            print("[batch_train] Skipping to next...")
            continue

    print("\n" + "="*65)
    print_table(json.load(open(QUEUE_FILE)))

if __name__ == "__main__":
    main()
