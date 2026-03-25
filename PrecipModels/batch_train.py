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

import argparse, json, sys, time, traceback
from argparse import Namespace
from datetime import datetime
from pathlib import Path

_DEFAULT_QUEUE = Path(__file__).parent / "TRAINING_QUEUE.json"
OUTPUTS_DIR    = Path(__file__).parent / "outputs_v3"
LOG_FILE    = OUTPUTS_DIR / "batch_log.jsonl"

# All fields train_model(args) reads; None → falls back to MODEL_DEFAULTS / ARCH_DEFAULTS
_NS_DEFAULTS = dict(
    # Training scalars (None → MODEL_DEFAULTS)
    max_epochs=1000, lr=None, batch_size=128, kl_warmup=500,
    latent_size=None, latent_occ=None, latent_amt=None,
    normalization_mode=None,
    # Infrastructure (match argparse defaults)
    device="auto", output_dir=str(OUTPUTS_DIR),
    n_samples=1000, holdout_ratio=0.0,
    resume=False, optimize=False, num_threads=None,
    # Architecture (None → ARCH_DEFAULTS)
    hidden_size=None, n_layers=None, n_coupling=None,
    hidden_occ=None, hidden_amt=None,
    gru_hidden=None, context_dim=None, window_size=7,
    hidden_dim=None, t_embed_dim=None, n_sample_steps=None,
    rnn_hidden=None, rnn_type=None, n_steps=None, mf_ratio=None,
    mf_warmup=None, jvp_eps=None, occ_weight=None,   # new
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

def log_job(record: dict):
    """Append one JSON record to the batch log file."""
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")


def fmt_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m{s:02d}s"


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
    p.add_argument("--queue",     default=str(_DEFAULT_QUEUE),
                   help="Queue JSON file to use (default: TRAINING_QUEUE.json)")
    args = p.parse_args()

    queue_file = Path(args.queue)
    with open(queue_file) as f:
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

    completed_times = []
    n_ok = 0
    n_err = 0

    for i, (entry, status) in enumerate(to_run):
        vn = entry["variant_name"]
        remaining = len(to_run) - i
        if completed_times:
            avg_s = sum(completed_times) / len(completed_times)
            eta_str = fmt_duration(avg_s * remaining)
            eta_msg = f"  ETA ~{eta_str} for {remaining} remaining job(s)"
        else:
            eta_msg = ""
        print(f"\n{'#'*65}")
        print(f"# [{i+1}/{len(to_run)}] {vn}  status={status}{eta_msg}")
        print(f"# Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'#'*65}\n")

        ns = build_namespace(entry, args.data_path, args.device)
        if status == "partial":
            ns.resume = True
            print(f"[batch_train] Partial run detected — auto-resuming {vn}")

        t0 = time.perf_counter()
        job_ok = True
        err_msg = None
        try:
            train_model(ns)
            n_ok += 1
        except KeyboardInterrupt:
            elapsed = time.perf_counter() - t0
            print(f"\n[batch_train] Interrupted at '{vn}' after {fmt_duration(elapsed)}. Stopping.")
            log_job({"variant": vn, "status": "interrupted",
                     "elapsed_s": round(elapsed, 1),
                     "timestamp": datetime.now().isoformat()})
            break
        except Exception as e:
            job_ok = False
            err_msg = traceback.format_exc()
            n_err += 1
            print(f"\n[batch_train] ERROR in '{vn}':\n{err_msg}")
            print("[batch_train] Skipping to next...")

        elapsed = time.perf_counter() - t0
        completed_times.append(elapsed)
        log_job({
            "variant": vn,
            "status": "ok" if job_ok else "error",
            "elapsed_s": round(elapsed, 1),
            "elapsed_fmt": fmt_duration(elapsed),
            "error": err_msg,
            "timestamp": datetime.now().isoformat(),
        })
        status_tag = "OK" if job_ok else "ERROR"
        print(f"\n[batch_train] [{i+1}/{len(to_run)}] {vn} — {status_tag} in {fmt_duration(elapsed)}")

    total_s = sum(completed_times)
    print(f"\n{'='*65}")
    print(f"Batch complete: {n_ok} OK, {n_err} errors  |  Total time: {fmt_duration(total_s)}")
    if LOG_FILE.exists():
        print(f"Log written to: {LOG_FILE}")
    print_table(json.load(open(queue_file)))

if __name__ == "__main__":
    main()
