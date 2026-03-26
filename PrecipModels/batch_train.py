"""
PrecipModels/batch_train.py — Batch training queue runner.

Usage (from PrecipModels/):
    python batch_train.py                        # train all pending (TRAINING_QUEUE.json)
    python batch_train.py --status               # print table, exit
    python batch_train.py --force                # re-run done runs too
    python batch_train.py --only ar_vae_small    # specific variants
    python batch_train.py --data_path ../dados_sabesp/dayprecip.dat
    python batch_train.py --device cuda

    # Multi-queue: merge all QUEUE*.json files in a folder
    python batch_train.py --queue_dir queues/
    python batch_train.py --queue_dir queues/ --status
    python batch_train.py --queue_dir queues/ --device cuda

    # Single alternate file (new or old format)
    python batch_train.py --queue QUEUE_24_v5_res.json --status

Queue file formats
------------------
Old (plain array, backward-compatible):
    [ {"variant_name": "...", "model": "...", "output_dir": "results/..."}, ... ]

New (dict with defaults — avoids repeating data_path, output_dir, etc.):
    {
      "defaults": {
        "data_path": "../dados_sabesp/data_precip.dat",
        "output_dir": "results/data_precip/outputs_v6_glow",
        "holdout_ratio": 0.15,
        "max_epochs": 2000
      },
      "entries": [
        {"variant_name": "ar_glow_h64_w30", "model": "ar_glow", "hidden_size": 64, "window_size": 30},
        {"variant_name": "ar_glow_h128",    "model": "ar_glow", "hidden_size": 128}
      ]
    }
  Entry-level fields override defaults.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse, json, sys, time, traceback
from argparse import Namespace
from datetime import datetime
from pathlib import Path

_DEFAULT_QUEUE = Path(__file__).parent / "TRAINING_QUEUE.json"
LOG_FILE = Path(__file__).parent / "batch_log.jsonl"

# All fields train_model(args) reads; None → falls back to MODEL_DEFAULTS / ARCH_DEFAULTS
_NS_DEFAULTS = dict(
    # Training scalars (None → MODEL_DEFAULTS)
    max_epochs=1000, lr=None, batch_size=128, kl_warmup=500,
    latent_size=None, latent_occ=None, latent_amt=None,
    normalization_mode=None,
    # Infrastructure (match argparse defaults)
    device="auto", output_dir=None,
    n_samples=1000, holdout_ratio=0.0,
    resume=False, optimize=False, num_threads=None,
    # Architecture (None → ARCH_DEFAULTS)
    hidden_size=None, n_layers=None, n_coupling=None,
    hidden_occ=None, hidden_amt=None,
    gru_hidden=None, context_dim=None, window_size=7,
    hidden_dim=None, t_embed_dim=None, n_sample_steps=None,
    rnn_hidden=None, rnn_type=None, n_steps=None, mf_ratio=None,
    mf_warmup=None, jvp_eps=None, occ_weight=None,
)


def _parse_queue_file(path: Path) -> list:
    """Load a queue file in either old (plain array) or new (dict+defaults) format."""
    raw = json.loads(path.read_text())
    if isinstance(raw, list):
        return raw  # old format — unchanged
    defaults = raw.get("defaults", {})
    return [{**defaults, **entry} for entry in raw.get("entries", [])]


def load_queue_from_dir(queue_dir: Path) -> list:
    """Discover, load, and merge all QUEUE*.json files in queue_dir (sorted by name)."""
    files = sorted(queue_dir.glob("QUEUE*.json"))
    if not files:
        raise SystemExit(f"No QUEUE*.json files found in {queue_dir}")
    all_entries = []
    seen = {}  # variant_name -> source filename
    for qf in files:
        entries = _parse_queue_file(qf)
        print(f"  [queue] {qf.name}: {len(entries)} entries")
        for entry in entries:
            vn = entry["variant_name"]
            if vn in seen:
                print(f"  [queue] WARNING: duplicate '{vn}' in {qf.name} "
                      f"(already in {seen[vn]}) — skipping")
                continue
            seen[vn] = qf.name
            entry["_source"] = qf.name
            all_entries.append(entry)
    print(f"  [queue] Total: {len(all_entries)} unique variants from {len(files)} file(s)")
    return all_entries


def get_status(variant_name, output_dir=None):
    if not output_dir:
        return "pending"
    out = Path(output_dir) / variant_name
    if (out / "metrics.json").exists():  return "done"
    if (out / "config.json").exists():   return "partial"
    return "pending"


def build_namespace(entry, data_path_override, device_override):
    kw = dict(_NS_DEFAULTS)
    for k, v in entry.items():
        if k in ("variant_name", "_source"):
            continue
        kw[k] = v
    kw["name"] = entry["variant_name"]
    if data_path_override:
        kw["data_path"] = data_path_override
    if device_override:
        kw["device"] = device_override
    if "data_path" not in kw or kw.get("data_path") is None:
        kw["data_path"] = "../dados_sabesp/dayprecip.dat"
    if not kw.get("output_dir"):
        raise ValueError(
            f"No output_dir for variant '{entry['variant_name']}' — "
            "add output_dir to the queue entry or to the file's 'defaults' block"
        )
    return Namespace(**kw)


def log_job(record: dict):
    """Append one JSON record to the batch log file."""
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
    show_source = any("_source" in e for e in queue)
    statuses = {e["variant_name"]: get_status(e["variant_name"], e.get("output_dir")) for e in queue}
    icons = {"done": "[DONE]", "partial": "[PARTIAL]", "pending": "[PENDING]"}
    counts = {s: sum(1 for v in statuses.values() if v == s) for s in icons}
    width = 80 if show_source else 65
    print(f"\n{'='*width}")
    print(f"  QUEUE: {len(queue)} runs  |  Done:{counts['done']}  Partial:{counts['partial']}  Pending:{counts['pending']}")
    print(f"{'='*width}")
    for e in queue:
        vn = e["variant_name"]
        src = f"  [{e['_source']}]" if show_source else ""
        print(f"  {icons[statuses[vn]]:<10} {vn:<40} ({e['model']}){src}")
    print()


def main():
    # Must run from PrecipModels/
    if not Path("MODEL_DEFAULTS.json").exists():
        print("ERROR: Run from PrecipModels/ directory.")
        sys.exit(1)

    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=__doc__,
    )
    p.add_argument("--status",    action="store_true",
                   help="Print queue status table and exit")
    p.add_argument("--force",     action="store_true",
                   help="Re-run variants that are already done")
    p.add_argument("--only",      nargs="+", metavar="VARIANT",
                   help="Run only these variant names")
    p.add_argument("--data_path", default=None,
                   help="Override data_path for all entries")
    p.add_argument("--device",    default=None,
                   help="Override device for all entries (e.g. cuda, cpu)")
    p.add_argument("--queue",     default=str(_DEFAULT_QUEUE),
                   help="Single queue JSON file (default: TRAINING_QUEUE.json)")
    p.add_argument("--queue_dir", default=None,
                   help="Directory to scan for QUEUE*.json files and merge. "
                        "Mutually exclusive with --queue.")
    args = p.parse_args()

    if args.queue_dir and args.queue != str(_DEFAULT_QUEUE):
        p.error("--queue_dir and --queue are mutually exclusive")

    if args.queue_dir:
        queue_dir = Path(args.queue_dir)
        if not queue_dir.is_dir():
            raise SystemExit(f"queue_dir not found: {queue_dir}")
        queue = load_queue_from_dir(queue_dir)
        queue_file = None
    else:
        queue_file = Path(args.queue)
        queue = _parse_queue_file(queue_file)

    # Validate uniqueness (within a single file; cross-file dedup is done by load_queue_from_dir)
    if queue_file is not None:
        names = [e["variant_name"] for e in queue]
        if len(names) != len(set(names)):
            dupes = {n for n in names if names.count(n) > 1}
            print(f"ERROR: duplicate variant_names: {dupes}"); sys.exit(1)

    if args.only:
        queue = [e for e in queue if e["variant_name"] in args.only]

    print_table(queue)
    if args.status:
        return

    to_run = [(e, st) for e in queue
              for st in [get_status(e["variant_name"], e.get("output_dir"))]
              if args.force or st != "done"]

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
    print(f"Log: {LOG_FILE}")

    # Re-print status table (reload from file only if single-file mode)
    if queue_file is not None:
        print_table(_parse_queue_file(queue_file))
    else:
        print_table(queue)


if __name__ == "__main__":
    main()
