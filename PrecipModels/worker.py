"""
worker.py — Distributed training job poller.

Polls queue_server.py for jobs, executes train.py or evaluate.py,
syncs outputs to Dropbox via rclone copy, and reports completion.

Usage (from PrecipModels/):
    python worker.py --server http://oracle-ip:5000 --machine home_pc --gpu_tier 1
    python worker.py --config worker_config.json

On Windows: set output_base to the Dropbox folder root; leave dropbox_remote empty
(Dropbox client syncs automatically, but worker.py still calls rclone copy explicitly
to ensure files are uploaded before posting /complete).

On Oracle VM: set output_base to a local path and set dropbox_remote to
"dropbox:PrecipModels" so rclone copy uploads after each job.

Crash recovery: if current_job.json exists on startup, the previous job was
interrupted. Worker re-runs or skips the job based on existing output files.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import requests

CURRENT_JOB_FILE = Path("current_job.json")

_TRAIN_PARAMS = (
    "data_path", "max_epochs", "lr", "batch_size", "latent_size",
    "latent_occ", "latent_amt", "kl_warmup", "mf_warmup",
    "hidden_size", "n_layers", "n_coupling", "hidden_occ", "hidden_amt",
    "gru_hidden", "context_dim", "window_size", "hidden_dim", "t_embed_dim",
    "n_sample_steps", "rnn_hidden", "rnn_type", "n_steps",
    "mf_ratio", "jvp_eps", "occ_weight",
    "holdout_ratio", "normalization_mode",
)


def _local_output_path(job: dict, output_base: str) -> Path:
    """Return the local directory where model outputs are stored.
    Formula: Path(output_base) / output_dir / variant_name
    For eval jobs, variant is the last component of model_dir.
    """
    output_dir = job.get("output_dir", "outputs")
    if job.get("job_type") == "eval":
        variant = Path(job["model_dir"]).name
    else:
        variant = job["variant_name"]
    return Path(output_base) / output_dir / variant


def _rclone_copy(src: str, dst: str):
    """rclone copy src to dst. Logs warning on failure, does not raise."""
    result = subprocess.run(["rclone", "copy", src, dst], capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[worker] rclone warning: {result.stderr.strip()}")


def _post(server: str, path: str, data: dict, max_wait: int = 600) -> dict | None:
    """POST with exponential backoff up to max_wait seconds."""
    delay = 10
    elapsed = 0
    while elapsed < max_wait:
        try:
            r = requests.post(f"{server}{path}", json=data, timeout=30)
            if r.ok:
                return r.json()
        except Exception as e:
            print(f"[worker] POST {path} failed: {e} — retrying in {delay}s")
        time.sleep(delay)
        elapsed += delay
        delay = min(delay * 2, 120)
    print(f"[worker] POST {path} gave up after {elapsed}s")
    return None


def _get(server: str, path: str, params: dict = None) -> dict | None:
    try:
        r = requests.get(f"{server}{path}", params=params, timeout=30)
        if r.ok:
            return r.json()
    except Exception as e:
        print(f"[worker] GET {path} failed: {e}")
    return None


def run_job(job: dict, args) -> bool:
    """Execute a single job. Returns True if successful (or already done)."""
    job_type = job.get("job_type", "train")
    out_path = _local_output_path(job, args.output_base)

    if job_type == "train":
        if (out_path / "model.pt").exists():
            print(f"[worker] {job['variant_name']}: model.pt exists, skipping (crash recovery)")
            return True
        output_dir_path = Path(args.output_base) / job.get("output_dir", "outputs")
        cmd = [
            sys.executable, "train.py",
            "--skip_eval",
            "--name", job["variant_name"],
            "--model", job["model"],
            "--output_dir", str(output_dir_path),
        ]
        for key in _TRAIN_PARAMS:
            val = job.get(key)
            if val is not None:
                cmd += [f"--{key}", str(val)]
        result = subprocess.run(cmd)
        return result.returncode == 0

    elif job_type == "eval":
        metrics_path = out_path / "metrics.json"
        model_pt_path = out_path / "model.pt"
        if (metrics_path.exists() and model_pt_path.exists()
                and metrics_path.stat().st_mtime > model_pt_path.stat().st_mtime):
            print(f"[worker] {job.get('variant_name', 'unknown')}: metrics.json up-to-date, skipping (crash recovery)")
            return True
        result = subprocess.run([sys.executable, "evaluate.py", "--model_dir", str(out_path)])
        return result.returncode == 0

    print(f"[worker] Unknown job_type: {job_type}")
    return False


def _sync_output(job: dict, output_base: str, dropbox_remote: str):
    if not dropbox_remote:
        return
    out_path = _local_output_path(job, output_base)
    output_dir = job.get("output_dir", "outputs")
    variant = Path(job["model_dir"]).name if job.get("job_type") == "eval" else job["variant_name"]
    remote_dst = f"{dropbox_remote}/{output_dir}/{variant}/"
    print(f"[worker] Syncing {out_path}/ → {remote_dst}")
    _rclone_copy(str(out_path) + "/", remote_dst)


def main():
    parser = argparse.ArgumentParser(
        description="Distributed training worker.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--server", help="queue_server URL (e.g. http://oracle-ip:5000)")
    parser.add_argument("--machine", help="Machine identifier (e.g. home_pc)")
    parser.add_argument("--gpu_tier", type=int, default=None)
    parser.add_argument("--poll_interval", type=int, default=None)
    parser.add_argument("--output_base", default=None)
    parser.add_argument("--dropbox_remote", default=None)
    parser.add_argument("--config", default=None, help="JSON config file")
    args = parser.parse_args()

    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        for k, v in cfg.items():
            if not k.startswith("_") and getattr(args, k, None) is None:
                setattr(args, k, v)

    # Apply final defaults for args that were neither set on CLI nor in config
    if args.gpu_tier is None:
        args.gpu_tier = 0
    if args.poll_interval is None:
        args.poll_interval = 30
    if args.output_base is None:
        args.output_base = "."
    if args.dropbox_remote is None:
        args.dropbox_remote = ""

    if not args.server or not args.machine:
        parser.error("--server and --machine are required (or set in --config)")

    # Crash recovery
    if CURRENT_JOB_FILE.exists():
        job = json.loads(CURRENT_JOB_FILE.read_text())
        print(f"[worker] Recovering interrupted job: {job['variant_name']}")
        success = run_job(job, args)
        if success:
            _sync_output(job, args.output_base, args.dropbox_remote)
            _post(args.server, "/complete", {
                "variant_name": job["variant_name"],
                "queue_file": job.get("_queue_file", ""),
            })
        else:
            _post(args.server, "/release", {
                "variant_name": job["variant_name"],
                "queue_file": job.get("_queue_file", ""),
            })
        CURRENT_JOB_FILE.unlink(missing_ok=True)

    # Clean shutdown
    def _on_exit(sig, frame):
        if CURRENT_JOB_FILE.exists():
            job = json.loads(CURRENT_JOB_FILE.read_text())
            print(f"[worker] Clean shutdown: releasing {job['variant_name']}")
            _post(args.server, "/release", {
                "variant_name": job["variant_name"],
                "queue_file": job.get("_queue_file", ""),
            })
        sys.exit(0)

    signal.signal(signal.SIGINT, _on_exit)
    signal.signal(signal.SIGTERM, _on_exit)

    print(f"[worker] Ready: machine={args.machine}, gpu_tier={args.gpu_tier}, server={args.server}")

    while True:
        response = _get(args.server, "/claim", {"machine": args.machine, "gpu_tier": args.gpu_tier})
        if response is None or response.get("status") == "idle":
            time.sleep(args.poll_interval)
            continue

        job = response["job"]
        job["_queue_file"] = response.get("queue_file", "")
        CURRENT_JOB_FILE.write_text(json.dumps(job, indent=2))
        print(f"[worker] Claimed: {job['variant_name']} (type={job.get('job_type','train')})")

        stop_hb = threading.Event()

        def _heartbeat(j=job, stop=stop_hb):
            while not stop.is_set():
                _post(args.server, "/heartbeat", {
                    "variant_name": j["variant_name"],
                    "queue_file": j.get("_queue_file", ""),
                })
                stop.wait(timeout=60)

        threading.Thread(target=_heartbeat, daemon=True).start()

        try:
            success = run_job(job, args)
            if success:
                _sync_output(job, args.output_base, args.dropbox_remote)
                _post(args.server, "/complete", {
                    "variant_name": job["variant_name"],
                    "queue_file": job.get("_queue_file", ""),
                })
                print(f"[worker] Completed: {job['variant_name']}")
            else:
                print(f"[worker] Failed: {job['variant_name']} — releasing")
                _post(args.server, "/release", {
                    "variant_name": job["variant_name"],
                    "queue_file": job.get("_queue_file", ""),
                })
        finally:
            stop_hb.set()
            CURRENT_JOB_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
