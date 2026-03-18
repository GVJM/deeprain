"""
queue_server.py — Distributed training job queue coordinator.

Usage (from PrecipModels/):
    python queue_server.py
    python queue_server.py --queues_dir . --port 5000 --timeout 30
    python queue_server.py --machine oracle_vm --dropbox_remote "dropbox:PrecipModels"

Only ONE instance should run at a time. Startup checks coordinator_lock.json in
Dropbox — aborts if another coordinator has been active in the last 5 minutes.
"""
import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request


def create_app(
    queues_dir: str = ".",
    timeout_secs: int = 1800,
    machine: str = "oracle_vm",
    dropbox_remote: str = "",
) -> Flask:
    app = Flask(__name__)

    _lock = threading.Lock()
    _jobs: dict = {}        # key: (queue_file_str, variant_name) -> record
    _config = {
        "queues_dir": Path(queues_dir),
        "timeout_secs": timeout_secs,
        "machine": machine,
        "dropbox_remote": dropbox_remote,
    }

    def _load_queues():
        qdir = _config["queues_dir"]
        patterns = list(qdir.glob("QUEUE_*.json")) + [qdir / "TRAINING_QUEUE.json"]
        for p in sorted(patterns):
            if not p.exists():
                continue
            entries = json.loads(p.read_text())
            for entry in entries:
                vname = entry["variant_name"]
                key = (str(p), vname)
                if key in _jobs:
                    continue
                # Read persisted status from entry (_status written by _save_queue)
                status = entry.get("_status", "pending")
                _jobs[key] = {
                    "spec": {k: v for k, v in entry.items() if not k.startswith("_")},
                    "queue_file": str(p),
                    "status": status,
                    "machine": entry.get("_machine"),
                    "started_at": None,
                    "heartbeat_at": None,
                }

    def _save_queue(queue_file_str: str):
        """Write specs with persisted _status/_machine to disk and rclone copy."""
        path = Path(queue_file_str)
        entries = [
            {**r["spec"], "_status": r["status"], "_machine": r["machine"]}
            for (qf, _), r in _jobs.items() if qf == queue_file_str
        ]
        path.write_text(json.dumps(entries, indent=2))
        remote = _config["dropbox_remote"]
        if remote:
            subprocess.run(
                ["rclone", "copy", str(path), f"{remote}/"],
                capture_output=True,
            )

    def _check_timeouts():
        """Return timed-out running jobs to pending. Must be called inside _lock."""
        timeout = _config["timeout_secs"]
        now = time.time()
        for record in _jobs.values():
            if record["status"] != "running":
                continue
            last = record["heartbeat_at"] or record["started_at"] or 0
            if now - last > timeout:
                print(f"[server] Timeout: returning {record['spec']['variant_name']} to pending")
                record.update(status="pending", machine=None, started_at=None, heartbeat_at=None)

    def _find_queue_file(variant_name: str, queue_file: str) -> str:
        """Resolve full queue file path key used in _jobs.
        Workers send basename; server stores full path. Match either.
        """
        for (qf, vn) in _jobs:
            if vn != variant_name:
                continue
            if not queue_file or qf == queue_file or Path(qf).name == queue_file:
                return qf
        return ""

    @app.get("/status")
    def status():
        with _lock:
            _check_timeouts()
            rows = [
                {
                    "variant_name": vn,
                    "job_type": r["spec"].get("job_type", "train"),
                    "status": r["status"],
                    "machine": r["machine"],
                    "queue_file": Path(qf).name,
                    "started_at": r["started_at"],
                }
                for (qf, vn), r in sorted(_jobs.items())
            ]
        return jsonify(rows)

    @app.get("/claim")
    def claim():
        machine = request.args.get("machine", "unknown")
        gpu_tier = int(request.args.get("gpu_tier", 0))
        with _lock:
            _check_timeouts()
            for (qf, vn), record in _jobs.items():
                if record["status"] != "pending":
                    continue
                if record["spec"].get("gpu_tier", 1) > gpu_tier:
                    continue
                record.update(
                    status="running",
                    machine=machine,
                    started_at=time.time(),
                    heartbeat_at=time.time(),
                )
                return jsonify({
                    "status": "ok",
                    "job": record["spec"],
                    "queue_file": Path(qf).name,
                })
        return jsonify({"status": "idle"})

    @app.post("/complete")
    def complete():
        data = request.json or {}
        variant = data.get("variant_name", "")
        queue_file = _find_queue_file(variant, data.get("queue_file", ""))
        key = (queue_file, variant)
        with _lock:
            if key not in _jobs:
                return jsonify({"error": f"job not found: {variant}"}), 404
            record = _jobs[key]
            record.update(status="done", machine=None)
            spec = record["spec"]
            if spec.get("job_type", "train") == "train":
                out_dir = spec.get("output_dir", "outputs")
                eval_spec = {
                    "variant_name": f"{variant}__eval",
                    "job_type": "eval",
                    "model_dir": f"{out_dir}/{variant}",
                    "gpu_tier": 0,
                    "output_dir": out_dir,
                }
                eval_key = (queue_file, eval_spec["variant_name"])
                if eval_key not in _jobs:
                    _jobs[eval_key] = {
                        "spec": eval_spec,
                        "queue_file": queue_file,
                        "status": "pending",
                        "machine": None,
                        "started_at": None,
                        "heartbeat_at": None,
                    }
            _save_queue(queue_file)
        return jsonify({"status": "ok"})

    @app.post("/heartbeat")
    def heartbeat():
        data = request.json or {}
        variant = data.get("variant_name", "")
        queue_file = _find_queue_file(variant, data.get("queue_file", ""))
        key = (queue_file, variant)
        with _lock:
            if key not in _jobs:
                return jsonify({"error": "job not found"}), 404
            _jobs[key]["heartbeat_at"] = time.time()
        return jsonify({"status": "ok"})

    @app.post("/release")
    def release():
        data = request.json or {}
        variant = data.get("variant_name", "")
        queue_file = _find_queue_file(variant, data.get("queue_file", ""))
        key = (queue_file, variant)
        with _lock:
            if key not in _jobs:
                return jsonify({"error": "job not found"}), 404
            record = _jobs[key]
            if record["status"] == "running":
                record.update(status="pending", machine=None, started_at=None, heartbeat_at=None)
        return jsonify({"status": "ok"})

    _load_queues()
    return app


def _acquire_coordinator_lock(dropbox_remote: str, machine: str):
    """Write coordinator_lock.json; abort if another coordinator is active."""
    lock_file = Path("coordinator_lock.json")
    if dropbox_remote:
        subprocess.run(
            ["rclone", "copy", f"{dropbox_remote}/coordinator_lock.json", "."],
            capture_output=True,
        )
    if lock_file.exists():
        lock = json.loads(lock_file.read_text())
        age = time.time() - lock.get("timestamp", 0)
        if lock.get("machine") != machine and age < 300:
            raise RuntimeError(
                f"Another coordinator '{lock['machine']}' is active (age: {age:.0f}s). "
                f"To take over, wait 5 minutes or delete coordinator_lock.json from Dropbox."
            )
    lock_file.write_text(json.dumps({"machine": machine, "timestamp": time.time()}, indent=2))
    if dropbox_remote:
        subprocess.run(
            ["rclone", "copy", str(lock_file), f"{dropbox_remote}/"],
            capture_output=True,
        )


def _pull_queues_from_dropbox(dropbox_remote: str, queues_dir: Path):
    if not dropbox_remote:
        return
    for pattern in ("TRAINING_QUEUE.json", "QUEUE_*.json"):
        subprocess.run(
            ["rclone", "copy", "--include", pattern, f"{dropbox_remote}/", str(queues_dir)],
            capture_output=True,
        )


def main():
    parser = argparse.ArgumentParser(
        description="Job queue coordinator for distributed training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--queues_dir", default=".")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--timeout", type=int, default=30,
                        help="Minutes before a running job times out and returns to pending")
    parser.add_argument("--machine", default="oracle_vm")
    parser.add_argument("--dropbox_remote", default="",
                        help="rclone remote path, e.g. 'dropbox:PrecipModels'")
    args = parser.parse_args()

    queues_dir = Path(args.queues_dir)
    _pull_queues_from_dropbox(args.dropbox_remote, queues_dir)
    _acquire_coordinator_lock(args.dropbox_remote, args.machine)

    app = create_app(
        queues_dir=str(queues_dir),
        timeout_secs=args.timeout * 60,
        machine=args.machine,
        dropbox_remote=args.dropbox_remote,
    )
    print(f"[server] Starting on port {args.port} | queues_dir={queues_dir} | machine={args.machine}")
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
