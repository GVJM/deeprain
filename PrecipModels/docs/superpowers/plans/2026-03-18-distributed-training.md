# Distributed Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Distribute model training across multiple machines (home PC, company PC, Oracle VM) using a pull-based HTTP queue with GPU-aware routing, deferred evaluation, and Dropbox file sync.

**Architecture:** Oracle VM runs `queue_server.py` (Flask HTTP coordinator); each machine runs `worker.py` which polls for jobs, trains with `train.py --skip_eval`, syncs outputs to Dropbox, then reports completion. Eval jobs are auto-inserted and claimed by any available worker.

**Tech Stack:** Python 3.9+, Flask, requests, rclone (Oracle VM), pytest

**Spec:** `docs/superpowers/specs/2026-03-18-distributed-training-design.md`

---

## File Map

| File | Action | Responsibility |
|------|--------|---------------|
| `train.py` | Modify | Add `--skip_eval` flag |
| `evaluate.py` | Create | Standalone evaluator (extracted from `train.py` lines 1187–1230) |
| `queue_server.py` | Create | Flask HTTP job queue coordinator |
| `worker.py` | Create | Job polling loop + crash recovery |
| `make_queue.py` | Modify | Add `--gpu_tier` and `--output_dir` flags |
| `TRAINING_QUEUE.json` | Modify | Add `gpu_tier`, `job_type`, `output_dir` to all entries |
| `worker_config.json.example` | Create | Example worker configuration |
| `tests/test_train_skip_eval.py` | Create | Tests for `--skip_eval` flag |
| `tests/test_evaluate.py` | Create | Tests for `evaluate.py` |
| `tests/test_queue_server.py` | Create | Tests for all queue_server endpoints |
| `tests/test_worker.py` | Create | Tests for worker path logic + crash recovery |
| `tests/test_make_queue.py` | Create | Tests for new make_queue.py flags |

---

## Task 1: `train.py` — `--skip_eval` flag

**Files:**
- Modify: `train.py` (argparser ~line 1266, eval block ~lines 1187–1230)
- Create: `tests/test_train_skip_eval.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_train_skip_eval.py
"""Test that --skip_eval prevents metrics.json from being written."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_skip_eval_does_not_write_metrics(tmp_path):
    """Running train.py --skip_eval must NOT create metrics.json."""
    result = subprocess.run(
        [
            sys.executable, "train.py",
            "--model", "copula",
            "--name", "test_skip_eval_copula",
            "--output_dir", str(tmp_path),
            "--skip_eval",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"train.py failed:\n{result.stderr}"
    out_dir = tmp_path / "test_skip_eval_copula"
    assert not (out_dir / "metrics.json").exists(), "metrics.json must not exist when --skip_eval is set"


def test_without_skip_eval_writes_metrics(tmp_path):
    """Running train.py without --skip_eval MUST create metrics.json."""
    result = subprocess.run(
        [
            sys.executable, "train.py",
            "--model", "copula",
            "--name", "test_no_skip_eval_copula",
            "--output_dir", str(tmp_path),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"train.py failed:\n{result.stderr}"
    out_dir = tmp_path / "test_no_skip_eval_copula"
    assert (out_dir / "metrics.json").exists(), "metrics.json must exist when --skip_eval is not set"
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd PrecipModels && pytest tests/test_train_skip_eval.py -v
```

Expected: `FAIL` — `unrecognized arguments: --skip_eval`

- [ ] **Step 3: Add `--skip_eval` to the argparser**

In `train.py`, after the `--n_samples` argument (~line 1267), add:

```python
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip post-training evaluation; do not write metrics.json. "
                             "Used by distributed worker — eval runs as a separate job.")
```

- [ ] **Step 4: Guard the eval block in `train_model()`**

In `train.py`, wrap lines 1187–1230 (the `# ── Métricas ──` block) with a check:

```python
    # ── Métricas ──
    if getattr(args, "skip_eval", False):
        print(f"[{variant_name}] --skip_eval set: skipping evaluation (metrics.json will not be written).")
        return
    model.eval()
    # ... (rest of eval block unchanged)
```

- [ ] **Step 5: Run test to verify it passes**

```bash
cd PrecipModels && pytest tests/test_train_skip_eval.py -v
```

Expected: `PASSED` for both tests

- [ ] **Step 6: Commit**

```bash
git add train.py tests/test_train_skip_eval.py
git commit -m "feat(train): add --skip_eval flag for distributed worker"
```

---

## Task 2: `evaluate.py` — Standalone Evaluator

**Files:**
- Create: `evaluate.py`
- Create: `tests/test_evaluate.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_evaluate.py
"""Test that evaluate.py loads a checkpoint and writes metrics.json."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_evaluate_writes_metrics(tmp_path):
    """evaluate.py must write metrics.json given a trained model dir.
    Uses hurdle_simple (not copula) — copula saves copula.pkl, not model.pt.
    """
    # First train a model without eval to get a checkpoint
    result = subprocess.run(
        [
            sys.executable, "train.py",
            "--model", "hurdle_simple",
            "--name", "test_eval_hurdle",
            "--output_dir", str(tmp_path),
            "--max_epochs", "2",
            "--skip_eval",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"train.py setup failed:\n{result.stderr}"
    model_dir = tmp_path / "test_eval_hurdle"
    assert (model_dir / "config.json").exists()
    assert (model_dir / "model.pt").exists(), "hurdle_simple must produce model.pt"
    assert not (model_dir / "metrics.json").exists()

    # Now run evaluate.py
    result = subprocess.run(
        [sys.executable, "evaluate.py", "--model_dir", str(model_dir), "--n_samples", "50"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"evaluate.py failed:\n{result.stderr}"
    assert (model_dir / "metrics.json").exists(), "metrics.json must be written by evaluate.py"

    metrics = json.loads((model_dir / "metrics.json").read_text())
    assert len(metrics) > 0


def test_evaluate_respects_n_samples_override(tmp_path):
    """evaluate.py --n_samples override must be accepted without error."""
    subprocess.run(
        [sys.executable, "train.py", "--model", "hurdle_simple",
         "--name", "test_eval_ns", "--output_dir", str(tmp_path),
         "--max_epochs", "2", "--skip_eval"],
        capture_output=True,
    )
    model_dir = tmp_path / "test_eval_ns"
    result = subprocess.run(
        [sys.executable, "evaluate.py", "--model_dir", str(model_dir), "--n_samples", "50"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"evaluate.py --n_samples failed:\n{result.stderr}"
    assert (model_dir / "metrics.json").exists()
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd PrecipModels && pytest tests/test_evaluate.py -v
```

Expected: `FAIL` — `evaluate.py` does not exist

- [ ] **Step 3: Implement `evaluate.py`**

Create `PrecipModels/evaluate.py`:

```python
"""
evaluate.py — Standalone model evaluator.

Loads a trained checkpoint from a model directory and writes metrics.json.
Used by distributed workers to decouple evaluation from GPU training time.

Usage (from PrecipModels/):
    python evaluate.py --model_dir outputs_sabesp/ar_vae
    python evaluate.py --model_dir outputs/vae --n_samples 1000

IMPORTANT: Float arch params (occ_weight, jvp_eps, mf_ratio) must be loaded
as float, not int — see CLAUDE.md "compare_ar.py rollout loader" and commit b64f0a1.
Any new float arch params added to AR models must be added to _FLOAT_PARAMS below.
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
from pathlib import Path

import torch

from data_utils import load_data, load_data_with_cond
from metrics import evaluate_model
from models import get_model
from train import (
    _TEMPORAL_MODELS,
    _MC_MODELS,
    temporal_holdout_split,
    temporal_holdout_split_with_cond,
    compute_norm_params,
    normalize_with_params,
)

# Float params: must be cast to float(), not int(), when loading from config.json
_FLOAT_PARAMS = ("occ_weight", "jvp_eps", "mf_ratio")

# Int architectural params
_INT_PARAMS = (
    "latent_size", "hidden_size", "n_layers", "n_coupling",
    "gru_hidden", "window_size", "t_embed_dim", "n_sample_steps",
    "rnn_hidden", "n_steps", "latent_occ", "latent_amt",
    "hidden_occ", "hidden_amt", "context_dim", "hidden_dim",
)


def evaluate_from_dir(model_dir: Path, n_samples_override: int = None) -> dict:
    """Load a trained model from model_dir and write metrics.json."""
    model_dir = Path(model_dir)
    cfg = json.loads((model_dir / "config.json").read_text())

    model_name = cfg["model"]
    data_path = cfg["data_path"]
    holdout_ratio = cfg.get("holdout_ratio", 0.0)
    norm_mode = cfg.get("normalization_mode", "scale_only")
    is_mc = model_name in _MC_MODELS
    is_temporal = model_name in _TEMPORAL_MODELS

    # Load data — must use same split as training to get correct mu/std
    if is_mc or is_temporal:
        _, data_raw_full, _, _, station_names, cond_arrays_full = load_data_with_cond(
            data_path=data_path, normalization_mode="scale_only"
        )
        train_raw, eval_raw, _, _ = temporal_holdout_split_with_cond(
            data_raw_full, cond_arrays_full, holdout_ratio
        )
    else:
        _, data_raw_full, _, _, station_names = load_data(
            data_path=data_path, normalization_mode="scale_only"
        )
        train_raw, eval_raw = temporal_holdout_split(data_raw_full, holdout_ratio)

    mu, std = compute_norm_params(train_raw, norm_mode)
    input_size = data_raw_full.shape[1]

    # Reconstruct model — mirrors compare_ar.py load_ar_model() logic
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_kwargs = {"input_size": input_size}
    for key in _INT_PARAMS:
        val = cfg.get(key)
        if val is not None:
            model_kwargs[key] = int(val)
    rnn_type = cfg.get("rnn_type")
    if rnn_type:
        model_kwargs["rnn_type"] = rnn_type
    for key in _FLOAT_PARAMS:
        val = cfg.get(key)
        if val is not None:
            model_kwargs[key] = float(val)   # MUST be float, not int

    model = get_model(model_name, **model_kwargs)
    ckpt_path = model_dir / "model.pt"
    if not ckpt_path.exists():
        # Copula models use copula.pkl, not model.pt — call fit and save
        raise FileNotFoundError(f"model.pt not found in {model_dir}. "
                                f"Copula models do not support deferred evaluation.")
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    # Evaluate — same caps as train.py for AR models
    n_samples = n_samples_override or cfg.get("n_samples", 5000)
    if is_temporal:
        eval_n_samples = min(n_samples, 200)
        timing_n_samples = 100
        timing_n_trials = 1
        print(f"[evaluate] AR model — using {eval_n_samples} samples, 1 timing trial")
    else:
        eval_n_samples = n_samples
        timing_n_samples = 1000
        timing_n_trials = 5

    print(f"[evaluate] Evaluating {model_name} from {model_dir} ...")
    metrics = evaluate_model(
        model, eval_raw, mu, std,
        n_samples=eval_n_samples,
        station_names=station_names,
        timing_n_samples=timing_n_samples,
        timing_n_trials=timing_n_trials,
    )

    out_path = model_dir / "metrics.json"
    out_path.write_text(json.dumps(metrics, indent=2))
    print(f"[evaluate] Metrics saved to {out_path}")
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model and write metrics.json.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model_dir", required=True,
                        help="Path to trained model directory (contains config.json + model.pt)")
    parser.add_argument("--n_samples", type=int, default=None,
                        help="Override n_samples from config.json")
    args = parser.parse_args()
    evaluate_from_dir(Path(args.model_dir), args.n_samples)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd PrecipModels && pytest tests/test_evaluate.py -v
```

Expected: `PASSED` for both tests

- [ ] **Step 5: Commit**

```bash
git add evaluate.py tests/test_evaluate.py
git commit -m "feat: add evaluate.py standalone evaluator for distributed workers"
```

---

## Task 3: `queue_server.py` — Core Job Queue

**Files:**
- Create: `queue_server.py`
- Create: `tests/test_queue_server.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_queue_server.py
"""Tests for queue_server.py HTTP endpoints."""
import json
import time
from pathlib import Path

import pytest


@pytest.fixture
def queue_file(tmp_path):
    """Write a minimal TRAINING_QUEUE.json for testing."""
    entries = [
        {
            "variant_name": "test_vae",
            "model": "vae",
            "job_type": "train",
            "gpu_tier": 1,
            "output_dir": "outputs",
            "data_path": "../dados_sabesp/dayprecip.dat",
        },
        {
            "variant_name": "test_copula",
            "model": "copula",
            "job_type": "train",
            "gpu_tier": 0,
            "output_dir": "outputs",
        },
    ]
    qfile = tmp_path / "TRAINING_QUEUE.json"
    qfile.write_text(json.dumps(entries))
    return tmp_path


@pytest.fixture
def client(queue_file):
    from queue_server import create_app
    app = create_app(
        queues_dir=str(queue_file),
        timeout_secs=30,
        machine="test_server",
        dropbox_remote="",
    )
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_status_returns_all_jobs(client):
    r = client.get("/status")
    assert r.status_code == 200
    jobs = r.get_json()
    assert len(jobs) == 2
    assert all(j["status"] == "pending" for j in jobs)


def test_claim_returns_highest_priority_pending_job(client):
    r = client.get("/claim?machine=worker1&gpu_tier=1")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert "job" in data
    assert data["job"]["variant_name"] in ("test_vae", "test_copula")


def test_claim_respects_gpu_tier(client):
    # gpu_tier=0 worker should only get tier-0 jobs
    r = client.get("/claim?machine=cpu_worker&gpu_tier=0")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert data["job"]["variant_name"] == "test_copula"  # only tier-0 job


def test_claim_marks_job_as_running(client):
    client.get("/claim?machine=worker1&gpu_tier=1")
    r = client.get("/status")
    jobs = r.get_json()
    running = [j for j in jobs if j["status"] == "running"]
    assert len(running) == 1
    assert running[0]["machine"] == "worker1"


def test_complete_marks_job_done_and_inserts_eval(client):
    # Claim the train job
    claim_r = client.get("/claim?machine=worker1&gpu_tier=1")
    job = claim_r.get_json()["job"]
    queue_file_name = claim_r.get_json().get("queue_file", "TRAINING_QUEUE.json")

    # Complete it
    r = client.post("/complete", json={
        "variant_name": job["variant_name"],
        "queue_file": queue_file_name,
    })
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"

    # Status should show done + new eval job pending
    status = client.get("/status").get_json()
    done = [j for j in status if j["status"] == "done"]
    assert any(j["variant_name"] == job["variant_name"] for j in done)
    pending = [j for j in status if j["status"] == "pending"]
    eval_jobs = [j for j in pending if j["variant_name"].endswith("__eval")]
    assert len(eval_jobs) == 1


def test_release_returns_job_to_pending(client):
    claim_r = client.get("/claim?machine=worker1&gpu_tier=1")
    job = claim_r.get_json()["job"]
    queue_file_name = claim_r.get_json().get("queue_file", "TRAINING_QUEUE.json")

    r = client.post("/release", json={
        "variant_name": job["variant_name"],
        "queue_file": queue_file_name,
    })
    assert r.status_code == 200

    status = client.get("/status").get_json()
    released = next(j for j in status if j["variant_name"] == job["variant_name"])
    assert released["status"] == "pending"
    assert released["machine"] is None


def test_claim_returns_idle_when_no_matching_jobs(client):
    # Claim all available jobs for gpu_tier=0
    client.get("/claim?machine=w1&gpu_tier=0")
    # Claim again — should be idle (only tier-1 job left for a tier-0 worker)
    r = client.get("/claim?machine=w2&gpu_tier=0")
    data = r.get_json()
    assert data["status"] == "idle"


def test_heartbeat_updates_timestamp(client):
    claim_r = client.get("/claim?machine=worker1&gpu_tier=1")
    job = claim_r.get_json()["job"]
    queue_file_name = claim_r.get_json().get("queue_file", "TRAINING_QUEUE.json")

    r = client.post("/heartbeat", json={
        "variant_name": job["variant_name"],
        "queue_file": queue_file_name,
    })
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd PrecipModels && pytest tests/test_queue_server.py -v
```

Expected: `FAIL` — `queue_server.py` does not exist

- [ ] **Step 3: Implement `queue_server.py`**

Create `PrecipModels/queue_server.py`:

```python
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

    # ── Queue loading ──────────────────────────────────────────────────────

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
                    continue   # preserve in-memory status on reload
                # Read persisted status from the entry itself (written by _save_queue).
                # This survives server restarts and multi-machine deployments where
                # metrics files may only exist on remote machines / Dropbox.
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
        """Write in-memory specs (with persisted _status/_machine) to disk and rclone copy."""
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

    # ── Helpers ───────────────────────────────────────────────────────────

    def _check_timeouts():
        """Return timed-out running jobs to pending. Call inside _lock."""
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
        """Resolve the full queue file path key used in _jobs.

        Workers receive only the basename from /claim (e.g. 'TRAINING_QUEUE.json').
        This function matches that basename against the full path keys in _jobs.
        """
        for (qf, vn) in _jobs:
            if vn != variant_name:
                continue
            # Match if queue_file is the full path OR just the filename
            if not queue_file or qf == queue_file or Path(qf).name == queue_file:
                return qf
        return ""

    # ── Endpoints ─────────────────────────────────────────────────────────

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
            # Auto-insert eval job for completed train jobs
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

    # ── Startup ───────────────────────────────────────────────────────────
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
    """Pull all QUEUE_*.json and TRAINING_QUEUE.json from Dropbox."""
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
    parser.add_argument("--queues_dir", default=".",
                        help="Directory containing TRAINING_QUEUE.json and QUEUE_*.json files")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--timeout", type=int, default=30,
                        help="Minutes before a running job times out and returns to pending")
    parser.add_argument("--machine", default="oracle_vm",
                        help="Identifier for this coordinator machine")
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
```

- [ ] **Step 4: Install Flask (if not already installed)**

```bash
pip install flask requests
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd PrecipModels && pytest tests/test_queue_server.py -v
```

Expected: all 8 tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add queue_server.py tests/test_queue_server.py
git commit -m "feat: add queue_server.py distributed job coordinator"
```

---

## Task 4: `worker.py` — Polling Loop

**Files:**
- Create: `worker.py`
- Create: `worker_config.json.example`
- Create: `tests/test_worker.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_worker.py
"""Tests for worker.py path construction and crash recovery logic."""
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def worker_args(tmp_path):
    """Minimal args namespace matching worker.py argparse output."""
    from argparse import Namespace
    return Namespace(
        server="http://localhost:5000",
        machine="test_pc",
        gpu_tier=1,
        poll_interval=1,
        output_base=str(tmp_path),
        dropbox_remote="",
        config=None,
    )


def test_output_path_construction(worker_args, tmp_path):
    """local_path = Path(output_base) / output_dir / variant_name"""
    from worker import _local_output_path
    job = {"variant_name": "ar_vae", "output_dir": "outputs_sabesp", "job_type": "train"}
    result = _local_output_path(job, worker_args.output_base)
    expected = tmp_path / "outputs_sabesp" / "ar_vae"
    assert result == expected


def test_eval_output_path_uses_model_dir(worker_args, tmp_path):
    """Eval jobs use model_dir to find the output path."""
    from worker import _local_output_path
    job = {
        "variant_name": "ar_vae__eval",
        "job_type": "eval",
        "model_dir": "outputs_sabesp/ar_vae",
        "output_dir": "outputs_sabesp",
    }
    result = _local_output_path(job, worker_args.output_base)
    expected = tmp_path / "outputs_sabesp" / "ar_vae"
    assert result == expected


def test_crash_recovery_train_skips_if_model_exists(worker_args, tmp_path):
    """If model.pt exists, train step is skipped (already trained before crash)."""
    from worker import run_job

    job = {
        "variant_name": "test_model",
        "model": "vae",
        "job_type": "train",
        "output_dir": "outputs",
    }
    # Simulate model already trained
    out_dir = tmp_path / "outputs" / "test_model"
    out_dir.mkdir(parents=True)
    (out_dir / "model.pt").write_bytes(b"fake")

    with patch("worker.subprocess.run") as mock_run:
        success = run_job(job, worker_args)

    assert success is True
    mock_run.assert_not_called()   # train.py must NOT be invoked


def test_crash_recovery_eval_skips_if_metrics_newer_than_model(worker_args, tmp_path):
    """If metrics.json is newer than model.pt, eval step is skipped."""
    from worker import run_job

    job = {
        "variant_name": "test_model__eval",
        "job_type": "eval",
        "model_dir": "outputs/test_model",
        "output_dir": "outputs",
    }
    out_dir = tmp_path / "outputs" / "test_model"
    out_dir.mkdir(parents=True)
    model_pt = out_dir / "model.pt"
    metrics_json = out_dir / "metrics.json"
    model_pt.write_bytes(b"fake_model")
    time.sleep(0.01)
    metrics_json.write_text('{"ks_stat": 0.1}')  # written after model.pt

    with patch("worker.subprocess.run") as mock_run:
        success = run_job(job, worker_args)

    assert success is True
    mock_run.assert_not_called()  # evaluate.py must NOT be invoked


def test_train_job_calls_train_py_with_skip_eval(worker_args, tmp_path):
    """Train job must invoke train.py --skip_eval."""
    from worker import run_job

    job = {
        "variant_name": "new_vae",
        "model": "vae",
        "job_type": "train",
        "output_dir": "outputs",
        "max_epochs": 5,
    }
    with patch("worker.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        run_job(job, worker_args)

    call_args = mock_run.call_args[0][0]   # first positional arg = command list
    assert "--skip_eval" in call_args
    assert "train.py" in " ".join(call_args)


def test_eval_job_calls_evaluate_py(worker_args, tmp_path):
    """Eval job must invoke evaluate.py."""
    from worker import run_job

    job = {
        "variant_name": "new_vae__eval",
        "job_type": "eval",
        "model_dir": "outputs/new_vae",
        "output_dir": "outputs",
    }
    with patch("worker.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        run_job(job, worker_args)

    call_args = mock_run.call_args[0][0]
    assert "evaluate.py" in " ".join(call_args)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd PrecipModels && pytest tests/test_worker.py -v
```

Expected: `FAIL` — `worker.py` does not exist

- [ ] **Step 3: Implement `worker.py`**

Create `PrecipModels/worker.py`:

```python
"""
worker.py — Distributed training job poller.

Polls queue_server.py for jobs, executes train.py or evaluate.py,
syncs outputs to Dropbox, and reports completion.

Usage (from PrecipModels/):
    python worker.py --server http://oracle-ip:5000 --machine home_pc --gpu_tier 1
    python worker.py --config worker_config.json

Crash recovery: if current_job.json exists on startup, the previous job was
interrupted. Recovery checks for model.pt / metrics.json and resumes or skips.
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

# All train.py params that can appear in queue entries
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
    For eval jobs, variant is derived from model_dir (last path component).
    """
    output_dir = job.get("output_dir", "outputs")
    if job.get("job_type") == "eval":
        variant = Path(job["model_dir"]).name
    else:
        variant = job["variant_name"]
    return Path(output_base) / output_dir / variant


def _rclone_copy(src: str, dst: str):
    """rclone copy src to dst. Logs warning on failure but does not raise."""
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
            print(f"[worker] {job['variant_name']}: model.pt exists, skipping training (crash recovery)")
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
            print(f"[worker] {job['variant_name']}: metrics.json up-to-date, skipping eval (crash recovery)")
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
    parser.add_argument("--machine", help="Identifier for this machine (e.g. home_pc)")
    parser.add_argument("--gpu_tier", type=int, default=0, help="GPU capability: 0=CPU, 1=small GPU, 2=large GPU")
    parser.add_argument("--poll_interval", type=int, default=30, help="Seconds between /claim polls when idle")
    parser.add_argument("--output_base", default=".", help="Base directory containing all output folders")
    parser.add_argument("--dropbox_remote", default="", help="rclone remote path (e.g. 'dropbox:PrecipModels')")
    parser.add_argument("--config", default=None, help="JSON config file (overrides CLI defaults)")
    args = parser.parse_args()

    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        for k, v in cfg.items():
            if getattr(args, k, None) is None:
                setattr(args, k, v)

    if not args.server or not args.machine:
        parser.error("--server and --machine are required (or set in --config)")

    # ── Crash recovery ─────────────────────────────────────────────────
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
            # Return to pending so another worker or retry can pick it up
            _post(args.server, "/release", {
                "variant_name": job["variant_name"],
                "queue_file": job.get("_queue_file", ""),
            })
        CURRENT_JOB_FILE.unlink(missing_ok=True)

    # ── Clean shutdown handler ─────────────────────────────────────────
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

    # ── Main polling loop ──────────────────────────────────────────────
    while True:
        response = _get(args.server, "/claim", {"machine": args.machine, "gpu_tier": args.gpu_tier})
        if response is None or response.get("status") == "idle":
            time.sleep(args.poll_interval)
            continue

        job = response["job"]
        job["_queue_file"] = response.get("queue_file", "")
        CURRENT_JOB_FILE.write_text(json.dumps(job, indent=2))
        print(f"[worker] Claimed: {job['variant_name']} (type={job.get('job_type','train')})")

        # Heartbeat thread — sends keepalive every 60s
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
                print(f"[worker] Failed: {job['variant_name']} — releasing back to pending")
                _post(args.server, "/release", {
                    "variant_name": job["variant_name"],
                    "queue_file": job.get("_queue_file", ""),
                })
        finally:
            stop_hb.set()
            CURRENT_JOB_FILE.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create `worker_config.json.example`**

```json
{
  "server": "http://ORACLE_VM_IP:5000",
  "machine": "home_pc",
  "gpu_tier": 1,
  "poll_interval": 30,
  "output_base": "C:/Users/yourname/Dropbox/PrecipModels",
  "dropbox_remote": ""
}
```

Note: Windows machines set `output_base` to the Dropbox folder and leave `dropbox_remote` empty (Dropbox client handles sync). Oracle VM sets `output_base` to a local path and sets `dropbox_remote` to `"dropbox:PrecipModels"`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd PrecipModels && pytest tests/test_worker.py -v
```

Expected: all 6 tests `PASSED`

- [ ] **Step 6: Commit**

```bash
git add worker.py worker_config.json.example tests/test_worker.py
git commit -m "feat: add worker.py distributed training poller with crash recovery"
```

---

## Task 5: `make_queue.py` updates + queue file migration

**Files:**
- Modify: `make_queue.py` (argparser and `_base_entry()` / `build_tier0()` / `build_tier1()`)
- Modify: `TRAINING_QUEUE.json`
- Modify: any other `QUEUE_*.json` files
- Create: `tests/test_make_queue.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_make_queue.py
"""Tests for make_queue.py --gpu_tier and --output_dir flags."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture
def dat_file(tmp_path):
    """Create a minimal .dat file with a header of 5 stations."""
    dat = tmp_path / "test.dat"
    dat.write_text("date,s1,s2,s3,s4,s5\n2020-01-01,0,1,0,0,0\n")
    return dat


def test_gpu_tier_default_is_1(dat_file, tmp_path):
    out = tmp_path / "queue.json"
    subprocess.run(
        [sys.executable, "make_queue.py",
         "--data_path", str(dat_file),
         "--prefix", "t",
         "--output", str(out)],
        check=True,
    )
    entries = json.loads(out.read_text())
    assert all(e.get("gpu_tier", 1) == 1 for e in entries)


def test_gpu_tier_flag_sets_value(dat_file, tmp_path):
    out = tmp_path / "queue.json"
    subprocess.run(
        [sys.executable, "make_queue.py",
         "--data_path", str(dat_file),
         "--prefix", "t",
         "--gpu_tier", "2",
         "--output", str(out)],
        check=True,
    )
    entries = json.loads(out.read_text())
    assert all(e["gpu_tier"] == 2 for e in entries)


def test_output_dir_default_is_outputs(dat_file, tmp_path):
    out = tmp_path / "queue.json"
    subprocess.run(
        [sys.executable, "make_queue.py",
         "--data_path", str(dat_file),
         "--prefix", "t",
         "--output", str(out)],
        check=True,
    )
    entries = json.loads(out.read_text())
    assert all(e.get("output_dir", "outputs") == "outputs" for e in entries)


def test_output_dir_flag_sets_value(dat_file, tmp_path):
    out = tmp_path / "queue.json"
    subprocess.run(
        [sys.executable, "make_queue.py",
         "--data_path", str(dat_file),
         "--prefix", "t",
         "--output_dir", "outputs_sabesp",
         "--output", str(out)],
        check=True,
    )
    entries = json.loads(out.read_text())
    assert all(e["output_dir"] == "outputs_sabesp" for e in entries)


def test_job_type_is_train(dat_file, tmp_path):
    out = tmp_path / "queue.json"
    subprocess.run(
        [sys.executable, "make_queue.py",
         "--data_path", str(dat_file),
         "--prefix", "t",
         "--output", str(out)],
        check=True,
    )
    entries = json.loads(out.read_text())
    assert all(e.get("job_type", "train") == "train" for e in entries)
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd PrecipModels && pytest tests/test_make_queue.py -v
```

Expected: `FAIL` — `--gpu_tier` and `--output_dir` are not recognized

- [ ] **Step 3: Add flags to `make_queue.py` argparser**

In `make_queue.py`'s `main()`, add after the `--tiers` argument:

```python
    p.add_argument("--gpu_tier", type=int, default=1, choices=[0, 1, 2],
                   help="GPU tier for all generated entries (0=CPU, 1=small GPU, 2=large GPU)")
    p.add_argument("--output_dir", default="outputs",
                   help="Output directory field for all generated entries (e.g. 'outputs_sabesp')")
```

- [ ] **Step 4: Thread `gpu_tier` and `output_dir` through entry builders**

In `make_queue.py`, update `_base_entry()` signature and body:

```python
def _base_entry(model: str, prefix: str, data_path: str, arch: dict,
                gpu_tier: int = 1, output_dir: str = "outputs") -> dict:
    entry = {
        "variant_name": f"{prefix}_{model}",
        "model":        model,
        "data_path":    data_path,
        "job_type":     "train",
        "gpu_tier":     gpu_tier,
        "output_dir":   output_dir,
    }
    # ... rest of body unchanged
```

Update `build_tier0()` and `build_tier1()` to accept and pass through `gpu_tier` and `output_dir`:

```python
def build_tier0(prefix, data_path, arch, gpu_tier=1, output_dir="outputs"):
    return [_base_entry(m, prefix, data_path, arch, gpu_tier, output_dir) for m in _TIER0_MODELS]

def build_tier1(prefix, data_path, arch, gpu_tier=1, output_dir="outputs"):
    # ... update all entry dicts to include "job_type": "train", "gpu_tier": gpu_tier, "output_dir": output_dir
```

Pass `args.gpu_tier` and `args.output_dir` when calling `build_tier0()` / `build_tier1()` in `main()`.

- [ ] **Step 5: Migrate existing queue files**

Add `"job_type": "train"`, `"gpu_tier": 1`, `"output_dir": "outputs"` to all entries in `TRAINING_QUEUE.json` and any other `QUEUE_*.json` files. Entries without these fields default correctly at runtime, but explicit fields make intent clear.

Run this one-liner to check what files exist:
```bash
ls QUEUE_*.json TRAINING_QUEUE.json 2>/dev/null
```

Edit each file: add the three fields to every entry. Existing queue entries use `"output_dir": "outputs"` unless they are SABESP-specific, in which case use `"output_dir": "outputs_sabesp"`. Check `data_path` in each entry to determine the correct output dir.

- [ ] **Step 6: Run tests to verify they pass**

```bash
cd PrecipModels && pytest tests/test_make_queue.py -v
```

Expected: all 5 tests `PASSED`

- [ ] **Step 7: Run full test suite**

```bash
cd PrecipModels && pytest tests/ -v
```

Expected: all tests pass

- [ ] **Step 8: Commit**

```bash
git add make_queue.py TRAINING_QUEUE.json tests/test_make_queue.py
git commit -m "feat(make_queue): add --gpu_tier and --output_dir flags; migrate existing queue files"
```

---

## Task 6: End-to-End Smoke Test

Verify the full distributed flow works locally before deploying to real machines.

- [ ] **Step 1: Start the queue server locally**

```bash
cd PrecipModels
python queue_server.py --queues_dir . --port 5000 --machine local_test --timeout 5
```

Expected output: `[server] Starting on port 5000 | queues_dir=. | machine=local_test`

- [ ] **Step 2: Check status endpoint**

```bash
curl http://localhost:5000/status | python -m json.tool | head -30
```

Expected: JSON list of jobs with `"status": "pending"`

- [ ] **Step 3: Run a worker for one job (CPU tier)**

In a second terminal:

```bash
cd PrecipModels
python worker.py \
  --server http://localhost:5000 \
  --machine local_worker \
  --gpu_tier 0 \
  --output_base . \
  --poll_interval 5
```

Worker should claim a tier-0 job, train it with `--skip_eval`, then wait for the auto-inserted eval job to appear and claim that too.

- [ ] **Step 4: Verify outputs**

After the worker completes a job cycle:

```bash
# Should show model.pt, config.json, metrics.json
ls outputs/<variant_name>/
```

- [ ] **Step 5: Check final queue status**

```bash
curl http://localhost:5000/status | python -m json.tool | grep '"status"'
```

Expected: the tested variant shows `"done"`, its eval variant shows `"done"`

- [ ] **Step 6: Commit smoke test results (if any config changes were needed)**

```bash
git add -p   # stage only intentional changes
git commit -m "chore: verified distributed training smoke test"
```

---

## Setup Notes (not code — reference for deployment)

**Oracle VM:**
```bash
pip install flask requests
# Install rclone and configure Dropbox:
# https://rclone.org/dropbox/
rclone config   # follow prompts to add "dropbox" remote
python queue_server.py --machine oracle_vm --dropbox_remote "dropbox:PrecipModels"
python worker.py --config worker_config.json  # in a second terminal
```

**Windows machines (home PC / company PC):**
```bash
pip install requests
# Copy worker_config.json.example → worker_config.json, fill in values
# output_base = path to your Dropbox/PrecipModels folder
# dropbox_remote = ""  (Dropbox client handles sync)
python worker.py --config worker_config.json
```

**Failover (if Oracle VM goes down permanently):**
```bash
# On any other machine with rclone configured:
python queue_server.py --machine home_pc --dropbox_remote "dropbox:PrecipModels"
# Update worker_config.json on all workers: change "server" to new coordinator IP
```
