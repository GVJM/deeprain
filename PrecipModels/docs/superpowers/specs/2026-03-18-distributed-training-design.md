# Distributed Training Design
**Date:** 2026-03-18
**Project:** DeepRain-Mix / PrecipModels
**Status:** Approved

---

## Problem

Training many model variants sequentially on a single machine is slow. Three machines are available (home PC with small GPU, company PC, Oracle Ubuntu VM) but are in separate locations with different network constraints. Goal: distribute training jobs across all machines in parallel, with GPU-aware routing, resilient to any machine going offline.

---

## Architecture

```
Dropbox (enterprise) — source of truth for files + queue state
├── PrecipModels/TRAINING_QUEUE.json
├── PrecipModels/QUEUE_SABESP_24/...
├── PrecipModels/outputs/<variant>/
├── PrecipModels/outputs_sabesp/<variant>/
└── PrecipModels/outputs_inmet_btg_barragens/<variant>/

Oracle VM (Ubuntu) — coordinator (replaceable)
└── queue_server.py   # HTTP API, reads/writes queue files from Dropbox via rclone

Home PC (Windows, small GPU) — worker
└── worker.py         # polls Oracle VM, trains, syncs outputs to Dropbox

Company PC (Windows) — worker
└── worker.py         # same script, different gpu_tier declaration

Oracle VM — also acts as a worker (gpu_tier=0, CPU jobs)
└── worker.py         # runs on the same machine as queue_server.py
```

All inter-machine communication is **outbound from workers to Oracle VM** (port 443 or configurable). This works behind corporate VPN.

---

## Queue Format

Each entry in any `QUEUE_*.json` file gains two new fields:

```json
{
  "variant_name": "ar_vae",
  "model": "ar_vae",
  "gpu_tier": 1,
  "job_type": "train",
  "output_dir": "outputs_sabesp",
  "data_path": "../dados_sabesp/dayprecip.dat",
  "max_epochs": 300
}
```

**`gpu_tier`** — worker capability matching:

| Tier | Meaning | Eligible workers |
|------|---------|-----------------|
| `0` | CPU only (copula, hurdle_simple, eval jobs) | Oracle VM, any machine |
| `1` | Small GPU OK | Home PC, company PC, any GPU machine |
| `2` | Prefers large GPU | Home PC (or best available, falls back to tier 1) |

Rule: a worker with declared `gpu_tier=N` can claim any job where `job.gpu_tier <= N`.

**`job_type`** — `"train"` or `"eval"`. Eval jobs are auto-inserted by the server when a train job completes (see Evaluation section).

**`output_dir`** — relative path from `PrecipModels/`. Defaults to `"outputs"` if absent (backward-compatible with existing queue files).

**`make_queue.py`** gains `--gpu_tier` (default: `1`) and `--output_dir` flags. Entries without `gpu_tier` default to `1` at runtime.

---

## Components

### `queue_server.py` (Oracle VM)

Tiny Flask HTTP server. Manages all `QUEUE_*.json` files in a configurable `--queues_dir`.

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/status` | Full queue table: variant, job_type, status, machine, started_at |
| `GET` | `/claim?machine=X&gpu_tier=N` | Atomically claim next eligible pending job; returns job JSON |
| `POST` | `/complete` | Mark job done; auto-insert eval job if job_type=train |
| `POST` | `/heartbeat` | Worker keepalive for running job |
| `POST` | `/release?variant=X` | Return job to pending (clean shutdown or manual) |

**Atomic claiming:** queue files are loaded into memory on startup and written back to disk (and synced to Dropbox via `rclone copy`) on every state change. A threading lock prevents concurrent claim races within a single server process. Only one `queue_server.py` instance should run at a time — on startup, the server writes a `coordinator_lock.json` to Dropbox containing its machine name and timestamp; if a lock file from another machine exists and is newer than 5 minutes, startup aborts with a clear error message.

**Heartbeat timeout:** if a `running` job has not received a heartbeat for `--timeout` minutes (default: 30), the server returns it to `pending` on the next `/claim` or `/status` call. This recovers from worker crashes automatically.

**Queue persistence:** after every state change, server writes the updated queue file to disk and calls `rclone copy` to push it to Dropbox. On startup, it pulls all queue files from Dropbox first via `rclone copy`.

**Multiple queues:** all `QUEUE_*.json` files in `--queues_dir` are loaded and managed together. Jobs are identified by `(queue_file, variant_name)`.

### `worker.py` (all machines)

Polling loop that claims and executes jobs.

**Configuration** (via CLI args or `worker_config.json`):
- `--server` — Oracle VM URL (e.g., `http://oracle-vm-ip:5000`)
- `--machine` — machine identifier (e.g., `home_pc`, `company_pc`, `oracle_vm`)
- `--gpu_tier` — declared GPU capability (0, 1, or 2)
- `--poll_interval` — seconds between `/claim` attempts (default: 30)
- `--output_base` — local base directory containing all output folders

**Local path construction:** `local_path = Path(output_base) / output_dir / variant_name`. On Windows machines, `output_base` is the Dropbox folder root (files sync automatically via Dropbox client). On Oracle VM, `output_base` is a local directory (synced explicitly via rclone).

**Job execution flow:**

```
1. GET /claim  →  receive job spec
2. Save spec to current_job.json  (crash recovery)
3. If job_type == "train":
     run: python train.py --skip_eval --name <variant> --output_dir <output_dir> ...
4. If job_type == "eval":
     run: python evaluate.py --model_dir <output_dir>/<variant>
5. rclone copy <local_output_dir>/<variant>/ dropbox:PrecipModels/<output_dir>/<variant>/
   (use rclone copy, not rclone sync — avoids deleting files written by other machines
    e.g. metrics.json written by an eval job on another machine)
   On Windows: Dropbox client syncs automatically; call rclone copy explicitly anyway
   to ensure files are uploaded before posting /complete.
6. POST /complete
7. Delete current_job.json
8. Go to step 1
```

**On startup:** if `current_job.json` exists, the previous job was interrupted. Recovery depends on `job_type` from `current_job.json`:
- `train`: check if `model.pt` exists in the output dir. If yes, job finished but `/complete` failed — go to step 5. If no, resume from step 3.
- `eval`: check if `metrics.json` exists AND its modification time is newer than `model.pt` (confirming eval wrote it, not a stale file from a prior run). If yes, go to step 5. If no, resume from step 4.

**On Ctrl-C / clean shutdown:** worker calls `POST /release` before exiting.

**If server unreachable:** worker retries `/complete` and sync with exponential backoff (up to 10 minutes). Training output is safe locally and in Dropbox regardless.

**If Oracle VM is permanently down:** update `--server` to point to any other machine running `queue_server.py`. Queue state is in Dropbox, so any machine can take over coordination by running `queue_server.py --queues_dir .`.

### `evaluate.py` (standalone)

Extracted from the end of `train.py`. Loads a trained model and writes `metrics.json`.

```bash
python evaluate.py --model_dir outputs_sabesp/ar_vae
python evaluate.py --model_dir outputs_sabesp/ar_vae --n_samples 1000
```

Reads `config.json` to reconstruct model architecture, loads `model.pt`, calls `evaluate_model()`, writes `metrics.json`. Accepts `--n_samples` override.

**Important:** must replicate the float-param loading logic from `compare_ar.py`'s `load_ar_model()` — specifically the explicit loop for `occ_weight`, `jvp_eps`, and `mf_ratio` float parameters. Loading these as int from `config.json` silently breaks AR models (see CLAUDE.md "compare_ar.py rollout loader" and commit `b64f0a1`). Any new float arch params added to AR models must also be added here.

### `train.py` changes

Add `--skip_eval` flag. When set, skip the `evaluate_model()` call and do not write `metrics.json`. Training ends after saving `model.pt` and `config.json`.

**Note for operators:** output directories from distributed training will contain `model.pt` and `config.json` but no `metrics.json` until the eval job completes. `batch_train.py`'s `get_status()` will show these as `"partial"` until eval finishes — this is expected behavior, not a failure.

No other changes to `train.py`, model code, or `compare_ar.py`.

### `batch_train.py` changes

None. Single-machine local training remains fully functional.

---

## Evaluation Flow (Deferred, Any-Machine)

When a worker POSTs `/complete` for a `train` job, the server auto-inserts an eval job **into the same queue file** as the originating train job:

```json
{
  "variant_name": "ar_vae__eval",
  "job_type": "eval",
  "model_dir": "outputs_sabesp/ar_vae",
  "gpu_tier": 0,
  "output_dir": "outputs_sabesp"
}
```

`variant_name` uniqueness is enforced **per queue file**. Across multiple queue files, the same base name may appear (e.g., `ar_vae` in both `TRAINING_QUEUE.json` and `QUEUE_SABESP_24.json`), but since eval jobs are inserted into the originating queue file, the `(queue_file, variant_name)` pair is globally unique.

Any available worker (including Oracle VM) claims this eval job. GPU is not required — `gpu_tier: 0` means all workers are eligible.

This means:
- GPU is freed immediately after training
- Eval runs on whichever machine is next available
- If Oracle VM is down, home PC or company PC picks up eval jobs
- Eval jobs follow the same heartbeat/timeout/release resilience as train jobs

---

## File Sync (Dropbox)

**Windows machines (home PC, company PC):**
- Dropbox client installed; `PrecipModels/` lives inside Dropbox folder
- Output folders sync automatically in the background
- No manual sync needed

**Oracle VM (Linux):**
- `rclone` configured with Dropbox credentials
- `worker.py` calls `rclone copy` explicitly after each job (not `rclone sync` — avoids deleting files written by other machines, e.g. `metrics.json` from an eval job)
- `queue_server.py` calls `rclone copy` after each queue state change

**Structure in Dropbox:**
```
PrecipModels/
├── TRAINING_QUEUE.json          ← queue state (managed by server)
├── QUEUE_SABESP_24.json
├── outputs/<variant>/
│   ├── model.pt
│   ├── config.json
│   └── metrics.json
├── outputs_sabesp/<variant>/
└── outputs_inmet_btg_barragens/<variant>/
```

Large data files (`dayprecip.dat`, CSVs) remain in the git repo, not Dropbox. Each machine is expected to have the repo cloned with data present.

---

## Resilience Matrix

| Failure scenario | Behavior |
|-----------------|----------|
| Worker crashes mid-training | Heartbeat timeout → job returns to `pending`; another worker picks it up |
| Worker loses network mid-job | Continues training locally; retries sync + `/complete` with backoff when connection returns |
| Worker Ctrl-C / clean shutdown | Calls `/release`; job immediately returns to `pending` |
| Oracle VM goes down (temporary) | Workers finish current job (training continues locally), buffer `/complete` with backoff retry. New job claims are blocked and eval jobs are not dispatched until VM returns — degraded but safe |
| Oracle VM goes down (permanent) | Any machine runs `queue_server.py`; queue state recovered from Dropbox |
| Dropbox outage | Workers train and store locally; sync resumes when Dropbox returns |
| Two workers claim same job | Prevented by threading lock in queue server |

---

## GPU Routing Examples

| Model | gpu_tier | Runs on |
|-------|----------|---------|
| `copula`, `hurdle_simple` | 0 | Oracle VM, any machine |
| `vae`, `real_nvp`, `flow_match` | 1 | Home PC, company PC |
| `ar_vae`, `ar_mean_flow` | 1 | Home PC, company PC |
| All eval jobs | 0 | Any machine |

Home PC declares `gpu_tier=1`; can run tier-0 and tier-1 jobs.
Oracle VM declares `gpu_tier=0`; runs CPU-only and eval jobs.
Company PC: declare `gpu_tier=1` if any GPU is available, `gpu_tier=0` if CPU-only.

---

## Files Changed / Created

**New:**
- `queue_server.py`
- `worker.py`
- `evaluate.py`
- `worker_config.json.example`

**Modified:**
- `train.py` — add `--skip_eval` flag
- `make_queue.py` — add `--gpu_tier` and `--output_dir` flags
- `TRAINING_QUEUE.json` and all `QUEUE_*.json` — add `gpu_tier`, `job_type`, `output_dir` fields to entries

**Unchanged:**
- `batch_train.py`, `compare_ar.py`, `compare.py`, all model code

---

## Out of Scope

- Distributed training of a single model across multiple GPUs (each job trains independently)
- Automatic repo/data sync (each machine manages its own git clone)
- Authentication on `queue_server.py` (assumed internal use; can add basic auth later)
