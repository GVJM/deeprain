# DeepRain-Mix

Precipitation scenario generation using generative ML models. Compares multiple architectures (VAE, flows, copulas, diffusion) on daily rainfall data from INMET/SABESP stations.

## Working Directory

All commands must be run from `PrecipModels/`:
```bash
cd PrecipModels/
```

## Key Commands


```bash
# Train a single model
python train.py --model <name>
python train.py --model vae --max_epochs 200 --lr 0.001 --latent_size 64
python train.py --model hurdle_simple --lr 0.0001 --name hs_lr0001

# AR/temporal models — require SABESP data path (data_precip.dat = 17 stations, most experiments)
python train.py --model ar_vae --data_path ../dados_sabesp/data_precip.dat --max_epochs 300
python train.py --model ar_flow_match --data_path ../dados_sabesp/data_precip.dat

# AR regularization flags (all default to 0/disabled for backward compat)
python train.py --model ar_flow_match --data_path ../dados_sabesp/data_precip.dat \
  --ema_decay 0.999 --dropout 0.05 --input_noise_std 0.05 --weight_decay 1e-4 \
  --lr_schedule cosine

# Train all models + compare
python compare.py
python compare.py --max_epochs 200 --n_samples 1000   # quick test
python compare.py --skip_training                      # re-run analysis on saved metrics

# Compare all trained AR models (Tier 1 + Tier 2)
python compare_ar.py --skip_rollouts                          # Tier 1 only (<1 min, no GPU)
python compare_ar.py --models ar_vae ar_mean_flow --n_days 30 --n_scenarios 5  # quick Tier 2 test
python compare_ar.py --n_days 365 --n_scenarios 50            # full run
python compare_ar.py --skip_rollouts --output_dir results/data_precip  # discovers models in all subdirs recursively
python compare_ar.py --n_workers 4 --n_days 365 --n_scenarios 50    # parallel Tier 2 rollouts (multi-GPU)
python compare_ar.py --force_rollouts --models ar_vae               # recompute rollouts even if cached
python compare_ar.py --top_n_per_family 3                           # keep only top-3 per family in charts
python compare_ar.py --recompute_tier1 --n_samples 1000 --models ar_vae  # re-eval Tier 1 with 1000 samples (overwrites metrics.json)
python compare_ar.py --recompute_tier1 --n_samples 1000 --skip_rollouts  # re-eval all models, Tier 1 only

# Ablation study for hurdle_vae_cond
python ablation_hurdle_vae_cond.py --skip_training            # re-plot saved results
python ablation_hurdle_vae_cond.py --max_epochs 500 --n_samples 2000  # full ablation

# Batch training queue
python batch_train.py                              # run all pending jobs in TRAINING_QUEUE.json
python batch_train.py --queue_dir queues/          # merge all QUEUE*.json files in queues/ folder
python batch_train.py --queue queues/QUEUE_foo.json # single alternate file
python batch_train.py --queue_dir queues/ --status  # status table across all queues
python batch_train.py --only ar_vae --status       # filter to specific variant
# Queue log written to PrecipModels/batch_log.jsonl

# Queue generator — auto-sizes architecture for a new dataset
python make_queue.py --data_path ../dados_sabesp/data_precip.dat \
                     --prefix dp24 --output QUEUE_precip24.json
python make_queue.py --data_path ../dados_sabesp/data_precip.dat \
                     --prefix dp24 --tiers 0 1 --append TRAINING_QUEUE.json

# Validation and visualization
python validate_holdout.py
python plot_model.py
python generate_scenarios.py --model ar_vae --n_scenarios 50 --n_days 365
```

Outputs go to `outputs/<name>/`: `model.pt` (or `copula.pkl`), `config.json`, `metrics.json`.

## Results & Queue Organization

```
PrecipModels/
├── queues/                          # all QUEUE*.json files (use --queue_dir queues/)
│   ├── QUEUE_dp_v3_glow.json        # naming: QUEUE_<dataset>_<version>_<family>.json
│   ├── QUEUE_dp_v6_latentfm.json    # dp = data_precip, dp24 = 24-station variant
│   └── LEGACY_*.json                # old flat-array format — excluded from --queue_dir
├── batch_log.jsonl                  # central log for all batch runs (grep by variant field)
└── results/
    ├── data_precip/                 # 17-station SABESP dataset (data_precip.dat)
    │   ├── merge/                   # all trained batches merged for cross-batch comparison
    │   │   └── comparison_ar/       # compare_ar.py outputs (reports, CSVs, charts)
    │   ├── outputs_v6_glow/         # per-batch output dirs (new runs go here)
    │   └── outputs_v6_latentfm/
    └── dayprecip/                   # 91-station SABESP dataset (dayprecip.dat)
        ├── outputs_sabesp/
        └── outputs_v5_res/
```

**Queue file format** — new `defaults` + `entries` format (old plain array still works):
```json
{
  "defaults": {
    "data_path": "../dados_sabesp/data_precip.dat",
    "output_dir": "results/data_precip/outputs_v6_glow",
    "holdout_ratio": 0.15, "val_ratio": 0.1, "val_freq": 5,
    "early_stop_patience": 50, "max_epochs": 2000
  },
  "entries": [
    {"variant_name": "ar_glow_h64_w30", "model": "ar_glow", "hidden_size": 64, "window_size": 30}
  ]
}
```
Entry-level fields override defaults. `output_dir` is required (no silent fallback).

**Datasets:**
- `data_precip.dat` — 17 SABESP stations (most AR experiments use this)
- `dayprecip.dat` — 91 SABESP stations (older experiments)
- **Mismatch gotcha:** running `compare_ar.py` against models trained on one dataset with the other's `--data_path` causes a shape crash in `transition_probability_error` (e.g., `(91,) vs (17,)`).

## Models

**Core:**
`copula`, `vae`, `hurdle_simple`, `hurdle_vae`, `real_nvp`, `glow`, `flow_match`, `flow_match_film`, `latent_flow`, `ldm`, `hurdle_flow`

**Month-conditional (`_mc`):**
`hurdle_simple_mc`, `vae_mc`, `real_nvp_mc`, `glow_mc`, `flow_match_mc`, `flow_match_film_mc`, `latent_fm_mc`, `hurdle_latent_fm_mc`, `hurdle_vae_cond_mc`

**Thresholded:**
`thresholded_latent_fm_mc`, `thresholded_vae_mc`, `thresholded_real_nvp_mc`, `thresholded_glow_mc`

**Other variants:**
`hurdle_vae_cond`, `hurdle_vae_cond_nll`

**Autoregressive (`ar_*`):**
`ar_vae`, `ar_vae_v2`, `ar_flow_match`, `ar_latent_fm`
`ar_real_nvp`, `ar_real_nvp_lstm`, `ar_glow`, `ar_glow_lstm`
`ar_mean_flow`, `ar_mean_flow_lstm`, `ar_mean_flow_v2`, `ar_mean_flow_ayfm`
`ar_flow_map`, `ar_flow_map_lstm`, `ar_flow_map_ms`, `ar_flow_map_sd`, `ar_flow_map_res`
`ar_ddpm`, `ar_ddpm_lstm`

**Temporal/Hurdle-Temporal:**
`hurdle_temporal`

Full registry: `PrecipModels/models/__init__.py`

## Running Tests

```bash
conda run -n pytorch_env python -m pytest tests/ -v
```

- `pytorch_env` is the conda env with PyTorch + flask + requests
- Tests cover: train --skip_eval, evaluate.py, queue_server, worker, make_queue

## Architecture

```
PrecipModels/
├── train.py            # unified training entry point
├── evaluate.py         # standalone evaluator (used by distributed workers)
├── queue_server.py     # distributed job coordinator (run on Oracle VM)
├── worker.py           # distributed training worker (run on each machine)
├── worker_config.json.example  # worker configuration template
├── compare.py          # train all models + comparative analysis
├── compare_ar.py       # AR model Tier 1 + Tier 2 comparison (rollouts, seasonal, extremes)
├── validate_holdout.py # holdout validation
├── plot_model.py       # visualize model outputs
├── data_utils.py       # data loading, normalization
├── base_model.py       # BaseModel interface
├── metrics.py          # evaluation metrics
├── datasets.py         # TemporalDataset, TemporalCondDataset
├── generate_scenarios.py # AR scenario rollout + evaluation
├── batch_train.py      # queue-based batch training runner
├── make_queue.py       # generate TRAINING_QUEUE.json for new datasets (auto-sizes arch)
├── MODEL_DEFAULTS.json # per-model hyperparameter defaults
├── models/             # all model implementations
│   ├── __init__.py     # model registry + get_model()
│   ├── conditioning.py # monthly conditioning utils
│   └── *.py            # individual model files
├── ar/                 # AR model utilities
│   ├── loader.py       # load_ar_model() — reads config.json + checkpoint
│   ├── plots.py        # AR-specific plotting helpers
│   └── reports.py      # Tier 1/2 report generation
└── outputs/            # training artifacts (gitignored)

dados/                  # INMET data (relative to repo root)
dados_sabesp/           # SABESP data
dados_barragens_btg/    # BTG dam data
```

## Data Paths (relative to `PrecipModels/`)

- INMET: `../dados/inmet_relevant_data.csv`
- SABESP: `../dados_sabesp/`
- BTG dams: `../dados_barragens_btg/`

## Common Gotchas

- **AR evaluation bottleneck:** `evaluate_model()` calls `sample(n)` 6× total; for AR models this is O(n) sequential steps. Mitigated in `train.py` (200 samples, 1 timing trial for `_TEMPORAL_MODELS`). Change caps in the `is_temporal` block ~line 1192.
- **AR `sample()` progress:** All AR models log progress every 25% of steps (`flush=True`). Do not add duplicate logging.
- **Batch log:** `PrecipModels/batch_log.jsonl` — one JSON record per job, appended atomically. Survives crashes; grep by `variant` field.
- **KL collapse (VAE):** increase `--kl_warmup` (try 100+)
- **RealNVP NLL explosion:** use `--lr 0.0001`
- **Flow Matching negative samples:** clip to 0 post-generation
- **HurdleSimple direct use:** call `fit_copulas()` before generating
- **Working directory:** data paths break if not run from `PrecipModels/`
- `KMP_DUPLICATE_LIB_OK=TRUE` is set automatically in train.py (Intel MKL conflict)
- **AR models require SABESP path:** default data path is INMET; always pass `--data_path ../dados_sabesp/data_precip.dat` for AR/temporal models (17-station dataset; use `dayprecip.dat` only for older 91-station experiments)
- **ARGlow inverse cache:** `_InvLinearLU.inverse()` caches `W⁻¹`; cleared on `train()`. Without it, evaluation does 8000 matrix inversions instead of 8
- **`compare_ar.py` output_dir:** AR checkpoints default to `./outputs`; always pass `--output_dir` matching where models actually live (e.g., `--output_dir results/data_precip/merge`). Also pass `--data_path` matching the training dataset or Tier 2 rollouts will crash with a shape mismatch.
- **st_362 outlier station:** Wasserstein 50–100× higher than all other 91 SABESP stations across every model. Excluded from per-station win ranking but shown in heatmaps. Constant `OUTLIER_STATION` in `compare_ar.py`.
- **`hurdle_temporal` is NOT in `_TEMPORAL_MODELS`** (`train.py` ~line 143): uses a different training path; AR evaluation caps do not apply to it.
- **`--data_path` string-matched against `DEFAULT_DATA_PATH`:** only the exact string `"../dados/inmet_relevant_data.csv"` (or `"../dados_barragens_btg/inmet_relevant_data.csv"`) triggers the BTG/INMET loader; everything else routes to SABESP. Absolute paths work correctly since ceb793c. Relative paths still require running from `PrecipModels/`.
- **New conditional `nn.Parameter` in AR models requires 3 coordinated changes:** (1) guard creation with `if hyperparam > 0` in model `__init__`, (2) save hyperparam to config dict in `train.py` (~line 824), (3) add to float-param loop in `compare_ar.py`'s `load_ar_model()`. Missing any one causes checkpoint load failure on rollout.
- **`compare_ar.py` rollout loader:** reads int arch params automatically but float params (`occ_weight`, `jvp_eps`, `mf_ratio`) must be explicitly listed in the float-param loop (~line 240).
- **Distributed training:** `queue_server.py` runs on Oracle VM; `worker.py` polls it. Queue state persisted in `TRAINING_QUEUE.json`/`QUEUE_*.json` (`_status` field). Use `--skip_eval` on workers — eval jobs auto-inserted at gpu_tier=0. See `docs/superpowers/specs/2026-03-18-distributed-training-design.md`.
- **evaluate.py float params:** Same rule as compare_ar.py — `occ_weight`, `jvp_eps`, `mf_ratio` must be cast as `float()` when loading from config.json.
- **`ar_flow_map_ms` is in `OUTLIER_MODELS`:** excluded from top-level Tier 2 scenario charts (but kept in Tier 1 tables and `families/` subdirs). Constant `OUTLIER_MODELS` in `compare_ar.py`. Add models here when pathological outputs distort shared axes.
- **Tier 2 performance at n_scenarios > 200:** metric and plot functions are vectorized (cumsum rolling sum, diff-based RLE, searchsorted exceedance). Per-family plots log per-step timing via `_tplot` — slow `seasonal` or `rxnday` steps indicate a regression.
- **New AR models need entries in BOTH `ARCH_DEFAULTS` (train.py) AND `MODEL_DEFAULTS.json`:** `ARCH_DEFAULTS` controls architecture hyperparameters; `MODEL_DEFAULTS.json` controls training hyperparameters (lr, batch_size, max_epochs, normalization_mode). Missing either causes `KeyError: 'model_name'` at line ~769 in train.py.
- **`ar_flow_map_res` uses residual parameterization:** output = `z_s + (t-s) * MLP(...)` instead of `MLP(...)`. This fixes stochastic collapse (MSE → conditional mean ≈ 0 on sparse data) by forcing the output to depend on `z_s` structurally. `use_residual=True` must be in `config.json` for checkpoint loading — it's a bool param handled in `ar/loader.py` and `evaluate.py` `_BOOL_PARAMS`.
- **`ar_ddpm` teacher model:** VP-SDE with v-prediction and continuous time. Implements `velocity()` interface for AYF distillation compatibility. Trained via `_TEMPORAL_MODELS` path. Schedule: β(t) = 0.1 + 19.9t, α̅(t) = exp(-(0.1t + 9.95t²)). Sampling: DDIM deterministic, t=1→0 in n_sample_steps steps.
- **EMA checkpoint transparency:** EMA saves the averaged model's `state_dict` directly into `model.pt` — no wrapper object. `evaluate.py` and `ar/loader.py` load it identically to non-EMA checkpoints; no special handling needed at load time.
- **Dropout backward compatibility:** AR model dropout uses `if dropout > 0.0` guards before inserting `nn.Dropout` layers. This means `dropout=0.0` checkpoints have no Dropout keys in `state_dict` — existing checkpoints load identically. Never use `nn.Dropout(0.0)` unconditionally (adds spurious keys).
- **Windows conda multiline scripts:** `conda run -n pytorch_env python -c "..."` fails with multiline scripts on Windows. Write to a temp `.py` file, run it, then delete it.
- **`compare_ar.py` recursive discovery:** `discover_ar_models()` now recurses into subdirs — point `--output_dir` at any ancestor folder (e.g. `results/data_precip/`) and it finds models in `outputs_v6_glow/`, `outputs_v6_latentfm/`, etc. Dirs named `comparison_ar`, `scenarios`, `families`, `stations` are skipped automatically (`_SKIP_DIR_NAMES` in `ar/loader.py`). Add new output-only dir names there to keep them out of discovery.
- **`discover_ar_models()` returns `dict[str, Path]`** (not `list[str]`): maps variant name → absolute dir. Callers that previously iterated a list still work (dict iteration yields keys), but any code that builds `Path(output_dir) / variant` must now use `_resolve_model_dir(variant, output_dir, variant_dirs)` from `ar/loader.py` instead.
- **`variant_dirs` param in `ar/` functions:** `load_ar_model`, `get_family`, `load_all_metrics`, `run_rollout`, `recompute_tier1_metrics`, and the affected plot/report functions all accept `variant_dirs: dict[str, Path] = None`. New functions that load model files should follow the same pattern to support non-flat layouts.
- **`--models` filter warns on missing names:** previously bypassed discovery silently; now discovers recursively first then filters, printing `[warn]` for any name not found under `--output_dir`.
