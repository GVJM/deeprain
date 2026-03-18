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

# AR/temporal models — require SABESP data path
python train.py --model ar_vae --data_path ../dados_sabesp/dayprecip.dat --max_epochs 300
python train.py --model ar_flow_match --data_path ../dados_sabesp/dayprecip.dat

# Train all models + compare
python compare.py
python compare.py --max_epochs 200 --n_samples 1000   # quick test
python compare.py --skip_training                      # re-run analysis on saved metrics

# Compare all trained AR models (Tier 1 + Tier 2)
python compare_ar.py --skip_rollouts                          # Tier 1 only (<1 min, no GPU)
python compare_ar.py --models ar_vae ar_mean_flow --n_days 30 --n_scenarios 5  # quick Tier 2 test
python compare_ar.py --n_days 365 --n_scenarios 50            # full run
python compare_ar.py --skip_rollouts --output_dir ./outputs_sabesp  # if AR models live in outputs_sabesp/

# Ablation study for hurdle_vae_cond
python ablation_hurdle_vae_cond.py --skip_training            # re-plot saved results
python ablation_hurdle_vae_cond.py --max_epochs 500 --n_samples 2000  # full ablation

# Batch training queue
python batch_train.py           # run all pending jobs in TRAINING_QUEUE.json

# Validation and visualization
python validate_holdout.py
python plot_model.py
python generate_scenarios.py --model ar_vae --n_scenarios 50 --n_days 365
```

Outputs go to `outputs/<name>/`: `model.pt` (or `copula.pkl`), `config.json`, `metrics.json`.

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
`ar_mean_flow`, `ar_mean_flow_lstm`, `ar_mean_flow_v2`, `ar_flow_map`, `ar_flow_map_lstm`, `ar_flow_map_ms`

**Temporal/Hurdle-Temporal:**
`hurdle_temporal`

Full registry: `PrecipModels/models/__init__.py`

## Architecture

```
PrecipModels/
├── train.py            # unified training entry point
├── compare.py          # train all models + comparative analysis
├── validate_holdout.py # holdout validation
├── plot_model.py       # visualize model outputs
├── data_utils.py       # data loading, normalization
├── base_model.py       # BaseModel interface
├── metrics.py          # evaluation metrics
├── datasets.py         # TemporalDataset, TemporalCondDataset
├── generate_scenarios.py # AR scenario rollout + evaluation
├── batch_train.py      # queue-based batch training runner
├── MODEL_DEFAULTS.json # per-model hyperparameter defaults
├── models/             # all model implementations
│   ├── __init__.py     # model registry + get_model()
│   ├── conditioning.py # monthly conditioning utils
│   └── *.py            # individual model files
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

- **AR evaluation bottleneck:** `evaluate_model()` calls `sample(n)` 6× total; for AR models this is O(n) sequential steps. Mitigated in `train.py` (200 samples, 1 timing trial for `_TEMPORAL_MODELS`). Change caps in the `is_temporal` block ~line 1153.
- **AR `sample()` progress:** All AR models log progress every 25% of steps (`flush=True`). Do not add duplicate logging.
- **Batch log:** `outputs/batch_log.jsonl` — one JSON record per job, appended atomically. Survives crashes; grep by `variant` field.
- **KL collapse (VAE):** increase `--kl_warmup` (try 100+)
- **RealNVP NLL explosion:** use `--lr 0.0001`
- **Flow Matching negative samples:** clip to 0 post-generation
- **HurdleSimple direct use:** call `fit_copulas()` before generating
- **Working directory:** data paths break if not run from `PrecipModels/`
- `KMP_DUPLICATE_LIB_OK=TRUE` is set automatically in train.py (Intel MKL conflict)
- **AR models require SABESP path:** default data path is INMET; always pass `--data_path ../dados_sabesp/dayprecip.dat` for AR/temporal models
- **ARGlow inverse cache:** `_InvLinearLU.inverse()` caches `W⁻¹`; cleared on `train()`. Without it, evaluation does 8000 matrix inversions instead of 8
- **`compare_ar.py` output_dir:** AR checkpoints default to `./outputs`; if models were trained to `outputs_sabesp/`, pass `--output_dir ./outputs_sabesp`
- **st_362 outlier station:** Wasserstein 50–100× higher than all other 91 SABESP stations across every model. Excluded from per-station win ranking but shown in heatmaps. Constant `OUTLIER_STATION` in `compare_ar.py`.
- **`hurdle_temporal` is NOT in `_TEMPORAL_MODELS`** (`train.py` line 134): uses a different training path; AR evaluation caps do not apply to it.
- **`--data_path` string-matched against constants:** `data_utils.py` compares the path string to `SABESP_DATA_PATH`/`DEFAULT_DATA_PATH`; relative paths only work from `PrecipModels/`. Use absolute paths when running from worktrees or non-standard CWDs.
- **New conditional `nn.Parameter` in AR models requires 3 coordinated changes:** (1) guard creation with `if hyperparam > 0` in model `__init__`, (2) save hyperparam to config dict in `train.py` (~line 824), (3) add to float-param loop in `compare_ar.py`'s `load_ar_model()`. Missing any one causes checkpoint load failure on rollout.
- **`compare_ar.py` rollout loader:** reads int arch params automatically but float params (`occ_weight`, `jvp_eps`, `mf_ratio`) must be explicitly listed in the float-param loop (~line 240).
