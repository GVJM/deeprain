# Design: Parallel Tier 2 Rollouts in `compare_ar.py`

**Date:** 2026-03-24
**Status:** Approved
**Scope:** `PrecipModels/compare_ar.py` only

---

## Problem

Tier 2 in `compare_ar.py` runs all model rollouts sequentially — one model at a time. With many AR variants and slow models (ARGlow ~50s, ARRealNVP ~67s per rollout), total wall time is proportional to the number of variants. Each rollout is independent of all others, making this embarrassingly parallelizable between models.

The intra-model day-by-day loop within `sample_rollout` remains inherently sequential (each day depends on the previous day's window output) and is not in scope.

---

## Goal

Run 2–4 tier 2 rollouts in parallel via `--n_workers N`, reducing wall time roughly by N. Default `--n_workers 1` preserves current behavior exactly.

---

## Approach: `ProcessPoolExecutor` with per-variant worker function

Each worker process independently handles the full pipeline for one model:
`load_ar_model → sample_rollout → compute_tier2_metrics → save cache`

`ProcessPoolExecutor` is chosen over `ThreadPoolExecutor` because:
- Separate Python interpreters eliminate GIL contention
- Each process gets its own CUDA context — GPU time-slicing is handled by the driver
- Works identically for CPU-only runs (true multiprocessing parallelism)

---

## Implementation

### 1. New `_rollout_worker(args_tuple)` function

A module-level function (required for pickling by `multiprocessing`). Single-arg tuple for `ProcessPoolExecutor` compatibility.

**Signature:**
```python
def _rollout_worker(args_tuple) -> tuple[str, dict | None]:
```

**Unpacks:**
```
(variant, output_dir, data_norm, std, data_raw, obs_months,
 n_days, n_scenarios, device_str, force)
```

**Steps:**
1. Reconstruct `device = torch.device(device_str)`
2. Call `run_rollout(variant, ...)` — returns `sc_mm` or `None`; the function already saves `sc_mm` to `outputs/<variant>/scenarios/scenarios.npy`
3. If `sc_mm` is not `None`: call `compute_tier2_metrics(sc_mm, data_raw, obs_months)`
4. Return `(variant, tier2_dict)` or `(variant, None)` on any failure

The worker does **not** return `sc_mm` across the process boundary. Since `run_rollout` already caches to disk, the main process loads `sc_mm` from disk after collecting the tier2 metrics dict. This avoids pickling large `(n_scenarios, n_days, S)` arrays through the inter-process channel.

Exceptions are caught and logged; worker always returns a valid 2-tuple.

**Important constraint:** `_rollout_worker` must not call any function that reads the module-level `args_global`. On Windows `spawn`, each worker process imports the module fresh and `args_global` remains `None` in the worker. All required data (`output_dir`, `data_norm`, etc.) must be passed explicitly in the args tuple.

### 2. New `--n_workers` CLI argument

```python
parser.add_argument("--n_workers", type=int, default=1,
                    help="Number of parallel rollout workers (default: 1 = sequential). "
                         "On a single GPU, --n_workers 2 is recommended; higher values "
                         "share VRAM and may OOM. On CPU-only, can match physical core count.")
```

### 3. Changes to `main()` rollout loop

**Before (lines ~2454–2474):**
```python
for variant in ckpt_variants:
    sc_mm = run_rollout(...)
    t2 = compute_tier2_metrics(sc_mm, data_raw, obs_months)
    scenarios_by_model[variant] = sc_mm
    tier2_metrics[variant] = t2
```

**After:**
```python
if args.n_workers == 1:
    # unchanged sequential path
    for variant in ckpt_variants:
        ...
else:
    worker_args = [
        (v, args.output_dir, data_norm, std, data_raw, obs_months,
         args.n_days, args.n_scenarios, str(device), args.force_rollouts)
        for v in ckpt_variants
    ]
    with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
        futures = {pool.submit(_rollout_worker, a): a[0] for a in worker_args}
        for fut in as_completed(futures):
            variant, t2 = fut.result()
            if t2 is None:
                continue
            # Load sc_mm from the cache written by the worker
            cache_path = Path(args.output_dir) / variant / "scenarios" / "scenarios.npy"
            sc_mm = np.load(str(cache_path))[:args.n_scenarios, :args.n_days, :]
            scenarios_by_model[variant] = sc_mm
            tier2_metrics[variant] = t2
            print(f"  [done] {variant}: ACF={t2['multi_lag_acf_rmse']:.4f} ...")

    # Re-sort to original ckpt_variants order for deterministic plot output
    scenarios_by_model = {v: scenarios_by_model[v] for v in ckpt_variants
                          if v in scenarios_by_model}
    tier2_metrics = {v: tier2_metrics[v] for v in ckpt_variants
                     if v in tier2_metrics}
```

`as_completed` lets results be printed as each model finishes. After all futures complete, both dicts are re-sorted to match `ckpt_variants` order so that downstream plot legend ordering is deterministic and consistent with the sequential path.

### 4. Import additions

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
```

---

## Edge Cases

| Scenario | Behavior |
|---|---|
| Worker crashes (OOM, load error, rollout fail) | Returns `(variant, None)`; main skips that variant (same as current) |
| GPU OOM with N workers | Worker returns None; user reduces `--n_workers` |
| Cache hit | `run_rollout` returns cached array immediately — worker still completes quickly |
| `n_workers=1` | Exact current behavior, no `ProcessPoolExecutor` created |
| Windows (default `spawn` start method) | Compatible with CUDA — no `set_start_method` call needed |
| `as_completed` non-deterministic ordering | Re-sort after collection restores `ckpt_variants` order |
| `args_global` is `None` in workers | Workers never call functions that read `args_global`; all inputs passed explicitly |

---

## Non-Goals

- No automatic fallback from GPU to CPU on OOM
- No intra-model parallelism (day-by-day loop stays sequential)
- No changes to plotting, metrics computation, or output structure
- No changes to `--n_workers` semantics for Tier 1

---

## Files Changed

- `PrecipModels/compare_ar.py`: add `_rollout_worker`, `--n_workers` arg, parallel branch in `main()`

No other files touched.
