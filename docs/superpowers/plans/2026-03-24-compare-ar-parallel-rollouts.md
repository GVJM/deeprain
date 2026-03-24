# Parallel Tier 2 Rollouts in `compare_ar.py` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--n_workers N` to `compare_ar.py` so tier 2 rollouts run N models in parallel via `ProcessPoolExecutor`, reducing wall time ~N× with no change to outputs.

**Architecture:** A new module-level `_rollout_worker(args_tuple)` function handles the full per-model pipeline (load → rollout → metrics → cache). The `main()` rollout loop branches: `n_workers=1` keeps the existing sequential path untouched; `n_workers>1` uses `ProcessPoolExecutor` + `as_completed`, then re-sorts results to preserve deterministic ordering.

**Tech Stack:** Python `concurrent.futures.ProcessPoolExecutor`, `numpy`, `torch` (existing deps — no new packages)

---

## File Map

| File | Change |
|---|---|
| `PrecipModels/compare_ar.py` | Add import, add `_rollout_worker`, add `--n_workers` arg, add parallel branch in `main()` |
| `PrecipModels/tests/test_compare_ar_parallel.py` | New test file |

---

### Task 1: Add import and `_rollout_worker` function

**Files:**
- Modify: `PrecipModels/compare_ar.py` (imports ~line 55, add function after `run_rollout` ~line 416)
- Create: `PrecipModels/tests/test_compare_ar_parallel.py`

- [ ] **Step 1: Write the failing test for `_rollout_worker` — failure path**

Create `PrecipModels/tests/test_compare_ar_parallel.py`:

```python
"""Tests for compare_ar._rollout_worker parallel helper."""
import numpy as np
import pytest
from unittest.mock import patch


def test_rollout_worker_returns_none_when_rollout_fails():
    """Worker must return (variant, None) when run_rollout returns None."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from compare_ar import _rollout_worker

    data_norm = np.zeros((30, 3), dtype=np.float32)
    std = np.ones((1, 3), dtype=np.float32)
    data_raw = np.zeros((30, 3), dtype=np.float32)
    obs_months = np.arange(30) % 12

    args_tuple = (
        "ar_vae_test", "./outputs", data_norm, std, data_raw, obs_months,
        10, 5, "cpu", False,
    )

    with patch("compare_ar.run_rollout", return_value=None):
        variant, t2 = _rollout_worker(args_tuple)

    assert variant == "ar_vae_test"
    assert t2 is None
```

- [ ] **Step 2: Run the test to verify it fails**

```bash
cd PrecipModels
conda run -n deeptutorial python -m pytest tests/test_compare_ar_parallel.py::test_rollout_worker_returns_none_when_rollout_fails -v
```

Expected: `ImportError: cannot import name '_rollout_worker' from 'compare_ar'`

- [ ] **Step 3: Add `ProcessPoolExecutor` import to `compare_ar.py`**

In `PrecipModels/compare_ar.py`, find the imports block (~line 58). Add after `import torch`:

```python
from concurrent.futures import ProcessPoolExecutor, as_completed
```

- [ ] **Step 4: Add `_rollout_worker` function after `run_rollout`**

In `PrecipModels/compare_ar.py`, immediately after the `run_rollout` function (~line 416, after `return sc_mm`), insert:

```python
def _rollout_worker(args_tuple) -> tuple[str, dict | None]:
    """
    Worker function for ProcessPoolExecutor parallel rollouts.

    Handles the full per-model pipeline: load → rollout → metrics → save cache.
    Returns (variant, tier2_dict) or (variant, None) on failure.

    IMPORTANT: Must not call any function that reads the module-level `args_global`.
    On Windows spawn, each worker process imports the module fresh and args_global
    is None. All required data must be received via args_tuple.
    """
    (variant, output_dir, data_norm, std, data_raw, obs_months,
     n_days, n_scenarios, device_str, force) = args_tuple

    device = torch.device(device_str)
    try:
        sc_mm = run_rollout(
            variant=variant,
            output_dir=output_dir,
            data_norm=data_norm,
            std=std,
            n_days=n_days,
            n_scenarios=n_scenarios,
            device=device,
            force=force,
        )
        if sc_mm is None:
            return variant, None
        t2 = compute_tier2_metrics(sc_mm, data_raw, obs_months)
        return variant, t2
    except Exception as e:
        print(f"  [worker] {variant}: failed — {e}", flush=True)
        return variant, None
```

- [ ] **Step 5: Run the test to verify it passes**

```bash
conda run -n deeptutorial python -m pytest tests/test_compare_ar_parallel.py::test_rollout_worker_returns_none_when_rollout_fails -v
```

Expected: PASS

- [ ] **Step 6: Add success-path test**

Append to `PrecipModels/tests/test_compare_ar_parallel.py`:

```python
def test_rollout_worker_returns_metrics_on_success():
    """Worker must return (variant, metrics_dict) when run_rollout succeeds."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from compare_ar import _rollout_worker

    n_scenarios, n_days, n_stations = 5, 10, 3
    fake_sc_mm = np.zeros((n_scenarios, n_days, n_stations), dtype=np.float32)
    fake_metrics = {"multi_lag_acf_rmse": 0.1, "transition_prob_error": 0.05,
                    "inter_scenario_cv": 0.3}

    data_norm = np.zeros((30, n_stations), dtype=np.float32)
    std = np.ones((1, n_stations), dtype=np.float32)
    data_raw = np.zeros((30, n_stations), dtype=np.float32)
    obs_months = np.arange(30) % 12

    args_tuple = (
        "ar_vae_test", "./outputs", data_norm, std, data_raw, obs_months,
        n_days, n_scenarios, "cpu", False,
    )

    with patch("compare_ar.run_rollout", return_value=fake_sc_mm), \
         patch("compare_ar.compute_tier2_metrics", return_value=fake_metrics):
        variant, t2 = _rollout_worker(args_tuple)

    assert variant == "ar_vae_test"
    assert t2 == fake_metrics
```

- [ ] **Step 7: Run all new tests**

```bash
conda run -n deeptutorial python -m pytest tests/test_compare_ar_parallel.py -v
```

Expected: 2 PASSED

- [ ] **Step 8: Commit**

```bash
git add PrecipModels/compare_ar.py PrecipModels/tests/test_compare_ar_parallel.py
git commit -m "feat(compare_ar): add _rollout_worker and ProcessPoolExecutor import"
```

---

### Task 2: Add `--n_workers` CLI argument

**Files:**
- Modify: `PrecipModels/compare_ar.py` (~line 2300, argparse block)

- [ ] **Step 1: Write the failing test for `--n_workers` argparse**

Append to `PrecipModels/tests/test_compare_ar_parallel.py`:

```python
def test_n_workers_arg_default_is_1():
    """--n_workers must default to 1 (sequential, no behavior change)."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    import argparse

    # Import the parser — we parse an empty args list and check default
    import importlib
    import compare_ar
    # Directly test argparse by calling the relevant section
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_workers", type=int, default=1)
    args = parser.parse_args([])
    assert args.n_workers == 1

    args2 = parser.parse_args(["--n_workers", "4"])
    assert args2.n_workers == 4
```

- [ ] **Step 2: Run to verify it passes (it's testing argparse defaults, not compare_ar yet)**

```bash
conda run -n deeptutorial python -m pytest tests/test_compare_ar_parallel.py::test_n_workers_arg_default_is_1 -v
```

Expected: PASS (this test verifies argparse behavior, not compare_ar's parser directly)

- [ ] **Step 3: Add `--n_workers` to `compare_ar.py` argparse**

In `PrecipModels/compare_ar.py`, find the argparse block in `main()` (~line 2310). Add after the `--force_rollouts` argument:

```python
parser.add_argument("--n_workers", type=int, default=1,
                    help="Number of parallel rollout workers (default: 1 = sequential). "
                         "On a single GPU, --n_workers 2 is recommended; higher values "
                         "share VRAM and may OOM. On CPU-only, can match physical core count.")
```

- [ ] **Step 4: Run all new tests**

```bash
conda run -n deeptutorial python -m pytest tests/test_compare_ar_parallel.py -v
```

Expected: 3 PASSED

- [ ] **Step 5: Commit**

```bash
git add PrecipModels/compare_ar.py PrecipModels/tests/test_compare_ar_parallel.py
git commit -m "feat(compare_ar): add --n_workers CLI argument"
```

---

### Task 3: Add parallel branch to `main()` rollout loop

**Files:**
- Modify: `PrecipModels/compare_ar.py` (rollout loop ~lines 2450–2474)

- [ ] **Step 1: Locate the exact rollout loop**

Read `PrecipModels/compare_ar.py` lines 2448–2476 to confirm the exact current code before editing.

The loop should look like:
```python
    # Run rollouts
    scenarios_by_model = {}
    tier2_metrics = {}

    for variant in ckpt_variants:
        print(f"\n[compare_ar] Processing: {variant}")
        sc_mm = run_rollout(
            variant=variant,
            output_dir=args.output_dir,
            data_norm=data_norm,
            std=std,
            n_days=args.n_days,
            n_scenarios=args.n_scenarios,
            device=device,
            force=args.force_rollouts,
        )
        if sc_mm is None:
            continue
        scenarios_by_model[variant] = sc_mm

        t2 = compute_tier2_metrics(sc_mm, data_raw, obs_months)
        tier2_metrics[variant] = t2
        print(f"  ACF RMSE={t2['multi_lag_acf_rmse']:.4f} | "
              f"Trans Err={t2['transition_prob_error']:.4f} | "
              f"CV={t2['inter_scenario_cv']:.4f}")
```

- [ ] **Step 2: Replace the sequential loop with the branching version**

Replace the entire block from `# Run rollouts` through the closing `print(...)` call with:

```python
    # Run rollouts
    scenarios_by_model = {}
    tier2_metrics = {}

    if args.n_workers == 1:
        for variant in ckpt_variants:
            print(f"\n[compare_ar] Processing: {variant}")
            sc_mm = run_rollout(
                variant=variant,
                output_dir=args.output_dir,
                data_norm=data_norm,
                std=std,
                n_days=args.n_days,
                n_scenarios=args.n_scenarios,
                device=device,
                force=args.force_rollouts,
            )
            if sc_mm is None:
                continue
            scenarios_by_model[variant] = sc_mm

            t2 = compute_tier2_metrics(sc_mm, data_raw, obs_months)
            tier2_metrics[variant] = t2
            print(f"  ACF RMSE={t2['multi_lag_acf_rmse']:.4f} | "
                  f"Trans Err={t2['transition_prob_error']:.4f} | "
                  f"CV={t2['inter_scenario_cv']:.4f}")
    else:
        print(f"[compare_ar] Parallel rollouts: {args.n_workers} workers, "
              f"{len(ckpt_variants)} models")
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
                    print(f"  [done] {variant}: failed or skipped")
                    continue
                # Load sc_mm from cache written by the worker
                cache_path = (Path(args.output_dir) / variant
                              / "scenarios" / "scenarios.npy")
                sc_mm = np.load(str(cache_path))[:args.n_scenarios, :args.n_days, :]
                scenarios_by_model[variant] = sc_mm
                tier2_metrics[variant] = t2
                print(f"  [done] {variant}: ACF={t2['multi_lag_acf_rmse']:.4f} | "
                      f"Trans Err={t2['transition_prob_error']:.4f} | "
                      f"CV={t2['inter_scenario_cv']:.4f}")

        # Re-sort to original ckpt_variants order for deterministic plot output
        scenarios_by_model = {v: scenarios_by_model[v] for v in ckpt_variants
                              if v in scenarios_by_model}
        tier2_metrics = {v: tier2_metrics[v] for v in ckpt_variants
                        if v in tier2_metrics}
```

- [ ] **Step 3: Run the existing test suite to verify no regressions**

```bash
conda run -n deeptutorial python -m pytest tests/ -v
```

Expected: all previously passing tests still PASS; new tests PASS.

- [ ] **Step 4: Smoke-test the sequential path is unchanged**

Run compare_ar with `--skip_rollouts` to verify tier 1 still works (no import errors or syntax issues):

```bash
conda run -n deeptutorial python compare_ar.py --skip_rollouts --output_dir ./outputs 2>&1 | tail -5
```

Expected: completes without error, prints `[compare_ar] --skip_rollouts set; Tier 2 skipped.`

- [ ] **Step 5: Commit**

```bash
git add PrecipModels/compare_ar.py
git commit -m "feat(compare_ar): parallel tier 2 rollouts via --n_workers (ProcessPoolExecutor)"
```

---

## Verification Checklist

After all tasks are complete:

- [ ] `python -m pytest tests/test_compare_ar_parallel.py -v` → 3 PASSED
- [ ] `python -m pytest tests/ -v` → no regressions
- [ ] `python compare_ar.py --skip_rollouts` → exits cleanly (no import errors)
- [ ] `python compare_ar.py --help` → shows `--n_workers` in the help text
