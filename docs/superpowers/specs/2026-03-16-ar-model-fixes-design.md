# AR Model Fixes — Design Spec
**Date:** 2026-03-16
**Scope:** Diagnose and repair broken/underperforming AR model families in `PrecipModels/`

---

## Context

`compare_ar.py` Tier 1 results show two catastrophically broken families and two underperforming ones:

| Family | Composite (best→worst) | Status |
|---|---|---|
| ar_glow | 0.044 – 0.116 | Baseline reference |
| ar_real_nvp | 0.052 – 0.297 | Inconsistent — stochastic training failure + structural gap |
| ar_flow_match | 0.074 – 0.203 | Acceptable |
| ar_vae | 0.076 – 0.192 | High dry-spell error |
| ar_flow_map | 0.211 – 0.381 | Broken — sampling direction bug |
| ar_mean_flow | 0.319 – 0.703 | Catastrophic — correction instability |

---

## Infrastructure background

`train.py` has two configuration layers:

- **`MODEL_DEFAULTS.json`** — training hyperparameters (lr, max_epochs, kl_warmup, normalization_mode, latent_size, batch_size). Read explicitly in `train_model()` by name (e.g. `kl_warmup = args.kl_warmup if args.kl_warmup is not None else defaults["kl_warmup"]`). Every model **must** have an entry — the lookup at line 715 is a hard `MODEL_DEFAULTS[model_name]` with no fallback.
- **`ARCH_DEFAULTS`** dict in `train.py` (lines 75–128) — model architecture parameters (hidden_size, rnn_hidden, n_steps, mf_ratio, etc.). Read via `_arch(name)` helper, collected into `extra_model_kwargs` if non-None, and passed to the model constructor.

`batch_train.py` builds an `argparse.Namespace` from `_NS_DEFAULTS` (lines 26–41) mirroring all argparse keys. Any new `--flag` added to `train.py` must also appear in `_NS_DEFAULTS` in `batch_train.py`.

---

## Fix 1 — `ar_flow_map`: Sampling Direction Bug

### Diagnosis

The OT path is `z_t = (1-t)·z₀ + t·x`: noise at `t=0`, data at `t=1`. Inference must start from noise at `s=0` and map to data at `t=1`.

The current `_generate_sample` does the opposite: it samples `z~N(0,I)` (named `z_1` in code) and queries the model with `s=1` (the data endpoint) and `t=0` (the noise endpoint). During training, `s=1` always corresponded to actual precipitation data `x`, never Gaussian noise — this is fully out-of-distribution input. The file-level docstring confirms the confusion: it says `z_1 ~ N(0,I) [pure noise, t=1 on OT scale]`, which is wrong.

Additionally, single-step MSE with the correct `s=0→t=1` direction collapses diversity: since `z₀` and `x` are independent in OT coupling, `E[x|z₀] = E[x|h]` regardless of `z₀`. Multi-step ODE integration preserves `z₀` stochasticity through intermediate states.

### Changes

**In `models/ar_flow_map.py` — `_generate_sample` (no retrain):**
```python
# Start from noise at s=0, predict data at t=1
z_0   = torch.randn(n, self.n_stations, device=h_cond.device)
s_emb = self.t_embed(torch.zeros(n, device=h_cond.device))  # s=0  (noise)
t_emb = self.t_embed(torch.ones(n, device=h_cond.device))   # t=1  (data)
return self.flow_map(z_0, s_emb, t_emb, h_cond).clamp(min=0.0)
```

Update the file-level docstring: `z_1 ~ N(0,I) [pure noise, t=1 on OT scale]` → `z_0 ~ N(0,I) [pure noise, t=0 on OT scale]`.

**New model `ar_flow_map_ms` (multi-step ODE, no retrain):**

Same `ARFlowMap` class — `models/__init__.py` maps `"ar_flow_map_ms"` → `ARFlowMap`. Add `n_steps: int = 1` constructor parameter used in `_generate_sample`. For `ar_flow_map_ms`, `n_steps=10` arrives via `ARCH_DEFAULTS`. The existing base `ar_flow_map` has no `n_steps` in ARCH_DEFAULTS, so `_arch("n_steps")` returns `None`, the key is skipped in `extra_model_kwargs`, and the constructor default `n_steps=1` applies — correct.

Multi-step loop replacing the single-step call in `_generate_sample`:
```python
ds = 1.0 / self.n_steps
z = torch.randn(n, self.n_stations, device=h_cond.device)
for i in range(self.n_steps):
    s_val = i * ds
    t_val = s_val + ds
    s_emb = self.t_embed(torch.full((n,), s_val, device=h_cond.device))
    t_emb = self.t_embed(torch.full((n,), t_val, device=h_cond.device))
    z = self.flow_map(z, s_emb, t_emb, h_cond)
return z.clamp(min=0.0)
```

The `steps` kwarg in `sample()` and `sample_rollout()` (already in the signature, currently ignored) will override `self.n_steps` at call time.

**Exact `ARCH_DEFAULTS` entry for `ar_flow_map_ms`** (add to `train.py` alongside `ar_flow_map`):
```python
"ar_flow_map_ms": {"rnn_hidden": 128, "hidden_size": 256, "n_layers": 4,
                    "t_embed_dim": 64, "window_size": 30, "rnn_type": "gru",
                    "n_steps": 10},
```

**Exact `MODEL_DEFAULTS.json` entry for `ar_flow_map_ms`** (same as `ar_flow_map`):
```json
"ar_flow_map_ms": {
    "normalization_mode": "scale_only",
    "max_epochs": 300,
    "lr": 0.0003,
    "batch_size": 128,
    "kl_warmup": 0,
    "latent_size": 0
}
```

### Files changed
- `models/ar_flow_map.py` — fix `_generate_sample` direction (rename `z_1`→`z_0`, swap `s`/`t` embeddings), update file-level docstring, add `n_steps: int = 1` constructor param, implement multi-step loop, wire `steps` kwarg in `sample()`/`sample_rollout()` to override `self.n_steps`
- `models/__init__.py` — register `"ar_flow_map_ms"` → `ARFlowMap`
- `train.py` — add `"ar_flow_map_ms"` to `_TEMPORAL_MODELS`; add `"ar_flow_map_ms"` entry to `ARCH_DEFAULTS` (exact entry above)
- `MODEL_DEFAULTS.json` — add `ar_flow_map_ms` entry (exact entry above)

---

## Fix 2 — `ar_mean_flow`: Correction Instability

### Diagnosis

Three compounding problems:

1. **No correction warmup**: The MeanFlow correction fires at 25% batch ratio from epoch 1. With untrained weights, `jvp_z` and `du_dt` are essentially random noise, making `u_target = v_cond - (t-r)·[noise]` explosive. Bad weights produce worse corrections — a vicious cycle that the model never escapes. Training loss ~0.9 confirms it never converges.

2. **Unstable finite difference**: `jvp_eps=0.01` is too small for `scale_only`-normalized precipitation (values span 0–100+ mm/day), accumulating floating-point error in `(u(t+eps) - u(t)) / eps`.

3. **Inverse size-performance relationship** (h512 worst 0.703, h128 best 0.319) confirms gradient instability.

### New model: `ar_mean_flow_v2`

Same `ARMeanFlow` class, three targeted changes:

**Change 1 — MF warmup (`mf_warmup` parameter):**

Add to `train.py` alongside `get_beta()`:
```python
def get_mf_ratio(epoch: int, mf_warmup: int, mf_ratio: float) -> float:
    """Linear ramp from 0 to mf_ratio over 100 epochs after mf_warmup."""
    if mf_warmup <= 0:
        return mf_ratio
    ramp = min(1.0, max(0.0, (epoch - mf_warmup) / 100.0))
    return mf_ratio * ramp
```

`mf_warmup` threading — parallel to `kl_warmup`:
1. Add `--mf_warmup` argparse argument (default `None`)
2. In `train_model()`, read: `mf_warmup = args.mf_warmup if args.mf_warmup is not None else defaults.get("mf_warmup", 0)`
3. Add `mf_warmup: int = 0` parameter to `train_neural_model_temporal()` signature (after `kl_warmup`)
4. Pass `mf_warmup=mf_warmup` at the call site (line 1075)

In the training loop inside `train_neural_model_temporal()` (~line 385), replace the single `loss_dict = model.loss(...)` call with a guard:
```python
# Alongside existing: beta = get_beta(epoch, kl_warmup)
if hasattr(model, 'mf_ratio'):
    eff_mf = get_mf_ratio(epoch, mf_warmup, model.mf_ratio)
    loss_dict = model.loss((window_batch, target_batch, cond_batch),
                           beta=beta, effective_mf_ratio=eff_mf)
else:
    loss_dict = model.loss((window_batch, target_batch, cond_batch), beta=beta)
```

This guard ensures all other temporal models (`ar_flow_map`, `ar_glow`, `ar_vae`, etc.) are unaffected.

In `ARMeanFlow.loss()`, update the signature and use `mf_r` throughout:
```python
def loss(self, x, beta=1.0, effective_mf_ratio=None):
    ...
    mf_r = effective_mf_ratio if effective_mf_ratio is not None else self.mf_ratio
    is_mf = torch.rand(B, device=device) < mf_r
    ...
    total = (1 - mf_r) * loss_fm + mf_r * loss_mf
```

**Change 2 — Correction magnitude clipping:**

After `correction = (jvp_z + du_dt).detach()` (shape `[mf_batch, n_stations]`), add:
```python
correction = correction / correction.norm(dim=-1, keepdim=True).clamp(min=1.0)
```
Unit-norm clamp: normalizes vectors with norm > 1, leaves smaller ones unchanged.

**Change 3 — Configurable `jvp_eps` via ARCH_DEFAULTS:**

Do NOT change the `jvp_eps=0.01` default in `ARMeanFlow.__init__`. Instead:
- Add `"jvp_eps"` to the `_arch()` enumeration and `extra_model_kwargs` loop in `train.py`
- Add `--jvp_eps` argparse argument (default `None`)
- Set `"jvp_eps": 0.05` in `ARCH_DEFAULTS["ar_mean_flow_v2"]`
- Add `jvp_eps=None` to `_NS_DEFAULTS` in `batch_train.py`

Existing `ar_mean_flow` keeps `jvp_eps=0.01` via its constructor default.

**Exact `ARCH_DEFAULTS` entry for `ar_mean_flow_v2`** (add to `train.py`):
```python
"ar_mean_flow_v2": {"rnn_hidden": 128, "hidden_size": 256, "n_layers": 4,
                     "t_embed_dim": 64, "window_size": 30,
                     "mf_ratio": 0.25, "rnn_type": "gru",
                     "jvp_eps": 0.05},
```

**Exact `MODEL_DEFAULTS.json` entry for `ar_mean_flow_v2`**:
```json
"ar_mean_flow_v2": {
    "normalization_mode": "scale_only",
    "max_epochs": 1000,
    "lr": 0.0003,
    "batch_size": 128,
    "kl_warmup": 0,
    "latent_size": 0,
    "mf_warmup": 300
}
```

### Files changed
- `models/ar_mean_flow.py` — update `loss()` signature with `effective_mf_ratio=None`, replace `self.mf_ratio` with `mf_r` in batch split and loss weighting, add correction clipping
- `models/__init__.py` — register `"ar_mean_flow_v2"` → `ARMeanFlow`
- `train.py` — add `get_mf_ratio()` helper; add `--mf_warmup` and `--jvp_eps` argparse args; add `mf_warmup` read in `train_model()` parallel to `kl_warmup`; add `mf_warmup: int = 0` to `train_neural_model_temporal()` signature and its call site at line 1075; add guarded `effective_mf_ratio` call in training loop; add `"jvp_eps"` to `_arch()` enumeration and `extra_model_kwargs`; add `"ar_mean_flow_v2"` to `_TEMPORAL_MODELS`; add `"ar_mean_flow_v2"` entry to `ARCH_DEFAULTS` (exact entry above)
- `MODEL_DEFAULTS.json` — add `ar_mean_flow_v2` entry (exact entry above)
- `batch_train.py` — add `mf_warmup=None` and `jvp_eps=None` to `_NS_DEFAULTS`

---

## Fix 3 — `ar_real_nvp`: Stochastic Training Failure

### Diagnosis

All three GRU variants inherit `lr=0.0001` from `MODEL_DEFAULTS.json["ar_real_nvp"]` — no lr override in `TRAINING_QUEUE.json`. The performance gap is not a learning rate artefact:

- `ar_real_nvp_c16_gru`: composite 0.052, energy score 53
- `ar_real_nvp_c8_gru`: composite 0.168, energy score 836
- `ar_real_nvp_c4_gru`: composite 0.297, energy score 133

The baseline `ar_real_nvp` (also n_coupling=8) achieves energy score 273 — far better than `c8_gru` (836) with the same architecture. This points to a bad random initialization or training run for `c8_gru`, not a systematic structural gap. The c4→c8→c16 trend in composite score is genuine and structural: the 90-station joint distribution requires many coupling layers for adequate coverage.

### Fix

Retrain `ar_real_nvp_c8_gru` and `ar_real_nvp_c4_gru` by deleting their output directories and re-running via `batch_train.py`. No source code or config changes needed.

```bash
cd PrecipModels/
rm -rf outputs/ar_real_nvp_c8_gru outputs/ar_real_nvp_c4_gru
python batch_train.py
```

If after retraining `c8_gru`'s energy score remains >> c16_gru, the gap is structural and 16 coupling layers should be the minimum recommended count.

### Files changed
- None

---

## Fix 4 — `ar_vae`: Learned Zero Threshold

### Diagnosis

The VAE decoder ends with `nn.ReLU()` — all outputs are `x_hat ≥ 0`. Near-zero values (0.001–0.5 mm) count as "wet days", producing spurious drizzle that fragments dry spells: `dry_spell_length_error` 4.5–8.4 days vs ~1.0–1.5 for flow models.

### New model: `ar_vae_v2`

Adds a learnable per-station threshold `self.threshold` (`nn.Parameter`, shape `[n_stations]`, initialized to zeros) and `occ_weight: float` constructor parameter.

**In `loss()` — occurrence loss term (uses existing `x_hat` variable):**
```python
# x_hat: (B, S) — from self.decode(z, h, cond_emb), already in loss()
pred_logit = x_hat - self.threshold.abs()          # (B, S)
true_occ   = (target > 0).float()                  # (B, S)
occ_loss   = F.binary_cross_entropy_with_logits(pred_logit, true_occ)
total      = mse + beta * kl + self.occ_weight * occ_loss
```

Return dict: `{'total': total, 'mse': mse, 'kl': kl, 'occ': occ_loss}`.

**In `sample()` and `sample_rollout()` — apply threshold after each `decode()` call:**
```python
x_hat = self.decode(z, h, cond_emb)
y = x_hat * (x_hat > self.threshold.abs().detach())
```

`decode()` is unchanged.

**`occ_weight` threading** (same `_arch()` / `extra_model_kwargs` pattern as `mf_ratio`):
- Add `"occ_weight"` to `_arch()` enumeration and `extra_model_kwargs` loop in `train.py`
- Add `--occ_weight` argparse argument (default `None`)
- Set `"occ_weight": 0.1` in `ARCH_DEFAULTS["ar_vae_v2"]`
- Add `occ_weight=None` to `_NS_DEFAULTS` in `batch_train.py`

**Exact `ARCH_DEFAULTS` entry for `ar_vae_v2`** (add to `train.py`):
```python
"ar_vae_v2": {"gru_hidden": 128, "hidden_size": 256, "window_size": 30,
               "occ_weight": 0.1},
```

**Exact `MODEL_DEFAULTS.json` entry for `ar_vae_v2`**:
```json
"ar_vae_v2": {
    "normalization_mode": "scale_only",
    "max_epochs": 1000,
    "lr": 0.0003,
    "batch_size": 128,
    "kl_warmup": 100,
    "latent_size": 64
}
```

### Files changed
- `models/ar_vae.py` — add `threshold: nn.Parameter` and `occ_weight: float = 0.1` to `__init__`; update `loss()` to compute `occ_loss` and return it; update `sample()` and `sample_rollout()` to apply threshold mask after `decode()`
- `models/__init__.py` — register `"ar_vae_v2"` → `ARVAE`
- `train.py` — add `--occ_weight` argparse arg; add `"occ_weight"` to `_arch()` enumeration and `extra_model_kwargs`; add `"ar_vae_v2"` to `_TEMPORAL_MODELS`; add `"ar_vae_v2"` entry to `ARCH_DEFAULTS` (exact entry above)
- `MODEL_DEFAULTS.json` — add `ar_vae_v2` entry (exact entry above)
- `batch_train.py` — add `occ_weight=None` to `_NS_DEFAULTS`

---

## Summary of Deliverables

| Item | Type | Retrain? |
|---|---|---|
| Fix `ar_flow_map._generate_sample` direction + docstring | Code fix | No |
| New `ar_flow_map_ms` (multi-step ODE, 10 steps) | New model variant | No |
| New `ar_mean_flow_v2` (MF warmup + clipping + jvp_eps=0.05) | New model variant | Yes |
| Retrain `ar_real_nvp_c8_gru`, `ar_real_nvp_c4_gru` | Training re-run | Yes |
| New `ar_vae_v2` (learned per-station threshold) | New model variant | Yes |

All existing checkpoints and metrics remain untouched. New/retrained variants are discovered automatically by `compare_ar.py` via `outputs/` scan.

---

## Evaluation

After all changes, regenerate Tier 1 comparison:
```bash
cd PrecipModels/
python compare_ar.py --skip_rollouts
```

To validate the `ar_flow_map` direction fix in isolation (no GPU training):
```bash
python compare_ar.py --skip_rollouts --models ar_flow_map ar_flow_map_ms
```
