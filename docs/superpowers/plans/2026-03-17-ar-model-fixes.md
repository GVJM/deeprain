# AR Model Fixes Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix four broken/underperforming AR model families: correct ar_flow_map's inverted sampling direction, stabilize ar_mean_flow training with correction warmup, retrain ar_real_nvp GRU variants that had stochastic training failures, and add a learned zero threshold to ar_vae to reduce dry-spell fragmentation.

**Architecture:** Pure code fixes to existing model files and training infrastructure. New model variants (`ar_flow_map_ms`, `ar_mean_flow_v2`, `ar_vae_v2`) reuse existing classes with parameter differences and register as new names. No new files are created.

**Tech Stack:** PyTorch, `torch.func.jvp`, existing `train.py` / `batch_train.py` / `compare_ar.py` infrastructure.

---

## File Map

| File | Change type | Reason |
|---|---|---|
| `PrecipModels/models/ar_flow_map.py` | Modify | Fix sampling direction; add `n_steps` param and multi-step loop |
| `PrecipModels/models/ar_mean_flow.py` | Modify | Add `effective_mf_ratio` kwarg + correction clipping |
| `PrecipModels/models/ar_vae.py` | Modify | Add `threshold` param and `occ_weight` + occurrence loss |
| `PrecipModels/models/__init__.py` | Modify | Register three new model names |
| `PrecipModels/train.py` | Modify | Add `get_mf_ratio()`, `--mf_warmup`, `--jvp_eps`, `--occ_weight`, update ARCH_DEFAULTS, `_TEMPORAL_MODELS`, `train_neural_model_temporal` signature and loop |
| `PrecipModels/MODEL_DEFAULTS.json` | Modify | Add entries for `ar_flow_map_ms`, `ar_mean_flow_v2`, `ar_vae_v2` |
| `PrecipModels/batch_train.py` | Modify | Add `mf_warmup`, `jvp_eps`, `occ_weight` to `_NS_DEFAULTS` |

All commands assume working directory `PrecipModels/` unless noted.

---

## Chunk 1: ar_flow_map — Direction Fix and Multi-Step Variant

### Task 1: Fix sampling direction in `ar_flow_map._generate_sample`

**Files:**
- Modify: `models/ar_flow_map.py`

**Background:** The OT path runs noise(t=0) → data(t=1). `_generate_sample` currently passes `z~N(0,I)` at `s=1` (the data side), which is out-of-distribution. The fix swaps to `s=0, t=1` (noise → data) and renames the variable from `z_1` to `z_0`.

- [ ] **Step 1: Write a smoke test to capture current (broken) behavior**

Create `smoke_test_flowmap.py` in `PrecipModels/`:
```python
"""Smoke test: verify ar_flow_map direction fix."""
import torch
from models.ar_flow_map import ARFlowMap

model = ARFlowMap(input_size=10, window_size=7, rnn_hidden=32, hidden_size=64, n_layers=2)
model.eval()

# Test _generate_sample: h_cond is a dummy context vector
h_cond = torch.zeros(5, 32 + model.cond_dim)  # 5 samples, rnn_hidden + cond_dim

with torch.no_grad():
    samples = model._generate_sample(h_cond, 5)

print(f"Sample shape: {samples.shape}")          # expect (5, 10)
print(f"Sample min: {samples.min():.4f}")        # expect >= 0.0 (clamped)
print(f"Sample std across batch: {samples.std(0).mean():.4f}")  # diversity check
# Before fix: model receives OOD input at s=1; diversity should be near-zero or chaotic
# After fix: samples should have reasonable std (> 0.01)
```

- [ ] **Step 2: Run the smoke test to capture baseline**
```bash
cd PrecipModels
python smoke_test_flowmap.py
```
Note the std value — this is the before-fix baseline.

- [ ] **Step 3a: Add `n_steps` to `__init__` in `models/ar_flow_map.py`**

Find the current `__init__` signature and add `n_steps=1`:
```python
def __init__(self, input_size=90, window_size=30, rnn_hidden=128,
             hidden_size=256, n_layers=4, t_embed_dim=64,
             rnn_type='gru', n_steps=1, **kwargs):
    super().__init__()
    self.n_stations  = input_size
    self.window_size = window_size
    self.rnn_type    = rnn_type
    self.n_steps     = n_steps          # ← new
    ...  # rest of __init__ unchanged
```

- [ ] **Step 3b: Fix `_generate_sample` in `models/ar_flow_map.py`**

Find the current `_generate_sample` (lines ~112–118):
```python
def _generate_sample(self, h_cond, n):
    """1-step: Phi(z_1, s=1, t=0, h) → data space."""
    device = h_cond.device
    z_1   = torch.randn(n, self.n_stations, device=device)
    s_emb = self.t_embed(torch.ones(n, device=device))
    t_emb = self.t_embed(torch.zeros(n, device=device))
    return self.flow_map(z_1, s_emb, t_emb, h_cond).clamp(min=0.0)
```

Replace with:
```python
def _generate_sample(self, h_cond, n):
    """n_steps Euler steps from noise(t=0) to data(t=1).

    Correct OT direction: z_0 ~ N(0,I) at t=0 (noise side),
    iteratively mapped to t=1 (data side) via self.flow_map.
    With n_steps=1: single-step prediction (may collapse diversity).
    With n_steps>1: multi-step ODE preserves z_0 stochasticity.
    """
    device = h_cond.device
    z = torch.randn(n, self.n_stations, device=device)
    ds = 1.0 / self.n_steps
    for i in range(self.n_steps):
        s_val = i * ds
        t_val = s_val + ds
        s_emb = self.t_embed(torch.full((n,), s_val, device=device))
        t_emb = self.t_embed(torch.full((n,), t_val, device=device))
        z = self.flow_map(z, s_emb, t_emb, h_cond)
    return z.clamp(min=0.0)
```

- [ ] **Step 3c: Wire `steps` kwarg in `sample()` and `sample_rollout()`**

The `steps` parameter already exists in the signatures (currently ignored). Wire it to override `self.n_steps`:

In `sample()`, add at the start of the method body:
```python
_n_steps_orig = self.n_steps
if steps is not None:
    self.n_steps = steps
```
And add at the end (before `return`):
```python
self.n_steps = _n_steps_orig
```

Apply the same pattern in `sample_rollout()` (same `steps` kwarg, same save/restore pattern).

- [ ] **Step 3d: Update the file-level module docstring**

Find and replace the wrong description near the top of the file. Look for the block describing the sampling convention (z_1, s=1, t=0). Replace the entire description with:
```
Sampling: z_0 ~ N(0,I) [pure noise, t=0 on OT scale].
          Euler steps from s=0 to t=1 via self.flow_map.
          n_steps=1 gives single-step prediction;
          n_steps>1 gives multi-step ODE (better diversity).
```

- [ ] **Step 4: Re-run smoke test to verify improvement**
```bash
cd PrecipModels
python smoke_test_flowmap.py
```
Expected: `Sample std across batch > 0.01` (confirms diversity is no longer collapsed).

- [ ] **Step 5: Commit**
```bash
git add models/ar_flow_map.py
git commit -m "fix(ar_flow_map): correct sampling direction s=0→t=1 and add n_steps multi-step ODE"
```

---

### Task 2: Register `ar_flow_map_ms` in infrastructure

**Files:**
- Modify: `models/__init__.py`
- Modify: `train.py` (ARCH_DEFAULTS and _TEMPORAL_MODELS)
- Modify: `MODEL_DEFAULTS.json`

- [ ] **Step 1: Add `ar_flow_map_ms` to `models/__init__.py`**

In `_MODEL_REGISTRY` dict (after the `"ar_flow_map"` entry):
```python
"ar_flow_map_ms": ARFlowMap,
```
No new import needed — `ARFlowMap` is already imported.

- [ ] **Step 2: Add `ar_flow_map_ms` to `ARCH_DEFAULTS` in `train.py`**

After the `"ar_flow_map_lstm"` entry (~line 127):
```python
"ar_flow_map_ms":  {"rnn_hidden": 128, "hidden_size": 256, "n_layers": 4,
                     "t_embed_dim": 64, "window_size": 30, "rnn_type": "gru",
                     "n_steps": 10},
```

- [ ] **Step 3: Add `ar_flow_map_ms` to `_TEMPORAL_MODELS` in `train.py`**

Add to the set (~line 135):
```python
_TEMPORAL_MODELS = {
    "ar_vae", "ar_flow_match", "ar_latent_fm",
    "ar_real_nvp", "ar_real_nvp_lstm",
    "ar_glow", "ar_glow_lstm",
    "ar_mean_flow", "ar_mean_flow_lstm",
    "ar_flow_map", "ar_flow_map_lstm",
    "ar_flow_map_ms",               # ← new
}
```

- [ ] **Step 4: Add `ar_flow_map_ms` to `MODEL_DEFAULTS.json`**

After the `"ar_flow_map_lstm"` entry, add:
```json
"ar_flow_map_ms": {
    "normalization_mode": "scale_only",
    "max_epochs": 300,
    "lr": 0.0003,
    "batch_size": 128,
    "kl_warmup": 0,
    "latent_size": 0
},
```

- [ ] **Step 5: Smoke-test that the model name resolves correctly**

`get_model` accepts `**kwargs` directly (not an `extra_kwargs` dict):
```bash
python -c "
from models import get_model
import torch
m = get_model('ar_flow_map_ms', input_size=10, n_steps=10, hidden_size=32,
              n_layers=2, rnn_hidden=16, rnn_type='gru', window_size=7, t_embed_dim=16)
print('n_steps:', m.n_steps)  # expect 10
s = m.sample(3, start_day=1)
print('sample shape:', s.shape)  # expect (3, 10)
"
```

- [ ] **Step 6: Delete smoke test file**
```bash
rm smoke_test_flowmap.py
```

- [ ] **Step 7: Commit**
```bash
git add models/__init__.py train.py MODEL_DEFAULTS.json
git commit -m "feat: register ar_flow_map_ms (10-step ODE variant, no retraining needed)"
```

---

## Chunk 2: Shared Infrastructure — New train.py / batch_train.py Parameters

### Task 3: Add `mf_warmup`, `jvp_eps`, `occ_weight` to train.py and batch_train.py

**Files:**
- Modify: `train.py`
- Modify: `batch_train.py`

This task adds three new parameters to the training infrastructure. They are needed before Tasks 4 and 5 can be tested.

- [ ] **Step 1: Add `get_mf_ratio()` helper to `train.py`**

After the `get_beta()` function (~line 150), add:
```python
def get_mf_ratio(epoch: int, mf_warmup: int, mf_ratio: float) -> float:
    """Linear ramp of mf_ratio from 0 to mf_ratio over 100 epochs after mf_warmup.

    Epochs 0 to mf_warmup-1: returns 0.0 (pure flow matching, no MeanFlow correction).
    Epochs mf_warmup to mf_warmup+99: linearly ramps from 0.0 to mf_ratio.
    Epoch mf_warmup+100 onwards: returns mf_ratio (full correction).

    If mf_warmup <= 0, returns mf_ratio immediately (no warmup).
    """
    if mf_warmup <= 0:
        return mf_ratio
    ramp = min(1.0, max(0.0, (epoch - mf_warmup) / 100.0))
    return mf_ratio * ramp
```

- [ ] **Step 2: Add `--mf_warmup`, `--jvp_eps`, `--occ_weight` argparse arguments to `train.py`**

In the argparse section (~line 1261, after `--mf_ratio`):
```python
parser.add_argument("--mf_warmup", type=int, default=None,
                    help="Epochs of pure FM before MeanFlow correction ramp (ar_mean_flow_v2)")
parser.add_argument("--jvp_eps", type=float, default=None,
                    help="Finite-difference epsilon for du/dt in MeanFlow (ar_mean_flow family)")
parser.add_argument("--occ_weight", type=float, default=None,
                    help="Weight of occurrence BCE loss (ar_vae_v2)")
```

- [ ] **Step 3: Add `mf_warmup` read in `train_model()` (parallel to `kl_warmup`)**

In `train_model()`, after the `kl_warmup` read (~line 725):
```python
mf_warmup = args.mf_warmup if args.mf_warmup is not None else defaults.get("mf_warmup", 0)
```

- [ ] **Step 4: Add `jvp_eps` and `occ_weight` to the `_arch()` enumeration and `extra_model_kwargs` loop**

In the `extra_model_kwargs` loop (~line 844), `mf_ratio` is already present. Append only the two new entries after it:
```python
("mf_ratio",       _arch("mf_ratio")),   # already exists — do not duplicate
("jvp_eps",        _arch("jvp_eps")),    # ← new
("occ_weight",     _arch("occ_weight")), # ← new
```

- [ ] **Step 5: Add `mf_warmup` to `train_neural_model_temporal()` signature**

Change the function signature at ~line 329 from:
```python
def train_neural_model_temporal(
    model: BaseModel,
    train_norm: np.ndarray,
    window_size: int,
    max_epochs: int,
    lr: float,
    batch_size: int,
    kl_warmup: int,
    device: torch.device,
    ...
```
to:
```python
def train_neural_model_temporal(
    model: BaseModel,
    train_norm: np.ndarray,
    window_size: int,
    max_epochs: int,
    lr: float,
    batch_size: int,
    kl_warmup: int,
    device: torch.device,
    mf_warmup: int = 0,             # ← new: MeanFlow correction warmup epochs
    ...
```

- [ ] **Step 6: Pass `mf_warmup` at the call site**

At the call to `train_neural_model_temporal` (~line 1075), add `mf_warmup=mf_warmup`:
```python
history, ms_per_epoch, final_opt_state, interrupted = train_neural_model_temporal(
    model=model,
    train_norm=train_norm,
    train_cond=train_cond,
    window_size=window_size or 30,
    max_epochs=max_epochs,
    lr=lr,
    batch_size=batch_size,
    kl_warmup=kl_warmup,
    mf_warmup=mf_warmup,            # ← new
    device=device,
    ...
```

- [ ] **Step 7: Add guarded `effective_mf_ratio` call in the training loop**

Inside `train_neural_model_temporal`, the `loss_dict = model.loss(...)` call is at ~line 398 **inside the `for batch in loader:` loop** (not at the epoch-level `beta` line). Replace that specific line:

```python
# BEFORE (line ~398, inside `for batch in loader:` loop):
loss_dict = model.loss((window_batch, target_batch, cond_batch), beta=beta)

# AFTER — replace with this guarded block (same indentation level):
# MeanFlow correction warmup: only applies to models with mf_ratio attribute
if hasattr(model, 'mf_ratio'):
    eff_mf = get_mf_ratio(epoch, mf_warmup, model.mf_ratio)
    loss_dict = model.loss((window_batch, target_batch, cond_batch),
                           beta=beta, effective_mf_ratio=eff_mf)
else:
    loss_dict = model.loss((window_batch, target_batch, cond_batch), beta=beta)
```

The `beta = get_beta(epoch, kl_warmup)` line at ~line 385 (epoch level, outside the batch loop) is **not modified**.

- [ ] **Step 8: Add new keys to `_NS_DEFAULTS` in `batch_train.py`**

In `_NS_DEFAULTS` at the end of the architecture section (~line 40):
```python
rnn_hidden=None, rnn_type=None, n_steps=None, mf_ratio=None,
mf_warmup=None, jvp_eps=None, occ_weight=None,   # ← new
```

- [ ] **Step 9: Verify existing AR model training loop still works**
```bash
python train.py --model ar_vae --data_path ../dados_sabesp/dayprecip.dat --max_epochs 2 --name smoke_arvae_infra
```
Expected: trains 2 epochs without error, produces `outputs/smoke_arvae_infra/`.

```bash
rm -rf outputs/smoke_arvae_infra
```

- [ ] **Step 10: Commit**
```bash
git add train.py batch_train.py
git commit -m "feat(train): add mf_warmup, jvp_eps, occ_weight params and get_mf_ratio() helper"
```

---

## Chunk 3: ar_mean_flow_v2 — Correction Warmup and Clipping

### Task 4: Update `ARMeanFlow.loss()` and register `ar_mean_flow_v2`

**Files:**
- Modify: `models/ar_mean_flow.py`
- Modify: `models/__init__.py`
- Modify: `train.py` (ARCH_DEFAULTS, _TEMPORAL_MODELS)
- Modify: `MODEL_DEFAULTS.json`

- [ ] **Step 1: Update `ARMeanFlow.loss()` signature and correction logic**

In `models/ar_mean_flow.py`, change the `loss()` method signature from:
```python
def loss(self, x, beta=1.0):
```
to:
```python
def loss(self, x, beta=1.0, effective_mf_ratio=None):
```

At the top of the method body, add:
```python
mf_r = effective_mf_ratio if effective_mf_ratio is not None else self.mf_ratio
```

Replace both occurrences of `self.mf_ratio` in the `loss()` method body (only two exist in the method — the one in `__init__` is not modified):
- `is_mf = torch.rand(B, device=device) < self.mf_ratio` → `< mf_r`
- `total = (1 - self.mf_ratio) * loss_fm + self.mf_ratio * loss_mf` → `(1 - mf_r) * loss_fm + mf_r * loss_mf`

Add correction magnitude clipping immediately after `correction = (jvp_z + du_dt).detach()`:
```python
correction = (jvp_z + du_dt).detach()
# Clip correction magnitude to prevent explosive targets early in training
correction = correction / correction.norm(dim=-1, keepdim=True).clamp(min=1.0)
```

- [ ] **Step 2: Smoke-test that the updated loss() still runs for existing ar_mean_flow**
```python
# In a terminal: python -c "..."
python -c "
import torch
from models.ar_mean_flow import ARMeanFlow
m = ARMeanFlow(input_size=10, window_size=7, rnn_hidden=32, hidden_size=64, n_layers=2)
m.train()
window = torch.zeros(4, 7, 10)
target = torch.rand(4, 10)
# Test default path (no effective_mf_ratio)
d = m.loss((window, target, None))
print('Default loss keys:', list(d.keys()))   # expect total, fm_loss, mf_loss
print('Total loss:', d['total'].item())
# Test with effective_mf_ratio=0 (warmup start)
d0 = m.loss((window, target, None), effective_mf_ratio=0.0)
print('mf_r=0 total:', d0['total'].item())    # should equal fm_loss (no MF correction)
# Test with effective_mf_ratio=0.25 (full)
d25 = m.loss((window, target, None), effective_mf_ratio=0.25)
print('mf_r=0.25 total:', d25['total'].item())
"
```

- [ ] **Step 3: Register `ar_mean_flow_v2` in `models/__init__.py`**

In `_MODEL_REGISTRY` dict (after `"ar_mean_flow"`):
```python
"ar_mean_flow_v2": ARMeanFlow,
```

- [ ] **Step 4: Add `ar_mean_flow_v2` to `ARCH_DEFAULTS` and `_TEMPORAL_MODELS` in `train.py`**

In `ARCH_DEFAULTS` (after `"ar_mean_flow_lstm"` entry, ~line 123):
```python
"ar_mean_flow_v2": {"rnn_hidden": 128, "hidden_size": 256, "n_layers": 4,
                     "t_embed_dim": 64, "window_size": 30,
                     "mf_ratio": 0.25, "rnn_type": "gru",
                     "jvp_eps": 0.05},
```

In `_TEMPORAL_MODELS`:
```python
"ar_mean_flow", "ar_mean_flow_lstm",
"ar_mean_flow_v2",                  # ← new
```

- [ ] **Step 5: Add `ar_mean_flow_v2` to `MODEL_DEFAULTS.json`**

After the `ar_mean_flow_lstm` entry:
```json
"ar_mean_flow_v2": {
    "normalization_mode": "scale_only",
    "max_epochs": 1000,
    "lr": 0.0003,
    "batch_size": 128,
    "kl_warmup": 0,
    "latent_size": 0,
    "mf_warmup": 300
},
```

- [ ] **Step 6: Verify 3-epoch training run of `ar_mean_flow_v2`**
```bash
python train.py --model ar_mean_flow_v2 --data_path ../dados_sabesp/dayprecip.dat \
    --max_epochs 3 --name smoke_amf_v2
```
Expected: runs 3 epochs, prints loss values. Epoch 1–3 should show `effective_mf_ratio=0` (all below mf_warmup=300), so only FM loss should contribute. Produces `outputs/smoke_amf_v2/`.

```bash
rm -rf outputs/smoke_amf_v2
```

- [ ] **Step 7: Commit**
```bash
git add models/ar_mean_flow.py models/__init__.py train.py MODEL_DEFAULTS.json
git commit -m "feat: add ar_mean_flow_v2 with MF correction warmup, clipping, jvp_eps=0.05"
```

---

## Chunk 4: ar_vae_v2 — Learned Zero Threshold

### Task 5: Add threshold parameter and occurrence loss to `ARVAE` and register `ar_vae_v2`

**Files:**
- Modify: `models/ar_vae.py`
- Modify: `models/__init__.py`
- Modify: `train.py` (ARCH_DEFAULTS, _TEMPORAL_MODELS)
- Modify: `MODEL_DEFAULTS.json`

- [ ] **Step 1: Add `threshold` and `occ_weight` to `ARVAE.__init__`**

In `models/ar_vae.py`, update `__init__` signature from:
```python
def __init__(
    self,
    input_size: int = 90,
    window_size: int = 30,
    gru_hidden: int = 128,
    latent_size: int = 64,
    hidden_size: int = 256,
    **kwargs,
):
```
to:
```python
def __init__(
    self,
    input_size: int = 90,
    window_size: int = 30,
    gru_hidden: int = 128,
    latent_size: int = 64,
    hidden_size: int = 256,
    occ_weight: float = 0.0,    # ← new: default 0 = disabled (backward compatible)
    **kwargs,
):
```

In the `__init__` body, after `super().__init__()`:
```python
self.occ_weight = occ_weight
# Learnable per-station zero threshold (used only when occ_weight > 0)
self.threshold = nn.Parameter(torch.zeros(input_size))
```

- [ ] **Step 2: Update `loss()` to compute occurrence loss**

In `loss()`, find the existing return block (line ~196–199). The existing KL formula is `kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())` — **preserve this formula exactly**. Only change the last three lines:

```python
# BEFORE (lines ~196–199):
mse   = F.mse_loss(x_hat, target)
kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
total = mse + beta * kl
return {"total": total, "mse": mse, "kl": kl}

# AFTER — preserve KL formula, add occ_loss:
mse   = F.mse_loss(x_hat, target)
kl    = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
occ_loss = torch.tensor(0.0, device=target.device)
if self.occ_weight > 0:
    pred_logit = x_hat - self.threshold.abs()        # (B, S)
    true_occ   = (target > 0).float()                # (B, S)
    occ_loss   = F.binary_cross_entropy_with_logits(pred_logit, true_occ)
total = mse + beta * kl + self.occ_weight * occ_loss
return {"total": total, "mse": mse, "kl": kl, "occ": occ_loss}
```

- [ ] **Step 3: Apply threshold in `sample()` and `sample_rollout()`**

`sample()` has two decode sites — **both** must be updated: the warmup loop (lines ~224–227) and the collection loop (lines ~237–240). The pattern at each site:

```python
# BEFORE (both loops):
y = self.decode(z, h, cond_emb)
window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)

# AFTER (both loops — same change):
y = self.decode(z, h, cond_emb)
if self.occ_weight > 0:
    y = y * (y > self.threshold.abs().detach())
window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)
```

In `sample_rollout()`, the same decode site at line ~286:
```python
# BEFORE:
y = self.decode(z, h, cond_emb)                         # (n_sc, n_stations)
window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)

# AFTER:
y = self.decode(z, h, cond_emb)                         # (n_sc, n_stations)
if self.occ_weight > 0:
    y = y * (y > self.threshold.abs().detach())
window = torch.cat([window[:, 1:, :], y.unsqueeze(1)], dim=1)
```

- [ ] **Step 4: Smoke-test backward compatibility (`occ_weight=0`, existing behavior)**
```bash
python -c "
import torch
from models.ar_vae import ARVAE
# Default: occ_weight=0, threshold unused
m = ARVAE(input_size=10, window_size=7, gru_hidden=32, latent_size=16, hidden_size=64)
m.train()
window = torch.zeros(4, 7, 10)
target = torch.rand(4, 10)
d = m.loss((window, target, None))
print('Keys:', list(d.keys()))        # expect total, mse, kl, occ
print('occ (disabled):', d['occ'].item())   # expect 0.0
"
```

- [ ] **Step 5: Smoke-test `ar_vae_v2` behavior (occ_weight=0.1)**
```bash
python -c "
import torch
from models.ar_vae import ARVAE
m = ARVAE(input_size=10, window_size=7, gru_hidden=32, latent_size=16, hidden_size=64, occ_weight=0.1)
m.train()
window = torch.zeros(4, 7, 10)
target = torch.rand(4, 10)
d = m.loss((window, target, None))
print('occ_loss (enabled):', d['occ'].item())   # expect > 0
print('total:', d['total'].item())
print('threshold initial:', m.threshold[:3].detach())  # expect [0, 0, 0]
# Check gradient flows to threshold
d['total'].backward()
print('threshold.grad non-zero:', m.threshold.grad is not None and m.threshold.grad.abs().sum() > 0)
"
```

- [ ] **Step 6: Register `ar_vae_v2` in `models/__init__.py`**

In `_MODEL_REGISTRY` (after `"ar_vae"`):
```python
"ar_vae_v2": ARVAE,
```

- [ ] **Step 7: Add `ar_vae_v2` to `ARCH_DEFAULTS` and `_TEMPORAL_MODELS` in `train.py`**

In `ARCH_DEFAULTS` (after `"ar_vae"` entry, ~line 105):
```python
"ar_vae_v2": {"gru_hidden": 128, "hidden_size": 256, "window_size": 30,
               "occ_weight": 0.1},
```

In `_TEMPORAL_MODELS`:
```python
"ar_vae", "ar_flow_match", "ar_latent_fm",
"ar_vae_v2",                    # ← new
```

- [ ] **Step 8: Add `ar_vae_v2` to `MODEL_DEFAULTS.json`**

After the `"ar_vae"` entry:
```json
"ar_vae_v2": {
    "normalization_mode": "scale_only",
    "max_epochs": 1000,
    "lr": 0.0003,
    "batch_size": 128,
    "kl_warmup": 100,
    "latent_size": 64
},
```

- [ ] **Step 9: Verify 3-epoch training run**
```bash
python train.py --model ar_vae_v2 --data_path ../dados_sabesp/dayprecip.dat \
    --max_epochs 3 --name smoke_vae_v2
```
Expected: trains without error, loss dict includes `occ` component, produces `outputs/smoke_vae_v2/`.

```bash
rm -rf outputs/smoke_vae_v2
```

- [ ] **Step 10: Commit**
```bash
git add models/ar_vae.py models/__init__.py train.py MODEL_DEFAULTS.json
git commit -m "feat: add ar_vae_v2 with learned per-station zero threshold (occ_weight=0.1)"
```

---

## Chunk 5: ar_real_nvp Retraining and Final Evaluation

### Task 6: Retrain failed ar_real_nvp GRU variants

**Background:** `ar_real_nvp_c8_gru` (energy score 836) and `ar_real_nvp_c4_gru` had bad random initialization runs. All config is already correct (`lr=0.0001` inherited from MODEL_DEFAULTS). Simply delete their outputs and re-run.

- [ ] **Step 1: Delete failed output directories**
```bash
cd PrecipModels
rm -rf outputs/ar_real_nvp_c8_gru outputs/ar_real_nvp_c4_gru
```

- [ ] **Step 2: Re-run via batch_train.py**
```bash
python batch_train.py --data_path ../dados_sabesp/dayprecip.dat
```
This will retrain only the two deleted variants (others are already marked `done` via `metrics.json`). Training will take approximately 2–4 hours per variant depending on GPU.

Expected: `outputs/ar_real_nvp_c8_gru/metrics.json` and `outputs/ar_real_nvp_c4_gru/metrics.json` created with `energy_score` values significantly lower than the previous 836 and 133.

---

### Task 7: Queue new model variants for training

The new variants need to be added to `TRAINING_QUEUE.json` so they can be batch-trained.

- [ ] **Step 1: Add new variants to `TRAINING_QUEUE.json`**

In `TRAINING_QUEUE.json`, add the following entries (append to the array):
```json
{"variant_name": "ar_mean_flow_v2",  "model": "ar_mean_flow_v2",  "data_path": "../dados_sabesp/dayprecip.dat"},
{"variant_name": "ar_vae_v2",        "model": "ar_vae_v2",         "data_path": "../dados_sabesp/dayprecip.dat"}
```

Note: `ar_flow_map_ms` does NOT need training — it shares the checkpoint with `ar_flow_map`. It can be evaluated directly via `compare_ar.py`.

- [ ] **Step 1b: Verify output dirs don't already exist (would cause silent skip)**
```bash
ls outputs/ar_mean_flow_v2 outputs/ar_vae_v2 2>/dev/null && echo "EXISTS — delete before running" || echo "OK — clean"
```
If they exist (from aborted prior runs), delete them: `rm -rf outputs/ar_mean_flow_v2 outputs/ar_vae_v2`

- [ ] **Step 2: Run batch training for new variants**
```bash
python batch_train.py --data_path ../dados_sabesp/dayprecip.dat
```
This trains `ar_mean_flow_v2` (~1000 epochs × ~2.3 s/epoch = ~38 min) and `ar_vae_v2` (~1000 epochs).

Note: the `--data_path` CLI flag overrides the per-entry `data_path` in the queue for all entries. Since both new entries already carry the same SABESP path, this is harmless — but do not pass a different path here.

- [ ] **Step 3: Commit TRAINING_QUEUE.json update**
```bash
git add TRAINING_QUEUE.json
git commit -m "chore: queue ar_mean_flow_v2 and ar_vae_v2 for training"
```

---

### Task 8: Final comparison run and validation

- [ ] **Step 1: Run Tier 1 comparison including all new variants**
```bash
cd PrecipModels
python compare_ar.py --skip_rollouts
```
Expected: `outputs/comparison_ar/ar_comparison_report.txt` updated with new rows for `ar_flow_map` (direction fixed), `ar_flow_map_ms`, `ar_mean_flow_v2`, `ar_vae_v2`, and retrained `ar_real_nvp_c8_gru`/`ar_real_nvp_c4_gru`.

- [ ] **Step 2: Spot-check key metrics in the report**

Verify these improvements vs. the original values:
- `ar_flow_map` composite should drop from ~0.264 toward the ar_glow range (0.04–0.12). If still poor, mean-collapse from single-step s=0→t=1 is the remaining issue — `ar_flow_map_ms` (10-step) should show better results.
- `ar_mean_flow_v2` training loss should be much lower than original ar_mean_flow (~0.9). Check `outputs/ar_mean_flow_v2/metrics.json` for `final_train_loss`.
- `ar_vae_v2` `dry_spell_length_error` should be < 3.0 (down from 4.5–8.4).
- `ar_real_nvp_c8_gru` energy score should be < 300 (down from 836).

- [ ] **Step 3: Commit final state**
```bash
git add outputs/comparison_ar/
git commit -m "chore: update comparison_ar outputs with fixed model results"
```
