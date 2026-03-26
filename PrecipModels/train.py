"""
train.py — Script de treino unificado para todos os modelos.

Uso:
    python train.py --model vae
    python train.py --model hurdle_simple
    python train.py --model hurdle_vae
    python train.py --model real_nvp
    python train.py --model flow_match
    python train.py --model copula          # apenas ajuste estatístico, sem gradiente

    python train.py --model vae --max_epochs 500 --latent_size 64
    python train.py --model vae --device cuda --batch_size 256

    # Variantes nomeadas (--name define o diretório de saída)
    python train.py --model hurdle_simple --lr 0.0001 --name hs_lr0001
    python train.py --model flow_match --hidden_size 512 --n_layers 6 --name fm_large

Saídas em outputs/<name>/: model.pt (ou copula.pkl), config.json, metrics.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset

from datasets import TemporalDataset, TemporalCondDataset
from data_utils import load_data, load_data_with_cond, denormalize
from base_model import BaseModel
from models import get_model, MODEL_NAMES
from metrics import evaluate_model
from models.conditioning import DEFAULT_CATEGORICALS
import json

def _setup_intel_opts(model, args):
    """Apply Intel CPU optimizations. Returns (model, use_amp, amp_dtype)."""
    n_threads = args.num_threads if args.num_threads is not None else 4
    torch.set_num_threads(n_threads)
    torch.set_num_interop_threads(1)
    print(f"[optimize] Threads: intra-op={n_threads}, inter-op=1")
    print("[optimize] BF16 autocast active — watch for NLL instability with log-likelihood losses")

    # Try IPEX (model-only optimize, inplace, no optimizer needed)
    try:
        import intel_extension_for_pytorch as ipex
        model = ipex.optimize(model, dtype=torch.bfloat16, inplace=True)
        print("[optimize] IPEX applied (model-only, BF16)")
    except ImportError:
        # Fall back to torch.compile
        try:
            model = torch.compile(model, backend="inductor", fullgraph=False)
            print("[optimize] torch.compile applied (inductor, fullgraph=False)")
        except Exception as e:
            print(f"[optimize] torch.compile failed ({e}) — thread tuning only")

    return model, True, torch.bfloat16


# ──────────────────────────────────────────────────────────
# DEFAULTS POR MODELO
# ──────────────────────────────────────────────────────────
with open("MODEL_DEFAULTS.json", "r+") as f:
    MODEL_DEFAULTS = json.load(f)

# Defaults de arquitetura por modelo (usados quando o CLI não sobrescreve)
ARCH_DEFAULTS = {
    "copula":             {},
    "vae":                {},
    "hurdle_simple":      {"hidden_occ": 32, "hidden_amt": 64},
    "hurdle_vae":         {},
    "hurdle_vae_cond":    {},
    "hurdle_vae_cond_mc":    {},
    "hurdle_vae_cond_nll":{},
    "real_nvp":           {"hidden_size": 256, "n_coupling": 12},
    "hurdle_flow":        {},
    "flow_match":         {"hidden_size": 256, "n_layers": 4, "t_embed_dim": 64, "n_sample_steps": 50},
    "flow_match_film":    {},
    "ldm":                {},
    "hurdle_temporal":    {"hidden_occ": 64, "hidden_amt": 128, "gru_hidden": 64,
                           "context_dim": 32, "window_size": 30},
    "latent_flow":        {"n_layers": 4, "window_size": 30, "hidden_dim": 128},
    "hurdle_simple_mc":   {"hidden_occ": 32, "hidden_amt": 64},
    "vae_mc":             {},
    "glow":               {"hidden_size": 128, "n_layers": 8},
    "real_nvp_mc":        {"hidden_size": 256, "n_coupling": 12},
    "glow_mc":            {"hidden_size": 128, "n_layers": 8},
    "flow_match_mc":      {"hidden_size": 256, "n_layers": 4, "t_embed_dim": 64, "n_sample_steps": 50},
    "flow_match_film_mc":      {"hidden_size": 256, "n_layers": 4, "t_embed_dim": 64, "n_sample_steps": 50},
    "latent_fm_mc":         {"hidden_size": 256, "n_layers": 6, "t_embed_dim": 64, "n_sample_steps": 100},
    "hurdle_latent_fm_mc":  {"hidden_size": 256, "n_layers": 6, "t_embed_dim": 64, "n_sample_steps": 100},
    "thresholded_latent_fm_mc": { "hidden_size": 384, "n_layers": 6, "t_embed_dim": 64, "n_sample_steps": 100, "vae_layers": 3},
    "thresholded_vae_mc":       {},
    "thresholded_real_nvp_mc":  {"hidden_size": 256, "n_coupling": 12},
    "thresholded_glow_mc":      {"hidden_size": 128, "n_layers": 8},
    # ── AR Models ─────────────────────────────────────────────────────────────
    "ar_vae":            {"gru_hidden": 128, "hidden_size": 256, "window_size": 30},
    "ar_vae_v2":         {"gru_hidden": 128, "hidden_size": 256, "window_size": 30,
                          "occ_weight": 0.1},
    "ar_flow_match":     {"gru_hidden": 128, "hidden_size": 256, "n_layers": 4,
                          "t_embed_dim": 64, "n_sample_steps": 50, "window_size": 30},
    "ar_latent_fm":      {"gru_hidden": 128, "hidden_size": 256, "n_layers": 4,
                          "t_embed_dim": 64, "n_sample_steps": 50, "window_size": 30},
    "ar_real_nvp":       {"rnn_hidden": 128, "n_coupling": 8, "hidden_size": 256,
                          "window_size": 30, "rnn_type": "gru"},
    "ar_real_nvp_lstm":  {"rnn_hidden": 128, "n_coupling": 8, "hidden_size": 256,
                          "window_size": 30, "rnn_type": "lstm"},
    "ar_glow":           {"rnn_hidden": 128, "n_steps": 8, "hidden_size": 128,
                          "window_size": 30, "rnn_type": "gru"},
    "ar_glow_lstm":      {"rnn_hidden": 128, "n_steps": 8, "hidden_size": 128,
                          "window_size": 30, "rnn_type": "lstm"},
    "ar_mean_flow":      {"rnn_hidden": 128, "hidden_size": 256, "n_layers": 4,
                          "t_embed_dim": 64, "window_size": 30,
                          "mf_ratio": 0.25, "rnn_type": "gru"},
    "ar_mean_flow_lstm": {"rnn_hidden": 128, "hidden_size": 256, "n_layers": 4,
                          "t_embed_dim": 64, "window_size": 30,
                          "mf_ratio": 0.25, "rnn_type": "lstm"},
    "ar_mean_flow_v2":   {"rnn_hidden": 128, "hidden_size": 256, "n_layers": 4,
                          "t_embed_dim": 64, "window_size": 30,
                          "mf_ratio": 0.25, "rnn_type": "gru",
                          "jvp_eps": 0.05},
    "ar_flow_map":       {"rnn_hidden": 128, "hidden_size": 256, "n_layers": 4,
                          "t_embed_dim": 64, "window_size": 30, "rnn_type": "gru"},
    "ar_flow_map_lstm":  {"rnn_hidden": 128, "hidden_size": 256, "n_layers": 4,
                          "t_embed_dim": 64, "window_size": 30, "rnn_type": "lstm"},
    "ar_flow_map_ms":    {"rnn_hidden": 128, "hidden_size": 256, "n_layers": 4,
                          "t_embed_dim": 64, "window_size": 30, "rnn_type": "gru",
                          "n_steps": 10},
    "ar_flow_map_sd":    {"rnn_hidden": 128, "hidden_size": 256, "n_layers": 4,
                          "t_embed_dim": 64, "window_size": 30, "rnn_type": "gru",
                          "n_steps": 1, "lsd_weight": 0.1},
    "ar_mean_flow_ayfm": {"rnn_hidden": 128, "hidden_size": 256, "n_layers": 4,
                          "t_embed_dim": 64, "window_size": 30, "rnn_type": "gru",
                          "mf_ratio": 0.25, "jvp_eps": 0.01,
                          "tangent_warmup_steps": 5000,
                          "improved_interval_sampling": True,
                          "mu_sad": 0.0, "sigma_sad": 1.0},
}


# Modelos que usam condicionamento (_mc suffix)
_MC_MODELS = {"hurdle_simple_mc", "vae_mc", "real_nvp_mc", "glow_mc", "flow_match_mc", "flow_match_film_mc", "hurdle_vae_cond_mc", "latent_fm_mc", "hurdle_latent_fm_mc", "thresholded_latent_fm_mc", "thresholded_vae_mc", "thresholded_real_nvp_mc", "thresholded_glow_mc"}

_TEMPORAL_MODELS = {
    "ar_vae", "ar_vae_v2", "ar_flow_match", "ar_latent_fm",
    "ar_real_nvp", "ar_real_nvp_lstm",
    "ar_glow", "ar_glow_lstm",
    "ar_mean_flow", "ar_mean_flow_lstm", "ar_mean_flow_v2",
    "ar_flow_map", "ar_flow_map_lstm", "ar_flow_map_ms",
    "ar_flow_map_sd",
    "ar_mean_flow_ayfm",
}


def get_beta(epoch: int, kl_warmup: int) -> float:
    """
    KL annealing: rampa linear 0 → 1 ao longo de kl_warmup épocas.
    Retorna 1.0 se kl_warmup=0 (sem annealing).
    """
    if kl_warmup <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / kl_warmup)


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


from data_utils import (
    temporal_holdout_split,
    temporal_holdout_split_with_cond,
    temporal_train_val_test_split,
    temporal_train_val_test_split_with_cond,
    compute_norm_params,
    normalize_with_params,
)


class EarlyStopper:
    """Monitor val loss and signal when training should stop.

    patience=0 disables early stopping (never stops).
    Patience counts validation *checks*, not epochs — use val_freq to control
    how often checks occur.
    """
    def __init__(self, patience: int):
        self.patience = patience
        self.best_loss = float('inf')
        self._strikes = 0

    def update(self, val_loss: float) -> bool:
        """Returns True when training should stop."""
        if self.patience <= 0:
            return False
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self._strikes = 0
        else:
            self._strikes += 1
        return self._strikes >= self.patience


def instantiate_model(
    model_name: str,
    input_size: int,
    latent_size: int,
    device: torch.device,
    extra_kwargs: dict = None,
) -> BaseModel:
    model_kwargs = dict(input_size=input_size)
    if latent_size > 0:
        model_kwargs["latent_size"] = latent_size
    if extra_kwargs:
        model_kwargs.update(extra_kwargs)
    model = get_model(model_name, **model_kwargs)
    model = model.to(device)
    return model


def train_neural_model(
    model: BaseModel,
    train_norm: np.ndarray,
    max_epochs: int,
    lr: float,
    batch_size: int,
    kl_warmup: int,
    device: torch.device,
    model_name: str,
    print_every: int = 50,
    optimizer_state: dict = None,
    eval_norm: np.ndarray = None,
    out_dir: str = None,
    opt_config: tuple = None,   # (use_amp, amp_dtype) or None
    early_stop_patience: int = 0,
    val_freq: int = 1,
) -> Tuple[List[dict], float, dict]:
    _use_amp, _amp_dtype = opt_config if opt_config is not None else (False, torch.float32)
    t_data = torch.FloatTensor(train_norm).to(device)
    dataset = TensorDataset(t_data, t_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    eval_tensor = torch.FloatTensor(eval_norm).to(device) if eval_norm is not None else None

    print(f"\n[{model_name}] Iniciando treino: {max_epochs} épocas, lr={lr}, batch={batch_size}")
    print("-" * 60)
    print(f"{'Epoch':>7}  {'Loss Total':>12}  {'Sub-losses':>30}  {'Beta':>6}")
    print("-" * 60)

    train_start = time.perf_counter()
    history = []
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    stopper = EarlyStopper(early_stop_patience)

    interrupted = False
    try:
        for epoch in range(max_epochs):
            model.train()
            running = {}
            n_samples = 0

            beta = get_beta(epoch, kl_warmup)

            for x_batch, _ in loader:
                optimizer.zero_grad()
                with torch.autocast(device_type='cpu', dtype=_amp_dtype, enabled=_use_amp):
                    loss_dict = model.loss(x_batch, beta=beta)
                loss_dict['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                bsz = x_batch.shape[0]
                for k, v in loss_dict.items():
                    running[k] = running.get(k, 0.0) + v.item() * bsz
                n_samples += bsz

            avg = {k: v / max(n_samples, 1) for k, v in running.items()}
            avg['epoch'] = epoch + 1
            avg['beta'] = round(beta, 4)

            if eval_tensor is not None and (epoch + 1) % val_freq == 0:
                model.eval()
                with torch.no_grad():
                    with torch.autocast(device_type='cpu', dtype=_amp_dtype, enabled=_use_amp):
                        vd = model.loss(eval_tensor, beta=beta)
                for k, v in vd.items():
                    avg[f'val_{k}'] = v.item()
                model.train()
                if out_dir and avg['val_total'] < best_val_loss:
                    best_val_loss = avg['val_total']
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               os.path.join(out_dir, 'model_best_val.pt'))
                if stopper.update(avg['val_total']):
                    print(f"\n[{model_name}] Early stopping at epoch {epoch+1} "
                          f"(patience={early_stop_patience}, no improvement for {stopper._strikes} checks)")
                    interrupted = True
                    break

            if out_dir and avg['total'] < best_train_loss:
                best_train_loss = avg['total']
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(out_dir, 'model_best_train.pt'))

            history.append(avg)

            if (epoch + 1) % print_every == 0 or epoch == max_epochs - 1:
                sub_str = "  ".join(
                    f"{k}={v:.4f}" for k, v in avg.items()
                    if k not in ('total', 'epoch', 'beta') and not k.startswith('val_')
                )
                val_str = f"  val={avg['val_total']:.4f}" if 'val_total' in avg else ""
                print(f"{epoch+1:7d}  {avg['total']:12.4f}  {sub_str:<30}  {beta:6.4f}{val_str}")
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[{model_name}] Training interrupted at epoch {epoch+1}/{max_epochs}. Saving best checkpoint...")

    train_elapsed = time.perf_counter() - train_start
    ms_per_epoch = train_elapsed * 1000 / max(len(history), 1)
    print(f"\n[{model_name}] Treino concluído: {train_elapsed:.1f}s ({ms_per_epoch:.1f} ms/época)")
    return history, ms_per_epoch, optimizer.state_dict(), interrupted


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
    model_name: str = "",
    train_cond: dict | None = None,
    print_every: int = 50,
    optimizer_state: dict = None,
    eval_norm: np.ndarray = None,
    eval_cond: dict | None = None,
    out_dir: str = None,
    opt_config: tuple = None,
    early_stop_patience: int = 0,
    val_freq: int = 1,
) -> Tuple[List[dict], float, dict]:
    """Training loop for autoregressive temporal models.
    Uses TemporalDataset or TemporalCondDataset; passes tuple to model.loss()."""
    _use_amp, _amp_dtype = opt_config if opt_config is not None else (False, torch.float32)

    if train_cond is not None:
        dataset = TemporalCondDataset(train_norm, train_cond, window_size)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    else:
        dataset = TemporalDataset(train_norm, window_size)
        loader  = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)
    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    if eval_norm is not None and eval_cond is not None:
        eval_dataset = TemporalCondDataset(eval_norm, eval_cond, window_size)
    elif eval_norm is not None:
        eval_dataset = TemporalDataset(eval_norm, window_size)
    else:
        eval_dataset = None

    print(f"\n[{model_name}] Temporal training: {max_epochs} epochs, lr={lr}, batch={batch_size}, window={window_size}")
    print("-" * 60)
    print(f"{'Epoch':>7}  {'Loss Total':>12}  {'Sub-losses':>30}  {'Beta':>6}")
    print("-" * 60)

    train_start = time.perf_counter()
    history = []
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    stopper = EarlyStopper(early_stop_patience)

    interrupted = False
    global_step = 0
    try:
        for epoch in range(max_epochs):
            model.train()
            running = {}
            n_samples = 0
            beta = get_beta(epoch, kl_warmup)
            eff_mf = get_mf_ratio(epoch, mf_warmup, model.mf_ratio) if hasattr(model, 'mf_ratio') else None

            for batch in loader:
                if train_cond is not None:
                    window_batch, target_batch, cond_batch = batch
                    cond_batch = {k: v.to(device) for k, v in cond_batch.items()}
                else:
                    window_batch, target_batch = batch
                    cond_batch = None
                window_batch = window_batch.to(device)
                target_batch = target_batch.to(device)
                optimizer.zero_grad()
                with torch.autocast(device_type='cpu', dtype=_amp_dtype, enabled=_use_amp):
                    # MeanFlow correction warmup: only applies to models with mf_ratio attribute
                    if eff_mf is not None:
                        loss_dict = model.loss((window_batch, target_batch, cond_batch),
                                               beta=beta, effective_mf_ratio=eff_mf, training_step=global_step)
                    else:
                        loss_dict = model.loss((window_batch, target_batch, cond_batch), beta=beta, training_step=global_step)
                loss_dict['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                global_step += 1

                bsz = target_batch.shape[0]
                for k, v in loss_dict.items():
                    running[k] = running.get(k, 0.0) + v.item() * bsz
                n_samples += bsz

            avg = {k: v / max(n_samples, 1) for k, v in running.items()}
            avg['epoch'] = epoch + 1
            avg['beta']  = round(beta, 4)

            if eval_dataset is not None and (epoch + 1) % val_freq == 0:
                model.eval()
                eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)
                val_running = {}
                val_n = 0
                with torch.no_grad():
                    for batch in eval_loader:
                        if eval_cond is not None:
                            w_val, t_val, c_val = batch
                            c_val = {k: v.to(device) for k, v in c_val.items()}
                        else:
                            w_val, t_val = batch
                            c_val = None
                        w_val, t_val = w_val.to(device), t_val.to(device)
                        with torch.autocast(device_type='cpu', dtype=_amp_dtype, enabled=_use_amp):
                            vd = model.loss((w_val, t_val, c_val), beta=beta)
                        bsz = t_val.shape[0]
                        for k, v in vd.items():
                            val_running[k] = val_running.get(k, 0.0) + v.item() * bsz
                        val_n += bsz
                for k, v in val_running.items():
                    avg[f'val_{k}'] = v / max(val_n, 1)
                model.train()
                if out_dir and avg['val_total'] < best_val_loss:
                    best_val_loss = avg['val_total']
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               os.path.join(out_dir, 'model_best_val.pt'))
                if stopper.update(avg['val_total']):
                    print(f"\n[{model_name}] Early stopping at epoch {epoch+1} "
                          f"(patience={early_stop_patience}, no improvement for {stopper._strikes} checks)")
                    interrupted = True
                    break

            if out_dir and avg['total'] < best_train_loss:
                best_train_loss = avg['total']
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(out_dir, 'model_best_train.pt'))

            history.append(avg)

            if (epoch + 1) % print_every == 0 or epoch == max_epochs - 1:
                sub_str = "  ".join(
                    f"{k}={v:.4f}" for k, v in avg.items()
                    if k not in ('total', 'epoch', 'beta') and not k.startswith('val_')
                )
                val_str = f"  val={avg['val_total']:.4f}" if 'val_total' in avg else ""
                print(f"{epoch+1:7d}  {avg['total']:12.4f}  {sub_str:<30}  {beta:6.4f}{val_str}")
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[{model_name}] Training interrupted at epoch {epoch+1}/{max_epochs}. Saving best checkpoint...")

    train_elapsed = time.perf_counter() - train_start
    ms_per_epoch = train_elapsed * 1000 / max(len(history), 1)
    print(f"\n[{model_name}] Training complete: {train_elapsed:.1f}s ({ms_per_epoch:.1f} ms/epoch)")
    return history, ms_per_epoch, optimizer.state_dict(), interrupted


class _CondDataset(Dataset):
    """Dataset que retorna (x_batch, cond_batch_dict) para modelos condicionados."""

    def __init__(self, data: torch.Tensor, cond_tensors: dict):
        self.data = data
        self.cond_tensors = cond_tensors

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        return self.data[idx], {k: v[idx] for k, v in self.cond_tensors.items()}


def _cond_collate(batch):
    """Collate function para _CondDataset."""
    xs = torch.stack([item[0] for item in batch])
    cond = {k: torch.stack([item[1][k] for item in batch]) for k in batch[0][1]}
    return xs, cond


def train_neural_model_mc(
    model: BaseModel,
    train_norm: np.ndarray,
    train_cond: dict,
    max_epochs: int,
    lr: float,
    batch_size: int,
    kl_warmup: int,
    device: torch.device,
    model_name: str,
    print_every: int = 50,
    optimizer_state: dict = None,
    eval_norm: np.ndarray = None,
    eval_cond: dict = None,
    out_dir: str = None,
    opt_config: tuple = None,   # (use_amp, amp_dtype) or None
    early_stop_patience: int = 0,
    val_freq: int = 1,
) -> Tuple[List[dict], float, dict]:
    """
    Loop de treino para modelos condicionados (_mc).
    Cada batch contém (x_batch, cond_batch_dict) onde cond é um dict de LongTensors.
    """
    _use_amp, _amp_dtype = opt_config if opt_config is not None else (False, torch.float32)
    t_data = torch.FloatTensor(train_norm).to(device)
    _cat_names = {name for name, _, _ in DEFAULT_CATEGORICALS}
    cond_tensors = {
        k: (torch.LongTensor(v) if k in _cat_names else torch.FloatTensor(v)).to(device)
        for k, v in train_cond.items()
    }

    dataset = _CondDataset(t_data, cond_tensors)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=_cond_collate)

    print(f'Model Name: {model_name}')
    if "latent_fm_mc/vae" == model_name:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    elif "latent_fm_mc/flow" == model_name:
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    else: 
        optimizer = optim.Adam(model.parameters(), lr=lr)

    if optimizer_state is not None:
        optimizer.load_state_dict(optimizer_state)

    eval_tensor = torch.FloatTensor(eval_norm).to(device) if eval_norm is not None else None
    eval_cond_tensors = None
    if eval_cond is not None and eval_tensor is not None:
        eval_cond_tensors = {
            k: (torch.LongTensor(v) if k in _cat_names else torch.FloatTensor(v)).to(device)
            for k, v in eval_cond.items()
        }

    print(f"\n[{model_name}] Iniciando treino (condicionado): {max_epochs} épocas, lr={lr}, batch={batch_size}")
    print("-" * 60)
    print(f"{'Epoch':>7}  {'Loss Total':>12}  {'Sub-losses':>30}  {'Beta':>6}")
    print("-" * 60)

    train_start = time.perf_counter()
    history = []
    best_val_loss = float('inf')
    best_train_loss = float('inf')
    stopper = EarlyStopper(early_stop_patience)

    interrupted = False
    try:
        for epoch in range(max_epochs):
            model.train()
            running = {}
            n_samples = 0

            beta = get_beta(epoch, kl_warmup)

            for x_batch, cond_batch in loader:
                optimizer.zero_grad()
                with torch.autocast(device_type='cpu', dtype=_amp_dtype, enabled=_use_amp):
                    loss_dict = model.loss(x_batch, beta=beta, cond=cond_batch)
                loss_dict['total'].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

                bsz = x_batch.shape[0]
                for k, v in loss_dict.items():
                    running[k] = running.get(k, 0.0) + v.item() * bsz
                n_samples += bsz

            avg = {k: v / max(n_samples, 1) for k, v in running.items()}
            avg['epoch'] = epoch + 1
            avg['beta'] = round(beta, 4)

            if eval_tensor is not None and (epoch + 1) % val_freq == 0:
                model.eval()
                val_running = {}
                val_n_samples = 0
                with torch.no_grad():
                    for start in range(0, eval_tensor.shape[0], batch_size):
                        end = min(start + batch_size, eval_tensor.shape[0])
                        x_val = eval_tensor[start:end]
                        cond_val = {k: v[start:end] for k, v in eval_cond_tensors.items()} if eval_cond_tensors is not None else None
                        with torch.autocast(device_type='cpu', dtype=_amp_dtype, enabled=_use_amp):
                            vd = model.loss(x_val, beta=beta, cond=cond_val)
                        bsz = x_val.shape[0]
                        for k, v in vd.items():
                            val_running[k] = val_running.get(k, 0.0) + v.item() * bsz
                        val_n_samples += bsz
                for k, v in val_running.items():
                    avg[f'val_{k}'] = v / max(val_n_samples, 1)
                model.train()
                if out_dir and avg['val_total'] < best_val_loss:
                    best_val_loss = avg['val_total']
                    torch.save({'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict()},
                               os.path.join(out_dir, 'model_best_val.pt'))
                if stopper.update(avg['val_total']):
                    print(f"\n[{model_name}] Early stopping at epoch {epoch+1} "
                          f"(patience={early_stop_patience}, no improvement for {stopper._strikes} checks)")
                    interrupted = True
                    break

            if out_dir and avg['total'] < best_train_loss:
                best_train_loss = avg['total']
                torch.save({'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(out_dir, 'model_best_train.pt'))

            history.append(avg)

            if (epoch + 1) % print_every == 0 or epoch == max_epochs - 1:
                sub_str = "  ".join(
                    f"{k}={v:.4f}" for k, v in avg.items()
                    if k not in ('total', 'epoch', 'beta') and not k.startswith('val_')
                )
                val_str = f"  val={avg['val_total']:.4f}" if 'val_total' in avg else ""
                print(f"{epoch+1:7d}  {avg['total']:12.4f}  {sub_str:<30}  {beta:6.4f}{val_str}")
    except KeyboardInterrupt:
        interrupted = True
        print(f"\n[{model_name}] Training interrupted at epoch {epoch+1}/{max_epochs}. Saving best checkpoint...")

    train_elapsed = time.perf_counter() - train_start
    ms_per_epoch = train_elapsed * 1000 / max(len(history), 1)
    print(f"\n[{model_name}] Treino concluído: {train_elapsed:.1f}s ({ms_per_epoch:.1f} ms/época)")
    return history, ms_per_epoch, optimizer.state_dict(), interrupted


def _plot_training_history(history: list, out_dir: str, model_name: str, is_log_axis=False):
    """Plota as curvas de perda por época e salva em training_loss.png."""
    import matplotlib.pyplot as plt

    epochs = [h['epoch'] for h in history]
    total = [h['total'] for h in history]

    # Detecta sub-losses (exclui chaves de metadados e val_*)
    meta_keys = {'epoch', 'beta', 'total', 'stage'}
    sub_keys = sorted(set(k for h in history for k in h if k not in meta_keys and not k.startswith('val_')))

    n_plots = 1 + len(sub_keys) if sub_keys else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

    # Para LDM: sombreia os dois estágios em todos os subplots
    has_stages = 'stage' in history[0]
    stage_switch = None
    if has_stages:
        stage_switch = next(
            (h['epoch'] for h in history if h.get('stage') == 'ldm'), None
        )

    def shade_stages(ax):
        if has_stages and stage_switch:
            ax.axvspan(epochs[0], stage_switch - 1, alpha=0.07, color='steelblue', label='VAE stage')
            ax.axvspan(stage_switch, epochs[-1], alpha=0.07, color='darkorange', label='LDM stage')

    has_val = any('val_total' in h for h in history)

    # Loss total
    ax = axes[0]
    ax.plot(epochs, total, color='steelblue', linewidth=1.2, label='Treino')
    if has_val:
        val_pairs = [(h['epoch'], h['val_total']) for h in history if 'val_total' in h]
        if val_pairs:
            ep_v, vals_v = zip(*val_pairs)
            ax.plot(ep_v, vals_v, color='darkorange', linewidth=1.2, linestyle='--', label='Validação')
    shade_stages(ax)
    ax.set_title('Total Loss', fontsize=10)
    ax.set_xlabel('Época', fontsize=9)
    ax.set_ylabel('Loss', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=8)
    if has_stages or has_val:
        ax.legend(fontsize=8)
    if is_log_axis:
        ax.set_yscale('log')

    # Sub-losses
    colors = ['#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    for i, key in enumerate(sub_keys):
        ax = axes[i + 1]
        pairs = [(h['epoch'], h[key]) for h in history if key in h]
        if not pairs:
            continue
        ep_filt, vals = zip(*pairs)
        ax.plot(ep_filt, vals, color=colors[i % len(colors)], linewidth=1.2, label='Treino')
        val_key = f'val_{key}'
        val_pairs = [(h['epoch'], h[val_key]) for h in history if val_key in h]
        if val_pairs:
            ep_v, vals_v = zip(*val_pairs)
            ax.plot(ep_v, vals_v, color=colors[i % len(colors)], linewidth=1.2, linestyle='--', alpha=0.7, label='Validação')
            ax.legend(fontsize=8)
        shade_stages(ax)
        ax.set_title(key.capitalize(), fontsize=10)
        ax.set_xlabel('Época', fontsize=9)
        ax.grid(axis='y', alpha=0.3)
        ax.tick_params(labelsize=8)
        if is_log_axis:
            ax.set_yscale('log')

    fig.suptitle(f'Histórico de Treinamento — {model_name}', fontsize=11, y=1.02)
    fig.tight_layout()
    name = 'training_loss_log.png' if is_log_axis else 'training_loss.png'
    out_path = os.path.join(out_dir, name)
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"  [train] Loss curve salva em {out_path}")

def plot_training_history(history, out_dir, model_name):
    _plot_training_history(history=history, out_dir=out_dir, model_name=model_name, is_log_axis=False)
    # _plot_training_history(history=history, out_dir=out_dir, model_name=model_name, is_log_axis=True)

def train_model(args):
    """Loop de treino principal."""
    model_name = args.model
    is_mc = model_name in _MC_MODELS
    is_temporal = model_name in _TEMPORAL_MODELS
    defaults = MODEL_DEFAULTS[model_name]
    arch_defs = ARCH_DEFAULTS.get(model_name, {})

    # Nome da variante (determina o diretório de saída)
    variant_name = args.name if args.name else model_name

    # Parâmetros efetivos de treino (CLI > defaults)
    max_epochs = args.max_epochs if args.max_epochs is not None else defaults["max_epochs"]
    lr = args.lr if args.lr is not None else defaults["lr"]
    batch_size = args.batch_size if args.batch_size is not None else defaults["batch_size"]
    kl_warmup = args.kl_warmup if args.kl_warmup is not None else defaults["kl_warmup"]
    mf_warmup = args.mf_warmup if args.mf_warmup is not None else defaults.get("mf_warmup", 0)
    latent_size = args.latent_size if args.latent_size is not None else defaults["latent_size"]
    norm_mode = args.normalization_mode if args.normalization_mode is not None else defaults["normalization_mode"]
    val_ratio = getattr(args, "val_ratio", 0.0) or 0.0
    early_stop_patience = getattr(args, "early_stop_patience", 0) or 0
    val_freq = getattr(args, "val_freq", 1) or 1

    latent_occ = args.latent_occ if args.latent_occ is not None else defaults.get("latent_occ")
    latent_amt = args.latent_amt if args.latent_amt is not None else defaults.get("latent_amt")


    # Parâmetros de arquitetura (CLI > ARCH_DEFAULTS do modelo; None = não relevante para esse modelo)
    def _arch(name):
        cli_val = getattr(args, name, None)
        return cli_val if cli_val is not None else arch_defs.get(name)

    hidden_size   = _arch("hidden_size")
    n_layers      = _arch("n_layers")
    n_coupling    = _arch("n_coupling")
    hidden_occ    = _arch("hidden_occ")
    hidden_amt    = _arch("hidden_amt")
    gru_hidden    = _arch("gru_hidden")
    context_dim   = _arch("context_dim")
    window_size   = _arch("window_size")
    hidden_dim    = _arch("hidden_dim")
    t_embed_dim   = _arch("t_embed_dim")
    n_sample_steps = _arch("n_sample_steps")

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[{variant_name}] Device: {device}")

    # Diretório de saída (usa variant_name, não model_name)
    out_dir = os.path.join(args.output_dir, variant_name)
    os.makedirs(out_dir, exist_ok=True)

    # Config — salva todos os parâmetros para reconstrução exata do modelo
    config = {
        "model": model_name,
        "variant_name": variant_name,
        "data_path": args.data_path,
        "max_epochs": max_epochs,
        "lr": lr,
        "batch_size": batch_size,
        "kl_warmup": kl_warmup,
        "mf_warmup": mf_warmup,
        "latent_size": latent_size,
        "latent_occ": latent_occ,
        "latent_amt": latent_amt,
        "normalization_mode": norm_mode,
        "device": str(device),
        "n_samples": args.n_samples,
        "holdout_ratio": args.holdout_ratio,
        "val_ratio": val_ratio,
        "early_stop_patience": early_stop_patience,
        "val_freq": val_freq,
        # Parâmetros de arquitetura
        "hidden_size":    hidden_size,
        "n_layers":       n_layers,
        "n_coupling":     n_coupling,
        "hidden_occ":     hidden_occ,
        "hidden_amt":     hidden_amt,
        "gru_hidden":     gru_hidden,
        "context_dim":    context_dim,
        "window_size":    window_size,
        "hidden_dim":     hidden_dim,
        "t_embed_dim":    t_embed_dim,
        "n_sample_steps": n_sample_steps,
        "rnn_hidden":     _arch("rnn_hidden"),
        "rnn_type":       _arch("rnn_type"),
        "n_steps":        _arch("n_steps"),
        "mf_ratio":       _arch("mf_ratio"),
        "jvp_eps":        _arch("jvp_eps"),
        "occ_weight":     _arch("occ_weight"),
        "lsd_weight":              _arch("lsd_weight"),
        "ayf_weight":              _arch("ayf_weight"),
        "ayf_delta_t":             _arch("ayf_delta_t"),
        "tangent_warmup_steps":    _arch("tangent_warmup_steps"),
        "improved_interval_sampling": _arch("improved_interval_sampling"),
        "mu_sad":                  _arch("mu_sad"),
        "sigma_sad":               _arch("sigma_sad"),
    }
    # Parâmetros específicos do LDM (estágio 2: DDPM)
    if model_name == "ldm":
        for k in ("ldm_epochs", "ldm_lr", "ldm_timesteps", "ldm_hidden_size",
                  "ldm_num_layers", "ldm_time_embed_dim"):
            if k in defaults:
                config[k] = defaults[k]
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    # config.json para _mc: cond_probs salvo após set_cond_distribution (abaixo)

    # ── Dados (brutos completos em ordem temporal) ──
    if is_mc or is_temporal:
        _, data_raw_full, _, _, station_names, cond_arrays_full = load_data_with_cond(
            data_path=args.data_path,
            normalization_mode="scale_only",
        )
        train_raw, val_raw, test_raw, train_cond, val_cond, test_cond = \
            temporal_train_val_test_split_with_cond(
                data_raw_full, cond_arrays_full, args.holdout_ratio, val_ratio
            )
    else:
        _, data_raw_full, _, _, station_names = load_data(
            data_path=args.data_path,
            normalization_mode="scale_only",
        )
        train_raw, val_raw, test_raw = temporal_train_val_test_split(
            data_raw_full, args.holdout_ratio, val_ratio
        )
        train_cond = val_cond = test_cond = None
    mu, std = compute_norm_params(train_raw, norm_mode)
    train_norm = normalize_with_params(train_raw, mu, std)
    val_norm   = normalize_with_params(val_raw, mu, std) if val_raw is not None else None
    test_norm  = normalize_with_params(test_raw, mu, std)  # noqa: F841 (available if needed)
    # eval_norm/eval_cond → val set for in-loop checkpoint selection (None disables in-loop val)
    # eval_raw            → test set for final evaluate_model() call
    eval_norm  = val_norm
    eval_raw   = test_raw
    eval_cond  = val_cond

    input_size = train_norm.shape[1]
    val_shape_str = str(val_raw.shape) if val_raw is not None else "N/A"
    print(f"[{variant_name}] Dados totais: {data_raw_full.shape} | treino: {train_raw.shape} | "
          f"val: {val_shape_str} | test: {test_raw.shape}")
    print(f"[{variant_name}] Normalização ajustada apenas no treino ({norm_mode}).")

    # Parâmetros extras de arquitetura a passar ao modelo (apenas os não-None)
    extra_model_kwargs = {}
    for key, val in [
        ("hidden_size",    hidden_size),
        ("n_layers",       n_layers),
        ("n_coupling",     n_coupling),
        ("hidden_occ",     hidden_occ),
        ("hidden_amt",     hidden_amt),
        ("gru_hidden",     gru_hidden),
        ("context_dim",    context_dim),
        ("window_size",    window_size),
        ("hidden_dim",     hidden_dim),
        ("t_embed_dim",    t_embed_dim),
        ("n_sample_steps", n_sample_steps),
        ("rnn_hidden",     _arch("rnn_hidden")),
        ("rnn_type",       _arch("rnn_type")),
        ("n_steps",        _arch("n_steps")),
        ("mf_ratio",       _arch("mf_ratio")),
        ("jvp_eps",        _arch("jvp_eps")),    # new
        ("occ_weight",     _arch("occ_weight")), # new
        ("lsd_weight",             _arch("lsd_weight")),
        ("ayf_weight",             _arch("ayf_weight")),
        ("ayf_delta_t",            _arch("ayf_delta_t")),
        ("tangent_warmup_steps",   _arch("tangent_warmup_steps")),
        ("improved_interval_sampling", _arch("improved_interval_sampling")),
        ("mu_sad",                 _arch("mu_sad")),
        ("sigma_sad",              _arch("sigma_sad")),
    ]:
        if val is not None:
            extra_model_kwargs[key] = val

    # parâmetros latentes específicos de hurdle
    if latent_occ is not None:
        extra_model_kwargs["latent_occ"] = latent_occ
    if latent_amt is not None:
        extra_model_kwargs["latent_amt"] = latent_amt

    # Parâmetros extras do LDM (estágio 2: DDPM)
    if model_name == "ldm":
        for k in ("ldm_timesteps", "ldm_hidden_size", "ldm_num_layers", "ldm_time_embed_dim"):
            v = defaults.get(k)
            if v is not None:
                extra_model_kwargs[k] = v

    # ── Instancia modelo ──
    model = instantiate_model(model_name, input_size, latent_size, device, extra_model_kwargs)
    print(f"[{variant_name}] Parâmetros: {model.count_parameters():,}")

    # Teacher loading for AYF-EMD distillation
    if getattr(args, 'teacher_checkpoint', None):
        from ar.loader import load_ar_model as _load_ar_model
        _teacher_variant = os.path.basename(args.teacher_checkpoint.rstrip('/\\'))
        _teacher_dir = os.path.dirname(args.teacher_checkpoint.rstrip('/\\'))
        print(f"Loading teacher from {args.teacher_checkpoint}...")
        _teacher = _load_ar_model(
            variant=_teacher_variant,
            output_dir=_teacher_dir,
            input_size=input_size,
            device=device,
        )
        _teacher.to(device)
        if hasattr(model, 'set_teacher'):
            model.set_teacher(_teacher)
            print(f"Teacher attached: {_teacher_variant}")
        else:
            print(f"WARNING: model has no set_teacher() method; --teacher_checkpoint ignored")

    # ── Retoma o treinamento se a flag --resume for usada ──
    model_path = os.path.join(out_dir, "model.pt")
    optimizer_state = None
    
    if getattr(args, "resume", False):
        if os.path.exists(model_path):
            print(f"[{variant_name}] Flag --resume detectada. Carregando checkpoint de {model_path}...")
            checkpoint = torch.load(model_path, map_location=device)
            
            # Verifica se é o formato novo (dict com model e optimizer) ou antigo (só model)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer_state = checkpoint['optimizer_state_dict']
            else:
                print(f"[{variant_name}] Formato de checkpoint antigo detectado. Carregando apenas o modelo.")
                model.load_state_dict(checkpoint)
        else:
            print(f"[{variant_name}] Aviso: --resume passado, mas {model_path} não existe. Começando do zero.")

    _opt_config = None
    if getattr(args, 'optimize', False):
        if str(device) != 'cpu':
            print(f"[optimize] Ignored: device is {device}, not cpu")
        else:
            model, use_amp, amp_dtype = _setup_intel_opts(model, args)
            _opt_config = (use_amp, amp_dtype)

    # ── Cópula: ajuste analítico ──
    if model_name == "copula":
        model.fit(train_raw)
        save_path = os.path.join(out_dir, "copula.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[{variant_name}] Salvo em {save_path}")

        if getattr(args, "skip_eval", False):
            print(f"[{variant_name}] --skip_eval set: skipping evaluation (metrics.json will not be written).")
            return

        metrics = evaluate_model(
            model,
            eval_raw,
            mu,
            std,
            n_samples=args.n_samples,
            station_names=station_names,
            samples_are_normalized=False,
        )
        metrics["evaluation_protocol"] = {
            "type": "temporal_holdout",
            "holdout_ratio": args.holdout_ratio,
            "train_size": int(train_raw.shape[0]),
            "eval_size": int(eval_raw.shape[0]),
        }
        with open(os.path.join(out_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"[{variant_name}] Métricas salvas em {out_dir}/metrics.json")
        return

    # ── Pré-ajustes específicos por modelo ──────────────────────────────────────
    if model_name == "hurdle_simple":
        model.fit_copulas(train_raw)

    if model_name == "hurdle_simple_mc":
        model.fit_copulas(train_raw)
        model.set_cond_distribution(train_cond)

    if model_name == "vae_mc":
        model.set_cond_distribution(train_cond)

    if model_name in ("real_nvp_mc", "glow_mc", "flow_match_mc", "flow_match_film_mc", "hurdle_vae_cond_mc", "latent_fm_mc", "hurdle_latent_fm_mc", "thresholded_latent_fm_mc", "thresholded_vae_mc", "thresholded_real_nvp_mc", "thresholded_glow_mc"):
        model.set_cond_distribution(train_cond)

    if model_name == "hurdle_temporal":
        model.fit_copulas(train_raw)
        model.fit_temporal(train_norm)

    if model_name == "latent_flow":
        model.fit_flow(train_raw, std_scale=std)

    # ── Atualiza config.json com cond_probs (modelos _mc) ─────────────────────
    if is_mc:
        config["cond_probs"] = {
            k: v.tolist() for k, v in model._cond_probs.items()
        }
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    # ── Treino ─────────────────────────────────────────────────────────────────

    if model_name == "ldm":
        # ── LDM: Estágio 1 (VAE) ──────────────────────────────────────────────
        model.set_stage("vae")
        history, ms_per_epoch, opt_state, intr1 = train_neural_model(
            model=model,
            train_norm=train_norm,
            max_epochs=max_epochs,
            lr=lr,
            batch_size=batch_size,
            kl_warmup=kl_warmup,
            device=device,
            model_name=f"{model_name}/vae",
            optimizer_state=optimizer_state,
            eval_norm=eval_norm,
            out_dir=out_dir,
            opt_config=_opt_config,
            early_stop_patience=early_stop_patience,
            val_freq=val_freq,
        )

        # ── LDM: Estágio 2 (DDPM no espaço latente) ──────────────────────────
        ldm_epochs = defaults.get("ldm_epochs", 300)
        ldm_lr = defaults.get("ldm_lr", 0.001)
        model.set_stage("ldm")
        history2, ms2, final_opt_state, intr2 = train_neural_model(
            model=model,
            train_norm=train_norm,
            max_epochs=ldm_epochs,
            lr=ldm_lr,
            batch_size=batch_size,
            kl_warmup=0,
            device=device,
            model_name=f"{model_name}/ddpm",
            print_every=25,
            optimizer_state=optimizer_state,
            eval_norm=eval_norm,
            out_dir=out_dir,
            opt_config=_opt_config,
            early_stop_patience=early_stop_patience,
            val_freq=val_freq,
        )
        interrupted = intr1 or intr2
        for entry in history:
            entry['stage'] = 'vae'
        for entry in history2:
            entry['stage'] = 'ldm'
        history = history + history2
        # Renumera épocas globalmente para eixo x monotônico
        for i, h in enumerate(history):
            h['epoch'] = i + 1
        # ms_per_epoch combinado ponderado pelo número de épocas
        total_epochs = max_epochs + ldm_epochs
        ms_per_epoch = (ms_per_epoch * max_epochs + ms2 * ldm_epochs) / max(total_epochs, 1)

    elif model_name in ("latent_fm_mc", "hurdle_latent_fm_mc", "thresholded_latent_fm_mc"):
        # ── Estágio 1: VAE ────────────────────────────────────────────────────
        flow_epochs = defaults.get("flow_epochs", 500)
        flow_lr     = defaults.get("flow_lr", 0.0002)

        model.set_stage("vae")
        history, ms_per_epoch, opt_state, intr1 = train_neural_model_mc(
            model=model,
            train_norm=train_norm,
            train_cond=train_cond,
            max_epochs=max_epochs,
            lr=lr,
            batch_size=batch_size,
            kl_warmup=kl_warmup,
            device=device,
            model_name=f"{model_name}/vae",
            optimizer_state=optimizer_state,
            eval_norm=eval_norm,
            eval_cond=eval_cond,
            out_dir=out_dir,
            opt_config=_opt_config,
            early_stop_patience=early_stop_patience,
            val_freq=val_freq,
        )

        # ── Estágio 2: Flow Matching ──────────────────────────────────────────
        model.set_stage("flow")
        history2, ms2, final_opt_state, intr2 = train_neural_model_mc(
            model=model,
            train_norm=train_norm,
            train_cond=train_cond,
            max_epochs=flow_epochs,
            lr=flow_lr,
            batch_size=batch_size,
            kl_warmup=0,
            device=device,
            model_name=f"{model_name}/flow",
            optimizer_state=None,   # optimizer novo para o flow
            eval_norm=eval_norm,
            eval_cond=eval_cond,
            out_dir=out_dir,
            opt_config=_opt_config,
            early_stop_patience=early_stop_patience,
            val_freq=val_freq,
        )

        # ── Combina histórico (igual ao LDM) ──────────────────────────────────
        interrupted = intr1 or intr2
        for entry in history:
            entry['stage'] = 'vae'
        for entry in history2:
            entry['stage'] = 'flow'
        history = history + history2
        for i, h in enumerate(history):
            h['epoch'] = i + 1
        total_epochs = max_epochs + flow_epochs
        ms_per_epoch = (ms_per_epoch * max_epochs + ms2 * flow_epochs) / max(total_epochs, 1)

    elif is_mc:
        history, ms_per_epoch, final_opt_state, interrupted = train_neural_model_mc(
            model=model,
            train_norm=train_norm,
            train_cond=train_cond,
            max_epochs=max_epochs,
            lr=lr,
            batch_size=batch_size,
            kl_warmup=kl_warmup,
            device=device,
            model_name=model_name,
            optimizer_state=optimizer_state,
            eval_norm=eval_norm,
            eval_cond=eval_cond,
            out_dir=out_dir,
            opt_config=_opt_config,
            early_stop_patience=early_stop_patience,
            val_freq=val_freq,
        )
    elif is_temporal:
        history, ms_per_epoch, final_opt_state, interrupted = train_neural_model_temporal(
            model=model,
            train_norm=train_norm,
            train_cond=train_cond,
            window_size=window_size or 30,
            max_epochs=max_epochs,
            lr=lr,
            batch_size=batch_size,
            kl_warmup=kl_warmup,
            device=device,
            mf_warmup=mf_warmup,
            model_name=model_name,
            optimizer_state=optimizer_state,
            eval_norm=eval_norm,
            eval_cond=eval_cond,
            out_dir=out_dir,
            opt_config=_opt_config,
            early_stop_patience=early_stop_patience,
            val_freq=val_freq,
        )
    else:
        history, ms_per_epoch, final_opt_state, interrupted = train_neural_model(
            model=model,
            train_norm=train_norm,
            max_epochs=max_epochs,
            lr=lr,
            batch_size=batch_size,
            kl_warmup=kl_warmup,
            device=device,
            model_name=model_name,
            optimizer_state=optimizer_state,
            eval_norm=eval_norm,
            out_dir=out_dir,
            opt_config=_opt_config,
            early_stop_patience=early_stop_patience,
            val_freq=val_freq,
        )

    # ── Se interrompido, carrega melhor checkpoint salvo em disco ──────────
    if interrupted:
        best_val_path   = os.path.join(out_dir, "model_best_val.pt")
        best_train_path = os.path.join(out_dir, "model_best_train.pt")
        ckpt_path = best_val_path if os.path.exists(best_val_path) else (
                    best_train_path if os.path.exists(best_train_path) else None)
        if ckpt_path:
            ckpt = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            final_opt_state = ckpt.get("optimizer_state_dict", final_opt_state)
            print(f"[{model_name}] Loaded checkpoint: {os.path.basename(ckpt_path)}")
        else:
            print(f"[{model_name}] No checkpoint found — evaluating in-memory state.")

    # ── Salva modelo e otimizador ──
    model_path = os.path.join(out_dir, "model.pt")
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': final_opt_state
    }, model_path)
    print(f"[{variant_name}] Checkpoint (Modelo + Otimizador) salvo em {model_path}")

    # ── Salva histórico de treino e plota curvas de perda ──
    history_path = os.path.join(out_dir, "training_history.json")
    # Se estiver retomando, anexa o novo histórico ao antigo
    if getattr(args, "resume", False) and os.path.exists(history_path):
        with open(history_path, "r") as f:
            old_history = json.load(f)
        
        # Ajusta a numeração das épocas no novo histórico
        last_epoch = old_history[-1]['epoch'] if old_history else 0
        for h in history:
            h['epoch'] += last_epoch
            
        history = old_history + history

    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[{variant_name}] Histórico salvo em {history_path}")
    plot_training_history(history, out_dir, variant_name)

    # ── Métricas ──
    if getattr(args, "skip_eval", False):
        print(f"[{variant_name}] --skip_eval set: skipping evaluation (metrics.json will not be written).")
        return
    model.eval()
    # AR/temporal models do sequential rollouts — cap samples and timing trials
    # to keep evaluation under ~5 min instead of 30+ min
    if is_temporal:
        eval_n_samples = min(args.n_samples, 200)
        timing_n_samples = 100
        timing_n_trials = 1
        print(f"[{variant_name}] [Phase: Evaluation] AR model detected — "
              f"using {eval_n_samples} samples, {timing_n_trials} timing trial "
              f"(reduced from {args.n_samples} / 5 to avoid multi-hour eval)")
    else:
        eval_n_samples = args.n_samples
        timing_n_samples = 1000
        timing_n_trials = 5
        print(f"[{variant_name}] [Phase: Evaluation] Generating {eval_n_samples} samples...")
    metrics = evaluate_model(
        model,
        eval_raw,
        mu,
        std,
        n_samples=eval_n_samples,
        station_names=station_names,
        timing_n_samples=timing_n_samples,
        timing_n_trials=timing_n_trials,
    )
    metrics['final_epoch'] = len(history)
    metrics['final_train_loss'] = history[-1]['total']
    metrics['best_train_loss'] = min(h['total'] for h in history)
    val_losses = [h['val_total'] for h in history if 'val_total' in h]
    if val_losses:
        metrics['best_val_loss'] = min(val_losses)
    metrics['training_ms_per_epoch'] = ms_per_epoch
    metrics['n_parameters'] = model.count_parameters()
    metrics["evaluation_protocol"] = {
        "type": "temporal_train_val_test" if val_ratio > 0 else "temporal_holdout",
        "holdout_ratio": args.holdout_ratio,
        "val_ratio": val_ratio,
        "train_size": int(train_raw.shape[0]),
        "val_size": int(val_raw.shape[0]) if val_raw is not None else 0,
        "test_size": int(eval_raw.shape[0]),
        "early_stopped": interrupted and early_stop_patience > 0,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{variant_name}] Métricas salvas em {out_dir}/metrics.json")


def main():
    parser = argparse.ArgumentParser(
        description="Treina um modelo generativo para precipitação.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=MODEL_NAMES,
                        help="Modelo a treinar")
    parser.add_argument("--name", type=str, default=None,
                        help="Nome da variante (padrão: nome do modelo). Usado como diretório de saída.")
    # ── Parâmetros de treino ──────────────────────────────────────────────────
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Número de épocas (usa default do modelo se omitido)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Taxa de aprendizado")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Tamanho do batch")
    parser.add_argument("--latent_size", type=int, default=None,
                        help="Dimensão do espaço latente (VAE/HurdleVAE)")
        
    parser.add_argument("--latent_occ", type=int, default=None,
                        help="Dimensão latente de ocorrência (hurdle_vae, etc.)")
    parser.add_argument("--latent_amt", type=int, default=None,
                        help="Dimensão latente de quantidade (hurdle_vae, etc.)")
    
    parser.add_argument("--kl_warmup", type=int, default=None,
                        help="Épocas de warmup do KL (VAE/HurdleVAE)")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cpu', ou 'cuda'")
    parser.add_argument("--data_path", type=str, default="../dados_barragens_btg/inmet_relevant_data.csv",
                        help="Caminho para inmet_relevant_data.csv")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Diretório base para salvar resultados")
    parser.add_argument("--n_samples", type=int, default=5000,
                        help="Amostras a gerar na avaliação")
    parser.add_argument("--holdout_ratio", type=float, default=0.0,
                        help="Proporção final da série usada como test set temporal")
    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="Fração do pré-test reservada para validação (split 3-vias). "
                             "0.0 (padrão) mantém comportamento atual (sem validação interna). "
                             "Exemplo: --val_ratio 0.1 usa 10%% do pré-holdout como val.")
    parser.add_argument("--early_stop_patience", type=int, default=0,
                        help="Early stopping: para após N checks de val sem melhora. "
                             "0 (padrão) desativado. Requer --val_ratio > 0.")
    parser.add_argument("--val_freq", type=int, default=1,
                        help="Valida a cada N épocas (reduz overhead). "
                             "Paciência do early stopping conta em checks, não épocas.")
    # ── Parâmetros de arquitetura ─────────────────────────────────────────────
    parser.add_argument("--hidden_size", type=int, default=None,
                        help="Tamanho das camadas ocultas (real_nvp, flow_match)")
    parser.add_argument("--n_layers", type=int, default=None,
                        help="Número de camadas (flow_match, latent_flow)")
    parser.add_argument("--n_coupling", type=int, default=None,
                        help="Número de camadas de acoplamento (real_nvp)")
    parser.add_argument("--hidden_occ", type=int, default=None,
                        help="Camadas ocultas do MLP de ocorrência (hurdle_simple, hurdle_temporal)")
    parser.add_argument("--hidden_amt", type=int, default=None,
                        help="Camadas ocultas do MLP de quantidade (hurdle_simple, hurdle_temporal)")
    parser.add_argument("--gru_hidden", type=int, default=None,
                        help="Dimensão oculta do GRU (hurdle_temporal)")
    parser.add_argument("--context_dim", type=int, default=None,
                        help="Dimensão do vetor de contexto (hurdle_temporal)")
    parser.add_argument("--window_size", type=int, default=None,
                        help="Tamanho da janela temporal (hurdle_temporal, latent_flow)")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Dimensão oculta do Transformer (latent_flow)")
    parser.add_argument("--t_embed_dim", type=int, default=None,
                        help="Dimensão do embedding de tempo (flow_match)")
    parser.add_argument("--n_sample_steps", type=int, default=None,
                        help="Número de passos de integração na amostragem (flow_match)")
    parser.add_argument("--rnn_hidden", type=int, default=None,
                        help="RNN hidden size (AR models)")
    parser.add_argument("--rnn_type", type=str, default=None, choices=["gru", "lstm"],
                        help="RNN type (AR models)")
    parser.add_argument("--n_steps", type=int, default=None,
                        help="Number of Glow steps (ar_glow)")
    parser.add_argument("--mf_ratio", type=float, default=None,
                        help="Mean flow ratio (ar_mean_flow)")
    parser.add_argument("--mf_warmup", type=int, default=None,
                        help="Epochs of pure FM before MeanFlow correction ramp (ar_mean_flow_v2)")
    parser.add_argument("--jvp_eps", type=float, default=None,
                        help="Finite-difference epsilon for du/dt in MeanFlow (ar_mean_flow family)")
    parser.add_argument("--occ_weight", type=float, default=None,
                        help="Weight of occurrence BCE loss (ar_vae_v2)")
    parser.add_argument("--lsd_weight", type=float, default=None,
                        help="LSD loss weight for ar_flow_map_sd (0.0=disabled)")
    parser.add_argument("--ayf_weight", type=float, default=None,
                        help="AYF-EMD distillation weight (0.0=disabled)")
    parser.add_argument("--ayf_delta_t", type=float, default=None,
                        help="Backstep size for AYF-EMD teacher distillation")
    parser.add_argument("--tangent_warmup_steps", type=int, default=None,
                        help="Steps to ramp tangent correction 0->1 (ar_mean_flow_ayfm)")
    parser.add_argument("--improved_interval_sampling", action="store_true", default=None,
                        help="Use N(mu,sigma)+sigmoid interval sampling (ar_mean_flow_ayfm)"
                        # NOTE: default=None (not False) is intentional — allows _arch() to fall
                        # through to ARCH_DEFAULTS, so ar_mean_flow_ayfm gets True by default.
                        # Changing to default=False would break the ARCH_DEFAULTS override.
                        )
    parser.add_argument("--mu_sad", type=float, default=None,
                        help="Mean of normal dist for improved interval sampling")
    parser.add_argument("--sigma_sad", type=float, default=None,
                        help="Std of normal dist for improved interval sampling")
    parser.add_argument("--teacher_checkpoint", type=str, default=None,
                        help="Path to teacher model outputs dir for AYF-EMD distillation")
    parser.add_argument("--skip_eval", action="store_true",
                        help="Skip post-training evaluation; do not write metrics.json. "
                             "Used by distributed worker — eval runs as a separate job.")

    parser.add_argument("--optimize", action="store_true",
                        help="Enable Intel CPU optimizations: torch.compile, BF16 autocast, thread tuning, IPEX if available")
    parser.add_argument("--num_threads", type=int, default=None,
                        help="Intra-op thread count for --optimize (default: 4, tuned for P-cores)")
    parser.add_argument("--resume", action="store_true",
                        help="Carrega o model.pt existente e retoma o treinamento de onde parou")
    parser.add_argument("--normalization_mode", type=str, default=None,
                        choices=["scale_only", "standardize"],
                        help="Normalizacao: scale-only = valor/std e standardize - (valor - media)/std")
    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
