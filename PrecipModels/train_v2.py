"""
train_v2.py — Script de treino com suporte a otimizadores separados por componente.

Extensão do train.py que ativa o protocolo multi-otimizador para modelos que
implementam get_optimizer_groups(). Modelos sem esse método usam o loop padrão
(compatibilidade total com train.py).

Uso:
    python train_v2.py --model hurdle_simple_v2
    python train_v2.py --model hurdle_flow_v2
    python train_v2.py --model hurdle_temporal_v2
    python train_v2.py --model hurdle_vae_v2 --kl_warmup 25

    # Modelos originais continuam funcionando (fallback automático):
    python train_v2.py --model vae --max_epochs 500
    python train_v2.py --model hurdle_simple

    # Smoke tests:
    python train_v2.py --model hurdle_simple_v2 --max_epochs 5
    python train_v2.py --model hurdle_flow_v2 --max_epochs 5
    python train_v2.py --model hurdle_temporal_v2 --max_epochs 5
    python train_v2.py --model hurdle_vae_v2 --max_epochs 5

Saídas em outputs/<model>/: model.pt, config.json, metrics.json
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import pickle
import time
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from data_utils import load_data, denormalize
from base_model import BaseModel
from models import get_model
from metrics import evaluate_model

# Registros v2 — apenas train_v2.py conhece esses modelos
from models_v2 import V2_MODEL_REGISTRY


# ──────────────────────────────────────────────────────────
# DEFAULTS POR MODELO
# ──────────────────────────────────────────────────────────

MODEL_DEFAULTS = {
    "copula": {
        "normalization_mode": "scale_only",
        "max_epochs": 0,
        "latent_size": 0,
        "lr": 0.0,
        "batch_size": 128,
        "kl_warmup": 0,
    },
    "vae": {
        "normalization_mode": "standardize",
        "max_epochs": 1000,
        "latent_size": 128,
        "lr": 0.0003,
        "batch_size": 128,
        "kl_warmup": 100,
    },
    "hurdle_simple": {
        "normalization_mode": "scale_only",
        "max_epochs": 1000,
        "latent_size": 0,
        "lr": 0.001,
        "batch_size": 128,
        "kl_warmup": 0,
    },
    "hurdle_vae": {
        "normalization_mode": "scale_only",
        "max_epochs": 2000,
        "latent_size": 64,
        "lr": 0.001,
        "batch_size": 128,
        "kl_warmup": 25,
    },
    "hurdle_vae_cond": {
        "normalization_mode": "scale_only",
        "max_epochs": 2000,
        "latent_size": 64,
        "lr": 0.001,
        "batch_size": 128,
        "kl_warmup": 200,
    },
    "hurdle_vae_cond_nll": {
        "normalization_mode": "scale_only",
        "max_epochs": 2000,
        "latent_size": 64,
        "lr": 0.001,
        "batch_size": 128,
        "kl_warmup": 200,
    },
    "real_nvp": {
        "normalization_mode": "standardize",
        "max_epochs": 2000,
        "latent_size": 0,
        "lr": 0.001,
        "batch_size": 128,
        "kl_warmup": 0,
    },
    "hurdle_flow": {
        "normalization_mode": "scale_only",
        "max_epochs": 1000,
        "latent_size": 0,
        "lr": 0.001,
        "batch_size": 128,
        "kl_warmup": 0,
    },
    "flow_match": {
        "normalization_mode": "standardize",
        "max_epochs": 1000,
        "latent_size": 0,
        "lr": 0.001,
        "batch_size": 128,
        "kl_warmup": 0,
    },
    "flow_match_film": {
        "normalization_mode": "standardize",
        "max_epochs": 1000,
        "latent_size": 0,
        "lr": 0.001,
        "batch_size": 128,
        "kl_warmup": 0,
    },
    "ldm": {
        "normalization_mode": "standardize",
        "max_epochs": 1000,
        "latent_size": 128,
        "lr": 0.005,
        "batch_size": 128,
        "kl_warmup": 25,
        "ldm_epochs": 1000,
        "ldm_lr": 0.001,
        "ldm_timesteps": 100,
        "ldm_hidden_size": 128,
        "ldm_num_layers": 3,
        "ldm_time_embed_dim": 64,
    },
    "hurdle_temporal": {
        "normalization_mode": "scale_only",
        "max_epochs": 1000,
        "latent_size": 0,
        "lr": 0.001,
        "batch_size": 128,
        "kl_warmup": 0,
    },
    "latent_flow": {
        "normalization_mode": "standardize",
        "max_epochs": 1000,
        "latent_size": 0,
        "lr": 0.0002,
        "batch_size": 128,
        "kl_warmup": 0,
    },
}

# Mapeamento nome-base → nome-v2 (para herdar defaults e pré-ajustes)
_BASE_NAME = {
    "hurdle_simple_v2":   "hurdle_simple",
    "hurdle_flow_v2":     "hurdle_flow",
    "hurdle_temporal_v2": "hurdle_temporal",
    "hurdle_vae_v2":      "hurdle_vae",
}

# Adiciona defaults v2 herdando do modelo base
for _v2, _base in _BASE_NAME.items():
    MODEL_DEFAULTS[_v2] = dict(MODEL_DEFAULTS[_base])


# ──────────────────────────────────────────────────────────
# FUNÇÕES AUXILIARES
# ──────────────────────────────────────────────────────────

def get_beta(epoch: int, kl_warmup: int) -> float:
    """KL annealing: rampa linear 0 → 1 ao longo de kl_warmup épocas."""
    if kl_warmup <= 0:
        return 1.0
    return min(1.0, (epoch + 1) / kl_warmup)


def temporal_holdout_split(data_raw: np.ndarray, holdout_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """Split temporal sem embaralhar: treino = parte inicial, avaliação = parte final."""
    if holdout_ratio <= 0:
        return data_raw, data_raw
    n_total = data_raw.shape[0]
    n_eval = max(1, int(round(n_total * holdout_ratio)))
    n_eval = min(n_eval, n_total - 1)
    return data_raw[:-n_eval], data_raw[-n_eval:]


def compute_norm_params(train_raw: np.ndarray, normalization_mode: str) -> Tuple[np.ndarray, np.ndarray]:
    std = np.std(train_raw, axis=0, keepdims=True)
    std = np.clip(std, 1e-8, None)
    if normalization_mode == "scale_only":
        mu = np.zeros((1, train_raw.shape[1]), dtype=train_raw.dtype)
    elif normalization_mode == "standardize":
        mu = np.mean(train_raw, axis=0, keepdims=True)
    else:
        raise ValueError(f"normalization_mode inválido: '{normalization_mode}'")
    return mu, std


def normalize_with_params(data_raw: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    return (data_raw - mu) / std


def instantiate_model(
    model_name: str,
    input_size: int,
    latent_size: int,
    device: torch.device,
    extra_kwargs: dict = None,
) -> BaseModel:
    """Instancia modelo: verifica primeiro V2_MODEL_REGISTRY, depois o registry original."""
    model_kwargs = dict(input_size=input_size)
    if latent_size > 0:
        model_kwargs["latent_size"] = latent_size
    if extra_kwargs:
        model_kwargs.update(extra_kwargs)

    if model_name in V2_MODEL_REGISTRY:
        model = V2_MODEL_REGISTRY[model_name](**model_kwargs)
    else:
        model = get_model(model_name, **model_kwargs)

    return model.to(device)


# ──────────────────────────────────────────────────────────
# LOOP DE TREINO — OTIMIZADOR ÚNICO (original)
# ──────────────────────────────────────────────────────────

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
) -> Tuple[List[dict], float]:
    t_data = torch.FloatTensor(train_norm).to(device)
    dataset = TensorDataset(t_data, t_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    print(f"\n[{model_name}] Iniciando treino: {max_epochs} épocas, lr={lr}, batch={batch_size}")
    print("-" * 60)
    print(f"{'Epoch':>7}  {'Loss Total':>12}  {'Sub-losses':>30}  {'Beta':>6}")
    print("-" * 60)

    train_start = time.perf_counter()
    history = []

    for epoch in range(max_epochs):
        model.train()
        running = {}
        n_batches = 0

        beta = get_beta(epoch, kl_warmup)

        for x_batch, _ in loader:
            optimizer.zero_grad()
            loss_dict = model.loss(x_batch, beta=beta)
            loss_dict['total'].backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + v.item()
            n_batches += 1

        avg = {k: v / n_batches for k, v in running.items()}
        avg['epoch'] = epoch + 1
        avg['beta'] = round(beta, 4)
        history.append(avg)

        if (epoch + 1) % print_every == 0 or epoch == max_epochs - 1:
            sub_str = "  ".join(
                f"{k}={v:.4f}" for k, v in avg.items()
                if k not in ('total', 'epoch', 'beta')
            )
            print(f"{epoch+1:7d}  {avg['total']:12.4f}  {sub_str:<30}  {beta:6.4f}")

    train_elapsed = time.perf_counter() - train_start
    ms_per_epoch = train_elapsed * 1000 / max(max_epochs, 1)
    print(f"\n[{model_name}] Treino concluído: {train_elapsed:.1f}s ({ms_per_epoch:.1f} ms/época)")
    return history, ms_per_epoch


# ──────────────────────────────────────────────────────────
# LOOP DE TREINO — MULTI-OTIMIZADOR (v2)
# ──────────────────────────────────────────────────────────

def train_neural_model_multi_opt(
    model: BaseModel,
    train_norm: np.ndarray,
    max_epochs: int,
    lr: float,
    batch_size: int,
    kl_warmup: int,
    device: torch.device,
    model_name: str,
    print_every: int = 50,
) -> Tuple[List[dict], float]:
    """
    Loop de treino com otimizadores separados por componente.

    Detecta automaticamente se o modelo implementa get_optimizer_groups().
    Se não implementar, faz fallback para train_neural_model() (comportamento idêntico).

    Protocolo multi-otimizador:
        - Grupos com combined=False: backward na sua própria loss, step imediato.
        - Grupos com combined=True: gradientes acumulam de todos os backward passes
          anteriores; step ocorre após todos os grupos normais.
        - retain_graph=True em todos os backward exceto o último, para preservar
          o grafo computacional de tensores compartilhados (e.g. GRU em hurdle_temporal).
    """
    if not hasattr(model, 'get_optimizer_groups'):
        return train_neural_model(
            model, train_norm, max_epochs, lr, batch_size,
            kl_warmup, device, model_name, print_every,
        )

    groups = model.get_optimizer_groups()
    normal_groups   = [g for g in groups if not g.get('combined', False)]
    combined_groups = [g for g in groups if g.get('combined', False)]

    group_names = [g['name'] for g in groups]
    print(f"\n[{model_name}] Protocolo multi-otimizador ativado.")
    print(f"[{model_name}] Grupos: {group_names}")

    opts_normal = [
        optim.Adam(g['params'], lr=lr * g.get('lr_scale', 1.0))
        for g in normal_groups
    ]
    opts_combined = [
        optim.Adam(g['params'], lr=lr * g.get('lr_scale', 1.0))
        for g in combined_groups
    ]
    all_opts = opts_normal + opts_combined

    t_data = torch.FloatTensor(train_norm).to(device)
    dataset = TensorDataset(t_data, t_data)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print(f"\n[{model_name}] Iniciando treino: {max_epochs} épocas, lr={lr}, batch={batch_size}")
    print("-" * 60)
    print(f"{'Epoch':>7}  {'Loss Total':>12}  {'Sub-losses':>30}  {'Beta':>6}")
    print("-" * 60)

    train_start = time.perf_counter()
    history = []
    n_normal = len(normal_groups)

    for epoch in range(max_epochs):
        model.train()
        running = {}
        n_batches = 0

        beta = get_beta(epoch, kl_warmup)

        for x_batch, _ in loader:
            # Zera todos os gradientes de uma vez
            for opt in all_opts:
                opt.zero_grad()

            # Computa todas as losses para o batch atual
            loss_dict = model.loss(x_batch, beta=beta)

            # Acumula total para logging (sem backward — apenas .item())
            total_val = loss_dict['total'].item()

            # Grupos normais: backward na própria loss, step imediato
            for i, (g, opt) in enumerate(zip(normal_groups, opts_normal)):
                if g.get('loss_fn') is not None:
                    target = g['loss_fn'](loss_dict, beta)
                else:
                    target = loss_dict[g['loss_key']]

                # retain_graph=True exceto no último backward (incluindo se há combined)
                is_last_backward = (i == n_normal - 1) and len(combined_groups) == 0
                target.backward(retain_graph=not is_last_backward)
                clip_grad_norm_(g['params'], 1.0)
                opt.step()

            # Grupos combined: gradientes já acumulados dos backward passes acima
            for g, opt in zip(combined_groups, opts_combined):
                clip_grad_norm_(g['params'], 1.0)
                opt.step()

            # Logging: acumula valores escalares
            for k, v in loss_dict.items():
                running[k] = running.get(k, 0.0) + v.item()
            n_batches += 1

        avg = {k: v / n_batches for k, v in running.items()}
        avg['epoch'] = epoch + 1
        avg['beta'] = round(beta, 4)
        history.append(avg)

        if (epoch + 1) % print_every == 0 or epoch == max_epochs - 1:
            sub_str = "  ".join(
                f"{k}={v:.4f}" for k, v in avg.items()
                if k not in ('total', 'epoch', 'beta')
            )
            print(f"{epoch+1:7d}  {avg['total']:12.4f}  {sub_str:<30}  {beta:6.4f}")

    train_elapsed = time.perf_counter() - train_start
    ms_per_epoch = train_elapsed * 1000 / max(max_epochs, 1)
    print(f"\n[{model_name}] Treino concluído: {train_elapsed:.1f}s ({ms_per_epoch:.1f} ms/época)")
    return history, ms_per_epoch


# ──────────────────────────────────────────────────────────
# PLOT DO HISTÓRICO
# ──────────────────────────────────────────────────────────

def _plot_training_history(history: list, out_dir: str, model_name: str, is_log_axis=False):
    """Plota as curvas de perda por época e salva em training_loss.png."""
    import matplotlib.pyplot as plt

    epochs = [h['epoch'] for h in history]
    total = [h['total'] for h in history]

    meta_keys = {'epoch', 'beta', 'total', 'stage'}
    sub_keys = sorted(set(k for h in history for k in h if k not in meta_keys))

    n_plots = 1 + len(sub_keys) if sub_keys else 1

    fig, axes = plt.subplots(1, n_plots, figsize=(4.5 * n_plots, 4))
    if n_plots == 1:
        axes = [axes]

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

    ax = axes[0]
    ax.plot(epochs, total, color='steelblue', linewidth=1.2)
    shade_stages(ax)
    ax.set_title('Total Loss', fontsize=10)
    ax.set_xlabel('Época', fontsize=9)
    ax.set_ylabel('Loss', fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    ax.tick_params(labelsize=8)
    if has_stages:
        ax.legend(fontsize=8)
    if is_log_axis:
        ax.set_yscale('log')

    colors = ['#e74c3c', '#2ecc71', '#9b59b6', '#f39c12']
    for i, key in enumerate(sub_keys):
        ax = axes[i + 1]
        pairs = [(h['epoch'], h[key]) for h in history if key in h]
        if not pairs:
            continue
        ep_filt, vals = zip(*pairs)
        ax.plot(ep_filt, vals, color=colors[i % len(colors)], linewidth=1.2)
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


# ──────────────────────────────────────────────────────────
# LOOP PRINCIPAL
# ──────────────────────────────────────────────────────────

def train_model(args):
    """Loop de treino principal (suporta modelos originais e v2)."""
    model_name = args.model
    defaults = MODEL_DEFAULTS[model_name]

    # Nome base para pré-ajustes (hurdle_simple_v2 → hurdle_simple, etc.)
    effective_base = _BASE_NAME.get(model_name, model_name)

    # Parâmetros efetivos (CLI > defaults)
    max_epochs = args.max_epochs if args.max_epochs is not None else defaults["max_epochs"]
    lr = args.lr if args.lr is not None else defaults["lr"]
    batch_size = args.batch_size if args.batch_size is not None else defaults["batch_size"]
    kl_warmup = args.kl_warmup if args.kl_warmup is not None else defaults["kl_warmup"]
    latent_size = args.latent_size if args.latent_size is not None else defaults["latent_size"]
    norm_mode = defaults["normalization_mode"]

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[{model_name}] Device: {device}")

    # Diretório de saída
    out_dir = os.path.join(args.output_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    # Config
    config = {
        "model": model_name,
        "max_epochs": max_epochs,
        "lr": lr,
        "batch_size": batch_size,
        "kl_warmup": kl_warmup,
        "latent_size": latent_size,
        "normalization_mode": norm_mode,
        "device": str(device),
        "n_samples": args.n_samples,
        "holdout_ratio": args.holdout_ratio,
    }
    if model_name == "ldm":
        for k in ("ldm_epochs", "ldm_lr", "ldm_timesteps", "ldm_hidden_size",
                  "ldm_num_layers", "ldm_time_embed_dim"):
            if k in defaults:
                config[k] = defaults[k]
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # ── Dados ──
    _, data_raw_full, _, _, station_names = load_data(
        data_path=args.data_path,
        normalization_mode="scale_only",
    )
    train_raw, eval_raw = temporal_holdout_split(data_raw_full, args.holdout_ratio)
    mu, std = compute_norm_params(train_raw, norm_mode)
    train_norm = normalize_with_params(train_raw, mu, std)

    input_size = train_norm.shape[1]
    print(f"[{model_name}] Dados totais: {data_raw_full.shape} | treino: {train_raw.shape} | holdout: {eval_raw.shape}")
    print(f"[{model_name}] Normalização ajustada apenas no treino ({norm_mode}).")

    # Parâmetros extras específicos por modelo
    extra_model_kwargs = {}
    if effective_base == "ldm":
        for k in ("ldm_timesteps", "ldm_hidden_size", "ldm_num_layers", "ldm_time_embed_dim"):
            v = defaults.get(k)
            if v is not None:
                extra_model_kwargs[k] = v

    # ── Instancia modelo ──
    model = instantiate_model(model_name, input_size, latent_size, device, extra_model_kwargs)
    print(f"[{model_name}] Parâmetros: {model.count_parameters():,}")

    # ── Cópula: ajuste analítico ──
    if effective_base == "copula":
        model.fit(train_raw)
        save_path = os.path.join(out_dir, "copula.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(model, f)
        print(f"[copula] Salvo em {save_path}")

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
        print(f"[copula] Métricas salvas em {out_dir}/metrics.json")
        return

    # ── Pré-ajustes específicos por modelo (usando effective_base) ──────────
    if effective_base == "hurdle_simple":
        model.fit_copulas(train_raw)

    if effective_base == "hurdle_temporal":
        model.fit_copulas(train_raw)
        model.fit_temporal(train_norm)

    if effective_base == "latent_flow":
        model.fit_flow(train_raw, std_scale=std)

    # ── Treino ─────────────────────────────────────────────────────────────
    if effective_base == "ldm":
        # Estágio 1: VAE
        model.set_stage("vae")
        history, ms_per_epoch = train_neural_model(
            model=model,
            train_norm=train_norm,
            max_epochs=max_epochs,
            lr=lr,
            batch_size=batch_size,
            kl_warmup=kl_warmup,
            device=device,
            model_name=f"{model_name}/vae",
        )

        # Estágio 2: DDPM no espaço latente
        ldm_epochs = defaults.get("ldm_epochs", 300)
        ldm_lr = defaults.get("ldm_lr", 0.001)
        model.set_stage("ldm")
        history2, ms2 = train_neural_model(
            model=model,
            train_norm=train_norm,
            max_epochs=ldm_epochs,
            lr=ldm_lr,
            batch_size=batch_size,
            kl_warmup=0,
            device=device,
            model_name=f"{model_name}/ddpm",
            print_every=25,
        )
        for entry in history:
            entry['stage'] = 'vae'
        for entry in history2:
            entry['stage'] = 'ldm'
        history = history + history2
        # Renumera épocas globalmente para eixo x monotônico
        for i, h in enumerate(history):
            h['epoch'] = i + 1
        total_epochs = max_epochs + ldm_epochs
        ms_per_epoch = (ms_per_epoch * max_epochs + ms2 * ldm_epochs) / max(total_epochs, 1)
    else:
        # Todos os outros modelos: tenta multi-otimizador, faz fallback automático
        history, ms_per_epoch = train_neural_model_multi_opt(
            model=model,
            train_norm=train_norm,
            max_epochs=max_epochs,
            lr=lr,
            batch_size=batch_size,
            kl_warmup=kl_warmup,
            device=device,
            model_name=model_name,
        )

    # ── Salva modelo ──
    model_path = os.path.join(out_dir, "model.pt")
    torch.save(model.state_dict(), model_path)
    print(f"[{model_name}] Modelo salvo em {model_path}")

    # ── Salva histórico e plota curvas ──
    history_path = os.path.join(out_dir, "training_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[{model_name}] Histórico salvo em {history_path}")
    plot_training_history(history, out_dir, model_name)

    # ── Métricas ──
    model.eval()
    metrics = evaluate_model(
        model,
        eval_raw,
        mu,
        std,
        n_samples=args.n_samples,
        station_names=station_names,
    )
    metrics['final_epoch'] = max_epochs
    metrics['final_train_loss'] = history[-1]['total']
    metrics['training_ms_per_epoch'] = ms_per_epoch
    metrics['n_parameters'] = model.count_parameters()
    metrics["evaluation_protocol"] = {
        "type": "temporal_holdout",
        "holdout_ratio": args.holdout_ratio,
        "train_size": int(train_raw.shape[0]),
        "eval_size": int(eval_raw.shape[0]),
    }

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[{model_name}] Métricas salvas em {out_dir}/metrics.json")


def main():
    # Todos os modelos suportados (originais + v2)
    all_model_choices = [
        "copula",
        "vae",
        "hurdle_simple",
        "hurdle_vae",
        "hurdle_vae_cond",
        "hurdle_vae_cond_nll",
        "real_nvp",
        "hurdle_flow",
        "flow_match",
        "flow_match_film",
        "ldm",
        "hurdle_temporal",
        "latent_flow",
        # Variantes v2 com otimizadores separados
        "hurdle_simple_v2",
        "hurdle_flow_v2",
        "hurdle_temporal_v2",
        "hurdle_vae_v2",
    ]

    parser = argparse.ArgumentParser(
        description="Treina um modelo generativo para precipitação (suporta otimizadores separados para modelos v2).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", type=str, required=True,
                        choices=all_model_choices,
                        help="Modelo a treinar")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Número de épocas (usa default do modelo se omitido)")
    parser.add_argument("--lr", type=float, default=None,
                        help="Taxa de aprendizado")
    parser.add_argument("--batch_size", type=int, default=None,
                        help="Tamanho do batch")
    parser.add_argument("--latent_size", type=int, default=None,
                        help="Dimensão do espaço latente (VAE/HurdleVAE)")
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
    parser.add_argument("--holdout_ratio", type=float, default=0.2,
                        help="Proporção final da série usada como avaliação temporal")

    args = parser.parse_args()
    train_model(args)


if __name__ == "__main__":
    main()
