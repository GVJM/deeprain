"""
ablation_hurdle_vae_cond.py — Ablation sistemático do melhor modelo (hurdle_vae_cond).

Varre uma grade de hiperparâmetros e compara composite scores:
    - latent_occ × latent_amt ∈ {16, 32, 64, 128}
    - kl_warmup ∈ {10, 25, 50, 100}

Uso:
    # Teste rápido (grade pequena, poucas épocas)
    python ablation_hurdle_vae_cond.py --max_epochs 50 --n_samples 500 \\
           --latent_sizes 16 64 --warmup_values 10 50

    # Ablation completo
    python ablation_hurdle_vae_cond.py --max_epochs 500 --n_samples 2000

    # Pula treino e só plota (se já tiver resultados)
    python ablation_hurdle_vae_cond.py --skip_training

Saídas em --output_dir (default: ./outputs/ablation/):
    ablation_results.json        — todas as métricas por config
    ablation_latent_heatmap.png  — heatmap composite: latent_occ × latent_amt
    ablation_warmup_bar.png      — bar chart composite por kl_warmup
    ablation_full_table.txt      — tabela texto com todas as métricas
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import argparse
import itertools
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch

# Garante imports do diretório PrecipModels
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_utils import load_data, denormalize
from metrics import evaluate_model
from models import get_model
from train import train_model  # reutiliza o loop de treino existente


# ──────────────────────────────────────────────────────────
# GRADE DE HIPERPARÂMETROS
# ──────────────────────────────────────────────────────────

DEFAULT_LATENT_SIZES = [16, 32, 64, 128]
DEFAULT_WARMUP_VALUES = [10, 25, 50, 100]

# Hiperparâmetros fixos durante o ablation (valores default do modelo)
FIXED_LR = 0.001
FIXED_EPOCHS = 500
FIXED_BATCH_SIZE = 128
FIXED_N_SAMPLES = 2000


def config_key(latent_occ: int, latent_amt: int, kl_warmup: int) -> str:
    """Gera chave única para uma configuração."""
    return f"lo{latent_occ}_la{latent_amt}_kw{kl_warmup}"


# ──────────────────────────────────────────────────────────
# TREINO E AVALIAÇÃO DE UMA CONFIGURAÇÃO
# ──────────────────────────────────────────────────────────

def train_and_evaluate(
    latent_occ: int,
    latent_amt: int,
    kl_warmup: int,
    data_norm: np.ndarray,
    data_raw: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    station_names: list,
    max_epochs: int,
    n_samples: int,
    device: torch.device,
) -> dict:
    """
    Instancia, treina e avalia um HurdleVAECond com os hiperparâmetros dados.

    Returns:
        dict de métricas (output de evaluate_model + config usada)
    """
    print(f"\n  Treinando: latent_occ={latent_occ}, latent_amt={latent_amt}, kl_warmup={kl_warmup}")

    model = get_model(
        "hurdle_vae_cond",
        input_size=data_raw.shape[1],
        latent_occ=latent_occ,
        latent_amt=latent_amt,
    ).to(device)

    # Dataset e DataLoader
    tensor = torch.FloatTensor(data_norm).to(device)
    dataset = torch.utils.data.TensorDataset(tensor)
    loader = torch.utils.data.DataLoader(dataset, batch_size=FIXED_BATCH_SIZE, shuffle=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=FIXED_LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=50
    )

    best_loss = float("inf")
    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_loss = 0.0
        # KL annealing: beta sobe de 0 → 1 ao longo de kl_warmup épocas
        beta = min(1.0, epoch / max(1, kl_warmup))
        for (xb,) in loader:
            optimizer.zero_grad()
            loss_dict = model.loss(xb, beta=beta)
            loss_dict["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss_dict["total"].item() * len(xb)
        epoch_loss /= len(tensor)
        scheduler.step(epoch_loss)
        best_loss = min(best_loss, epoch_loss)

        if epoch % 100 == 0 or epoch == max_epochs:
            print(f"    Época {epoch}/{max_epochs} | loss={epoch_loss:.4f} | beta={beta:.3f}")

    model.eval()
    metrics = evaluate_model(
        model, data_raw, mu, std,
        n_samples=n_samples,
        station_names=station_names,
        samples_are_normalized=True,
    )
    metrics["config"] = {
        "latent_occ": latent_occ,
        "latent_amt": latent_amt,
        "kl_warmup": kl_warmup,
        "max_epochs": max_epochs,
        "n_parameters": model.count_parameters(),
    }
    return metrics


# ──────────────────────────────────────────────────────────
# COMPOSITE SCORE (mesmo critério do compare.py)
# ──────────────────────────────────────────────────────────

QUALITY_METRIC_KEYS = [
    "mean_wasserstein",
    "corr_rmse",
    "wet_day_freq_error_mean",
    "extreme_q90_mean",
    "extreme_q95_mean",
    "extreme_q99_mean",
    "energy_score",
]


def compute_composite_single(all_results: dict) -> dict:
    """
    Normaliza métricas de qualidade em [0,1] e faz média por configuração.
    Retorna dict {config_key: composite_score}.
    """
    keys = list(all_results.keys())
    scores = {}

    metric_arrays = {}
    for mk in QUALITY_METRIC_KEYS:
        vals = []
        for k in keys:
            v = all_results[k].get(mk, np.nan)
            vals.append(float(v) if v is not None else np.nan)
        metric_arrays[mk] = np.array(vals, dtype=float)

    normalized = {k: {} for k in keys}
    for mk, vals in metric_arrays.items():
        valid = ~np.isnan(vals)
        if valid.sum() < 2:
            continue
        mn, mx = vals[valid].min(), vals[valid].max()
        rng = mx - mn + 1e-12
        norm = (vals - mn) / rng  # lower is better → norm=0 é melhor
        for i, k in enumerate(keys):
            normalized[k][mk] = float(norm[i]) if valid[i] else np.nan

    for k in keys:
        vals = [v for v in normalized[k].values() if not np.isnan(v)]
        scores[k] = float(np.mean(vals)) if vals else np.nan

    return scores


# ──────────────────────────────────────────────────────────
# GRÁFICOS
# ──────────────────────────────────────────────────────────

def plot_latent_heatmap(
    all_results: dict,
    scores: dict,
    latent_sizes: list,
    kl_warmup_best: int,
    out_dir: str,
):
    """
    Heatmap: linhas = latent_occ, colunas = latent_amt.
    Usa o melhor kl_warmup encontrado para fixar o 3º eixo.
    Valor em cada célula = composite score (menor = melhor).
    """
    n = len(latent_sizes)
    grid = np.full((n, n), np.nan)

    for i, lo in enumerate(latent_sizes):
        for j, la in enumerate(latent_sizes):
            key = config_key(lo, la, kl_warmup_best)
            if key in scores:
                grid[i, j] = scores[key]

    fig, ax = plt.subplots(figsize=(7, 6))
    vmin = np.nanmin(grid)
    vmax = np.nanmax(grid)
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels([f"latent_amt={la}" for la in latent_sizes], fontsize=10)
    ax.set_yticklabels([f"latent_occ={lo}" for lo in latent_sizes], fontsize=10)
    ax.set_title(
        f"Ablation: Composite Score (kl_warmup={kl_warmup_best})\n"
        f"Verde = melhor  |  Vermelho = pior",
        fontsize=12, fontweight="bold"
    )

    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Composite Score (↓ melhor)", fontsize=9)

    for i in range(n):
        for j in range(n):
            val = grid[i, j]
            if not np.isnan(val):
                # Destacar mínimo
                is_best = (val == np.nanmin(grid))
                color = "black"
                weight = "bold" if is_best else "normal"
                marker = "★ " if is_best else ""
                ax.text(j, i, f"{marker}{val:.4f}", ha="center", va="center",
                        fontsize=9, color=color, fontweight=weight)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "ablation_latent_heatmap.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[ablation] Heatmap salvo: {out_path}")


def plot_warmup_bar(
    all_results: dict,
    scores: dict,
    warmup_values: list,
    latent_occ_best: int,
    latent_amt_best: int,
    out_dir: str,
):
    """
    Bar chart: composite score por kl_warmup (fixando latent_occ e latent_amt ótimos).
    """
    warmup_scores = []
    for kw in warmup_values:
        key = config_key(latent_occ_best, latent_amt_best, kw)
        warmup_scores.append(scores.get(key, np.nan))

    warmup_scores = np.array(warmup_scores)
    best_idx = int(np.nanargmin(warmup_scores))

    colors = ["#2ecc71" if i == best_idx else "#3498db" for i in range(len(warmup_values))]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar([str(w) for w in warmup_values], warmup_scores, color=colors, edgecolor="white", linewidth=0.5)

    for bar, val in zip(bars, warmup_scores):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{val:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    ax.set_xlabel("KL Warmup (épocas)", fontsize=11)
    ax.set_ylabel("Composite Score (↓ melhor)", fontsize=11)
    ax.set_title(
        f"Ablation: KL Warmup\n"
        f"(latent_occ={latent_occ_best}, latent_amt={latent_amt_best})\n"
        f"Verde = melhor",
        fontsize=12, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(0, max(np.nanmax(warmup_scores) * 1.15, 0.01))

    import matplotlib.patches as mpatches
    best_patch = mpatches.Patch(color="#2ecc71", label=f"Melhor: warmup={warmup_values[best_idx]}")
    ax.legend(handles=[best_patch], fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "ablation_warmup_bar.png")
    plt.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[ablation] Bar chart warmup salvo: {out_path}")


def plot_metric_breakdown(all_results: dict, scores: dict, out_dir: str):
    """
    Para as top-5 configurações: bar chart multi-métrica para comparação detalhada.
    """
    # Ordena por composite score e pega top 5
    sorted_keys = sorted(scores.keys(), key=lambda k: scores.get(k, float("inf")))
    top_keys = sorted_keys[:min(5, len(sorted_keys))]

    metrics_to_plot = [
        ("mean_wasserstein", "Wasserstein"),
        ("corr_rmse", "Corr RMSE"),
        ("wet_day_freq_error_mean", "Wet Day Freq"),
        ("extreme_q90_mean", "Q90 Err"),
        ("energy_score", "Energy Score"),
    ]

    n_metrics = len(metrics_to_plot)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))

    colors = plt.cm.tab10(np.linspace(0, 1, len(top_keys)))

    for ax, (mkey, mlabel) in zip(axes, metrics_to_plot):
        vals = [all_results[k].get(mkey, np.nan) for k in top_keys]
        labels = [k.replace("lo", "occ").replace("la", ",amt=").replace("_kw", "\nkw=") for k in top_keys]
        bar_colors = ["#2ecc71" if i == int(np.nanargmin(vals)) else "#3498db" for i in range(len(vals))]
        ax.bar(range(len(top_keys)), vals, color=bar_colors)
        ax.set_xticks(range(len(top_keys)))
        ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
        ax.set_title(mlabel, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Breakdown de Métricas — Top 5 Configurações", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "ablation_metric_breakdown.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[ablation] Metric breakdown salvo: {out_path}")


def print_table(all_results: dict, scores: dict, warmup_values: list, latent_sizes: list) -> str:
    """Imprime e retorna tabela texto com todas as configurações rankeadas."""
    sorted_keys = sorted(scores.keys(), key=lambda k: scores.get(k, float("inf")))
    lines = [
        "",
        "=" * 90,
        "  ABLATION: hurdle_vae_cond — Ranking por Composite Score",
        "=" * 90,
        f"{'Config':<28} {'Wasserstein':>12} {'Corr RMSE':>10} {'Wet Day':>9} {'Q90':>8} {'Q99':>8} {'Energy':>10} {'Composite':>11} {'Params':>8}",
        "-" * 90,
    ]
    for rank, key in enumerate(sorted_keys, 1):
        m = all_results[key]
        cfg = m.get("config", {})
        lo = cfg.get("latent_occ", "?")
        la = cfg.get("latent_amt", "?")
        kw = cfg.get("kl_warmup", "?")
        label = f"occ={lo},amt={la},kw={kw}"

        w = m.get("mean_wasserstein", np.nan)
        c = m.get("corr_rmse", np.nan)
        wf = m.get("wet_day_freq_error_mean", np.nan)
        q90 = m.get("extreme_q90_mean", np.nan)
        q99 = m.get("extreme_q99_mean", np.nan)
        es = m.get("energy_score", np.nan)
        sc = scores.get(key, np.nan)
        np_ = cfg.get("n_parameters", 0)

        lines.append(
            f"#{rank:<2} {label:<26} {w:>12.4f} {c:>10.4f} {wf:>9.4f} {q90:>8.4f} {q99:>8.4f} {es:>10.4f} {sc:>11.4f} {np_:>8,}"
        )

    lines.append("=" * 90)

    # Vencedor global
    best_key = sorted_keys[0]
    bcfg = all_results[best_key].get("config", {})
    lines.append(
        f"\nMELHOR CONFIGURAÇÃO: latent_occ={bcfg.get('latent_occ')} | latent_amt={bcfg.get('latent_amt')} | kl_warmup={bcfg.get('kl_warmup')}"
    )
    lines.append(f"Composite Score: {scores[best_key]:.4f}")
    lines.append("")

    report = "\n".join(lines)
    print(report)
    return report


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Ablation sistemático do hurdle_vae_cond (latente + KL warmup).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--max_epochs", type=int, default=FIXED_EPOCHS,
                        help="Épocas de treino por configuração")
    parser.add_argument("--n_samples", type=int, default=FIXED_N_SAMPLES,
                        help="Amostras para avaliação")
    parser.add_argument("--latent_sizes", type=int, nargs="+", default=DEFAULT_LATENT_SIZES,
                        help="Tamanhos de espaço latente a testar (usado para latent_occ E latent_amt)")
    parser.add_argument("--warmup_values", type=int, nargs="+", default=DEFAULT_WARMUP_VALUES,
                        help="Valores de kl_warmup a testar")
    parser.add_argument("--output_dir", type=str, default="./outputs/ablation",
                        help="Diretório de saída")
    parser.add_argument("--data_path", type=str, default="../dados_sabesp/dayprecip.dat",
                        help="Caminho para o arquivo de dados")
    parser.add_argument("--skip_training", action="store_true",
                        help="Pula treino e carrega resultados existentes de ablation_results.json")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cpu', ou 'cuda'")
    args = parser.parse_args()

    # Dispositivo
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[ablation] Usando device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)
    results_path = os.path.join(args.output_dir, "ablation_results.json")

    # ── Dados ─────────────────────────────────────────────
    print("[ablation] Carregando dados...")
    data_norm, data_raw, mu, std, station_names = load_data(
        data_path=args.data_path,
        normalization_mode="scale_only",
    )
    print(f"[ablation] Dados: {data_raw.shape} | Estações: {len(station_names)}")

    # ── Treino ou carregamento ─────────────────────────────
    if args.skip_training and os.path.exists(results_path):
        print(f"[ablation] Carregando resultados de: {results_path}")
        with open(results_path, encoding="utf-8") as f:
            all_results = json.load(f)
    else:
        all_results = {}
        grid = list(itertools.product(args.latent_sizes, args.latent_sizes, args.warmup_values))
        total = len(grid)
        print(f"[ablation] Grade: {total} configurações | {args.max_epochs} épocas cada")
        print(f"[ablation] latent_sizes={args.latent_sizes} | warmup_values={args.warmup_values}\n")

        for i, (lo, la, kw) in enumerate(grid, 1):
            key = config_key(lo, la, kw)
            print(f"[ablation] [{i}/{total}] {key}")
            try:
                metrics = train_and_evaluate(
                    latent_occ=lo,
                    latent_amt=la,
                    kl_warmup=kw,
                    data_norm=data_norm,
                    data_raw=data_raw,
                    mu=mu,
                    std=std,
                    station_names=station_names,
                    max_epochs=args.max_epochs,
                    n_samples=args.n_samples,
                    device=device,
                )
                # Converte arrays numpy para listas para serialização JSON
                serializable = {}
                for mk, mv in metrics.items():
                    if isinstance(mv, dict):
                        serializable[mk] = {
                            k2: (v2.tolist() if hasattr(v2, "tolist") else v2)
                            for k2, v2 in mv.items()
                        }
                    elif hasattr(mv, "tolist"):
                        serializable[mk] = mv.tolist()
                    else:
                        serializable[mk] = mv
                all_results[key] = serializable

                # Salva incrementalmente (segurança em caso de falha)
                with open(results_path, "w", encoding="utf-8") as f:
                    json.dump(all_results, f, indent=2)

            except Exception as e:
                print(f"[AVISO] Falha em {key}: {e}")

    if not all_results:
        print("[ablation] Nenhum resultado disponível. Abortando.")
        return

    # ── Composite scores ───────────────────────────────────
    scores = compute_composite_single(all_results)

    # ── Relatório texto ────────────────────────────────────
    report = print_table(all_results, scores, args.warmup_values, args.latent_sizes)
    with open(os.path.join(args.output_dir, "ablation_full_table.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # ── Identifica melhor configuração ────────────────────
    best_key = min(scores, key=scores.get)
    best_cfg = all_results[best_key].get("config", {})
    best_lo = best_cfg.get("latent_occ", args.latent_sizes[0])
    best_la = best_cfg.get("latent_amt", args.latent_sizes[0])
    best_kw = best_cfg.get("kl_warmup", args.warmup_values[0])

    # Para o heatmap: fixa melhor warmup e varia latentes
    # Para o bar chart: fixa melhor latente e varia warmup
    plot_latent_heatmap(all_results, scores, args.latent_sizes, best_kw, args.output_dir)
    plot_warmup_bar(all_results, scores, args.warmup_values, best_lo, best_la, args.output_dir)
    plot_metric_breakdown(all_results, scores, args.output_dir)

    print(f"\n[ablation] ✓ Concluído! Resultados em: {args.output_dir}/")
    print(f"[ablation] Melhor: {best_key} → composite={scores[best_key]:.4f}")


if __name__ == "__main__":
    main()
