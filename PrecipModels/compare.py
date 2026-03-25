"""
compare.py — Treina todos os modelos e gera tabela + gráficos comparativos.

Uso:
    python compare.py                                      # treina todos e compara
    python compare.py --skip_training                      # carrega métricas existentes
    python compare.py --max_epochs 200                     # teste rápido
    python compare.py --models vae copula                  # compara subconjunto
    python compare.py --solver_grid --models flow_match_film ldm  # variantes de solver/steps

Saídas em outputs/comparison/:
    composite_scores.json     — pontuação composta normalizada
    comparison_quality.png    — bar charts por métrica (verde = melhor)
    radar.png                 — radar chart comparativo
    station_samples_comparison.png — distribuição de amostras por estação e modelo
    overall_correlation_chart.png  — correlações espaciais (real + modelos)
    scenario_comparison_by_model/<model>/plt03_*.png e plt04_*.png — via auxplot.py
    comparison_report.txt     — tabela texto
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import pickle
import subprocess
import sys
import math
import io
import contextlib
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Metrics importadas para avaliação direta
import torch
from data_utils import load_data, load_data_with_cond, denormalize
from metrics import evaluate_model
from models import MODEL_NAMES, get_model

# Reuso de plots legados do módulo VAE_Tests.
VAE_TESTS_DIR = Path(__file__).resolve().parent.parent / "VAE_Tests"
if str(VAE_TESTS_DIR) not in sys.path:
    sys.path.insert(0, str(VAE_TESTS_DIR))
try:
    from auxplot import plot_scenario_comparison as auxplot_scenario_comparison
except Exception:
    auxplot_scenario_comparison = None


ALL_MODELS = MODEL_NAMES  # inclui ldm, hurdle_temporal, latent_flow


# ──────────────────────────────────────────────────────────
# SOLVER GRID — variantes de sampling por solver/passos
# ──────────────────────────────────────────────────────────

# Cada entrada: (model_name, steps, method, display_tag)
# Usado apenas quando --solver_grid é passado para compare.py.
SOLVER_GRID = [
    ("flow_match",      10,  "euler",    "euler_10"),
    ("flow_match",      50,  "euler",    "euler_50"),
    ("flow_match",      10,  "heun",     "heun_10"),
    ("flow_match",      50,  "heun",     "heun_50"),
    ("flow_match_film", 10,  "euler",    "euler_10"),
    ("flow_match_film", 50,  "euler",    "euler_50"),
    ("flow_match_film", 10,  "heun",     "heun_10"),
    ("flow_match_film", 50,  "heun",     "heun_50"),
    ("latent_flow",     10,  "euler",    "euler_10"),
    ("latent_flow",     30,  "midpoint", "midpoint_30"),
    ("ldm",             20,  "ddim",     "ddim_20"),
    ("ldm",             100, "ddpm",     "ddpm_100"),
]


# ──────────────────────────────────────────────────────────
# COMPOSITE SCORE
# ──────────────────────────────────────────────────────────

from scoring import QUALITY_METRICS, _metric_array, compute_composite


# ──────────────────────────────────────────────────────────
# TREINO DE TODOS OS MODELOS
# ──────────────────────────────────────────────────────────

def train_all(models: list, max_epochs: int, n_samples: int, output_dir: str, data_path: str):
    """Chama train.py para cada modelo em sequência."""
    for model_name in models:
        print(f"\n{'='*60}")
        print(f"  Treinando: {model_name.upper()}")
        print(f"{'='*60}")

        cmd = [
            sys.executable, "train.py",
            "--model", model_name,
            "--output_dir", output_dir,
            "--n_samples", str(n_samples),
            "--data_path", data_path,
        ]
        if max_epochs is not None and model_name != "copula":
            cmd += ["--max_epochs", str(max_epochs)]

        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        if result.returncode != 0:
            print(f"[AVISO] Treino de '{model_name}' falhou (código {result.returncode})")


# ──────────────────────────────────────────────────────────
# CARREGA MÉTRICAS
# ──────────────────────────────────────────────────────────

def load_all_metrics(models: list, output_dir: str) -> dict:
    """Carrega metrics.json de cada modelo."""
    all_metrics = {}
    for m in models:
        path = os.path.join(output_dir, m, "metrics.json")
        if os.path.exists(path):
            with open(path) as f:
                all_metrics[m] = json.load(f)
        else:
            print(f"[AVISO] Métricas não encontradas: {path}")
    return all_metrics


# ──────────────────────────────────────────────────────────
# GRÁFICOS
# ──────────────────────────────────────────────────────────

def plot_bar_charts(all_metrics: dict, scores: dict, out_dir: str):
    """Bar charts por métrica, verde = melhor."""
    model_names = list(all_metrics.keys())
    n_metrics = len(QUALITY_METRICS)
    n_cols = 3
    n_rows = (n_metrics + 1) // n_cols + 1  # +1 para composite

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    axes = axes.flatten()

    for idx, (key, label, lower_is_better) in enumerate(QUALITY_METRICS):
        ax = axes[idx]
        vals_np = _metric_array(all_metrics, model_names, key)
        valid = ~np.isnan(vals_np)
        if valid.any():
            best_idx = np.argmin(vals_np[valid]) if lower_is_better else np.argmax(vals_np[valid])
            best_name = np.array(model_names)[valid][best_idx]
            colors = ['#2ecc71' if m == best_name else 'steelblue' for m in model_names]
            plot_vals = vals_np
        else:
            colors = ['#95a5a6' for _ in model_names]
            plot_vals = np.zeros_like(vals_np)

        bars = ax.bar(model_names, plot_vals, color=colors)
        ax.set_title(label, fontsize=10, fontweight='bold')
        ax.set_ylabel('Valor (menor = melhor)' if lower_is_better else 'Valor (maior = melhor)', fontsize=8)
        ax.tick_params(axis='x', rotation=45, labelsize=8)
        ax.grid(axis='y', alpha=0.3)

        # Anota valores válidos
        for bar, val in zip(bars, vals_np):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f'{val:.3f}', ha='center', va='bottom', fontsize=7)
        if not valid.any():
            ax.text(0.5, 0.5, "Sem dados", transform=ax.transAxes,
                    ha='center', va='center', fontsize=9, color='#7f8c8d')

    # Composite score
    ax_comp = axes[n_metrics]
    comp_np = np.array([float(scores.get(m, np.nan)) for m in model_names], dtype=float)
    valid = ~np.isnan(comp_np)
    if valid.any():
        best_idx = np.argmin(comp_np[valid])
        best_name = np.array(model_names)[valid][best_idx]
        colors = ['#2ecc71' if m == best_name else '#e67e22' for m in model_names]
        plot_comp = comp_np
    else:
        colors = ['#95a5a6' for _ in model_names]
        plot_comp = np.zeros_like(comp_np)
    ax_comp.bar(model_names, plot_comp, color=colors)
    ax_comp.set_title("Composite Score (menor = melhor)", fontsize=10, fontweight='bold')
    ax_comp.tick_params(axis='x', rotation=45, labelsize=8)
    ax_comp.grid(axis='y', alpha=0.3)
    for bar, val in zip(ax_comp.patches, comp_np):
        if not np.isnan(val):
            ax_comp.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                         f'{val:.3f}', ha='center', va='bottom', fontsize=7)
    if not valid.any():
        ax_comp.text(0.5, 0.5, "Sem dados", transform=ax_comp.transAxes,
                     ha='center', va='center', fontsize=9, color='#7f8c8d')

    # Esconde eixos extras
    for i in range(n_metrics + 1, len(axes)):
        axes[i].set_visible(False)

    green_patch = mpatches.Patch(color='#2ecc71', label='Melhor nesta métrica')
    fig.legend(handles=[green_patch], loc='lower right', fontsize=9)
    fig.suptitle("Comparação de Modelos Generativos — Precipitação", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "comparison_quality.png"), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Gráfico salvo: {out_dir}/comparison_quality.png")


def plot_radar(all_metrics: dict, normalized: dict, out_dir: str):
    """Radar chart comparativo — área menor = modelo melhor."""
    model_names = list(all_metrics.keys())
    metric_labels = [label for _, label, _ in QUALITY_METRICS]
    N = len(metric_labels)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # fecha o polígono

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    colors = plt.cm.tab10(np.linspace(0, 1, len(model_names)))

    for i, m in enumerate(model_names):
        vals = [normalized.get(m, {}).get(key, 0.5) for key, _, _ in QUALITY_METRICS]
        vals += vals[:1]
        ax.plot(angles, vals, 'o-', linewidth=2, color=colors[i], label=m)
        ax.fill(angles, vals, alpha=0.1, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=7)
    ax.set_title("Radar Chart — Qualidade Normalizada\n(menor área = melhor)", fontsize=12)
    ax.legend(loc='upper right', bbox_to_anchor=(1.35, 1.1), fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "radar.png"), dpi=120, bbox_inches='tight')
    plt.close()
    print(f"Radar salvo: {out_dir}/radar.png")


def _load_model_config(model_name: str, output_dir: str) -> dict:
    path = os.path.join(output_dir, model_name, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_trained_model(variant_name: str, output_dir: str, input_size: int, device: torch.device):
    model_dir = os.path.join(output_dir, variant_name)
    cfg = _load_model_config(variant_name, output_dir)

    # Classe real do modelo (pode diferir do nome da variante)
    model_class = cfg.get("model", variant_name)

    if model_class == "copula":
        copula_path = os.path.join(model_dir, "copula.pkl")
        if not os.path.exists(copula_path):
            raise FileNotFoundError(copula_path)
        with open(copula_path, "rb") as f:
            model = pickle.load(f)
        return model

    # Constrói kwargs a partir do config — todos os parâmetros de arquitetura
    model_kwargs = dict(input_size=input_size)
    for key in (
        "latent_size",
        "latent_occ", "latent_amt",
        "hidden_size", "n_layers", "n_coupling",
        "hidden_occ", "hidden_amt",
        "gru_hidden", "context_dim", "window_size",
        "hidden_dim", "t_embed_dim", "n_sample_steps",
        "ldm_timesteps", "ldm_hidden_size", "ldm_num_layers", "ldm_time_embed_dim",
    ):
        val = cfg.get(key)
        if val is not None:
            model_kwargs[key] = int(val)

    model = get_model(model_class, **model_kwargs)
    model_path = os.path.join(model_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(model_path)
    state = torch.load(model_path, map_location=device)
    # Compatibilidade: checkpoints podem vir como state_dict puro
    # ou como dict com {"model_state_dict", "optimizer_state_dict"}.
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # Restaura distribuição empírica de condicionadores para modelos _mc
    cond_probs_raw = cfg.get("cond_probs")
    if cond_probs_raw and hasattr(model, "_cond_probs"):
        model._cond_probs = {k: np.array(v) for k, v in cond_probs_raw.items()}

    return model


def _sample_generated_mm(
    model_name: str,
    model,
    n_samples: int,
    norm_mode: str,
    norm_cache: dict,
    steps: int | None = None,
    method: str | None = None,
) -> np.ndarray:
    with torch.no_grad():
        samples = model.sample(n_samples, steps=steps, method=method)
    if isinstance(samples, torch.Tensor):
        samples = samples.detach().cpu().numpy()

    if model_name == "copula":
        generated = samples
    else:
        mu, std = norm_cache[norm_mode]
        generated = denormalize(samples, mu, std)
    return np.clip(generated, 0, None)


def _uniform_subset(data: np.ndarray, n: int) -> np.ndarray:
    if data.shape[0] <= n:
        return data
    idx = np.random.choice(data.shape[0], n, replace=False)
    return data[idx]


def plot_station_samples_comparison(
    real_data: np.ndarray,
    generated_by_model: dict,
    station_names: list,
    out_dir: str,
    max_points_per_series: int,
):
    """Distribuição por estação: Real vs modelos (boxplots)."""
    model_names = list(generated_by_model.keys())
    S = len(station_names)
    n_cols = 3
    n_rows = math.ceil(S / n_cols)

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 3.8 * n_rows))
    axes = np.array(axes).reshape(-1)

    real_sub = _uniform_subset(real_data, max_points_per_series)
    series_colors = ["#2c3e50"] + list(plt.cm.Set2(np.linspace(0, 1, len(model_names))))

    for i, st_name in enumerate(station_names):
        ax = axes[i]
        values = [real_sub[:, i]]
        labels = ["real"]

        for m in model_names:
            values.append(_uniform_subset(generated_by_model[m], max_points_per_series)[:, i])
            labels.append(m)

        bp = ax.boxplot(values, tick_labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(bp["boxes"], series_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.85)

        ax.set_title(st_name, fontsize=9, fontweight="bold")
        ax.tick_params(axis="x", rotation=45, labelsize=8)
        ax.grid(axis="y", alpha=0.25)
        if i % n_cols == 0:
            ax.set_ylabel("Precipitação (mm/dia)", fontsize=8)

    for i in range(S, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle("Comparação de Amostras por Estação (Real vs Modelos)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "station_samples_comparison.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {out_dir}/station_samples_comparison.png")


def plot_auxplot_scenarios(real_data: np.ndarray, generated_by_model: dict, out_dir: str):
    """
    Reusa VAE_Tests/auxplot.plot_scenario_comparison de forma organizada:
      outputs/comparison/scenario_comparison_by_model/<modelo>/
    """
    if auxplot_scenario_comparison is None:
        print("[AVISO] auxplot.plot_scenario_comparison indisponível; pulando plot legado.")
        return

    base_dir = os.path.join(out_dir, "scenario_comparison_by_model")
    os.makedirs(base_dir, exist_ok=True)

    for model_name, generated in generated_by_model.items():
        model_out = os.path.join(base_dir, model_name)
        os.makedirs(model_out, exist_ok=True)
        try:
            auxplot_scenario_comparison(model_out, real_data, generated)
        except Exception as e:
            print(f"[AVISO] Falha no auxplot para '{model_name}': {e}")


def plot_overall_correlation_chart(
    real_data: np.ndarray,
    generated_by_model: dict,
    out_dir: str,
    all_metrics: dict,
    max_points_per_series: int,
):
    """Painel com matrizes de correlação: dados reais + cada modelo."""
    panels = [("real", _uniform_subset(real_data, max_points_per_series))]
    for model_name, data in generated_by_model.items():
        panels.append((model_name, _uniform_subset(data, max_points_per_series)))

    corr_mats = []
    for _, data in panels:
        corr_mats.append(np.corrcoef(data, rowvar=False))

    n_panels = len(panels)
    n_cols = min(3, n_panels)
    n_rows = math.ceil(n_panels / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.8 * n_cols, 4.5 * n_rows))
    axes = np.array(axes).reshape(-1)

    for i, ((name, _), mat) in enumerate(zip(panels, corr_mats)):
        ax = axes[i]
        im = ax.imshow(mat, cmap="coolwarm", vmin=-1, vmax=1)
        if name == "real":
            title = "Real (referência)"
        else:
            rmse = all_metrics.get(name, {}).get("corr_rmse", float("nan"))
            title = f"{name} (corr RMSE={rmse:.3f})" if not np.isnan(rmse) else name
        ax.set_title(title, fontsize=10, fontweight="bold")
        ax.set_xticks([])
        ax.set_yticks([])

    for i in range(n_panels, len(axes)):
        axes[i].set_visible(False)

    cbar = fig.colorbar(im, ax=axes[:n_panels], shrink=0.8)
    cbar.set_label("Correlação", fontsize=9)
    fig.suptitle("Overall Correlation Chart (Real + Modelos)", fontsize=13, fontweight="bold")
    fig.subplots_adjust(left=0.04, right=0.92, top=0.90, bottom=0.05, wspace=0.12, hspace=0.18)
    plt.savefig(os.path.join(out_dir, "overall_correlation_chart.png"), dpi=120, bbox_inches="tight")
    plt.close()
    print(f"Gráfico salvo: {out_dir}/overall_correlation_chart.png")


def plot_qqplots_by_station(
    real_data: np.ndarray,
    generated_by_model: dict,
    station_names: list,
    out_dir: str,
    n_quantiles: int = 100,
):
    """
    QQ-plots por estação: quantis empíricos do real (eixo X) vs gerado (eixo Y).

    Cada linha = uma estação. Cada coluna = um modelo.
    A diagonal preta tracejada representa a referência perfeita (gerado = real).
    Desvios acima da diagonal = modelo gera valores mais altos que o real.
    Desvios abaixo = modelo gera valores mais baixos (sub-estimação).

    Salva: outputs/comparison/qqplots_by_station.png
    """
    model_names = list(generated_by_model.keys())
    S = len(station_names)
    n_models = len(model_names)

    if n_models == 0:
        print("[compare] Nenhum modelo para QQ-plots.")
        return

    # Grid: linhas = estações, colunas = modelos (máx 5 por linha)
    n_cols = min(n_models, 5)
    n_rows = S

    fig_width = max(4 * n_cols, 12)
    fig_height = max(2.8 * n_rows, 8)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height), squeeze=False)

    quantile_levels = np.linspace(0.01, 0.99, n_quantiles)
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))

    for row, st_name in enumerate(station_names):
        real_col = real_data[:, row]
        q_real = np.quantile(real_col, quantile_levels)

        for col_idx, (m_name, color) in enumerate(zip(model_names, colors)):
            if col_idx >= n_cols:
                break
            ax = axes[row][col_idx]
            gen_col = generated_by_model[m_name][:, row]
            q_gen = np.quantile(gen_col, quantile_levels)

            # Linha de referência diagonal perfeita
            ref_min = min(float(q_real.min()), float(q_gen.min()))
            ref_max = max(float(q_real.max()), float(q_gen.max()))
            ax.plot([ref_min, ref_max], [ref_min, ref_max],
                    "k--", linewidth=1.0, alpha=0.6, zorder=1)

            # Pontos QQ coloridos
            ax.scatter(q_real, q_gen, s=8, alpha=0.75, color=color, zorder=2)

            # Título = modelo (primeira linha), ylabel = estação (primeira coluna)
            if row == 0:
                ax.set_title(m_name, fontsize=8, fontweight="bold")
            if col_idx == 0:
                ax.set_ylabel(st_name, fontsize=7, rotation=0, labelpad=55, va="center")
            if row == n_rows - 1:
                ax.set_xlabel("Quantis reais (mm/dia)", fontsize=7)

            ax.tick_params(labelsize=6)
            ax.grid(alpha=0.2)

        # Esconde colunas extras se houver menos modelos que n_cols
        for col_idx in range(n_models, n_cols):
            axes[row][col_idx].set_visible(False)

    fig.suptitle(
        "QQ-Plots por Estação — Quantis Reais vs Gerados\n"
        "(pontos na diagonal = distribuição perfeita; acima = super-estimação; abaixo = sub-estimação)",
        fontsize=12, fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = os.path.join(out_dir, "qqplots_by_station.png")
    plt.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close()
    print(f"QQ-plots salvos: {out_path}")


def plot_seasonal_metrics(
    mc_models: dict,
    data_raw: np.ndarray,
    norm_cache: dict,
    out_dir: str,
    data_raw_cond_arrays: dict,
    n_per_month: int = 500,
    station_names: list | None = None,
):
    """
    Gera análise sazonal para modelos condicionados (_mc):
      - seasonal_wet_freq.png: frequência de dias chuvosos por mês (real vs modelos)
      - seasonal_wasserstein.png: Wasserstein médio por mês

    Args:
        mc_models: dict[variant_name, (model, norm_mode)]
        data_raw: (N, S) dados reais completos
        norm_cache: dict[norm_mode, (mu, std)]
        n_per_month: amostras geradas por mês para avaliação
    """
    from scipy.stats import wasserstein_distance

    months = np.arange(12)
    month_names = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                   "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

    # Frequência real de dias chuvosos por mês
    # Usa todos os dados disponíveis (sem filtrar por holdout)
    # data_raw não tem índice temporal aqui — usa distribuição global
    months_full = data_raw_cond_arrays["month"]

    real_wet_freq = []
    for m in months:
        month_mask = (months_full == m)
        month_subset = data_raw[month_mask]
        month_real_wet_freq = (month_subset > 0.1).mean()
        real_wet_freq.append(month_real_wet_freq)

    # ── Gera amostras por mês para cada modelo _mc ──────────────────────────
    model_month_samples: dict[str, list] = {}  # [model] = list de 12 arrays (n_per_month, S)

    for variant_name, (model, norm_mode) in mc_models.items():
        mu, std = norm_cache[norm_mode]
        monthly_samples = []
        for m in months:
            cond = model.cond_block.sample_cond(
                n_per_month,
                model._cond_probs,
                continuous_data=getattr(model, '_continuous_data', None),
            )
            cond["month"] = torch.LongTensor([m] * n_per_month)
            with torch.no_grad():
                samp = model.sample(n_per_month, cond=cond)
            if isinstance(samp, torch.Tensor):
                samp = samp.detach().cpu().numpy()
            samp_mm = np.clip(denormalize(samp, mu, std), 0, None)
            monthly_samples.append(samp_mm)
        model_month_samples[variant_name] = monthly_samples

    if not model_month_samples:
        print("[compare] Nenhum modelo _mc disponível para plot sazonal.")
        return

    # ── Plot 1: Frequência de dias chuvosos por mês ────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(model_month_samples)))

    ax.plot(months, real_wet_freq, 'o-', color='black', label='Dados Reais')
    for i, (name, monthly) in enumerate(model_month_samples.items()):
        wet_freqs = [np.mean(monthly[m] > 0.1) for m in months]
        ax.plot(months, wet_freqs, 'o-', color=colors[i], label=name, linewidth=2)

    ax.set_xticks(months)
    ax.set_xticklabels(month_names, fontsize=9)
    ax.set_ylabel("Frequência de dias chuvosos", fontsize=10)
    ax.set_title("Frequência de Dias Chuvosos por Mês — Modelos Condicionados", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "seasonal_wet_freq.png")
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico sazonal salvo: {out_path}")

    # ── Plot 2: Wasserstein médio por mês ──────────────────────────────────
    # Para cada mês m: calcula Wasserstein entre amostras geradas e dados reais globais
    fig, ax = plt.subplots(figsize=(10, 5))

    for i, (name, monthly) in enumerate(model_month_samples.items()):
        wass_by_month = []
        S = data_raw.shape[1]
        for m in months:
            samp_mm = monthly[m]
            wass_stations = []
            for s in range(S):
                w = wasserstein_distance(data_raw[:, s], samp_mm[:, s])
                wass_stations.append(w)
            wass_by_month.append(np.mean(wass_stations))
        ax.plot(months, wass_by_month, 'o-', color=colors[i], label=name, linewidth=2)

    ax.set_xticks(months)
    ax.set_xticklabels(month_names, fontsize=9)
    ax.set_ylabel("Wasserstein médio (mm/dia)", fontsize=10)
    ax.set_title("Wasserstein Médio por Mês — Modelos Condicionados", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "seasonal_wasserstein.png")
    fig.savefig(out_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"Gráfico sazonal salvo: {out_path}")

    # ── Per-station: one image per station, 12 month subplots (3×4) ──────────
    S = data_raw.shape[1]
    st_labels = station_names if station_names and len(station_names) == S else [f"Est. {s}" for s in range(S)]

    # Precompute real masks once
    month_masks = [(data_raw_cond_arrays["month"] == m) for m in months]

    model_list = list(model_month_samples.keys())
    n_models = len(model_list)
    bar_colors_models = [colors[i] for i in range(n_models)]

    wf_dir = os.path.join(out_dir, "seasonal_wet_freq_per_station")
    ws_dir = os.path.join(out_dir, "seasonal_wasserstein_per_station")
    os.makedirs(wf_dir, exist_ok=True)
    os.makedirs(ws_dir, exist_ok=True)

    for s in range(S):
        st_name = st_labels[s]
        safe_name = st_name.replace("/", "_").replace(" ", "_")

        # ── Wet freq: one line chart per station ─────────────────────────────
        fig_wf, ax_wf = plt.subplots(figsize=(10, 4))
        real_wf = [float((data_raw[month_masks[m]][:, s] > 0.1).mean()) for m in months]
        ax_wf.plot(months, real_wf, "o-", color="black", label="Real", linewidth=2)
        for i, name in enumerate(model_list):
            wf = [float(np.mean(model_month_samples[name][m][:, s] > 0.1)) for m in months]
            ax_wf.plot(months, wf, "o-", color=bar_colors_models[i], label=name, linewidth=2)
        ax_wf.set_xticks(months)
        ax_wf.set_xticklabels(month_names, fontsize=9)
        ax_wf.set_ylabel("Frequência de dias chuvosos", fontsize=10)
        ax_wf.set_title(f"Frequência de Dias Chuvosos — {st_name}", fontsize=11, fontweight="bold")
        ax_wf.legend(fontsize=9, frameon=True)
        ax_wf.grid(axis="y", alpha=0.3)
        fig_wf.tight_layout()
        out_wf = os.path.join(wf_dir, f"{safe_name}.png")
        fig_wf.savefig(out_wf, dpi=120, bbox_inches="tight")
        plt.close(fig_wf)

        # ── Wasserstein: one line chart per station ──────────────────────────
        fig_ws, ax_ws = plt.subplots(figsize=(10, 4))
        for i, name in enumerate(model_list):
            wass = [wasserstein_distance(data_raw[:, s], model_month_samples[name][m][:, s])
                    for m in months]
            ax_ws.plot(months, wass, "o-", color=bar_colors_models[i], label=name, linewidth=2)
        ax_ws.set_xticks(months)
        ax_ws.set_xticklabels(month_names, fontsize=9)
        ax_ws.set_ylabel("Wasserstein (mm/dia)", fontsize=10)
        ax_ws.set_title(f"Wasserstein por Mês — {st_name}", fontsize=11, fontweight="bold")
        ax_ws.legend(fontsize=9, loc="best", frameon=True)
        ax_ws.grid(axis="y", alpha=0.3)
        fig_ws.tight_layout()
        out_ws = os.path.join(ws_dir, f"{safe_name}.png")
        fig_ws.savefig(out_ws, dpi=120, bbox_inches="tight")
        plt.close(fig_ws)

    print(f"Gráficos sazonais por estação salvos em: {wf_dir}/")
    print(f"Gráficos Wasserstein por estação salvos em: {ws_dir}/")


def plot_monthly_metrics_comparison(
    mc_models: dict,
    data_raw: np.ndarray,
    norm_cache: dict,
    out_dir: str,
    data_raw_cond_arrays: dict,
    n_per_month: int = 500,
):
    """
    Gera comparação rica de métricas mensais para modelos condicionados (_mc):
      - monthly_metrics_heatmap.png: heatmap por métrica (rows=models, cols=months)
      - monthly_composite_heatmap.png: heatmap composto normalizado
      - monthly_metrics_lines.png: line charts por métrica (1 linha por modelo)

    Métricas calculadas por mês (média across estações):
      - Wasserstein
      - Wet day frequency error
      - Q90, Q95, Q99 absolute error
    """
    from scipy.stats import wasserstein_distance

    months = np.arange(12)
    month_names = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                   "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    months_full = data_raw_cond_arrays["month"]
    S = data_raw.shape[1]

    # ── Compute real monthly stats ──────────────────────────────────────────
    real_wet_freq_by_month = []
    real_q90_by_month = []
    real_q95_by_month = []
    real_q99_by_month = []
    for m in months:
        mask = (months_full == m)
        subset = data_raw[mask]  # (days_m, S)
        real_wet_freq_by_month.append((subset > 0.1).mean())
        real_q90_by_month.append(np.quantile(subset, 0.90, axis=0))  # (S,)
        real_q95_by_month.append(np.quantile(subset, 0.95, axis=0))
        real_q99_by_month.append(np.quantile(subset, 0.99, axis=0))

    # ── Generate monthly samples for each model ────────────────────────────
    model_month_samples: dict[str, list] = {}
    for variant_name, (model, norm_mode) in mc_models.items():
        mu, std = norm_cache[norm_mode]
        monthly = []
        for m in months:
            cond = model.cond_block.sample_cond(
                n_per_month,
                model._cond_probs,
                continuous_data=getattr(model, '_continuous_data', None),
            )
            cond["month"] = torch.LongTensor([m] * n_per_month)
            with torch.no_grad():
                samp = model.sample(n_per_month, cond=cond)
            if isinstance(samp, torch.Tensor):
                samp = samp.detach().cpu().numpy()
            samp_mm = np.clip(denormalize(samp, mu, std), 0, None)
            monthly.append(samp_mm)
        model_month_samples[variant_name] = monthly

    if not model_month_samples:
        print("[compare] Nenhum modelo _mc disponível para plot mensal.")
        return

    model_names = list(model_month_samples.keys())
    n_models = len(model_names)
    metric_keys = ["Wasserstein", "Wet Freq Err", "Q90 Err", "Q95 Err", "Q99 Err"]

    # ── Compute per-model per-month metrics ────────────────────────────────
    # Shape: (n_models, 12) for each metric
    all_wass    = np.zeros((n_models, 12))
    all_wetfreq = np.zeros((n_models, 12))
    all_q90     = np.zeros((n_models, 12))
    all_q95     = np.zeros((n_models, 12))
    all_q99     = np.zeros((n_models, 12))

    for i, name in enumerate(model_names):
        monthly = model_month_samples[name]
        for m in months:
            samp = monthly[m]  # (n_per_month, S)
            # Wasserstein: mean across stations
            mask = (months_full == m)
            real_m = data_raw[mask]
            wass_vals = [wasserstein_distance(real_m[:, s], samp[:, s]) for s in range(S)]
            all_wass[i, m] = np.mean(wass_vals)
            # Wet freq error
            all_wetfreq[i, m] = abs(np.mean(samp > 0.1) - real_wet_freq_by_month[m])
            # Quantile errors (mean across stations)
            all_q90[i, m] = np.mean(np.abs(np.quantile(samp, 0.90, axis=0) - real_q90_by_month[m]))
            all_q95[i, m] = np.mean(np.abs(np.quantile(samp, 0.95, axis=0) - real_q95_by_month[m]))
            all_q99[i, m] = np.mean(np.abs(np.quantile(samp, 0.99, axis=0) - real_q99_by_month[m]))

    metrics_arrays = [all_wass, all_wetfreq, all_q90, all_q95, all_q99]

    # ── Plot 1: Heatmap por métrica ────────────────────────────────────────
    fig, axes = plt.subplots(len(metric_keys), 1, figsize=(12, 3 * len(metric_keys)))
    if len(metric_keys) == 1:
        axes = [axes]

    for ax, key, arr in zip(axes, metric_keys, metrics_arrays):
        im = ax.imshow(arr, aspect="auto", cmap="RdYlGn_r")
        ax.set_xticks(np.arange(12))
        ax.set_xticklabels(month_names, fontsize=8)
        ax.set_yticks(np.arange(n_models))
        ax.set_yticklabels(model_names, fontsize=8)
        ax.set_title(key, fontsize=10)
        plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
        for row in range(n_models):
            for col in range(12):
                ax.text(col, row, f"{arr[row, col]:.2f}", ha="center", va="center",
                        fontsize=6, color="black")

    fig.suptitle("Monthly Metrics — Conditional Models", fontsize=13, y=1.01)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "monthly_metrics_heatmap.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap mensal salvo: {out_path}")

    # ── Plot 2: Composite heatmap ──────────────────────────────────────────
    # Stack all metrics into (n_models, 12, n_metrics) and min-max normalize globally
    stacked = np.stack(metrics_arrays, axis=-1)  # (n_models, 12, 5)
    vmin, vmax = stacked.min(), stacked.max()
    if vmax > vmin:
        composite = (stacked - vmin) / (vmax - vmin)
    else:
        composite = np.zeros_like(stacked)
    composite_score = composite.mean(axis=-1)  # (n_models, 12)

    fig, ax = plt.subplots(figsize=(12, max(3, n_models * 0.8 + 1)))
    im = ax.imshow(composite_score, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=1)
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(month_names, fontsize=9)
    ax.set_yticks(np.arange(n_models))
    ax.set_yticklabels(model_names, fontsize=9)
    ax.set_title("Composite Monthly Score (min-max normalizado, menor = melhor)", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    for row in range(n_models):
        for col in range(12):
            ax.text(col, row, f"{composite_score[row, col]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")
    fig.tight_layout()
    out_path = os.path.join(out_dir, "monthly_composite_heatmap.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap composto mensal salvo: {out_path}")

    # ── Plot 3: Line charts per metric ────────────────────────────────────
    colors = plt.cm.tab10(np.linspace(0, 1, n_models))
    fig, axes = plt.subplots(len(metric_keys), 1, figsize=(11, 3 * len(metric_keys)), sharex=True)
    if len(metric_keys) == 1:
        axes = [axes]

    for ax, key, arr in zip(axes, metric_keys, metrics_arrays):
        for i, name in enumerate(model_names):
            ax.plot(months, arr[i], "o-", color=colors[i], label=name, linewidth=2, markersize=5)
        ax.set_xticks(months)
        ax.set_xticklabels(month_names, fontsize=8)
        ax.set_ylabel(key, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Monthly Metrics by Model", fontsize=13)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "monthly_metrics_lines.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Line chart mensal salvo: {out_path}")


class _VariantModel:
    """
    Wrapper leve para testar um modelo com parâmetros de amostragem específicos.

    Compatível com evaluate_model() de metrics.py: expõe sample() e count_parameters().
    """

    def __init__(self, model, steps: int | None, method: str | None):
        self._model = model
        self._steps = steps
        self._method = method

    def sample(self, n: int):
        return self._model.sample(n, steps=self._steps, method=self._method)

    def count_parameters(self) -> int:
        return self._model.count_parameters()


def _evaluate_solver_grid(
    grid: list,
    models_filter: list,
    output_dir: str,
    data_path: str,
    n_samples: int,
) -> tuple:
    """
    Carrega modelos, avalia métricas e gera amostras para cada variante do SOLVER_GRID.

    Returns:
        (all_metrics, generated_by_model, data_raw, station_names)
        onde as chaves são f"{model_name}_{tag}" por variante.
    """
    def _silent_load(mode):
        with contextlib.redirect_stdout(io.StringIO()):
            return load_data(data_path=data_path, normalization_mode=mode)

    _, data_raw, _, _, station_names = _silent_load("scale_only")
    norm_cache = {}
    for mode in ("scale_only", "standardize"):
        _, _, mu, std, _ = _silent_load(mode)
        norm_cache[mode] = (mu, std)

    device = torch.device("cpu")

    # Carrega cada modelo único uma só vez
    loaded_models: dict = {}
    norm_modes: dict = {}
    for (model_name, _, _, _) in grid:
        if model_name in models_filter and model_name not in loaded_models:
            try:
                cfg = _load_model_config(model_name, output_dir)
                model_class = cfg.get("model", model_name)
                norm_modes[model_name] = cfg.get("normalization_mode", "scale_only")
                model = _load_trained_model(model_name, output_dir, len(station_names), device)

                # Configurações especiais pós-carregamento
                if model_class == "latent_flow":
                    _, _, _, std_sc, _ = _silent_load("scale_only")
                    model.fit_flow(data_raw, std_scale=std_sc)
                if model_class == "hurdle_temporal":
                    _, _, _, std_sc, _ = _silent_load("scale_only")
                    model.fit_temporal(data_raw / std_sc)

                loaded_models[model_name] = model
                print(f"[SolverGrid] Modelo carregado: {model_name}")
            except Exception as e:
                print(f"[AVISO] Falha ao carregar '{model_name}': {e}")

    all_metrics: dict = {}
    generated_by_model: dict = {}

    for (model_name, steps, method, tag) in grid:
        if model_name not in models_filter:
            continue
        if model_name not in loaded_models:
            continue

        variant_key = f"{model_name}_{tag}"
        model = loaded_models[model_name]
        norm_mode = norm_modes[model_name]
        mu, std = norm_cache[norm_mode]

        print(f"\n[SolverGrid] Avaliando: {variant_key}  (steps={steps}, method={method})")

        # Avalia métricas via wrapper
        variant = _VariantModel(model, steps, method)
        samples_are_norm = (model_name != "copula")
        try:
            metrics = evaluate_model(
                variant, data_raw, mu, std,
                n_samples=n_samples,
                station_names=list(station_names),
                samples_are_normalized=samples_are_norm,
            )
            all_metrics[variant_key] = metrics
        except Exception as e:
            print(f"[AVISO] Falha ao avaliar '{variant_key}': {e}")
            continue

        # Gera amostras para gráficos visuais
        try:
            generated_by_model[variant_key] = _sample_generated_mm(
                model_name=model_name,
                model=model,
                n_samples=n_samples,
                norm_mode=norm_mode,
                norm_cache=norm_cache,
                steps=steps,
                method=method,
            )
        except Exception as e:
            print(f"[AVISO] Falha ao gerar amostras para '{variant_key}': {e}")

    return all_metrics, generated_by_model, data_raw, list(station_names)


def generate_visual_comparisons(
    models: list,
    output_dir: str,
    data_path: str,
    out_dir: str,
    all_metrics: dict,
    n_samples: int,
    max_points_per_series: int,
):
    """Gera comparações visuais de amostras por estação e correlação geral."""
    def _silent_load_data(norm_mode: str):
        with contextlib.redirect_stdout(io.StringIO()):
            return load_data(data_path=data_path, normalization_mode=norm_mode)

    # Usa scale_only como referência para obter data_raw e nomes de estação.
    _, data_raw, _, _, station_names = _silent_load_data("scale_only")

    # Cache de normalização para converter amostras dos modelos neurais.
    norm_cache = {}
    for mode in ("scale_only", "standardize"):
        _, _, mu, std, _ = _silent_load_data(mode)
        norm_cache[mode] = (mu, std)

    device = torch.device("cpu")
    generated_by_model = {}
    for variant_name in models:
        if variant_name not in all_metrics:
            continue
        try:
            cfg = _load_model_config(variant_name, output_dir)
            model_class = cfg.get("model", variant_name)
            norm_mode = cfg.get("normalization_mode", "scale_only")
            model = _load_trained_model(variant_name, output_dir, len(station_names), device)

            # latent_flow precisa de fit_flow() para restaurar transform interno
            if model_class == "latent_flow":
                _, _, mu_sc, std_sc, _ = _silent_load_data("scale_only")
                model.fit_flow(data_raw, std_scale=std_sc)

            # hurdle_temporal precisa de fit_temporal() para janelas de contexto
            if model_class == "hurdle_temporal":
                _, _, mu_sc, std_sc, _ = _silent_load_data("scale_only")
                train_norm_cmp = data_raw / std_sc  # scale_only: mu=0
                model.fit_temporal(train_norm_cmp)

            generated_by_model[variant_name] = _sample_generated_mm(
                model_name=model_class,
                model=model,
                n_samples=n_samples,
                norm_mode=norm_mode,
                norm_cache=norm_cache,
            )
        except Exception as e:
            print(f"[AVISO] Falha ao carregar/gerar amostras para '{variant_name}': {e}")

    if not generated_by_model:
        print("[AVISO] Nenhuma amostra gerada para os gráficos de comparação.")
        return

    plot_auxplot_scenarios(
        real_data=data_raw,
        generated_by_model=generated_by_model,
        out_dir=out_dir,
    )
    plot_station_samples_comparison(
        real_data=data_raw,
        generated_by_model=generated_by_model,
        station_names=station_names,
        out_dir=out_dir,
        max_points_per_series=max_points_per_series,
    )
    plot_overall_correlation_chart(
        real_data=data_raw,
        generated_by_model=generated_by_model,
        out_dir=out_dir,
        all_metrics=all_metrics,
        max_points_per_series=max_points_per_series,
    )
    # QQ-plots por estação (novo)
    try:
        plot_qqplots_by_station(
            real_data=data_raw,
            generated_by_model=generated_by_model,
            station_names=station_names,
            out_dir=out_dir,
        )
    except Exception as e:
        print(f"[AVISO] Falha nos QQ-plots: {e}")

    # Heatmaps por estação (wet freq + wasserstein)
    try:
        plot_station_wetfreq_heatmap(all_metrics, data_raw, station_names, out_dir)
    except Exception as e:
        print(f"[AVISO] Falha no heatmap wet freq por estação: {e}")
    try:
        plot_station_wasserstein_heatmap(all_metrics, station_names, out_dir)
    except Exception as e:
        print(f"[AVISO] Falha no heatmap Wasserstein por estação: {e}")


_WET_DAY_THRESHOLD_MM = 0.1


def compute_station_reference_wetfreq(data_raw: np.ndarray, station_names: list) -> dict:
    """Retorna frequência de dias chuvosos reais por estação."""
    return {st: float((data_raw[:, i] > _WET_DAY_THRESHOLD_MM).mean())
            for i, st in enumerate(station_names)}


def plot_station_wetfreq_heatmap(all_metrics: dict, data_raw: np.ndarray,
                                  station_names: list, out_dir: str) -> None:
    """Heatmap: linhas = modelos, colunas = estações, valores = wet_day_freq_error_per_station.
    Linha extra no topo com frequência real de referência."""
    model_names = sorted(all_metrics.keys())
    ref = compute_station_reference_wetfreq(data_raw, station_names)
    n_models = len(model_names)
    n_stations = len(station_names)

    # Matriz de erros (n_models x n_stations)
    err_matrix = np.full((n_models, n_stations), np.nan)
    for i, m in enumerate(model_names):
        per_st = all_metrics[m].get("wet_day_freq_error_per_station", {})
        for j, st in enumerate(station_names):
            err_matrix[i, j] = per_st.get(st, np.nan)

    ref_row = np.array([ref.get(st, np.nan) for st in station_names])

    fig, axes = plt.subplots(2, 1, figsize=(max(10, n_stations * 0.8), 3 + n_models * 0.5),
                             gridspec_kw={"height_ratios": [1, n_models]})

    # Linha de referência
    ax_ref = axes[0]
    im_ref = ax_ref.imshow(ref_row[np.newaxis, :], aspect="auto", cmap="Blues",
                            vmin=0, vmax=1)
    ax_ref.set_xticks(np.arange(n_stations))
    ax_ref.set_xticklabels(station_names, rotation=45, ha="right", fontsize=7)
    ax_ref.set_yticks([0])
    ax_ref.set_yticklabels(["Real (ref)"], fontsize=8)
    plt.colorbar(im_ref, ax=ax_ref, fraction=0.02, pad=0.02)
    for j, v in enumerate(ref_row):
        if not np.isnan(v):
            ax_ref.text(j, 0, f"{v:.2f}", ha="center", va="center", fontsize=6)

    # Matriz de erros
    ax_err = axes[1]
    vmax_err = np.nanmax(np.abs(err_matrix)) if not np.all(np.isnan(err_matrix)) else 1.0
    im_err = ax_err.imshow(err_matrix, aspect="auto", cmap="RdYlGn_r",
                            vmin=0, vmax=vmax_err)
    ax_err.set_xticks(np.arange(n_stations))
    ax_err.set_xticklabels(station_names, rotation=45, ha="right", fontsize=7)
    ax_err.set_yticks(np.arange(n_models))
    ax_err.set_yticklabels(model_names, fontsize=8)
    plt.colorbar(im_err, ax=ax_err, fraction=0.02, pad=0.02)
    for i in range(n_models):
        for j in range(n_stations):
            v = err_matrix[i, j]
            if not np.isnan(v):
                ax_err.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)

    fig.suptitle("Wet Day Freq Error per Station (models) + Reference (top)", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "station_wetfreq_heatmap.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap wet freq por estação salvo: {out_path}")


def plot_station_wasserstein_heatmap(all_metrics: dict, station_names: list,
                                      out_dir: str) -> None:
    """Heatmap: linhas = modelos, colunas = estações, valores = wasserstein_per_station."""
    model_names = sorted(all_metrics.keys())
    n_models = len(model_names)
    n_stations = len(station_names)

    matrix = np.full((n_models, n_stations), np.nan)
    for i, m in enumerate(model_names):
        per_st = all_metrics[m].get("wasserstein_per_station", {})
        for j, st in enumerate(station_names):
            matrix[i, j] = per_st.get(st, np.nan)

    fig, ax = plt.subplots(figsize=(max(10, n_stations * 0.8), 2 + n_models * 0.5))
    vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 1.0
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn_r", vmin=0, vmax=vmax)
    ax.set_xticks(np.arange(n_stations))
    ax.set_xticklabels(station_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(np.arange(n_models))
    ax.set_yticklabels(model_names, fontsize=8)
    ax.set_title("Wasserstein Distance per Station", fontsize=11)
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    for i in range(n_models):
        for j in range(n_stations):
            v = matrix[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6)

    fig.tight_layout()
    out_path = os.path.join(out_dir, "station_wasserstein_heatmap.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Heatmap Wasserstein por estação salvo: {out_path}")


# ──────────────────────────────────────────────────────────
# RELATÓRIO TEXTO
# ──────────────────────────────────────────────────────────

def print_report(all_metrics: dict, scores: dict, data_raw: np.ndarray,
                 station_names: list = None) -> str:
    """Imprime e retorna tabela texto com ranking."""
    model_names = sorted(scores.keys(), key=lambda m: scores.get(m, float('inf')))
    col = 18
    lines = []

    lines.append("\n" + "=" * 80)
    lines.append("  RELATÓRIO COMPARATIVO — MODELOS GENERATIVOS PARA PRECIPITAÇÃO")
    lines.append("  (ordenado por composite score: menor = melhor)")
    lines.append("=" * 80)

    # ref_wet_day_freq = 

    # Cabeçalho
    header = f"{'Modelo':<{col}}"
    for _, label, _ in QUALITY_METRICS:
        header += f"  {label[:12]:>12}"
    header += f"  {'Composite':>10}  {'Cov90%':>7}  {'Params':>9}  {'T_samp(ms)':>10}"
    lines.append(header)
    lines.append("-" * len(header))

    for rank, m in enumerate(model_names, 1):
        mt = all_metrics.get(m, {})
        row = f"{m:<{col}}"
        for key, _, _ in QUALITY_METRICS:
            v = mt.get(key, float('nan'))
            row += f"  {v:>12.4f}" if not np.isnan(v) else f"  {'N/A':>12}"
        row += f"  {scores.get(m, float('nan')):>10.4f}"
        cov90 = mt.get('coverage_90', float('nan'))
        row += f"  {cov90:>7.3f}" if not np.isnan(cov90) else f"  {'N/A':>7}"
        row += f"  {mt.get('n_parameters', 0):>9,}"
        row += f"  {mt.get('sampling_time_ms', float('nan')):>10.1f}"
        lines.append(f"#{rank}  {row}")

    lines.append("=" * 80)

    # Vencedor por métrica
    lines.append("\nVENCEDOR POR MÉTRICA:")
    for key, label, lower_is_better in QUALITY_METRICS:
        vals = {m: all_metrics.get(m, {}).get(key, float('nan')) for m in model_names}
        valid = {m: v for m, v in vals.items() if not np.isnan(v)}
        if valid:
            winner = min(valid, key=valid.get) if lower_is_better else max(valid, key=valid.get)
            lines.append(f"  {label:<28}: {winner} ({valid[winner]:.4f})")

    lines.append(f"\nRANKING GERAL (composite score):")
    for rank, m in enumerate(model_names, 1):
        lines.append(f"  #{rank} {m}: {scores.get(m, float('nan')):.4f}")

    # Métricas temporais (informativas — não afetam composite score)
    temporal_metrics = [
        ("wet_spell_length_error", "Wet spell length err"),
        ("dry_spell_length_error", "Dry spell length err"),
        ("lag1_autocorr_error",    "Lag-1 autocorr err"),
    ]
    has_temporal = any(
        any(all_metrics.get(m, {}).get(k) is not None for m in model_names)
        for k, _ in temporal_metrics
    )
    if has_temporal:
        lines.append("\nMÉTRICAS TEMPORAIS (informativas — fora do composite score):")
        for key, label in temporal_metrics:
            vals = {m: all_metrics.get(m, {}).get(key, float('nan')) for m in model_names}
            valid = {m: v for m, v in vals.items() if isinstance(v, float) and not np.isnan(v)}
            if valid:
                winner = min(valid, key=valid.get)
                parts = "  ".join(f"{m}={valid[m]:.3f}" for m in model_names if m in valid)
                lines.append(f"  {label:<28}: {parts}  → best: {winner}")

    # Per-station wet day freq table
    if station_names is not None and data_raw is not None:
        ref = compute_station_reference_wetfreq(data_raw, station_names)
        col_w = max(10, max(len(s) for s in station_names) + 1)
        lines.append("\nFREQUÊNCIA DE DIAS CHUVOSOS POR ESTAÇÃO:")
        lines.append("  (ref = frequência real; modelos mostram erro absoluto)")
        header_st = f"  {'Modelo':<18}" + "".join(f"  {st[:col_w]:>{col_w}}" for st in station_names)
        lines.append(header_st)
        lines.append("  " + "-" * (len(header_st) - 2))
        ref_row_str = f"  {'Real (ref)':<18}" + "".join(
            f"  {ref.get(st, float('nan')):>{col_w}.3f}" for st in station_names)
        lines.append(ref_row_str)
        for m in model_names:
            per_st = all_metrics.get(m, {}).get("wet_day_freq_error_per_station", {})
            row_st = f"  {m:<18}" + "".join(
                f"  {per_st.get(st, float('nan')):>{col_w}.3f}" if not np.isnan(per_st.get(st, float('nan')))
                else f"  {'N/A':>{col_w}}"
                for st in station_names)
            lines.append(row_st)

    report = "\n".join(lines)
    print(report)
    return report


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compara todos os modelos generativos para precipitação.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--skip_training", action="store_true",
                        help="Pula treino e usa métricas já salvas")
    parser.add_argument("--max_epochs", type=int, default=None,
                        help="Épocas máximas (sobrescreve defaults; útil para teste rápido)")
    parser.add_argument("--n_samples", type=int, default=5000,
                        help="Amostras para avaliação")
    parser.add_argument("--chart_samples", type=int, default=3000,
                        help="Amostras sintéticas para gráficos de comparação")
    parser.add_argument("--chart_max_points", type=int, default=1000,
                        help="Máximo de pontos por série nos boxplots/heatmaps")
    parser.add_argument("--models", nargs="+", default=ALL_MODELS,
                        # choices=ALL_MODELS,
                        help="Nomes de variantes a comparar (diretórios em outputs/). "
                             "Aceita tanto nomes de modelos padrão quanto variantes nomeadas (--name).")
    parser.add_argument("--solver_grid", action="store_true",
                        help="Compara variantes de solver/steps do SOLVER_GRID (sem re-treinar)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Diretório base de saída")
    parser.add_argument("--data_path", type=str, default="../dados/inmet_relevant_data.csv",
                        help="Caminho para inmet_relevant_data.csv")

    args = parser.parse_args()

    out_dir = os.path.join(args.output_dir, "comparison")
    os.makedirs(out_dir, exist_ok=True)
    
    def _silent_load(mode):
        with contextlib.redirect_stdout(io.StringIO()):
            return load_data_with_cond(data_path=args.data_path, normalization_mode=mode)
    
    _, data_raw_cmp, _, _, station_names_cmp, data_raw_cond_arrays = _silent_load("scale_only")

    # ── Modo solver_grid: avalia variantes de solver/steps sem re-treinar ──────
    if args.solver_grid:
        print("[compare] Modo --solver_grid: comparando variantes de solver/steps.")
        all_metrics, generated_by_model, data_raw, station_names = _evaluate_solver_grid(
            grid=SOLVER_GRID,
            models_filter=args.models,
            output_dir=args.output_dir,
            data_path=args.data_path,
            n_samples=args.n_samples,
        )
        if not all_metrics:
            print("Nenhuma variante avaliada. Verifique se os modelos foram treinados.")
            return

        scores, normalized = compute_composite(all_metrics)
        report = print_report(all_metrics, scores, data_raw_cmp, list(station_names_cmp))
        with open(os.path.join(out_dir, "comparison_report_solver_grid.txt"), "w", encoding="utf-8") as f:
            f.write(report + "\n")
        with open(os.path.join(out_dir, "composite_scores_solver_grid.json"), "w") as f:
            json.dump(scores, f, indent=2)

        try:
            plot_bar_charts(all_metrics, scores, out_dir)
            plot_radar(all_metrics, normalized, out_dir)
            if generated_by_model:
                plot_station_samples_comparison(
                    real_data=data_raw,
                    generated_by_model=generated_by_model,
                    station_names=station_names,
                    out_dir=out_dir,
                    max_points_per_series=args.chart_max_points,
                )
                plot_overall_correlation_chart(
                    real_data=data_raw,
                    generated_by_model=generated_by_model,
                    out_dir=out_dir,
                    all_metrics=all_metrics,
                    max_points_per_series=args.chart_max_points,
                )
        except Exception as e:
            print(f"[AVISO] Erro ao gerar gráficos: {e}")

        print(f"\nSolver grid completo! Resultados em: {out_dir}/")
        return

    # ── Modo normal ────────────────────────────────────────────────────────────

    # 1. Treino (se necessário)
    if not args.skip_training:
        train_all(args.models, args.max_epochs, args.n_samples, args.output_dir, args.data_path)
    else:
        print("[compare] Pulando treino — carregando métricas existentes.")

    # 2. Carrega métricas
    all_metrics = load_all_metrics(args.models, args.output_dir)
    if not all_metrics:
        print("Nenhuma métrica encontrada. Execute sem --skip_training primeiro.")
        return

    # 3. Composite score
    scores, normalized = compute_composite(all_metrics)

    # 4. Relatório texto
    report = print_report(all_metrics, scores, data_raw_cmp)
    with open(os.path.join(out_dir, "comparison_report.txt"), "w", encoding="utf-8") as f:
        f.write(report + "\n")

    # 5. Salva composite scores
    with open(os.path.join(out_dir, "composite_scores.json"), "w") as f:
        json.dump(scores, f, indent=2)

    # 7. Análise sazonal para modelos _mc
    mc_model_names = [m for m in args.models if ("_mc" in m) and m in all_metrics]
    if mc_model_names:
        try:
            
            norm_cache_cmp = {}
            for mode in ("scale_only", "standardize"):
                _, _, mu_m, std_m, _, _ = _silent_load(mode)
                norm_cache_cmp[mode] = (mu_m, std_m)

            device_cmp = torch.device("cpu")
            mc_models_dict = {}
            for variant_name in mc_model_names:
                try:
                    cfg = _load_model_config(variant_name, args.output_dir)
                    norm_mode = cfg.get("normalization_mode", "scale_only")
                    model = _load_trained_model(
                        variant_name, args.output_dir, len(station_names_cmp), device_cmp
                    )
                    mc_models_dict[variant_name] = (model, norm_mode)
                except Exception as e:
                    print(f"[AVISO] Falha ao carregar '{variant_name}' para plot sazonal: {e}")

            if mc_models_dict:
                plot_seasonal_metrics(
                    mc_models=mc_models_dict,
                    data_raw=data_raw_cmp,
                    norm_cache=norm_cache_cmp,
                    out_dir=out_dir,
                    data_raw_cond_arrays=data_raw_cond_arrays,
                    station_names=list(station_names_cmp),
                )
                plot_monthly_metrics_comparison(
                    mc_models=mc_models_dict,
                    data_raw=data_raw_cmp,
                    norm_cache=norm_cache_cmp,
                    out_dir=out_dir,
                    data_raw_cond_arrays=data_raw_cond_arrays,
                    n_per_month=500,
                )
        except Exception as e:
            print(f"[AVISO] Erro ao gerar análise sazonal: {e}")

    # 6. Gráficos
    try:
        plot_bar_charts(all_metrics, scores, out_dir)
        plot_radar(all_metrics, normalized, out_dir)
        generate_visual_comparisons(
            models=args.models,
            output_dir=args.output_dir,
            data_path=args.data_path,
            out_dir=out_dir,
            all_metrics=all_metrics,
            n_samples=args.chart_samples,
            max_points_per_series=args.chart_max_points,
        )
    except Exception as e:
        print(f"[AVISO] Erro ao gerar gráficos: {e}")

    

    print(f"\nComparação completa! Resultados em: {out_dir}/")


if __name__ == "__main__":
    main()
