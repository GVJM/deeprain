"""
plot_model.py — Análise detalhada de um único modelo generativo de precipitação.

Uso:
    python plot_model.py --model vae_mc
    python plot_model.py --model flow_match_mc --n_samples 2000 --output_dir ./outputs
    python plot_model.py --model hurdle_simple --no_monthly

Para modelos não condicionados (_mc), --no_monthly é forçado automaticamente.

Saídas em outputs/<model>/analysis/:
    monthly_kde_by_station.png       — KDE por estação × mês
    monthly_exceedance_by_station.png — P(X > x) por estação × mês
    monthly_wet_freq.png             — frequência de dias chuvosos por mês
    monthly_quantile_errors.png      — erros Q90/Q95/Q99 por mês
    monthly_metrics_radar.png        — radar chart por mês
    joint_distribution.png           — pairwise scatter (real vs gerado)
    marginal_distributions.png       — histograma/KDE por estação
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch

# ── Adiciona PrecipModels ao path para imports locais ────────────────────────
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from data_utils import load_data, load_data_with_cond, denormalize
from compare import _load_trained_model, _load_model_config


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _sample_model_mm(model, norm_mode: str, norm_cache: dict, n: int) -> np.ndarray:
    """Sample n rows from model (no conditioning) and denormalize to mm/day."""
    with torch.no_grad():
        samp = model.sample(n)
    if isinstance(samp, torch.Tensor):
        samp = samp.detach().cpu().numpy()
    mu, std = norm_cache[norm_mode]
    return np.clip(denormalize(samp, mu, std), 0, None)


def _sample_model_mm_cond(model, norm_mode: str, norm_cache: dict, n: int, month: int) -> np.ndarray:
    """Sample n rows conditioned on a single month (0-11)."""
    cond = {"month": torch.LongTensor([month] * n)}
    with torch.no_grad():
        samp = model.sample(n, cond=cond)
    if isinstance(samp, torch.Tensor):
        samp = samp.detach().cpu().numpy()
    mu, std = norm_cache[norm_mode]
    return np.clip(denormalize(samp, mu, std), 0, None)


def _exceedance(values: np.ndarray):
    """Returns (sorted_x, P(X > x)) for empirical exceedance curve."""
    x = np.sort(values)
    n = len(x)
    p_exceed = 1.0 - np.arange(1, n + 1) / n
    return x, p_exceed


# ─────────────────────────────────────────────────────────────────────────────
# Monthly plots (only for _mc models)
# ─────────────────────────────────────────────────────────────────────────────

def plot_monthly_kde_by_station(
    model, norm_mode, norm_cache, data_raw, months_full, station_names, n_samples, out_dir
):
    """KDE per station per month — 12 colored lines, one panel per station."""
    from scipy.stats import gaussian_kde

    months = np.arange(12)
    month_names = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                   "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    S = len(station_names)
    colors = cm.tab20(np.linspace(0, 1, 12))

    ncols = min(4, S)
    nrows = int(np.ceil(S / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)

    for s_idx in range(S):
        ax = axes[s_idx // ncols][s_idx % ncols]
        for m in months:
            samp = _sample_model_mm_cond(model, norm_mode, norm_cache, n_samples, m)
            vals = samp[:, s_idx]
            # Skip if all zeros
            if vals.max() < 1e-6:
                continue
            try:
                kde = gaussian_kde(vals, bw_method="scott")
                x_grid = np.linspace(0, np.percentile(vals, 99), 200)
                ax.plot(x_grid, kde(x_grid), color=colors[m], label=month_names[m], linewidth=1.2)
            except Exception:
                pass
        ax.set_title(station_names[s_idx], fontsize=8)
        ax.set_xlabel("mm/dia", fontsize=7)
        ax.set_ylabel("densidade", fontsize=7)
        ax.tick_params(labelsize=6)

    # Hide empty subplots
    for idx in range(S, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    # Single shared legend
    handles = [plt.Line2D([0], [0], color=colors[m], linewidth=1.5, label=month_names[m])
               for m in months]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("KDE por Estação × Mês (gerado)", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = os.path.join(out_dir, "monthly_kde_by_station.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"KDE mensal salvo: {out_path}")


def plot_monthly_exceedance_by_station(
    model, norm_mode, norm_cache, data_raw, months_full, station_names, n_samples, out_dir
):
    """P(X > x) per station per month."""
    months = np.arange(12)
    month_names = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                   "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    S = len(station_names)
    colors = cm.tab20(np.linspace(0, 1, 12))

    ncols = min(4, S)
    nrows = int(np.ceil(S / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)

    for s_idx in range(S):
        ax = axes[s_idx // ncols][s_idx % ncols]
        for m in months:
            samp = _sample_model_mm_cond(model, norm_mode, norm_cache, n_samples, m)
            vals = samp[:, s_idx]
            x, p = _exceedance(vals)
            ax.plot(x, p, color=colors[m], label=month_names[m], linewidth=1.2)
        ax.set_title(station_names[s_idx], fontsize=8)
        ax.set_xlabel("mm/dia", fontsize=7)
        ax.set_ylabel("P(X > x)", fontsize=7)
        ax.set_yscale("log")
        ax.tick_params(labelsize=6)

    for idx in range(S, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    handles = [plt.Line2D([0], [0], color=colors[m], linewidth=1.5, label=month_names[m])
               for m in months]
    fig.legend(handles=handles, loc="lower center", ncol=6, fontsize=7, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Exceedance P(X > x) por Estação × Mês (gerado)", fontsize=12)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    out_path = os.path.join(out_dir, "monthly_exceedance_by_station.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Exceedance mensal salvo: {out_path}")


def plot_monthly_wet_freq(
    model, norm_mode, norm_cache, data_raw, months_full, n_samples, out_dir
):
    """Bar chart: real vs model wet day frequency per month."""
    months = np.arange(12)
    month_names = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                   "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

    real_freq = []
    model_freq = []
    for m in months:
        mask = (months_full == m)
        real_freq.append((data_raw[mask] > 0.1).mean())
        samp = _sample_model_mm_cond(model, norm_mode, norm_cache, n_samples, m)
        model_freq.append((samp > 0.1).mean())

    x = np.arange(12)
    width = 0.35
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.bar(x - width / 2, real_freq, width, label="Real", color="steelblue")
    ax.bar(x + width / 2, model_freq, width, label="Modelo", color="tomato")
    ax.set_xticks(x)
    ax.set_xticklabels(month_names)
    ax.set_ylabel("Frequência dias chuvosos (>0.1 mm)")
    ax.set_title("Frequência de Dias Chuvosos por Mês")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "monthly_wet_freq.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Wet freq mensal salvo: {out_path}")


def plot_monthly_quantile_errors(
    model, norm_mode, norm_cache, data_raw, months_full, n_samples, out_dir
):
    """Q90/Q95/Q99 absolute error by month."""
    from scipy.stats import wasserstein_distance

    months = np.arange(12)
    month_names = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                   "Jul", "Ago", "Set", "Out", "Nov", "Dez"]

    errs = {q: [] for q in [0.90, 0.95, 0.99]}
    for m in months:
        mask = (months_full == m)
        real_m = data_raw[mask]
        samp = _sample_model_mm_cond(model, norm_mode, norm_cache, n_samples, m)
        for q in [0.90, 0.95, 0.99]:
            real_q = np.quantile(real_m, q, axis=0)
            model_q = np.quantile(samp, q, axis=0)
            errs[q].append(np.mean(np.abs(model_q - real_q)))

    fig, ax = plt.subplots(figsize=(11, 5))
    colors_q = {"0.90": "steelblue", "0.95": "darkorange", "0.99": "crimson"}
    for q in [0.90, 0.95, 0.99]:
        ax.plot(months, errs[q], "o-", label=f"Q{int(q*100)}", linewidth=2,
                color=colors_q[str(q)])
    ax.set_xticks(months)
    ax.set_xticklabels(month_names)
    ax.set_ylabel("Erro absoluto médio (mm/dia)")
    ax.set_title("Erros de Quantil Extremo por Mês")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "monthly_quantile_errors.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Quantile errors mensais salvos: {out_path}")


def plot_monthly_metrics_radar(
    model, norm_mode, norm_cache, data_raw, months_full, n_samples, out_dir
):
    """Radar chart with 12 month polygons across 5 metrics."""
    from scipy.stats import wasserstein_distance

    months = np.arange(12)
    month_names = ["Jan", "Fev", "Mar", "Abr", "Mai", "Jun",
                   "Jul", "Ago", "Set", "Out", "Nov", "Dez"]
    metric_labels = ["Wasserstein", "Wet Freq Err", "Q90 Err", "Q95 Err", "Q99 Err"]
    n_metrics = len(metric_labels)

    # Compute raw metric values per month
    raw = np.zeros((12, n_metrics))
    for m in months:
        mask = (months_full == m)
        real_m = data_raw[mask]
        samp = _sample_model_mm_cond(model, norm_mode, norm_cache, n_samples, m)
        S = data_raw.shape[1]
        # Wasserstein
        wass = np.mean([wasserstein_distance(real_m[:, s], samp[:, s]) for s in range(S)])
        raw[m, 0] = wass
        # Wet freq err
        raw[m, 1] = abs((samp > 0.1).mean() - (real_m > 0.1).mean())
        # Quantile errors
        for qi, q in enumerate([0.90, 0.95, 0.99]):
            raw[m, 2 + qi] = np.mean(np.abs(
                np.quantile(samp, q, axis=0) - np.quantile(real_m, q, axis=0)
            ))

    # Normalize each metric to [0, 1]
    col_min = raw.min(axis=0, keepdims=True)
    col_max = raw.max(axis=0, keepdims=True)
    denom = np.where(col_max > col_min, col_max - col_min, 1.0)
    normalized = (raw - col_min) / denom

    # Radar setup
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    colors = cm.hsv(np.linspace(0, 1, 12))
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))

    for m in months:
        values = normalized[m].tolist() + [normalized[m][0]]
        ax.plot(angles, values, color=colors[m], linewidth=1.2, label=month_names[m])
        ax.fill(angles, values, color=colors[m], alpha=0.05)

    ax.set_thetagrids(np.degrees(angles[:-1]), metric_labels, fontsize=9)
    ax.set_ylim(0, 1)
    ax.set_title("Métricas Mensais — Radar Chart (normalizado)", fontsize=12, pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=8, ncol=2)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "monthly_metrics_radar.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Radar mensal salvo: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Overall plots (all models)
# ─────────────────────────────────────────────────────────────────────────────

def plot_marginal_distributions(real_data, generated, station_names, out_dir):
    """Histogram/KDE overlay per station: real (grey) vs generated (colored)."""
    from scipy.stats import gaussian_kde

    S = len(station_names)
    ncols = min(4, S)
    nrows = int(np.ceil(S / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4, nrows * 3), squeeze=False)

    for s_idx in range(S):
        ax = axes[s_idx // ncols][s_idx % ncols]
        r = real_data[:, s_idx]
        g = generated[:, s_idx]
        x_max = max(np.percentile(r, 99), np.percentile(g, 99))
        x_grid = np.linspace(0, x_max, 300)

        # Real KDE
        try:
            kde_r = gaussian_kde(r, bw_method="scott")
            ax.fill_between(x_grid, kde_r(x_grid), alpha=0.4, color="grey", label="Real")
            ax.plot(x_grid, kde_r(x_grid), color="grey", linewidth=1)
        except Exception:
            pass

        # Generated KDE
        try:
            kde_g = gaussian_kde(g, bw_method="scott")
            ax.fill_between(x_grid, kde_g(x_grid), alpha=0.4, color="steelblue", label="Gerado")
            ax.plot(x_grid, kde_g(x_grid), color="steelblue", linewidth=1.5)
        except Exception:
            pass

        ax.set_title(station_names[s_idx], fontsize=8)
        ax.set_xlabel("mm/dia", fontsize=7)
        ax.tick_params(labelsize=6)
        if s_idx == 0:
            ax.legend(fontsize=7)

    for idx in range(S, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Distribuições Marginais: Real vs Gerado", fontsize=12)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "marginal_distributions.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Marginais salvas: {out_path}")


def plot_joint_distribution(real_data, generated, station_names, out_dir, max_stations=6):
    """Pairwise scatter matrix (real vs generated) for up to max_stations stations."""
    S_plot = min(max_stations, len(station_names))
    names_plot = station_names[:S_plot]
    n_pts = min(real_data.shape[0], generated.shape[0])

    fig, axes = plt.subplots(S_plot, S_plot, figsize=(S_plot * 2.5, S_plot * 2.5), squeeze=False)

    for i in range(S_plot):
        for j in range(S_plot):
            ax = axes[i][j]
            if i == j:
                # Diagonal: histogram overlay
                ax.hist(real_data[:n_pts, i], bins=30, alpha=0.5, color="grey", density=True, label="Real")
                ax.hist(generated[:n_pts, i], bins=30, alpha=0.5, color="steelblue", density=True, label="Gerado")
                ax.set_title(names_plot[i], fontsize=7)
            else:
                # Off-diagonal: scatter real vs generated
                ax.scatter(real_data[:n_pts, j], generated[:n_pts, i],
                           alpha=0.15, s=3, color="steelblue", rasterized=True)
                # Identity line reference
                lim = max(real_data[:n_pts, j].max(), generated[:n_pts, i].max())
                ax.plot([0, lim], [0, lim], "r--", linewidth=0.8, alpha=0.6)
            ax.tick_params(labelsize=5)
            if j == 0:
                ax.set_ylabel(names_plot[i], fontsize=6)
            if i == S_plot - 1:
                ax.set_xlabel(names_plot[j], fontsize=6)

    fig.suptitle(f"Distribuição Conjunta: Real vs Gerado (primeiras {S_plot} estações)", fontsize=11)
    fig.tight_layout()
    out_path = os.path.join(out_dir, "joint_distribution.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Distribuição conjunta salva: {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Análise detalhada de um único modelo de precipitação.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model", required=True,
                        help="Nome do modelo (diretório em output_dir/)")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Diretório base de saída")
    parser.add_argument("--data_path", type=str, default="../dados_sabesp/dayprecip.dat",
                        help="Caminho para o CSV de dados")
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Amostras por mês (ou total para modelos não condicionados)")
    parser.add_argument("--no_monthly", action="store_true",
                        help="Pula análise mensal (forçado para modelos sem _mc)")

    args = parser.parse_args()

    is_conditional = args.model.endswith("_mc")
    run_monthly = is_conditional and not args.no_monthly

    # ── Setup output dir ────────────────────────────────────────────────────
    analysis_dir = os.path.join(args.output_dir, args.model, "analysis")
    os.makedirs(analysis_dir, exist_ok=True)
    print(f"[plot_model] Modelo: {args.model}")
    print(f"[plot_model] Saídas em: {analysis_dir}")

    # ── Load data ───────────────────────────────────────────────────────────
    import contextlib, io as _io

    def _silent_load_cond(mode):
        with contextlib.redirect_stdout(_io.StringIO()):
            return load_data_with_cond(data_path=args.data_path, normalization_mode=mode)

    def _silent_load(mode):
        with contextlib.redirect_stdout(_io.StringIO()):
            return load_data(data_path=args.data_path, normalization_mode=mode)

    if run_monthly:
        _, data_raw, _, _, station_names, cond_arrays = _silent_load_cond("scale_only")
        months_full = cond_arrays["month"]
    else:
        _, data_raw, _, _, station_names = _silent_load("scale_only")
        months_full = None

    # Build norm_cache for both modes
    norm_cache = {}
    for mode in ("scale_only", "standardize"):
        if run_monthly:
            _, _, mu_m, std_m, _, _ = _silent_load_cond(mode)
        else:
            _, _, mu_m, std_m, _ = _silent_load(mode)
        norm_cache[mode] = (mu_m, std_m)

    # ── Load model ──────────────────────────────────────────────────────────
    device = torch.device("cpu")
    cfg = _load_model_config(args.model, args.output_dir)
    norm_mode = cfg.get("normalization_mode", "scale_only")
    input_size = len(station_names)

    print(f"[plot_model] Carregando modelo '{args.model}' (norm_mode={norm_mode})…")
    model = _load_trained_model(args.model, args.output_dir, input_size, device)
    model.eval()
    print("[plot_model] Modelo carregado.")

    # ── Monthly plots ───────────────────────────────────────────────────────
    if run_monthly:
        print("[plot_model] Gerando análise mensal…")

        plot_monthly_kde_by_station(
            model, norm_mode, norm_cache, data_raw, months_full,
            station_names, args.n_samples, analysis_dir
        )
        plot_monthly_exceedance_by_station(
            model, norm_mode, norm_cache, data_raw, months_full,
            station_names, args.n_samples, analysis_dir
        )
        plot_monthly_wet_freq(
            model, norm_mode, norm_cache, data_raw, months_full,
            args.n_samples, analysis_dir
        )
        plot_monthly_quantile_errors(
            model, norm_mode, norm_cache, data_raw, months_full,
            args.n_samples, analysis_dir
        )
        plot_monthly_metrics_radar(
            model, norm_mode, norm_cache, data_raw, months_full,
            args.n_samples, analysis_dir
        )
    else:
        if is_conditional:
            print("[plot_model] --no_monthly especificado: pulando análise mensal.")
        else:
            print(f"[plot_model] Modelo '{args.model}' não é _mc: pulando análise mensal.")

    # ── Overall marginal and joint distributions ────────────────────────────
    print("[plot_model] Gerando distribuições globais…")
    n_overall = min(args.n_samples * 12 if run_monthly else args.n_samples, 5000)
    generated_overall = _sample_model_mm(model, norm_mode, norm_cache, n_overall)

    # Subsample real data to same size for fair comparison
    n_real = min(data_raw.shape[0], n_overall)
    idx = np.random.choice(data_raw.shape[0], n_real, replace=False)
    real_sub = data_raw[idx]
    gen_sub = generated_overall[:n_real]

    plot_marginal_distributions(real_sub, gen_sub, station_names, analysis_dir)
    plot_joint_distribution(real_sub, gen_sub, station_names, analysis_dir)

    print(f"\n[plot_model] Análise completa! Resultados em: {analysis_dir}/")


if __name__ == "__main__":
    main()
