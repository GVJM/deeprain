"""
generate_scenarios.py — Geração e validação de cenários temporais com ar_vae.

Uso:
    python generate_scenarios.py
    python generate_scenarios.py --model ar_vae --n_scenarios 50 --n_days 365
    python generate_scenarios.py --model ar_vae --output_dir ./outputs --seed_days 30

Carrega o modelo treinado de outputs/<model>/, usa os últimos `seed_days` dias
dos dados reais como janela semente, e gera N cenários de T dias via rollout
autorregressivo.

Saídas em outputs/<model>/scenarios/:
    scenarios.npy        — array (N, T, S) com cenários gerados (mm/dia)
    fig_timeseries.png   — séries temporais (primeiras 3 estações, 5 cenários)
    fig_autocorr.png     — autocorrelação lags 1–30 vs. observado
    fig_spells.png       — distribuição de spell lengths vs. observado
    fig_seasonal.png     — precipitação média mensal vs. observado
    fig_spread.png       — percentis entre cenários (variabilidade)
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Adiciona PrecipModels/ ao path
sys.path.insert(0, os.path.dirname(__file__))

from data_utils import load_data, denormalize, SABESP_DATA_PATH
from models.ar_vae import ARVAE


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de métricas temporais
# ─────────────────────────────────────────────────────────────────────────────

def _autocorr_series(series: np.ndarray, max_lag: int = 30) -> np.ndarray:
    """
    Autocorrelação de uma série 1-D para lags 1..max_lag.
    Calcula manualmente para evitar dependência de statsmodels.
    """
    n   = len(series)
    mu  = series.mean()
    var = ((series - mu) ** 2).mean()
    if var < 1e-10:
        return np.zeros(max_lag)
    acf = np.array([
        np.mean((series[:n - lag] - mu) * (series[lag:] - mu)) / var
        for lag in range(1, max_lag + 1)
    ])
    return acf


def _mean_autocorr(data: np.ndarray, max_lag: int = 30) -> np.ndarray:
    """Autocorrelação média sobre as estações. data: (T, S)"""
    acfs = np.stack([_autocorr_series(data[:, s], max_lag) for s in range(data.shape[1])])
    return acfs.mean(axis=0)


def _spell_lengths(series: np.ndarray, threshold: float = 0.1) -> dict:
    """
    Extrai distribuição de wet/dry spell lengths de uma série 1-D (mm/dia).
    Returns dict {'wet': [...], 'dry': [...]}
    """
    wet    = series > threshold
    wet_sp = []
    dry_sp = []
    count  = 1
    for i in range(1, len(wet)):
        if wet[i] == wet[i - 1]:
            count += 1
        else:
            (wet_sp if wet[i - 1] else dry_sp).append(count)
            count = 1
    (wet_sp if wet[-1] else dry_sp).append(count)
    return {"wet": wet_sp, "dry": dry_sp}


def _mean_spell_dist(data: np.ndarray, threshold: float = 0.1, max_len: int = 30) -> dict:
    """
    Distribuição média de spell lengths sobre as estações.
    data: (T, S) — mm/dia

    Returns {'wet': array(max_len,), 'dry': array(max_len,)} — frequências relativas
    """
    wet_counts = np.zeros(max_len)
    dry_counts = np.zeros(max_len)

    for s in range(data.shape[1]):
        sp = _spell_lengths(data[:, s], threshold)
        for l in sp["wet"]:
            if 1 <= l <= max_len:
                wet_counts[l - 1] += 1
        for l in sp["dry"]:
            if 1 <= l <= max_len:
                dry_counts[l - 1] += 1

    wet_counts /= max(wet_counts.sum(), 1)
    dry_counts /= max(dry_counts.sum(), 1)
    return {"wet": wet_counts, "dry": dry_counts}


def _monthly_mean(data: np.ndarray, months: np.ndarray) -> np.ndarray:
    """
    Precipitação média por mês sobre todas as estações.
    data: (T, S) mm/dia, months: (T,) int 0..11

    Returns array (12,)
    """
    return np.array([
        data[months == m].mean() if (months == m).any() else 0.0
        for m in range(12)
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Carregamento do modelo
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_dir: str, device: torch.device, n_stations: int) -> ARVAE:
    """
    Instancia e carrega pesos de um checkpoint ar_vae.

    Args:
        n_stations: número de estações — passado externamente pois config.json
                    não salva input_size (é determinado pelos dados).
    """
    config_path = os.path.join(model_dir, "config.json")
    model_path  = os.path.join(model_dir, "model.pt")

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"config.json não encontrado em {model_dir}")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pt não encontrado em {model_dir}")

    with open(config_path) as f:
        cfg = json.load(f)

    model = ARVAE(
        input_size=n_stations,
        window_size=cfg.get("window_size", 30),
        gru_hidden=cfg.get("gru_hidden", 128),
        latent_size=cfg.get("latent_size", 64),
        hidden_size=cfg.get("hidden_size", 256),
    )
    ckpt = torch.load(model_path, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    print(f"[generate] Modelo carregado: {model.count_parameters():,} parâmetros")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────────────────────────────────────

def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  [generate] Salvo: {path}")


def plot_timeseries(scenarios_mm, obs_mm, out_dir, n_plot=5, n_stations=3):
    """Séries temporais: primeiras n_stations estações, n_plot cenários + obs."""
    T = scenarios_mm.shape[1]
    days = np.arange(T)

    fig, axes = plt.subplots(n_stations, 1, figsize=(14, 3 * n_stations), sharex=True)
    if n_stations == 1:
        axes = [axes]

    obs_T = obs_mm[-T:]  # últimos T dias observados para comparação visual
    colors = plt.cm.tab10(np.linspace(0, 0.6, n_plot))

    for ax, st in zip(axes, range(n_stations)):
        for i in range(min(n_plot, scenarios_mm.shape[0])):
            ax.plot(days, scenarios_mm[i, :, st], alpha=0.5, lw=0.8,
                    color=colors[i], label=f"Cen. {i+1}" if st == 0 else None)
        if obs_T.shape[0] == T:
            ax.plot(days, obs_T[:, st], color="black", lw=1.2, alpha=0.9,
                    label="Observado" if st == 0 else None)
        ax.set_ylabel("mm/dia", fontsize=8)
        ax.set_title(f"Estação {st + 1}", fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.tick_params(labelsize=7)

    axes[0].legend(fontsize=7, ncol=3, loc="upper right")
    axes[-1].set_xlabel("Dia", fontsize=9)
    fig.suptitle("Séries Temporais — Cenários vs. Observado", fontsize=11)
    _save(fig, os.path.join(out_dir, "fig_timeseries.png"))


def plot_autocorr(scenarios_mm, obs_mm, out_dir, max_lag=30):
    """Autocorrelação lags 1–max_lag: envelopes dos cenários + observado."""
    lags = np.arange(1, max_lag + 1)
    obs_acf = _mean_autocorr(obs_mm, max_lag)

    sc_acfs = np.stack([_mean_autocorr(scenarios_mm[i], max_lag)
                        for i in range(scenarios_mm.shape[0])])
    p5, p50, p95 = np.percentile(sc_acfs, [5, 50, 95], axis=0)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.fill_between(lags, p5, p95, alpha=0.25, color="steelblue", label="Cenários (5–95%)")
    ax.plot(lags, p50, color="steelblue", lw=1.5, label="Cenários (mediana)")
    ax.plot(lags, obs_acf, color="black", lw=2, linestyle="--", label="Observado")
    ax.axhline(0, color="gray", lw=0.8, linestyle=":")
    ax.set_xlabel("Lag (dias)", fontsize=10)
    ax.set_ylabel("Autocorrelação", fontsize=10)
    ax.set_title("Autocorrelação Temporal — Média sobre Estações", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    _save(fig, os.path.join(out_dir, "fig_autocorr.png"))


def plot_spells(scenarios_mm, obs_mm, out_dir, threshold=0.1, max_len=20):
    """Distribuição de wet/dry spell lengths."""
    obs_sp = _mean_spell_dist(obs_mm, threshold, max_len)

    sc_wet = np.stack([_mean_spell_dist(scenarios_mm[i], threshold, max_len)["wet"]
                       for i in range(scenarios_mm.shape[0])])
    sc_dry = np.stack([_mean_spell_dist(scenarios_mm[i], threshold, max_len)["dry"]
                       for i in range(scenarios_mm.shape[0])])

    lens = np.arange(1, max_len + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    for ax, sc_data, obs_data, title in [
        (ax1, sc_wet, obs_sp["wet"], "Wet Spells"),
        (ax2, sc_dry, obs_sp["dry"], "Dry Spells"),
    ]:
        p5, p50, p95 = np.percentile(sc_data, [5, 50, 95], axis=0)
        ax.fill_between(lens, p5, p95, alpha=0.3, color="steelblue", label="Cenários (5–95%)")
        ax.plot(lens, p50, color="steelblue", lw=1.5, label="Cenários (mediana)")
        ax.bar(lens, obs_data, alpha=0.5, color="black", label="Observado", width=0.4)
        ax.set_xlabel("Comprimento (dias)", fontsize=9)
        ax.set_ylabel("Frequência relativa", fontsize=9)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(f"Spell Lengths (threshold={threshold} mm/dia)", fontsize=11)
    _save(fig, os.path.join(out_dir, "fig_spells.png"))


def plot_seasonal(scenarios_mm, obs_mm, obs_months, out_dir):
    """Precipitação média mensal: cenários vs. observado."""
    obs_monthly = _monthly_mean(obs_mm, obs_months)

    # Para cenários, não temos meses reais — assumimos sequência contínua
    # a partir do mês do primeiro dia da série de avaliação
    sc_monthly_all = np.stack([
        scenarios_mm[i].mean(axis=1).mean()  # média simples por cenário
        for i in range(scenarios_mm.shape[0])
    ])

    # Melhor: calcular mean total por cenário
    sc_means = scenarios_mm.mean(axis=(1, 2))  # (n_sc,)

    fig, ax = plt.subplots(figsize=(9, 4))
    month_labels = ["Jan","Fev","Mar","Abr","Mai","Jun","Jul","Ago","Set","Out","Nov","Dez"]
    x = np.arange(12)

    ax.bar(x - 0.2, obs_monthly, width=0.35, label="Observado", color="black", alpha=0.7)

    # Precipitação diária média dos cenários (escalar, mostrado como linha horizontal)
    sc_mean = scenarios_mm.mean()
    ax.axhline(sc_mean, color="steelblue", lw=2, linestyle="--",
               label=f"Cenários (média={sc_mean:.2f} mm/dia)")

    ax.set_xticks(x)
    ax.set_xticklabels(month_labels, fontsize=9)
    ax.set_ylabel("Precipitação média (mm/dia)", fontsize=10)
    ax.set_title("Sazonalidade — Observado vs. Cenários", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(out_dir, "fig_seasonal.png"))


def plot_spread(scenarios_mm, obs_mm, out_dir):
    """Variabilidade entre cenários: percentis diários agregados."""
    # Média sobre estações por dia
    sc_daily_mean = scenarios_mm.mean(axis=2)   # (n_sc, T)
    p5, p25, p50, p75, p95 = np.percentile(sc_daily_mean, [5, 25, 50, 75, 95], axis=0)
    T = sc_daily_mean.shape[1]
    days = np.arange(T)

    obs_mean = obs_mm[-T:].mean(axis=1) if obs_mm.shape[0] >= T else None

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(days, p5, p95, alpha=0.15, color="steelblue", label="5–95%")
    ax.fill_between(days, p25, p75, alpha=0.30, color="steelblue", label="25–75%")
    ax.plot(days, p50, color="steelblue", lw=1.2, label="Mediana")
    if obs_mean is not None:
        ax.plot(days, obs_mean, color="black", lw=0.8, alpha=0.6, label="Observado")
    ax.set_xlabel("Dia", fontsize=10)
    ax.set_ylabel("Precipitação média (mm/dia)", fontsize=10)
    ax.set_title("Spread entre Cenários — Média sobre Estações", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    _save(fig, os.path.join(out_dir, "fig_spread.png"))


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def generate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[generate] Device: {device}")

    # ── Carrega dados ──
    data_norm, data_raw, mu, std, station_names = load_data(
        data_path=args.data_path,
        normalization_mode="scale_only",
        missing_strategy="impute_station_median",
    )
    n_days_obs, n_stations = data_raw.shape
    print(f"[generate] Dados: {data_raw.shape} | estações: {n_stations}")

    # ── Carrega modelo ──
    model_dir = os.path.join(args.output_dir, args.model)
    model = load_model(model_dir, device, n_stations=n_stations)

    # ── Janela semente: últimos seed_days dias dos dados ──
    W = model.window_size
    seed_days = max(W, args.seed_days)
    seed_raw = data_raw[-seed_days:]
    seed_norm = data_norm[-seed_days:]

    # Usa exatamente os últimos W dias normalizados como window
    seed_window = torch.FloatTensor(seed_norm[-W:])  # (W, S)
    print(f"[generate] Seed: últimos {W} dias dos dados reais (normalizado)")

    # ── Gera cenários ──
    print(f"[generate] Gerando {args.n_scenarios} cenários de {args.n_days} dias...")
    with torch.no_grad():
        scenarios_norm = model.sample_rollout(
            seed_window=seed_window,
            n_days=args.n_days,
            n_scenarios=args.n_scenarios,
        )  # (n_sc, n_days, n_stations) — espaço normalizado

    # Denormaliza: (n_sc, n_days, S) → mm/dia
    scenarios_np = scenarios_norm.cpu().numpy()
    scenarios_mm = scenarios_np * std[0]  # std shape (1, S) → broadcast ok

    print(f"[generate] Cenários: shape={scenarios_mm.shape}, "
          f"min={scenarios_mm.min():.3f}, max={scenarios_mm.max():.3f} mm/dia")
    print(f"[generate] Dias chuvosos (>0.1mm): "
          f"{(scenarios_mm > 0.1).mean() * 100:.1f}% (obs: "
          f"{(data_raw > 0.1).mean() * 100:.1f}%)")

    # ── Diretório de saída ──
    out_dir = os.path.join(model_dir, "scenarios")
    os.makedirs(out_dir, exist_ok=True)

    # ── Salva array ──
    npy_path = os.path.join(out_dir, "scenarios.npy")
    np.save(npy_path, scenarios_mm)
    print(f"[generate] Cenários salvos em {npy_path}")

    # ── Estatísticas resumo ──
    summary = {
        "n_scenarios":      args.n_scenarios,
        "n_days":           args.n_days,
        "n_stations":       n_stations,
        "mean_precip_mm":   float(scenarios_mm.mean()),
        "wet_frac_scen":    float((scenarios_mm > 0.1).mean()),
        "wet_frac_obs":     float((data_raw > 0.1).mean()),
        "lag1_acf_scen":    float(_mean_autocorr(scenarios_mm[0], 1)[0]),
        "lag1_acf_obs":     float(_mean_autocorr(data_raw, 1)[0]),
    }
    import json
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[generate] Resumo: lag1_acf_scen={summary['lag1_acf_scen']:.3f}, "
          f"lag1_acf_obs={summary['lag1_acf_obs']:.3f}")

    # ── Plots ──
    print("[generate] Gerando plots...")
    plot_timeseries(scenarios_mm, data_raw, out_dir)
    plot_autocorr(scenarios_mm, data_raw, out_dir)
    plot_spells(scenarios_mm, data_raw, out_dir)
    plot_spread(scenarios_mm, data_raw, out_dir)

    # Para sazonalidade: estima meses dos dados observados
    try:
        import pandas as pd
        from data_utils import load_sabesp_daily_precip
        df = load_sabesp_daily_precip()
        months_obs = (df.index.month.values - 1).astype(int)
        # alinha ao data_raw (que pode ter sido filtrado)
        months_obs = months_obs[-n_days_obs:]
        plot_seasonal(scenarios_mm, data_raw, months_obs, out_dir)
    except Exception as e:
        print(f"[generate] Aviso: seasonal plot não gerado ({e})")

    print(f"\n[generate] Concluído. Figuras em {out_dir}/")


def main():
    parser = argparse.ArgumentParser(
        description="Gera cenários temporais com modelo autorregressivo.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model",        type=str, default="ar_vae",
                        help="Nome do modelo (deve ter outputs/<model>/model.pt)")
    parser.add_argument("--n_scenarios",  type=int, default=50,
                        help="Número de cenários a gerar")
    parser.add_argument("--n_days",       type=int, default=365,
                        help="Dias por cenário")
    parser.add_argument("--seed_days",    type=int, default=30,
                        help="Dias de observação usados como janela semente")
    parser.add_argument("--data_path",    type=str, default=SABESP_DATA_PATH,
                        help="Caminho para os dados de precipitação")
    parser.add_argument("--output_dir",   type=str, default="./outputs",
                        help="Diretório base onde estão os modelos treinados")
    args = parser.parse_args()
    generate(args)


if __name__ == "__main__":
    main()
