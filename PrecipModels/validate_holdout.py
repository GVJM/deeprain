"""
validate_holdout.py — Validação temporal holdout para modelos de precipitação.

Avalia modelos treinados em dados "out-of-sample": os últimos `holdout_frac`
dos dados (em ordem temporal) são usados como conjunto de teste.

O objetivo é verificar se os modelos generalizam para períodos fora do treinamento,
comparando métricas in-sample (salvas por train.py) com métricas out-of-sample
(calculadas aqui com os dados de teste).

Uso:
    # Avalia todos os modelos com holdout de 20%
    python validate_holdout.py

    # Subconjunto de modelos
    python validate_holdout.py --models hurdle_vae_cond flow_match copula

    # Holdout de 30% (últimos 30% dos dados)
    python validate_holdout.py --holdout_frac 0.30 --n_samples 1000

    # Smoke test rápido
    python validate_holdout.py --models hurdle_vae_cond copula --n_samples 300

Saídas em --holdout_dir (default: ./outputs/holdout/):
    holdout_metrics.json       — métricas out-of-sample por modelo
    holdout_comparison.png     — comparação in-sample vs out-of-sample
    holdout_table.txt          — tabela texto com ranking e degradação
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import sys
import json
import argparse
import pickle
import contextlib
import io
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch

# Garante imports do diretório PrecipModels
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from data_utils import load_data, denormalize
from metrics import evaluate_model
from models import get_model


# ──────────────────────────────────────────────────────────
# CARREGAMENTO DE MODELO TREINADO
# ──────────────────────────────────────────────────────────

def _load_config(model_name: str, output_dir: str) -> dict:
    path = os.path.join(output_dir, model_name, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load_trained_model(variant_name: str, output_dir: str, input_size: int, device: torch.device):
    """Carrega checkpoint treinado de output_dir/variant_name/model.pt."""
    model_dir = os.path.join(output_dir, variant_name)
    cfg = _load_config(variant_name, output_dir)
    model_class = cfg.get("model", variant_name)

    # Cópula: não é PyTorch
    if model_class == "copula":
        copula_path = os.path.join(model_dir, "copula.pkl")
        if not os.path.exists(copula_path):
            raise FileNotFoundError(copula_path)
        with open(copula_path, "rb") as f:
            return pickle.load(f), cfg

    # Constrói kwargs de arquitetura a partir do config salvo
    model_kwargs = dict(input_size=input_size)
    for key in (
        "latent_size", "hidden_size", "n_layers", "n_coupling",
        "hidden_occ", "hidden_amt", "gru_hidden", "context_dim", "window_size",
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
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model = model.to(device)
    model.eval()

    # Restaura cond_probs para modelos _mc
    cond_probs_raw = cfg.get("cond_probs")
    if cond_probs_raw and hasattr(model, "_cond_probs"):
        model._cond_probs = {k: np.array(v) for k, v in cond_probs_raw.items()}

    return model, cfg


# ──────────────────────────────────────────────────────────
# AVALIAÇÃO HOLDOUT
# ──────────────────────────────────────────────────────────

def evaluate_holdout(
    model_name: str,
    output_dir: str,
    test_raw: np.ndarray,
    test_norm: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    station_names: list,
    n_samples: int,
    device: torch.device,
    data_path: str,
) -> dict:
    """
    Carrega o modelo treinado e avalia no conjunto de teste (holdout).

    Returns:
        dict de métricas out-of-sample
    """
    model, cfg = _load_trained_model(model_name, output_dir, test_raw.shape[1], device)
    model_class = cfg.get("model", model_name)
    norm_mode = cfg.get("normalization_mode", "scale_only")

    # Modelos especiais precisam de dados de treino para contexto/transform interno
    if model_class == "latent_flow":
        # Recarrega dados completos para reconstruir o transform interno
        def _silent_load(mode):
            with contextlib.redirect_stdout(io.StringIO()):
                return load_data(data_path=data_path, normalization_mode=mode)
        _, data_raw_full, _, std_sc, _ = _silent_load("scale_only")
        model.fit_flow(data_raw_full, std_scale=std_sc)

    if model_class == "hurdle_temporal":
        def _silent_load(mode):
            with contextlib.redirect_stdout(io.StringIO()):
                return load_data(data_path=data_path, normalization_mode=mode)
        _, data_raw_full, _, std_sc, _ = _silent_load("scale_only")
        train_norm_cmp = data_raw_full / std_sc
        model.fit_temporal(train_norm_cmp)

    samples_are_normalized = (model_class != "copula")

    metrics = evaluate_model(
        model,
        test_raw,       # dados de teste (holddout) como referência
        mu,
        std,
        n_samples=n_samples,
        station_names=station_names,
        samples_are_normalized=samples_are_normalized,
    )
    return metrics


# ──────────────────────────────────────────────────────────
# GRÁFICO DE COMPARAÇÃO
# ──────────────────────────────────────────────────────────

def plot_holdout_comparison(
    in_sample: dict,
    out_sample: dict,
    model_names: list,
    out_dir: str,
):
    """
    Grouped bar chart: in-sample vs out-of-sample por métrica.

    Métricas comparadas: Wasserstein, Q90, Energy Score, Coverage 90%.
    Barras agrupadas: cada grupo = um modelo; barra clara = in-sample, barra escura = holdout.
    """
    metrics_to_compare = [
        ("mean_wasserstein",       "Wasserstein Médio (mm/dia)"),
        ("extreme_q90_mean",       "Q90 Erro (mm/dia)"),
        ("energy_score",           "Energy Score"),
        ("wet_day_freq_error_mean","Wet Day Freq Error"),
    ]
    # Filtra modelos com ambas as métricas disponíveis
    valid_models = [m for m in model_names if m in in_sample and m in out_sample]
    if not valid_models:
        print("[holdout] Nenhum modelo com dados suficientes para o gráfico.")
        return

    n_metrics = len(metrics_to_compare)
    fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    x = np.arange(len(valid_models))
    width = 0.35
    palette_in = "#3498db"    # azul claro = in-sample
    palette_out = "#e74c3c"   # vermelho = out-of-sample (possivelmente pior)

    for ax, (mkey, mlabel) in zip(axes, metrics_to_compare):
        vals_in  = [in_sample[m].get(mkey, np.nan) for m in valid_models]
        vals_out = [out_sample[m].get(mkey, np.nan) for m in valid_models]

        bars_in  = ax.bar(x - width / 2, vals_in,  width, label="In-sample",   color=palette_in,  alpha=0.85, edgecolor="white")
        bars_out = ax.bar(x + width / 2, vals_out, width, label="Out-of-sample (holdout)", color=palette_out, alpha=0.85, edgecolor="white")

        # Anotações numéricas acima das barras
        for bar, val in zip(bars_in, vals_in):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7, color=palette_in)
        for bar, val in zip(bars_out, vals_out):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7, color=palette_out)

        ax.set_xticks(x)
        ax.set_xticklabels(valid_models, rotation=35, ha="right", fontsize=8)
        ax.set_title(mlabel, fontsize=10, fontweight="bold")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylabel("Valor (↓ melhor)", fontsize=8)

    # Legenda compartilhada
    patch_in  = mpatches.Patch(color=palette_in,  label="In-sample (treino)")
    patch_out = mpatches.Patch(color=palette_out, label="Out-of-sample (holdout)")
    fig.legend(handles=[patch_in, patch_out], loc="upper center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, 1.02))

    fig.suptitle("Validação Holdout Temporal\n(Comparação In-sample vs Out-of-sample)",
                 fontsize=13, fontweight="bold", y=1.05)
    plt.tight_layout()
    out_path = os.path.join(out_dir, "holdout_comparison.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[holdout] Gráfico de comparação salvo: {out_path}")


def plot_degradation_bar(
    in_sample: dict,
    out_sample: dict,
    model_names: list,
    out_dir: str,
):
    """
    Barras de degradação relativa: (out - in) / in * 100%.
    Positivo = piorou no holdout; negativo = melhorou (improvável).
    """
    valid_models = [m for m in model_names if m in in_sample and m in out_sample]
    if not valid_models:
        return

    metrics_for_degradation = [
        ("mean_wasserstein",  "Wasserstein"),
        ("extreme_q90_mean",  "Q90"),
        ("energy_score",      "Energy"),
    ]
    n_metrics = len(metrics_for_degradation)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4.5 * n_metrics, 5))
    if n_metrics == 1:
        axes = [axes]

    for ax, (mkey, mlabel) in zip(axes, metrics_for_degradation):
        degradation = []
        for m in valid_models:
            vin  = in_sample[m].get(mkey, np.nan)
            vout = out_sample[m].get(mkey, np.nan)
            if not np.isnan(vin) and not np.isnan(vout) and vin > 1e-12:
                degradation.append((vout - vin) / vin * 100.0)
            else:
                degradation.append(np.nan)

        colors = ["#e74c3c" if d > 0 else "#2ecc71" for d in degradation]
        bars = ax.bar(valid_models, degradation, color=colors, alpha=0.85, edgecolor="white")

        ax.axhline(0, color="black", linewidth=1.0, linestyle="--", alpha=0.5)
        ax.set_xticks(range(len(valid_models)))
        ax.set_xticklabels(valid_models, rotation=35, ha="right", fontsize=8)
        ax.set_title(f"Degradação {mlabel} (%)", fontsize=10, fontweight="bold")
        ax.set_ylabel("(holdout - in-sample) / in-sample × 100%", fontsize=7)
        ax.grid(axis="y", alpha=0.3)

        for bar, val in zip(bars, degradation):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + (0.5 if val >= 0 else -2.5),
                        f"{val:+.1f}%", ha="center", va="bottom", fontsize=8, fontweight="bold")

    fig.suptitle("Degradação Relativa no Holdout Temporal\n(vermelho = piorou; verde = melhorou)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    out_path = os.path.join(out_dir, "holdout_degradation.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[holdout] Gráfico de degradação salvo: {out_path}")


# ──────────────────────────────────────────────────────────
# RELATÓRIO TEXTO
# ──────────────────────────────────────────────────────────

def print_holdout_table(in_sample: dict, out_sample: dict, model_names: list, holdout_frac: float) -> str:
    key_pairs = [
        ("mean_wasserstein",        "Wasserstein"),
        ("corr_rmse",               "Corr RMSE"),
        ("wet_day_freq_error_mean",  "Wet Day"),
        ("extreme_q90_mean",        "Q90"),
        ("extreme_q99_mean",        "Q99"),
        ("energy_score",            "Energy"),
        ("coverage_90",             "Cov90%"),
    ]
    col = 24
    lines = [
        "",
        "=" * 110,
        f"  VALIDAÇÃO HOLDOUT TEMPORAL (últimos {holdout_frac:.0%} dos dados como teste)",
        f"  Formato: IN | OUT  (IN = in-sample, OUT = out-of-sample holdout)",
        "=" * 110,
    ]

    # Cabeçalho
    hdr = f"{'Modelo':<{col}}"
    for _, lbl in key_pairs:
        hdr += f"  {'IN':>8} {'OUT':>8} {'Δ%':>6}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for m in model_names:
        mi = in_sample.get(m, {})
        mo = out_sample.get(m, {})
        row = f"{m:<{col}}"
        for mkey, _ in key_pairs:
            vin  = float(mi.get(mkey, np.nan))
            vout = float(mo.get(mkey, np.nan))
            if not np.isnan(vin):
                row += f"  {vin:>8.3f}"
            else:
                row += f"  {'N/A':>8}"
            if not np.isnan(vout):
                row += f" {vout:>8.3f}"
            else:
                row += f" {'N/A':>8}"
            # Degradação relativa
            if not np.isnan(vin) and not np.isnan(vout) and vin > 1e-12:
                delta = (vout - vin) / vin * 100.0
                row += f" {delta:>+6.1f}%"
            else:
                row += f"  {'N/A':>6}"
        lines.append(row)

    lines.append("=" * 110)
    lines.append("\n(Δ% positivo em métricas de erro = modelo PIOROU no holdout; negativo = melhorou)")
    lines.append("(Δ% positivo em Coverage = maior cobertura = modelo mais conservador no holdout)\n")

    report = "\n".join(lines)
    print(report)
    return report


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validação temporal holdout para modelos de precipitação.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--models", nargs="+", default=None,
                        help="Modelos a avaliar (nomes dos subdiretórios em --output_dir). "
                             "Se não especificado, usa todos os modelos com metrics.json salvo.")
    parser.add_argument("--output_dir", type=str, default="./outputs",
                        help="Diretório base com modelos treinados")
    parser.add_argument("--holdout_dir", type=str, default="./outputs/holdout",
                        help="Diretório de saída para resultados holdout")
    parser.add_argument("--data_path", type=str, default="../dados_sabesp/dayprecip.dat",
                        help="Caminho para o arquivo de dados")
    parser.add_argument("--holdout_frac", type=float, default=0.20,
                        help="Fração dos dados (em ordem temporal) usada como teste holdout")
    parser.add_argument("--n_samples", type=int, default=2000,
                        help="Amostras sintéticas para avaliação de cada modelo")
    parser.add_argument("--device", type=str, default="auto",
                        help="'auto', 'cpu', ou 'cuda'")
    args = parser.parse_args()

    # Dispositivo
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"[holdout] Device: {device}")

    os.makedirs(args.holdout_dir, exist_ok=True)

    # ── Carrega dados e faz split temporal ────────────────
    print("[holdout] Carregando dados...")
    data_norm, data_raw, mu, std, station_names = load_data(
        data_path=args.data_path,
        normalization_mode="scale_only",
    )
    N = data_raw.shape[0]
    n_test = int(N * args.holdout_frac)
    n_train = N - n_test

    test_raw  = data_raw[n_train:]
    test_norm = data_norm[n_train:]

    print(f"[holdout] Total de dias: {N} | Treino (in-sample): {n_train} | Teste (holdout): {n_test}")
    print(f"[holdout] Estações: {len(station_names)}")

    # ── Descobre modelos disponíveis ─────────────────────
    if args.models is None:
        # Detecta automaticamente todos os subdiretórios com metrics.json
        args.models = []
        for entry in os.scandir(args.output_dir):
            if entry.is_dir() and entry.name != "comparison" and entry.name != "holdout" and entry.name != "ablation":
                metrics_path = os.path.join(entry.path, "metrics.json")
                if os.path.exists(metrics_path):
                    args.models.append(entry.name)
        args.models.sort()
        print(f"[holdout] Modelos detectados automaticamente: {args.models}")

    # ── Carrega métricas in-sample (salvas por train.py) ──
    in_sample_metrics = {}
    for m in args.models:
        path = os.path.join(args.output_dir, m, "metrics.json")
        if os.path.exists(path):
            with open(path, encoding="utf-8") as f:
                in_sample_metrics[m] = json.load(f)
        else:
            print(f"[AVISO] metrics.json não encontrado para '{m}' — será excluído da comparação in-sample")

    # ── Avalia cada modelo no conjunto de teste ────────────
    out_sample_metrics = {}
    out_file = os.path.join(args.holdout_dir, "holdout_metrics.json")

    for m in args.models:
        print(f"\n[holdout] Avaliando: {m}")
        try:
            metrics = evaluate_holdout(
                model_name=m,
                output_dir=args.output_dir,
                test_raw=test_raw,
                test_norm=test_norm,
                mu=mu,
                std=std,
                station_names=list(station_names),
                n_samples=args.n_samples,
                device=device,
                data_path=args.data_path,
            )
            out_sample_metrics[m] = metrics
        except FileNotFoundError as e:
            print(f"[AVISO] Modelo '{m}' não encontrado: {e}")
        except Exception as e:
            print(f"[AVISO] Falha ao avaliar '{m}': {e}")
            import traceback
            traceback.print_exc()

    if not out_sample_metrics:
        print("[holdout] Nenhum modelo avaliado com sucesso. Abortando.")
        return

    # Salva métricas holdout (serializa arrays numpy)
    serializable_out = {}
    for m, mdict in out_sample_metrics.items():
        serializable_out[m] = {}
        for k, v in mdict.items():
            if isinstance(v, dict):
                serializable_out[m][k] = {
                    k2: (v2.tolist() if hasattr(v2, "tolist") else v2)
                    for k2, v2 in v.items()
                }
            elif hasattr(v, "tolist"):
                serializable_out[m][k] = v.tolist()
            else:
                serializable_out[m][k] = v

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(serializable_out, f, indent=2)
    print(f"\n[holdout] Métricas holdout salvas: {out_file}")

    # ── Relatório texto ────────────────────────────────────
    model_names_eval = list(out_sample_metrics.keys())
    report = print_holdout_table(in_sample_metrics, out_sample_metrics, model_names_eval, args.holdout_frac)
    with open(os.path.join(args.holdout_dir, "holdout_table.txt"), "w", encoding="utf-8") as f:
        f.write(report)

    # ── Gráficos ───────────────────────────────────────────
    try:
        plot_holdout_comparison(in_sample_metrics, out_sample_metrics, model_names_eval, args.holdout_dir)
    except Exception as e:
        print(f"[AVISO] Falha no gráfico de comparação: {e}")

    try:
        plot_degradation_bar(in_sample_metrics, out_sample_metrics, model_names_eval, args.holdout_dir)
    except Exception as e:
        print(f"[AVISO] Falha no gráfico de degradação: {e}")

    print(f"\n[holdout] ✓ Validação holdout completa! Resultados em: {args.holdout_dir}/")


if __name__ == "__main__":
    main()
