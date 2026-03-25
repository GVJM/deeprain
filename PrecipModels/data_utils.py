"""
data_utils.py — Carregamento e pré-processamento dos dados de precipitação INMET.

Portagem limpa de VAE_Tests/experiment.py:load_data() com suporte a dois modos
de normalização e retorno dos parâmetros para denormalização posterior.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List


# Caminho padrão relativo a PrecipModels/
DEFAULT_DATA_PATH = "../dados_barragens_btg/inmet_relevant_data.csv"
SABESP_DATA_PATH = "../dados_sabesp/dayprecip.dat"

def load_btg_barragens_inmet_df():
    raw_path = Path(DEFAULT_DATA_PATH)
    candidates = [raw_path]
    if not raw_path.is_absolute():
        module_dir = Path(__file__).resolve().parent
        candidates.append(module_dir / raw_path)
        candidates.append(module_dir.parent / raw_path)

    path = None
    for cand in candidates:
        if cand.exists():
            path = cand
            break

    if path is None:
        tried = "\n".join(f"  - {c.resolve()}" for c in candidates)
        raise FileNotFoundError(
            "Arquivo de dados não encontrado.\n"
            f"Caminhos testados:\n{tried}\n"
            "Verifique se o caminho aponta para 'inmet_relevant_data.csv'."
        )

    inmet_data = pd.read_csv(path)

    # Limpa código de estação (trailing ';')
    inmet_data['cd_inmet_station'] = inmet_data['cd_inmet_station'].str.replace(";", "", regex=False)

    # Normaliza datas
    inmet_data['dt'] = inmet_data['dt'].str.replace(r"/", "-", regex=False)
    inmet_data['hr'] = inmet_data['hr'].str.slice(stop=2).astype(int)

    # Precipitação: vírgula → ponto, -9999 → NaN
    inmet_data['precipitacao'] = (
        inmet_data['precipitacao']
        .str.replace(",", ".", regex=False)
        .astype(float)
        .replace(-9999, np.nan)
    )

    # Cria datetime completo
    inmet_data['datetime'] = (
        pd.to_datetime(inmet_data['dt'], format='%Y-%m-%d')
        + pd.to_timedelta(inmet_data['hr'], unit='h')
    )

    # Pivot para tabela (datetime × estação).
    # Usa pivot_table para tolerar duplicatas de (datetime, estação) no CSV.
    pivot = inmet_data.pivot_table(
        index='datetime',
        columns='cd_inmet_station',
        values='precipitacao',
        aggfunc='mean',
    )

    # Reindex para série horária completa e resample diário
    date_range = pd.date_range(
        start=inmet_data['datetime'].min(),
        end=inmet_data['datetime'].max(),
        freq='h'
    )
    pivot = pivot.reindex(date_range)
    pivot = pivot.resample("D").sum(min_count=1)

    return pivot


def load_sabesp_daily_precip(path: str = SABESP_DATA_PATH):
    dados = pd.read_csv(path)
    dados['datetime'] = pd.to_datetime(dados['datetime'])

    dodos_count = dados.count()

    total_days = dodos_count['datetime']
    non_relevant_stations = dodos_count[dodos_count < (total_days * 0.99)]

    dados.drop(non_relevant_stations.index, axis=1, inplace=True)
    dados.set_index('datetime', inplace=True)
    dados = dados.add_prefix("st_")

    return dados

def load_data(
    data_path: str = SABESP_DATA_PATH,
    normalization_mode: str = "scale_only",
    dropna: bool = True,
    missing_strategy: str = "drop_any",
):
    """
    Carrega e pré-processa os dados de precipitação INMET.

    Etapas:
        1. Lê CSV, limpa separadores e códigos de estação
        2. Converte precipitação: vírgula → ponto, -9999 → NaN
        3. Pivot horário → resample diário (sum, min_count=1)
        4. Trata NaNs (imputação por estação por padrão)
        5. Normaliza conforme mode

    Args:
        data_path: caminho para inmet_relevant_data.csv
        normalization_mode: "scale_only" (divide por std; NÃO subtrai a média — mu=zeros apenas para denormalização) ou
                            "standardize" (z-score completo)
        dropna: compat legado.
            - True: aplica `missing_strategy`
            - False: não aplica tratamento extra (mantém NaNs)
        missing_strategy:
            - "impute_station_median" (padrão): remove apenas dias com todas as
              estações NaN e imputa o restante pela mediana da estação.
            - "drop_any": remove linhas com qualquer NaN (comportamento antigo).

    Returns:
        data_norm: np.ndarray (N, S) — dados normalizados
        data_raw:  np.ndarray (N, S) — dados brutos em mm/dia (após dropna)
        mu:        np.ndarray (1, S) — média subtraída (zeros em scale_only, média real em standardize)
        std:       np.ndarray (1, S) — desvio padrão usado
        station_names: list[str]
    """
    if data_path == DEFAULT_DATA_PATH:
        pivot = load_btg_barragens_inmet_df()
    else:
        pivot = load_sabesp_daily_precip(data_path)

    station_names = list(pivot.columns)
    data_np = pivot.to_numpy()

    # Tratamento de NaN
    if dropna:
        if missing_strategy == "drop_any":
            nan_rows = np.any(np.isnan(data_np), axis=1)
            data_np = data_np[~nan_rows]
            print(f"Linhas removidas (NaN em qualquer estação): {nan_rows.sum()} -> dados limpos: {data_np.shape}")
        elif missing_strategy == "impute_station_median":
            all_nan_rows = np.all(np.isnan(data_np), axis=1)
            data_np = data_np[~all_nan_rows]
            med = np.nanmedian(data_np, axis=0, keepdims=True)
            med = np.where(np.isnan(med), 0.0, med)
            nan_mask = np.isnan(data_np)
            data_np = np.where(nan_mask, med, data_np)
            print(
                f"Linhas removidas (todas estações NaN): {all_nan_rows.sum()} | "
                f"valores imputados: {nan_mask.sum()} -> dados limpos: {data_np.shape}"
            )
        else:
            raise ValueError(
                f"missing_strategy inválido: '{missing_strategy}'. "
                "Use 'impute_station_median' ou 'drop_any'."
            )

    data_raw = data_np.copy()

    # Normalização
    std = np.nanstd(data_np, axis=0, keepdims=True)
    std = np.clip(std, 1e-8, None)

    if normalization_mode == "scale_only":
        mu = np.zeros((1, data_np.shape[1]), dtype=data_np.dtype)
        data_norm = data_np / std
    elif normalization_mode == "standardize":
        mu = np.nanmean(data_np, axis=0, keepdims=True)
        data_norm = (data_np - mu) / std
    else:
        raise ValueError(f"normalization_mode inválido: '{normalization_mode}'. Use 'scale_only' ou 'standardize'.")

    print(f"Dados carregados: {data_norm.shape} | estações: {len(station_names)} | modo: {normalization_mode}")
    return data_norm, data_raw, mu, std, station_names


def load_data_with_cond(
    data_path: str = SABESP_DATA_PATH,
    normalization_mode: str = "scale_only",
    dropna: bool = True,
    missing_strategy: str = "drop_any",
):
    """
    Igual a load_data(), mas também retorna arrays de condicionamento.

    O dict cond_arrays contém arrays inteiros alinhados com data_norm/data_raw.
    Adicionar novo condicionador = computar array a partir do índice datetime
    e incluí-lo no dict retornado.

    Returns:
        data_norm, data_raw, mu, std, station_names,
        cond_arrays: dict[str, np.ndarray(N,)] — ex: {"month": array(0..11)}
    """
    if data_path == DEFAULT_DATA_PATH:
        pivot = load_btg_barragens_inmet_df()
    else:
        pivot = load_sabesp_daily_precip(data_path)

    station_names = list(pivot.columns)
    data_np = pivot.to_numpy()

    # Mês (0–11) antes do dropna para manter alinhamento
    months_full = (pivot.index.month.values - 1).astype(np.int64)
    days_ciclic = 2.0 * np.pi * pivot.index.day.values.astype(np.float64) / 365.25
    day_sin = np.sin(days_ciclic)
    day_cos = np.cos(days_ciclic)

    # Tratamento de NaN (mesmo comportamento de load_data)
    if dropna:
        if missing_strategy == "drop_any":
            nan_rows = np.any(np.isnan(data_np), axis=1)
            data_np = data_np[~nan_rows]
            months_full = months_full[~nan_rows]
            day_sin = day_sin[~nan_rows]
            day_cos = day_cos[~nan_rows]
            print(f"Linhas removidas (NaN em qualquer estação): {nan_rows.sum()} -> dados limpos: {data_np.shape}")
        elif missing_strategy == "impute_station_median":
            all_nan_rows = np.all(np.isnan(data_np), axis=1)
            data_np = data_np[~all_nan_rows]
            months_full = months_full[~all_nan_rows]
            day_sin = day_sin[~all_nan_rows]
            day_cos = day_cos[~all_nan_rows]
            med = np.nanmedian(data_np, axis=0, keepdims=True)
            med = np.where(np.isnan(med), 0.0, med)
            nan_mask = np.isnan(data_np)
            data_np = np.where(nan_mask, med, data_np)
            print(
                f"Linhas removidas (todas estações NaN): {all_nan_rows.sum()} | "
                f"valores imputados: {nan_mask.sum()} -> dados limpos: {data_np.shape}"
            )
        else:
            raise ValueError(
                f"missing_strategy inválido: '{missing_strategy}'. "
                "Use 'impute_station_median' ou 'drop_any'."
            )

    data_raw = data_np.copy()

    # Normalização
    std = np.nanstd(data_np, axis=0, keepdims=True)
    std = np.clip(std, 1e-8, None)

    if normalization_mode == "scale_only":
        mu = np.zeros((1, data_np.shape[1]), dtype=data_np.dtype)
        data_norm = data_np / std
    elif normalization_mode == "standardize":
        mu = np.nanmean(data_np, axis=0, keepdims=True)
        data_norm = (data_np - mu) / std
    else:
        raise ValueError(f"normalization_mode inválido: '{normalization_mode}'.")

    cond_arrays = {"month": months_full, "day_sin": day_sin, "day_cos": day_cos}
    # Para adicionar ENSO no futuro: cond_arrays["enso"] = enso_phases

    print(f"Dados carregados (com cond): {data_norm.shape} | estações: {len(station_names)} | modo: {normalization_mode}")
    return data_norm, data_raw, mu, std, station_names, cond_arrays


def denormalize(data_norm: np.ndarray, mu: np.ndarray, std: np.ndarray) -> np.ndarray:
    """Inverte a normalização: retorna mm/dia."""
    return data_norm * std + mu


# ─────────────────────────────────────────────────────────────────────────────
# Temporal split and normalization helpers (shared by train.py and evaluate.py)
# ─────────────────────────────────────────────────────────────────────────────

def temporal_holdout_split_with_cond(
    data_raw: np.ndarray,
    cond_arrays: dict,
    holdout_ratio: float,
) -> Tuple[np.ndarray, np.ndarray, dict, dict]:
    """
    Split temporal para dados com condicionamento.
    Aplica o mesmo índice de corte a data_raw e a todos os arrays em cond_arrays.

    Returns:
        train_raw, eval_raw, train_cond, eval_cond
    """
    if holdout_ratio <= 0:
        return data_raw, data_raw, cond_arrays, cond_arrays

    n_total = data_raw.shape[0]
    n_eval = max(1, int(round(n_total * holdout_ratio)))
    n_eval = min(n_eval, n_total - 1)
    cut = n_total - n_eval

    train_raw = data_raw[:cut]
    eval_raw = data_raw[cut:]
    train_cond = {k: v[:cut] for k, v in cond_arrays.items()}
    eval_cond = {k: v[cut:] for k, v in cond_arrays.items()}
    return train_raw, eval_raw, train_cond, eval_cond


def temporal_holdout_split(data_raw: np.ndarray, holdout_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split temporal (sem embaralhar):
      - treino = parte inicial
      - avaliação = parte final (holdout)
    """
    if holdout_ratio <= 0:
        return data_raw, data_raw

    n_total = data_raw.shape[0]
    n_eval = max(1, int(round(n_total * holdout_ratio)))
    n_eval = min(n_eval, n_total - 1)
    return data_raw[:-n_eval], data_raw[-n_eval:]


def temporal_train_val_test_split(
    data_raw: np.ndarray,
    holdout_ratio: float,
    val_ratio: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray]:
    """
    3-way temporal split (chronological, no shuffle):
      [train | val | test]
    - test  = last holdout_ratio fraction (same as current holdout)
    - val   = last val_ratio fraction of pre-test data (None when val_ratio <= 0)
    - train = everything before val

    When val_ratio=0 returns (train, None, test) — backward-compatible with 2-way split.
    """
    n = data_raw.shape[0]
    n_test = max(1, int(round(n * holdout_ratio))) if holdout_ratio > 0 else 0
    n_test = min(n_test, n - 1)
    n_remaining = n - n_test
    n_val = max(1, int(round(n_remaining * val_ratio))) if val_ratio > 0 else 0
    if n_val > 0:
        n_val = min(n_val, n_remaining - 1)
    n_train = n_remaining - n_val

    train_raw = data_raw[:n_train]
    val_raw   = data_raw[n_train:n_train + n_val] if n_val > 0 else None
    test_raw  = data_raw[n_train + n_val:] if n_test > 0 else data_raw
    return train_raw, val_raw, test_raw


def temporal_train_val_test_split_with_cond(
    data_raw: np.ndarray,
    cond_arrays: dict,
    holdout_ratio: float,
    val_ratio: float,
) -> Tuple[np.ndarray, Optional[np.ndarray], np.ndarray, dict, Optional[dict], dict]:
    """
    3-way temporal split for data + conditioning arrays.
    Returns: train_raw, val_raw, test_raw, train_cond, val_cond, test_cond
    val_raw and val_cond are None when val_ratio <= 0.
    """
    n = data_raw.shape[0]
    n_test = max(1, int(round(n * holdout_ratio))) if holdout_ratio > 0 else 0
    n_test = min(n_test, n - 1)
    n_remaining = n - n_test
    n_val = max(1, int(round(n_remaining * val_ratio))) if val_ratio > 0 else 0
    if n_val > 0:
        n_val = min(n_val, n_remaining - 1)
    n_train = n_remaining - n_val

    train_raw = data_raw[:n_train]
    val_raw   = data_raw[n_train:n_train + n_val] if n_val > 0 else None
    test_raw  = data_raw[n_train + n_val:] if n_test > 0 else data_raw

    train_cond = {k: v[:n_train] for k, v in cond_arrays.items()}
    val_cond   = {k: v[n_train:n_train + n_val] for k, v in cond_arrays.items()} if n_val > 0 else None
    test_cond  = {k: v[n_train + n_val:] for k, v in cond_arrays.items()} if n_test > 0 else cond_arrays
    return train_raw, val_raw, test_raw, train_cond, val_cond, test_cond


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
