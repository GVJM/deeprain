"""
metrics.py — Métricas de qualidade e performance para comparação de modelos generativos.

Métricas de qualidade:
    wasserstein_per_station   — Distância Wasserstein 1D por estação (distribuição marginal)
    correlation_rmse          — RMSE entre matrizes de correlação espacial
    wet_day_frequency_error   — |P(chuva_real) - P(chuva_gerado)| por estação
    extreme_quantile_error    — Erro absoluto nos quantis 90%, 95%, 99% por estação
    energy_score              — Escore de energia multivariado (regra de pontuação própria)

Métricas de performance:
    sampling_time_ms          — Mediana de 5 medições para gerar 1000 amostras
    training_ms_per_epoch     — Tempo total de treino / n_epochs
    n_parameters              — Parâmetros treináveis (0 para Cópula)
"""

import time
import numpy as np
import torch
from scipy import stats

WET_DAY_THRESHOLD_MM = 0.1


# ──────────────────────────────────────────────────────────
# MÉTRICAS TEMPORAIS (adicionais — informativas, fora do composite score)
# ──────────────────────────────────────────────────────────

def _run_lengths(series: np.ndarray, wet: bool) -> np.ndarray:
    """Comprimentos de sequências de dias chuvosos (wet=True) ou secos (wet=False)."""
    binary = (series > WET_DAY_THRESHOLD_MM) if wet else (series <= WET_DAY_THRESHOLD_MM)
    lengths = []
    count = 0
    for v in binary:
        if v:
            count += 1
        elif count > 0:
            lengths.append(count)
            count = 0
    if count > 0:
        lengths.append(count)
    return np.array(lengths, dtype=float) if lengths else np.array([0.0])


def wet_spell_length_error(real: np.ndarray, generated: np.ndarray) -> dict:
    """
    Erro no comprimento médio de sequências chuvosas (wet spells) por estação.

    Captura se o modelo preserva a persistência temporal da precipitação.
    Menor = melhor.

    Returns:
        dict com 'per_station' (array S) e 'mean' (escalar)
    """
    S = real.shape[1]
    err = np.zeros(S)
    for i in range(S):
        real_mean = _run_lengths(real[:, i], wet=True).mean()
        gen_mean = _run_lengths(generated[:, i], wet=True).mean()
        err[i] = abs(real_mean - gen_mean)
    return {"per_station": err, "mean": float(err.mean())}


def dry_spell_length_error(real: np.ndarray, generated: np.ndarray) -> dict:
    """
    Erro no comprimento médio de sequências secas (dry spells) por estação.

    Returns:
        dict com 'per_station' (array S) e 'mean' (escalar)
    """
    S = real.shape[1]
    err = np.zeros(S)
    for i in range(S):
        real_mean = _run_lengths(real[:, i], wet=False).mean()
        gen_mean = _run_lengths(generated[:, i], wet=False).mean()
        err[i] = abs(real_mean - gen_mean)
    return {"per_station": err, "mean": float(err.mean())}


def lag1_autocorr_error(real: np.ndarray, generated: np.ndarray) -> dict:
    """
    Erro absoluto na autocorrelação lag-1 por estação.

    Mede se o modelo captura a persistência temporal (tendência de dias chuvosos
    seguirem dias chuvosos).
    Menor = melhor.

    Returns:
        dict com 'per_station' (array S) e 'mean' (escalar)
    """
    S = real.shape[1]
    err = np.zeros(S)
    for i in range(S):
        r = real[:, i]
        g = generated[:, i]
        # Correlação de Pearson entre x[t] e x[t-1]
        if np.std(r) > 1e-12 and len(r) > 2:
            ac_real = float(np.corrcoef(r[:-1], r[1:])[0, 1])
        else:
            ac_real = 0.0
        if np.std(g) > 1e-12 and len(g) > 2:
            ac_gen = float(np.corrcoef(g[:-1], g[1:])[0, 1])
        else:
            ac_gen = 0.0
        err[i] = abs(ac_real - ac_gen)
    return {"per_station": err, "mean": float(err.mean())}


# ──────────────────────────────────────────────────────────
# MÉTRICAS DE QUALIDADE
# ──────────────────────────────────────────────────────────

def wasserstein_per_station(real: np.ndarray, generated: np.ndarray) -> dict:
    """
    Distância Wasserstein-1 (Earth Mover's Distance) por estação.

    Mede quão diferente é a distribuição marginal de cada estação.
    Menor = melhor.

    Args:
        real:      (N_real, S) — dados reais em mm/dia
        generated: (N_gen, S) — dados gerados em mm/dia

    Returns:
        dict com 'per_station' (array S) e 'mean' (escalar)
    """
    S = real.shape[1]
    w = np.zeros(S)
    for i in range(S):
        w[i] = stats.wasserstein_distance(real[:, i], generated[:, i])
    return {'per_station': w, 'mean': float(w.mean())}


def correlation_rmse(real: np.ndarray, generated: np.ndarray) -> float:
    """
    RMSE entre matrizes de correlação espacial (real vs gerada).

    Uma boa geração deve preservar a estrutura de correlação entre estações.
    Menor = melhor.

    Returns:
        float — RMSE (raiz do erro quadrático médio entre as correlações)
    """
    # Estações com variância ~zero tornam a correlação indefinida (NaN).
    std_real = np.std(real, axis=0)
    std_gen = np.std(generated, axis=0)
    valid = (std_real > 1e-12) & (std_gen > 1e-12)

    if valid.sum() < 2:
        return float("nan")

    cx = np.corrcoef(real[:, valid], rowvar=False)
    cy = np.corrcoef(generated[:, valid], rowvar=False)
    diff = cx - cy
    return float(np.sqrt(np.nanmean(diff ** 2)))


def wet_day_frequency_error(real: np.ndarray, generated: np.ndarray) -> dict:
    """
    Erro absoluto na frequência de dias chuvosos por estação.

    |P(precip_real > threshold) - P(precip_gerado > threshold)| por estação.
    Menor = melhor.

    Returns:
        dict com 'per_station' (array S) e 'mean' (escalar)
    """
    p_real = (real > WET_DAY_THRESHOLD_MM).mean(axis=0)
    p_gen = (generated > WET_DAY_THRESHOLD_MM).mean(axis=0)
    err = np.abs(p_real - p_gen)
    return {'per_station': err, 'mean': float(err.mean())}


def extreme_quantile_error(real: np.ndarray, generated: np.ndarray,
                           quantiles=(0.90, 0.95, 0.99)) -> dict:
    """
    Erro absoluto nos quantis extremos por estação.

    Captura se o modelo reproduz eventos raros (importância para gestão de riscos).
    Menor = melhor.

    Returns:
        dict com:
            'q{q}': {'per_station': array S, 'mean': float}
            'mean_all': média sobre todos os quantis e estações
    """
    S = real.shape[1]
    results = {}
    all_errors = []

    for q in quantiles:
        qr = np.nanquantile(real, q, axis=0)
        qg = np.nanquantile(generated, q, axis=0)
        err = np.abs(qr - qg)
        key = f'q{int(q*100)}'
        results[key] = {'per_station': err, 'mean': float(err.mean())}
        all_errors.append(err)

    results['mean_all'] = float(np.mean(all_errors))
    return results


def _mean_pairwise_l2(a: np.ndarray, b: np.ndarray) -> float:
    """
    Distância euclidiana média entre pares de linhas de a e b.
    """
    diff = a[:, None, :] - b[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    return float(np.mean(d))


def energy_score(real: np.ndarray, generated: np.ndarray, n_pairs: int = 1000) -> float:
    """
    Escore de energia multivariado (regra de pontuação própria).

    ES(P, x) = E||Y - x|| - 0.5 * E||Y - Y'||
    onde Y, Y' ~ P são amostras independentes da distribuição gerada.

    Estima com média de distâncias pareadas em subamostras fixas.
    Menor = melhor.

    Args:
        n_pairs: tamanho da subamostra para estimar as expectâncias

    Returns:
        float — escore de energia (idealmente não-negativo; menor é melhor)
    """
    N_real = real.shape[0]
    N_gen = generated.shape[0]
    n_pairs = min(n_pairs, N_real, N_gen)
    if n_pairs < 2:
        return float("nan")

    idx_real = np.random.choice(N_real, n_pairs, replace=False)
    idx_gen = np.random.choice(N_gen, n_pairs, replace=False)

    x = real[idx_real]         # (n_pairs, S)
    y = generated[idx_gen]     # (n_pairs, S)

    term1 = _mean_pairwise_l2(x, y)
    term2 = 0.5 * _mean_pairwise_l2(y, y)
    # Pequenas violações negativas podem ocorrer numericamente; truncamos em 0.
    return float(max(0.0, term1 - term2))


def coverage_test(
    real: np.ndarray,
    generated: np.ndarray,
    alphas: tuple = (0.80, 0.90, 0.95),
) -> dict:
    """
    Teste de cobertura dos intervalos preditivos do modelo gerador.

    Para cada nível alpha, calcula um intervalo de credibilidade simétrico
    baseado nas amostras geradas e mede qual fração das observações reais
    cai dentro desse intervalo — por estação e em média.

    Calibração ideal: se alpha=0.90, a cobertura esperada é ≈0.90.
    Cobertura < alpha → sub-dispersão (intervalos muito estreitos).
    Cobertura > alpha → sobre-dispersão (intervalos muito largos).

    Args:
        real:      (N_real, S) — dados reais em mm/dia
        generated: (N_gen, S) — amostras sintéticas em mm/dia
        alphas:    níveis de cobertura a testar (default: 0.80, 0.90, 0.95)

    Returns:
        dict com uma entrada por alpha:
            {'coverage_80': {'per_station': array S, 'mean': float, 'ideal': 0.80, 'error': float},
             'coverage_90': ...,
             'coverage_95': ...}
    """
    S = real.shape[1]
    results = {}

    for alpha in alphas:
        lo_q = (1.0 - alpha) / 2.0        # ex: alpha=0.90 → lo=0.05
        hi_q = 1.0 - lo_q                  # ex: alpha=0.90 → hi=0.95

        lo_bound = np.quantile(generated, lo_q, axis=0)  # (S,)
        hi_bound = np.quantile(generated, hi_q, axis=0)  # (S,)

        covered = np.zeros(S)
        for i in range(S):
            covered[i] = float(np.mean(
                (real[:, i] >= lo_bound[i]) & (real[:, i] <= hi_bound[i])
            ))

        key = f'coverage_{int(alpha * 100)}'
        results[key] = {
            'per_station': covered,
            'mean': float(covered.mean()),
            'ideal': alpha,
            'error': float(abs(covered.mean() - alpha)),
        }

    return results


# ──────────────────────────────────────────────────────────
# MÉTRICAS DE PERFORMANCE
# ──────────────────────────────────────────────────────────

def sampling_time_ms(model, n_samples: int = 1000, n_trials: int = 5) -> float:
    """
    Tempo mediano para gerar n_samples amostras (ms).

    Repete n_trials vezes e usa a mediana para robustez.
    """
    times = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        with torch.no_grad():
            _ = model.sample(n_samples)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    return float(np.median(times))


# ──────────────────────────────────────────────────────────
# AVALIAÇÃO COMPLETA
# ──────────────────────────────────────────────────────────

def evaluate_model(
    model,
    data_raw: np.ndarray,
    mu: np.ndarray,
    std: np.ndarray,
    n_samples: int = 5000,
    station_names: list = None,
    samples_are_normalized: bool = True,
    timing_n_samples: int = 1000,
    timing_n_trials: int = 5,
) -> dict:
    """
    Avalia um modelo treinado computando todas as métricas.

    Args:
        model: instância de BaseModel (já treinado / ajustado)
        data_raw: (N, S) — dados reais em mm/dia (sem normalização)
        mu, std: parâmetros de normalização para denormalizar amostras
        n_samples: quantas amostras sintéticas gerar
        station_names: nomes das estações para o relatório
        samples_are_normalized: se True, aplica denormalização com (mu, std)

    Returns:
        dict com todas as métricas
    """
    S = data_raw.shape[1]
    if station_names is None:
        station_names = [f'st{i}' for i in range(S)]

    # Gera amostras e denormaliza
    with torch.no_grad():
        samples_norm = model.sample(n_samples)
    if isinstance(samples_norm, torch.Tensor):
        samples_norm = samples_norm.cpu().numpy()
    if samples_are_normalized:
        generated = samples_norm * std + mu
    else:
        generated = samples_norm

    # Garante não-negatividade (precipitação física)
    generated = np.clip(generated, 0, None)

    # --- Métricas ---
    w = wasserstein_per_station(data_raw, generated)
    corr = correlation_rmse(data_raw, generated)
    wet = wet_day_frequency_error(data_raw, generated)
    quant = extreme_quantile_error(data_raw, generated)
    es = energy_score(data_raw, generated)
    coverage = coverage_test(data_raw, generated)
    t_ms = sampling_time_ms(model, n_samples=timing_n_samples, n_trials=timing_n_trials)
    n_params = model.count_parameters()

    # Métricas temporais (informativas — não afetam composite score)
    wet_spell = wet_spell_length_error(data_raw, generated)
    dry_spell = dry_spell_length_error(data_raw, generated)
    ac_err = lag1_autocorr_error(data_raw, generated)

    metrics = {
        'mean_wasserstein': w['mean'],
        'wasserstein_per_station': {station_names[i]: float(w['per_station'][i]) for i in range(S)},
        'corr_rmse': corr,
        'wet_day_freq_error_mean': wet['mean'],
        'wet_day_freq_error_per_station': {station_names[i]: float(wet['per_station'][i]) for i in range(S)},
        'extreme_q90_mean': quant['q90']['mean'],
        'extreme_q95_mean': quant['q95']['mean'],
        'extreme_q99_mean': quant['q99']['mean'],
        'energy_score': es,
        'sampling_time_ms': t_ms,
        'n_parameters': n_params,
        # Métricas temporais
        'wet_spell_length_error': wet_spell['mean'],
        'dry_spell_length_error': dry_spell['mean'],
        'lag1_autocorr_error': ac_err['mean'],
        # Cobertura dos intervalos preditivos (ideal: coverage_xx ≈ xx/100)
        'coverage_80': coverage['coverage_80']['mean'],
        'coverage_80_error': coverage['coverage_80']['error'],
        'coverage_90': coverage['coverage_90']['mean'],
        'coverage_90_error': coverage['coverage_90']['error'],
        'coverage_95': coverage['coverage_95']['mean'],
        'coverage_95_error': coverage['coverage_95']['error'],
    }

    _print_report(metrics, station_names)
    return metrics



# ──────────────────────────────────────────────────────────
# TEMPORAL METRICS (scenario-level: (n_sc, T, S) vs (N, S))
# ──────────────────────────────────────────────────────────

def _autocorr_1d(series: np.ndarray, max_lag: int = 30) -> np.ndarray:
    """Autocorrelação lag-1..max_lag de uma série 1-D."""
    n = len(series)
    mu = series.mean()
    var = ((series - mu) ** 2).mean()
    if var < 1e-10:
        return np.zeros(max_lag)
    return np.array([
        np.mean((series[:n - lag] - mu) * (series[lag:] - mu)) / var
        for lag in range(1, max_lag + 1)
    ])


def _mean_autocorr(data: np.ndarray, max_lag: int = 30) -> np.ndarray:
    """Autocorrelação média sobre as estações. data: (T, S)"""
    return np.stack([_autocorr_1d(data[:, s], max_lag) for s in range(data.shape[1])]).mean(axis=0)


def _acf_batched(data3d: np.ndarray, max_lag: int) -> np.ndarray:
    """data3d: (B, T, S) → ACF averaged over S, shape (B, max_lag).

    Vectorized over batch (B) and stations (S); only loops over max_lag=30 lags.
    Replaces B*S calls to _autocorr_1d with 30 array multiplications.
    """
    B, T, S = data3d.shape
    mu  = data3d.mean(axis=1, keepdims=True)          # (B, 1, S)
    x   = data3d - mu                                 # (B, T, S) — zero-mean
    var = (x ** 2).mean(axis=1)                       # (B, S)
    acfs = np.zeros((B, max_lag))
    for lag in range(1, max_lag + 1):
        cov = (x[:, :-lag, :] * x[:, lag:, :]).mean(axis=1)   # (B, S)
        acfs[:, lag - 1] = (cov / np.maximum(var, 1e-12)).mean(axis=1)
    return acfs


def multi_lag_autocorr_rmse(
    scenarios: np.ndarray,
    observed: np.ndarray,
    max_lag: int = 30,
) -> float:
    """
    RMSE da autocorrelação média (lags 1..max_lag) entre cenários e observado.

    Args:
        scenarios: (n_sc, T, S) — cenários em mm/dia
        observed:  (N, S) — dados observados em mm/dia

    Returns:
        float — RMSE médio sobre os lags e estações
    """
    obs_acf     = _mean_autocorr(observed, max_lag)             # (max_lag,)
    sc_mean_acf = _acf_batched(scenarios, max_lag).mean(axis=0) # (max_lag,)
    return float(np.sqrt(np.mean((obs_acf - sc_mean_acf) ** 2)))


def transition_probability_error(
    scenarios: np.ndarray,
    observed: np.ndarray,
    threshold: float = 0.1,
) -> dict:
    """
    Erro nas probabilidades de transição 2×2 (wet/dry) entre cenários e observado.

    Returns:
        dict com 'p_ww', 'p_wd', 'p_dw', 'p_dd' (erro absoluto médio por estação)
        e 'mean' (média dos 4 erros)
    """
    def _trans(data: np.ndarray):
        """Calcula P(wet|wet), P(wet|dry), P(dry|wet), P(dry|dry) por estação."""
        S = data.shape[1]
        p_ww = np.zeros(S); p_wd = np.zeros(S)
        p_dw = np.zeros(S); p_dd = np.zeros(S)
        for i in range(S):
            wet = (data[:, i] > threshold).astype(float)
            n_w = wet[:-1].sum(); n_d = (1 - wet[:-1]).sum()
            p_ww[i] = (wet[:-1] * wet[1:]).sum() / max(n_w, 1)
            p_wd[i] = (wet[:-1] * (1 - wet[1:])).sum() / max(n_w, 1)
            p_dw[i] = ((1 - wet[:-1]) * wet[1:]).sum() / max(n_d, 1)
            p_dd[i] = ((1 - wet[:-1]) * (1 - wet[1:])).sum() / max(n_d, 1)
        return p_ww, p_wd, p_dw, p_dd

    obs_ww, obs_wd, obs_dw, obs_dd = _trans(observed)

    # Average transition probs across scenarios — vectorized over all n_sc at once
    wet_sc = (scenarios > threshold).astype(float)        # (n_sc, T, S)
    n_w_sc = wet_sc[:, :-1, :].sum(axis=1)                # (n_sc, S)
    n_d_sc = (1.0 - wet_sc[:, :-1, :]).sum(axis=1)
    sc_ww = ((wet_sc[:, :-1, :] * wet_sc[:, 1:, :]).sum(axis=1)
             / np.maximum(n_w_sc, 1)).mean(axis=0)
    sc_wd = ((wet_sc[:, :-1, :] * (1.0 - wet_sc[:, 1:, :])).sum(axis=1)
             / np.maximum(n_w_sc, 1)).mean(axis=0)
    sc_dw = (((1.0 - wet_sc[:, :-1, :]) * wet_sc[:, 1:, :]).sum(axis=1)
             / np.maximum(n_d_sc, 1)).mean(axis=0)
    sc_dd = (((1.0 - wet_sc[:, :-1, :]) * (1.0 - wet_sc[:, 1:, :])).sum(axis=1)
             / np.maximum(n_d_sc, 1)).mean(axis=0)

    e_ww = float(np.abs(obs_ww - sc_ww).mean())
    e_wd = float(np.abs(obs_wd - sc_wd).mean())
    e_dw = float(np.abs(obs_dw - sc_dw).mean())
    e_dd = float(np.abs(obs_dd - sc_dd).mean())

    return {
        'p_ww': e_ww, 'p_wd': e_wd, 'p_dw': e_dw, 'p_dd': e_dd,
        'mean': float(np.mean([e_ww, e_wd, e_dw, e_dd])),
    }


def max_consecutive_dry_days_error(
    scenarios: np.ndarray,
    observed: np.ndarray,
    threshold: float = 0.1,
) -> float:
    """
    Erro no número médio de dias secos consecutivos máximos por estação.

    Returns:
        float — erro absoluto médio sobre estações
    """
    def _max_cdd(data: np.ndarray) -> np.ndarray:
        """data: (T, S) → max consecutive dry days per station via diff-based RLE."""
        dry = (data <= threshold)
        T, S = dry.shape
        pad = np.zeros((1, S), dtype=bool)
        padded = np.concatenate([pad, dry, pad], axis=0)  # (T+2, S)
        changes = np.diff(padded.astype(np.int8), axis=0)  # (T+1, S)
        result = np.zeros(S, dtype=float)
        for s in range(S):
            starts = np.where(changes[:, s] == 1)[0]
            ends   = np.where(changes[:, s] == -1)[0]
            if len(starts) > 0:
                result[s] = float((ends - starts).max())
        return result

    obs_cdd = _max_cdd(observed)
    sc_cdd = np.mean([_max_cdd(scenarios[i]) for i in range(scenarios.shape[0])], axis=0)
    return float(np.abs(obs_cdd - sc_cdd).mean())


def annual_max_daily_error(
    scenarios: np.ndarray,
    observed: np.ndarray,
) -> float:
    """
    Erro no máximo diário anual médio por estação.

    Returns:
        float — erro absoluto médio sobre estações
    """
    obs_max = observed.max(axis=0)  # (S,)
    sc_max = np.mean([scenarios[i].max(axis=0) for i in range(scenarios.shape[0])], axis=0)
    return float(np.abs(obs_max - sc_max).mean())


def monthly_mean_error(
    scenarios: np.ndarray,
    observed: np.ndarray,
    obs_months: np.ndarray,
    start_month: int = 0,
) -> float:
    """
    Erro na precipitação média mensal (mm/dia), média sobre estações e meses.

    Args:
        scenarios:   (n_sc, T, S) — cenários
        observed:    (N, S) — dados observados
        obs_months:  (N,) — mês (0..11) para cada linha de observed
        start_month: mês inicial dos cenários (0=Jan) para atribuição cíclica

    Returns:
        float — MAE médio sobre meses e estações
    """
    obs_mm = np.array([observed[obs_months == m].mean(axis=0) if (obs_months == m).any()
                       else np.zeros(observed.shape[1]) for m in range(12)])  # (12, S)

    T = scenarios.shape[1]
    sc_months = np.array([(start_month + t) % 12 for t in range(T)])

    sc_mm_all = []
    for i in range(scenarios.shape[0]):
        sc_mm_all.append(np.array([scenarios[i][sc_months == m].mean(axis=0) if (sc_months == m).any()
                                   else np.zeros(observed.shape[1]) for m in range(12)]))
    sc_mm = np.mean(sc_mm_all, axis=0)  # (12, S)

    return float(np.abs(obs_mm - sc_mm).mean())


def monthly_variance_error(
    scenarios: np.ndarray,
    observed: np.ndarray,
    obs_months: np.ndarray,
    start_month: int = 0,
) -> float:
    """
    Erro na variância mensal da precipitação, média sobre estações e meses.

    Returns:
        float — MAE médio sobre meses e estações
    """
    obs_var = np.array([observed[obs_months == m].var(axis=0) if (obs_months == m).any()
                        else np.zeros(observed.shape[1]) for m in range(12)])  # (12, S)

    T = scenarios.shape[1]
    sc_months = np.array([(start_month + t) % 12 for t in range(T)])

    sc_var_all = []
    for i in range(scenarios.shape[0]):
        sc_var_all.append(np.array([scenarios[i][sc_months == m].var(axis=0) if (sc_months == m).any()
                                    else np.zeros(observed.shape[1]) for m in range(12)]))
    sc_var = np.mean(sc_var_all, axis=0)  # (12, S)

    return float(np.abs(obs_var - sc_var).mean())


def rx5day_distribution_error(
    scenarios: np.ndarray,
    observed: np.ndarray,
    window: int = 5,
) -> float:
    """
    Wasserstein-1 entre a distribuição do acumulado máximo em `window` dias
    consecutivos (Rxnday) dos cenários vs. observado, médio sobre estações.

    Captura se o modelo reproduz bem a cauda de eventos extremos acumulados
    (e.g. cheias multi-dia), seguindo a recomendação ETCCDI de índices climáticos.

    Args:
        scenarios:  (n_sc, T, S) — cenários em mm/dia
        observed:   (N, S) — dados observados em mm/dia
        window:     tamanho da janela de acumulação em dias (default 5 = Rx5day)

    Returns:
        float — Wasserstein-1 médio sobre estações
    """
    n_sc, T, S = scenarios.shape
    errors = []

    # Precompute cumsum for vectorized rolling sum (avoids n_sc np.convolve calls per station)
    cs_sc  = np.cumsum(scenarios, axis=1)  # (n_sc, T, S)
    cs_obs = np.cumsum(observed,  axis=0)  # (N, S)

    for s in range(S):
        obs_rx = cs_obs[window:, s] - cs_obs[:-window, s]          # (N-window+1,)
        sc_rx  = (cs_sc[:, window:, s] - cs_sc[:, :-window, s]).ravel()  # (n_sc*(T-window+1),)
        errors.append(stats.wasserstein_distance(obs_rx, sc_rx))

    return float(np.mean(errors))


def seasonal_accumulation_error(
    scenarios: np.ndarray,
    observed: np.ndarray,
    obs_months: np.ndarray,
    wet_months: tuple = (10, 11, 0, 1, 2, 3),
    start_month: int = 0,
) -> dict:
    """
    Wasserstein-1 entre a distribuição dos totais mensais de chuva na estação
    chuvosa (Nov–Abr) e seca (Mai–Out), médio sobre estações.

    Captura se o modelo reproduz corretamente o volume acumulado sazonal —
    relevante para gestão de reservatórios.

    Args:
        scenarios:   (n_sc, T, S) — cenários em mm/dia
        observed:    (N, S) — dados observados em mm/dia
        obs_months:  (N,) mês 0-indexado (0=Jan .. 11=Dez) para cada linha de observed
        wet_months:  meses da estação chuvosa (default Nov–Abr, 0-indexado)
        start_month: mês inicial dos cenários (0=Jan)

    Returns:
        dict com 'wet_season_error' e 'dry_season_error' (floats)
    """
    dry_months = tuple(m for m in range(12) if m not in wet_months)
    n_sc, T, S = scenarios.shape
    MONTH_DAYS = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    _ml, _m, _d = [], start_month, 0
    for _t in range(T):
        _ml.append(_m)
        _d += 1
        if _d >= MONTH_DAYS[_m % 12]:
            _m = (_m + 1) % 12
            _d = 0
    sc_months = np.array(_ml)

    def _monthly_totals_2d(data: np.ndarray, months_mask: np.ndarray, season: tuple) -> np.ndarray:
        """data: (T, S) → (n_runs, S)"""
        totals = []
        for m in season:
            indices = np.where(months_mask == m)[0]
            if len(indices) == 0:
                continue
            breaks = np.where(np.diff(indices) > 1)[0] + 1
            for run in np.split(indices, breaks):
                totals.append(data[run].sum(axis=0))  # (S,)
        return np.array(totals) if totals else np.empty((0, data.shape[1]))

    def _monthly_totals_batched(data3d: np.ndarray, months_mask: np.ndarray, season: tuple) -> np.ndarray:
        """data3d: (n_sc, T, S) → (n_runs, n_sc, S) — all scenarios at once."""
        totals = []
        for m in season:
            indices = np.where(months_mask == m)[0]
            if len(indices) == 0:
                continue
            breaks = np.where(np.diff(indices) > 1)[0] + 1
            for run in np.split(indices, breaks):
                totals.append(data3d[:, run, :].sum(axis=1))  # (n_sc, S)
        if not totals:
            return np.empty((0, data3d.shape[0], data3d.shape[2]))
        return np.stack(totals, axis=0)  # (n_runs, n_sc, S)

    obs_wet = _monthly_totals_2d(observed, obs_months, wet_months)  # (n_runs, S)
    obs_dry = _monthly_totals_2d(observed, obs_months, dry_months)

    # Vectorize over all n_sc scenarios at once — eliminates the n_sc inner loop
    sc_wet_batched = _monthly_totals_batched(scenarios, sc_months, wet_months)  # (n_runs, n_sc, S)
    sc_dry_batched = _monthly_totals_batched(scenarios, sc_months, dry_months)

    wet_errors, dry_errors = [], []
    for s in range(S):
        # Flatten (n_runs, n_sc) → 1-D pool of all scenario monthly totals for station s
        sc_wet = sc_wet_batched[:, :, s].ravel() if sc_wet_batched.shape[0] > 0 else np.array([])
        sc_dry = sc_dry_batched[:, :, s].ravel() if sc_dry_batched.shape[0] > 0 else np.array([])

        if obs_wet.shape[0] > 0 and len(sc_wet) > 0:
            wet_errors.append(stats.wasserstein_distance(obs_wet[:, s], sc_wet))
        if obs_dry.shape[0] > 0 and len(sc_dry) > 0:
            dry_errors.append(stats.wasserstein_distance(obs_dry[:, s], sc_dry))

    return {
        "wet_season_error": float(np.mean(wet_errors)) if wet_errors else float("nan"),
        "dry_season_error": float(np.mean(dry_errors)) if dry_errors else float("nan"),
    }


def inter_scenario_cv(scenarios: np.ndarray) -> float:
    """
    Coeficiente de variação entre cenários (diversidade).

    CV = std(média_diária_por_cenário) / mean(média_diária_por_cenário)
    Maior = mais diverso.

    Args:
        scenarios: (n_sc, T, S) — cenários em mm/dia

    Returns:
        float — CV (adimensional)
    """
    sc_means = scenarios.mean(axis=(1, 2))  # (n_sc,)
    mu = sc_means.mean()
    if mu < 1e-10:
        return 0.0
    return float(sc_means.std() / mu)


def _print_report(metrics: dict, station_names: list):
    """Imprime relatório de métricas formatado."""
    print("\n" + "=" * 55)
    print("  MÉTRICAS DE AVALIAÇÃO")
    print("=" * 55)
    print(f"  Wasserstein médio:        {metrics['mean_wasserstein']:.4f}")
    print(f"  Correlation RMSE:         {metrics['corr_rmse']:.4f}")
    print(f"  Wet day freq error:       {metrics['wet_day_freq_error_mean']:.4f}")
    print(f"  Quantil 90% erro:         {metrics['extreme_q90_mean']:.4f}")
    print(f"  Quantil 95% erro:         {metrics['extreme_q95_mean']:.4f}")
    print(f"  Quantil 99% erro:         {metrics['extreme_q99_mean']:.4f}")
    print(f"  Energy score:             {metrics['energy_score']:.4f}")
    print(f"  --- Cobertura dos intervalos preditivos (ideal = alpha) ---")
    cov80 = metrics.get('coverage_80', float('nan'))
    cov90 = metrics.get('coverage_90', float('nan'))
    cov95 = metrics.get('coverage_95', float('nan'))
    err80 = metrics.get('coverage_80_error', float('nan'))
    err90 = metrics.get('coverage_90_error', float('nan'))
    err95 = metrics.get('coverage_95_error', float('nan'))
    print(f"  Coverage 80% (ideal=0.80): {cov80:.4f}  |  erro={err80:.4f}")
    print(f"  Coverage 90% (ideal=0.90): {cov90:.4f}  |  erro={err90:.4f}")
    print(f"  Coverage 95% (ideal=0.95): {cov95:.4f}  |  erro={err95:.4f}")
    print(f"  --- Métricas temporais ---")
    print(f"  Wet spell length error:   {metrics.get('wet_spell_length_error', float('nan')):.4f}")
    print(f"  Dry spell length error:   {metrics.get('dry_spell_length_error', float('nan')):.4f}")
    print(f"  Lag-1 autocorr error:     {metrics.get('lag1_autocorr_error', float('nan')):.4f}")
    print(f"  --- Performance ---")
    print(f"  Tempo sampling (ms):      {metrics['sampling_time_ms']:.1f}")
    print(f"  Parâmetros:               {metrics['n_parameters']:,}")
    print("=" * 55 + "\n")
