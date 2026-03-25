"""ar/ — Autoregressive model analysis sub-package.

Re-exports the complete public API so callers can use:
    from ar import load_ar_model, plot_composite_bar, ...
or
    from ar.loader import load_ar_model
    from ar.plots import plot_composite_bar
"""

from ar.loader import (
    AR_FAMILIES,
    OUTLIER_STATION,
    OUTLIER_MODELS,
    _is_ar_dir,
    discover_ar_models,
    discover_ar_models_with_checkpoints,
    get_family,
    load_all_metrics,
    _load_config,
    load_ar_model,
)
from ar.rollout import (
    TIER2_METRICS,
    COMBINED_TIER2_METRICS,
    recompute_tier1_metrics,
    run_rollout,
    compute_tier2_metrics,
)
from ar.scoring import (
    compute_combined_score,
    select_top_n_per_family,
)
from ar.plots import (
    _family_colors,
    plot_composite_bar,
    plot_radar_by_family,
    plot_pareto,
    plot_heatmap,
    plot_family_grouped_bars,
    plot_hyperparameter_sensitivity,
    plot_training_loss_overlay,
    _group_scenarios_by_family,
    _mean_autocorr_nd,
    _spell_dist,
    _monthly_means,
    plot_autocorr_multilag,
    plot_spell_length_comparison,
    plot_monthly_precip,
    plot_spread_envelopes,
    plot_transition_probs,
    plot_return_period,
    plot_rx5day_distribution,
    plot_rxnday_multi,
    plot_seasonal_accumulation,
    plot_exceedance_frequency,
    plot_family_detail_dirs,
)
from ar.station import (
    _load_station_names,
    plot_station_heatmap_ar,
    plot_station_score_distribution,
    plot_best_model_per_station,
    write_per_station_report,
    plot_station_rank_heatmap,
    plot_per_station_detail,
    plot_per_station_lag1_scatter,
    plot_per_station_wetfreq_scatter,
)
from ar.reports import (
    write_comparison_report,
    write_family_summary,
    write_hyperparameter_sensitivity_report,
)
