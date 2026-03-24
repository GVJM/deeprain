"""Tests for compare_ar._rollout_worker parallel helper."""
import numpy as np
import pytest
from unittest.mock import patch


def test_rollout_worker_returns_none_when_rollout_fails():
    """Worker must return (variant, None) when run_rollout returns None."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from compare_ar import _rollout_worker

    data_norm = np.zeros((30, 3), dtype=np.float32)
    std = np.ones((1, 3), dtype=np.float32)
    data_raw = np.zeros((30, 3), dtype=np.float32)
    obs_months = np.arange(30) % 12

    args_tuple = (
        "ar_vae_test", "./outputs", data_norm, std, data_raw, obs_months,
        10, 5, "cpu", False,
    )

    with patch("compare_ar.run_rollout", return_value=None):
        variant, t2 = _rollout_worker(args_tuple)

    assert variant == "ar_vae_test"
    assert t2 is None


def test_rollout_worker_returns_metrics_on_success():
    """Worker must return (variant, metrics_dict) when run_rollout succeeds."""
    import sys, os
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from compare_ar import _rollout_worker

    n_scenarios, n_days, n_stations = 5, 10, 3
    fake_sc_mm = np.zeros((n_scenarios, n_days, n_stations), dtype=np.float32)
    fake_metrics = {"multi_lag_acf_rmse": 0.1, "transition_prob_error": 0.05,
                    "inter_scenario_cv": 0.3}

    data_norm = np.zeros((30, n_stations), dtype=np.float32)
    std = np.ones((1, n_stations), dtype=np.float32)
    data_raw = np.zeros((30, n_stations), dtype=np.float32)
    obs_months = np.arange(30) % 12

    args_tuple = (
        "ar_vae_test", "./outputs", data_norm, std, data_raw, obs_months,
        n_days, n_scenarios, "cpu", False,
    )

    with patch("compare_ar.run_rollout", return_value=fake_sc_mm), \
         patch("compare_ar.compute_tier2_metrics", return_value=fake_metrics):
        variant, t2 = _rollout_worker(args_tuple)

    assert variant == "ar_vae_test"
    assert t2 == fake_metrics
