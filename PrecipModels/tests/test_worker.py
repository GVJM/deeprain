# tests/test_worker.py
"""Tests for worker.py path construction and crash recovery logic."""
import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture
def worker_args(tmp_path):
    from argparse import Namespace
    return Namespace(
        server="http://localhost:5000",
        machine="test_pc",
        gpu_tier=1,
        poll_interval=1,
        output_base=str(tmp_path),
        dropbox_remote="",
        config=None,
    )


def test_output_path_construction(worker_args, tmp_path):
    """local_path = Path(output_base) / output_dir / variant_name"""
    from worker import _local_output_path
    job = {"variant_name": "ar_vae", "output_dir": "outputs_sabesp", "job_type": "train"}
    result = _local_output_path(job, worker_args.output_base)
    expected = tmp_path / "outputs_sabesp" / "ar_vae"
    assert result == expected


def test_eval_output_path_uses_model_dir(worker_args, tmp_path):
    """Eval jobs use model_dir last component for the output path."""
    from worker import _local_output_path
    job = {
        "variant_name": "ar_vae__eval",
        "job_type": "eval",
        "model_dir": "outputs_sabesp/ar_vae",
        "output_dir": "outputs_sabesp",
    }
    result = _local_output_path(job, worker_args.output_base)
    expected = tmp_path / "outputs_sabesp" / "ar_vae"
    assert result == expected


def test_crash_recovery_train_skips_if_model_exists(worker_args, tmp_path):
    """If model.pt exists, train step is skipped (already trained before crash)."""
    from worker import run_job
    job = {
        "variant_name": "test_model",
        "model": "vae",
        "job_type": "train",
        "output_dir": "outputs",
    }
    out_dir = tmp_path / "outputs" / "test_model"
    out_dir.mkdir(parents=True)
    (out_dir / "model.pt").write_bytes(b"fake")

    with patch("worker.subprocess.run") as mock_run:
        success = run_job(job, worker_args)

    assert success is True
    mock_run.assert_not_called()


def test_crash_recovery_eval_skips_if_metrics_newer_than_model(worker_args, tmp_path):
    """If metrics.json is newer than model.pt, eval step is skipped."""
    from worker import run_job
    job = {
        "variant_name": "test_model__eval",
        "job_type": "eval",
        "model_dir": "outputs/test_model",
        "output_dir": "outputs",
    }
    out_dir = tmp_path / "outputs" / "test_model"
    out_dir.mkdir(parents=True)
    model_pt = out_dir / "model.pt"
    metrics_json = out_dir / "metrics.json"
    model_pt.write_bytes(b"fake_model")
    time.sleep(0.05)
    metrics_json.write_text('{"ks_stat": 0.1}')

    with patch("worker.subprocess.run") as mock_run:
        success = run_job(job, worker_args)

    assert success is True
    mock_run.assert_not_called()


def test_train_job_calls_train_py_with_skip_eval(worker_args, tmp_path):
    """Train job must invoke train.py --skip_eval."""
    from worker import run_job
    job = {
        "variant_name": "new_vae",
        "model": "vae",
        "job_type": "train",
        "output_dir": "outputs",
        "max_epochs": 5,
    }
    with patch("worker.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        run_job(job, worker_args)

    call_args = mock_run.call_args[0][0]
    assert "--skip_eval" in call_args
    assert "train.py" in " ".join(call_args)


def test_eval_job_calls_evaluate_py(worker_args, tmp_path):
    """Eval job must invoke evaluate.py."""
    from worker import run_job
    job = {
        "variant_name": "new_vae__eval",
        "job_type": "eval",
        "model_dir": "outputs/new_vae",
        "output_dir": "outputs",
    }
    with patch("worker.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(returncode=0)
        run_job(job, worker_args)

    call_args = mock_run.call_args[0][0]
    assert "evaluate.py" in " ".join(call_args)
