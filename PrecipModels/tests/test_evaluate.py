# tests/test_evaluate.py
"""Test that evaluate.py loads a checkpoint and writes metrics.json."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_evaluate_writes_metrics(tmp_path):
    """evaluate.py must write metrics.json given a trained model dir.
    Uses hurdle_simple (not copula) — copula saves copula.pkl, not model.pt.
    """
    result = subprocess.run(
        [
            sys.executable, "train.py",
            "--model", "hurdle_simple",
            "--name", "test_eval_hurdle",
            "--output_dir", str(tmp_path),
            "--max_epochs", "2",
            "--skip_eval",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"train.py setup failed:\n{result.stderr}"
    model_dir = tmp_path / "test_eval_hurdle"
    assert (model_dir / "config.json").exists()
    assert (model_dir / "model.pt").exists(), "hurdle_simple must produce model.pt"
    assert not (model_dir / "metrics.json").exists()

    result = subprocess.run(
        [sys.executable, "evaluate.py", "--model_dir", str(model_dir), "--n_samples", "50"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"evaluate.py failed:\n{result.stderr}"
    assert (model_dir / "metrics.json").exists(), "metrics.json must be written by evaluate.py"

    metrics = json.loads((model_dir / "metrics.json").read_text())
    assert len(metrics) > 0


def test_evaluate_respects_n_samples_override(tmp_path):
    """evaluate.py --n_samples override must be accepted without error."""
    subprocess.run(
        [sys.executable, "train.py", "--model", "hurdle_simple",
         "--name", "test_eval_ns", "--output_dir", str(tmp_path),
         "--max_epochs", "2", "--skip_eval"],
        capture_output=True,
    )
    model_dir = tmp_path / "test_eval_ns"
    result = subprocess.run(
        [sys.executable, "evaluate.py", "--model_dir", str(model_dir), "--n_samples", "50"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"evaluate.py --n_samples failed:\n{result.stderr}"
    assert (model_dir / "metrics.json").exists()
