# tests/test_train_skip_eval.py
"""Test that --skip_eval prevents metrics.json from being written."""
import json
import subprocess
import sys
from pathlib import Path

import pytest


def test_skip_eval_does_not_write_metrics(tmp_path):
    """Running train.py --skip_eval must NOT create metrics.json."""
    result = subprocess.run(
        [
            sys.executable, "train.py",
            "--model", "copula",
            "--name", "test_skip_eval_copula",
            "--output_dir", str(tmp_path),
            "--skip_eval",
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"train.py failed:\n{result.stderr}"
    out_dir = tmp_path / "test_skip_eval_copula"
    assert not (out_dir / "metrics.json").exists(), "metrics.json must not exist when --skip_eval is set"


def test_without_skip_eval_writes_metrics(tmp_path):
    """Running train.py without --skip_eval MUST create metrics.json."""
    result = subprocess.run(
        [
            sys.executable, "train.py",
            "--model", "copula",
            "--name", "test_no_skip_eval_copula",
            "--output_dir", str(tmp_path),
        ],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, f"train.py failed:\n{result.stderr}"
    out_dir = tmp_path / "test_no_skip_eval_copula"
    assert (out_dir / "metrics.json").exists(), "metrics.json must exist when --skip_eval is not set"
