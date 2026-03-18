# tests/test_make_queue.py
import json, subprocess, sys
from pathlib import Path
import pytest

@pytest.fixture
def dat_file(tmp_path):
    dat = tmp_path / "test.dat"
    dat.write_text("date,s1,s2,s3,s4,s5\n2020-01-01,0,1,0,0,0\n")
    return dat

def test_gpu_tier_default_is_1(dat_file, tmp_path):
    out = tmp_path / "queue.json"
    subprocess.run([sys.executable, "make_queue.py", "--data_path", str(dat_file),
                    "--prefix", "t", "--output", str(out)], check=True)
    entries = json.loads(out.read_text())
    assert all(e.get("gpu_tier", 1) == 1 for e in entries)

def test_gpu_tier_flag_sets_value(dat_file, tmp_path):
    out = tmp_path / "queue.json"
    subprocess.run([sys.executable, "make_queue.py", "--data_path", str(dat_file),
                    "--prefix", "t", "--gpu_tier", "2", "--output", str(out)], check=True)
    entries = json.loads(out.read_text())
    assert all(e["gpu_tier"] == 2 for e in entries)

def test_output_dir_default_is_outputs(dat_file, tmp_path):
    out = tmp_path / "queue.json"
    subprocess.run([sys.executable, "make_queue.py", "--data_path", str(dat_file),
                    "--prefix", "t", "--output", str(out)], check=True)
    entries = json.loads(out.read_text())
    assert all(e.get("output_dir", "outputs") == "outputs" for e in entries)

def test_output_dir_flag_sets_value(dat_file, tmp_path):
    out = tmp_path / "queue.json"
    subprocess.run([sys.executable, "make_queue.py", "--data_path", str(dat_file),
                    "--prefix", "t", "--output_dir", "outputs_sabesp", "--output", str(out)], check=True)
    entries = json.loads(out.read_text())
    assert all(e["output_dir"] == "outputs_sabesp" for e in entries)

def test_job_type_is_train(dat_file, tmp_path):
    out = tmp_path / "queue.json"
    subprocess.run([sys.executable, "make_queue.py", "--data_path", str(dat_file),
                    "--prefix", "t", "--output", str(out)], check=True)
    entries = json.loads(out.read_text())
    assert all(e.get("job_type", "train") == "train" for e in entries)
