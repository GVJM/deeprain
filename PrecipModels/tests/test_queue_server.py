# tests/test_queue_server.py
"""Tests for queue_server.py HTTP endpoints."""
import json
import time
from pathlib import Path

import pytest


@pytest.fixture
def queue_file(tmp_path):
    entries = [
        {
            "variant_name": "test_vae",
            "model": "vae",
            "job_type": "train",
            "gpu_tier": 1,
            "output_dir": "outputs",
            "data_path": "../dados_sabesp/dayprecip.dat",
        },
        {
            "variant_name": "test_copula",
            "model": "copula",
            "job_type": "train",
            "gpu_tier": 0,
            "output_dir": "outputs",
        },
    ]
    qfile = tmp_path / "TRAINING_QUEUE.json"
    qfile.write_text(json.dumps(entries))
    return tmp_path


@pytest.fixture
def client(queue_file):
    from queue_server import create_app
    app = create_app(
        queues_dir=str(queue_file),
        timeout_secs=30,
        machine="test_server",
        dropbox_remote="",
    )
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


def test_status_returns_all_jobs(client):
    r = client.get("/status")
    assert r.status_code == 200
    jobs = r.get_json()
    assert len(jobs) == 2
    assert all(j["status"] == "pending" for j in jobs)


def test_claim_returns_highest_priority_pending_job(client):
    r = client.get("/claim?machine=worker1&gpu_tier=1")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert "job" in data
    assert data["job"]["variant_name"] in ("test_vae", "test_copula")


def test_claim_respects_gpu_tier(client):
    r = client.get("/claim?machine=cpu_worker&gpu_tier=0")
    assert r.status_code == 200
    data = r.get_json()
    assert data["status"] == "ok"
    assert data["job"]["variant_name"] == "test_copula"


def test_claim_marks_job_as_running(client):
    client.get("/claim?machine=worker1&gpu_tier=1")
    r = client.get("/status")
    jobs = r.get_json()
    running = [j for j in jobs if j["status"] == "running"]
    assert len(running) == 1
    assert running[0]["machine"] == "worker1"


def test_complete_marks_job_done_and_inserts_eval(client):
    claim_r = client.get("/claim?machine=worker1&gpu_tier=1")
    data = claim_r.get_json()
    job = data["job"]
    queue_file_name = data.get("queue_file", "TRAINING_QUEUE.json")

    r = client.post("/complete", json={
        "variant_name": job["variant_name"],
        "queue_file": queue_file_name,
    })
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"

    status = client.get("/status").get_json()
    done = [j for j in status if j["status"] == "done"]
    assert any(j["variant_name"] == job["variant_name"] for j in done)
    pending = [j for j in status if j["status"] == "pending"]
    eval_jobs = [j for j in pending if j["variant_name"].endswith("__eval")]
    assert len(eval_jobs) == 1


def test_release_returns_job_to_pending(client):
    claim_r = client.get("/claim?machine=worker1&gpu_tier=1")
    data = claim_r.get_json()
    job = data["job"]
    queue_file_name = data.get("queue_file", "TRAINING_QUEUE.json")

    r = client.post("/release", json={
        "variant_name": job["variant_name"],
        "queue_file": queue_file_name,
    })
    assert r.status_code == 200

    status = client.get("/status").get_json()
    released = next(j for j in status if j["variant_name"] == job["variant_name"])
    assert released["status"] == "pending"
    assert released["machine"] is None


def test_claim_returns_idle_when_no_matching_jobs(client):
    client.get("/claim?machine=w1&gpu_tier=0")
    r = client.get("/claim?machine=w2&gpu_tier=0")
    data = r.get_json()
    assert data["status"] == "idle"


def test_heartbeat_updates_timestamp(client):
    claim_r = client.get("/claim?machine=worker1&gpu_tier=1")
    data = claim_r.get_json()
    job = data["job"]
    queue_file_name = data.get("queue_file", "TRAINING_QUEUE.json")

    r = client.post("/heartbeat", json={
        "variant_name": job["variant_name"],
        "queue_file": queue_file_name,
    })
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"
