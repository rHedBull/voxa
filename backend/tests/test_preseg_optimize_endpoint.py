import time
import numpy as np
from fastapi.testclient import TestClient

import main
from point_cloud import PointCloud


client = TestClient(main.app)


def _load_synthetic():
    rng = np.random.default_rng(42)
    plane = rng.uniform(-1, 1, (3000, 2))
    plane = np.hstack([plane, rng.normal(0, 1e-3, (3000, 1))])
    cyl_theta = rng.uniform(0, 2 * np.pi, 3000)
    cyl = np.column_stack([
        3 + 0.3 * np.cos(cyl_theta),
        0.3 * np.sin(cyl_theta),
        rng.uniform(0, 1, 3000),
    ])
    xyz = np.vstack([plane, cyl]).astype(np.float32)
    colors = np.tile(np.array([0.5, 0.5, 0.5], dtype=np.float32), (len(xyz), 1))
    main._state["pc"] = PointCloud(points=xyz, colors=colors)
    main._state.pop("seg", None)
    main._state.pop("preseg_opt_job", None)


def _wait_for_terminal(job_id, timeout_s=10.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        r = client.get(f"/api/segment/presegment/optimize/status?job_id={job_id}")
        s = r.json()["status"]
        if s in ("aborted", "done", "error"):
            return s
        time.sleep(0.1)
    return "timeout"


def test_optimize_lifecycle_abort():
    _load_synthetic()
    r = client.post("/api/segment/presegment/optimize",
                    json={"n_trials": 20, "subsample_n": 2000})
    assert r.status_code == 200
    job_id = r.json()["job_id"]
    r = client.get(f"/api/segment/presegment/optimize/status?job_id={job_id}")
    assert r.status_code == 200
    assert r.json()["status"] in ("running", "done")
    r = client.post(f"/api/segment/presegment/optimize/abort?job_id={job_id}")
    assert r.status_code == 200
    final = _wait_for_terminal(job_id, timeout_s=30.0)
    assert final in ("aborted", "done"), f"unexpected terminal status {final}"


def test_optimize_rejects_concurrent():
    _load_synthetic()
    r1 = client.post("/api/segment/presegment/optimize",
                     json={"n_trials": 20, "subsample_n": 2000})
    assert r1.status_code == 200
    r2 = client.post("/api/segment/presegment/optimize",
                     json={"n_trials": 20, "subsample_n": 2000})
    assert r2.status_code == 409
    job_id = r1.json()["job_id"]
    client.post(f"/api/segment/presegment/optimize/abort?job_id={job_id}")
    _wait_for_terminal(job_id, timeout_s=30.0)


def test_optimize_requires_scene():
    main._state.pop("pc", None)
    main._state.pop("seg", None)
    main._state.pop("preseg_opt_job", None)
    r = client.post("/api/segment/presegment/optimize", json={})
    assert r.status_code == 409


def test_status_unknown_job_id():
    r = client.get("/api/segment/presegment/optimize/status?job_id=nope")
    assert r.status_code == 404
