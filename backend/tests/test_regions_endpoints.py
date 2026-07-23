"""Tests for the /api/regions routes (eval-labeling phase 1)."""
from __future__ import annotations

import json

import numpy as np
import pytest

PRISM = {"polygon": [[-0.01, -0.01], [0.2, -0.01], [0.2, 0.2], [-0.01, 0.2]],
         "y0": -0.5, "height": 1.0}


def test_409_without_session(client):
    assert client.get("/api/regions").status_code == 409
    assert client.post("/api/regions", json={"prism": PRISM}).status_code == 409
    assert client.get("/api/regions/stats").status_code == 409


def test_crud_roundtrip(client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    assert client.get("/api/regions").json() == {"regions": []}

    r = client.post("/api/regions", json={"prism": PRISM})
    assert r.status_code == 200
    region = r.json()
    assert region["id"] == 1
    assert region["name"] == "Region 1"
    assert region["status"] == "draft"

    # persisted at the SCAN root, not the session dir
    on_disk = json.loads((scan_dir_for_loaded_scene / "eval_regions.json").read_text())
    assert on_disk["next_region_id"] == 2

    r = client.patch("/api/regions/1", json={"name": "skid A"})
    assert r.status_code == 200 and r.json()["name"] == "skid A"

    r = client.patch("/api/regions/1", json={"prism": {**PRISM, "height": 2.0}})
    assert r.status_code == 200 and r.json()["prism"]["height"] == 2.0

    assert client.delete("/api/regions/1").status_code == 200
    assert client.get("/api/regions").json() == {"regions": []}


def test_validation_and_404(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    bad = {"polygon": [[0, 0], [1, 0]], "y0": 0.0, "height": 1.0}
    assert client.post("/api/regions", json={"prism": bad}).status_code == 422
    r404 = client.patch("/api/regions/99", json={"name": "x"})
    assert r404.status_code == 404
    assert r404.json()["detail"] == "no region with id 99"   # no repr-quoting
    assert client.delete("/api/regions/99").status_code == 404
    client.post("/api/regions", json={"prism": PRISM})
    assert client.patch("/api/regions/1", json={}).status_code == 422   # empty patch


def test_gate_refuses_without_raw_source(client_with_loaded_annotated_scene):
    # The default fixture has no raw source registered.
    client = client_with_loaded_annotated_scene
    client.post("/api/regions", json={"prism": {"polygon": [[-10, -10], [10, -10], [10, 10], [-10, 10]],
                                                "y0": -10.0, "height": 20.0}})
    r = client.patch("/api/regions/1", json={"status": "eval_grade"})
    assert r.status_code == 422
    assert "raw source" in r.json()["detail"]


def test_gate_refuses_below_point_floor_with_raw_registered(monkeypatch, tmp_path):
    import main
    from fastapi.testclient import TestClient
    from tests.conftest import build_annotated_root, register_raw_source

    few_pts = [[i * 0.005, 0.0, 0.0] for i in range(25)]
    root, _sid = build_annotated_root(tmp_path, pts=np.asarray(few_pts, dtype=np.float32),
                                      n_instance0_points=25)
    register_raw_source(root, "demo", few_pts)
    monkeypatch.setattr("app.constants.LIDAR_ROOT", root, raising=False)
    client = TestClient(main.app)
    r = client.post("/api/load", json={"name": "annotated/demo", "max_points": 100})
    assert r.status_code == 200

    client.post("/api/regions", json={"prism": {"polygon": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
                                                "y0": -1.0, "height": 2.0}})
    r = client.patch("/api/regions/1", json={"status": "eval_grade"})
    assert r.status_code == 422
    assert "100" in r.json()["detail"]


def test_gate_passes_on_dense_scene_and_locks(client_with_dense_annotated_scene):
    client = client_with_dense_annotated_scene
    client.post("/api/regions", json={"prism": PRISM})
    r = client.patch("/api/regions/1", json={"status": "eval_grade"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "eval_grade"
    assert body["accuracy"]["p90"] == pytest.approx(0.005, abs=1e-4)
    assert body["accuracy"]["loa"] == "LOA40"

    # locked: geometry edits and delete refuse; rename still fine
    assert client.patch("/api/regions/1", json={"prism": PRISM}).status_code == 422
    assert client.delete("/api/regions/1").status_code == 422
    assert client.patch("/api/regions/1", json={"name": "ok"}).status_code == 200

    # draft flip unlocks + clears accuracy
    r = client.patch("/api/regions/1", json={"status": "draft"})
    assert r.status_code == 200 and "accuracy" not in r.json()
    assert client.delete("/api/regions/1").status_code == 200


def test_patch_cannot_unlock_and_redraw_in_one_request(client_with_dense_annotated_scene):
    """Pins the field ordering in patch_region: prism is applied BEFORE status,
    so a combined {"status":"draft","prism":X} can never unlock-then-mutate an
    eval-grade region. Swapping those two blocks would silently allow it."""
    client = client_with_dense_annotated_scene
    client.post("/api/regions", json={"prism": PRISM})
    assert client.patch("/api/regions/1", json={"status": "eval_grade"}).status_code == 200

    other = {**PRISM, "height": 7.0}
    assert client.patch("/api/regions/1", json={"status": "draft", "prism": other}).status_code == 422

    after = client.get("/api/regions").json()["regions"][0]
    assert after["prism"]["height"] == PRISM["height"]
    assert after["status"] == "eval_grade"


def test_stats(client_with_dense_annotated_scene):
    client = client_with_dense_annotated_scene
    client.post("/api/regions", json={"prism": PRISM})
    r = client.get("/api/regions/stats")
    assert r.status_code == 200
    s = r.json()["regions"][0]
    assert s["id"] == 1
    assert s["n_points"] == 400          # whole grid inside
    # dense fixture labels inst 0 on its first 2 points (conftest)
    assert s["n_unlabeled"] == 398
    assert s["instances"] == {"0": {"inside": 2, "total": 2}}


def test_frame_roundtrip_with_recenter(monkeypatch, tmp_path):
    """A scene whose coords exceed 1e3 triggers _recenter; the stored file
    must hold stored-frame geometry while the API speaks the runtime frame."""
    import main
    from fastapi.testclient import TestClient
    from tests.conftest import build_annotated_root, dense_grid_pts, register_raw_source

    off = (5000.0, 0.0, 3000.0)
    pts = dense_grid_pts(offset=off)
    root, _sid = build_annotated_root(tmp_path, pts=pts)
    register_raw_source(root, "demo", pts)
    monkeypatch.setattr("app.constants.LIDAR_ROOT", root, raising=False)
    client = TestClient(main.app)
    r = client.post("/api/load", json={"name": "annotated/demo", "max_points": 100})
    assert r.status_code == 200
    recenter = r.json()["recenter_offset"]
    assert abs(recenter[0]) > 1e3        # recenter actually triggered

    # Create in the RUNTIME frame (near origin, like the viewer sees it).
    client.post("/api/regions", json={"prism": PRISM})
    stored = json.loads((tmp_path / "lidar" / "annotated" / "demo" /
                         "eval_regions.json").read_text())
    sx, sz = stored["regions"][0]["prism"]["polygon"][0]
    # stored = runtime + recenter_offset
    assert sx == pytest.approx(PRISM["polygon"][0][0] + recenter[0], abs=1e-3)
    assert sz == pytest.approx(PRISM["polygon"][0][1] + recenter[2], abs=1e-3)
    # ...and it comes back out in the runtime frame.
    out = client.get("/api/regions").json()["regions"][0]["prism"]["polygon"][0]
    assert out[0] == pytest.approx(PRISM["polygon"][0][0], abs=1e-3)
    # The gate works across the frame shift too (grid is 5 mm).
    assert client.patch("/api/regions/1", json={"status": "eval_grade"}).status_code == 200


def _mark_review(client, indices):
    import base64

    import numpy as np
    return client.post("/api/segment/apply", json={
        "op": "set_category",
        "indices": base64.b64encode(np.asarray(indices, np.int32).tobytes()).decode(),
        "payload": {"category": "excluded_review"},
    })


def test_gate_refuses_over_review_budget(client_with_dense_annotated_scene):
    """excluded-review is a budget, not a bin: >3% of a region's points blocks
    the eval-grade flip (eval-labeling phase 2)."""
    client = client_with_dense_annotated_scene
    client.post("/api/regions", json={"prism": PRISM})
    assert _mark_review(client, range(20)).status_code == 200      # 20/400 = 5%
    r = client.patch("/api/regions/1", json={"status": "eval_grade"})
    assert r.status_code == 422
    assert "budget" in r.json()["detail"]
    assert client.get("/api/regions").json()["regions"][0]["status"] == "draft"


def test_gate_passes_at_the_budget_edge(client_with_dense_annotated_scene):
    client = client_with_dense_annotated_scene
    client.post("/api/regions", json={"prism": PRISM})
    assert _mark_review(client, range(12)).status_code == 200      # 12/400 = 3.0%
    assert client.patch("/api/regions/1", json={"status": "eval_grade"}).status_code == 200


def test_stats_report_review_points(client_with_dense_annotated_scene):
    client = client_with_dense_annotated_scene
    client.post("/api/regions", json={"prism": PRISM})
    _mark_review(client, range(5))
    s = client.get("/api/regions/stats").json()["regions"][0]
    assert s["n_review"] == 5
