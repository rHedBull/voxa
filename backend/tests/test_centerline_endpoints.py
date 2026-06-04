"""Tests for /api/segment/centerline-* endpoints."""
from __future__ import annotations

import json


def _apply_body(**over):
    body = {
        "paths": [{"points": [[-1e5, -1e5, -1e5], [1e5, 1e5, 1e5]],
                   "radius": 1e6, "smooth": False}],   # tube swallows the whole cloud
        "target_class": "pipe",
        "target_inst": -1,
        "merged_from": [],
    }
    body.update(over)
    return body


def test_centerline_apply_labels_points_and_persists(
        client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/centerline-apply", json=_apply_body())
    assert r.status_code == 200
    j = r.json()
    assert j["n_affected"] > 0
    assert j["instance_id"] == j["new_instance_id"]
    # Stored in the active session's centerlines.json.
    sessions = list((scan_dir_for_loaded_scene / "sessions").iterdir())
    docs = [d / "centerlines.json" for d in sessions if (d / "centerlines.json").exists()]
    assert len(docs) == 1
    doc = json.loads(docs[0].read_text())
    assert len(doc["paths"]) == 1
    assert doc["paths"][0]["instance_id"] == j["instance_id"]


def test_centerline_apply_empty_tube_returns_zero_no_store(
        client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    body = _apply_body(paths=[{"points": [[9e6, 9e6, 9e6], [9e6 + 1, 9e6, 9e6]],
                               "radius": 0.01, "smooth": False}])
    r = client.post("/api/segment/centerline-apply", json=body)
    assert r.status_code == 200
    j = r.json()
    assert j["n_affected"] == 0
    assert "after_class" not in j          # spec: keys absent on empty delta
    assert "instance_id" not in j
    files = list((scan_dir_for_loaded_scene / "sessions").rglob("centerlines.json"))
    assert files == []                     # nothing persisted


def test_centerline_apply_undo_reverts_labels(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    before = client.get("/api/segment/state").json()
    r = client.post("/api/segment/centerline-apply", json=_apply_body())
    assert r.json()["n_affected"] > 0
    u = client.post("/api/segment/undo")
    assert u.status_code == 200
    after = client.get("/api/segment/state").json()
    assert after["n_assigned"] == before["n_assigned"]


def test_centerline_apply_reapply_same_instance_replaces(
        client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    j1 = client.post("/api/segment/centerline-apply", json=_apply_body()).json()
    inst = j1["instance_id"]
    j2 = client.post("/api/segment/centerline-apply",
                     json=_apply_body(target_inst=inst)).json()
    assert "new_instance_id" not in j2     # reused, not allocated
    assert j2["instance_id"] == inst
    doc = json.loads(next((scan_dir_for_loaded_scene / "sessions")
                          .rglob("centerlines.json")).read_text())
    assert len(doc["paths"]) == 1          # replaced, not appended


def test_centerline_apply_validation_errors(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    # < 2 points
    r = client.post("/api/segment/centerline-apply", json=_apply_body(
        paths=[{"points": [[0, 0, 0]], "radius": 0.1, "smooth": False}]))
    assert r.status_code == 422
    # radius <= 0
    r = client.post("/api/segment/centerline-apply", json=_apply_body(
        paths=[{"points": [[0, 0, 0], [1, 0, 0]], "radius": 0, "smooth": False}]))
    assert r.status_code == 422
    # unknown class name
    r = client.post("/api/segment/centerline-apply", json=_apply_body(
        target_class="not-a-class"))
    assert r.status_code == 400


def test_centerline_apply_409_without_session(client):
    # Plain client: nothing loaded.
    r = client.post("/api/segment/centerline-apply", json=_apply_body())
    assert r.status_code == 409


def test_get_centerlines_empty_then_populated(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.get("/api/segment/centerlines")
    assert r.status_code == 200
    assert r.json() == {"paths": []}
    client.post("/api/segment/centerline-apply", json=_apply_body())
    j = client.get("/api/segment/centerlines").json()
    assert len(j["paths"]) == 1
    assert {"points", "radius", "smooth", "class_id", "instance_id"} <= set(j["paths"][0])


def test_get_centerlines_409_without_session(client):
    r = client.get("/api/segment/centerlines")
    assert r.status_code == 409
