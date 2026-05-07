"""Tests for /api/segment/* endpoints."""
from __future__ import annotations

import base64

import numpy as np


def _b64_to_int32(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.int32)


def _b64_int32(values: list[int]) -> str:
    return base64.b64encode(np.array(values, dtype=np.int32).tobytes()).decode("ascii")


# ── Task 9: brush-query ──────────────────────────────────────────────────────

def test_brush_query_returns_indices(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    body = {"center": [0.0, 0.0, 0.0], "radius": 100.0}
    r = client.post("/api/segment/brush-query", json=body)
    assert r.status_code == 200
    j = r.json()
    assert "indices" in j and "n" in j
    arr = _b64_to_int32(j["indices"])
    assert arr.size == j["n"]


def test_brush_query_409_when_no_session(client_with_loaded_annotated_scene, monkeypatch):
    import main
    monkeypatch.setitem(main._state, "seg", None)
    client = client_with_loaded_annotated_scene
    body = {"center": [0.0, 0.0, 0.0], "radius": 1.0}
    r = client.post("/api/segment/brush-query", json=body)
    assert r.status_code == 409


# ── Task 10: apply ───────────────────────────────────────────────────────────

def test_apply_set_class_changes_state(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    body = {
        "op": "set_class",
        "indices": _b64_int32([1, 2]),
        "payload": {"class_id": 2},
    }
    r = client.post("/api/segment/apply", json=body)
    assert r.status_code == 200
    j = r.json()
    assert j["op"] == "set_class"
    assert j["n_affected"] == 2


def test_apply_merge_routes_to_session(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    body = {"op": "merge", "payload": {"source_inst": 2, "target_inst": 0}}
    r = client.post("/api/segment/apply", json=body)
    assert r.status_code == 200


def test_apply_400_when_indices_missing_for_set_class(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    body = {"op": "set_class", "payload": {"class_id": 2}}  # no indices
    r = client.post("/api/segment/apply", json=body)
    assert r.status_code == 400
    assert "indices" in r.json()["detail"]


def test_apply_400_on_unknown_op(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    body = {"op": "explode", "payload": {}}
    r = client.post("/api/segment/apply", json=body)
    assert r.status_code == 400


# ── Task 11: undo / redo ─────────────────────────────────────────────────────

def test_undo_returns_inverse_delta(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    client.post("/api/segment/apply", json={
        "op": "set_class",
        "indices": _b64_int32([1, 2]),
        "payload": {"class_id": 2},
    })
    r = client.post("/api/segment/undo")
    assert r.status_code == 200
    j = r.json()
    assert j["direction"] == "undo"
    assert j["n_affected"] == 2


def test_undo_returns_204_when_stack_empty(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/undo")
    assert r.status_code == 204


# ── Task 12: save ────────────────────────────────────────────────────────────

def test_save_writes_labels_to_disk(client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    client.post("/api/segment/apply", json={
        "op": "set_class",
        "indices": _b64_int32([1, 2]),
        "payload": {"class_id": 2},
    })
    r = client.put("/api/segment/save")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    # Demo fixture has 4 instances (ids 0,1,2,3) and 6 labeled points.
    assert j["n_segments"] == 4
    assert j["n_labeled_points"] == 6
    arr = np.load(scan_dir_for_loaded_scene / "labels" / "gt_class_ids.npy")
    assert int(arr[1]) == 2 and int(arr[2]) == 2


def test_save_drops_unclassified_preseg(client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    """A preseg-style point (inst≥0, class=-1) would violate invariant 3.
    The save endpoint must coerce those points to (-1, -1) and persist the
    rest, so partial labeling sessions can be saved without classifying
    every preseg first."""
    import main
    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    # Reset to a known state, then plant exactly one unclassified preseg
    # point alongside one fully-classified point.
    seg.class_ids[:] = -1
    seg.instance_ids[:] = -1
    seg.class_ids[0] = -1; seg.instance_ids[0] = 99   # unclassified preseg
    seg.class_ids[1] = 2;  seg.instance_ids[1] = 100  # classified

    r = client.put("/api/segment/save")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["n_dropped_preseg"] == 1
    assert j["n_labeled_points"] == 1  # only the classified point remains
    # The unclassified point was reset on disk and in-memory.
    arr_inst = np.load(scan_dir_for_loaded_scene / "labels" / "gt_segment_ids.npy")
    arr_cls  = np.load(scan_dir_for_loaded_scene / "labels" / "gt_class_ids.npy")
    assert int(arr_inst[0]) == -1
    assert int(seg.instance_ids[0]) == -1
    # Classified point survived intact.
    assert int(arr_inst[1]) == 100
    assert int(arr_cls[1]) == 2
