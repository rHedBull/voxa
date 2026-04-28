"""Tests for /api/segment/* endpoints."""
from __future__ import annotations

import base64

import numpy as np
import pytest


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
