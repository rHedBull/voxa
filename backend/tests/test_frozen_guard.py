"""Tests for the 422 write-guard on frozen (legacy) class ids.

FROZEN_ID = 0 ("pipe", legacy — frozen: true in classes.yaml).
LIVE_ID = 16 ("elbow", live primitive).
"""
from __future__ import annotations

import base64

import numpy as np

FROZEN_ID = 0
LIVE_ID = 16


def _b64_i32(vals) -> str:
    return base64.b64encode(np.array(vals, dtype=np.int32).tobytes()).decode()


def _huge_obb(**over):
    body = {"type": "obb", "center": [0.0, 0.0, 0.0],
            "size": [1e7, 1e7, 1e7], "rotation": [0.0, 0.0, 0.0]}
    body.update(over)
    return body


def test_apply_set_class_frozen_422(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/apply", json={
        "op": "set_class",
        "indices": _b64_i32([0, 1]),
        "payload": {"class_id": FROZEN_ID},
    })
    assert r.status_code == 422
    assert "frozen" in r.json()["detail"].lower()


def test_apply_reassign_frozen_422(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/apply", json={
        "op": "reassign",
        "indices": _b64_i32([0, 1]),
        "payload": {"target_inst": -1, "target_class": FROZEN_ID},
    })
    assert r.status_code == 422
    assert "frozen" in r.json()["detail"].lower()


def test_apply_shape_frozen_422(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/apply-shape", json={
        "shape": _huge_obb(), "target_class": FROZEN_ID,
        "target_inst": -1, "merged_from": [],
    })
    assert r.status_code == 422
    assert "frozen" in r.json()["detail"].lower()


def test_centerline_apply_frozen_422(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    paths = [{"points": [[-1e5, -1e5, -1e5], [1e5, 1e5, 1e5]],
              "radius": 1e6, "smooth": False}]
    r = client.post("/api/segment/centerline-apply", json={
        "paths": paths, "target_class": FROZEN_ID,
        "target_inst": -1, "merged_from": [],
    })
    assert r.status_code == 422
    assert "frozen" in r.json()["detail"].lower()


def test_live_class_still_applies(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/apply", json={
        "op": "set_class",
        "indices": _b64_i32([0, 1]),
        "payload": {"class_id": LIVE_ID},
    })
    assert r.status_code == 200


def test_frozen_class_by_name_422(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/apply-shape", json={
        "shape": _huge_obb(), "target_class": "pipe",
        "target_inst": -1, "merged_from": [],
    })
    assert r.status_code == 422
    assert "frozen" in r.json()["detail"].lower()


def test_cut_instance_frozen_422(client_with_loaded_annotated_scene):
    # Fixture instance 0 (points {1,2}) already carries class 0 ("pipe",
    # frozen) — seeded directly from the preseg summary in
    # conftest.build_annotated_root. Cutting from it must reject with a
    # message that tells the user to re-label the source instance first.
    import main
    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    src_inst_id = 0
    assert int(seg.class_ids[np.flatnonzero(seg.instance_ids == src_inst_id)[0]]) == FROZEN_ID

    r = client.post("/api/segment/cut-shape", json={
        "shape": _huge_obb(),
        "sources": [{"kind": "instance", "seg_id": src_inst_id}],
        "protect_instances": [],
    })
    assert r.status_code == 422
    assert "re-label" in r.json()["detail"].lower()
