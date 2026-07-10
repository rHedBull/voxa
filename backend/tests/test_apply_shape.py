"""Tests for the generic /api/segment/apply-shape endpoint."""
from __future__ import annotations


def _obb_body(**over):
    body = {
        "shape": {"type": "obb", "center": [0.0, 0.0, 0.0],
                  "size": [1e7, 1e7, 1e7], "rotation": [0.0, 0.0, 0.0]},
        "target_class": "pipe", "target_inst": -1, "merged_from": [],
    }
    body.update(over)
    return body


def test_apply_shape_obb_reassigns_and_allocates(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/apply-shape", json=_obb_body())
    assert r.status_code == 200
    j = r.json()
    assert j["n_affected"] > 0
    assert j["instance_id"] == j["new_instance_id"]


def test_apply_shape_obb_far_away_is_empty(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/apply-shape", json=_obb_body(
        shape={"type": "obb", "center": [9e6, 9e6, 9e6],
               "size": [0.01, 0.01, 0.01], "rotation": [0.0, 0.0, 0.0]}))
    assert r.status_code == 200
    assert r.json()["n_affected"] == 0


def test_apply_shape_tube_parity_with_centerline_apply(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    paths = [{"points": [[-1e5, -1e5, -1e5], [1e5, 1e5, 1e5]],
              "radius": 1e6, "smooth": False}]
    r1 = client.post("/api/segment/apply-shape",
                     json={"shape": {"type": "tube", "paths": paths},
                           "target_class": "pipe", "target_inst": -1,
                           "merged_from": []})
    client.post("/api/segment/undo")  # revert before the second apply
    r2 = client.post("/api/segment/centerline-apply",
                     json={"paths": paths, "target_class": "pipe",
                           "target_inst": -1, "merged_from": []})
    assert r1.json()["n_affected"] == r2.json()["n_affected"]


def test_apply_shape_unknown_type_400(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/apply-shape",
                    json={"shape": {"type": "blob"}, "target_class": "pipe"})
    assert r.status_code == 400
