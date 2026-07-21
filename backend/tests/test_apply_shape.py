"""Tests for the generic /api/segment/apply-shape endpoint."""
from __future__ import annotations


def _obb_body(**over):
    body = {
        "shape": {"type": "obb", "center": [0.0, 0.0, 0.0],
                  "size": [1e7, 1e7, 1e7], "rotation": [0.0, 0.0, 0.0]},
        "target_class": "elbow", "target_inst": -1, "merged_from": [],
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
                           "target_class": "elbow", "target_inst": -1,
                           "merged_from": []})
    client.post("/api/segment/undo")  # revert before the second apply
    r2 = client.post("/api/segment/centerline-apply",
                     json={"paths": paths, "target_class": "elbow",
                           "target_inst": -1, "merged_from": []})
    assert r1.json()["n_affected"] == r2.json()["n_affected"]


def test_apply_shape_skips_protected_instances(client_with_loaded_annotated_scene):
    # Overextending a box over an already-labeled (confirmed) instance must not
    # steal its points. First apply creates instance A; a second overlapping
    # apply that lists A in protect_instances leaves A's points untouched.
    client = client_with_loaded_annotated_scene
    r1 = client.post("/api/segment/apply-shape", json=_obb_body())
    a = r1.json()
    a_id, a_n = a["instance_id"], a["n_affected"]
    assert a_n > 0

    r2 = client.post("/api/segment/apply-shape", json=_obb_body(
        target_class="tank", protect_instances=[a_id]))
    b = r2.json()
    assert b["n_protected"] == a_n     # every one of A's points was skipped
    assert b["n_affected"] == 0        # nothing else was inside the box
    # A still exists as class 'pipe' — a third, unprotected apply would grab it.
    r3 = client.post("/api/segment/apply-shape", json=_obb_body(target_class="tank"))
    assert r3.json()["n_affected"] == a_n


def test_apply_shape_prism_labels_enclosed_points(client_with_loaded_annotated_scene):
    # Fixture cloud is `rng.default_rng(0).standard_normal((8, 3))` (see
    # conftest.py::write_scene_ply). A [-2,2]x[-2,2] XZ square combined with
    # a y-band of [-1.0, 0.5] encloses exactly points {0,1,3,5,7} (point 4 is
    # excluded by the XZ square, points 2 and 6 by the y-band) — verified by
    # calling prism_indices directly on the same fixture coordinates.
    client = client_with_loaded_annotated_scene
    polygon = [[-2.0, -2.0], [2.0, -2.0], [2.0, 2.0], [-2.0, 2.0]]
    r = client.post("/api/segment/apply-shape", json={
        "shape": {"type": "prism", "polygon": polygon, "y0": -1.0, "height": 1.5},
        "target_class": "elbow", "target_inst": -1, "merged_from": [],
        "protect_instances": [],
    })
    assert r.status_code == 200
    body = r.json()
    # Non-empty apply-shape responses are serialized by the same
    # `_serialize_apply` as every other shape type — `op` is the underlying
    # `apply_reassign` op ("reassign"), not the route name; the OBB test above
    # doesn't check it for the same reason.
    assert body["op"] == "reassign"
    assert body["n_affected"] == 5


def test_apply_shape_unknown_type_400(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/apply-shape",
                    json={"shape": {"type": "blob"}, "target_class": "elbow"})
    assert r.status_code == 400
