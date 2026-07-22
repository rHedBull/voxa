"""Point-category routes + persistence (eval-labeling phase 2)."""
from __future__ import annotations

import base64
import json

import numpy as np

from labeling.categories import (CATEGORY_ARTIFACT, CATEGORY_EXCLUDED_REVIEW,
                                 CATEGORY_NONE)


def _b64_int32(vals):
    return base64.b64encode(np.asarray(vals, dtype=np.int32).tobytes()).decode()


def _b64_decode_int8(s):
    return np.frombuffer(base64.b64decode(s), dtype=np.int8)


def _session_output_dir(scan_dir):
    sessions_root = scan_dir / "sessions"
    dirs = [d / "output" for d in sessions_root.iterdir() if (d / "output").is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def test_set_category_marks_points(client_with_loaded_annotated_scene):
    import main
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/apply", json={
        "op": "set_category",
        "indices": _b64_int32([1, 2]),
        "payload": {"category": "artifact"},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_affected"] == 2
    assert _b64_decode_int8(body["after_category"]).tolist() == [CATEGORY_ARTIFACT] * 2
    seg = main._state["seg"]
    assert bool((seg.categories[[1, 2]] == CATEGORY_ARTIFACT).all())
    assert bool((seg.class_ids[[1, 2]] == -1).all())


def test_set_category_review_returns_blob_instance(client_with_loaded_annotated_scene):
    import main
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/apply", json={
        "op": "set_category",
        "indices": _b64_int32([3, 4]),
        "payload": {"category": "excluded_review"},
    })
    assert r.status_code == 200, r.text
    blob = r.json()["new_instance_id"]
    seg = main._state["seg"]
    assert bool((seg.instance_ids[[3, 4]] == blob).all())
    assert bool((seg.class_ids[[3, 4]] == -1).all())


def test_set_category_unknown_name_is_400(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/apply", json={
        "op": "set_category",
        "indices": _b64_int32([1]),
        "payload": {"category": "banana"},
    })
    assert r.status_code == 400


def test_state_and_undo_carry_categories(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    client.post("/api/segment/apply", json={
        "op": "set_category", "indices": _b64_int32([1]),
        "payload": {"category": "transient"},
    })
    state = client.get("/api/segment/state").json()
    cats = _b64_decode_int8(state["full_categories"])
    assert cats[1] == 2 and cats[0] == CATEGORY_NONE

    undo = client.post("/api/segment/undo").json()
    assert _b64_decode_int8(undo["after_category"]).tolist() == [CATEGORY_NONE]
    cats = _b64_decode_int8(client.get("/api/segment/state").json()["full_categories"])
    assert cats[1] == CATEGORY_NONE


def test_apply_shape_marks_a_category(client_with_loaded_annotated_scene):
    import main
    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    lo = seg.positions.min(axis=0) - 1.0
    hi = seg.positions.max(axis=0) + 1.0
    center = ((lo + hi) / 2).tolist()
    size = (hi - lo).tolist()
    r = client.post("/api/segment/apply-shape", json={
        "shape": {"type": "obb", "center": center, "size": size, "rotation": [0, 0, 0]},
        "target_category": "artifact",
    })
    assert r.status_code == 200, r.text
    assert r.json()["instance_id"] is None      # artifact allocates no instance
    assert bool((seg.categories == CATEGORY_ARTIFACT).all())


def test_apply_shape_rejects_both_or_neither(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    shape = {"type": "obb", "center": [0, 0, 0], "size": [1, 1, 1], "rotation": [0, 0, 0]}
    assert client.post("/api/segment/apply-shape", json={"shape": shape}).status_code == 400
    r = client.post("/api/segment/apply-shape", json={
        "shape": shape, "target_class": 1, "target_category": "artifact"})
    assert r.status_code == 400


def test_frozen_class_is_now_rejected_everywhere(client_with_loaded_annotated_scene):
    """Phase 2 removed the denoise id-6 exemption; `unknown` is fully frozen."""
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/apply", json={
        "op": "reassign", "indices": _b64_int32([1]),
        "payload": {"target_inst": -1, "target_class": 6},
    })
    assert r.status_code == 422


def test_save_writes_category_and_component_arrays(
    client_with_loaded_annotated_scene, scan_dir_for_loaded_scene
):
    client = client_with_loaded_annotated_scene
    client.post("/api/segment/apply", json={
        "op": "set_category", "indices": _b64_int32([1, 2]),
        "payload": {"category": "excluded_review"},
    })
    assert client.put("/api/segment/save").status_code == 200
    out_dir = _session_output_dir(scan_dir_for_loaded_scene)

    cats = np.load(out_dir / "gt_point_category.npy")
    assert cats.dtype == np.int8
    assert bool((cats[[1, 2]] == CATEGORY_EXCLUDED_REVIEW).all())

    comps = np.load(out_dir / "gt_point_component_ids.npy")
    assert comps.dtype == np.int16
    assert comps.shape == cats.shape

    meta = json.loads((out_dir / "gt_segment_metadata.json").read_text())
    assert meta["categories"]["excluded_review"] == 2
    assert meta["component_link_radius_m"] == 0.05
    blobs = meta["review_blobs"]
    assert len(blobs) == 1 and blobs[0]["n_points"] == 2
    # the blob is stripped from gt_segment_ids (class-less) but recorded above
    inst = np.load(out_dir / "gt_segment_ids.npy")
    assert bool((inst[[1, 2]] == -1).all())


def test_categories_persist_across_reload(client_with_annotated_scene):
    import main
    client, scene_id, _sid = client_with_annotated_scene
    assert client.post("/api/load", json={"name": scene_id, "max_points": 100}).status_code == 200
    client.post("/api/segment/apply", json={
        "op": "set_category", "indices": _b64_int32([0]),
        "payload": {"category": "artifact"},
    })
    main._state["seg"].flush_autosave()
    # Re-load the scene: the session resumes from disk.
    r = client.post("/api/load", json={"name": scene_id, "max_points": 100,
                                       "want_full_labels": True})
    assert r.status_code == 200
    assert main._state["seg"].categories[0] == CATEGORY_ARTIFACT
    assert _b64_decode_int8(r.json()["full_categories"])[0] == CATEGORY_ARTIFACT
