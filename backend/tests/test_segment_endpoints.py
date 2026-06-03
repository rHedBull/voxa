"""Tests for /api/segment/* endpoints."""
from __future__ import annotations

import base64

import numpy as np


def _b64_to_int32(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.int32)


def _b64_int32(values: list[int]) -> str:
    return base64.b64encode(np.array(values, dtype=np.int32).tobytes()).decode("ascii")


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


def test_save_drops_unclassified_preseg_on_disk_only(
    client_with_loaded_annotated_scene, scan_dir_for_loaded_scene
):
    """A preseg-style point (inst≥0, class=-1) would violate invariant 3 in
    the labels/ export, so the save endpoint coerces those to (-1, -1) on
    disk. But the in-memory SegmentSession is the working canvas — it must
    KEEP the preseg ids so the user can keep labeling. The earlier
    in-place-mutation behavior collapsed every prelabel-derived segment to
    -1 on every save and leaked through the session/working_*.npy autosave
    (which runs as part of save), so reload couldn't recover preseg either.
    """
    import main
    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    seg.class_ids[:] = -1
    seg.instance_ids[:] = -1
    seg.class_ids[0] = -1; seg.instance_ids[0] = 99   # unclassified preseg
    seg.class_ids[1] = 2;  seg.instance_ids[1] = 100  # classified

    r = client.put("/api/segment/save")
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["n_dropped_preseg"] == 1
    assert j["n_labeled_points"] == 1  # only the classified point on disk

    # Disk export: invariant 3 satisfied — preseg point reset to (-1, -1).
    arr_inst = np.load(scan_dir_for_loaded_scene / "labels" / "gt_segment_ids.npy")
    arr_cls  = np.load(scan_dir_for_loaded_scene / "labels" / "gt_class_ids.npy")
    assert int(arr_inst[0]) == -1
    assert int(arr_cls[0]) == -1
    assert int(arr_inst[1]) == 100
    assert int(arr_cls[1]) == 2

    # In-memory: preseg-only point KEPT its instance id so the user can
    # continue labeling that segment. This is the regression guard.
    assert int(seg.instance_ids[0]) == 99
    assert int(seg.class_ids[0]) == -1


def test_save_then_reload_preserves_preseg_full_round_trip(
    client_with_annotated_scene,
):
    """End-to-end: set up preseg+labels → save → simulate page reload via a
    fresh /api/load → confirm the recovered SegmentSession brings back the
    full working canvas (preseg ids intact + user labels intact).

    This is the user-facing guarantee the previous bug broke: every save
    flattened preseg, and the next page load showed only the labeled
    segment with the rest as -1.
    """
    import main
    client, scene_id, session_id = client_with_annotated_scene

    r = client.post("/api/load", json={"name": scene_id, "max_points": 100})
    assert r.status_code == 200

    seg = main._state["seg"]
    n = len(seg.instance_ids)
    # Plant a recognisable mix: one labeled segment + several preseg-only
    # ids. Mirrors the real workflow — user runs preseg (5000+ instances),
    # then labels a handful while the rest stay preseg-coloured.
    seg.class_ids[:] = -1
    seg.instance_ids[:] = -1
    seg.class_ids[0] = 2; seg.instance_ids[0] = 500   # labeled
    seg.instance_ids[1] = 700                          # preseg only
    seg.instance_ids[2] = 700                          # same preseg id
    seg.instance_ids[3] = 701                          # different preseg id
    seg.instance_ids[4] = 702                          # different preseg id

    r = client.put("/api/segment/save")
    assert r.status_code == 200, r.text

    # Drop in-memory state to simulate the process restart / page reload
    # path. Next /api/load must hydrate the session from session/working_*.
    main._state["seg"] = None
    main._state["pc"] = None
    main._state["scene"] = None

    r = client.post("/api/load", json={"name": scene_id, "max_points": 100})
    assert r.status_code == 200

    # Verify recovery path was taken — the SegmentSession should be back
    # with full preseg structure, NOT the labels-sanitized version.
    seg2 = main._state["seg"]
    assert seg2 is not None
    assert len(seg2.instance_ids) == n
    assert int(seg2.instance_ids[0]) == 500
    assert int(seg2.class_ids[0]) == 2
    assert int(seg2.instance_ids[1]) == 700
    assert int(seg2.instance_ids[2]) == 700
    assert int(seg2.instance_ids[3]) == 701
    assert int(seg2.instance_ids[4]) == 702

    # Cross-check through the HTTP surface the frontend uses on reload.
    r = client.get("/api/segment/state")
    assert r.status_code == 200
    state = r.json()
    inst = _b64_to_int32(state["full_instance_ids"])
    assert int(inst[0]) == 500
    assert int(inst[1]) == 700
    assert int(inst[2]) == 700
    assert int(inst[3]) == 701
    assert int(inst[4]) == 702
    # 4 distinct preseg ids active (500, 700, 701, 702).
    assert state["n_segments"] == 4
    assert state["n_assigned"] == 5


def test_save_does_not_collapse_preseg_in_session_autosave(
    client_with_loaded_annotated_scene,
):
    """Regression: the session/working_*.npy file must contain the
    UNMUTATED in-memory state after save, so a page reload (which hydrates
    from session/) shows the user the same preseg + labels canvas they had
    before saving. The old in-place-mutation bug wrote the collapsed (post-
    invariant) arrays into session/, destroying preseg structure.
    """
    import main
    from labeling.segment_io import load_session_aux, load_working_arrays
    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    seg.class_ids[:] = -1
    seg.instance_ids[:] = -1
    # Plant: 1 labeled segment + 3 preseg-only points across two preseg ids.
    seg.class_ids[0] = 2; seg.instance_ids[0] = 100
    seg.instance_ids[1] = 200    # preseg only
    seg.instance_ids[2] = 200    # same preseg id
    seg.instance_ids[3] = 201    # another preseg id

    r = client.put("/api/segment/save")
    assert r.status_code == 200, r.text

    session_dir = seg.session_dir
    aux = load_session_aux(session_dir)
    assert aux is not None
    wa = load_working_arrays(session_dir, n_points=len(seg.instance_ids))
    assert wa is not None
    working_cls, working_inst = wa
    # Session must carry the full working canvas, not the labels-sanitized
    # one. Preseg ids 200 and 201 survive in session/ even though they were
    # stripped from labels/.
    assert int(working_inst[1]) == 200
    assert int(working_inst[2]) == 200
    assert int(working_inst[3]) == 201
    assert int(working_inst[0]) == 100
    assert int(working_cls[0]) == 2


# ── segment/state hydration ──────────────────────────────────────────────────

def test_segment_state_surfaces_full_session_aux(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "max_points": 100})
    assert r.status_code == 200

    r = client.get("/api/segment/state")
    body = r.json()
    # Should include the fields needed for FE hydration.
    for k in ("has_seg", "n_points", "preseg_id", "preseg_fingerprint",
              "source_fingerprint", "is_from_prelabel",
              "dirty"):
        assert k in body, f"missing field: {k}"
