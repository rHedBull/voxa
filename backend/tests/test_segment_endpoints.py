"""Tests for /api/segment/* endpoints."""
from __future__ import annotations

import base64
import json

import numpy as np
import pytest


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
    # Target instance 1 (class 1/"tank", live) — instance 0 carries frozen
    # legacy class 0/"pipe" and merging into it is now rejected (see
    # test_frozen_guard.py::test_merge_into_frozen_instance_422).
    body = {"op": "merge", "payload": {"source_inst": 2, "target_inst": 1}}
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

def _session_output_dir(scan_dir):
    """Return the single sessions/<id>/output dir that save created."""
    sessions_root = scan_dir / "sessions"
    dirs = [d / "output" for d in sessions_root.iterdir()
            if (d / "output").is_dir()]
    assert len(dirs) == 1, f"expected 1 output dir, got {[str(d) for d in dirs]}"
    return dirs[0]


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
    out_dir = _session_output_dir(scan_dir_for_loaded_scene)
    arr = np.load(out_dir / "gt_class_ids.npy")
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
    out_dir = _session_output_dir(scan_dir_for_loaded_scene)
    arr_inst = np.load(out_dir / "gt_segment_ids.npy")
    arr_cls  = np.load(out_dir / "gt_class_ids.npy")
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
              "dirty", "session_id"):
        assert k in body, f"missing field: {k}"
    assert body["session_id"] == session_id


# ── Task 5: save writes into active session's output/ ────────────────────────

def test_save_writes_into_session_output(client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    r = client.put("/api/segment/save")
    assert r.status_code == 200
    sessions_root = scan_dir_for_loaded_scene / "sessions"
    outs = [d / "output" / "gt_class_ids.npy" for d in sessions_root.iterdir()
            if (d / "output").is_dir()]
    assert len(outs) == 1 and outs[0].exists()
    assert not (scan_dir_for_loaded_scene / "labels").exists()  # v2: no top-level labels/
    meta = json.loads((outs[0].parent / "gt_segment_metadata.json").read_text())
    assert "preseg_fingerprint" in meta


def test_save_persists_dirty_false_to_session_json(client_with_annotated_scene):
    """After an edit + save, session.json must record dirty:false. The
    autosave inside save persists the still-True flag first, so the endpoint
    has to re-persist the cleared flag — otherwise the session list keeps
    reporting the session as unsaved after reload (stuck "● unsaved" badge)."""
    import main
    client, scene_id, session_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "max_points": 100})
    assert r.status_code == 200

    # An edit marks the working canvas dirty.
    client.post("/api/segment/apply", json={
        "op": "set_class",
        "indices": _b64_int32([1, 2]),
        "payload": {"class_id": 2},
    })
    seg = main._state["seg"]
    assert seg.dirty is True  # edit marks the working canvas dirty
    session_json = seg.session_dir / "session.json"

    r = client.put("/api/segment/save")
    assert r.status_code == 200, r.text
    assert json.loads(session_json.read_text())["dirty"] is False


def test_save_without_session_409(client_with_annotated_scene):
    """Saving without an active session_id in _state must return 409."""
    import main
    client, scene_id, _session_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "max_points": 100})
    assert r.status_code == 200
    # Artificially clear session_id to simulate the no-session condition.
    main._state["session_id"] = None
    r = client.put("/api/segment/save")
    assert r.status_code == 409
    assert "session" in r.json()["detail"].lower()


def test_resume_session_restores_sam_candidates(client_with_loaded_annotated_scene):
    """SAM candidates (working_sam_ids.npy + sam_segments.json) written directly
    to disk must be hydrated back into the in-memory SegmentSession on the next
    /api/load — mirrors the preseg+labels resume guarantee above, for the SAM
    candidate layer added in Task 1/2."""
    import main
    from labeling.segment_io import save_session_aux, save_sam_segments

    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    scene_id = main._state["scene"]
    n = len(seg.instance_ids)

    sam_ids = np.full(n, -1, dtype=np.int32)
    sam_ids[0] = 7
    sam_ids[1] = 7
    save_session_aux(seg.session_dir, seg._aux_payload(), sam_ids=sam_ids)
    save_sam_segments(seg.session_dir, {7: {"n_points": 2, "mask_score": 0.8,
                                            "created_at": "2026-07-13T00:00:00+00:00"}})

    # Drop in-memory state to force a real resume (not just a GET) on next load.
    main._state.update(seg=None, pc=None, scene=None)

    r = client.post("/api/load", json={"name": scene_id, "max_points": 100})
    assert r.status_code == 200

    seg2 = main._state["seg"]
    assert int(seg2.sam_ids[0]) == 7 and int(seg2.sam_ids[1]) == 7
    assert seg2.sam_segments[7]["n_points"] == 2
    assert seg2._next_sam_id == 8


def test_resume_session_raises_on_sam_ids_shape_mismatch(client_with_loaded_annotated_scene):
    """A working_sam_ids.npy whose length no longer matches the cloud (bad/foreign
    data directory) must fail loudly through /api/load, not be silently swallowed —
    unlike working_class_ids/working_segment_ids (which soft-fail to a controlled
    409), load_sam_ids raises directly and that must propagate unhandled."""
    import main
    from labeling.segment_io import save_session_aux

    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    scene_id = main._state["scene"]

    # Deliberately wrong length: any n_points != len(cloud).
    bad_sam_ids = np.array([-1, 0], dtype=np.int32)
    save_session_aux(seg.session_dir, seg._aux_payload(), sam_ids=bad_sam_ids)

    # Drop in-memory state to force a real resume (not just a GET) on next load.
    main._state.update(seg=None, pc=None, scene=None)

    with pytest.raises(ValueError):
        client.post("/api/load", json={"name": scene_id, "max_points": 100})


def test_segment_state_includes_full_sam_ids_and_sam_segments(
    client_with_loaded_annotated_scene,
):
    client = client_with_loaded_annotated_scene
    r = client.get("/api/segment/state")
    body = r.json()
    assert "full_sam_ids" in body
    assert "sam_segments" in body
    assert body["sam_segments"] == []  # nothing materialized yet


def test_segment_state_reflects_materialized_sam_segments(
    client_with_loaded_annotated_scene,
):
    import main
    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    seg.materialize_sam_segment(np.array([0, 1], dtype=np.int32), source="sam", mask_score=0.7)
    r = client.get("/api/segment/state")
    body = r.json()
    sam_ids = _b64_to_int32(body["full_sam_ids"])
    assert int(sam_ids[0]) == 0 and int(sam_ids[1]) == 0
    assert body["sam_segments"] == [
        {"id": 0, "n_points": 2, "source": "sam", "mask_score": 0.7, "created_at": body["sam_segments"][0]["created_at"]},
    ]
