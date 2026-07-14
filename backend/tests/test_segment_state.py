"""Editing-session state: apply / undo / redo, plus brush_query."""
from __future__ import annotations

import numpy as np
import pytest

from labeling.segment_state import SegmentSession


def _seed():
    cls = np.array([-1, 0, 0, 1, 1, 2, -1, 2], dtype=np.int8)
    inst = np.array([-1, 0, 0, 1, 1, 2, -1, 2], dtype=np.int32)
    return SegmentSession(class_ids=cls.copy(), instance_ids=inst.copy(),
                          positions=np.zeros((8, 3), dtype=np.float32))


def _seed_preseg(s, preseg_ids, *, preseg_id=None):
    """Seed the immutable preseg layer the way app.core._resume_session does
    (freeze_preseg was removed in v2 — sessions pin their preseg at creation)."""
    import numpy as np
    s.preseg_ids = preseg_ids.astype(np.int32, copy=False)
    s.preseg_id = preseg_id


def test_set_class_changes_specified_indices():
    s = _seed()
    delta = s.apply_set_class(indices=np.array([1, 2], dtype=np.int32),
                              class_id=2)
    assert int(s.class_ids[1]) == 2 and int(s.class_ids[2]) == 2
    # instance ids untouched.
    assert int(s.instance_ids[1]) == 0
    assert delta["op"] == "set_class"
    assert s.dirty is True


def test_undo_restores_pre_state_and_redo_reapplies():
    s = _seed()
    s.apply_set_class(np.array([1, 2], dtype=np.int32), class_id=2)
    s.undo()
    assert int(s.class_ids[1]) == 0 and int(s.class_ids[2]) == 0
    s.redo()
    assert int(s.class_ids[1]) == 2 and int(s.class_ids[2]) == 2


def test_undo_stack_is_bounded():
    s = _seed()
    s.history_cap = 3
    for cid in range(10):
        s.apply_set_class(np.array([1], dtype=np.int32), class_id=cid % 3)
    # Only 3 undos should succeed.
    for _ in range(3):
        assert s.undo() is not None
    assert s.undo() is None


def test_merge_moves_source_points_to_target_instance_and_class():
    s = _seed()
    s.apply_merge(source_inst=2, target_inst=0)
    assert (s.instance_ids[s.instance_ids != -1] == 0).any()
    # All previously-2 points are now 0.
    assert int((s.instance_ids == 2).sum()) == 0
    # And took target's class (which was 0).
    assert int(s.class_ids[5]) == 0


def test_reassign_with_negative_target_inst_allocates_new_id():
    s = _seed()
    out = s.apply_reassign(np.array([0, 6], dtype=np.int32),
                           target_inst=-1, target_class=2)
    new_id = out["new_instance_id"]
    assert int(s.instance_ids[0]) == new_id
    assert int(s.class_ids[6]) == 2


def test_fresh_instance_ids_are_never_reused_after_undo():
    # An undone apply's id may still be referenced by the frontend's instance
    # doc (the row survives Ctrl+Z until reconciliation); re-issuing it would
    # cross-link two instances. Fresh ids must be monotonic for the session.
    s = _seed()
    out1 = s.apply_reassign(np.array([0], dtype=np.int32),
                            target_inst=-1, target_class=2)
    s.undo()
    out2 = s.apply_reassign(np.array([6], dtype=np.int32),
                            target_inst=-1, target_class=2)
    assert out2["new_instance_id"] > out1["new_instance_id"]


def test_reassign_skips_points_in_protected_instances():
    # "Confirmed = locked": a reassign must never overwrite points that belong
    # to a protected (confirmed) instance. _seed: idx 1,2 -> inst 0; idx 3,4 ->
    # inst 1. Protecting inst 0 while reassigning [1,2,3,4] must touch only 3,4.
    s = _seed()
    out = s.apply_reassign(np.array([1, 2, 3, 4], dtype=np.int32),
                           target_inst=-1, target_class=5,
                           protect_instances=[0])
    # Protected inst-0 points untouched.
    assert int(s.class_ids[1]) == 0 and int(s.instance_ids[1]) == 0
    assert int(s.class_ids[2]) == 0 and int(s.instance_ids[2]) == 0
    # Unprotected inst-1 points reassigned to the fresh instance + class 5.
    new_id = out["new_instance_id"]
    assert int(s.instance_ids[3]) == new_id and int(s.class_ids[3]) == 5
    assert int(s.instance_ids[4]) == new_id and int(s.class_ids[4]) == 5
    assert out["n_affected"] == 2
    assert out["n_protected"] == 2


def test_reassign_all_protected_is_a_noop():
    # Every candidate point is locked -> no mutation, no fresh id, nothing dirty.
    s = _seed()
    s.dirty = False
    out = s.apply_reassign(np.array([1, 2], dtype=np.int32),
                           target_inst=-1, target_class=5,
                           protect_instances=[0])
    assert out["n_affected"] == 0
    assert out["n_protected"] == 2
    assert "new_instance_id" not in out
    assert int(s.class_ids[1]) == 0 and int(s.instance_ids[1]) == 0
    assert s.dirty is False


def test_reassign_never_protects_its_own_target_inst():
    # Re-applying an existing box reuses its instance id; passing that id in
    # protect_instances (defensive frontend) must not make the write a no-op.
    s = _seed()
    out = s.apply_reassign(np.array([3, 4], dtype=np.int32),
                           target_inst=1, target_class=5,
                           protect_instances=[1])
    assert out["n_affected"] == 2
    assert int(s.class_ids[3]) == 5 and int(s.instance_ids[3]) == 1


def test_reassign_never_protects_unlabeled_points():
    # A stray -1 in protect_instances must not lock every unlabeled point out of
    # labeling — unlabeled points are always the target of a fresh apply.
    s = _seed()  # idx 0, 6 are unlabeled (-1, -1)
    out = s.apply_reassign(np.array([0, 6], dtype=np.int32),
                           target_inst=-1, target_class=2,
                           protect_instances=[-1])
    assert out["n_affected"] == 2
    assert out["n_protected"] == 0
    assert int(s.class_ids[0]) == 2 and int(s.class_ids[6]) == 2


def test_reassign_with_both_none_erases_to_unlabeled():
    s = _seed()
    s.apply_reassign(np.array([1, 2], dtype=np.int32),
                     target_inst=None, target_class=None)
    assert int(s.instance_ids[1]) == -1
    assert int(s.class_ids[1]) == -1


def test_reassign_raises_when_target_class_missing_but_target_inst_set():
    """A non-erase reassign without a target class would risk invariant 4."""
    s = _seed()
    with pytest.raises(ValueError, match="target_class"):
        s.apply_reassign(np.array([1], dtype=np.int32),
                         target_inst=-1, target_class=None)


def test_brush_query_returns_indices_within_radius():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((1000, 3)).astype(np.float32)
    s = SegmentSession(
        class_ids=np.full(1000, -1, dtype=np.int8),
        instance_ids=np.full(1000, -1, dtype=np.int32),
        positions=pts,
    )
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    R = 0.5
    got = set(s.brush_query(center, R).tolist())
    expected = set(np.flatnonzero(np.linalg.norm(pts - center, axis=1) <= R).tolist())
    assert got == expected


def test_is_from_prelabel_stored_on_session():
    cls = np.array([0, 1], dtype=np.int8)
    inst = np.array([0, 1], dtype=np.int32)
    pos = np.zeros((2, 3), dtype=np.float32)
    s = SegmentSession(class_ids=cls, instance_ids=inst, positions=pos,
                       is_from_prelabel=True)
    assert s.is_from_prelabel is True
    s2 = SegmentSession(class_ids=cls, instance_ids=inst, positions=pos)
    assert s2.is_from_prelabel is False


def test_brush_query_depth_cull_excludes_far_points_along_ray():
    # Place a near cluster at z=0 and a far cluster at z=10, both within sphere R.
    near = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float32)
    far = np.array([[0.0, 0.0, 10.0], [0.01, 0.0, 10.0]], dtype=np.float32)
    pts = np.concatenate([near, far], axis=0)
    s = SegmentSession(
        class_ids=np.full(4, -1, dtype=np.int8),
        instance_ids=np.full(4, -1, dtype=np.int32),
        positions=pts,
    )
    # Sphere big enough to cover both clusters; depth-cull along +z eliminates far.
    got = s.brush_query(
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        radius=20.0,
        camera_ray=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        depth_cull=2.0,
    )
    assert set(got.tolist()) == {0, 1}


def test_segment_session_has_preseg_layer():
    import numpy as np
    from labeling.segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.full(10, -1, dtype=np.int8),
        instance_ids=np.full(10, -1, dtype=np.int32),
        positions=pts,
    )
    assert s.preseg_ids.shape == (10,)
    assert s.preseg_ids.dtype == np.int32
    assert (s.preseg_ids == -1).all()
    assert s.preseg_id is None
    assert s.preseg_fingerprint is None


def test_preseg_layer_immutable_through_merge():
    import numpy as np
    from labeling.segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(10, dtype=np.int8),
        instance_ids=np.array([0]*5 + [1]*5, dtype=np.int32),
        positions=pts,
    )
    _seed_preseg(s, np.array([0]*5 + [1]*5, dtype=np.int32))
    s.apply_merge(source_inst=0, target_inst=1)
    assert (s.instance_ids == 1).all()
    np.testing.assert_array_equal(s.preseg_ids, np.array([0]*5 + [1]*5))


def test_current_inst_ids_for_preseg_after_merge():
    """Core bug-fix scenario: hide(preseg=0) after a merge resolves to live id 1."""
    import numpy as np
    from labeling.segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(10, dtype=np.int8),
        instance_ids=np.array([0]*5 + [1]*5, dtype=np.int32),
        positions=pts,
    )
    _seed_preseg(s, s.instance_ids.copy())
    s.apply_merge(source_inst=0, target_inst=1)
    assert s.current_inst_ids_for_preseg(0) == {1}
    assert s.current_inst_ids_for_preseg(1) == {1}


def test_hide_unhide_inst():
    import numpy as np
    from labeling.segment_state import SegmentSession
    pts = np.zeros((4, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(4, dtype=np.int8),
        instance_ids=np.array([0, 0, 1, 1], dtype=np.int32),
        positions=pts,
    )
    s.hide_instance(0)
    s.hide_instance(1)
    assert s.hidden_inst_ids == {0, 1}
    s.unhide_instance(0)
    assert s.hidden_inst_ids == {1}
    s.unhide_instance(999)  # no-op
    assert s.hidden_inst_ids == {1}


def test_hide_survives_merge():
    """preseg=0 hidden → merge into preseg=1's live id → hide still resolves to the merged live id."""
    import numpy as np
    from labeling.segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(10, dtype=np.int8),
        instance_ids=np.array([0]*5 + [1]*5, dtype=np.int32),
        positions=pts,
    )
    _seed_preseg(s, s.instance_ids.copy())
    s.hide_instance(0)
    s.apply_merge(source_inst=0, target_inst=1)
    assert s.current_inst_ids_for_preseg(0) == {1}


def test_snap_to_preseg_reverts_merged_object():
    import numpy as np
    from labeling.segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.full(10, 2, dtype=np.int8),
        instance_ids=np.array([0]*5 + [1]*5, dtype=np.int32),
        positions=pts,
    )
    _seed_preseg(s, np.array([0]*5 + [1]*5, dtype=np.int32))
    s.apply_merge(source_inst=0, target_inst=1)
    assert (s.instance_ids == 1).all()
    s.snap_to_preseg([1])
    np.testing.assert_array_equal(s.instance_ids, np.array([0]*5 + [1]*5))


def test_snap_to_preseg_undoable():
    import numpy as np
    from labeling.segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(10, dtype=np.int8),
        instance_ids=np.array([0]*5 + [1]*5, dtype=np.int32),
        positions=pts,
    )
    _seed_preseg(s, np.array([0]*5 + [1]*5, dtype=np.int32))
    s.apply_merge(0, 1)
    s.snap_to_preseg([1])
    s.undo()
    assert (s.instance_ids == 1).all()


def test_autosave_writes_working_files_and_session_json(tmp_path):
    import numpy as np, json
    from labeling.segment_state import SegmentSession
    pts = np.zeros((4, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.full(4, -1, dtype=np.int8),
        instance_ids=np.full(4, -1, dtype=np.int32),
        positions=pts,
        session_dir=tmp_path,
        autosave_debounce_s=0.0,
    )
    s.apply_set_class(np.array([0, 1], dtype=np.int32), class_id=2)
    s.flush_autosave()
    assert (tmp_path / "session.json").exists()
    assert (tmp_path / "working_class_ids.npy").exists()
    assert (tmp_path / "working_segment_ids.npy").exists()
    payload = json.loads((tmp_path / "session.json").read_text())
    assert payload["dirty"] is True
    assert payload["schema_version"] == 2
    assert "is_from_prelabel" not in payload


def test_autosave_writes_sam_ids_and_sam_segments(tmp_path):
    import numpy as np, json
    from labeling.segment_state import SegmentSession
    pts = np.zeros((4, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.full(4, -1, dtype=np.int8),
        instance_ids=np.full(4, -1, dtype=np.int32),
        positions=pts,
        session_dir=tmp_path,
        autosave_debounce_s=0.0,
    )
    out = s.materialize_sam_segment(np.array([0, 1], dtype=np.int32))
    s.flush_autosave()
    assert (tmp_path / "working_sam_ids.npy").exists()
    assert (tmp_path / "sam_segments.json").exists()
    sam_ids = np.load(tmp_path / "working_sam_ids.npy")
    assert int(sam_ids[0]) == out["sam_seg_id"] and int(sam_ids[1]) == out["sam_seg_id"]
    assert int(sam_ids[2]) == -1 and int(sam_ids[3]) == -1
    payload = json.loads((tmp_path / "sam_segments.json").read_text())
    segments = {seg["id"]: seg for seg in payload["segments"]}
    assert segments[out["sam_seg_id"]]["n_points"] == 2


def test_autosave_includes_hidden_and_preseg_run(tmp_path):
    import numpy as np, json
    from labeling.segment_state import SegmentSession
    pts = np.zeros((4, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(4, dtype=np.int8),
        instance_ids=np.array([0, 0, 1, 1], dtype=np.int32),
        positions=pts,
        session_dir=tmp_path,
        autosave_debounce_s=0.0,
    )
    _seed_preseg(s, np.array([0, 0, 1, 1], dtype=np.int32), preseg_id="r1")
    # In production the pin is set from session.json by from_aux; the
    # autosave payload must carry it through unchanged.
    s.preseg_fingerprint = "sha256:pinned-at-create"
    s.hide_instance(0)
    s.flush_autosave()
    payload = json.loads((tmp_path / "session.json").read_text())
    assert payload["preseg_id"] == "r1"
    assert payload["hidden_inst_ids"] == [0]
    assert payload["preseg_fingerprint"] == "sha256:pinned-at-create"
    assert "is_from_prelabel" not in payload


def test_autosave_disabled_when_no_session_dir(tmp_path):
    import numpy as np
    from labeling.segment_state import SegmentSession
    pts = np.zeros((4, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.full(4, -1, dtype=np.int8),
        instance_ids=np.full(4, -1, dtype=np.int32),
        positions=pts,
    )
    s.apply_set_class(np.array([0], dtype=np.int32), class_id=1)
    assert not (tmp_path / "session.json").exists()


def test_materialize_sam_segment_allocates_fresh_id_and_writes_sam_ids():
    s = _seed()
    out = s.materialize_sam_segment(np.array([0, 6], dtype=np.int32))
    assert out["sam_seg_id"] == 0
    assert out["n_affected"] == 2
    assert out["n_protected"] == 0
    assert int(s.sam_ids[0]) == 0 and int(s.sam_ids[6]) == 0
    # instance_ids/class_ids are untouched — this is the whole point.
    assert int(s.instance_ids[0]) == -1 and int(s.class_ids[0]) == -1
    assert s.sam_segments[0]["n_points"] == 2
    assert s.dirty is False  # not a working-array edit


def test_materialize_sam_segment_ids_increment():
    s = _seed()
    a = s.materialize_sam_segment(np.array([0], dtype=np.int32))
    b = s.materialize_sam_segment(np.array([6], dtype=np.int32))
    assert a["sam_seg_id"] == 0 and b["sam_seg_id"] == 1


def test_materialize_sam_segment_respects_protect_instances():
    s = _seed()
    # Point 3 already belongs to instance 1 (see _seed()); protect it.
    out = s.materialize_sam_segment(np.array([0, 3], dtype=np.int32),
                                     protect_instances=[1])
    assert out["n_affected"] == 1
    assert out["n_protected"] == 1
    assert int(s.sam_ids[3]) == -1   # protected point never got a sam id
    assert int(s.sam_ids[0]) == 0


def test_materialize_sam_segment_all_protected_creates_nothing():
    s = _seed()
    out = s.materialize_sam_segment(np.array([3, 4], dtype=np.int32),
                                     protect_instances=[1])
    assert out["sam_seg_id"] is None
    assert out["n_affected"] == 0
    assert out["n_protected"] == 2
    assert len(s.sam_segments) == 0


def test_materialize_sam_segment_overlap_is_last_write_wins():
    s = _seed()
    a = s.materialize_sam_segment(np.array([0, 6], dtype=np.int32))
    b = s.materialize_sam_segment(np.array([6], dtype=np.int32))  # overlaps a
    assert int(s.sam_ids[6]) == b["sam_seg_id"]
    assert int(s.sam_ids[0]) == a["sam_seg_id"]
    # a's summary shrank from 2 to 1 (point 6 moved to b), not deleted.
    assert s.sam_segments[a["sam_seg_id"]]["n_points"] == 1
    assert s.sam_segments[b["sam_seg_id"]]["n_points"] == 1


def test_materialize_sam_segment_full_overlap_drops_old_summary_entry():
    s = _seed()
    a = s.materialize_sam_segment(np.array([6], dtype=np.int32))
    s.materialize_sam_segment(np.array([6], dtype=np.int32))  # fully re-covers a
    assert a["sam_seg_id"] not in s.sam_segments


def test_apply_reassign_retires_overlapping_sam_ids():
    """Any tool labeling a point (apply_reassign, set_class, merge) must
    retire that point's SAM candidacy — it's no longer up for grabs."""
    s = _seed()
    out = s.materialize_sam_segment(np.array([0, 6], dtype=np.int32))
    sam_id = out["sam_seg_id"]
    s.apply_reassign(np.array([6], dtype=np.int32), target_inst=-1, target_class=0)
    assert int(s.sam_ids[6]) == -1
    assert int(s.sam_ids[0]) == sam_id       # untouched point keeps its candidacy
    assert s.sam_segments[sam_id]["n_points"] == 1


def test_materialize_sam_segment_not_on_undo_stack():
    s = _seed()
    s.materialize_sam_segment(np.array([0], dtype=np.int32))
    assert s.undo() is None  # nothing to undo — matches preseg_ids' non-edit status
