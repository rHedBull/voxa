"""Editing-session state: apply / undo / redo, plus brush_query."""
from __future__ import annotations

import numpy as np
import pytest

from segment_state import SegmentSession


def _seed():
    cls = np.array([-1, 0, 0, 1, 1, 2, -1, 2], dtype=np.int8)
    inst = np.array([-1, 0, 0, 1, 1, 2, -1, 2], dtype=np.int32)
    return SegmentSession(class_ids=cls.copy(), instance_ids=inst.copy(),
                          positions=np.zeros((8, 3), dtype=np.float32))


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
    from segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.full(10, -1, dtype=np.int8),
        instance_ids=np.full(10, -1, dtype=np.int32),
        positions=pts,
    )
    assert s.preseg_ids.shape == (10,)
    assert s.preseg_ids.dtype == np.int32
    assert (s.preseg_ids == -1).all()
    assert s.preseg_run_id is None
    assert s.preseg_fingerprint is None


def test_freeze_preseg_stamps_run_id_and_fingerprint():
    import numpy as np
    from segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.full(10, -1, dtype=np.int8),
        instance_ids=np.full(10, -1, dtype=np.int32),
        positions=pts,
    )
    new_pre = np.arange(10, dtype=np.int32)
    s.freeze_preseg(new_pre, run_id="abc")
    np.testing.assert_array_equal(s.preseg_ids, new_pre)
    assert s.preseg_run_id == "abc"
    assert s.preseg_fingerprint.startswith("sha256:")


def test_freeze_preseg_immutable_through_merge():
    import numpy as np
    from segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(10, dtype=np.int8),
        instance_ids=np.array([0]*5 + [1]*5, dtype=np.int32),
        positions=pts,
    )
    s.freeze_preseg(np.array([0]*5 + [1]*5, dtype=np.int32))
    s.apply_merge(source_inst=0, target_inst=1)
    assert (s.instance_ids == 1).all()
    np.testing.assert_array_equal(s.preseg_ids, np.array([0]*5 + [1]*5))


def test_current_inst_ids_for_preseg_after_merge():
    """Core bug-fix scenario: hide(preseg=0) after a merge resolves to live id 1."""
    import numpy as np
    from segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(10, dtype=np.int8),
        instance_ids=np.array([0]*5 + [1]*5, dtype=np.int32),
        positions=pts,
    )
    s.freeze_preseg(s.instance_ids.copy())
    s.apply_merge(source_inst=0, target_inst=1)
    assert s.current_inst_ids_for_preseg(0) == {1}
    assert s.current_inst_ids_for_preseg(1) == {1}


def test_hide_unhide_inst():
    import numpy as np
    from segment_state import SegmentSession
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
    from segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(10, dtype=np.int8),
        instance_ids=np.array([0]*5 + [1]*5, dtype=np.int32),
        positions=pts,
    )
    s.freeze_preseg(s.instance_ids.copy())
    s.hide_instance(0)
    s.apply_merge(source_inst=0, target_inst=1)
    assert s.current_inst_ids_for_preseg(0) == {1}


def test_snap_to_preseg_reverts_merged_object():
    import numpy as np
    from segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.full(10, 2, dtype=np.int8),
        instance_ids=np.array([0]*5 + [1]*5, dtype=np.int32),
        positions=pts,
    )
    s.freeze_preseg(np.array([0]*5 + [1]*5, dtype=np.int32))
    s.apply_merge(source_inst=0, target_inst=1)
    assert (s.instance_ids == 1).all()
    s.snap_to_preseg([1])
    np.testing.assert_array_equal(s.instance_ids, np.array([0]*5 + [1]*5))


def test_snap_to_preseg_undoable():
    import numpy as np
    from segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(10, dtype=np.int8),
        instance_ids=np.array([0]*5 + [1]*5, dtype=np.int32),
        positions=pts,
    )
    s.freeze_preseg(np.array([0]*5 + [1]*5, dtype=np.int32))
    s.apply_merge(0, 1)
    s.snap_to_preseg([1])
    s.undo()
    assert (s.instance_ids == 1).all()


def test_autosave_writes_working_files_and_current_json(tmp_path):
    import numpy as np, json
    from segment_state import SegmentSession
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
    assert (tmp_path / "current.json").exists()
    assert (tmp_path / "working_class_ids.npy").exists()
    assert (tmp_path / "working_segment_ids.npy").exists()
    payload = json.loads((tmp_path / "current.json").read_text())
    assert payload["dirty"] is True
    assert payload["schema_version"] == 1


def test_autosave_includes_hidden_and_preseg_run(tmp_path):
    import numpy as np, json
    from segment_state import SegmentSession
    pts = np.zeros((4, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(4, dtype=np.int8),
        instance_ids=np.array([0, 0, 1, 1], dtype=np.int32),
        positions=pts,
        session_dir=tmp_path,
        autosave_debounce_s=0.0,
    )
    s.freeze_preseg(np.array([0, 0, 1, 1], dtype=np.int32), run_id="r1")
    s.hide_instance(0)
    s.flush_autosave()
    payload = json.loads((tmp_path / "current.json").read_text())
    assert payload["preseg_run_id"] == "r1"
    assert payload["hidden_inst_ids"] == [0]
    assert payload["preseg_fingerprint"].startswith("sha256:")


def test_autosave_disabled_when_no_session_dir(tmp_path):
    import numpy as np
    from segment_state import SegmentSession
    pts = np.zeros((4, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.full(4, -1, dtype=np.int8),
        instance_ids=np.full(4, -1, dtype=np.int32),
        positions=pts,
    )
    s.apply_set_class(np.array([0], dtype=np.int32), class_id=1)
    assert not (tmp_path / "current.json").exists()
