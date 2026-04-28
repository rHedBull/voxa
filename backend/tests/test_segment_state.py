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
