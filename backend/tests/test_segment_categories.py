# backend/tests/test_segment_categories.py
import numpy as np
import pytest

from labeling.categories import (
    CATEGORY_ARTIFACT,
    CATEGORY_EXCLUDED_REVIEW,
    CATEGORY_NONE,
    CATEGORY_TRANSIENT,
)
from labeling.segment_state import SegmentSession


def _session(n=50):
    pos = np.random.default_rng(0).normal(size=(n, 3)).astype(np.float32)
    return SegmentSession(np.full(n, -1, np.int8), np.full(n, -1, np.int32), pos)


def test_fresh_session_has_all_none_categories():
    seg = _session()
    assert seg.categories.dtype == np.int8
    assert seg.categories.shape == (50,)
    assert bool((seg.categories == CATEGORY_NONE).all())


def test_mark_artifact_erases_class_and_instance():
    seg = _session()
    idx = np.arange(0, 10, dtype=np.int32)
    seg.apply_reassign(idx, target_inst=-1, target_class=3)
    out = seg.apply_category(idx, CATEGORY_ARTIFACT)
    assert out["n_affected"] == 10
    assert out.get("new_instance_id") is None
    assert bool((seg.categories[idx] == CATEGORY_ARTIFACT).all())
    assert bool((seg.class_ids[idx] == -1).all())
    assert bool((seg.instance_ids[idx] == -1).all())


def test_mark_review_allocates_one_blob_instance():
    seg = _session()
    idx = np.arange(0, 8, dtype=np.int32)
    out = seg.apply_category(idx, CATEGORY_EXCLUDED_REVIEW)
    blob = out["new_instance_id"]
    assert blob is not None and blob >= 0
    assert bool((seg.instance_ids[idx] == blob).all())
    # class stays -1: a review blob carries shape + note, never a class
    assert bool((seg.class_ids[idx] == -1).all())
    assert bool((seg.categories[idx] == CATEGORY_EXCLUDED_REVIEW).all())
    # ids stay session-monotonic (the stable-id contract)
    second = seg.apply_category(np.arange(20, 25, dtype=np.int32),
                                CATEGORY_EXCLUDED_REVIEW)
    assert second["new_instance_id"] > blob


def test_clear_category_returns_points_to_unlabeled():
    seg = _session()
    idx = np.arange(0, 5, dtype=np.int32)
    blob = seg.apply_category(idx, CATEGORY_EXCLUDED_REVIEW)["new_instance_id"]
    seg.apply_category(idx, CATEGORY_NONE)
    assert bool((seg.categories[idx] == CATEGORY_NONE).all())
    assert bool((seg.instance_ids[idx] == -1).all())
    assert int((seg.instance_ids == blob).sum()) == 0


def test_protect_instances_blocks_confirmed_points():
    seg = _session()
    keep = np.arange(0, 5, dtype=np.int32)
    confirmed = seg.apply_reassign(keep, target_inst=-1, target_class=2)["new_instance_id"]
    idx = np.arange(0, 10, dtype=np.int32)
    out = seg.apply_category(idx, CATEGORY_TRANSIENT, protect_instances=[confirmed])
    assert out["n_protected"] == 5
    assert out["n_affected"] == 5
    assert bool((seg.instance_ids[keep] == confirmed).all())
    assert bool((seg.categories[keep] == CATEGORY_NONE).all())
    assert bool((seg.categories[np.arange(5, 10)] == CATEGORY_TRANSIENT).all())


def test_fully_protected_mark_is_a_no_op():
    seg = _session()
    idx = np.arange(0, 5, dtype=np.int32)
    inst = seg.apply_reassign(idx, target_inst=-1, target_class=2)["new_instance_id"]
    before = seg.categories.copy()
    out = seg.apply_category(idx, CATEGORY_ARTIFACT, protect_instances=[inst])
    assert out["n_affected"] == 0 and out["n_protected"] == 5
    assert np.array_equal(seg.categories, before)


def test_undo_redo_restores_categories():
    seg = _session()
    idx = np.arange(0, 6, dtype=np.int32)
    seg.apply_reassign(idx, target_inst=-1, target_class=1)
    seg.apply_category(idx, CATEGORY_ARTIFACT)
    d = seg.undo()
    assert bool((seg.categories[idx] == CATEGORY_NONE).all())
    assert bool((seg.class_ids[idx] == 1).all())
    assert "after_category" in d
    seg.redo()
    assert bool((seg.categories[idx] == CATEGORY_ARTIFACT).all())
    assert bool((seg.class_ids[idx] == -1).all())


def test_labeling_a_marked_point_clears_its_category():
    seg = _session()
    idx = np.arange(0, 6, dtype=np.int32)
    seg.apply_category(idx, CATEGORY_ARTIFACT)
    seg.apply_reassign(idx, target_inst=-1, target_class=1)
    assert bool((seg.categories[idx] == CATEGORY_NONE).all())
    # ...and undoing the label brings the mark back
    seg.undo()
    assert bool((seg.categories[idx] == CATEGORY_ARTIFACT).all())


def test_empty_indices_is_a_no_op():
    seg = _session()
    out = seg.apply_category(np.empty(0, np.int32), CATEGORY_ARTIFACT)
    assert out["n_affected"] == 0
    assert seg.dirty is False


def test_unknown_category_raises():
    seg = _session()
    with pytest.raises(ValueError):
        seg.apply_category(np.arange(3, dtype=np.int32), 9)


def test_artifact_with_allocate_instance_mints_a_blob():
    seg = _session()
    idx = np.arange(0, 8, dtype=np.int32)
    out = seg.apply_category(idx, CATEGORY_ARTIFACT, allocate_instance=True)
    assert out["n_affected"] == 8
    assert out.get("new_instance_id") is not None
    blob = out["new_instance_id"]
    assert bool((seg.categories[idx] == CATEGORY_ARTIFACT).all())
    assert bool((seg.class_ids[idx] == -1).all())
    assert bool((seg.instance_ids[idx] == blob).all())


def test_artifact_default_still_erases_instance():
    seg = _session()
    idx = np.arange(0, 8, dtype=np.int32)
    out = seg.apply_category(idx, CATEGORY_ARTIFACT)
    assert out.get("new_instance_id") is None
    assert bool((seg.instance_ids[idx] == -1).all())


def test_review_default_still_allocates_blob():
    seg = _session()
    idx = np.arange(0, 8, dtype=np.int32)
    out = seg.apply_category(idx, CATEGORY_EXCLUDED_REVIEW)
    assert out.get("new_instance_id") is not None


def test_allocate_instance_false_forces_no_blob_on_review():
    seg = _session()
    idx = np.arange(0, 8, dtype=np.int32)
    out = seg.apply_category(idx, CATEGORY_EXCLUDED_REVIEW, allocate_instance=False)
    assert out.get("new_instance_id") is None
    assert bool((seg.instance_ids[idx] == -1).all())
