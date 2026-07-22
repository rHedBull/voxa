import numpy as np
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from migrate_eval_invariants import migrate_session


def test_migrate_session_backfills_missing_arrays(tmp_path):
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    n = 5
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([2, 2, -1, -1, -1], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([0, 0, -1, -1, -1], dtype=np.int32))
    migrate_session(session_dir, n_points=n, dry_run=False)
    cats = np.load(session_dir / "output" / "gt_point_category.npy")
    comps = np.load(session_dir / "output" / "gt_point_component_ids.npy")
    assert (cats == 0).all()          # all backfilled to `none`
    assert comps.tolist() == [0, 0, -1, -1, -1]   # one component per instance


def test_migrate_session_converts_legacy_class_6(tmp_path):
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([6, 6, 2], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([3, 3, 9], dtype=np.int32))
    migrate_session(session_dir, n_points=3, dry_run=False)
    new_cls = np.load(session_dir / "output" / "gt_class_ids.npy")
    new_inst = np.load(session_dir / "output" / "gt_segment_ids.npy")
    cats = np.load(session_dir / "output" / "gt_point_category.npy")
    assert new_cls.tolist() == [-1, -1, 2]          # class-6 points erased
    assert new_inst.tolist() == [-1, -1, 9]         # instance stripped (review blob, class-less)
    assert cats.tolist() == [3, 3, 0]               # excluded_review
    meta = json.loads((session_dir / "output" / "gt_segment_metadata.json").read_text())
    assert any(b["instance_id"] == 3 for b in meta["review_blobs"])
    assert meta["review_blobs"][0]["n_points"] == 2   # correctness check on the count, not just presence


def test_migrate_session_dry_run_writes_nothing(tmp_path):
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([2], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([0], dtype=np.int32))
    migrate_session(session_dir, n_points=1, dry_run=True)
    assert not (session_dir / "output" / "gt_point_category.npy").exists()


def test_migrate_session_preserves_non_legacy_points_byte_for_byte(tmp_path):
    # The core "additive-only except class-6" guarantee: any point whose class
    # is NOT 6 must be byte-for-byte unchanged (class id, instance id) after
    # migration, even if other frozen legacy classes (0/3/5/13) are present.
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    orig_cls = np.array([0, 3, 5, 13, 2, 6], dtype=np.int32)   # mix of frozen (non-6) + real + legacy-6
    orig_inst = np.array([1, 2, 3, 4, 5, 6], dtype=np.int32)
    np.save(session_dir / "output" / "gt_class_ids.npy", orig_cls.copy())
    np.save(session_dir / "output" / "gt_segment_ids.npy", orig_inst.copy())
    migrate_session(session_dir, n_points=6, dry_run=False)
    new_cls = np.load(session_dir / "output" / "gt_class_ids.npy")
    new_inst = np.load(session_dir / "output" / "gt_segment_ids.npy")
    non_legacy_6 = orig_cls != 6
    assert np.array_equal(new_cls[non_legacy_6], orig_cls[non_legacy_6])
    assert np.array_equal(new_inst[non_legacy_6], orig_inst[non_legacy_6])
    assert new_cls[-1] == -1 and new_inst[-1] == -1   # the class-6 point IS changed


def test_migrate_session_second_run_is_idempotent(tmp_path):
    # Running the migration twice on an already-migrated session must not
    # error and must not change anything further (no legacy-6 points remain
    # after the first run).
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([6, 2], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([3, 9], dtype=np.int32))
    migrate_session(session_dir, n_points=2, dry_run=False)
    result2 = migrate_session(session_dir, n_points=2, dry_run=False)
    assert result2["n_legacy_converted"] == 0
    cls_after = np.load(session_dir / "output" / "gt_class_ids.npy")
    assert cls_after.tolist() == [-1, 2]
