import numpy as np
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from migrate_eval_invariants import migrate_session, strip_orphaned_presegments


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


def test_migrate_session_updates_instances_gt_json_for_review_blob(tmp_path):
    # Reviewer repro: a pre-phase-2 session has a normal instances_gt.json row
    # (cls:"unknown", confirmed:false) for what becomes a converted class-6
    # instance. Migration must null out `cls` on that row so it matches the
    # post-migration arrays (a review blob), or eval-invariant 3 rejects it.
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([6, 6, 2], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([3, 3, 9], dtype=np.int32))
    instances_gt = {
        "scene": "test", "kind": "gt",
        "instances": [
            {"id": "a", "kind": "pointset", "segId": 3, "cls": "unknown", "confirmed": False},
            {"id": "b", "kind": "pointset", "segId": 9, "cls": "pipe", "confirmed": True},
        ],
    }
    (session_dir / "instances_gt.json").write_text(json.dumps(instances_gt))
    migrate_session(session_dir, n_points=3, dry_run=False)
    updated = json.loads((session_dir / "instances_gt.json").read_text())
    row3 = next(r for r in updated["instances"] if r["segId"] == 3)
    row9 = next(r for r in updated["instances"] if r["segId"] == 9)
    assert row3["cls"] is None
    assert row3["confirmed"] is False
    assert row9["cls"] == "pipe"          # untouched, non-converted instance


def test_migrate_session_corrects_confirmed_review_blob_row(tmp_path):
    # A pre-migration row that was BOTH cls:"unknown" AND confirmed:true must
    # have confirmed forced to False post-migration (eval-invariant 9: a
    # confirmed instance can never be a review blob).
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([6], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([3], dtype=np.int32))
    instances_gt = {
        "scene": "test", "kind": "gt",
        "instances": [
            {"id": "a", "kind": "pointset", "segId": 3, "cls": "unknown", "confirmed": True,
             "note": "keep me"},
        ],
    }
    (session_dir / "instances_gt.json").write_text(json.dumps(instances_gt))
    migrate_session(session_dir, n_points=1, dry_run=False)
    updated = json.loads((session_dir / "instances_gt.json").read_text())
    row = updated["instances"][0]
    assert row["cls"] is None
    assert row["confirmed"] is False
    assert row["note"] == "keep me"       # other metadata untouched


def test_migrate_session_dry_run_does_not_write_instances_gt_json(tmp_path):
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([6], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([3], dtype=np.int32))
    instances_gt = {
        "scene": "test", "kind": "gt",
        "instances": [{"id": "a", "kind": "pointset", "segId": 3, "cls": "unknown", "confirmed": False}],
    }
    original_text = json.dumps(instances_gt)
    (session_dir / "instances_gt.json").write_text(original_text)
    migrate_session(session_dir, n_points=1, dry_run=True)
    assert (session_dir / "instances_gt.json").read_text() == original_text


def test_migrate_session_no_instances_gt_json_is_fine(tmp_path):
    # Absence of instances_gt.json (or no matching row) is not itself a
    # violation — the array-level review_blobs entry is authoritative.
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([6, 2], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([3, 9], dtype=np.int32))
    result = migrate_session(session_dir, n_points=2, dry_run=False)
    assert result["n_legacy_converted"] == 1
    assert not (session_dir / "instances_gt.json").exists()


def test_migrate_session_writes_real_category_histogram(tmp_path):
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([6, 6, 2], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([3, 3, 9], dtype=np.int32))
    migrate_session(session_dir, n_points=3, dry_run=False)
    meta = json.loads((session_dir / "output" / "gt_segment_metadata.json").read_text())
    assert meta["categories"] == {"none": 1, "artifact": 0, "transient": 0, "excluded_review": 2}


def test_migrate_session_end_to_end_passes_real_eval_invariant_3(tmp_path):
    # Full round-trip against the REAL scan_schema.eval_invariants gate, not a
    # reimplementation: reproduces the reviewer's exact scenario and asserts
    # it no longer raises after migration.
    from scan_schema.eval_invariants import check_instance_class_consistency
    from labeling.instances_doc import load_instances_for_invariants

    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([6, 6, 2], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([3, 3, 9], dtype=np.int32))
    instances_gt = {
        "scene": "test", "kind": "gt",
        "instances": [
            {"id": "a", "kind": "pointset", "segId": 3, "cls": "unknown", "confirmed": False},
            {"id": "b", "kind": "pointset", "segId": 9, "cls": "pipe", "confirmed": True},
        ],
    }
    (session_dir / "instances_gt.json").write_text(json.dumps(instances_gt))

    migrate_session(session_dir, n_points=3, dry_run=False)

    new_cls = np.load(session_dir / "output" / "gt_class_ids.npy")
    new_inst = np.load(session_dir / "output" / "gt_segment_ids.npy")
    cats = np.load(session_dir / "output" / "gt_point_category.npy")
    meta = json.loads((session_dir / "output" / "gt_segment_metadata.json").read_text())
    instances = load_instances_for_invariants(session_dir)

    # must not raise
    check_instance_class_consistency(
        segment_ids=new_inst, class_ids=new_cls, categories=cats,
        instances=instances, review_blobs=meta["review_blobs"],
    )


def test_migrate_session_end_to_end_confirmed_row_becomes_unconfirmed(tmp_path):
    from scan_schema.eval_invariants import check_instance_class_consistency
    from labeling.instances_doc import load_instances_for_invariants

    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([6], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([3], dtype=np.int32))
    instances_gt = {
        "scene": "test", "kind": "gt",
        "instances": [
            {"id": "a", "kind": "pointset", "segId": 3, "cls": "unknown", "confirmed": True},
        ],
    }
    (session_dir / "instances_gt.json").write_text(json.dumps(instances_gt))

    migrate_session(session_dir, n_points=1, dry_run=False)

    updated = json.loads((session_dir / "instances_gt.json").read_text())
    row = updated["instances"][0]
    assert row["confirmed"] is False
    assert row["cls"] is None

    new_cls = np.load(session_dir / "output" / "gt_class_ids.npy")
    new_inst = np.load(session_dir / "output" / "gt_segment_ids.npy")
    cats = np.load(session_dir / "output" / "gt_point_category.npy")
    meta = json.loads((session_dir / "output" / "gt_segment_metadata.json").read_text())
    instances = load_instances_for_invariants(session_dir)

    check_instance_class_consistency(
        segment_ids=new_inst, class_ids=new_cls, categories=cats,
        instances=instances, review_blobs=meta["review_blobs"],
    )


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


def test_strip_orphaned_presegments_no_instances_doc(tmp_path):
    # No instances_gt.json at all -> must skip, strip nothing
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([4, 4, 2], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([39, 39, 9], dtype=np.int32))
    result = strip_orphaned_presegments(session_dir, n_points=3, dry_run=False)
    assert result["skipped_no_instances_doc"] is True
    cls = np.load(session_dir / "output" / "gt_class_ids.npy")
    assert cls.tolist() == [4, 4, 2]   # untouched


def test_strip_orphaned_presegments_strips_ids_missing_from_doc(tmp_path):
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([4, 4, 2], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([39, 39, 9], dtype=np.int32))
    np.save(session_dir / "output" / "gt_point_category.npy", np.array([0, 0, 0], dtype=np.int8))
    np.save(session_dir / "output" / "gt_point_component_ids.npy", np.array([0, 0, 0], dtype=np.int16))
    (session_dir / "output" / "gt_segment_metadata.json").write_text(json.dumps({
        "n_points": 3, "segments": [
            {"gt_id": 39, "class_id": 4, "n_points": 2},
            {"gt_id": 9, "class_id": 2, "n_points": 1},
        ], "review_blobs": [],
    }))
    # instances_gt.json has a row for segId 9 but NOT for segId 39 -- 39 is orphaned
    (session_dir / "instances_gt.json").write_text(json.dumps({
        "scene": "x", "kind": "gt",
        "instances": [{"id": "a", "kind": "pointset", "segId": 9, "cls": "pipe", "confirmed": True}],
    }))
    result = strip_orphaned_presegments(session_dir, n_points=3, dry_run=False)
    assert result["n_orphaned_ids"] == 1
    assert result["orphaned_ids"] == [39]
    assert result["n_points_stripped"] == 2
    cls = np.load(session_dir / "output" / "gt_class_ids.npy")
    inst = np.load(session_dir / "output" / "gt_segment_ids.npy")
    comp = np.load(session_dir / "output" / "gt_point_component_ids.npy")
    assert cls.tolist() == [-1, -1, 2]
    assert inst.tolist() == [-1, -1, 9]
    assert comp.tolist() == [-1, -1, 0]   # component cleared for stripped points
    meta = json.loads((session_dir / "output" / "gt_segment_metadata.json").read_text())
    assert [s["gt_id"] for s in meta["segments"]] == [9]   # stale entry for 39 removed
    assert meta["review_blobs"] == []   # NOT treated as a review blob


def test_strip_orphaned_presegments_dry_run_writes_nothing(tmp_path):
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([4], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([39], dtype=np.int32))
    (session_dir / "instances_gt.json").write_text(json.dumps({"scene": "x", "kind": "gt", "instances": []}))
    result = strip_orphaned_presegments(session_dir, n_points=1, dry_run=True)
    assert result["n_orphaned_ids"] == 1
    cls = np.load(session_dir / "output" / "gt_class_ids.npy")
    assert cls.tolist() == [4]   # unchanged -- dry run wrote nothing


def test_strip_orphaned_presegments_leaves_non_orphaned_untouched(tmp_path):
    # A session with a mix: instance 5 has a real instances_gt.json row (not
    # orphaned) and must be completely unaffected; instance 39 is orphaned.
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([4, 2, 2], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([39, 5, 5], dtype=np.int32))
    (session_dir / "instances_gt.json").write_text(json.dumps({
        "scene": "x", "kind": "gt",
        "instances": [{"id": "a", "kind": "pointset", "segId": 5, "cls": "pipe", "confirmed": False}],
    }))
    strip_orphaned_presegments(session_dir, n_points=3, dry_run=False)
    cls = np.load(session_dir / "output" / "gt_class_ids.npy")
    inst = np.load(session_dir / "output" / "gt_segment_ids.npy")
    assert cls.tolist() == [-1, 2, 2]
    assert inst.tolist() == [-1, 5, 5]


def test_strip_orphaned_presegments_surfaces_nonnone_category(tmp_path):
    # An orphaned point that unexpectedly carries a non-none category must be
    # surfaced in the result, not silently overwritten or crashed on.
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([4, 2], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([39, 5], dtype=np.int32))
    np.save(session_dir / "output" / "gt_point_category.npy", np.array([1, 0], dtype=np.int8))  # artifact
    (session_dir / "instances_gt.json").write_text(json.dumps({
        "scene": "x", "kind": "gt",
        "instances": [{"id": "a", "kind": "pointset", "segId": 5, "cls": "pipe", "confirmed": False}],
    }))
    result = strip_orphaned_presegments(session_dir, n_points=2, dry_run=False)
    assert result["orphaned_with_nonnone_category"] == 1
    cats = np.load(session_dir / "output" / "gt_point_category.npy")
    assert cats.tolist() == [1, 0]   # category left untouched by this function
