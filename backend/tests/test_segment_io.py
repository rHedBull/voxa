"""Tests for prelabel ingestion + label save."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np

from labeling.segment_io import (
    atomic_write_json,
    atomic_write_npy,
    compute_fingerprint,
    load_prelabel,
    load_session_aux,
    load_working_arrays,
    prune_history,
    save_labels,
    save_session_aux,
)


def _write_prelabel(scan_dir: Path, instance_ids: np.ndarray, summary: list[dict]):
    pre = scan_dir / "prelabel"
    pre.mkdir(parents=True, exist_ok=True)
    np.save(pre / "ransac_instance_ids.npy", instance_ids.astype(np.int32))
    (pre / "ransac_segment_summary.json").write_text(
        json.dumps({"segments": summary})
    )


def test_load_prelabel_returns_aligned_arrays(tmp_path):
    scan_dir = tmp_path / "annotated" / "demo"
    inst = np.array([-1, 0, 0, 1, 1, 2, -1, 2], dtype=np.int32)
    summary = [
        {"id": 0, "class_id": 0},
        {"id": 1, "class_id": 1},
        {"id": 2, "class_id": 2},
    ]
    _write_prelabel(scan_dir, inst, summary)

    out = load_prelabel(scan_dir, n_points=8)
    assert out is not None
    cls, ii = out
    assert ii.dtype == np.int32 and ii.shape == (8,)
    assert cls.dtype == np.int8 and cls.shape == (8,)
    np.testing.assert_array_equal(ii, inst)
    np.testing.assert_array_equal(cls, np.array([-1, 0, 0, 1, 1, 2, -1, 2], dtype=np.int8))


def test_load_prelabel_returns_none_when_missing(tmp_path):
    scan_dir = tmp_path / "annotated" / "demo"
    scan_dir.mkdir(parents=True)
    assert load_prelabel(scan_dir, n_points=8) is None


def test_load_prelabel_returns_none_on_size_mismatch(tmp_path):
    scan_dir = tmp_path / "annotated" / "demo"
    _write_prelabel(scan_dir, np.zeros(7, dtype=np.int32), [])
    assert load_prelabel(scan_dir, n_points=8) is None


def test_load_prelabel_returns_none_when_summary_is_not_a_dict(tmp_path):
    scan_dir = tmp_path / "annotated" / "demo"
    pre = scan_dir / "prelabel"
    pre.mkdir(parents=True, exist_ok=True)
    np.save(pre / "ransac_instance_ids.npy", np.zeros(8, dtype=np.int32))
    (pre / "ransac_segment_summary.json").write_text("[]")
    assert load_prelabel(scan_dir, n_points=8) is None


def test_load_prelabel_returns_none_when_segment_missing_keys(tmp_path):
    scan_dir = tmp_path / "annotated" / "demo"
    _write_prelabel(scan_dir, np.zeros(8, dtype=np.int32),
                    [{"id": 0}])  # class_id missing
    assert load_prelabel(scan_dir, n_points=8) is None


def _read_npy(path: Path) -> np.ndarray:
    return np.load(path)


def test_save_labels_writes_aligned_arrays_and_metadata(tmp_path):
    scan_dir = tmp_path / "annotated" / "demo"
    cls = np.array([-1, 0, 0, 1, 1], dtype=np.int8)
    inst = np.array([-1, 0, 0, 1, 1], dtype=np.int32)
    save_labels(scan_dir, cls, inst, write_history=False)

    np.testing.assert_array_equal(
        _read_npy(scan_dir / "labels" / "gt_class_ids.npy"), cls.astype(np.int32),
    )
    np.testing.assert_array_equal(
        _read_npy(scan_dir / "labels" / "gt_segment_ids.npy"), inst.astype(np.int32),
    )
    meta = json.loads((scan_dir / "labels" / "gt_segment_metadata.json").read_text())
    assert meta["n_points"] == 5
    assert meta["n_labeled_points"] == 4
    assert meta["n_gt_segments"] == 2
    seg_ids = sorted(s["gt_id"] for s in meta["segments"])
    assert seg_ids == [0, 1]


def test_save_labels_rejects_invariant_violation(tmp_path):
    import pytest
    scan_dir = tmp_path / "annotated" / "demo"
    cls = np.array([0, 0], dtype=np.int8)
    inst = np.array([-1, 0], dtype=np.int32)
    with pytest.raises(ValueError, match="invariant"):
        save_labels(scan_dir, cls, inst, write_history=False)


def test_save_labels_rejects_class_inconsistency(tmp_path):
    import pytest
    scan_dir = tmp_path / "annotated" / "demo"
    cls = np.array([0, 1], dtype=np.int8)
    inst = np.array([0, 0], dtype=np.int32)
    with pytest.raises(ValueError, match="invariant"):
        save_labels(scan_dir, cls, inst, write_history=False)


def test_save_labels_writes_history_snapshot(tmp_path):
    scan_dir = tmp_path / "annotated" / "demo"
    save_labels(scan_dir, np.array([0], dtype=np.int8), np.array([0], dtype=np.int32),
                write_history=True)
    hist = scan_dir / "annotation_history"
    assert hist.exists()
    snaps = list(hist.iterdir())
    assert len(snaps) == 1
    assert re.match(r"^\d{8}_\d{6}$", snaps[0].name)


def test_save_labels_snapshots_existing_labels_before_overwrite(tmp_path):
    """v1 → save → v2 → save: v1's class array must be preserved in history."""
    scan_dir = tmp_path / "annotated" / "demo"
    cls_v1 = np.array([0], dtype=np.int8)
    inst_v1 = np.array([0], dtype=np.int32)
    save_labels(scan_dir, cls_v1, inst_v1, write_history=False)

    cls_v2 = np.array([1], dtype=np.int8)
    inst_v2 = np.array([1], dtype=np.int32)
    save_labels(scan_dir, cls_v2, inst_v2, write_history=True)

    snaps = list((scan_dir / "annotation_history").iterdir())
    assert len(snaps) == 1
    saved_cls = np.load(snaps[0] / "gt_class_ids.npy")
    np.testing.assert_array_equal(saved_cls, cls_v1.astype(np.int32))
    saved_meta = json.loads((snaps[0] / "gt_segment_metadata.json").read_text())
    assert saved_meta["segments"][0]["class_id"] == 0


def _write_classes_json(lidar_root: Path, version: int, classes: list[dict]):
    lidar_root.mkdir(parents=True, exist_ok=True)
    (lidar_root / "classes.json").write_text(json.dumps({
        "version": version, "unlabeled_id": -1, "classes": classes,
    }))


def _write_meta(scan_dir: Path, version: int):
    scan_dir.mkdir(parents=True, exist_ok=True)
    (scan_dir / "meta.json").write_text(json.dumps({"class_map_version": version}))


def test_save_labels_enriches_segment_metadata_with_label_from_classes_json(tmp_path):
    lidar_root = tmp_path / "lidar"
    scan_dir = lidar_root / "annotated" / "demo"
    _write_classes_json(lidar_root, 2, [
        {"id": 0, "name": "pipe"}, {"id": 1, "name": "tank"},
    ])
    save_labels(scan_dir,
                np.array([0, 1], dtype=np.int8),
                np.array([0, 1], dtype=np.int32),
                write_history=False)

    meta = json.loads((scan_dir / "labels" / "gt_segment_metadata.json").read_text())
    assert meta["class_map_version"] == 2
    by_id = {s["gt_id"]: s for s in meta["segments"]}
    assert by_id[0]["label"] == "pipe"
    assert by_id[1]["label"] == "tank"


def test_save_labels_rejects_unknown_class_id_when_registry_present(tmp_path):
    import pytest
    lidar_root = tmp_path / "lidar"
    scan_dir = lidar_root / "annotated" / "demo"
    _write_classes_json(lidar_root, 1, [{"id": 0, "name": "pipe"}])
    with pytest.raises(ValueError, match="invariant 5"):
        save_labels(scan_dir,
                    np.array([7], dtype=np.int8),
                    np.array([0], dtype=np.int32),
                    write_history=False)


def test_save_labels_rejects_class_map_version_mismatch(tmp_path):
    import pytest
    lidar_root = tmp_path / "lidar"
    scan_dir = lidar_root / "annotated" / "demo"
    _write_classes_json(lidar_root, 2, [{"id": 0, "name": "pipe"}])
    _write_meta(scan_dir, version=1)
    with pytest.raises(ValueError, match="invariant 6"):
        save_labels(scan_dir,
                    np.array([0], dtype=np.int8),
                    np.array([0], dtype=np.int32),
                    write_history=False)


def test_compute_fingerprint_is_content_addressed():
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([1, 2, 3], dtype=np.int32)
    c = np.array([1, 2, 4], dtype=np.int32)
    assert compute_fingerprint(a) == compute_fingerprint(b)
    assert compute_fingerprint(a) != compute_fingerprint(c)
    assert compute_fingerprint(a).startswith("sha256:")


def test_compute_fingerprint_handles_non_contiguous_views():
    base = np.array([[1, 2, 3], [4, 5, 6]], dtype=np.int32)
    view = base.T  # non-contiguous
    contig = np.ascontiguousarray(view)
    assert compute_fingerprint(view) == compute_fingerprint(contig)


def test_atomic_write_npy_round_trip(tmp_path):
    p = tmp_path / "x.npy"
    arr = np.arange(100, dtype=np.int32)
    atomic_write_npy(p, arr)
    assert p.exists()
    assert not (tmp_path / "x.npy.tmp").exists()
    np.testing.assert_array_equal(np.load(p), arr)


def test_atomic_write_json_round_trip(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_json(p, {"a": 1, "b": [2, 3]})
    assert p.exists()
    assert not (tmp_path / "x.json.tmp").exists()
    assert json.loads(p.read_text()) == {"a": 1, "b": [2, 3]}


def test_prune_history_keeps_only_timestamped_dirs(tmp_path):
    hist = tmp_path / "annotation_history"
    hist.mkdir()
    valid = [hist / f"2026010{i % 10}_{100000 + i:06d}" for i in range(12)]
    for v in valid:
        v.mkdir()
    for i, v in enumerate(valid):
        os.utime(v, (1_700_000_000 + i, 1_700_000_000 + i))
    user = hist / "manual-backup"; user.mkdir()

    prune_history(hist, keep=10)

    remaining = sorted(p.name for p in hist.iterdir())
    assert "manual-backup" in remaining
    assert sum(1 for n in remaining if re.match(r"^\d{8}_\d{6}$", n)) == 10


def test_session_aux_round_trip(tmp_path):
    session_dir = tmp_path / "session"
    class_ids = np.full(100, -1, dtype=np.int8)
    class_ids[10:20] = 2
    inst_ids = np.full(100, -1, dtype=np.int32)
    inst_ids[10:20] = 7
    aux = {
        "preseg_id": "20260513-100000_ransac",
        "preseg_fingerprint": "sha256:abc",
        "source_fingerprint": "sha256:def",
        "hidden_inst_ids": [7],
        "dirty": True,
        "name": "test session",
        "created_at": "2026-05-13T10:00:00+00:00",
    }
    save_session_aux(session_dir, aux, class_ids=class_ids, instance_ids=inst_ids)
    assert (session_dir / "session.json").exists()
    assert (session_dir / "working_class_ids.npy").exists()
    assert (session_dir / "working_segment_ids.npy").exists()

    out = load_session_aux(session_dir)
    assert out is not None
    assert out["preseg_id"] == "20260513-100000_ransac"
    assert out["hidden_inst_ids"] == [7]

    wc, wi = load_working_arrays(session_dir, n_points=100)
    np.testing.assert_array_equal(wc, class_ids)
    np.testing.assert_array_equal(wi, inst_ids)


def test_load_working_arrays_returns_none_without_session_json(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    # Working arrays present but no session.json → ignore (commit-pointer rule).
    atomic_write_npy(session_dir / "working_class_ids.npy",
                     np.zeros(50, dtype=np.int8))
    atomic_write_npy(session_dir / "working_segment_ids.npy",
                     np.zeros(50, dtype=np.int32))
    assert load_working_arrays(session_dir, n_points=50) is None


def test_load_working_arrays_returns_none_on_shape_mismatch(tmp_path):
    session_dir = tmp_path / "session"
    save_session_aux(session_dir, {},
                     class_ids=np.zeros(50, dtype=np.int8),
                     instance_ids=np.zeros(50, dtype=np.int32))
    assert load_working_arrays(session_dir, n_points=999) is None


def test_save_labels_adds_fingerprints(tmp_path):
    """gt_segment_metadata.json must carry prelabel_fingerprint + source_fingerprint
    when supplied by the caller."""
    from labeling.segment_io import save_labels
    scan = tmp_path
    # minimal class registry so validators pass
    (scan / "labels").mkdir()
    class_ids = np.full(4, -1, dtype=np.int32)
    inst_ids = np.full(4, -1, dtype=np.int32)
    save_labels(
        scan,
        class_ids=class_ids,
        instance_ids=inst_ids,
        write_history=False,
        prelabel_fingerprint="sha256:abc",
        source_fingerprint="sha256:def",
    )
    meta = json.loads((scan / "labels" / "gt_segment_metadata.json").read_text())
    assert meta["prelabel_fingerprint"] == "sha256:abc"
    assert meta["source_fingerprint"] == "sha256:def"


def test_save_labels_omits_fingerprints_when_not_provided(tmp_path):
    from labeling.segment_io import save_labels
    scan = tmp_path
    (scan / "labels").mkdir()
    save_labels(
        scan,
        class_ids=np.full(4, -1, dtype=np.int32),
        instance_ids=np.full(4, -1, dtype=np.int32),
        write_history=False,
    )
    meta = json.loads((scan / "labels" / "gt_segment_metadata.json").read_text())
    assert "prelabel_fingerprint" not in meta
    assert "source_fingerprint" not in meta
