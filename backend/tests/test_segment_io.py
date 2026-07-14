"""Tests for label save, session aux, and working-array I/O."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pytest

from labeling.segment_io import (
    atomic_write_npy,
    load_session_aux,
    load_working_arrays,
    prune_history,
    save_labels,
    save_session_aux,
)


def _read_npy(path: Path) -> np.ndarray:
    return np.load(path)


def test_save_labels_writes_aligned_arrays_and_metadata(tmp_path):
    scan_dir = tmp_path / "annotated" / "demo"
    sid = "sess-001"
    cls = np.array([-1, 0, 0, 1, 1], dtype=np.int8)
    inst = np.array([-1, 0, 0, 1, 1], dtype=np.int32)
    save_labels(scan_dir, sid, cls, inst, write_history=False)

    out_dir = scan_dir / "sessions" / sid / "output"
    np.testing.assert_array_equal(
        _read_npy(out_dir / "gt_class_ids.npy"), cls.astype(np.int32),
    )
    np.testing.assert_array_equal(
        _read_npy(out_dir / "gt_segment_ids.npy"), inst.astype(np.int32),
    )
    meta = json.loads((out_dir / "gt_segment_metadata.json").read_text())
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
        save_labels(scan_dir, "sess-001", cls, inst, write_history=False)


def test_save_labels_rejects_class_inconsistency(tmp_path):
    import pytest
    scan_dir = tmp_path / "annotated" / "demo"
    cls = np.array([0, 1], dtype=np.int8)
    inst = np.array([0, 0], dtype=np.int32)
    with pytest.raises(ValueError, match="invariant"):
        save_labels(scan_dir, "sess-001", cls, inst, write_history=False)


def test_save_labels_writes_history_snapshot(tmp_path):
    scan_dir = tmp_path / "annotated" / "demo"
    sid = "sess-001"
    save_labels(scan_dir, sid, np.array([0], dtype=np.int8), np.array([0], dtype=np.int32),
                write_history=True)
    hist = scan_dir / "sessions" / sid / "history"
    assert hist.exists()
    snaps = list(hist.iterdir())
    assert len(snaps) == 1
    assert re.match(r"^\d{8}_\d{6}$", snaps[0].name)


def test_save_labels_snapshots_existing_labels_before_overwrite(tmp_path):
    """v1 → save → v2 → save: v1's class array must be preserved in history."""
    scan_dir = tmp_path / "annotated" / "demo"
    sid = "sess-001"
    cls_v1 = np.array([0], dtype=np.int8)
    inst_v1 = np.array([0], dtype=np.int32)
    save_labels(scan_dir, sid, cls_v1, inst_v1, write_history=False)

    cls_v2 = np.array([1], dtype=np.int8)
    inst_v2 = np.array([1], dtype=np.int32)
    save_labels(scan_dir, sid, cls_v2, inst_v2, write_history=True)

    hist = scan_dir / "sessions" / sid / "history"
    snaps = list(hist.iterdir())
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
    sid = "sess-001"
    _write_classes_json(lidar_root, 2, [
        {"id": 0, "name": "pipe"}, {"id": 1, "name": "tank"},
    ])
    save_labels(scan_dir, sid,
                np.array([0, 1], dtype=np.int8),
                np.array([0, 1], dtype=np.int32),
                write_history=False)

    meta = json.loads((scan_dir / "sessions" / sid / "output" / "gt_segment_metadata.json").read_text())
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
        save_labels(scan_dir, "sess-001",
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
        save_labels(scan_dir, "sess-001",
                    np.array([0], dtype=np.int8),
                    np.array([0], dtype=np.int32),
                    write_history=False)


# atomic_write_npy/json round-trip + tmp-cleanup are tested in the scan_schema
# package (tests/test_storage.py), which now owns those writers.


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
    """gt_segment_metadata.json must carry preseg_fingerprint + source_fingerprint
    when supplied by the caller."""
    from labeling.segment_io import save_labels
    scan = tmp_path
    sid = "sess-001"
    class_ids = np.full(4, -1, dtype=np.int32)
    inst_ids = np.full(4, -1, dtype=np.int32)
    save_labels(
        scan,
        sid,
        class_ids=class_ids,
        instance_ids=inst_ids,
        write_history=False,
        preseg_fingerprint="sha256:abc",
        source_fingerprint="sha256:def",
    )
    meta = json.loads((scan / "sessions" / sid / "output" / "gt_segment_metadata.json").read_text())
    assert meta["preseg_fingerprint"] == "sha256:abc"
    assert meta["source_fingerprint"] == "sha256:def"


def test_save_labels_omits_fingerprints_when_not_provided(tmp_path):
    from labeling.segment_io import save_labels
    scan = tmp_path
    sid = "sess-001"
    save_labels(
        scan,
        sid,
        class_ids=np.full(4, -1, dtype=np.int32),
        instance_ids=np.full(4, -1, dtype=np.int32),
        write_history=False,
    )
    meta = json.loads((scan / "sessions" / sid / "output" / "gt_segment_metadata.json").read_text())
    assert "preseg_fingerprint" not in meta
    assert "source_fingerprint" not in meta


def test_save_session_aux_writes_working_sam_ids(tmp_path):
    from labeling.segment_io import save_session_aux
    sam_ids = np.array([-1, 0, 0, -1], dtype=np.int32)
    save_session_aux(tmp_path, {"name": "x"}, sam_ids=sam_ids)
    assert (tmp_path / "working_sam_ids.npy").exists()
    loaded = np.load(tmp_path / "working_sam_ids.npy")
    assert (loaded == sam_ids).all()


def test_save_session_aux_without_sam_ids_does_not_write_file(tmp_path):
    from labeling.segment_io import save_session_aux
    save_session_aux(tmp_path, {"name": "x"})
    assert not (tmp_path / "working_sam_ids.npy").exists()


def test_load_sam_ids_roundtrip(tmp_path):
    from labeling.segment_io import save_session_aux, load_sam_ids
    sam_ids = np.array([-1, 0, 0, -1], dtype=np.int32)
    save_session_aux(tmp_path, {"name": "x"}, sam_ids=sam_ids)
    loaded = load_sam_ids(tmp_path, n_points=4)
    assert loaded is not None
    assert (loaded == sam_ids).all()


def test_load_sam_ids_absent_file_returns_none(tmp_path):
    from labeling.segment_io import load_sam_ids
    assert load_sam_ids(tmp_path, n_points=4) is None


def test_load_sam_ids_shape_mismatch_raises(tmp_path):
    from labeling.segment_io import save_session_aux, load_sam_ids
    save_session_aux(tmp_path, {"name": "x"},
                     sam_ids=np.array([-1, 0], dtype=np.int32))
    with pytest.raises(ValueError):
        load_sam_ids(tmp_path, n_points=99)


def test_save_and_load_sam_segments_roundtrip(tmp_path):
    from labeling.segment_io import save_sam_segments, load_sam_segments
    segs = {0: {"n_points": 5, "mask_score": 0.9, "created_at": "2026-07-13T00:00:00+00:00"},
            2: {"n_points": 3, "mask_score": None, "created_at": "2026-07-13T00:00:01+00:00"}}
    save_sam_segments(tmp_path, segs)
    assert (tmp_path / "sam_segments.json").exists()
    loaded = load_sam_segments(tmp_path)
    assert loaded == segs


def test_load_sam_segments_absent_file_returns_empty_dict(tmp_path):
    from labeling.segment_io import load_sam_segments
    assert load_sam_segments(tmp_path) == {}
