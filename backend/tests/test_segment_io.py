"""Tests for prelabel ingestion + label save."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from segment_io import load_prelabel


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


import os
import re

from segment_io import save_labels, prune_history


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
    assert re.match(r"^\d{8}_\d{4}$", snaps[0].name)


def test_prune_history_keeps_only_timestamped_dirs(tmp_path):
    hist = tmp_path / "annotation_history"
    hist.mkdir()
    valid = [hist / f"2026010{i % 10}_{1000 + i:04d}" for i in range(12)]
    for v in valid:
        v.mkdir()
    for i, v in enumerate(valid):
        os.utime(v, (1_700_000_000 + i, 1_700_000_000 + i))
    user = hist / "manual-backup"; user.mkdir()

    prune_history(hist, keep=10)

    remaining = sorted(p.name for p in hist.iterdir())
    assert "manual-backup" in remaining
    assert sum(1 for n in remaining if re.match(r"^\d{8}_\d{4}$", n)) == 10
