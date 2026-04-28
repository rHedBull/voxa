"""Tests for prelabel ingestion + label save."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

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
