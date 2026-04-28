"""Tests for the on-load merge-model inference path.

Real model inference needs xgboost + a trained .pkl + the segmentation
repo on disk — too heavy for unit tests. We verify the surrounding glue:
artifact discovery, shape validation, environment-driven path resolution,
prelabel cache write, and degraded-mode behavior (returns None when
deps are missing). The actual fit/predict math is covered by the
segmentation repo's own smoke tests.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from seg_inference import (
    _ransac_class_for_segment,
    _read_ransac_artifacts,
    _write_prelabel_cache,
)


def _write_artifacts(scan_dir: Path, n: int = 10) -> None:
    seg_dir = scan_dir / "fresh_run" / "segmentation"
    seg_dir.mkdir(parents=True, exist_ok=True)
    np.save(seg_dir / "instance_labels.npy",
            np.array([0] * (n // 2) + [1] * (n - n // 2), dtype=np.int32))
    np.save(seg_dir / "surface_labels.npy", np.zeros(n, dtype=np.int32))
    np.save(seg_dir / "k1.npy", np.zeros(n, dtype=np.float32))
    np.save(seg_dir / "k2.npy", np.zeros(n, dtype=np.float32))
    (seg_dir / "ransac_summary.json").write_text(json.dumps([
        {"id": 0, "type": "cylinder", "label": "pipe", "radius": 0.05},
        {"id": 1, "type": "cylinder", "label": "tank", "radius": 0.10},
    ]))


# ── Pure helpers ────────────────────────────────────────────────────────────

def test_ransac_class_for_segment_matches_case_insensitively():
    cm = {"pipe": 0, "tank": 1, "equipment": 2}
    assert _ransac_class_for_segment("pipe", cm) == 0
    assert _ransac_class_for_segment("PIPE", cm) == 0
    assert _ransac_class_for_segment("Tank", cm) == 1


def test_ransac_class_for_segment_returns_minus_one_on_unknown():
    cm = {"pipe": 0}
    assert _ransac_class_for_segment("widget", cm) == -1
    assert _ransac_class_for_segment("", cm) == -1
    assert _ransac_class_for_segment(None, cm) == -1   # type: ignore[arg-type]


def test_ransac_class_for_segment_matches_descriptive_ransac_labels():
    """ipl segment emits 'flat_surface', 'large_pipe', etc. — broader than
    voxa's class names. The keyword heuristic should still classify them."""
    cm = {"pipe": 0, "tank": 1, "equipment": 2, "structural": 3, "fitting": 4}
    assert _ransac_class_for_segment("large_pipe", cm) == 0
    assert _ransac_class_for_segment("short_pipe", cm) == 0
    assert _ransac_class_for_segment("small_pipe", cm) == 0
    assert _ransac_class_for_segment("flat_surface", cm) == 3
    assert _ransac_class_for_segment("vessel", cm) == 1


def test_ransac_class_for_segment_exact_match_takes_priority():
    """A label that's an exact class name should win over the keyword
    heuristic — e.g. "pipe" matches "pipe" exactly, not via the keyword."""
    cm = {"pipe": 0, "tank": 1}
    assert _ransac_class_for_segment("pipe", cm) == 0
    assert _ransac_class_for_segment("PIPE", cm) == 0


# ── Artifact discovery ──────────────────────────────────────────────────────

def test_read_ransac_artifacts_returns_none_when_missing(tmp_path):
    assert _read_ransac_artifacts(tmp_path) is None


def test_read_ransac_artifacts_returns_dict_when_complete(tmp_path):
    _write_artifacts(tmp_path, n=10)
    out = _read_ransac_artifacts(tmp_path)
    assert out is not None
    assert out["inst_ids"].shape == (10,)
    assert out["surface_labels"].shape == (10,)
    assert len(out["ransac_summary"]) == 2


def test_read_ransac_artifacts_returns_none_when_partial(tmp_path):
    """Missing one of the 5 files → None (we don't try to predict on incomplete inputs)."""
    _write_artifacts(tmp_path, n=10)
    (tmp_path / "fresh_run" / "segmentation" / "k1.npy").unlink()
    assert _read_ransac_artifacts(tmp_path) is None


# ── Prelabel cache write ────────────────────────────────────────────────────

def test_write_prelabel_cache_writes_schema_shaped_files(tmp_path):
    inst = np.array([-1, 0, 0, 1, 1], dtype=np.int32)
    summary = [{"id": 0, "label": "pipe"}, {"id": 1, "label": "tank"}]
    cm = {"pipe": 0, "tank": 1}
    _write_prelabel_cache(tmp_path, inst, summary, cm)

    cached = np.load(tmp_path / "prelabel" / "ransac_instance_ids.npy")
    np.testing.assert_array_equal(cached, inst)

    summary_out = json.loads(
        (tmp_path / "prelabel" / "ransac_segment_summary.json").read_text()
    )
    assert summary_out["segments"] == [
        {"id": 0, "class_id": 0, "label": "pipe"},
        {"id": 1, "class_id": 1, "label": "tank"},
    ]


def test_write_prelabel_cache_marks_unknown_label_class_as_minus_one(tmp_path):
    inst = np.array([0], dtype=np.int32)
    summary = [{"id": 0, "label": "widget"}]
    cm = {"pipe": 0, "tank": 1}
    _write_prelabel_cache(tmp_path, inst, summary, cm)
    summary_out = json.loads(
        (tmp_path / "prelabel" / "ransac_segment_summary.json").read_text()
    )
    assert summary_out["segments"][0]["class_id"] == -1


# ── Top-level predict_for_scene degraded-mode behavior ──────────────────────

def test_predict_for_scene_returns_none_when_repo_missing(tmp_path, monkeypatch):
    """Segmentation repo not on disk → None, no exception."""
    monkeypatch.setenv("VOXA_SEGMENTATION_REPO", str(tmp_path / "nonexistent"))
    monkeypatch.setenv("VOXA_MERGE_MODEL", str(tmp_path / "no.pkl"))
    # Module-level paths are read at import time — reload to pick up the env.
    import importlib
    import seg_inference
    importlib.reload(seg_inference)
    assert seg_inference.predict_for_scene(tmp_path, n_points=10, class_map={}) is None


def test_predict_for_scene_returns_none_when_no_artifacts(tmp_path, monkeypatch):
    """Even with a real model bundle, missing RANSAC artifacts → None."""
    monkeypatch.setenv("VOXA_MERGE_MODEL", str(tmp_path / "no.pkl"))
    import importlib, seg_inference
    importlib.reload(seg_inference)
    # No fresh_run/segmentation/ directory at all.
    assert seg_inference.predict_for_scene(tmp_path, n_points=10,
                                           class_map={"pipe": 0}) is None
