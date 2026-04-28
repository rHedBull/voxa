"""Annotated SCHEMA loader + LAZ stride sampling.

The LAZ test writes a tiny LAS file (lossless, no compression) so it doesn't
require the lazrs codec; load_laz works on both LAS and LAZ via laspy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from lidar_io import load_annotated, load_laz
from scene_registry import discover


def _write_ply(path: Path, n: int = 8, with_color: bool = True) -> None:
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    arr = np.zeros(n, dtype=dtype)
    arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    if with_color:
        arr['red'] = arr['green'] = arr['blue'] = 200
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(str(path))


def _build_annotated_root(tmp_path: Path) -> Path:
    root = tmp_path / "lidar"
    scan_dir = root / "annotated" / "demo"
    _write_ply(scan_dir / "source" / "scan.ply", n=8)
    (scan_dir / "labels").mkdir(parents=True, exist_ok=True)
    np.save(scan_dir / "labels" / "gt_class_ids.npy",
            np.array([-1, 0, 0, 1, 1, 2, -1, 2], dtype=np.int32))
    np.save(scan_dir / "labels" / "gt_segment_ids.npy",
            np.array([-1, 0, 0, 1, 1, 2, -1, 3], dtype=np.int32))
    (scan_dir / "labels" / "gt_segment_metadata.json").write_text(json.dumps({
        "n_points": 8, "n_gt_segments": 4, "n_labeled_points": 6,
        "class_map": {"pipe": 0, "tank": 1, "equipment": 2},
        "segments": [],
    }))
    (scan_dir / "meta.json").write_text(json.dumps({
        "scan_name": "demo", "n_points": 8, "coords": "world", "units": "meters",
    }))
    (root / "classes.json").write_text(json.dumps({
        "version": 1, "unlabeled_id": -1,
        "classes": [{"id": 0, "name": "pipe"},
                    {"id": 1, "name": "tank"},
                    {"id": 2, "name": "equipment"}],
    }))
    return root


def test_load_annotated_returns_aligned_label_arrays(tmp_path):
    root = _build_annotated_root(tmp_path)
    src = next(s for s in discover(tmp_path / "voxa-data", root) if s.tier == "annotated")
    out = load_annotated(src, root)

    assert len(out.pc) == 8
    assert out.labels is not None
    assert out.labels.class_ids.shape == (8,)
    assert out.labels.instance_ids.shape == (8,)
    assert out.labels.class_ids.dtype == np.int8
    assert int(out.n_classes) == 3   # max label id + 1
    assert int(out.n_instances) == 4


def test_load_annotated_palette_resolves_labels_from_classes_json(tmp_path):
    root = _build_annotated_root(tmp_path)
    src = next(s for s in discover(tmp_path / "voxa-data", root) if s.tier == "annotated")
    out = load_annotated(src, root)

    by_id = {p.id: p.label for p in out.palette}
    assert by_id[0] == "pipe"
    assert by_id[1] == "tank"
    assert by_id[2] == "equipment"
    # All entries get a color even though classes.json doesn't carry one.
    assert all(p.color.startswith("#") for p in out.palette)


def test_load_annotated_no_labels(tmp_path):
    """factory_large-style stub: no .npy files and no prelabel/ → all-(-1) arrays."""
    root = tmp_path / "lidar"
    scan_dir = root / "annotated" / "stub"
    _write_ply(scan_dir / "source" / "scan.ply", n=4)
    (scan_dir / "labels").mkdir(parents=True, exist_ok=True)
    (scan_dir / "meta.json").write_text(json.dumps({"scan_name": "stub", "n_points": 4}))

    src = next(s for s in discover(tmp_path / "voxa-data", root)
               if s.tier == "annotated" and s.name == "stub")
    out = load_annotated(src, root)
    assert out.labels is not None
    assert int(out.labels.class_ids.min()) == -1 and int(out.labels.class_ids.max()) == -1
    assert out.is_from_prelabel is False
    assert out.n_classes == 0
    assert out.n_instances == 0


def _write_tiny_las(path: Path, n: int = 1000) -> None:
    """Produce a real LAS 1.2 / point format 3 file for stride testing."""
    import laspy

    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    rng = np.random.default_rng(42)
    las.x = rng.uniform(0, 10, n).astype(np.float64)
    las.y = rng.uniform(0, 10, n).astype(np.float64)
    las.z = rng.uniform(0, 5, n).astype(np.float64)
    las.red = (rng.integers(0, 65535, n)).astype(np.uint16)
    las.green = (rng.integers(0, 65535, n)).astype(np.uint16)
    las.blue = (rng.integers(0, 65535, n)).astype(np.uint16)
    las.intensity = (rng.integers(0, 65535, n)).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(path))


def test_load_laz_stride_subsamples_to_target(tmp_path):
    p = tmp_path / "tiny.las"
    _write_tiny_las(p, n=1000)

    pc, intensity = load_laz(p, max_points=100)

    # stride = ceil(1000 / 100) = 10 → exactly 100 sampled points.
    assert len(pc) == 100
    assert intensity.shape == (100,)
    assert intensity.dtype == np.float32
    assert intensity.min() >= 0.0 and intensity.max() <= 1.0


def test_load_laz_intensity_normalized_to_unit_range(tmp_path):
    p = tmp_path / "tiny.las"
    _write_tiny_las(p, n=200)

    _pc, intensity = load_laz(p, max_points=200)
    # Intensity must be 0..1 regardless of LAS raw uint16 range.
    assert float(intensity.max()) == pytest.approx(1.0, abs=1e-6)


def test_load_laz_smaller_than_max_returns_all(tmp_path):
    p = tmp_path / "tiny.las"
    _write_tiny_las(p, n=50)

    pc, _intensity = load_laz(p, max_points=300_000)
    # stride collapses to 1; we keep every point.
    assert len(pc) == 50


def test_load_annotated_falls_through_to_prelabel(tmp_path):
    """When labels/gt_*.npy is absent, prelabel/ becomes the editable seed."""
    root = tmp_path / "lidar"
    scan_dir = root / "annotated" / "demo"
    _write_ply(scan_dir / "source" / "scan.ply", n=8)
    pre = scan_dir / "prelabel"; pre.mkdir(parents=True)
    np.save(pre / "ransac_instance_ids.npy",
            np.array([-1, 0, 0, 1, 1, 2, -1, 2], dtype=np.int32))
    (pre / "ransac_segment_summary.json").write_text(json.dumps({
        "segments": [{"id": 0, "class_id": 0},
                     {"id": 1, "class_id": 1},
                     {"id": 2, "class_id": 2}],
    }))
    (root / "classes.json").write_text(json.dumps({
        "version": 1, "unlabeled_id": -1,
        "classes": [{"id": 0, "name": "pipe"}, {"id": 1, "name": "tank"},
                    {"id": 2, "name": "equipment"}],
    }))

    src = next(s for s in discover(tmp_path / "voxa-data", root) if s.tier == "annotated")
    out = load_annotated(src, root)

    assert out.is_from_prelabel is True
    assert out.labels is not None
    assert out.labels.instance_ids.shape == (8,)
    assert int(out.labels.instance_ids[1]) == 0
    assert int(out.labels.class_ids[1]) == 0


def test_load_annotated_empty_when_no_labels_no_prelabel(tmp_path):
    root = tmp_path / "lidar"
    scan_dir = root / "annotated" / "demo"
    _write_ply(scan_dir / "source" / "scan.ply", n=8)
    src = next(s for s in discover(tmp_path / "voxa-data", root) if s.tier == "annotated")
    out = load_annotated(src, root)
    # No labels, no prelabel → labels arrays present but all -1, is_from_prelabel False.
    assert out.is_from_prelabel is False
    assert out.labels is not None
    assert int(out.labels.class_ids.min()) == -1 and int(out.labels.class_ids.max()) == -1
