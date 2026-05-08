import numpy as np
import pytest
from preseg_optimize import score_segmentation


def _plane_xyz(n=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    xy = rng.uniform(-1.0, 1.0, size=(n, 2))
    z = rng.normal(0.0, 0.001, size=(n, 1))
    return np.hstack([xy, z]).astype(np.float64)


def _cylinder_xyz(n=2000, radius=0.3, height=1.5, rng=None):
    rng = rng or np.random.default_rng(1)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    z = rng.uniform(0, height, size=n)
    r = radius + rng.normal(0.0, 0.001, size=n)
    x = 3.0 + r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z]).astype(np.float64)


def _two_primitive_cloud():
    plane = _plane_xyz()
    cyl = _cylinder_xyz()
    xyz = np.vstack([plane, cyl])
    ids = np.concatenate([np.zeros(len(plane), dtype=np.int32),
                          np.ones(len(cyl), dtype=np.int32)])
    return xyz, ids


def test_score_low_for_pure_primitives():
    xyz, ids = _two_primitive_cloud()
    pad_xyz = np.zeros((16, 3))
    pad_ids = np.repeat(np.arange(2, 6, dtype=np.int32), 4)
    xyz_full = np.vstack([xyz, pad_xyz])
    ids_full = np.concatenate([ids, pad_ids])
    score = score_segmentation(xyz_full, ids_full)
    assert score < 0.01, f"pure primitives should score low, got {score}"


def test_score_high_for_mixed_segment():
    xyz, _ = _two_primitive_cloud()
    ids = np.zeros(len(xyz), dtype=np.int32)
    pad_xyz = np.zeros((16, 3))
    pad_ids = np.repeat(np.arange(1, 5, dtype=np.int32), 4)
    xyz_full = np.vstack([xyz, pad_xyz])
    ids_full = np.concatenate([ids, pad_ids])
    score = score_segmentation(xyz_full, ids_full)
    assert score > 0.05, f"mixed segment should score high, got {score}"


def test_score_penalises_too_few_segments():
    xyz, _ = _two_primitive_cloud()
    ids = np.zeros(len(xyz), dtype=np.int32)
    score = score_segmentation(xyz, ids)
    assert score >= 1e5, f"degenerate (1 segment) should hit penalty, got {score}"


def test_score_penalises_too_many_segments():
    rng = np.random.default_rng(2)
    xyz = rng.uniform(-1, 1, size=(20000, 3)).astype(np.float64)
    ids = np.arange(len(xyz), dtype=np.int32)
    score = score_segmentation(xyz, ids)
    assert score >= 1e5
