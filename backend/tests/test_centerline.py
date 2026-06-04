"""Tests for labeling.centerline — tube extraction + path sampling + store."""
from __future__ import annotations

import numpy as np
import pytest

from labeling.centerline import tube_indices


def _cylinder_cloud(axis_len=2.0, radius=0.1, n=500, seed=0):
    """Synthetic pipe along +X: points on a cylinder surface around the X axis."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, axis_len, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    y = radius * np.cos(theta)
    z = radius * np.sin(theta)
    return np.column_stack([x, y, z]).astype(np.float32)


def test_tube_captures_cylinder_points():
    pts = _cylinder_cloud(radius=0.1)
    # Outlier far away must not be captured.
    cloud = np.vstack([pts, [[5.0, 5.0, 5.0]]]).astype(np.float32)
    paths = [{"points": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], "radius": 0.15, "smooth": False}]
    idx = tube_indices(cloud, paths)
    assert set(idx.tolist()) == set(range(len(pts)))


def test_tube_excludes_points_outside_radius():
    pts = _cylinder_cloud(radius=0.3)   # all at distance 0.3 from the axis
    paths = [{"points": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], "radius": 0.15, "smooth": False}]
    idx = tube_indices(pts, paths)
    assert idx.size == 0


def test_multi_segment_elbow():
    # L-shaped path: along +X then turning to +Y. One point near each leg,
    # one near the joint, one in the "inside corner" beyond the radius.
    cloud = np.array([
        [1.0, 0.05, 0.0],    # near leg 1
        [2.0, 1.0, 0.05],    # near leg 2
        [2.0, 0.02, 0.0],    # at the joint
        [1.0, 1.0, 0.0],     # inside corner, far from both segments
    ], dtype=np.float32)
    paths = [{"points": [[0, 0, 0], [2, 0, 0], [2, 2, 0]], "radius": 0.1, "smooth": False}]
    idx = tube_indices(cloud, paths)
    assert set(idx.tolist()) == {0, 1, 2}


def test_multiple_paths_union_unique():
    cloud = np.array([[0.5, 0, 0], [0.5, 1, 0], [0.5, 0.02, 0]], dtype=np.float32)
    paths = [
        {"points": [[0, 0, 0], [1, 0, 0]], "radius": 0.1, "smooth": False},
        {"points": [[0, 1, 0], [1, 1, 0]], "radius": 0.1, "smooth": False},
        # Overlapping with path 1 — union must stay unique.
        {"points": [[0, 0, 0], [1, 0, 0]], "radius": 0.05, "smooth": False},
    ]
    idx = tube_indices(cloud, paths)
    assert sorted(idx.tolist()) == [0, 1, 2]
    assert idx.dtype == np.int32


def test_per_path_radius_respected():
    cloud = np.array([[0.5, 0.2, 0.0]], dtype=np.float32)
    thin = [{"points": [[0, 0, 0], [1, 0, 0]], "radius": 0.1, "smooth": False}]
    thick = [{"points": [[0, 0, 0], [1, 0, 0]], "radius": 0.3, "smooth": False}]
    assert tube_indices(cloud, thin).size == 0
    assert tube_indices(cloud, thick).size == 1


def test_degenerate_zero_length_segment():
    # Two identical control points → segment degenerates to a sphere test.
    cloud = np.array([[0.0, 0.05, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    paths = [{"points": [[0, 0, 0], [0, 0, 0]], "radius": 0.1, "smooth": False}]
    idx = tube_indices(cloud, paths)
    assert idx.tolist() == [0]
