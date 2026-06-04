"""Tests for labeling.centerline — tube extraction + path sampling + store."""
from __future__ import annotations

import numpy as np
import pytest

from labeling.centerline import tube_indices
from labeling.centerline import sample_path
from labeling.centerline import load_centerlines, update_centerlines


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


def test_sample_path_straight_passthrough():
    p = {"points": [[0, 0, 0], [1, 0, 0], [1, 1, 0]], "radius": 0.1, "smooth": False}
    out = sample_path(p)
    assert np.allclose(out, p["points"])


def test_sample_path_smooth_interpolates_through_controls():
    p = {"points": [[0, 0, 0], [1, 1, 0], [2, 0, 0]], "radius": 0.2, "smooth": True}
    out = sample_path(p)
    # Control points must lie on the sampled curve.
    for cp in p["points"]:
        d = np.linalg.norm(out - np.asarray(cp, dtype=np.float32), axis=1)
        assert d.min() < 1e-5
    # Densified: more samples than control points. Chords must stay below
    # the tube radius so the segment-distance test can't visibly cut corners.
    # (Uniform-parameter sampling overshoots step≈radius/2 near the apex —
    # the guarantee that matters is "< radius", not "< step".)
    assert len(out) > 3
    chords = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert chords.max() <= 0.2


def test_sample_path_smooth_two_points_is_segment():
    p = {"points": [[0, 0, 0], [1, 0, 0]], "radius": 0.1, "smooth": True}
    out = sample_path(p)
    # Degenerates to a straight chordal run along the segment.
    assert np.allclose(out[0], [0, 0, 0]) and np.allclose(out[-1], [1, 0, 0])
    assert np.allclose(out[:, 1:], 0, atol=1e-6)


def test_smooth_tube_captures_curve_apex():
    # Point at the apex of the arc through (0,0),(1,1),(2,0) — outside the
    # straight-polyline corner-cutting chord, inside the smooth tube.
    cloud = np.array([[1.0, 1.05, 0.0]], dtype=np.float32)
    smooth = [{"points": [[0, 0, 0], [1, 1, 0], [2, 0, 0]], "radius": 0.15, "smooth": True}]
    assert tube_indices(cloud, smooth).size == 1


def _path(x=0.0, radius=0.15):
    return {"points": [[x, 0, 0], [x + 1, 0, 0]], "radius": radius, "smooth": False}


def test_store_roundtrip_append_and_load(tmp_path):
    assert load_centerlines(tmp_path) == {"paths": []}
    update_centerlines(tmp_path, instance_id=7, class_id=0, paths=[_path()], merged_from=[])
    doc = load_centerlines(tmp_path)
    assert len(doc["paths"]) == 1
    assert doc["paths"][0]["instance_id"] == 7
    assert doc["paths"][0]["class_id"] == 0
    assert doc["paths"][0]["radius"] == 0.15


def test_store_reapply_replaces_by_instance_id(tmp_path):
    update_centerlines(tmp_path, 7, 0, [_path(0.0), _path(5.0)], [])
    # Re-apply instance 7 with ONE edited path → both old entries replaced.
    update_centerlines(tmp_path, 7, 2, [_path(0.0, radius=0.3)], [])
    doc = load_centerlines(tmp_path)
    assert len(doc["paths"]) == 1
    assert doc["paths"][0]["radius"] == 0.3
    assert doc["paths"][0]["class_id"] == 2


def test_store_merged_from_deletes_absorbed_entries(tmp_path):
    update_centerlines(tmp_path, 7, 0, [_path(0.0)], [])
    update_centerlines(tmp_path, 9, 0, [_path(5.0)], [])
    # Merge 9 into 7: union of paths applied under 7, 9 absorbed.
    update_centerlines(tmp_path, 7, 0, [_path(0.0), _path(5.0)], merged_from=[9])
    doc = load_centerlines(tmp_path)
    assert sorted(p["instance_id"] for p in doc["paths"]) == [7, 7]


def test_store_distinct_instances_append(tmp_path):
    update_centerlines(tmp_path, 7, 0, [_path(0.0)], [])
    update_centerlines(tmp_path, 9, 1, [_path(5.0)], [])
    doc = load_centerlines(tmp_path)
    assert sorted(p["instance_id"] for p in doc["paths"]) == [7, 9]
