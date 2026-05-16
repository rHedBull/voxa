"""Regression: voxa rotates Z-up source → Y-up display frame on load.

Before the fix, /api/edit/export-ply wrote in-memory (Y-up) coords directly,
so cleaned PLYs came out 90° rotated relative to the source. The fix
inverts the rotation and re-applies the recenter offset on export.
"""
from __future__ import annotations

import numpy as np

from main import _z_up_to_y_up, _y_up_to_z_up_xyz, _to_display_frame  # type: ignore
from point_cloud import PointCloud  # type: ignore


def test_zup_to_yup_forward_mapping():
    """(x, y, z) → (x, z, -y)."""
    pts = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    out = _z_up_to_y_up(PointCloud(points=pts.copy(), colors=None)).points
    np.testing.assert_allclose(out, [[1.0, 3.0, -2.0]])


def test_yup_to_zup_inverse_mapping():
    """Inverse: (x, y, z) → (x, -z, y)."""
    pts = np.array([[1.0, 3.0, -2.0]])
    out = _y_up_to_z_up_xyz(pts)
    np.testing.assert_allclose(out, [[1.0, 2.0, 3.0]])


def test_inverse_roundtrip_random():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-50, 50, size=(1000, 3)).astype(np.float32)
    yup = _z_up_to_y_up(PointCloud(points=pts.copy(), colors=None)).points
    zup = _y_up_to_z_up_xyz(yup)
    np.testing.assert_allclose(zup, pts, rtol=0, atol=1e-6)


def test_to_display_frame_matches_load_path():
    """`_to_display_frame` reproduces the load-time z_up→y_up + recenter
    so source-frame points line up with OBB ops authored in the display
    frame."""
    rng = np.random.default_rng(1)
    pts = rng.uniform(-10, 10, size=(200, 3)).astype(np.float64)
    # Simulate a centered scene
    centered = _z_up_to_y_up(PointCloud(points=pts.copy().astype(np.float32),
                                        colors=None)).points
    offset = np.mean(centered, axis=0)
    expected = centered - offset
    got = _to_display_frame(pts, scene_is_z_up=True, offset=offset)
    np.testing.assert_allclose(got, expected, atol=1e-5)


def test_to_display_frame_yup_scene_just_recenters():
    """If scene_is_z_up is False the only thing to undo is the recenter."""
    pts = np.array([[1.0, 2.0, 3.0]])
    offset = np.array([1.0, 1.0, 1.0])
    got = _to_display_frame(pts, scene_is_z_up=False, offset=offset)
    np.testing.assert_allclose(got, [[0.0, 1.0, 2.0]])
