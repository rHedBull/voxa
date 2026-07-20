"""Tests for fit_gravity_obb + the /api/segment/fit-box endpoint."""
import numpy as np
import pytest

from labeling.fit_box import fit_gravity_obb
from labeling.shapes import obb_indices


def _obb_dict(center, size, rotation):
    return {"center": list(center), "size": list(size), "rotation": list(rotation)}


def test_axis_aligned_box_recovers_bounds():
    # A filled axis-aligned box; fit should recover its center + extents (+pad).
    # NOTE: 2D PCA picks the higher-variance horizontal axis (z, extent 6) as
    # dominant, so ry ≈ ±π/2 and the footprint axes (size[0]/size[2]) may SWAP.
    # The eigenvector sign from eigh is also arbitrary (ry can come back as π).
    # So assert on the SORTED extents + containment, never positional size[i].
    rng = np.random.default_rng(0)
    pts = rng.uniform([-2, -1, -3], [2, 1, 3], size=(2000, 3)).astype(np.float32)
    center, size, rotation = fit_gravity_obb(pts)
    assert rotation[0] == 0.0 and rotation[2] == 0.0
    np.testing.assert_allclose(center, [0, 0, 0], atol=0.05)
    # Sorted extents ~ (2, 4, 6) + small pad, order-independent.
    np.testing.assert_allclose(sorted(size), [2, 4, 6], atol=0.05)
    # And the box actually contains every point (the real invariant).
    assert obb_indices(pts, _obb_dict(center, size, rotation)).size == pts.shape[0]


def test_containment_parity_after_yaw():
    # Rotate a thin rectangle by a known yaw; the fitted OBB fed back through
    # obb_indices must contain 100% of the points. This locks the yaw sign /
    # Euler convention — a reversed yaw drops points.
    rng = np.random.default_rng(1)
    theta = 0.6
    local = rng.uniform([-3, -0.5, -0.4], [3, 0.5, 0.4], size=(3000, 3))
    c, s = np.cos(theta), np.sin(theta)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    pts = (local @ Ry.T).astype(np.float32)  # rotate about world Y
    center, size, rotation = fit_gravity_obb(pts)
    inside = obb_indices(pts, _obb_dict(center, size, rotation))
    assert inside.size == pts.shape[0]  # every point contained


def test_yaw_gives_tight_footprint_not_aabb():
    # For a diagonal thin rectangle, the tight footprint is much smaller than
    # the axis-aligned bounding box of the same points.
    rng = np.random.default_rng(2)
    theta = np.pi / 4
    local = rng.uniform([-3, 0, -0.3], [3, 1, 0.3], size=(3000, 3))
    c, s = np.cos(theta), np.sin(theta)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    pts = (local @ Ry.T).astype(np.float32)
    _, size, _ = fit_gravity_obb(pts)
    aabb_x = pts[:, 0].max() - pts[:, 0].min()
    # Tight footprint SHORT side is ~0.6, which an AABB could never be.
    short = min(size[0], size[2])
    assert short < 1.0
    assert min(aabb_x, pts[:, 2].max() - pts[:, 2].min()) > 3.0


def test_collinear_xz_falls_back_to_axis_aligned():
    # All points share one XZ line (vary only along x and y). No NaN, ry=0,
    # positive volume.
    pts = np.zeros((100, 3), dtype=np.float32)
    pts[:, 0] = np.linspace(-1, 1, 100)
    pts[:, 1] = np.linspace(0, 2, 100)
    center, size, rotation = fit_gravity_obb(pts)
    assert rotation[1] == 0.0
    assert all(np.isfinite(v) for v in (*center, *size))
    assert size[0] > 0 and size[1] > 0 and size[2] > 0


def test_fewer_than_three_points():
    pts = np.array([[0, 0, 0], [1, 2, 1]], dtype=np.float32)
    center, size, rotation = fit_gravity_obb(pts)
    assert rotation == [0.0, 0.0, 0.0]
    assert all(v > 0 for v in size)


def test_zero_points_raises():
    with pytest.raises(ValueError):
        fit_gravity_obb(np.empty((0, 3), dtype=np.float32))
