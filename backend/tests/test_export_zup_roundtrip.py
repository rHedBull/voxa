"""Regression test: voxa loaded a Z-up scene → Y-up display frame, but
export-ply was returning Y-up data. Fixed so the export inverse-rotates.
"""
import struct, base64
import numpy as np

from backend.main import _z_up_to_y_up, _y_up_to_z_up_xyz


def test_inverse_z_up_y_up_roundtrip():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-50, 50, size=(1000, 3)).astype(np.float32)
    from backend.point_cloud import PointCloud
    pc = PointCloud(points=pts.copy(), colors=None)
    yup = _z_up_to_y_up(pc).points
    # roundtrip back
    zup = _y_up_to_z_up_xyz(yup)
    np.testing.assert_allclose(zup, pts, rtol=1e-6)


def test_zup_to_yup_mapping_is_xyz_to_x_z_minus_y():
    """Forward mapping is (x, y, z) → (x, z, -y)."""
    from backend.point_cloud import PointCloud
    pts = np.array([[1.0, 2.0, 3.0]])
    out = _z_up_to_y_up(PointCloud(points=pts, colors=None)).points
    np.testing.assert_allclose(out, [[1.0, 3.0, -2.0]])


def test_yup_to_zup_xyz_mapping_is_xyz_to_x_minus_z_y():
    """Inverse mapping is (x, y, z) → (x, -z, y)."""
    pts = np.array([[1.0, 3.0, -2.0]])
    out = _y_up_to_z_up_xyz(pts)
    np.testing.assert_allclose(out, [[1.0, 2.0, 3.0]])
