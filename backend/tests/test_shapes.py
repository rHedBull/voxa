import numpy as np
from labeling.shapes import obb_indices
from labeling.shapes import shape_indices


def test_obb_axis_aligned_selects_interior():
    # 3x3x3 grid of points at integer coords 0..2 on each axis (27 pts).
    xs = np.array([0, 1, 2], dtype=np.float32)
    pts = np.array([[x, y, z] for x in xs for y in xs for z in xs],
                   dtype=np.float32).reshape(-1)
    # Box centered at (1,1,1), size 1.0 -> half-extent 0.5 -> only the center point.
    box = {"center": [1.0, 1.0, 1.0], "size": [1.0, 1.0, 1.0],
           "rotation": [0.0, 0.0, 0.0]}
    idx = obb_indices(pts, box)
    assert idx.tolist() == [13]  # the (1,1,1) point is index 13 in the grid


def test_obb_rotated_matches_local_frame():
    # A point offset +0.9 along world X. A box rotated 45 deg about Z with
    # half-extent 0.5 in local x should NOT contain it (0.9 > 0.5 in any frame),
    # but a box half-extent 0.5 centered ON it should.
    pts = np.array([0.9, 0.0, 0.0], dtype=np.float32)
    inside = {"center": [0.9, 0.0, 0.0], "size": [1.0, 1.0, 1.0],
              "rotation": [0.0, 0.0, np.pi / 4]}
    outside = {"center": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0],
               "rotation": [0.0, 0.0, np.pi / 4]}
    assert obb_indices(pts, inside).tolist() == [0]
    assert obb_indices(pts, outside).tolist() == []


def test_obb_multi_axis_rotation_matches_threejs_euler_xyz():
    # Three.js Euler 'XYZ' (what the viewer renders and mode-label previews)
    # composes as Rx @ Ry @ Rz. Multi-axis rotations distinguish it from the
    # reversed Rz @ Ry @ Rx order — single-axis tests cannot. The composition
    # itself is pinned independently in test_reproject.py; here we verify
    # obb_indices' vectorized containment against that basis.
    from scenes.reproject import euler_xyz_matrix
    rx, ry, rz = 0.4, -0.7, 0.3
    R = euler_xyz_matrix(rx, ry, rz)

    rng = np.random.default_rng(0)
    pts = rng.uniform(-2, 2, (5000, 3)).astype(np.float32)
    half = np.array([1.0, 0.5, 0.25])
    expected = np.flatnonzero(np.all(np.abs(pts @ R) <= half, axis=1))

    box = {"center": [0.0, 0.0, 0.0], "size": (half * 2).tolist(),
           "rotation": [rx, ry, rz]}
    np.testing.assert_array_equal(obb_indices(pts.reshape(-1), box), expected)


def test_shape_indices_obb_dispatch():
    pts = np.array([0.9, 0.0, 0.0], dtype=np.float32)
    shape = {"type": "obb", "center": [0.9, 0.0, 0.0],
             "size": [1.0, 1.0, 1.0], "rotation": [0.0, 0.0, 0.0]}
    assert shape_indices(pts, shape).tolist() == [0]


def test_shape_indices_tube_dispatch_matches_tube_indices():
    from labeling.centerline import tube_indices
    pts = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32).reshape(-1)
    paths = [{"points": [[0, 0, 0], [1, 0, 0]], "radius": 0.5, "smooth": False}]
    shape = {"type": "tube", "paths": paths}
    np.testing.assert_array_equal(
        shape_indices(pts, shape),
        tube_indices(np.asarray(pts, dtype=np.float32).reshape(-1, 3), paths))


def test_shape_indices_unknown_type_raises():
    import pytest
    with pytest.raises(ValueError):
        shape_indices(np.zeros(3, dtype=np.float32), {"type": "blob"})


def _beam_frame(a, b):
    """Replicates frontend beam-graph.js beamFrame(): u along the axis,
    (v, w) from the world axis least aligned with u."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    u = d / np.linalg.norm(d)
    ref = np.eye(3)[int(np.argmin(np.abs(u)))]
    v = np.cross(ref, u)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    return u, v, w


def _euler_xyz_from_basis(u, v, w):
    """Replicates frontend eulerXYZFromBasis() (THREE.Euler 'XYZ' algorithm):
    angles whose Rx.Ry.Rz composition equals the matrix with columns u,v,w."""
    m13, m23, m33 = w[0], w[1], w[2]
    m11, m12 = u[0], v[0]
    m22, m32 = v[1], v[2]
    y = np.arcsin(np.clip(m13, -1.0, 1.0))
    if abs(m13) < 0.9999999:
        x = np.arctan2(-m23, m33)
        z = np.arctan2(-m12, m11)
    else:
        x = np.arctan2(m32, m22)
        z = 0.0
    return [float(x), float(y), float(z)]


def test_beam_style_obb_matches_frame_membership():
    """A beam OBB built the frontend's way (frame -> euler) must select exactly
    the points inside the swept square box, on a skew (non-world-aligned) axis."""
    rng = np.random.default_rng(7)
    a = np.array([0.5, -0.2, 1.0])
    b = np.array([2.0, 1.3, -0.4])
    width = 0.3
    u, v, w = _beam_frame(a, b)
    length = np.linalg.norm(b - a)

    pts = rng.uniform(-1.5, 3.0, size=(5000, 3)).astype(np.float32)
    # Ground truth straight from the frame (the spec's containment definition).
    rel = pts.astype(np.float64) - a
    du, dv, dw = rel @ u, rel @ v, rel @ w
    inside = (du >= 0) & (du <= length) & (np.abs(dv) <= width / 2) & (np.abs(dw) <= width / 2)
    assert inside.sum() > 5           # sanity: the box actually contains points

    box = {
        "center": ((a + b) / 2).tolist(),
        "size": [float(length), width, width],
        "rotation": _euler_xyz_from_basis(u, v, w),
    }
    got = np.zeros(len(pts), dtype=bool)
    got[obb_indices(pts, box)] = True
    # float32 points on float64 boundaries: allow disagreement only within an
    # epsilon shell of the box faces.
    margin = 1e-5
    strict = (du >= margin) & (du <= length - margin) \
        & (np.abs(dv) <= width / 2 - margin) & (np.abs(dw) <= width / 2 - margin)
    outside = (du < -margin) | (du > length + margin) \
        | (np.abs(dv) > width / 2 + margin) | (np.abs(dw) > width / 2 + margin)
    assert got[strict].all()
    assert not got[outside].any()
