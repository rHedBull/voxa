"""Fit a gravity-aligned (yaw-only) oriented box to a set of points.

The box's up axis is locked to world-Y; the footprint is the tightest rotated
rectangle in the X/Z plane, found via rotating calipers over the convex hull
(the classic exact minimum-area-rectangle construction: for a filled convex
region the optimal rectangle always shares an edge with the hull, so it
suffices to test one orientation per hull edge — unlike a 2D-PCA/covariance
estimate, this is exact for axis-aligned data, not a statistical
approximation subject to sampling noise). Rotation is returned as Euler-XYZ
[0, ry, 0], matching the Rx·Ry·Rz convention that shapes.py::obb_indices /
scenes/reproject.py::euler_xyz_matrix compose. See
docs/superpowers/specs/2026-07-20-fit-box-to-selection-design.md.
"""
import numpy as np
from scipy.spatial import ConvexHull, QhullError

PAD = 0.005       # per-axis padding so the box provably contains all points
MIN_SIZE = 0.01   # floor for a degenerate (collinear / near-flat) footprint


def _min_area_rect_theta(xz):
    """Angle (radians) of the box-local x-axis that minimizes the AABB area
    of `xz` in the rotated frame, via rotating calipers over the convex
    hull. Falls back to 0.0 for degenerate (collinear / <3 point) input."""
    try:
        hull = ConvexHull(xz)
    except QhullError:
        return 0.0
    hp = xz[hull.vertices]
    m = hp.shape[0]
    best_area, best_ang = None, 0.0
    for i in range(m):
        edge = hp[(i + 1) % m] - hp[i]
        ang = np.arctan2(edge[1], edge[0])
        c, s = np.cos(ang), np.sin(ang)
        ru = hp[:, 0] * c + hp[:, 1] * s
        rv = -hp[:, 0] * s + hp[:, 1] * c
        area = (ru.max() - ru.min()) * (rv.max() - rv.min())
        if best_area is None or area < best_area:
            best_area, best_ang = area, ang
    return -best_ang


def fit_gravity_obb(points):
    """points: (N,3) array. Returns (center, size, rotation) as plain float
    lists: center [x,y,z], size [sx,sy,sz] (full side lengths), rotation
    [0, ry, 0]. Raises ValueError on an empty input."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n == 0:
        raise ValueError("fit_gravity_obb: empty point set")

    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    cy = (y0 + y1) / 2.0
    sy = (y1 - y0) + PAD

    xz = pts[:, [0, 2]]
    theta = _min_area_rect_theta(xz) if n >= 3 else 0.0

    c, s = np.cos(theta), np.sin(theta)
    # Project onto box-local horizontal axes u (local x) and v (local z).
    pu = xz[:, 0] * c - xz[:, 1] * s
    pv = xz[:, 0] * s + xz[:, 1] * c
    umin, umax = pu.min(), pu.max()
    vmin, vmax = pv.min(), pv.max()
    cu, cv = (umin + umax) / 2.0, (vmin + vmax) / 2.0
    sx = max(umax - umin, MIN_SIZE) + PAD
    sz = max(vmax - vmin, MIN_SIZE) + PAD
    # Center back to world XZ (u,v is an orthonormal rotation).
    cx = cu * c + cv * s
    cz = -cu * s + cv * c

    # Cast to Python float so the values serialize cleanly when the endpoint
    # returns this dict straight through FastAPI (np.float64 can trip the
    # JSON encoder).
    return ([float(cx), float(cy), float(cz)],
            [float(sx), float(sy), float(sz)],
            [0.0, float(theta), 0.0])
