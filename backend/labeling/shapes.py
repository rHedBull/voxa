"""Shape -> full-resolution point-index resolvers for the apply-shape endpoint.

Each labeling tool selects points via a geometric *shape*; the backend resolves
any shape to the indices of the full-res points inside it, then reassigns them.
The OBB math mirrors the frontend preview `pointsInsideOBBLabel` (mode-label.jsx)
so the applied label matches the on-screen selection box exactly.
"""
import numpy as np

from scenes.reproject import euler_xyz_matrix


def obb_indices(positions: np.ndarray, box: dict) -> np.ndarray:
    """Int32 indices of points inside the oriented box.

    box = {center:[3], size:[3] (full side lengths), rotation:[rx,ry,rz]}
    Rotation is Euler XYZ in radians, matching Three.js `Euler(..., 'XYZ')`.
    local = R^T . (p - center); a point is inside iff |local| <= size/2 per axis.
    """
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    center = np.asarray(box["center"], dtype=np.float64)
    half = np.asarray(box["size"], dtype=np.float64) / 2.0
    R = euler_xyz_matrix(*(float(v) for v in box["rotation"]))

    # local = R^T . (p - c), vectorized as (p - c) @ R.
    local = (positions.astype(np.float64) - center) @ R
    inside = np.all(np.abs(local) <= half, axis=1)
    return np.nonzero(inside)[0].astype(np.int32)


def prism_indices(positions: np.ndarray, prism: dict) -> np.ndarray:
    """Int32 indices of points inside a vertical prism.

    prism = {polygon:[[x,z],...] (>=3 verts, XZ world plane),
             y0: base-plane height, height: upward extrusion (>0)}
    A point is inside iff its (x,z) is inside the polygon AND
    y0 <= y <= y0 + height. Concave polygons are supported (ray-cast rule);
    < 3 vertices or a zero/negative height select nothing.
    """
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    poly = np.asarray(prism["polygon"], dtype=np.float64)
    height = float(prism["height"])
    if poly.shape[0] < 3 or height <= 0.0:
        return np.empty(0, dtype=np.int32)
    y0 = float(prism["y0"])
    y = positions[:, 1].astype(np.float64)
    in_band = (y >= y0) & (y <= y0 + height)
    inside = in_band.copy()
    if in_band.any():
        px = positions[in_band, 0].astype(np.float64)
        pz = positions[in_band, 2].astype(np.float64)
        inside[in_band] = _point_in_polygon_xz(px, pz, poly)
    return np.nonzero(inside)[0].astype(np.int32)


def _point_in_polygon_xz(px: np.ndarray, pz: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized even-odd ray-cast point-in-polygon for arrays of (px,pz)
    against polygon vertices `poly` (M×2, columns = x,z). Points exactly on an
    edge are treated as inside-or-out per the standard crossing rule (not
    guaranteed either way — acceptable at LiDAR density)."""
    x1 = poly[:, 0]; z1 = poly[:, 1]
    x2 = np.roll(x1, -1); z2 = np.roll(z1, -1)
    inside = np.zeros(px.shape[0], dtype=bool)
    for i in range(poly.shape[0]):
        # Edge (x1[i],z1[i]) -> (x2[i],z2[i]). A horizontal ray in +x from the
        # point crosses this edge iff the edge straddles pz and the crossing x
        # is to the right of px.
        cond = ((z1[i] > pz) != (z2[i] > pz))
        # Avoid div-by-zero on horizontal edges (cond is already False there).
        denom = np.where(z2[i] != z1[i], z2[i] - z1[i], 1.0)
        xints = x1[i] + (pz - z1[i]) * (x2[i] - x1[i]) / denom
        inside ^= cond & (px < xints)
    return inside


def shape_indices(positions: np.ndarray, shape: dict) -> np.ndarray:
    """Resolve any shape descriptor to the int32 indices of the full-res points
    it contains. Dispatches on shape['type']."""
    kind = shape.get("type")
    if kind == "obb":
        return obb_indices(positions, shape)
    if kind == "tube":
        from labeling.centerline import tube_indices
        pts = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        return tube_indices(pts, shape["paths"])
    if kind == "prism":
        return prism_indices(positions, shape)
    raise ValueError(f"unknown shape type: {kind!r}")
