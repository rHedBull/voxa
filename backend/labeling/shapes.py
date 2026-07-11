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
    raise ValueError(f"unknown shape type: {kind!r}")
