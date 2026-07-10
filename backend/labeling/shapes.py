"""Shape -> full-resolution point-index resolvers for the apply-shape endpoint.

Each labeling tool selects points via a geometric *shape*; the backend resolves
any shape to the indices of the full-res points inside it, then reassigns them.
The OBB math mirrors the frontend preview `pointsInsideOBBLabel` (mode-label.jsx)
so the applied label matches the on-screen selection box exactly.
"""
import numpy as np


def obb_indices(positions: np.ndarray, box: dict) -> np.ndarray:
    """Int32 indices of points inside the oriented box.

    box = {center:[3], size:[3] (full side lengths), rotation:[rx,ry,rz]}
    Rotation is Euler XYZ in radians, matching Three.js `Euler(..., 'XYZ')`.
    local = R^T . (p - center); a point is inside iff |local| <= size/2 per axis.
    """
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    cx, cy, cz = (float(v) for v in box["center"])
    sx, sy, sz = (float(v) for v in box["size"])
    rx, ry, rz = (float(v) for v in box["rotation"])
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0

    cxr, sxr = np.cos(rx), np.sin(rx)
    cyr, syr = np.cos(ry), np.sin(ry)
    czr, szr = np.cos(rz), np.sin(rz)
    # Columns of R (world = R.local); local = R^T.(p-c) picks these as the
    # projection axes -- identical basis to pointsInsideOBBLabel.
    ax0 = (cyr * czr,               cyr * szr,               -syr)
    ax1 = (sxr * syr * czr - cxr * szr, sxr * syr * szr + cxr * czr, sxr * cyr)
    ax2 = (cxr * syr * czr + sxr * szr, cxr * syr * szr - sxr * czr, cxr * cyr)

    rel = positions.astype(np.float64) - (cx, cy, cz)
    lx = rel @ np.asarray(ax0)
    ly = rel @ np.asarray(ax1)
    lz = rel @ np.asarray(ax2)
    inside = (np.abs(lx) <= hx) & (np.abs(ly) <= hy) & (np.abs(lz) <= hz)
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
