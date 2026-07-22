"""Per-instance convex-hull .glb generation for the labels-export endpoint.

Ported from the retired scripts/build_instance_meshes.py: each mesh is the
convex hull of that instance's own labeled points (not a reconstruction from
an external mesh), so it can never be in a different coordinate frame from
the point cloud that produced it. Convex, so it won't tightly hug a
bent/branching pipe, but it's real geometry, cheap, and always well-formed.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError

# Quality bar, not just the geometric floor (a hull needs >= 4 non-coplanar
# points to exist at all) -- a technically-valid hull from a handful of
# points is too noisy for collision detection to trust.
MIN_POINTS_FOR_MESH = 100


def build_instance_glbs(
    points: np.ndarray,
    instance_ids: np.ndarray,
    surviving_ids: set[int],
) -> tuple[dict[int, bytes], list[tuple[int, str]]]:
    """Convex-hull .glb per id in `surviving_ids`.

    Returns (glbs, skipped): glbs maps instance_id -> glb bytes; skipped
    lists (instance_id, reason) for ids with < MIN_POINTS_FOR_MESH points or
    a degenerate/coplanar hull (QhullError).
    """
    points = np.asarray(points, dtype=np.float64)
    instance_ids = np.asarray(instance_ids)

    glbs: dict[int, bytes] = {}
    skipped: list[tuple[int, str]] = []

    for asset_id in sorted(surviving_ids):
        pts = points[instance_ids == asset_id]
        if len(pts) < MIN_POINTS_FOR_MESH:
            skipped.append((asset_id, f"only {len(pts)} points"))
            continue
        try:
            hull = ConvexHull(pts)
        except QhullError as e:
            skipped.append((asset_id, f"degenerate/coplanar points ({e.__class__.__name__})"))
            continue
        mesh = trimesh.Trimesh(vertices=pts, faces=hull.simplices, process=True)
        glbs[asset_id] = mesh.export(file_type="glb")

    return glbs, skipped
