"""Per-segment convex hull computation.

Mirrors `3d-labeler/backend/supervoxels.py::compute_supervoxels`'s hull
output format, but works on any (points, instance_ids) pair so it's
preseg-mode agnostic. For each segment we attempt a scipy ConvexHull on
its points and fall back to an axis-aligned bounding box for degenerate
cases (n < 4 or coplanar points).

The result is packed into three flat arrays (already with global vertex
offsets baked in) so the frontend can drop them straight into a merged
BufferGeometry — exactly like the 3d-labeler's `SupervoxelHulls`
component.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError


def compute_hulls(
    points: np.ndarray,
    instance_ids: np.ndarray,
    *,
    resolution: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build packed convex-hull arrays for every segment.

    Parameters
    ----------
    points : (N, 3) float
    instance_ids : (N,) int32
        Segment id per point; -1 = unassigned (skipped).
    resolution : float
        Half-size used for the bbox fallback when a segment has < 4
        points (so single-point segments still show as a small cube).

    Returns
    -------
    vertices : (V, 3) float32   — concatenated hull vertices
    faces    : (F, 3) int32     — triangle indices into ``vertices``
    face_seg : (F,) int32       — segment id per triangle (for picking)
    """
    points = np.asarray(points, dtype=np.float64)
    instance_ids = np.asarray(instance_ids, dtype=np.int32)

    valid = instance_ids >= 0
    if not valid.any():
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )

    max_id = int(instance_ids.max())
    # Group point indices by segment via single argsort pass.
    order = np.argsort(instance_ids, kind="stable")
    sorted_ids = instance_ids[order]
    starts = np.searchsorted(sorted_ids, np.arange(max_id + 1))
    ends = np.searchsorted(sorted_ids, np.arange(max_id + 1) + 1)

    all_verts: list[np.ndarray] = []
    all_faces: list[np.ndarray] = []
    all_face_seg: list[np.ndarray] = []
    vertex_offset = 0
    half = resolution * 0.4

    for sv_id in range(max_id + 1):
        group = order[starts[sv_id]:ends[sv_id]]
        if group.size == 0:
            continue
        sv_pts = points[group]
        n = sv_pts.shape[0]

        verts: np.ndarray | None = None
        faces: np.ndarray | None = None

        if n >= 4:
            try:
                hull = ConvexHull(sv_pts)
                verts = sv_pts[hull.vertices].astype(np.float32)
                # Remap simplex indices (which point into sv_pts) to indices
                # into the hull's own vertex list.
                remap = np.full(n, -1, dtype=np.int32)
                remap[hull.vertices] = np.arange(len(hull.vertices), dtype=np.int32)
                faces = remap[hull.simplices].astype(np.int32)
            except QhullError:
                verts = None

        if verts is None or faces is None:
            if n == 1:
                center = sv_pts[0]
                mn = center - half
                mx = center + half
            else:
                pad = resolution * 0.05
                mn = sv_pts.min(axis=0) - pad
                mx = sv_pts.max(axis=0) + pad
            verts = np.array([
                [mn[0], mn[1], mn[2]],
                [mx[0], mn[1], mn[2]],
                [mx[0], mx[1], mn[2]],
                [mn[0], mx[1], mn[2]],
                [mn[0], mn[1], mx[2]],
                [mx[0], mn[1], mx[2]],
                [mx[0], mx[1], mx[2]],
                [mn[0], mx[1], mx[2]],
            ], dtype=np.float32)
            faces = np.array([
                [0, 2, 1], [0, 3, 2],
                [4, 5, 6], [4, 6, 7],
                [0, 1, 5], [0, 5, 4],
                [2, 3, 7], [2, 7, 6],
                [0, 4, 7], [0, 7, 3],
                [1, 2, 6], [1, 6, 5],
            ], dtype=np.int32)

        assert verts is not None and faces is not None  # for type checker
        all_verts.append(verts)
        all_faces.append(faces + vertex_offset)
        all_face_seg.append(np.full(faces.shape[0], sv_id, dtype=np.int32))
        vertex_offset += verts.shape[0]

    if not all_verts:
        return (
            np.zeros((0, 3), dtype=np.float32),
            np.zeros((0, 3), dtype=np.int32),
            np.zeros((0,), dtype=np.int32),
        )

    return (
        np.concatenate(all_verts, axis=0).astype(np.float32),
        np.concatenate(all_faces, axis=0).astype(np.int32),
        np.concatenate(all_face_seg, axis=0).astype(np.int32),
    )
