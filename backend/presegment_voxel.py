"""Voxel supervoxel presegmentation.

Replaces the RANSAC pipeline with Open3D voxel downsampling + trace, which
produces small spatially-compact supervoxels instead of large geometric
primitives (planes / cylinders). This matches the 3d-labeler's approach and
gives finer-grained, more useful presegments for interactive labeling.

Public API (unchanged):
    presegment(xyz, *, class_map=None, log=None) -> (instance_ids, summary)
"""
from __future__ import annotations

from typing import Callable, Optional

import numpy as np


def presegment(
    xyz: np.ndarray,
    *,
    class_map: Optional[dict[str, int]] = None,
    log: Callable[[str], None] = print,
    resolution: float = 0.05,
) -> tuple[np.ndarray, list[dict]]:
    """Run voxel supervoxel presegmentation.

    Parameters
    ----------
    xyz : (N, 3) float
        Point positions.
    class_map : dict[str, int] | None
        Unused (kept for API compatibility); all supervoxels get class_id=-1.
    log : callable
        Progress sink.
    resolution : float
        Voxel size in scene units (metres). Smaller → more segments.

    Returns
    -------
    instance_ids : (N,) int32
        Per-point supervoxel id (-1 = unassigned, should not occur).
    summary : list[dict]
        One entry per supervoxel with keys id, class_id, label, n_points.
    """
    import open3d as o3d

    points = np.asarray(xyz, dtype=np.float64)
    n_total = len(points)
    log(f"Voxel presegmentation: {n_total} pts, resolution={resolution}m")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    _, _, indices = pcd.voxel_down_sample_and_trace(
        voxel_size=resolution,
        min_bound=pcd.get_min_bound() - resolution,
        max_bound=pcd.get_max_bound() + resolution,
    )

    instance_ids = np.full(n_total, -1, dtype=np.int32)
    summary: list[dict] = []

    for sv_id, group in enumerate(indices):
        for idx in group:
            instance_ids[int(idx)] = sv_id
        summary.append({
            "id": sv_id,
            "class_id": -1,
            "label": "supervoxel",
            "n_points": len(group),
        })

    log(f"Done: {len(summary)} supervoxels, "
        f"{int((instance_ids >= 0).sum())}/{n_total} pts assigned")
    return instance_ids, summary
