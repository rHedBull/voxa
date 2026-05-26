"""Presegmentation dispatcher.

Three modes:
- ``voxel``  — uniform spatial supervoxels (Open3D voxel downsample + trace).
               Fast, finely grained, no semantic features.
- ``ransac`` — curvature + RANSAC primitive grouping (planes / cylinders /
               spheres). Slower, larger groups, primitive-aware.
- ``model``  — learned merge model. Not yet wired up (raises).

Public API:
    presegment(xyz, *, mode="voxel", class_map=None, log=print, resolution=0.05)
        -> (instance_ids, summary)
"""
from __future__ import annotations

from typing import Callable, Literal, Optional

import numpy as np

PresegMode = Literal["voxel", "ransac", "model"]


def presegment(
    xyz: np.ndarray,
    *,
    mode: PresegMode = "voxel",
    class_map: Optional[dict[str, int]] = None,
    log: Callable[[str], None] = print,
    resolution: float = 0.05,
    ransac_params: Optional[dict[str, float]] = None,
    labeler_strict: bool = False,
    features: Optional[np.ndarray] = None,
    feature_seen: Optional[np.ndarray] = None,
    feature_split_min_size: int = 3000,
    feature_split_target_size: int = 5000,
    feature_split_max_k: int = 8,
) -> tuple[np.ndarray, list[dict]]:
    if mode == "voxel":
        from preseg.presegment_voxel import presegment as _run
        return _run(xyz, class_map=class_map, log=log, resolution=resolution)
    if mode == "ransac":
        from preseg.presegment_ransac import presegment as _run
        return _run(
            xyz, class_map=class_map, log=log, params=ransac_params,
            labeler_strict=labeler_strict,
            features=features, feature_seen=feature_seen,
            feature_split_min_size=feature_split_min_size,
            feature_split_target_size=feature_split_target_size,
            feature_split_max_k=feature_split_max_k,
        )
    if mode == "model":
        raise NotImplementedError(
            "Model-based presegmentation is not enabled in this build."
        )
    raise ValueError(f"Unknown presegment mode: {mode!r}")  # pyright: ignore[reportUnreachable]
