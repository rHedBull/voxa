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
) -> tuple[np.ndarray, list[dict]]:
    if mode == "voxel":
        from presegment_voxel import presegment as _run
        return _run(xyz, class_map=class_map, log=log, resolution=resolution)
    if mode == "ransac":
        from presegment_ransac import presegment as _run
        return _run(
            xyz, class_map=class_map, log=log, params=ransac_params,
            labeler_strict=labeler_strict,
        )
    if mode == "model":
        raise NotImplementedError(
            "Model-based presegmentation is not enabled in this build."
        )
    raise ValueError(f"Unknown presegment mode: {mode!r}")  # pyright: ignore[reportUnreachable]
