"""PLY debug-visualization helpers shared by the dry_sam3 scripts.

Torch-free: read raw xyz, build a random per-instance color palette, and write an
instance-id-colored PLY. Distinct from ``backend/scenes/point_cloud.py`` (which
serializes a PointCloud) and from ``text_prompt_class_voting.write_labeled_ply``
(class palette + confidence fade) — this one colors arbitrary instance ids.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement


def load_xyz(ply_path: Path) -> np.ndarray:
    p = PlyData.read(str(ply_path))
    v = p["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)


def random_palette(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = rng.uniform(0, 1, n)
    s = rng.uniform(0.55, 0.95, n)
    v = rng.uniform(0.70, 1.0, n)
    i = (h * 6).astype(int) % 6
    f = h * 6 - i
    p = v * (1 - s); q = v * (1 - f * s); t = v * (1 - (1 - f) * s)
    rgb = np.zeros((n, 3), dtype=np.float32)
    for k, parts in enumerate([(v, t, p), (q, v, p), (p, v, t),
                                (p, q, v), (t, p, v), (v, p, q)]):
        sel = i == k
        rgb[sel] = np.stack([parts[0][sel], parts[1][sel], parts[2][sel]], -1)
    return (rgb * 255).astype(np.uint8)


def write_colored_ply(path: Path, xyz: np.ndarray, ids: np.ndarray, seed: int = 0):
    n_inst = int(ids.max()) + 1 if ids.size and ids.max() >= 0 else 0
    palette = random_palette(max(1, n_inst), seed=seed)
    colors = np.full((xyz.shape[0], 3), 60, dtype=np.uint8)
    has = ids >= 0
    colors[has] = palette[ids[has]]
    rec = np.empty(xyz.shape[0], dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("instance", "i4"),
    ])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    rec["instance"] = ids.astype(np.int32)
    PlyData([PlyElement.describe(rec, "vertex")], text=False).write(str(path))
