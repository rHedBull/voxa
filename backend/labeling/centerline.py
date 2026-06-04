"""Centerline tube extraction + per-session centerline persistence.

A "path" is dict {points: [[x,y,z],...], radius: float, smooth: bool} in the
recentered frame (same frame as SegmentSession.positions). Extraction is the
union over paths of all points within `radius` of the polyline — the implied
tube is segment cylinders plus spheres at the joints (distance-to-segment
metric), per the design spec.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CENTERLINES_FILENAME = "centerlines.json"


def _segment_mask(positions: np.ndarray, a: np.ndarray, b: np.ndarray,
                  r2: float) -> np.ndarray:
    """Bool mask: squared distance from each point to segment a→b ≤ r2."""
    d = b - a
    l2 = float(d @ d)
    if l2 < 1e-12:                      # degenerate segment → sphere test
        diff = positions - a
        return np.einsum("ij,ij->i", diff, diff) <= r2
    t = np.clip((positions - a) @ d / l2, 0.0, 1.0)
    closest = a + t[:, None] * d
    diff = positions - closest
    return np.einsum("ij,ij->i", diff, diff) <= r2


def tube_indices(positions: np.ndarray, paths: list[dict]) -> np.ndarray:
    """Unique int32 indices of points within any path's tube."""
    positions = np.asarray(positions, dtype=np.float32)
    mask = np.zeros(positions.shape[0], dtype=bool)
    for p in paths:
        pts = np.asarray(sample_path(p), dtype=np.float32)
        r2 = float(p["radius"]) ** 2
        for i in range(len(pts) - 1):
            mask |= _segment_mask(positions, pts[i], pts[i + 1], r2)
    return np.flatnonzero(mask).astype(np.int32)


def sample_path(path: dict) -> np.ndarray:
    return np.asarray(path["points"], dtype=np.float32)


def load_centerlines(session_dir: Path) -> dict:
    f = Path(session_dir) / CENTERLINES_FILENAME
    if not f.exists():
        return {"paths": []}
    return json.loads(f.read_text())


def update_centerlines(session_dir: Path, instance_id: int, class_id: int,
                       paths: list[dict], merged_from: list[int]) -> dict:
    """Replace-by-instance_id write: drop stored paths for `instance_id` and
    any id in `merged_from`, then append the new ones. Keeps re-editing a
    pipe from duplicating stored paths (spec: Persistence)."""
    doc = load_centerlines(session_dir)
    dead = {int(instance_id), *(int(m) for m in merged_from)}
    kept = [p for p in doc["paths"] if p.get("instance_id") not in dead]
    for p in paths:
        kept.append({
            "points": [[float(c) for c in pt] for pt in p["points"]],
            "radius": float(p["radius"]),
            "smooth": bool(p.get("smooth", False)),
            "class_id": int(class_id),
            "instance_id": int(instance_id),
        })
    doc = {"paths": kept}
    f = Path(session_dir) / CENTERLINES_FILENAME
    f.write_text(json.dumps(doc, indent=1))
    return doc
