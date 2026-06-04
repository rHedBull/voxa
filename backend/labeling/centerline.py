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
    """Unique int32 indices of points within any path's tube.

    AABB prefilter: for each path, restrict candidate points to those inside
    the bounding box of the sampled path points expanded by the tube radius.
    This avoids allocating O(N) temporaries over the full cloud per segment
    when paths are small relative to the total cloud volume.
    """
    positions = np.asarray(positions, dtype=np.float32)
    mask = np.zeros(positions.shape[0], dtype=bool)
    for p in paths:
        pts = np.asarray(sample_path(p), dtype=np.float32)
        radius = float(p["radius"])
        r2 = radius ** 2
        # AABB of sampled path points expanded by radius.
        lo = pts.min(axis=0) - radius
        hi = pts.max(axis=0) + radius
        cand = np.where(
            (positions[:, 0] >= lo[0]) & (positions[:, 0] <= hi[0]) &
            (positions[:, 1] >= lo[1]) & (positions[:, 1] <= hi[1]) &
            (positions[:, 2] >= lo[2]) & (positions[:, 2] <= hi[2])
        )[0]
        if cand.size == 0:
            continue
        sub = positions[cand]
        sub_mask = np.zeros(cand.size, dtype=bool)
        for i in range(len(pts) - 1):
            sub_mask |= _segment_mask(sub, pts[i], pts[i + 1], r2)
        mask[cand[sub_mask]] = True
    return np.flatnonzero(mask).astype(np.int32)


def sample_path(path: dict) -> np.ndarray:
    """Control points → polyline chords. Straight paths pass through
    unchanged; smooth paths get Catmull-Rom sampling with target step
    ≈ radius/2 (worst-case chord stays < radius near apexes), so the tube
    test on chords can't visibly cut corners."""
    pts = np.asarray(path["points"], dtype=np.float32)
    if not path.get("smooth") or len(pts) < 3:
        return pts
    step = max(float(path["radius"]) / 2.0, 1e-4)
    # Endpoint-duplicated control polygon so the curve spans all controls.
    ctrl = np.vstack([pts[0], pts, pts[-1]])
    out = [pts[0]]
    for i in range(1, len(ctrl) - 2):
        p0, p1, p2, p3 = ctrl[i - 1], ctrl[i], ctrl[i + 1], ctrl[i + 2]
        seg_len = float(np.linalg.norm(p2 - p1))
        n = max(int(np.ceil(seg_len / step)), 1)
        for t in np.linspace(0, 1, n + 1)[1:]:
            t2, t3 = t * t, t * t * t
            v = (0.5 * ((2 * p1) + (-p0 + p2) * t
                 + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
            out.append(v.astype(np.float32))
    return np.asarray(out, dtype=np.float32)


def load_centerlines(session_dir: Path) -> dict:
    f = Path(session_dir) / CENTERLINES_FILENAME
    if not f.exists():
        return {"paths": []}
    data = json.loads(f.read_text())
    if "paths" not in data:
        raise ValueError(
            f"malformed centerlines.json in {session_dir}: missing 'paths'"
        )
    return data


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
