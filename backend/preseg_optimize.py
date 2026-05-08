"""Heuristic scoring for RANSAC presegmentation parameter tuning.

`score_segmentation` evaluates an instance partition by fitting the three
primitives (plane / cylinder / sphere) to each segment and returning a
weighted-mean best-fit RMS, biased toward partitions with more segments via
a small log-count regulariser. Lower is better; degenerate segment counts
return a large penalty so the optimiser steers away from them.
"""
from __future__ import annotations

import numpy as np

PENALTY = 1e6
MIN_SEG_PTS = 30
LAMBDA_SEG = 1e-3
MIN_SEGMENTS = 5
MAX_SEGMENTS = 5000


def _plane_rms(pts: np.ndarray) -> float:
    centered = pts - pts.mean(axis=0)
    cov = (centered.T @ centered) / max(len(pts) - 1, 1)
    try:
        eigvals = np.linalg.eigvalsh(cov)
    except np.linalg.LinAlgError:
        return float("inf")
    smallest = float(eigvals[0])
    if not np.isfinite(smallest) or smallest < 0:
        return float("inf")
    return float(np.sqrt(smallest))


def _sphere_rms(pts: np.ndarray) -> float:
    n = len(pts)
    A = np.hstack([2.0 * pts, np.ones((n, 1))])
    b = np.sum(pts * pts, axis=1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf")
    cx, cy, cz, d = sol
    r2 = d + cx * cx + cy * cy + cz * cz
    if not np.isfinite(r2) or r2 <= 0:
        return float("inf")
    r = np.sqrt(r2)
    center = np.array([cx, cy, cz])
    dist = np.linalg.norm(pts - center, axis=1)
    residual = dist - r
    return float(np.sqrt(np.mean(residual * residual)))


def _cylinder_rms(pts: np.ndarray) -> float:
    centroid = pts.mean(axis=0)
    centered = pts - centroid
    try:
        _, _, vh = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return float("inf")
    axis = vh[0]
    basis = vh[1:3]
    proj = centered @ basis.T

    n = len(proj)
    A = np.hstack([2.0 * proj, np.ones((n, 1))])
    b = np.sum(proj * proj, axis=1)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except np.linalg.LinAlgError:
        return float("inf")
    cx, cy, d = sol
    r2 = d + cx * cx + cy * cy
    if not np.isfinite(r2) or r2 <= 0:
        return float("inf")
    r = np.sqrt(r2)
    center2d = np.array([cx, cy])
    radial = np.linalg.norm(proj - center2d, axis=1) - r
    rms = float(np.sqrt(np.mean(radial * radial)))
    _ = axis
    return rms


def _seg_score(pts: np.ndarray) -> float:
    candidates = [_plane_rms(pts), _cylinder_rms(pts), _sphere_rms(pts)]
    finite = [c for c in candidates if np.isfinite(c)]
    if not finite:
        return float("inf")
    return min(finite)


def score_segmentation(xyz: np.ndarray, instance_ids: np.ndarray) -> float:
    """Return heuristic objective for a presegmentation; lower is better."""
    xyz = np.asarray(xyz, dtype=np.float64)
    instance_ids = np.asarray(instance_ids)

    mask = instance_ids >= 0
    xyz = xyz[mask]
    ids = instance_ids[mask]

    unique_ids = np.unique(ids)
    n_segments = len(unique_ids)
    if n_segments < MIN_SEGMENTS or n_segments > MAX_SEGMENTS:
        return PENALTY

    rms_values: list[float] = []
    weights: list[int] = []
    for seg_id in unique_ids:
        seg_pts = xyz[ids == seg_id]
        if len(seg_pts) < MIN_SEG_PTS:
            continue
        score = _seg_score(seg_pts)
        if not np.isfinite(score):
            continue
        rms_values.append(score)
        weights.append(len(seg_pts))

    if not rms_values:
        return PENALTY

    rms_arr = np.asarray(rms_values, dtype=np.float64)
    w_arr = np.asarray(weights, dtype=np.float64)
    weighted_mean = float(np.sum(rms_arr * w_arr) / np.sum(w_arr))
    return weighted_mean - LAMBDA_SEG * float(np.log(n_segments))
