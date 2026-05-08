"""Heuristic scoring for RANSAC presegmentation parameter tuning.

`score_segmentation` evaluates an instance partition by fitting the three
primitives (plane / cylinder / sphere) to each segment and returning a
weighted-mean best-fit RMS, biased toward partitions with more segments via
a small log-count regulariser. Lower is better; degenerate segment counts
return a large penalty so the optimiser steers away from them.
"""
from __future__ import annotations

import threading
from typing import Any, Callable, Optional

import numpy as np

PENALTY = 1e6
MIN_SEG_PTS = 30
LAMBDA_SEG = 5e-3
MIN_SEGMENTS = 5
MAX_SEGMENTS = 5000
# Soft penalty when any single segment hogs more than this fraction of all
# assigned points. Each percentage-point above the threshold adds OVERSIZE_GAIN
# to the score, so the optimiser actively prefers splits.
OVERSIZE_FRAC = 0.20
OVERSIZE_GAIN = 0.5


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

    unique_ids, counts = np.unique(ids, return_counts=True)
    n_segments = len(unique_ids)
    if n_segments < MIN_SEGMENTS or n_segments > MAX_SEGMENTS:
        return PENALTY

    rms_values: list[float] = []
    for seg_id in unique_ids:
        seg_pts = xyz[ids == seg_id]
        if len(seg_pts) < MIN_SEG_PTS:
            continue
        score = _seg_score(seg_pts)
        if not np.isfinite(score):
            continue
        rms_values.append(score)

    if not rms_values:
        return PENALTY

    # Unweighted: each qualifying segment contributes equally so a single
    # giant pure segment can't drown out many small pure ones.
    mean_rms = float(np.mean(rms_values))
    log_bonus = LAMBDA_SEG * float(np.log(n_segments))
    total_assigned = int(counts.sum())
    largest_frac = float(counts.max() / max(total_assigned, 1))
    oversize = max(0.0, largest_frac - OVERSIZE_FRAC) * OVERSIZE_GAIN
    return mean_rms + oversize - log_bonus


SEARCH_SPACE: dict[str, tuple[float, float, str]] = {
    "plane_distance_threshold": (0.005, 0.10, "logfloat"),
    "plane_min_inliers":        (20, 300, "int"),
    "max_planes":               (5, 50, "int"),
    "flat_thresh":              (0.1, 2.0, "float"),
    "cylinder_ratio_thresh":    (1.5, 8.0, "float"),
    "cyl_search_radius":        (0.05, 0.30, "logfloat"),
    "cyl_axis_thresh":          (0.85, 0.99, "float"),
    "cyl_radius_ratio":         (1.2, 3.0, "float"),
    "cyl_distance_threshold":   (0.005, 0.10, "logfloat"),
    "merge_axis_dot":           (0.85, 0.99, "float"),
    "merge_radius_ratio":       (1.1, 3.0, "float"),
}


def _suggest_params(trial: Any) -> dict[str, float]:
    params: dict[str, float] = {}
    for name, (lo, hi, kind) in SEARCH_SPACE.items():
        if kind == "int":
            # Cast to float so downstream RANSAC code sees a uniform numeric type.
            params[name] = float(trial.suggest_int(name, int(lo), int(hi)))
        elif kind == "logfloat":
            params[name] = float(trial.suggest_float(name, lo, hi, log=True))
        else:
            params[name] = float(trial.suggest_float(name, lo, hi, log=False))
    return params


def run_study(
    xyz_sub: np.ndarray,
    *,
    n_trials: int,
    cancel_event: threading.Event,
    progress_cb: Callable[[dict], None],
    class_map: Optional[dict[str, int]] = None,
) -> dict:
    import optuna
    from presegment_ransac import presegment as _ransac

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ran = 0
    trials_log: list[dict] = []

    def objective(trial: "optuna.trial.Trial") -> float:
        nonlocal ran
        if cancel_event.is_set():
            raise optuna.TrialPruned()
        params = _suggest_params(trial)
        ran += 1
        try:
            instance_ids, _summary = _ransac(
                xyz_sub,
                class_map=class_map,
                log=lambda *_: None,
                params=params,
            )
            score = float(score_segmentation(xyz_sub, instance_ids))
        except Exception:
            score = float(PENALTY)
        trials_log.append({"params": dict(params), "score": score})
        return score

    sampler = optuna.samplers.TPESampler(seed=0)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    def _cb(study_: "optuna.Study", trial: "optuna.trial.FrozenTrial") -> None:
        try:
            best_score: Optional[float] = float(study_.best_value)
            best_params: Optional[dict] = dict(study_.best_params)
        except ValueError:
            best_score = None
            best_params = None
        progress_cb({
            "trial": trial.number,
            "total": n_trials,
            "best_score": best_score,
            "best_params": best_params,
        })
        if cancel_event.is_set():
            study_.stop()

    study.optimize(objective, n_trials=n_trials, callbacks=[_cb])

    try:
        best_params = dict(study.best_params)
        best_score = float(study.best_value)
    except ValueError:
        best_params = {}
        best_score = float(PENALTY)

    return {
        "best_params": best_params,
        "best_score": best_score,
        "n_trials_run": ran,
        "trials": trials_log,
    }
