# backend/labeling/outliers.py
"""Statistical Outlier Removal (SOR) over a point subset.

One pure function, shared by the global denoise and per-selection
"remove outliers" features. The population it judges against is the
caller-supplied `subset_idx`, so the same code flags a floating speck
against the whole cloud (global) or an edge-stray against a selection's
core (per-selection). See
docs/superpowers/specs/2026-07-20-outlier-detection-filtering-design.md.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def statistical_outlier_indices(
    positions: np.ndarray,
    subset_idx: np.ndarray,
    k: int = 16,
    std_ratio: float = 2.0,
    tree: "cKDTree | None" = None,
) -> np.ndarray:
    """Return the indices (into `positions`) of the spatial outliers among
    `positions[subset_idx]`.

    For each subset point, compute the mean distance to its `k` nearest
    neighbours within the subset; flag points whose mean distance exceeds
    `mean + std_ratio * std` of that distribution. Density-adaptive: the
    threshold comes from the subset's own distances.

    `tree`, if given, MUST be a `cKDTree` already built over exactly
    `positions[subset_idx]` in the same order — an optimisation the caller
    uses only for the whole-cloud case (`subset_idx` covers every point), so
    a re-run doesn't rebuild a multi-million-point tree. Omit it otherwise.
    """
    subset_idx = np.asarray(subset_idx, dtype=np.int64).ravel()
    n = subset_idx.size
    # Need at least k+1 points (self + k neighbours) to judge anything.
    if n < k + 1:
        return np.empty(0, dtype=np.int64)
    pts = positions[subset_idx]
    if tree is None:
        tree = cKDTree(pts)
    # query k+1 because the first neighbour is the point itself (dist 0).
    dists, _ = tree.query(pts, k=k + 1)
    mean_knn = dists[:, 1:].mean(axis=1)      # drop the self-distance column
    mu = float(mean_knn.mean())
    sigma = float(mean_knn.std())
    if sigma == 0.0:                           # zero variance -> no outliers
        return np.empty(0, dtype=np.int64)
    threshold = mu + std_ratio * sigma
    is_outlier = mean_knn > threshold
    return subset_idx[is_outlier]
