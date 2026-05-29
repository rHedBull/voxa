"""Merge multiple annotation runs into one (scan-schema v1.3 §4.6.3).

Pure array logic (no IO). All input runs must be on the SAME variant/point set:
``class_lists[i]`` and ``seg_lists[i]`` are ``(N,)`` int arrays, ``-1`` = unlabeled.
Returns ``(merged_class, merged_seg, conflicts, provenance)`` where ``conflicts`` is
a bool mask of points where ≥2 runs disagree on a (non -1) class, and ``provenance``
gives the winning run index per point (``-1`` where unlabeled). Sources are never
mutated.
"""
from __future__ import annotations

import numpy as np


def merge_runs(class_lists, seg_lists, *, policy: str = "priority",
               priority_order=None):
    C = np.stack([np.asarray(c, dtype=np.int64) for c in class_lists])  # (k, N)
    S = np.stack([np.asarray(s, dtype=np.int64) for s in seg_lists])    # (k, N)
    k, n = C.shape
    if S.shape != C.shape:
        raise ValueError("class and segment arrays must have matching shapes")
    idx = np.arange(n)
    labeled = C != -1

    # conflict = ≥2 runs labeled AND they disagree on class
    n_labeled = labeled.sum(0)
    masked_max = np.where(labeled, C, np.iinfo(np.int64).min).max(0)
    masked_min = np.where(labeled, C, np.iinfo(np.int64).max).min(0)
    conflicts = (n_labeled >= 2) & (masked_max != masked_min)

    if policy == "priority":
        order = list(priority_order) if priority_order is not None else list(range(k))
        winner = np.full(n, -1, dtype=np.int64)
        for ri in order:
            take = (winner == -1) & labeled[ri]
            winner[take] = ri
    elif policy == "vote":
        # votes[ri] = how many labeled runs share ri's class (0 if ri unlabeled)
        votes = np.zeros((k, n), dtype=np.int64)
        for ri in range(k):
            votes[ri] = ((C == C[ri]) & labeled & labeled[ri]).sum(0)
        best = votes.max(0)
        winner_run = votes.argmax(0)             # first run achieving `best`
        win_class = C[winner_run, idx]
        # tie if another run also at `best` carries a different labeled class
        at_best = (votes == best) & labeled
        tie = (at_best & (C != win_class)).any(0) & (best > 0)
        winner = np.where((best > 0) & ~tie, winner_run, -1)
    else:
        raise ValueError(f"unknown merge policy: {policy!r}")

    w = np.clip(winner, 0, k - 1)
    merged_class = np.where(winner >= 0, C[w, idx], -1).astype(np.int32)
    merged_seg = np.where(winner >= 0, S[w, idx], -1).astype(np.int32)
    return merged_class, merged_seg, conflicts, winner.astype(np.int32)
