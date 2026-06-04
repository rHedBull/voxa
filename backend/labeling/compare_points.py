"""Per-point comparison of two class labelings (scan-schema v2 Compare).

Pure numpy. No FastAPI, no in-memory state. Both inputs are int class
arrays of equal length; -1 means unlabeled.
"""
from __future__ import annotations

import numpy as np

CONFUSION_TOP_N = 20


def compare_class_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    """Agreement, per-class IoU/precision/recall (A as reference), and the
    top disagreeing class pairs. ``agreement`` is computed over points
    labeled in AT LEAST ONE side — both-unlabeled points carry no signal
    and would inflate agreement on sparse labelings; ``agreement_all``
    keeps them for reference."""
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: a has {a.shape[0]}, b has {b.shape[0]}")
    a = a.astype(np.int32, copy=False)
    b = b.astype(np.int32, copy=False)
    n = int(a.shape[0])
    labeled_a = a >= 0
    labeled_b = b >= 0
    either = labeled_a | labeled_b
    eq = a == b

    out: dict = {
        "n_points": n,
        "n_labeled_a": int(labeled_a.sum()),
        "n_labeled_b": int(labeled_b.sum()),
        # Total coverage gaps: labeled in one source, unlabeled in the other
        # (the per-class missed_a/missed_b columns sum to these).
        "n_missed_a": int((labeled_b & ~labeled_a).sum()),
        "n_missed_b": int((labeled_a & ~labeled_b).sum()),
        "agreement": float(eq[either].mean()) if either.any() else None,
        "agreement_all": float(eq.mean()) if n else None,
    }

    classes = np.union1d(np.unique(a[labeled_a]), np.unique(b[labeled_b]))
    per_class = []
    for c in classes.tolist():
        in_a = a == c
        in_b = b == c
        tp = int((in_a & in_b).sum())
        union = int((in_a | in_b).sum())
        n_a = int(in_a.sum())
        n_b = int(in_b.sum())
        per_class.append({
            "class_id": int(c),
            "iou": tp / union if union else None,
            "precision": tp / n_b if n_b else None,
            "recall": tp / n_a if n_a else None,
            "n_a": n_a,
            "n_b": n_b,
            # Coverage gaps: points one source calls class c that the OTHER
            # left entirely unlabeled — "should have been labeled" misses,
            # distinct from class confusion (which needs both sides labeled).
            "missed_a": int((in_b & ~labeled_a).sum()),
            "missed_b": int((in_a & ~labeled_b).sum()),
        })
    out["per_class"] = per_class

    # Confusion: both labeled, classes differ. Vectorized pair counting via a
    # single unique() over packed (a, b) pairs — no Python loop over points.
    both = labeled_a & labeled_b & ~eq
    if both.any():
        pairs = a[both].astype(np.int64) * 100_000 + b[both].astype(np.int64)
        uniq, counts = np.unique(pairs, return_counts=True)
        # stable sort so tied counts keep a deterministic (pair-id) order
        order = np.argsort(-counts, kind="stable")[:CONFUSION_TOP_N]
        out["confusion"] = [
            {"a_class": int(uniq[i] // 100_000), "b_class": int(uniq[i] % 100_000),
             "n": int(counts[i])}
            for i in order
        ]
    else:
        out["confusion"] = []
    return out
