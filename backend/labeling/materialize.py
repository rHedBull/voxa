"""Density-agnostic label materialization: produce per-point (positions, colors,
class_ids, instance_ids) for a session at any target density. Regime A =
down-sample by index-transfer (exact); regime B (denser) added in later tasks."""
import numpy as np
from app.core import _subsample_indices  # seeded, ascending indices into range(N)


def materialize_downsample(positions, colors, class_ids, instance_ids, n):
    """Regime A: exact down-sample. n >= len -> identity; else index every array
    by the same seeded subsample so labels are transferred, never interpolated."""
    N = len(positions)
    if n >= N:
        return positions, colors, class_ids, instance_ids
    idx = _subsample_indices(N, n)
    return positions[idx], colors[idx], class_ids[idx], instance_ids[idx]
