"""Deterministic content fingerprint for a point cloud (scan-schema v1.3 §3.2)."""
from __future__ import annotations

import hashlib

import numpy as np


def cloud_fingerprint(xyz: np.ndarray) -> str:
    """sha256 over xyz quantized to integer millimetres, lexicographically
    sorted, serialized as little-endian int32.

    - Order-independent (sorted) -> voxel/thinning reordering doesn't change it.
    - Robust to float round-trips (mm quantize).
    - xyz only; identifies a variant's stored cloud *in its own frame*
      (NOT frame-invariant, NOT equal across variants). Content-identity
      heuristic, not a cryptographic guarantee.
    """
    q = np.round(np.asarray(xyz, dtype=np.float64) * 1000.0).astype("<i4")
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {q.shape}")
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))
    digest = hashlib.sha256(np.ascontiguousarray(q[order]).tobytes()).hexdigest()
    return "sha256:" + digest
