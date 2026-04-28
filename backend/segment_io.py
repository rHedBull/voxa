"""SCHEMA-aware reader for prelabel/, writer for labels/, history pruning.

Pure I/O. No FastAPI, no in-memory state. Loaders return aligned arrays;
writers validate invariants and recompute gt_segment_metadata.json from the
arrays before flushing to disk.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


def load_prelabel(
    scan_dir: Path, n_points: int
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Read prelabel/ if present. Returns (class_ids int8, instance_ids int32)
    or None when no prelabel exists / arrays are malformed."""
    pre = scan_dir / "prelabel"
    inst_path = pre / "ransac_instance_ids.npy"
    summary_path = pre / "ransac_segment_summary.json"
    if not inst_path.exists() or not summary_path.exists():
        return None
    try:
        instance_ids = np.load(inst_path).astype(np.int32)
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if instance_ids.shape != (n_points,):
        return None
    seg_to_class = {int(s["id"]): int(s["class_id"]) for s in summary.get("segments", [])}
    class_ids = np.full(n_points, -1, dtype=np.int8)
    for sid, cid in seg_to_class.items():
        class_ids[instance_ids == sid] = cid
    return class_ids, instance_ids
