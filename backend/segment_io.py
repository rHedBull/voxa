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
        if instance_ids.shape != (n_points,):
            return None
        seg_to_class = {
            int(s["id"]): int(s["class_id"])
            for s in summary.get("segments", [])
        }
    except (OSError, ValueError, json.JSONDecodeError,
            AttributeError, KeyError, TypeError):
        return None
    class_ids = np.full(n_points, -1, dtype=np.int8)
    for sid, cid in seg_to_class.items():
        class_ids[instance_ids == sid] = cid
    return class_ids, instance_ids


_TS_RE = re.compile(r"^\d{8}_\d{6}$")


def _validate_invariants(class_ids: np.ndarray, instance_ids: np.ndarray) -> None:
    """SCHEMA invariants 3 & 4. Raises ValueError on violation."""
    cls_unl = class_ids == -1
    inst_unl = instance_ids == -1
    if not np.array_equal(cls_unl, inst_unl):
        n_bad = int(np.sum(cls_unl != inst_unl))
        raise ValueError(
            f"invariant 3: class_ids[i]==-1 ⟺ instance_ids[i]==-1 violated at {n_bad} points",
        )
    labeled = ~cls_unl
    if labeled.any():
        ii = instance_ids[labeled]
        ci = class_ids[labeled]
        order = np.argsort(ii, kind="stable")
        ii_s, ci_s = ii[order], ci[order]
        boundaries = np.concatenate(([True], ii_s[1:] != ii_s[:-1]))
        first_cls = ci_s[boundaries]
        group_idx = np.cumsum(boundaries) - 1
        if not np.array_equal(ci_s, first_cls[group_idx]):
            raise ValueError(
                "invariant 4: per-segment class consistency violated",
            )


def _build_segment_metadata(
    class_ids: np.ndarray, instance_ids: np.ndarray,
    positions: Optional[np.ndarray] = None,
) -> dict:
    n_points = int(instance_ids.shape[0])
    labeled = instance_ids >= 0
    n_labeled = int(labeled.sum())
    segments: list[dict] = []
    for sid in np.unique(instance_ids[labeled]):
        sid_i = int(sid)
        m = instance_ids == sid_i
        cid = int(class_ids[m][0])
        entry: dict = {
            "gt_id": sid_i,
            "class_id": cid,
            "n_points": int(m.sum()),
        }
        if positions is not None:
            sub = positions[m]
            mn = sub.min(axis=0); mx = sub.max(axis=0)
            entry["bbox"] = [float(mn[0]), float(mn[1]), float(mn[2]),
                              float(mx[0]), float(mx[1]), float(mx[2])]
        segments.append(entry)
    return {
        "n_points": n_points,
        "n_gt_segments": len(segments),
        "n_labeled_points": n_labeled,
        "class_map_version": 1,
        "segments": segments,
    }


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_labels(
    scan_dir: Path,
    class_ids: np.ndarray,
    instance_ids: np.ndarray,
    *,
    positions: Optional[np.ndarray] = None,
    write_history: bool = True,
    history_keep: int = 10,
) -> None:
    """Validate, snapshot existing labels, then write gt_*.npy + metadata.

    Writes are sequential (3 files); not atomic across files. A history
    snapshot is taken from the prior on-disk labels before overwrite.
    """
    _validate_invariants(class_ids, instance_ids)

    labels_dir = scan_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    if write_history and (labels_dir / "gt_class_ids.npy").exists():
        snap_dir = scan_dir / "annotation_history" / _utc_timestamp()
        snap_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("gt_class_ids.npy", "gt_segment_ids.npy",
                      "gt_segment_metadata.json"):
            src = labels_dir / fname
            if src.exists():
                shutil.copy2(src, snap_dir / fname)
        prune_history(scan_dir / "annotation_history", keep=history_keep)
    elif write_history:
        (scan_dir / "annotation_history" / _utc_timestamp()).mkdir(parents=True, exist_ok=True)
        prune_history(scan_dir / "annotation_history", keep=history_keep)

    np.save(labels_dir / "gt_class_ids.npy", class_ids.astype(np.int32))
    np.save(labels_dir / "gt_segment_ids.npy", instance_ids.astype(np.int32))
    meta = _build_segment_metadata(class_ids, instance_ids, positions)
    (labels_dir / "gt_segment_metadata.json").write_text(json.dumps(meta, indent=2))


def prune_history(history_dir: Path, *, keep: int = 10) -> None:
    """Keep the `keep` most-recent timestamp-named subdirs; leave others alone."""
    if not history_dir.exists():
        return
    timestamped = [p for p in history_dir.iterdir() if p.is_dir() and _TS_RE.match(p.name)]
    if len(timestamped) <= keep:
        return
    timestamped.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in timestamped[keep:]:
        shutil.rmtree(p)
