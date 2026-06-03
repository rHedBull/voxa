"""SCHEMA-aware reader for prelabel/, writer for labels/, history pruning.

Pure I/O. No FastAPI, no in-memory state. Loaders return aligned arrays;
writers validate invariants and recompute gt_segment_metadata.json from the
arrays before flushing to disk.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from scenes.scan_layout import ScanLayout


def compute_fingerprint(arr: np.ndarray) -> str:
    """Content-addressed sha256 of a numpy array's bytes. Stable across
    save/load (numpy preserves byte layout for fixed dtypes)."""
    h = hashlib.sha256()
    h.update(bytes(arr.dtype.str, "ascii"))
    h.update(b":")
    h.update(bytes(str(arr.shape), "ascii"))
    h.update(b":")
    h.update(np.ascontiguousarray(arr).tobytes())
    return f"sha256:{h.hexdigest()}"


def atomic_write_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def load_prelabel(
    scan_dir: Path, n_points: int
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Read prelabel/ if present. Returns (class_ids int8, instance_ids int32)
    or None when no prelabel exists / arrays are malformed."""
    lay = ScanLayout(scan_dir)
    inst_path = lay.ransac_instance_ids
    summary_path = lay.ransac_segment_summary
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


def _load_class_registry(scan_dir: Path) -> Optional[dict]:
    """Read `<lidar_root>/classes.json` from scan_dir's grandparent.

    Returns ``{"version": int, "by_id": {id: name}}`` or ``None`` if the
    registry isn't found / malformed. Tests using throwaway tmp dirs
    naturally hit the None branch, so callers must treat schema-aware
    enrichment + validation as optional.
    """
    candidate = ScanLayout(scan_dir).classes_json
    if not candidate.exists():
        return None
    try:
        raw = json.loads(candidate.read_text())
        version = int(raw["version"])
        by_id = {int(c["id"]): str(c["name"]) for c in raw["classes"]}
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
        return None
    return {"version": version, "by_id": by_id}


def _read_meta_class_map_version(scan_dir: Path) -> Optional[int]:
    meta_path = ScanLayout(scan_dir).meta_json
    if not meta_path.exists():
        return None
    try:
        raw = json.loads(meta_path.read_text())
        return int(raw["class_map_version"])
    except (OSError, ValueError, json.JSONDecodeError, KeyError, TypeError):
        return None


def _validate_invariants(
    class_ids: np.ndarray, instance_ids: np.ndarray,
    registry: Optional[dict] = None,
    meta_class_map_version: Optional[int] = None,
) -> None:
    """SCHEMA invariants 3, 4, and (when registry is supplied) 5 & 6."""
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
    if registry is not None and labeled.any():
        unknown = sorted({int(c) for c in np.unique(class_ids[labeled]).tolist()
                          if int(c) not in registry["by_id"]})
        if unknown:
            raise ValueError(
                f"invariant 5: class IDs {unknown} not in classes.json "
                f"(known: {sorted(registry['by_id'])})",
            )
    if (registry is not None and meta_class_map_version is not None
            and meta_class_map_version != registry["version"]):
        raise ValueError(
            f"invariant 6: meta.json::class_map_version "
            f"({meta_class_map_version}) != classes.json::version "
            f"({registry['version']})",
        )


def _build_segment_metadata(
    class_ids: np.ndarray, instance_ids: np.ndarray,
    positions: Optional[np.ndarray] = None,
    registry: Optional[dict] = None,
) -> dict:
    n_points = int(instance_ids.shape[0])
    labeled = instance_ids >= 0
    n_labeled = int(labeled.sum())
    by_id = registry["by_id"] if registry is not None else {}
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
        if cid in by_id:
            entry["label"] = by_id[cid]
        if positions is not None:
            sub = positions[m]
            mn = sub.min(axis=0); mx = sub.max(axis=0)
            entry["bbox"] = [float(mn[0]), float(mn[1]), float(mn[2]),
                              float(mx[0]), float(mx[1]), float(mx[2])]
        segments.append(entry)
    version = registry["version"] if registry is not None else 1
    return {
        "n_points": n_points,
        "n_gt_segments": len(segments),
        "n_labeled_points": n_labeled,
        "class_map_version": version,
        "segments": segments,
    }


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def save_labels(
    scan_dir: Path,
    session_id: str,
    class_ids: np.ndarray,
    instance_ids: np.ndarray,
    *,
    positions: Optional[np.ndarray] = None,
    write_history: bool = True,
    history_keep: int = 10,
    preseg_fingerprint: Optional[str] = None,
    source_fingerprint: Optional[str] = None,
) -> None:
    """Validate, snapshot existing labels, then write gt_*.npy + metadata.

    Writes into sessions/<session_id>/output/ under scan_dir. Writes are
    sequential (3 files); not atomic across files. A history snapshot is
    taken from the prior on-disk labels before overwrite.
    """
    registry = _load_class_registry(scan_dir)
    meta_version = _read_meta_class_map_version(scan_dir)
    _validate_invariants(class_ids, instance_ids,
                         registry=registry,
                         meta_class_map_version=meta_version)

    sp = ScanLayout(scan_dir).session(session_id)
    sp.output_dir.mkdir(parents=True, exist_ok=True)
    gt_files = (sp.output_gt_class_ids, sp.output_gt_segment_ids, sp.output_gt_segment_metadata)

    if write_history and sp.output_gt_class_ids.exists():
        snap_dir = sp.history_dir / _utc_timestamp()
        snap_dir.mkdir(parents=True, exist_ok=True)
        for src in gt_files:
            if src.exists():
                shutil.copy2(src, snap_dir / src.name)
        prune_history(sp.history_dir, keep=history_keep)
    elif write_history:
        (sp.history_dir / _utc_timestamp()).mkdir(parents=True, exist_ok=True)
        prune_history(sp.history_dir, keep=history_keep)

    np.save(sp.output_gt_class_ids, class_ids.astype(np.int32))
    np.save(sp.output_gt_segment_ids, instance_ids.astype(np.int32))
    meta = _build_segment_metadata(class_ids, instance_ids, positions,
                                   registry=registry)
    if preseg_fingerprint is not None:
        meta["preseg_fingerprint"] = preseg_fingerprint
    if source_fingerprint is not None:
        meta["source_fingerprint"] = source_fingerprint
    sp.output_gt_segment_metadata.write_text(json.dumps(meta, indent=2))


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


SESSION_SCHEMA_VERSION = 2


def filter_tiny_segments(class_ids: np.ndarray, instance_ids: np.ndarray,
                         min_points: int) -> tuple[np.ndarray, np.ndarray]:
    """Reset (class_id, instance_id) to (-1, -1) for any point belonging to
    an instance with fewer than ``min_points`` points. Returns fresh copies."""
    inst = np.asarray(instance_ids, dtype=np.int32)
    cls = np.asarray(class_ids, dtype=np.int8)
    if inst.size == 0 or min_points <= 1:
        return cls.copy(), inst.copy()
    labeled = inst >= 0
    if not labeled.any():
        return cls.copy(), inst.copy()
    ids, counts = np.unique(inst[labeled], return_counts=True)
    drop_ids = ids[counts < int(min_points)]
    if drop_ids.size == 0:
        return cls.copy(), inst.copy()
    drop_mask = np.isin(inst, drop_ids)
    new_cls = cls.copy(); new_inst = inst.copy()
    new_cls[drop_mask] = -1; new_inst[drop_mask] = -1
    return new_cls, new_inst


def save_session_aux(
    session_dir: Path,
    aux: dict,
    *,
    class_ids: Optional[np.ndarray] = None,
    instance_ids: Optional[np.ndarray] = None,
) -> dict:
    """Atomically persist editor session state. Returns the payload as
    written (callers use its ``saved_at`` stamp).

    Order: working_*.npy first, then session.json (commit pointer). On a
    crash between the npy renames and session.json rename, the next reload
    sees the previous-consistent session.json and ignores any half-updated
    working_*.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    if class_ids is not None:
        atomic_write_npy(session_dir / "working_class_ids.npy",
                         class_ids.astype(np.int8, copy=False))
    if instance_ids is not None:
        atomic_write_npy(session_dir / "working_segment_ids.npy",
                         instance_ids.astype(np.int32, copy=False))
    payload = dict(aux)
    payload.setdefault("schema_version", SESSION_SCHEMA_VERSION)
    payload["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    atomic_write_json(session_dir / "session.json", payload)
    return payload


def load_session_aux(session_dir: Path) -> Optional[dict]:
    """Read session.json or return None if absent/unreadable."""
    p = session_dir / "session.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_working_arrays(
    session_dir: Path, n_points: int,
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return (class_ids int8, instance_ids int32) iff session.json exists
    AND both working files are present AND shapes match n_points."""
    if load_session_aux(session_dir) is None:
        return None
    cp = session_dir / "working_class_ids.npy"
    ip = session_dir / "working_segment_ids.npy"
    if not (cp.exists() and ip.exists()):
        return None
    try:
        ci = np.load(cp).astype(np.int8, copy=False)
        ii = np.load(ip).astype(np.int32, copy=False)
    except (OSError, ValueError):
        return None
    if ci.shape != (n_points,) or ii.shape != (n_points,):
        return None
    return ci, ii
