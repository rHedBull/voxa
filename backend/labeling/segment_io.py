"""SCHEMA-aware session-output writer + working-state I/O + history pruning.

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

from scan_schema.layout import ScanLayout
from scan_schema.invariants import validate_invariants
# Crash-safe writers now live in the schema package; re-exported here so the
# existing labeling/preseg/migration callers import them unchanged.
from scan_schema.storage import atomic_write_npy, atomic_write_json  # noqa: F401


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


def review_blob_summary(categories: np.ndarray, instance_ids: np.ndarray) -> list[dict]:
    """[{instance_id, n_points}] for every review blob, from the WORKING
    arrays (before the save-path strip of class-less instances). Recorded in
    gt_segment_metadata.json so the blob ids survive that strip."""
    from labeling.categories import CATEGORY_EXCLUDED_REVIEW

    cats = np.asarray(categories)
    inst = np.asarray(instance_ids)
    mask = (cats == CATEGORY_EXCLUDED_REVIEW) & (inst >= 0)
    if not mask.any():
        return []
    ids, counts = np.unique(inst[mask], return_counts=True)
    return [{"instance_id": int(i), "n_points": int(c)}
            for i, c in zip(ids.tolist(), counts.tolist())]


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")


def utc_now_iso() -> str:
    """UTC timestamp for session/preseg metadata (ISO, second resolution).
    Single home — session_store, preseg_store and the migration script all
    stamp with this."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


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
    categories: Optional[np.ndarray] = None,
    review_blobs: Optional[list[dict]] = None,
) -> None:
    """Validate, snapshot existing labels, then write gt_*.npy + metadata.

    Writes into sessions/<session_id>/output/ under scan_dir. Writes are
    sequential; not atomic across files. A history snapshot is taken from the
    prior on-disk labels before overwrite.

    `categories` (phase 2) additionally writes gt_point_category.npy and the
    metadata category histogram + review-blob table; component ids are derived
    from `positions` + `instance_ids` and written to
    gt_point_component_ids.npy whenever positions are supplied.
    """
    registry = _load_class_registry(scan_dir)
    meta_version = _read_meta_class_map_version(scan_dir)
    validate_invariants(class_ids, instance_ids,
                        registry=registry,
                        meta_class_map_version=meta_version)

    sp = ScanLayout(scan_dir).session(session_id)
    sp.output_dir.mkdir(parents=True, exist_ok=True)
    category_path = sp.output_dir / "gt_point_category.npy"
    component_path = sp.output_dir / "gt_point_component_ids.npy"
    gt_files = (sp.output_gt_class_ids, sp.output_gt_segment_ids,
                sp.output_gt_segment_metadata, category_path, component_path)

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
    if categories is not None:
        from labeling.categories import category_histogram
        np.save(category_path, np.asarray(categories).astype(np.int8))
        meta["categories"] = category_histogram(categories)
        meta["review_blobs"] = list(review_blobs or [])
    if positions is not None:
        from labeling.components import LINK_RADIUS_M, component_ids
        np.save(component_path, component_ids(positions, instance_ids))
        meta["component_link_radius_m"] = LINK_RADIUS_M
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
    sam_ids: Optional[np.ndarray] = None,
    categories: Optional[np.ndarray] = None,
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
    if sam_ids is not None:
        atomic_write_npy(session_dir / "working_sam_ids.npy",
                         sam_ids.astype(np.int32, copy=False))
    if categories is not None:
        atomic_write_npy(session_dir / "working_categories.npy",
                         categories.astype(np.int8, copy=False))
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


def load_sam_ids(session_dir: Path, n_points: int) -> Optional[np.ndarray]:
    """Return the SAM candidate layer (int32) if working_sam_ids.npy exists,
    or None if absent (a session with no SAM captures yet — caller defaults
    to all -1). Unlike load_working_arrays (which soft-fails to None and lets
    the caller decide, since class_ids/instance_ids are confirmed labels a
    silent loss of which would be catastrophic), sam_ids is a disposable
    candidate layer — but a shape-mismatched file still signals a bad/foreign
    data directory, so this loader raises directly rather than soft-failing."""
    p = session_dir / "working_sam_ids.npy"
    if not p.exists():
        return None
    arr = np.load(p).astype(np.int32, copy=False)
    if arr.shape != (n_points,):
        raise ValueError(f"working_sam_ids.npy shape {arr.shape} != ({n_points},)")
    return arr


def load_categories(session_dir: Path, n_points: int) -> Optional[np.ndarray]:
    """Return the point-category layer (int8) if working_categories.npy exists,
    or None for a session written before phase 2 (caller defaults to all
    `none`). Shape mismatch raises, same contract as load_sam_ids: a misshapen
    file signals a foreign/corrupt data dir, not an empty layer."""
    p = session_dir / "working_categories.npy"
    if not p.exists():
        return None
    arr = np.load(p).astype(np.int8, copy=False)
    if arr.shape != (n_points,):
        raise ValueError(f"working_categories.npy shape {arr.shape} != ({n_points},)")
    return arr


def sam_segments_to_list(sam_segments: dict[int, dict]) -> list[dict]:
    """{sam_seg_id: meta} -> [{id, **meta}, ...] sorted by id — the shared
    wire/file shape used by both sam_segments.json and GET /api/segment/state."""
    return [{"id": sid, **meta} for sid, meta in sorted(sam_segments.items())]


def save_sam_segments(session_dir: Path, sam_segments: dict[int, dict]) -> None:
    """Atomically persist the SAM candidate-segment summary
    (sessions/<id>/sam_segments.json). Point membership lives in
    working_sam_ids.npy; this file is metadata only, mirroring prelabel's
    segment_summary.json shape."""
    session_dir.mkdir(parents=True, exist_ok=True)
    payload = {"segments": sam_segments_to_list(sam_segments)}
    atomic_write_json(session_dir / "sam_segments.json", payload)


def load_sam_segments(session_dir: Path) -> dict[int, dict]:
    """Read sam_segments.json -> {sam_seg_id: {n_points, source, mask_score,
    created_at}}. Missing file -> empty dict (no SAM captures yet)."""
    p = session_dir / "sam_segments.json"
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {int(e["id"]): {k: v for k, v in e.items() if k != "id"}
            for e in raw.get("segments", [])}
