"""One-shot migration: voxa scan-schema v1.3 → v2.

v1.3 layout:
  labels/gt_class_ids.npy, gt_segment_ids.npy, gt_segment_metadata.json
  session/current.json, working_class_ids.npy, working_segment_ids.npy  (optional)
  prelabel/ransac_instance_ids.npy + ransac_segment_summary.json        (optional)
  annotation_history/<YYYYMMDD_HHMMSS>/...                              (optional)
  meta.json (schema_version: "1.3" or absent)

v2 layout:
  prelabel/ransac/{instance_ids.npy, segment_summary.json, meta.json}
  sessions/legacy/{session.json, working_class_ids.npy, working_segment_ids.npy}
  sessions/legacy/output/{gt_class_ids.npy, gt_segment_ids.npy, gt_segment_metadata.json}
  sessions/legacy/history/<YYYYMMDD_HHMMSS>/...
  meta.json (schema_version: "2.0")

Usage:
  python scripts/migrate_scan_v2.py [--dry-run] [--scan NAME ...] LIDAR_ROOT

NOTE: the per-scan migration is not transactional — a crash mid-migration can
leave a scan partially migrated (e.g. sessions/ present but meta.json still at
1.3), which the next run REFUSES rather than guessing. Recovery is manual:
inspect the scan dir and finish or revert the listed steps by hand.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Add backend to sys.path — same pattern as scripts/preseg/presegment.py:40
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from labeling.segment_io import (atomic_write_json, atomic_write_npy,
                                 utc_now_iso)
from scan_schema.fingerprint import array_fingerprint
from preseg.preseg_store import register_preseg
from scenes.lidar_io import z_up_to_y_up_xyz
from scenes.scan_meta import is_z_up_from_meta
from scenes.point_cloud import load_ply
from scan_schema.layout import ScanLayout


def _read_ply_positions(ply_path: Path) -> np.ndarray:
    """Return the (N, 3) float32 xyz array EXACTLY as the loader produces it.

    Bit-exactness matters: float32 ``mean`` results differ in the last bits
    depending on the array's memory layout (load_ply returns a non-contiguous
    ``vstack().T`` view), and the recenter step bakes that mean into every
    coordinate. The source-fingerprint pin must match what /api/load computes,
    so we go through the loader's own reader rather than re-stacking columns.
    Colors/labels are loaded and discarded — wasteful for huge clouds, but
    reading only xyz would mean replicating load_ply's stacking, which risks
    exactly the divergence this function exists to prevent.
    """
    pc, _ = load_ply(ply_path)
    return pc.points


def _display_positions(pts: np.ndarray, z_up: bool) -> np.ndarray:
    """Apply the same load-time transforms used in routes/load.py so
    source_fingerprint matches what /api/load computes.

    1. z_up → y_up rotation via the loader's own ``z_up_to_y_up_xyz`` (the
       single source of truth — never replicate it).
    2. Recenter exactly as _recenter in app/core.py:114 does: subtract the
       mean centroid, but ONLY when the centroid magnitude itself exceeds
       1e3 (the trigger is on the mean, not on individual coords).

    Every op runs on the loader's own array object (see _read_ply_positions)
    so summation order — and therefore the float32 mean — is bit-identical
    to what /api/load computes.
    """
    out = z_up_to_y_up_xyz(pts) if z_up else pts
    # _recenter: subtract the mean centroid iff max(|mean|) >= 1e3
    center = out.mean(axis=0)
    if float(np.max(np.abs(center))) >= 1e3:
        out = (out - center).astype(np.float32)
    return out.astype(np.float32)


def _mtime_iso(path: Path) -> str:
    return datetime.fromtimestamp(
        path.stat().st_mtime, tz=timezone.utc
    ).isoformat(timespec="seconds")


def _check_shape(arr_path: Path, n_points: int, label: str) -> str | None:
    """Return an error string if arr_path exists and shape != (n_points,), else None."""
    if not arr_path.exists():
        return None
    try:
        arr = np.load(arr_path)
    except Exception as e:
        return f"{label}: could not load: {e}"
    if arr.shape != (n_points,):
        return f"{label}: shape {arr.shape} != ({n_points},)"
    return None


def _migrate_scan(scan_dir: Path, *, dry_run: bool) -> tuple[bool, str]:
    """Migrate one scan.  Returns (ok, message) where ok=True means success or skip.

    ok=True  → "already v2" (skip) or migrated
    ok=False → refused (caller sets exit code 1)
    """
    name = scan_dir.name
    lay = ScanLayout(scan_dir)

    # Step 1: idempotency
    meta_path = scan_dir / "meta.json"
    meta: dict = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text())
        except Exception:
            meta = {}
    if str(meta.get("schema_version", "")).startswith("2"):
        return True, f"SKIP {name} (already v2)"

    # ---- Refusal checks (all before any mutation) ----
    sessions_dir = scan_dir / "sessions"
    prelabel_dir = scan_dir / "prelabel"
    ply_path = scan_dir / "source" / "scan.ply"

    # sessions/ already exists
    if sessions_dir.exists():
        return False, f"REFUSED {name}: unexpected sessions/ dir on a v1.3 scan"

    # prelabel/ exists but has unexpected contents
    if prelabel_dir.exists():
        allowed = {"ransac_instance_ids.npy", "ransac_segment_summary.json"}
        actual = set()
        for item in prelabel_dir.iterdir():
            if item.is_dir():
                return False, (f"REFUSED {name}: prelabel/ contains a subdirectory "
                               f"'{item.name}' — not a plain v1.3 prelabel")
            actual.add(item.name)
        unexpected = actual - allowed
        if unexpected:
            return False, (f"REFUSED {name}: prelabel/ contains unexpected files: "
                           f"{sorted(unexpected)}")

    # source/scan.ply missing
    if not ply_path.exists():
        return False, f"REFUSED {name}: source/scan.ply missing"

    # Load positions (needed for fingerprint and shape checks)
    try:
        positions = _read_ply_positions(ply_path)
    except Exception as e:
        return False, f"REFUSED {name}: could not read source/scan.ply: {e}"
    n_points = len(positions)

    # Shape checks on all per-point arrays
    labels_dir = scan_dir / "labels"
    session_dir = scan_dir / "session"
    shape_checks = [
        (labels_dir / "gt_class_ids.npy", "labels/gt_class_ids.npy"),
        (labels_dir / "gt_segment_ids.npy", "labels/gt_segment_ids.npy"),
        (session_dir / "working_class_ids.npy", "session/working_class_ids.npy"),
        (session_dir / "working_segment_ids.npy", "session/working_segment_ids.npy"),
    ]
    if prelabel_dir.exists():
        shape_checks.append((
            prelabel_dir / "ransac_instance_ids.npy",
            "prelabel/ransac_instance_ids.npy",
        ))
    for arr_path, label in shape_checks:
        err = _check_shape(arr_path, n_points, label)
        if err:
            return False, f"REFUSED {name}: {err}"

    # ---- Plan (printed for --dry-run, also determines actions) ----
    has_ransac = (prelabel_dir / "ransac_instance_ids.npy").exists()
    has_labels = (labels_dir / "gt_class_ids.npy").exists()
    has_session = session_dir.exists() and (session_dir / "current.json").exists()
    history_dir = scan_dir / "annotation_history"
    has_history = history_dir.exists() and any(history_dir.iterdir())

    plan_lines = []
    if has_ransac:
        plan_lines.append(
            "  PRESEG: register_preseg ransac from prelabel/ransac_*.* → prelabel/ransac/"
        )
        plan_lines.append("  PRESEG: delete prelabel/ransac_instance_ids.npy + ransac_segment_summary.json")
    if has_labels or has_session:
        plan_lines.append("  SESSION: create sessions/legacy/")
        if has_session:
            plan_lines.append("  SESSION: copy working arrays from session/working_*.npy")
        elif has_labels:
            plan_lines.append("  SESSION: seed working arrays from gt_class_ids/gt_segment_ids (int8 cast)")
        else:
            plan_lines.append("  SESSION: seed working arrays as all -1")
        if has_labels:
            plan_lines.append("  SESSION: MOVE labels/ → sessions/legacy/output/")
        plan_lines.append("  SESSION: write sessions/legacy/session.json")
    if has_history:
        plan_lines.append("  HISTORY: MOVE annotation_history/ entries → sessions/legacy/history/")
    plan_lines.append("  CLEANUP: remove empty labels/, session/, annotation_history/ dirs")
    plan_lines.append(f"  META: set schema_version to \"2.0\" in meta.json")

    if dry_run:
        print(f"DRY-RUN plan for {name}:")
        for line in plan_lines:
            print(line)
        return True, f"DRY-RUN {name} (no changes made)"

    # ---- Step 3: Preseg ----
    preseg_id: str | None = None
    preseg_fingerprint: str | None = None
    if has_ransac:
        inst_path = prelabel_dir / "ransac_instance_ids.npy"
        summary_path = prelabel_dir / "ransac_segment_summary.json"
        inst = np.load(inst_path).astype(np.int32)
        summary = json.loads(summary_path.read_text())
        info = register_preseg(lay, "ransac", inst,
                               summary=summary, generator="ransac", params={})
        preseg_id = "ransac"
        preseg_fingerprint = info.fingerprint
        inst_path.unlink()
        summary_path.unlink()
        # remove prelabel/ if now empty
        try:
            prelabel_dir.rmdir()
        except OSError:
            pass  # not empty (the ransac/ subdir is there now)

    # ---- Step 4: Legacy session ----
    if has_labels or has_session:
        legacy_sp = lay.session("legacy")
        legacy_sp.dir.mkdir(parents=True, exist_ok=True)
        legacy_sp.output_dir.mkdir(parents=True, exist_ok=True)

        # Read old session state (if present)
        old_current: dict = {}
        if has_session:
            try:
                old_current = json.loads((session_dir / "current.json").read_text())
            except Exception:
                old_current = {}

        # Working arrays: session/working_* if readable, else GT cast, else -1
        def _try_load(p: Path, dtype) -> np.ndarray | None:
            if not p.exists():
                return None
            try:
                return np.load(p).astype(dtype)
            except Exception:
                return None

        working_cls: np.ndarray | None = None
        working_seg: np.ndarray | None = None
        if has_session:
            working_cls = _try_load(session_dir / "working_class_ids.npy", np.int8)
            working_seg = _try_load(session_dir / "working_segment_ids.npy", np.int32)

        if working_cls is None or working_seg is None:
            if has_labels:
                gt_cls = _try_load(labels_dir / "gt_class_ids.npy", np.int8)
                gt_seg = _try_load(labels_dir / "gt_segment_ids.npy", np.int32)
                working_cls = gt_cls if working_cls is None else working_cls
                working_seg = gt_seg if working_seg is None else working_seg
            if working_cls is None:
                working_cls = np.full(n_points, -1, dtype=np.int8)
            if working_seg is None:
                working_seg = np.full(n_points, -1, dtype=np.int32)

        # Write working arrays
        atomic_write_npy(legacy_sp.working_class_ids, working_cls.astype(np.int8, copy=False))
        atomic_write_npy(legacy_sp.working_segment_ids, working_seg.astype(np.int32, copy=False))

        # MOVE labels/ → sessions/legacy/output/
        if has_labels:
            for fname in ("gt_class_ids.npy", "gt_segment_ids.npy", "gt_segment_metadata.json"):
                src_p = labels_dir / fname
                if src_p.exists():
                    shutil.move(str(src_p), str(legacy_sp.output_dir / fname))

        # Compute source_fingerprint (must match what /api/load computes)
        z_up = is_z_up_from_meta(meta)
        display_pts = _display_positions(positions, z_up)
        source_fingerprint = array_fingerprint(display_pts)

        # Timestamp from labels mtime, fallback to session/current.json, fallback to now
        ts_source: Path | None = None
        if (legacy_sp.output_dir / "gt_class_ids.npy").exists():
            ts_source = legacy_sp.output_dir / "gt_class_ids.npy"
        elif (session_dir / "current.json").exists():
            ts_source = session_dir / "current.json"
        ts = _mtime_iso(ts_source) if ts_source else utc_now_iso()

        hidden_inst_ids = old_current.get("hidden_inst_ids", [])
        dirty = bool(old_current.get("dirty", False))

        # Synthesize session.json explicitly (not via save_session_aux; we
        # want mtime-derived timestamps, not "now" for created_at/saved_at).
        session_payload = {
            "schema_version": 2,
            "preseg_id": preseg_id,
            "preseg_fingerprint": preseg_fingerprint,
            "source_fingerprint": source_fingerprint,
            "hidden_inst_ids": hidden_inst_ids,
            "dirty": dirty,
            "name": "legacy",
            "created_at": ts,
            "saved_at": ts,
        }
        atomic_write_json(legacy_sp.session_json, session_payload)

    # ---- Step 5: History ----
    if has_history:
        legacy_history = lay.session("legacy").history_dir
        legacy_history.mkdir(parents=True, exist_ok=True)
        for entry in sorted(history_dir.iterdir()):
            shutil.move(str(entry), str(legacy_history / entry.name))

    # ---- Step 6: Cleanup ----
    # labels/ and session/ contents have been fully migrated; remove the old dirs.
    # annotation_history/ entries have been moved; remove the now-empty dir.
    for d in (labels_dir, session_dir, history_dir):
        if d.exists():
            shutil.rmtree(d)

    # Update meta.json
    meta["schema_version"] = "2.0"
    atomic_write_json(meta_path, meta)

    return True, f"MIGRATED {name}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Migrate voxa annotated scans from schema v1.3 → v2."
    )
    parser.add_argument("lidar_root", metavar="LIDAR_ROOT", type=Path)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the plan per scan; change nothing on disk.")
    parser.add_argument("--scan", metavar="NAME", nargs="+",
                        help="Only migrate these named scans (default: all).")
    args = parser.parse_args()

    lidar_root = args.lidar_root.resolve()
    annotated_root = lidar_root / "annotated"
    if not annotated_root.is_dir():
        print(f"ERROR: {annotated_root} is not a directory", file=sys.stderr)
        return 1

    if args.scan:
        scan_dirs = [annotated_root / name for name in args.scan]
        for sd in scan_dirs:
            if not sd.is_dir():
                print(f"ERROR: scan dir not found: {sd}", file=sys.stderr)
                return 1
    else:
        scan_dirs = sorted(p for p in annotated_root.iterdir() if p.is_dir())

    any_refused = False
    for sd in scan_dirs:
        ok, msg = _migrate_scan(sd, dry_run=args.dry_run)
        print(msg)
        if not ok:
            any_refused = True

    return 1 if any_refused else 0


if __name__ == "__main__":
    sys.exit(main())
