"""Migrate a pre-phase-2 session so it passes the phase-3 eval invariants.

Additive-only EXCEPT for legacy class-id-6 ('unknown') points, which are
rewritten to the phase-2 review-blob representation (category=excluded_review,
class=-1, instance stripped) because eval-invariant 4 rejects any frozen class
id — including 6 — with no grandfather path. Every other already-labeled
point (any class id other than 6, including the other frozen legacy ids
0/3/5/13, which stay readable-but-unassignable) is untouched byte-for-byte.

    .venv/bin/python scripts/migrate_eval_invariants.py <scan_dir> <session_id> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402

from labeling.categories import CATEGORY_NONE, CATEGORY_EXCLUDED_REVIEW, category_histogram  # noqa: E402
from scan_schema.layout import ScanLayout  # noqa: E402

LEGACY_UNKNOWN_CLASS_ID = 6


def migrate_session(session_dir: Path, n_points: int, *, dry_run: bool) -> dict:
    out = session_dir / "output"
    class_ids = np.load(out / "gt_class_ids.npy")
    instance_ids = np.load(out / "gt_segment_ids.npy")

    categories_path = out / "gt_point_category.npy"
    categories = (np.load(categories_path) if categories_path.exists()
                 else np.full(n_points, CATEGORY_NONE, dtype=np.int8))

    legacy = class_ids == LEGACY_UNKNOWN_CLASS_ID
    n_legacy = int(legacy.sum())

    # Per-instance point counts for the review_blobs metadata, computed from
    # the ORIGINAL (pre-mutation) instance_ids array in a single pass —
    # np.unique + return_counts, no re-load and no re-query of a mutated array.
    converted_ids: list[int] = []
    review_blob_counts: dict[int, int] = {}
    if n_legacy:
        ids, counts = np.unique(instance_ids[legacy], return_counts=True)
        converted_ids = sorted(int(i) for i in ids.tolist())
        review_blob_counts = {int(i): int(c) for i, c in zip(ids.tolist(), counts.tolist())}

    if n_legacy:
        categories = categories.copy()
        categories[legacy] = CATEGORY_EXCLUDED_REVIEW
        class_ids = class_ids.copy(); instance_ids = instance_ids.copy()
        class_ids[legacy] = -1
        instance_ids[legacy] = -1   # review blobs are class-less by construction

    component_path = out / "gt_point_component_ids.npy"
    if component_path.exists():
        comp_ids = np.load(component_path)
    else:
        # No positions available offline (source.ply not loaded here) — one
        # component per instance, matching "no fragment splitting is invented"
        comp_ids = np.where(instance_ids >= 0, 0, -1).astype(np.int16)

    meta_path = out / "gt_segment_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    # The pre-migration `segments` list may still carry an entry for each
    # converted instance (class_id: 6, the frozen legacy id) — that instance
    # no longer has any labeled points after migration (it's now a pure,
    # class-less review blob), so its stale `segments` entry must be dropped;
    # leaving it in place would both misrepresent the data (a "segment" with
    # zero remaining points) and reintroduce the very frozen-class-id that
    # eval-invariant 4 rejects, if anything ever re-validates the metadata
    # standalone.
    segments = [s for s in meta.get("segments", [])
                if int(s.get("gt_id", -1)) not in converted_ids]
    if "segments" in meta or segments:
        meta["segments"] = segments

    review_blobs = list(meta.get("review_blobs", []))
    for iid in converted_ids:
        review_blobs.append({"instance_id": iid, "n_points": review_blob_counts[iid]})
    meta["review_blobs"] = review_blobs
    meta["categories"] = category_histogram(categories)

    # instances_gt.json (frontend-owned, sibling of output/) carries its own
    # per-instance `cls`/`confirmed` — a converted instance's row there must
    # be brought in line with the arrays (now a class-less review blob) or
    # scan_schema.eval_invariants.check_instance_class_consistency /
    # check_confirmed_* reject the mismatch. segId is the join key (per
    # Cuboid.segId / labeling/instances_doc.py::load_instances_for_invariants).
    instances_gt_path = session_dir / "instances_gt.json"
    instances_gt = None
    if converted_ids and instances_gt_path.exists():
        instances_gt = json.loads(instances_gt_path.read_text())
        converted_set = set(converted_ids)
        for inst in instances_gt.get("instances", []):
            if inst.get("kind") != "pointset" or inst.get("segId") not in converted_set:
                continue
            inst["cls"] = None
            if inst.get("confirmed"):
                inst["confirmed"] = False

    result = {
        "n_legacy_converted": n_legacy,
        "converted_instance_ids": converted_ids,
    }
    if dry_run:
        return result

    np.save(out / "gt_class_ids.npy", class_ids)
    np.save(out / "gt_segment_ids.npy", instance_ids)
    np.save(categories_path, categories)
    np.save(component_path, comp_ids)
    meta_path.write_text(json.dumps(meta, indent=2))
    if instances_gt is not None:
        instances_gt_path.write_text(json.dumps(instances_gt, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", type=Path)
    ap.add_argument("session_id")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    meta = json.loads((a.scan_dir / "meta.json").read_text())
    n_points = int(meta["n_points"])
    session_dir = ScanLayout(a.scan_dir).session(a.session_id).dir
    result = migrate_session(session_dir, n_points, dry_run=a.dry_run)
    print(f"{'[dry-run] ' if a.dry_run else ''}{a.scan_dir.name}/{a.session_id}: "
          f"converted {result['n_legacy_converted']} legacy class-6 points "
          f"(instances {result['converted_instance_ids']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
