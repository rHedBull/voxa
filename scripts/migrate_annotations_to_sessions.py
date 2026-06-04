"""One-time migration: scene-global instance docs → per-session instances_gt.json.

Before session-scoped annotation routes, every labeling session of an annotated
scan shared one ``data/annotations/<tier>__<scan>/ground_truth.json`` — so the
right-panel instance list leaked across sessions. This script attributes each
instance in those global docs to the session that actually created it and moves
it into ``<scan>/sessions/<sid>/instances_gt.json``.

Attribution: a pointset instance's ``segId`` is a per-point instance id in
exactly one session's ``working_segment_ids.npy``. Candidates are sessions whose
working array contains the id; ties are broken by preferring the session where
the id is *promoted* (greater than the session's preseg id range — raw preseg
ids are not instance segIds unless promoted). Unattributable instances (and
cuboids, which carry no segId) go to the doc's majority session, else the
scan's last-worked session.

The global file is renamed to ``ground_truth.json.pre-session-migration`` (kept
as backup). Legacy data-dir scenes (no ``annotated__`` prefix) are untouched.

Run with voxa's .venv (numpy only):

    .venv/bin/python scripts/migrate_annotations_to_sessions.py            # dry-run
    .venv/bin/python scripts/migrate_annotations_to_sessions.py --apply
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))


def _session_id_sets(scan_dir: Path) -> dict[str, dict]:
    """Per session: unique working segment ids + the preseg id ceiling."""
    out = {}
    sessions_root = scan_dir / "sessions"
    if not sessions_root.is_dir():
        return out
    for sdir in sorted(p for p in sessions_root.iterdir() if p.is_dir()):
        wpath = sdir / "working_segment_ids.npy"
        if not wpath.exists():
            continue
        sj = {}
        try:
            sj = json.loads((sdir / "session.json").read_text())
        except (OSError, json.JSONDecodeError):
            pass
        preseg_max = -1
        preseg_id = sj.get("preseg_id")
        if preseg_id:
            inst_path = scan_dir / "prelabel" / preseg_id / "instance_ids.npy"
            if inst_path.exists():
                preseg_max = int(np.load(inst_path, mmap_mode="r").max())
        ids = np.unique(np.load(wpath, mmap_mode="r"))
        out[sdir.name] = {
            "ids": set(int(i) for i in ids[ids >= 0]),
            "preseg_max": preseg_max,
            "saved_at": sj.get("saved_at") or "",
        }
    return out


def _attribute(instances: list[dict], sess: dict[str, dict]) -> dict[str, list]:
    """Map session_id -> instances. Pointsets by segId; rest to majority."""
    by_session: dict[str, list] = {sid: [] for sid in sess}
    unplaced = []
    for inst in instances:
        seg_id = inst.get("segId")
        if seg_id is None:
            unplaced.append(inst)
            continue
        candidates = [sid for sid, s in sess.items() if int(seg_id) in s["ids"]]
        if len(candidates) > 1:
            promoted = [sid for sid in candidates
                        if int(seg_id) > sess[sid]["preseg_max"]]
            if promoted:
                candidates = promoted
        if len(candidates) == 1:
            by_session[candidates[0]].append(inst)
        else:
            unplaced.append(inst)
    if unplaced:
        # Majority session of the placed instances, else last-worked.
        winner = max(by_session, key=lambda sid: len(by_session[sid])) \
            if any(by_session.values()) else \
            max(sess, key=lambda sid: sess[sid]["saved_at"])
        by_session[winner].extend(unplaced)
    return {sid: lst for sid, lst in by_session.items() if lst}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annot-dir", type=Path, default=ROOT / "data" / "annotations")
    ap.add_argument("--lidar-root", type=Path,
                    default=Path("/home/hendrik/coding/engine/data/lidar"))
    ap.add_argument("--apply", action="store_true",
                    help="write the migration (default: dry-run report)")
    args = ap.parse_args()

    if not args.annot_dir.is_dir():
        print(f"nothing to do: {args.annot_dir} does not exist")
        return 0

    n_moved = 0
    for doc_dir in sorted(args.annot_dir.iterdir()):
        gt = doc_dir / "ground_truth.json"
        if not doc_dir.name.startswith("annotated__") or not gt.exists():
            continue
        scan_dir = args.lidar_root / "annotated" / doc_dir.name.split("__", 1)[1]
        sess = _session_id_sets(scan_dir)
        if not sess:
            print(f"[skip] {doc_dir.name}: no sessions under {scan_dir}")
            continue
        doc = json.loads(gt.read_text())
        instances = doc.get("instances", [])
        if not instances:
            print(f"[skip] {doc_dir.name}: empty doc")
            continue
        placed = _attribute(instances, sess)
        print(f"[{doc_dir.name}] {len(instances)} instances → "
              + ", ".join(f"{sid}: {len(lst)}" for sid, lst in placed.items()))
        if not args.apply:
            continue
        for sid, lst in placed.items():
            target = scan_dir / "sessions" / sid / "instances_gt.json"
            existing = []
            if target.exists():
                existing = json.loads(target.read_text()).get("instances", [])
                have = {i.get("id") for i in existing}
                lst = [i for i in lst if i.get("id") not in have]
            target.write_text(json.dumps({
                "scene": doc.get("scene", ""), "kind": "gt",
                "instances": existing + lst, "meta": doc.get("meta", {}),
            }, indent=2))
            n_moved += len(lst)
        gt.rename(gt.with_suffix(".json.pre-session-migration"))
        print(f"  backed up global doc → {gt.name}.pre-session-migration")
    print(f"\n{'moved' if args.apply else 'would move'} {n_moved} instances"
          if args.apply else "\ndry-run complete — re-run with --apply")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
