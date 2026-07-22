"""Read-only loader for instances_gt.json, for the save-time eval-invariant
gate (segment_io.save_labels). Deliberately NOT the same code path as the
route (backend/routes/compare.py's GET/PUT /api/annotations) — this must not
depend on FastAPI, and segment_io is pure I/O by convention."""
from __future__ import annotations

import json
from pathlib import Path


def load_instances_for_invariants(session_dir: Path) -> dict[int, dict]:
    """{segId: {"class_id": cls-or-None, "confirmed": bool}} for every
    kind:'pointset' instance with a segId. Returns {} if instances_gt.json is
    absent (a session with nothing confirmed yet, or a pre-instance-doc
    session) — callers must not treat this as an invariant violation on its
    own; an empty doc with non-empty gt_segment_ids.npy IS still a violation,
    caught by eval-invariant 3 itself."""
    p = Path(session_dir) / "instances_gt.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    result: dict[int, dict] = {}
    for inst in raw.get("instances", []):
        seg_id = inst.get("segId")
        if inst.get("kind") != "pointset" or seg_id is None:
            continue
        result[int(seg_id)] = {
            "class_id": inst.get("cls"),
            "confirmed": bool(inst.get("confirmed", False)),
        }
    return result
