"""Read-only loader for instances_gt.json, for the save-time eval-invariant
gate (segment_io.save_labels). Deliberately NOT the same code path as the
route (backend/routes/compare.py's GET/PUT /api/annotations), and deliberately
does NOT import app.schemas.Cuboid — backend/labeling/* is meant to stay
usable/testable independent of the backend/app/* app layer (the same spirit
as segment_io's "pure I/O, no in-memory state"), so this module hand-parses
the row shape it actually needs (`kind`, `segId`, `cls`, `confirmed`) instead
of importing the schema.

That means Cuboid and this loader's field expectations are NOT pinned
together by any shared import — if Cuboid's field names for these four keys
ever change, this loader silently keeps reading the old names and degrades
to returning {} / wrong values rather than erroring. See
test_instances_doc.py::test_expected_keys_are_cuboid_fields, which locks the
two in sync the same way test_class_config.py pins classes.yaml/classes.json."""
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
        cls = inst.get("cls")
        result[int(seg_id)] = {
            "class_id": cls,
            # A class-less blob (artifact/review) is a session-only review handle,
            # never a confirmed GT instance (labeler guide; eval-invariant 9). Its
            # in-session `confirmed` drives hide/protect only; strip it here.
            "confirmed": bool(inst.get("confirmed", False)) and cls is not None,
        }
    return result
