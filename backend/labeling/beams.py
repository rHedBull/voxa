"""Per-session Beam-tool structure persistence.

`structure.json` stores the node/edge graph (active, resumable) plus committed
beams (baked geometry for the read-only committed layer). Point extraction
needs no beam-specific backend code: the frontend converts each beam to an
OBB and applies it through /api/segment/apply-shape (see the beam spec's
as-built notes). Mirrors centerline.py's persistence half.
"""
from __future__ import annotations

import json
from pathlib import Path

STRUCTURE_FILENAME = "structure.json"

_KEYS = {"nodes", "edges", "committed_beams"}


def load_structure(session_dir: Path) -> dict:
    f = Path(session_dir) / STRUCTURE_FILENAME
    if not f.exists():
        return {"nodes": [], "edges": [], "committed_beams": []}
    data = json.loads(f.read_text())
    missing = _KEYS - data.keys()
    if missing:
        raise ValueError(
            f"malformed structure.json in {session_dir}: missing {sorted(missing)}"
        )
    return data


def save_structure(session_dir: Path, doc: dict) -> dict:
    from labeling.segment_io import atomic_write_json
    atomic_write_json(Path(session_dir) / STRUCTURE_FILENAME, doc)
    return doc
