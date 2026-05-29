"""Per-render-run provenance meta (renders/<run>/meta.json), scan-schema v1.3 §4.3."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from scenes.frame import Frame, frame_from_dict

META_NAME = "meta.json"   # sibling to manifest.json


def write_render_meta(run_dir: Path, *, run_id: str, generated_from: dict,
                      frame: Frame, intrinsics: dict, generated_at: str = "",
                      n_frames: Optional[int] = None) -> Path:
    doc = {
        "run_id": run_id,
        "generated_from": generated_from,
        "frame": frame.to_dict(),
        "intrinsics": intrinsics,
        "generated_at": generated_at,
    }
    if n_frames is not None:
        doc["n_frames"] = n_frames
    path = Path(run_dir) / META_NAME
    path.write_text(json.dumps(doc, indent=2))
    return path


def read_render_meta(run_dir: Path) -> Optional[dict]:
    path = Path(run_dir) / META_NAME
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    if "frame" in doc:
        doc["frame"] = frame_from_dict(doc["frame"])
    return doc
