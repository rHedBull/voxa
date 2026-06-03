"""prelabel/<preseg_id>/ store (scan-schema v2).

Each preseg result is {instance_ids.npy, segment_summary.json, meta.json}.
The only code that reads or writes this layout. Pure I/O — no FastAPI,
no in-memory state. Errors raise; callers decide how to surface them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np

from labeling.segment_io import atomic_write_json, atomic_write_npy, compute_fingerprint
from scenes.scan_layout import ScanLayout

_ID_RE = re.compile(r"^[a-z0-9_-]+$")


@dataclass(frozen=True)
class PresegInfo:
    preseg_id: str
    generator: str
    params: dict
    fingerprint: str
    created_at: str
    n_segments: int


def register_preseg(layout: ScanLayout, preseg_id: str, instance_ids: np.ndarray,
                    *, summary: dict, generator: str, params: dict) -> PresegInfo:
    """Publish a preseg result. The fingerprint and n_segments are computed
    once here and recorded in meta.json — meta.json is the preseg's identity:
    pin checks and listings read it instead of re-hashing the array."""
    if not _ID_RE.match(preseg_id):
        raise ValueError(f"preseg_id {preseg_id!r} must match {_ID_RE.pattern}")
    instance_ids = instance_ids.astype(np.int32, copy=False)
    d = layout.preseg_dir(preseg_id)
    atomic_write_npy(d / "instance_ids.npy", instance_ids)
    atomic_write_json(d / "segment_summary.json", summary)
    meta = {
        "preseg_id": preseg_id,
        "generator": generator,
        "params": params,
        "fingerprint": compute_fingerprint(instance_ids),
        "n_segments": int(np.unique(instance_ids[instance_ids >= 0]).size),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_json(d / "meta.json", meta)
    return PresegInfo(**meta)


def read_preseg_meta(layout: ScanLayout, preseg_id: str) -> dict:
    """Parsed prelabel/<id>/meta.json. Raises FileNotFoundError if the file
    is absent or missing required identity keys — both mean "no valid v2
    preseg here" to callers (listing → 500, pin check → mismatch)."""
    meta_path = layout.preseg_dir(preseg_id) / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"preseg '{preseg_id}' has no meta.json (not a v2 preseg?)")
    meta = json.loads(meta_path.read_text())
    for key in ("preseg_id", "fingerprint"):
        if key not in meta:
            raise FileNotFoundError(
                f"preseg '{preseg_id}': meta.json is missing required key "
                f"'{key}' — re-run register_preseg")
    return meta


def list_presegs(layout: ScanLayout) -> list[PresegInfo]:
    """Enumerate prelabel/*/meta.json. Pure metadata I/O — never loads the
    point arrays. A dir without a readable meta.json raises, because a
    malformed preseg silently vanishing from the picker hides bugs."""
    root = layout.presegs_root
    if not root.is_dir():
        return []
    out: list[PresegInfo] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        meta = read_preseg_meta(layout, d.name)
        out.append(PresegInfo(
            preseg_id=meta["preseg_id"], generator=meta.get("generator", "?"),
            params=meta.get("params", {}), fingerprint=meta["fingerprint"],
            created_at=meta.get("created_at", ""),
            n_segments=int(meta.get("n_segments", 0)),
        ))
    return out


def load_preseg(layout: ScanLayout, preseg_id: str, n_points: int,
                ) -> tuple[np.ndarray, np.ndarray]:
    """Return (class_ids int8, instance_ids int32) for one preseg result.
    Moved from v1.3 segment_io.load_prelabel; raises on missing files /
    shape mismatch instead of degrading to None."""
    d = layout.preseg_dir(preseg_id)
    inst_path = d / "instance_ids.npy"
    summary_path = d / "segment_summary.json"
    if not (inst_path.exists() and summary_path.exists()
            and (d / "meta.json").exists()):
        raise FileNotFoundError(f"preseg '{preseg_id}' incomplete under {d}")
    instance_ids = np.load(inst_path).astype(np.int32)
    if instance_ids.shape != (n_points,):
        raise ValueError(
            f"preseg '{preseg_id}': shape {instance_ids.shape} != ({n_points},)")
    summary = json.loads(summary_path.read_text())
    seg_to_class = {int(s["id"]): int(s["class_id"])
                    for s in summary.get("segments", [])}
    class_ids = np.full(n_points, -1, dtype=np.int8)
    if seg_to_class:
        # LUT gather: one pass over the cloud instead of one mask per segment.
        max_id = max(seg_to_class)
        lut = np.full(max_id + 1, -1, dtype=np.int8)
        for sid, cid in seg_to_class.items():
            if sid >= 0:
                if not (-128 <= cid <= 127):
                    raise ValueError(
                        f"preseg '{preseg_id}': segment {sid} has class_id {cid} "
                        f"outside int8 range")
                lut[sid] = np.int8(cid)
        m = (instance_ids >= 0) & (instance_ids <= max_id)
        class_ids[m] = lut[instance_ids[m]]
    return class_ids, instance_ids
