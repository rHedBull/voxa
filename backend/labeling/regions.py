"""Eval-region store, gate, and stats (eval-labeling phase 1).

``eval_regions.json`` lives at the SCAN root — scan-level truth shared by
every session on the scan. Geometry in the file is in the scan's STORED
frame; the routes convert to/from the runtime (recentered) frame via
``shift_prism`` + the load-time ``recenter_offset``. Caveat for z-up scans:
load also applies the z-up -> y-up rotation BEFORE recentering, so for those
scans "stored frame" means the y-up display frame plus offset, not literal
scan.ply coordinates — consistent within voxa (load is deterministic), but
the phase-3 manifest generator must replay the same rotation to read it.
The backend owns the eval-grade gate (p90 <= 10 mm over the region's
full-res points) and the geometry lock — same server-side-invariant
philosophy as ``reject_frozen_class``. See
docs/superpowers/specs/2026-07-21-eval-regions-design.md.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scan_schema import atomic_write_json

from labeling.materialize import loa_band, raw_sample_spacing
from labeling.shapes import prism_indices

REGIONS_FILE = "eval_regions.json"
MIN_GATE_POINTS = 100          # raw_sample_spacing returns (0,0) for n<2 —
EVAL_GRADE_P90_M = 0.010       # without a floor an empty region would PASS.


class RegionError(ValueError):
    """Invalid input or a locked/gated operation — routes map this to 422."""


class RegionNotFound(KeyError):
    """Unknown region id — routes map this to 404."""


def empty_doc() -> dict:
    return {"version": 1, "next_region_id": 1, "regions": []}


def load_regions(scan_dir) -> dict:
    p = Path(scan_dir) / REGIONS_FILE
    if not p.exists():
        return empty_doc()
    return json.loads(p.read_text())


def save_regions(scan_dir, doc: dict) -> None:
    atomic_write_json(Path(scan_dir) / REGIONS_FILE, doc)


def validate_prism(prism: dict) -> None:
    if len(prism.get("polygon") or []) < 3:
        raise RegionError("footprint needs at least 3 vertices")
    if not float(prism.get("height") or 0) > 0:
        raise RegionError("height must be > 0")


def shift_prism(prism: dict, delta) -> dict:
    """Translate a prism by (dx, dy, dz). stored = runtime + recenter_offset;
    pass the negated offset to go the other way."""
    dx, dy, dz = (float(v) for v in delta)
    return {"polygon": [[float(x) + dx, float(z) + dz] for x, z in prism["polygon"]],
            "y0": float(prism["y0"]) + dy, "height": float(prism["height"])}


def _get(doc: dict, rid: int) -> dict:
    for r in doc["regions"]:
        if r["id"] == rid:
            return r
    raise RegionNotFound(f"no region with id {rid}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_region(doc: dict, prism: dict, name: str | None = None) -> dict:
    validate_prism(prism)
    rid = int(doc["next_region_id"])
    doc["next_region_id"] = rid + 1
    region = {"id": rid, "name": name or f"Region {rid}", "status": "draft",
              "prism": shift_prism(prism, (0, 0, 0)),   # deep-copies + floats
              "created_at": _now()}
    doc["regions"].append(region)
    return region


def rename_region(doc: dict, rid: int, name: str) -> dict:
    name = (name or "").strip()
    if not name:
        raise RegionError("name must be non-empty")
    r = _get(doc, rid)
    r["name"] = name
    return r


def set_geometry(doc: dict, rid: int, prism: dict) -> dict:
    r = _get(doc, rid)
    if r["status"] == "eval_grade":
        raise RegionError("eval-grade geometry is locked — flip to draft first")
    validate_prism(prism)
    r["prism"] = shift_prism(prism, (0, 0, 0))
    return r


def delete_region(doc: dict, rid: int) -> None:
    r = _get(doc, rid)
    if r["status"] == "eval_grade":
        raise RegionError("eval-grade region is locked — flip to draft first")
    doc["regions"].remove(r)


def flip_status(doc: dict, rid: int, status: str, positions,
                offset=(0.0, 0.0, 0.0)) -> dict:
    """draft <-> eval_grade. The eval_grade flip is the gate: measure the
    region's local p50/p90 spacing over its full-res points and refuse
    (RegionError) below the point floor or above the 10 mm bar."""
    r = _get(doc, rid)
    if status == "draft":
        r["status"] = "draft"
        r.pop("accuracy", None)
        return r
    if status != "eval_grade":
        raise RegionError(f"unknown status {status!r}")
    if r["status"] == "eval_grade":
        return r
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    runtime = shift_prism(r["prism"], [-v for v in offset])
    idx = prism_indices(positions, runtime)
    if len(idx) < MIN_GATE_POINTS:
        raise RegionError(
            f"region holds {len(idx)} points — at least {MIN_GATE_POINTS} "
            "needed to measure spacing")
    p50, p90 = raw_sample_spacing(positions[idx])
    if p90 > EVAL_GRADE_P90_M:
        raise RegionError(
            f"measured p90 spacing {p90 * 1000:.1f} mm exceeds the "
            f"{EVAL_GRADE_P90_M * 1000:.0f} mm eval-grade bar")
    r["status"] = "eval_grade"
    r["accuracy"] = {"p50": p50, "p90": p90, "loa": loa_band(p90),
                     "measured_at": _now()}
    return r


def region_stats(doc: dict, positions, class_ids, instance_ids,
                 offset=(0.0, 0.0, 0.0)) -> list[dict]:
    """Per-region point/unlabeled/instance-overlap counts over the full-res
    working arrays. The caller (frontend) filters instances to confirmed and
    applies the majority rule — confirmed-status lives client-side, exactly
    like the protect_instances pattern."""
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    class_ids = np.asarray(class_ids)
    instance_ids = np.asarray(instance_ids)
    labeled_all = instance_ids >= 0
    n_inst = int(instance_ids[labeled_all].max()) + 1 if labeled_all.any() else 0
    totals = np.bincount(instance_ids[labeled_all], minlength=n_inst)
    out = []
    for r in doc["regions"]:
        runtime = shift_prism(r["prism"], [-v for v in offset])
        idx = prism_indices(positions, runtime)
        inst_in = instance_ids[idx]
        counts = (np.bincount(inst_in[inst_in >= 0], minlength=n_inst)
                  if n_inst else np.zeros(0, dtype=int))
        out.append({
            "id": r["id"],
            "n_points": int(len(idx)),
            "n_unlabeled": int((class_ids[idx] < 0).sum()),
            "instances": {int(i): {"inside": int(counts[i]), "total": int(totals[i])}
                          for i in np.nonzero(counts)[0]},
        })
    return out
