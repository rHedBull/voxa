"""Voxa-side viewer heuristics over scan meta.json.

The canonical meta.json reader (frame hydration + derivation normalization) now
lives in the shared package (``scan_schema.metadata.read_scan_meta``); this module
keeps only the voxa-specific view concerns: Y-up inference and the UI summary.
"""
from __future__ import annotations

from pathlib import Path

from scan_schema.metadata import read_scan_meta


def is_z_up_from_meta(meta: dict) -> bool:
    """Frame rule shared by scene discovery and the migration script: a scan
    is Y-up only when sampled from a mesh and not from a LAZ; default Z-up."""
    return not (meta.get("source_mesh") and not meta.get("source_laz"))


def frame_summary(scan_dir: Path) -> dict:
    """Compact frame/provenance summary for the API/UI. Returns ``{}`` when the
    scan has no meta.json (non-annotated tiers); never raises for that case."""
    if not (Path(scan_dir) / "meta.json").exists():
        return {}
    meta = read_scan_meta(scan_dir)
    f = meta["frame"]
    return {
        "schema_version": meta.get("schema_version"),
        "variant_id": meta["derivation"].get("variant_id"),
        "frame_canonical_id": f.canonical_id,
        "frame_uncertain": bool(f.frame_uncertain),
        "georef_offset": (f.georef or {}).get("offset_m"),
    }
