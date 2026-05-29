"""Scan/variant meta.json reader with v1.2 -> v1.3 back-compat (§4.1)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scenes.frame import Frame, frame_from_dict


def _legacy_frame(meta: dict) -> Frame:
    """Synthesize a frame for a pre-v1.3 meta.json.

    We don't know the aligned canonical for legacy clouds, so we treat the stored
    cloud as its own canonical-local (identity transform) and flag it uncertain so
    the §6 registration health-check is mandatory before trusting it. The legacy
    ``coord_offset_m`` (if any) is preserved as the world georeference offset.
    """
    scan_id = meta.get("scan_name", "unknown")
    offset = meta.get("coord_offset_m")
    georef = {"offset_m": offset} if offset else None
    return Frame(np.eye(4), f"{scan_id}#local",
                 units=meta.get("units", "meters"),
                 georef=georef, frame_uncertain=True)


def read_scan_meta(scan_dir: Path) -> dict:
    """Return the parsed meta.json with ``frame`` as a ``Frame`` and a normalized
    ``derivation`` block (synthesized for legacy scans)."""
    meta = json.loads((Path(scan_dir) / "meta.json").read_text())
    if "frame" in meta:
        meta["frame"] = frame_from_dict(meta["frame"])
    else:
        meta["frame"] = _legacy_frame(meta)
    if "derivation" not in meta:
        scan_id = meta.get("scan_name", "unknown")
        meta["derivation"] = {
            "scan_id": scan_id, "variant_id": scan_id,
            "parent": "original", "op": "asis", "varies": [],
            "source_fingerprint": None, "role": None,
        }
    return meta


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
