"""Shared helpers for the scripts/preseg/ CLIs.

Single home for the bits the RANSAC + SAM3 preseg scripts had each copied: the
classes.yaml -> id map, the PLY-header vertex count, and the v2 preseg publisher
(routes through preseg_store.register_preseg into prelabel/<preseg_id>/).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make backend importable when _common is imported from scripts/preseg/.
_BACKEND = Path(__file__).resolve().parents[2] / "backend"
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))


def classes_from_yaml(config_path: Path) -> dict[str, int]:
    """{name_lower: id} from voxa's classes.yaml.

    Thin delegate to backend ``app.core._voxa_class_name_to_id`` — the single
    home for the name↔id mapping (explicit ``id:`` with positional fallback,
    duplicate-id guard) — so the ids published into segment_summary.json can
    never drift from the app palette and exports.
    """
    from app.core import _voxa_class_name_to_id  # lazy backend import
    return _voxa_class_name_to_id(config_path)


def ply_vertex_count(path: Path) -> int:
    """Vertex count from a binary PLY header without loading the points.

    Lets a caller reject an oversized cloud before a multi-GB load would OOM.
    """
    with open(path, "rb") as f:
        for _ in range(60):  # headers are short; bail out defensively
            line = f.readline()
            if not line or line.strip() == b"end_header":
                break
            if line.startswith(b"element vertex"):
                return int(line.split()[2])
    raise ValueError(f"no 'element vertex' in PLY header: {path}")


def publish_preseg(scan_dir: Path, preseg_id: str, instance_ids: np.ndarray,
                   summary: list, *, generator: str, params: dict) -> "PresegInfo":  # noqa: F821 — lazy backend import below
    """Publish a preseg result into prelabel/<preseg_id>/ (scan-schema v2)
    via the backend's register_preseg — the single writer of that layout."""
    from preseg.preseg_store import register_preseg
    from scan_schema.layout import ScanLayout
    return register_preseg(
        ScanLayout(scan_dir), preseg_id, instance_ids,
        summary={"segments": [
            {"id": int(s["id"]), "class_id": int(s.get("class_id", -1)),
             "label": s.get("label", "")}
            for s in summary
        ]},
        generator=generator,
        params=params,
    )
