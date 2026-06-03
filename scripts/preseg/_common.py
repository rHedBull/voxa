"""Shared helpers for the scripts/preseg/ CLIs.

Single home for the bits the RANSAC + SAM3 preseg scripts had each copied: the
classes.yaml -> id map, the PLY-header vertex count, and the prelabel/ransac_*
writer + its filenames (the contract voxa's segment_io.load_prelabel reads).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml


def classes_from_yaml(config_path: Path) -> dict[str, int]:
    """{name_lower: id} from voxa's classes.yaml, ids by enumeration order.

    classes.yaml is keyed by name with no explicit id; ordering matches
    backend ``main.py::load_classes`` so the int ids line up with the palette.
    """
    if not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text()) or {}
    return {str(k).lower(): i for i, k in enumerate((data.get("classes", {})).keys())}


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


def prelabel_paths(out_dir: Path) -> tuple[Path, Path]:
    """The two files voxa's ``segment_io.load_prelabel`` reads back."""
    return (out_dir / "ransac_instance_ids.npy",
            out_dir / "ransac_segment_summary.json")


def write_prelabel(out_dir: Path, instance_ids: np.ndarray, summary: list) -> tuple[Path, Path]:
    """Write ``prelabel/ransac_instance_ids.npy`` + ``ransac_segment_summary.json``."""
    inst_path, summary_path = prelabel_paths(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(inst_path, instance_ids.astype(np.int32))
    summary_path.write_text(json.dumps({"segments": [
        {"id": int(s["id"]), "class_id": int(s.get("class_id", -1)),
         "label": s.get("label", "")}
        for s in summary
    ]}, indent=2))
    return inst_path, summary_path
