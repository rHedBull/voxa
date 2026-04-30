"""CLI: run RANSAC presegmentation and write `prelabel/ransac_*` for a scene.

Voxa's ``segment_io.load_prelabel`` reads two files from
``<scan>/prelabel/``:

  - ``ransac_instance_ids.npy``      int32 (N,)
  - ``ransac_segment_summary.json``  ``{"segments": [{id, class_id, label}]}``

This script generates both from a source PLY using ``presegment.presegment``
so any annotated scene gets a usable prelabel without depending on the
external segmentation repo or trained merge model.

Usage
-----

    python scripts/presegment.py <scene-id-or-scan-dir> [--output <dir>] [--force]

Examples:

    # SCHEMA scan directory (writes prelabel/ next to source/scan.ply)
    python scripts/presegment.py /lidar/annotated/munich_water_pump

    # Voxa-registered scene id (resolved via VOXA_LIDAR_ROOT / data/scenes)
    python scripts/presegment.py annotated/munich_water_pump

    # Bare PLY → write to the same dir
    python scripts/presegment.py /tmp/foo.ply --output /tmp/foo_prelabel
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from presegment import presegment  # noqa: E402


def _classes_from_yaml(config_path: Path) -> dict[str, int]:
    """Build a {name_lower: id} mapping from voxa's classes.yaml.

    Voxa's classes.yaml is keyed by name and has no explicit `id`. We
    assign ids by enumeration order, matching ``main.py::load_classes``.
    """
    if not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text()) or {}
    classes = data.get("classes", {})
    out: dict[str, int] = {}
    for i, key in enumerate(classes.keys()):
        out[str(key).lower()] = i
    return out


def _resolve_scene(arg: str) -> tuple[Path, Path]:
    """Resolve ``arg`` to (scan_dir, source_ply).

    Cases:
      - Path to a SCHEMA scan dir (has ``source/scan.ply``)
      - Path to a raw .ply file
      - Voxa scene id (looked up via scene_registry)
    """
    p = Path(arg)
    if p.is_dir() and (p / "source" / "scan.ply").exists():
        return p, p / "source" / "scan.ply"
    if p.is_file() and p.suffix.lower() == ".ply":
        return p.parent, p

    # Treat as scene id via voxa's registry
    from scene_registry import resolve
    data_dir = Path(os.environ.get("VOXA_DATA_DIR", str(ROOT / "data")))
    lidar_root = Path(os.environ.get("VOXA_LIDAR_ROOT",
                                     "/home/hendrik/coding/engine/data/lidar"))
    try:
        src = resolve(arg, data_dir, lidar_root if lidar_root.exists() else None)
    except KeyError as e:
        raise SystemExit(f"Unknown scene: {arg!r}") from e
    if src.tier == "annotated":
        scan = src.path  # path to scan dir
        return scan, scan / "source" / "scan.ply"
    # decimated / raw / legacy: src.path is a PLY file
    return src.path.parent, src.path


def _load_xyz(ply_path: Path) -> np.ndarray:
    """Load just XYZ from a PLY. Uses voxa's loader for consistency."""
    from point_cloud import load_ply
    pc, _mesh = load_ply(ply_path)
    return np.asarray(pc.points, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="Scene id, scan dir, or .ply path")
    ap.add_argument("--output", type=Path, default=None,
                    help="Override output dir (default: <scan>/prelabel)")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "classes.yaml",
                    help="Voxa classes.yaml (used to assign class_id per segment)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing prelabel files")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress logging")
    args = ap.parse_args()

    scan_dir, ply_path = _resolve_scene(args.scene)
    out_dir = args.output or (scan_dir / "prelabel")

    inst_path = out_dir / "ransac_instance_ids.npy"
    summary_path = out_dir / "ransac_segment_summary.json"
    if (inst_path.exists() or summary_path.exists()) and not args.force:
        print(f"Refusing to overwrite existing prelabel in {out_dir} (use --force).",
              file=sys.stderr)
        return 1

    print(f"Loading {ply_path}…")
    xyz = _load_xyz(ply_path)
    class_map = _classes_from_yaml(args.config)
    if not class_map:
        print(f"warning: no classes loaded from {args.config}; class_ids will be -1",
              file=sys.stderr)

    log = (lambda *_: None) if args.quiet else print
    instance_ids, summary = presegment(xyz, class_map=class_map, log=log)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(inst_path, instance_ids.astype(np.int32))
    summary_path.write_text(json.dumps({"segments": [
        {"id": int(s["id"]),
         "class_id": int(s.get("class_id", -1)),
         "label": s.get("label", "")}
        for s in summary
    ]}, indent=2))

    n_assigned = int((instance_ids >= 0).sum())
    print(f"\nWrote {inst_path.name} ({len(instance_ids)} pts, "
          f"{n_assigned} assigned, {len(summary)} segments)")
    print(f"Wrote {summary_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
