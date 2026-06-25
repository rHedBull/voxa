"""CLI: run RANSAC presegmentation and publish to prelabel/<preseg_id>/ (scan-schema v2).

Writes via ``preseg_store.register_preseg`` into
``<scan>/prelabel/<preseg_id>/{instance_ids.npy, segment_summary.json, meta.json}``.

Usage
-----

    python scripts/preseg/presegment.py <scene-id-or-scan-dir> [--preseg-id ID] [--force]

Examples:

    # SCHEMA scan directory (writes to prelabel/ransac/)
    python scripts/preseg/presegment.py /lidar/annotated/munich_water_pump

    # Custom preseg id
    python scripts/preseg/presegment.py /lidar/annotated/munich_water_pump --preseg-id ransac_v2

    # Voxa-registered scene id (resolved via VOXA_LIDAR_ROOT / data/scenes)
    python scripts/preseg/presegment.py annotated/munich_water_pump
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from preseg.presegment_ransac import presegment  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import classes_from_yaml, publish_preseg  # noqa: E402


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
    from scenes.scene_registry import resolve
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
    from scenes.point_cloud import load_ply
    pc, _mesh = load_ply(ply_path)
    return np.asarray(pc.points, dtype=np.float64)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", help="Scene id, scan dir, or .ply path")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "classes.yaml",
                    help="Voxa classes.yaml (used to assign class_id per segment)")
    ap.add_argument("--force", action="store_true",
                    help="Overwrite existing prelabel files")
    ap.add_argument("--quiet", action="store_true", help="Suppress progress logging")
    ap.add_argument("--preseg-id", default="ransac",
                    help="Preseg identifier written to prelabel/<id>/ (default: ransac)")
    args = ap.parse_args()

    scan_dir, ply_path = _resolve_scene(args.scene)

    from scan_schema.layout import ScanLayout
    preseg_dir = ScanLayout(scan_dir).preseg_dir(args.preseg_id)
    if preseg_dir.is_dir() and any(preseg_dir.iterdir()) and not args.force:
        print(f"Refusing to overwrite existing prelabel in {preseg_dir} (use --force).",
              file=sys.stderr)
        return 1

    print(f"Loading {ply_path}…")
    xyz = _load_xyz(ply_path)
    class_map = classes_from_yaml(args.config)
    if not class_map:
        print(f"warning: no classes loaded from {args.config}; class_ids will be -1",
              file=sys.stderr)

    log = (lambda *_: None) if args.quiet else print
    instance_ids, summary = presegment(xyz, class_map=class_map, log=log)

    publish_preseg(scan_dir, args.preseg_id, instance_ids, summary,
                   generator="ransac", params={})

    n_assigned = int((instance_ids >= 0).sum())
    print(f"\nWrote prelabel/{args.preseg_id}/instance_ids.npy "
          f"({len(instance_ids)} pts, {n_assigned} assigned, {len(summary)} segments)")
    print(f"Wrote prelabel/{args.preseg_id}/segment_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
