"""Stage 1 of the SAM3 presegmentation pipeline: extract + cache per-point features.

Run this with a python that has torch + CUDA + the ``sam3`` package — i.e. the
**anaconda base** interpreter, NOT voxa's ``.venv`` (which has no torch):

    /home/hendrik/anaconda3/bin/python scripts/presegment_sam3_features.py <scan_dir>

It discovers the render runs under ``<scan_dir>/renders/`` (lidar/SCHEMA.md
v1.2), runs the SAM3 image encoder per frame, projects every point, and caches
per-point features to ``<scan_dir>/sam3/<scene>/sam3_features.npz``.

Then run stage 2 (``presegment_sam3.py``) with voxa's ``.venv`` to turn those
features into ``prelabel/ransac_*``. The split exists because SAM3 needs torch
(anaconda only) while the RANSAC/Open3D preseg is stable only in ``.venv`` —
anaconda's Open3D build SIGSEGVs in plane extraction.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import sam3_features as sam3  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", type=Path, help="annotated/<scan> directory")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if a feature cache already exists")
    ap.add_argument("--fpn-level", type=int, default=0)
    ap.add_argument("--pca-dim", type=int, default=64)
    args = ap.parse_args()

    scan_dir: Path = args.scan_dir.resolve()
    ply_path = scan_dir / "source" / "scan.ply"
    if not ply_path.exists():
        print(f"ERROR: no source/scan.ply in {scan_dir}", file=sys.stderr)
        return 2

    from point_cloud import load_ply
    print(f"[load] {ply_path}")
    pc, _ = load_ply(ply_path)
    xyz = np.asarray(pc.points, dtype=np.float64)
    n = int(xyz.shape[0])
    print(f"[load] {n:,} points")

    runs = sam3.discover_render_runs(scan_dir.name, scan_dir=scan_dir)
    if not runs:
        print(f"ERROR: no render runs with manifest.json under {scan_dir}/renders/",
              file=sys.stderr)
        return 4
    render_dirs = [r.path for r in runs]
    print(f"[sam3] {len(render_dirs)} render run(s), "
          f"{sum(r.n_frames for r in runs)} frames total")
    for r in runs:
        print(f"        - {r.name}: {r.n_frames} frames")

    _features, seen, _meta = sam3.extract_or_load(
        xyz, scan_dir.name,
        render_dirs=render_dirs,
        cache_dir=scan_dir / "sam3",
        fpn_level=int(args.fpn_level),
        pca_dim=int(args.pca_dim),
        force=bool(args.force),
        log=print,
    )
    cache = scan_dir / "sam3" / scan_dir.name / "sam3_features.npz"
    print(f"\n[done] cached features → {cache}")
    print(f"[done] {int((seen > 0).sum()):,}/{n:,} points visible in >=1 frame")
    print(f"[done] next: .venv/bin/python scripts/presegment_sam3.py {scan_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
