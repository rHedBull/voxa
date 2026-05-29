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

import preseg.sam3_features as sam3  # noqa: E402
from app.constants import MAX_LABEL_POINTS  # noqa: E402


def _ply_vertex_count(path: Path) -> int:
    """Vertex count from a binary PLY header without loading the points."""
    with open(path, "rb") as f:
        for _ in range(60):
            line = f.readline()
            if not line or line.strip() == b"end_header":
                break
            if line.startswith(b"element vertex"):
                return int(line.split()[2])
    raise ValueError(f"no 'element vertex' in PLY header: {path}")


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", type=Path, help="annotated/<scan> directory")
    ap.add_argument("--force", action="store_true",
                    help="recompute even if a feature cache already exists")
    ap.add_argument("--fpn-level", type=int, default=0)
    ap.add_argument("--pca-dim", type=int, default=64)
    ap.add_argument("--skip-registration-check", action="store_true",
                    help="bypass the cloud↔renders registration health-check (not recommended)")
    args = ap.parse_args()

    scan_dir: Path = args.scan_dir.resolve()
    ply_path = scan_dir / "source" / "scan.ply"
    if not ply_path.exists():
        print(f"ERROR: no source/scan.ply in {scan_dir}", file=sys.stderr)
        return 2

    n_header = _ply_vertex_count(ply_path)
    if n_header > MAX_LABEL_POINTS:
        print(f"ERROR: {ply_path} has {n_header:,} points > label cap "
              f"{MAX_LABEL_POINTS:,}.\n       Voxa can't label a cloud this large, and "
              f"SAM3 projection over {n_header:,} points is infeasible. Downsample "
              f"source/scan.ply first (see docs/point-cloud-sizing.md).", file=sys.stderr)
        return 5

    from scenes.point_cloud import load_ply
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

    # Registration health-check (scan-schema v1.3 §6): refuse to spend SAM3 compute
    # if the cloud doesn't actually project into the renders' poses (the navvis
    # frame-mismatch bug). Uses the SAME "Z+" orientation extract_or_load defaults to.
    if not args.skip_registration_check:
        import json as _json

        from PIL import Image
        from preseg.registration import check_registration, registration_score
        from scenes.reproject import ORIENTATION_PRESETS, euler_xyz_matrix

        Rchk = euler_xyz_matrix(*ORIENTATION_PRESETS["Z+"])
        xyz_chk = xyz @ Rchk.T
        _c = getattr(pc, "colors", None)
        rgb = np.asarray(_c).astype(np.uint8) if _c is not None and len(_c) else None

        sample, W, H = [], None, None
        for r in runs:
            m = _json.loads((r.path / "manifest.json").read_text())
            frs = [f for f in m.get("frames", []) if (r.path / f["file"]).exists()][:8]
            for f in frs:
                f["_run"] = str(r.path)
                sample.append(f)
            if W is None and frs:
                W, H = Image.open(r.path / frs[0]["file"]).size

        def _loader(f):
            return np.array(Image.open(Path(f["_run"]) / f["file"]).convert("RGB"))

        if sample and W is not None:
            score = registration_score(xyz_chk, sample, fov_y_deg=60.0, W=W, H=H,
                                       rgb=rgb, image_loader=_loader)
            ok, reasons = check_registration(score)
            ph = f"{score['photometric']:.1%}" if score["photometric"] is not None else "n/a"
            print(f"[reg-check] coverage {score['coverage']:.1%}, photometric {ph} "
                  f"({score['n_seen']:,}/{score['n_points']:,})")
            if not ok:
                print("ERROR: cloud does not register to its renders — refusing to "
                      "compute SAM3 features:", file=sys.stderr)
                for reason in reasons:
                    print(f"  - {reason}", file=sys.stderr)
                print("  (run scripts/verify_registration.py to inspect; "
                      "re-run with --skip-registration-check to override)", file=sys.stderr)
                return 6

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
