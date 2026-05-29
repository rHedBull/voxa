"""Verify a scan's source/scan.ply registers to its renders/<run> poses.

Exit 0 if coverage+photometric pass thresholds; non-zero otherwise. Run with
voxa's .venv (no torch needed):

    .venv/bin/python scripts/verify_registration.py <scan_dir> [--run <name>]

This is the scan-schema v1.3 §6 registration health-check: it catches the case
where the labeling cloud and the renders' camera poses live in different frames
(the navvis incident) before any expensive SAM3 work.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from preseg.registration import check_registration, registration_score  # noqa: E402
from scenes.fingerprint import cloud_fingerprint  # noqa: E402
from scenes.point_cloud import load_ply  # noqa: E402
from scenes.reproject import ORIENTATION_PRESETS, euler_xyz_matrix  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", type=Path)
    ap.add_argument("--run", default=None, help="renders/<run> name (default: all runs)")
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--orientation", default="Z+", choices=list(ORIENTATION_PRESETS))
    ap.add_argument("--min-coverage", type=float, default=0.35)
    ap.add_argument("--min-photometric", type=float, default=0.5)
    args = ap.parse_args()

    pc, _ = load_ply(args.scan_dir / "source" / "scan.ply")   # load_ply takes NO orientation
    xyz_raw = np.asarray(pc.points, dtype=np.float64)
    R = euler_xyz_matrix(*ORIENTATION_PRESETS[args.orientation])  # rotate like extract_or_load
    xyz = xyz_raw @ R.T
    _c = getattr(pc, "colors", None)
    rgb = np.asarray(_c).astype(np.uint8) if _c is not None and len(_c) else None
    print(f"[verify] {args.scan_dir.name}: {len(xyz):,} pts  "
          f"fp={cloud_fingerprint(xyz_raw)[:23]}…  colours={'yes' if rgb is not None else 'no'}")

    renders_root = args.scan_dir / "renders"
    if args.run:
        runs = [renders_root / args.run]
    else:
        runs = sorted(d for d in renders_root.iterdir()
                      if d.is_dir() and (d / "manifest.json").exists())
    if not runs:
        print(f"ERROR: no render runs under {renders_root}", file=sys.stderr)
        return 3

    from PIL import Image
    ok_all = True
    for run in runs:
        m = json.loads((run / "manifest.json").read_text())
        frames = [f for f in m.get("frames", []) if (run / f["file"]).exists()]
        if not frames:
            print(f"  [skip] {run.name}: no frame images on disk")
            continue
        loader = lambda f, _run=run: np.array(Image.open(_run / f["file"]).convert("RGB"))
        W, H = Image.open(run / frames[0]["file"]).size
        s = registration_score(xyz, frames, fov_y_deg=args.fov, W=W, H=H,
                               rgb=rgb, image_loader=loader)
        ok, reasons = check_registration(s, min_coverage=args.min_coverage,
                                         min_photometric=args.min_photometric)
        flag = "OK  " if ok else "FAIL"
        ph = f"{s['photometric']:.1%}" if s["photometric"] is not None else "n/a"
        print(f"  [{flag}] {run.name}: coverage {s['coverage']:.1%}, photometric {ph} "
              f"({s['n_seen']:,}/{s['n_points']:,}, {s['n_frames']} frames)")
        for r in reasons:
            print(f"         - {r}")
        ok_all &= ok
    return 0 if ok_all else 2


if __name__ == "__main__":
    raise SystemExit(main())
