"""Verify a scan's source/scan.ply registers to its renders/<run> poses.

Exit 0 if all checkable runs pass; 2 if any fails OR nothing was verifiable;
3 if there are no render runs at all. Run with voxa's .venv (no torch needed):

    .venv/bin/python scripts/verify_registration.py <scan_dir>

This is the scan-schema v1.3 §6 registration health-check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from preseg.registration import verify_scan_registration  # noqa: E402
from scenes.reproject import ORIENTATION_PRESETS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", type=Path)
    ap.add_argument("--orientation", default="Z+", choices=list(ORIENTATION_PRESETS))
    ap.add_argument("--min-coverage", type=float, default=0.35)
    ap.add_argument("--min-photometric", type=float, default=0.5)
    args = ap.parse_args()

    renders_root = args.scan_dir / "renders"
    runs = [d for d in renders_root.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()] if renders_root.is_dir() else []
    if not runs:
        print(f"ERROR: no render runs under {renders_root}", file=sys.stderr)
        return 3

    v = verify_scan_registration(args.scan_dir, orientation=args.orientation,
                                 min_coverage=args.min_coverage,
                                 min_photometric=args.min_photometric, use_cache=False)
    print(f"[verify] {args.scan_dir.name}: checked={v['checked']} ok={v['ok']}")
    for r in v["runs"]:
        ph = f"{r['photometric']:.1%}" if r["photometric"] is not None else "n/a"
        flag = "OK  " if r["ok"] else "FAIL"
        print(f"  [{flag}] {r['run_id']}: coverage {r['coverage']:.1%}, photometric {ph} "
              f"({r['n_seen']:,}/{r['n_points']:,}, {r['n_frames']} frames)")
        for reason in r["reasons"]:
            print(f"         - {reason}")
    for reason in v["reasons"]:
        if not v["runs"]:
            print(f"  - {reason}")
    return 0 if (v["checked"] and v["ok"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
