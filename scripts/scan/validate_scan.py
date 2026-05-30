"""Lint a scan directory against the scan-schema v1.3 invariants (§7).

    .venv/bin/python scripts/validate_scan.py <scan_dir>

Exit 0 if clean, 1 if any invariant is violated. The checks live in
``validate_scan_dir(scan_dir) -> list[str]`` so they are unit-testable.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from scenes.frame import is_rigid  # noqa: E402

_ALLOWED_VARIES = {"density", "frame", "points", "color", "attributes"}


def validate_scan_dir(scan_dir: Path) -> list[str]:
    scan_dir = Path(scan_dir)
    violations: list[str] = []
    meta_path = scan_dir / "meta.json"
    if not meta_path.exists():
        return [f"{scan_dir}: missing meta.json"]
    meta = json.loads(meta_path.read_text())

    sv = meta.get("schema_version")
    if not sv:
        violations.append("missing schema_version")

    if sv and str(sv) >= "1.3":
        frame = meta.get("frame")
        deriv = meta.get("derivation")
        scan_id = (deriv or {}).get("scan_id") or meta.get("scan_name")
        if not frame:
            violations.append("v1.3 requires a frame block")
        else:
            M = np.asarray(frame.get("transform_to_canonical", []), dtype=float)
            if M.shape != (4, 4) or not is_rigid(M):
                violations.append("frame.transform_to_canonical must be a 4x4 rigid transform")
            if frame.get("canonical_id") != f"{scan_id}#local":
                violations.append(
                    f"frame.canonical_id should be '{scan_id}#local', "
                    f"got '{frame.get('canonical_id')}'")
        if not deriv:
            violations.append("v1.3 requires a derivation block")
        else:
            bad = set(deriv.get("varies", [])) - _ALLOWED_VARIES
            if bad:
                violations.append(f"derivation.varies has invalid entries: {sorted(bad)}")

    renders = scan_dir / "renders"
    if renders.exists():
        for run in sorted(renders.iterdir()):
            if not (run.is_dir() and (run / "manifest.json").exists()):
                continue
            rm = run / "meta.json"
            if not rm.exists():
                violations.append(f"renders/{run.name}: missing meta.json (render-run pin)")
            else:
                gf = json.loads(rm.read_text()).get("generated_from") or {}
                if not gf.get("variant_id"):
                    violations.append(
                        f"renders/{run.name}: meta.json missing generated_from.variant_id")
    return violations


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", type=Path)
    args = ap.parse_args()
    violations = validate_scan_dir(args.scan_dir)
    if not violations:
        print(f"[validate] {args.scan_dir.name}: OK")
        return 0
    print(f"[validate] {args.scan_dir.name}: {len(violations)} violation(s):", file=sys.stderr)
    for v in violations:
        print(f"  - {v}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
