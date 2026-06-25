"""Lint a single scan directory against the shared scan-schema contract.

    .venv/bin/python scripts/scan/validate_scan.py <scan_dir>

Meta-level checks delegate to ``scan_schema.metadata.check_meta`` (the single
schema definition) — 2.x scans grandfather frame/derivation as warnings, 3.x
treats them as errors. Voxa adds one supplement the package doesn't model: a
render-run pin check (renders/<run>/meta.json::generated_from.variant_id).

For whole-archive validation use the package CLI: ``python -m scan_schema
validate <lidar_root>``. The per-scan checks live in
``validate_scan_dir(scan_dir) -> list[str]`` so they stay unit-testable.
Exit 0 if clean, 1 if any error is found.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from scan_schema.metadata import check_meta  # noqa: E402


def validate_scan_dir(scan_dir: Path) -> list[str]:
    """Return hard errors for a scan dir (empty == clean). Schema-version /
    frame / derivation / required-field checks come from check_meta; the
    render-run pin check is a voxa-local supplement."""
    scan_dir = Path(scan_dir)
    meta_path = scan_dir / "meta.json"
    if not meta_path.exists():
        return [f"{scan_dir}: missing meta.json"]
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError) as exc:
        return [f"{scan_dir}: unreadable meta.json: {exc}"]

    errors, _warnings = check_meta(meta)
    violations: list[str] = list(errors)

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
