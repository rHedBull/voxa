"""Promote annotated scans to scan-schema v3.0 by routing lineage writes through
the shared Registry.

For each scan under <lidar_root>/annotated/:
  1. resolve its raw root (via meta.source_laz basename, else by scan name),
  2. ensure a valid frame block (kept if already present+valid; otherwise an
     identity ``<scan_id>#local`` frame is synthesized — the stored cloud is its
     own canonical-local, matching what backfill_scan_frame.py writes; georef
     ``coord_offset_m`` is preserved, frame flagged uncertain so the runtime
     registration health-check still runs),
  3. call Registry.set_derivation(..., bump_to_3=True) — the single lineage
     writer — which writes the v3.0 nested root/parent derivation and flips
     schema_version to 3.0 (check_meta gates frame/derivation as it writes).

Scans with no resolvable root (e.g. munich_water_pump) are skipped and stay 2.x.

    .venv/bin/python scripts/scan/promote_to_v3.py <lidar_root> [--dry-run] [--only NAME ...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402

from scan_schema.frame import Frame  # noqa: E402
from scan_schema.metadata import check_meta  # noqa: E402
from scan_schema.registry import Registry  # noqa: E402


def _resolve_root_id(reg: Registry, meta: dict, scan_name: str):
    src = meta.get("source_laz")
    if src:
        r = reg.root_by_basename(src)
        if r:
            return r.source_id
    r = reg.root(scan_name)          # matterport: root source_id == scan name
    return r.source_id if r else None


def _ensure_frame(meta: dict, scan_id: str) -> bool:
    """Add an identity frame if absent. Returns True if one was synthesized."""
    if meta.get("frame"):
        return False
    offset = meta.get("coord_offset_m")
    meta["frame"] = Frame(
        np.eye(4), f"{scan_id}#local",
        units=meta.get("units", "meters"),
        georef={"offset_m": offset} if offset else None,
        frame_uncertain=True,
    ).to_dict()
    return True


def promote_scan(scan_dir: Path, reg: Registry, *, dry_run: bool) -> tuple[str, str]:
    meta_path = scan_dir / "meta.json"
    meta = json.loads(meta_path.read_text())
    scan_name = meta.get("scan_name", scan_dir.name)
    deriv = meta.get("derivation") or {}
    scan_id = deriv.get("scan_id") or scan_name
    variant_id = deriv.get("variant_id") or scan_id
    varies = deriv.get("varies") or ["density"]
    role = deriv.get("role") or "labeling"

    root_id = _resolve_root_id(reg, meta, scan_name)
    if root_id is None:
        return ("skip", "no resolvable root (stays 2.x)")
    root_fp = reg.root(root_id).fingerprint

    new_frame = _ensure_frame(meta, scan_id)

    if dry_run:
        sim = dict(meta)
        sim["schema_version"] = "3.0"
        sim["derivation"] = {
            "scan_id": scan_id, "variant_id": variant_id, "varies": varies, "role": role,
            "root": {"source_id": root_id, "fingerprint": root_fp},
            "parent": {"ref": root_id, "fingerprint": root_fp},
        }
        errs, _ = check_meta(sim)
        status = "would-promote" if not errs else "WOULD-FAIL"
        return (status, f"root={root_id} frame={'synth' if new_frame else 'kept'} errors={errs}")

    if new_frame:
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    reg.set_derivation(scan_dir, root_id=root_id, parent_ref=root_id,
                       varies=varies, role=role, bump_to_3=True)
    return ("promoted", f"root={root_id} frame={'synth' if new_frame else 'kept'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lidar_root", type=Path)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", nargs="*", default=None, help="restrict to these scan dir names")
    a = ap.parse_args()

    reg = Registry.load(a.lidar_root)
    annotated = a.lidar_root / "annotated"
    rc = 0
    for scan_dir in sorted(p for p in annotated.iterdir() if p.is_dir()):
        if a.only and scan_dir.name not in a.only:
            continue
        status, detail = promote_scan(scan_dir, reg, dry_run=a.dry_run)
        print(f"  [{status:13}] {scan_dir.name:32} {detail}")
        if status == "WOULD-FAIL":
            rc = 1
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
