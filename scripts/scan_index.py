"""Generate a scan's ``variants.json`` index (scan-schema v1.3 §4.2).

The index is assembled from the scan's own ``meta.json`` (the labeling variant)
plus every ``renders/<run>/meta.json`` ``generated_from`` pin (the render-source
variants). It is a generated cache — the per-variant ``meta.json`` / render meta
remain authoritative — so it never drifts as long as it's regenerated.

    .venv/bin/python scripts/scan_index.py <scan_dir>   # writes <scan_dir>/variants.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from scenes.render_meta import read_render_meta  # noqa: E402
from scenes.scan_meta import read_scan_meta  # noqa: E402


def build_variants_index(scan_dir: Path) -> dict:
    scan_dir = Path(scan_dir)
    meta = read_scan_meta(scan_dir)
    deriv = meta["derivation"]
    frame = meta["frame"]
    scan_id = deriv["scan_id"]

    variants: dict[str, dict] = {}
    # the labeling variant (this scan dir)
    variants[deriv["variant_id"]] = {
        "variant_id": deriv["variant_id"],
        "varies": deriv.get("varies", []),
        "role": deriv.get("role"),
        "source_fingerprint": deriv.get("source_fingerprint"),
        "transform_to_canonical": frame.transform_to_canonical.tolist(),
        "path": str(scan_dir),
    }
    labeling_variant = deriv["variant_id"] if deriv.get("role") == "labeling" else None

    # render-source variants, from each render run's generated_from pin
    renders = scan_dir / "renders"
    if renders.exists():
        for run in sorted(d for d in renders.iterdir() if d.is_dir()):
            rm = read_render_meta(run)
            if rm is None:
                continue
            gf = rm.get("generated_from") or {}
            vid = gf.get("variant_id")
            if not vid or vid in variants:
                continue
            variants[vid] = {
                "variant_id": vid,
                "varies": ["frame", "density"],   # render cloud differs in frame (+density)
                "role": None,
                "source_fingerprint": gf.get("source_fingerprint"),
                "n_points": gf.get("n_points"),
                "source": gf.get("source"),
                "transform_to_canonical": rm["frame"].transform_to_canonical.tolist(),
                "path": None,   # external (Potree); not a scan-dir variant
            }

    return {
        "scan_id": scan_id,
        "canonical_id": frame.canonical_id,
        "labeling_variant": labeling_variant,
        "variants": list(variants.values()),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", type=Path)
    args = ap.parse_args()
    idx = build_variants_index(args.scan_dir)
    out = args.scan_dir / "variants.json"
    out.write_text(json.dumps(idx, indent=2))
    print(f"[scan_index] wrote {out} — {len(idx['variants'])} variant(s): "
          f"{[v['variant_id'] for v in idx['variants']]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
