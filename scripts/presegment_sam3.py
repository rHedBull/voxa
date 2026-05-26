"""Stage 2 of the SAM3 presegmentation pipeline: preseg from cached features.

Run this with voxa's ``.venv`` python (its Open3D build is stable; anaconda's
SIGSEGVs in plane extraction):

    .venv/bin/python scripts/presegment_sam3.py <scan_dir>

Loads the per-point SAM3 features cached by stage 1
(``presegment_sam3_features.py`` → ``<scan_dir>/sam3/<scene>/sam3_features.npz``),
runs RANSAC presegmentation with feature-aware splitting, and writes
``<scan_dir>/prelabel/ransac_instance_ids.npy`` + ``ransac_segment_summary.json``
(which voxa surfaces as a prelabel when ``labels/`` is empty).

The features are loaded directly with numpy, so torch is not needed here.

Scale note: the RANSAC/Open3D pipeline is only stable up to ~500k points (it
segfaults on 3M clouds). Above ``--preseg-points`` we presegment a random
subsample and propagate instance ids to the full cloud via nearest-neighbour.
The prelabel is only a seed — you refine at full resolution in Label mode — so
the propagation is adequate.
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from presegment import presegment  # noqa: E402

DEFAULT_PRESEG_POINTS = 500_000


def classes_from_yaml(config_path: Path) -> dict[str, int]:
    """{name_lower: id} by enumeration order — matches main.py::load_classes."""
    if not config_path.exists():
        return {}
    data = yaml.safe_load(config_path.read_text()) or {}
    return {str(k).lower(): i for i, k in enumerate((data.get("classes", {})).keys())}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", type=Path, help="annotated/<scan> directory")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing prelabel files")
    ap.add_argument("--preseg-points", type=int, default=DEFAULT_PRESEG_POINTS,
                    help="preseg a random subsample of this size then NN-propagate "
                         "to the full cloud (0 = no subsample)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "classes.yaml")
    args = ap.parse_args()

    scan_dir: Path = args.scan_dir.resolve()
    ply_path = scan_dir / "source" / "scan.ply"
    npz_path = scan_dir / "sam3" / scan_dir.name / "sam3_features.npz"
    out_dir = scan_dir / "prelabel"
    inst_path = out_dir / "ransac_instance_ids.npy"
    summary_path = out_dir / "ransac_segment_summary.json"

    if not npz_path.exists():
        print(f"ERROR: no cached SAM3 features at {npz_path}\n"
              f"       run stage 1 first: anaconda/bin/python "
              f"scripts/presegment_sam3_features.py {scan_dir}", file=sys.stderr)
        return 2
    if (inst_path.exists() or summary_path.exists()) and not args.force:
        print(f"ERROR: prelabel already exists in {out_dir} (use --force)", file=sys.stderr)
        return 1

    from point_cloud import load_ply
    print(f"[load] {ply_path}")
    pc, _ = load_ply(ply_path)
    xyz = np.asarray(pc.points, dtype=np.float64)
    n = int(xyz.shape[0])

    # np.load defaults to allow_pickle=False; we only read plain `features`/`seen`.
    cache = np.load(npz_path)
    features = np.asarray(cache["features"], dtype=np.float32)
    seen = np.asarray(cache["seen"], dtype=np.int32)
    print(f"[load] {n:,} points, cached features {features.shape}, "
          f"{int((seen > 0).sum()):,} visible")
    if features.shape[0] != n:
        print(f"ERROR: feature rows {features.shape[0]} != points {n} — "
              f"recompute stage 1 for this cloud", file=sys.stderr)
        return 3

    class_map = classes_from_yaml(args.config)
    cap = args.preseg_points
    if cap and n > cap:
        rng = np.random.default_rng(args.seed)
        sub = np.sort(rng.choice(n, cap, replace=False))
        print(f"[preseg] {n:,} > {cap:,} cap — preseg subsample then NN-propagate")
        inst_sub, summary = presegment(
            xyz[sub], mode="ransac", class_map=class_map,
            features=features[sub], feature_seen=seen[sub], log=print,
        )
        from scipy.spatial import cKDTree
        print("[preseg] propagating via nearest-neighbour…")
        _, nn = cKDTree(xyz[sub]).query(xyz, k=1, workers=-1)
        inst = inst_sub[nn]
    else:
        print(f"[preseg] ransac mode, {len(class_map)} classes, feature-aware split")
        inst, summary = presegment(
            xyz, mode="ransac", class_map=class_map,
            features=features, feature_seen=seen, log=print,
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(inst_path, inst.astype(np.int32))
    summary_path.write_text(json.dumps({"segments": [
        {"id": int(s["id"]), "class_id": int(s.get("class_id", -1)),
         "label": s.get("label", "")} for s in summary
    ]}, indent=2))
    n_assigned = int((inst >= 0).sum())
    print(f"\n[done] wrote {inst_path}")
    print(f"[done] wrote {summary_path}")
    print(f"[done] {n_assigned:,}/{n:,} assigned across {len(summary)} segments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
