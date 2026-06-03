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

Scale note: by default this presegments the FULL cloud — voxa's ``.venv`` Open3D
build handles 3M points fine (~6 min, ~20 GB RAM at 3M). The crashes seen with
the anaconda Open3D build are a broken-build artifact, not a scale limit, and
this stage runs under ``.venv`` anyway. For memory-constrained machines or clouds
near/over the label cap, pass ``--preseg-points N`` to presegment a random N-point
subsample and propagate instance ids to the full cloud via nearest-neighbour (the
prelabel is only a seed, refined at full resolution in Label mode).
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))

from preseg.presegment_ransac import presegment  # noqa: E402
from app.constants import MAX_LABEL_POINTS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import classes_from_yaml, ply_vertex_count, publish_preseg  # noqa: E402

# 0 = presegment the full cloud, bounded by a RAM-safe ceiling (see
# _ram_safe_ceiling). A positive value forces a subsample of that size.
DEFAULT_PRESEG_POINTS = 0

# RANSAC preseg peaks around this much resident memory per point (measured:
# ~19.5 GB RSS for 3M points). Used to pick a safe full-res ceiling.
BYTES_PER_POINT = 6_500


def _ram_safe_ceiling() -> int:
    """Largest full-res preseg this machine can run without thrashing.

    Budgets ~85% of total RAM at BYTES_PER_POINT, clamped to
    [500k, MAX_LABEL_POINTS]. Clouds above this are subsampled + propagated.
    """
    try:
        total = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        total = 16e9
    est = int(0.85 * total / BYTES_PER_POINT)
    return max(500_000, min(est, MAX_LABEL_POINTS))


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", type=Path, help="annotated/<scan> directory")
    ap.add_argument("--force", action="store_true",
                    help="overwrite existing prelabel files")
    ap.add_argument("--preseg-id", default="sam3",
                    help="Preseg identifier written to prelabel/<id>/ (default: sam3)")
    ap.add_argument("--preseg-points", type=int, default=DEFAULT_PRESEG_POINTS,
                    help="0 (default) = presegment the full cloud; a positive value "
                         "presegs a random subsample of that size then NN-propagates "
                         "to the full cloud (for memory-constrained / very large clouds)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "classes.yaml")
    args = ap.parse_args()

    scan_dir: Path = args.scan_dir.resolve()
    ply_path = scan_dir / "source" / "scan.ply"
    npz_path = scan_dir / "sam3" / scan_dir.name / "sam3_features.npz"

    from scenes.scan_layout import ScanLayout
    preseg_dir = ScanLayout(scan_dir).preseg_dir(args.preseg_id)

    if not npz_path.exists():
        print(f"ERROR: no cached SAM3 features at {npz_path}\n"
              f"       run stage 1 first: anaconda/bin/python "
              f"scripts/presegment_sam3_features.py {scan_dir}", file=sys.stderr)
        return 2
    if preseg_dir.is_dir() and any(preseg_dir.iterdir()) and not args.force:
        print(f"ERROR: prelabel already exists in {preseg_dir} (use --force)", file=sys.stderr)
        return 1

    # Cheap header check BEFORE the multi-GB load: a prelabel for a cloud over
    # the label cap is unusable (voxa refuses to load it for labeling), and the
    # load itself would OOM on a 100M+ cloud.
    n_header = ply_vertex_count(ply_path)
    if n_header > MAX_LABEL_POINTS:
        print(f"ERROR: {ply_path} has {n_header:,} points > label cap "
              f"{MAX_LABEL_POINTS:,}.\n       Voxa can't load this for labeling, so a "
              f"prelabel would be unusable. Downsample source/scan.ply first "
              f"(see docs/point-cloud-sizing.md).", file=sys.stderr)
        return 4

    from scenes.point_cloud import load_ply
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
    if cap == 0:
        # Full-res by default, but don't exceed what RAM can handle.
        ceiling = _ram_safe_ceiling()
        if n > ceiling:
            cap = ceiling
            print(f"[preseg] {n:,} points > RAM-safe full-res ceiling "
                  f"{ceiling:,} (~{BYTES_PER_POINT*n/1e9:.0f} GB needed) — "
                  f"subsampling to {ceiling:,} + NN-propagate. Pass "
                  f"--preseg-points to override.")
    if cap and n > cap:
        rng = np.random.default_rng(args.seed)
        sub = np.sort(rng.choice(n, cap, replace=False))
        print(f"[preseg] {n:,} > {cap:,} cap — preseg subsample then NN-propagate")
        inst_sub, summary = presegment(
            xyz[sub], class_map=class_map,
            features=features[sub], feature_seen=seen[sub], log=print,
        )
        from scipy.spatial import cKDTree
        print("[preseg] propagating via nearest-neighbour…")
        _, nn = cKDTree(xyz[sub]).query(xyz, k=1, workers=-1)
        inst = inst_sub[nn]
    else:
        print(f"[preseg] ransac mode, {len(class_map)} classes, feature-aware split")
        inst, summary = presegment(
            xyz, class_map=class_map,
            features=features, feature_seen=seen, log=print,
        )

    publish_preseg(scan_dir, args.preseg_id, inst, summary,
                   generator="sam3",
                   params={"preseg_points": args.preseg_points, "seed": args.seed})
    n_assigned = int((inst >= 0).sum())
    print(f"\n[done] wrote {preseg_dir}/instance_ids.npy")
    print(f"[done] wrote {preseg_dir}/segment_summary.json")
    print(f"[done] {n_assigned:,}/{n:,} assigned across {len(summary)} segments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
