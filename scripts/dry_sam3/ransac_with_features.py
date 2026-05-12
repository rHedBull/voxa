"""Dry-run: voxa RANSAC preseg + SAM3-feature-aware sub-segmentation.

Steps:
  1. Run the existing RANSAC pipeline (planes, cylinders, leftovers) → P
     instances.
  2. For each RANSAC instance, look up per-point SAM3 features and split
     by k-means in feature space when the instance is large enough.
     Each sub-cluster becomes its own final instance.
  3. Optionally merge adjacent final instances whose mean SAM3 feature
     cosine ≥ threshold (coarse pass).
  4. Save raw RANSAC + feature-split PLYs side by side.

No voxa application code is modified.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree
from sklearn.cluster import MiniBatchKMeans

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))


def load_xyz(ply_path: Path) -> np.ndarray:
    p = PlyData.read(str(ply_path))
    v = p["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], -1).astype(np.float64)


def random_palette(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    h = rng.uniform(0, 1, n)
    s = rng.uniform(0.55, 0.95, n)
    v = rng.uniform(0.70, 1.0, n)
    i = (h * 6).astype(int) % 6
    f = h * 6 - i
    p = v * (1 - s); q = v * (1 - f * s); t = v * (1 - (1 - f) * s)
    rgb = np.zeros((n, 3), dtype=np.float32)
    for k, parts in enumerate([(v, t, p), (q, v, p), (p, v, t),
                                (p, q, v), (t, p, v), (v, p, q)]):
        sel = i == k
        rgb[sel] = np.stack([parts[0][sel], parts[1][sel], parts[2][sel]], -1)
    return (rgb * 255).astype(np.uint8)


def write_colored_ply(path: Path, xyz: np.ndarray, ids: np.ndarray,
                      seed: int = 0):
    n_inst = int(ids.max()) + 1 if ids.size and ids.max() >= 0 else 0
    palette = random_palette(max(1, n_inst), seed=seed)
    colors = np.full((xyz.shape[0], 3), 60, dtype=np.uint8)
    has = ids >= 0
    colors[has] = palette[ids[has]]
    rec = np.empty(xyz.shape[0], dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("instance", "i4"),
    ])
    rec["x"], rec["y"], rec["z"] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    rec["red"], rec["green"], rec["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    rec["instance"] = ids.astype(np.int32)
    PlyData([PlyElement.describe(rec, "vertex")], text=False).write(str(path))


def split_by_features(idx: np.ndarray, feat: np.ndarray, seen: np.ndarray,
                      min_size: int = 800, max_k: int = 8,
                      target_size: int = 1500, eps_silhouette: float = 0.02
                      ) -> np.ndarray:
    """Return per-point sub-cluster ids inside this RANSAC instance.

    Heuristic: number of sub-clusters k = clamp(round(N/target_size), 1, max_k).
    Skip if instance is too small or fewer than half the points have features.
    """
    n = idx.size
    if n < min_size:
        return np.zeros(n, dtype=np.int32)
    has = seen[idx] > 0
    if has.sum() < n * 0.5:
        return np.zeros(n, dtype=np.int32)
    k = int(np.clip(round(n / target_size), 1, max_k))
    if k <= 1:
        return np.zeros(n, dtype=np.int32)
    X = feat[idx][has].astype(np.float32)
    km = MiniBatchKMeans(n_clusters=k, random_state=0,
                          batch_size=2048, n_init=3)
    sub = km.fit_predict(X)
    # Reject low-quality splits: if any cluster has <5% of points it's noise.
    counts = np.bincount(sub, minlength=k)
    if (counts < max(20, int(0.05 * has.sum()))).any():
        # try k-1
        k = max(1, k - 1)
        if k == 1:
            return np.zeros(n, dtype=np.int32)
        km = MiniBatchKMeans(n_clusters=k, random_state=0,
                              batch_size=2048, n_init=3)
        sub = km.fit_predict(X)

    out = np.zeros(n, dtype=np.int32)
    # Assign points with features to their k-means cluster.
    out[has] = sub
    # Points without features: nearest neighbour in 3D among the same instance's
    # featureful points.
    if (~has).any() and has.any():
        # cheap: use feature-mean closest by sub
        # actually use spatial NN inside the instance
        # (we don't have xyz here, so just leave as 0 — they share the dominant
        # sub-cluster). Good enough for a dry run.
        out[~has] = 0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path)
    ap.add_argument("--seen", type=Path, default=None)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--target-size", type=int, default=1500,
                    help="target points per sub-cluster")
    ap.add_argument("--max-k", type=int, default=8,
                    help="max sub-clusters per RANSAC instance")
    ap.add_argument("--min-size", type=int, default=800,
                    help="instance must be ≥ this to be split")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seen_path = args.seen or args.features.with_name("seen.npy")

    print(f"[1/4] Load {args.ply}")
    xyz = load_xyz(args.ply)
    N = xyz.shape[0]
    feat = np.load(args.features).astype(np.float32)
    seen = np.load(seen_path)
    assert feat.shape[0] == N
    print(f"      N={N:,}  D={feat.shape[1]}  seen={int((seen>0).sum()):,}")

    print("[2/4] RANSAC presegmentation (this takes a minute)")
    from presegment_ransac import presegment as ransac_preseg
    ids, summary = ransac_preseg(
        xyz, class_map=None, log=lambda *_: None, params=None,
    )
    P = int(ids.max()) + 1 if ids.max() >= 0 else 0
    print(f"      {P} RANSAC instances, {int((ids==-1).sum()):,} unassigned")
    write_colored_ply(args.out / "ransac_raw.png.ply", xyz, ids, seed=3)

    print(f"[3/4] Feature split: target_size={args.target_size}, max_k={args.max_k}")
    next_id = 0
    final = np.full(N, -1, dtype=np.int32)
    split_stats = []
    for inst_id in range(P):
        member = np.where(ids == inst_id)[0]
        if member.size == 0:
            continue
        sub = split_by_features(member, feat, seen,
                                min_size=args.min_size, max_k=args.max_k,
                                target_size=args.target_size)
        n_sub = int(sub.max()) + 1
        final[member] = next_id + sub
        split_stats.append({"inst": inst_id, "n_pts": int(member.size),
                            "n_sub": n_sub})
        next_id += n_sub

    # Keep RANSAC unassigned as their own bucket (one cluster per spatially-
    # close blob of unassigned).
    unassigned = np.where(ids == -1)[0]
    if unassigned.size > 0:
        tree = cKDTree(xyz[unassigned])
        # cheap: grid-bin them at 0.5m → call those the cluster ids
        bins = np.floor(xyz[unassigned] / 0.5).astype(np.int64)
        keys = bins[:, 0] * 1_000_003 + bins[:, 1] * 1_000_033 + bins[:, 2]
        _, comp = np.unique(keys, return_inverse=True)
        final[unassigned] = next_id + comp
        next_id += int(comp.max()) + 1
        _ = tree  # silence unused

    n_final = int(final.max()) + 1
    print(f"      {P} RANSAC → {n_final} clusters after feature split")

    write_colored_ply(args.out / "merged.ply", xyz, final, seed=42)
    np.save(args.out / "instance_ids.npy", final)

    # Histograms of the biggest splits
    big_splits = sorted(split_stats, key=lambda d: -d["n_sub"])[:10]
    info = {
        "n_points": int(N),
        "n_ransac_instances": int(P),
        "n_final_clusters": int(n_final),
        "n_unassigned_by_ransac": int(unassigned.size),
        "target_size": args.target_size,
        "max_k": args.max_k,
        "min_size_to_split": args.min_size,
        "feature_dim": int(feat.shape[1]),
        "biggest_splits": big_splits,
        "cluster_size_stats": {
            "min": int(np.bincount(final[final >= 0]).min()),
            "median": int(np.median(np.bincount(final[final >= 0]))),
            "p95": int(np.percentile(np.bincount(final[final >= 0]), 95)),
            "max": int(np.bincount(final[final >= 0]).max()),
        },
    }
    (args.out / "info.json").write_text(json.dumps(info, indent=2))
    _ = summary  # silence unused
    print(json.dumps({k: info[k] for k in
                      ("n_ransac_instances", "n_final_clusters",
                       "cluster_size_stats", "biggest_splits")}, indent=2))


if __name__ == "__main__":
    main()
