"""Dry-run: voxa voxel preseg + SAM3-feature-aware merge.

Pipeline (all offline, no voxa code changes):
  1. Load PLY, run voxa's voxel supervoxel preseg → N points → S supervoxels.
  2. Load SAM3 per-point features (from extract_features.py).
  3. Mean-pool features per supervoxel.
  4. Build supervoxel adjacency in 3D (kNN on supervoxel centroids).
  5. Greedy union-find merge of adjacent supervoxels whose mean features
     have cosine similarity above a threshold AND whose centroids are
     within a spatial cap.
  6. Save three PLYs: raw supervoxels, SAM3-merged, and a side-by-side
     bbox-tagged comparison.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial import cKDTree

# Make voxa backend importable.
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
    n_inst = int(ids.max()) + 1 if ids.max() >= 0 else 0
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


class UnionFind:
    def __init__(self, n: int):
        self.p = np.arange(n, dtype=np.int32)
        self.rank = np.zeros(n, dtype=np.int32)

    def find(self, x: int) -> int:
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.p[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ply", required=True, type=Path)
    ap.add_argument("--features", required=True, type=Path,
                    help="features.npy from extract_features.py")
    ap.add_argument("--seen", type=Path, default=None,
                    help="seen.npy; defaults to <features dir>/seen.npy")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--voxel", type=float, default=0.30,
                    help="voxel preseg resolution (m)")
    ap.add_argument("--knn", type=int, default=6,
                    help="supervoxel neighbors in adjacency graph")
    ap.add_argument("--max-merge-dist", type=float, default=0.8,
                    help="cap on supervoxel-centroid distance for a merge edge (m)")
    ap.add_argument("--cos-thresh", type=float, default=0.85,
                    help="cosine-sim threshold to merge two supervoxels")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    seen_path = args.seen or args.features.with_name("seen.npy")

    print(f"[1/6] Load {args.ply}")
    xyz = load_xyz(args.ply)
    N = xyz.shape[0]
    print(f"      {N:,} points")

    print(f"[2/6] Voxel preseg @ {args.voxel}m")
    from presegment_voxel import presegment as voxel_preseg
    sv_ids, summary = voxel_preseg(xyz, log=lambda *_: None,
                                    resolution=args.voxel)
    S = int(sv_ids.max()) + 1
    print(f"      {S:,} supervoxels  (median size "
          f"{int(np.median(np.bincount(sv_ids[sv_ids >= 0]))):,})")
    write_colored_ply(args.out / "supervoxels.ply", xyz, sv_ids, seed=1)

    print(f"[3/6] Load features {args.features}")
    feat = np.load(args.features).astype(np.float32)
    seen = np.load(seen_path)
    assert feat.shape[0] == N, f"feature length {feat.shape[0]} ≠ N={N}"
    D = feat.shape[1]
    print(f"      features (N={feat.shape[0]:,}, D={D}); "
          f"seen={int((seen>0).sum()):,}")

    print("[4/6] Mean-pool features and centroids per supervoxel")
    sv_feat = np.zeros((S, D), dtype=np.float32)
    sv_cnt = np.zeros(S, dtype=np.int32)
    sv_centroid = np.zeros((S, 3), dtype=np.float32)
    sv_pcnt = np.zeros(S, dtype=np.int32)
    valid = (sv_ids >= 0) & (seen > 0)
    np.add.at(sv_feat, sv_ids[valid], feat[valid])
    np.add.at(sv_cnt, sv_ids[valid], 1)
    # centroid uses all points
    np.add.at(sv_centroid, sv_ids[sv_ids >= 0], xyz[sv_ids >= 0].astype(np.float32))
    np.add.at(sv_pcnt, sv_ids[sv_ids >= 0], 1)
    has_feat = sv_cnt > 0
    sv_feat[has_feat] = sv_feat[has_feat] / sv_cnt[has_feat, None]
    sv_centroid = sv_centroid / np.maximum(sv_pcnt[:, None], 1)
    # L2 normalize for cosine
    norms = np.linalg.norm(sv_feat, axis=1, keepdims=True)
    sv_feat = sv_feat / np.maximum(norms, 1e-6)
    print(f"      supervoxels with features: {int(has_feat.sum()):,}/{S:,}")

    print(f"[5/6] Build kNN adjacency (k={args.knn}, max d={args.max_merge_dist}m)")
    tree = cKDTree(sv_centroid)
    dists, neighbors = tree.query(sv_centroid, k=args.knn + 1)
    # drop self
    dists = dists[:, 1:]
    neighbors = neighbors[:, 1:]
    edge_a = np.repeat(np.arange(S), args.knn)
    edge_b = neighbors.ravel()
    edge_d = dists.ravel()
    keep = (edge_d <= args.max_merge_dist) & (edge_a < edge_b)
    edge_a, edge_b, edge_d = edge_a[keep], edge_b[keep], edge_d[keep]
    print(f"      candidate edges: {len(edge_a):,}")

    print(f"[6/6] Greedy merge (cos ≥ {args.cos_thresh})")
    uf = UnionFind(S)
    cos = (sv_feat[edge_a] * sv_feat[edge_b]).sum(axis=1)
    both_have_feat = has_feat[edge_a] & has_feat[edge_b]
    merge_edge = both_have_feat & (cos >= args.cos_thresh)
    # also merge edges where one side has no feature, if very close in space
    spatial_merge = (~both_have_feat) & (edge_d < args.voxel * 1.2)
    do_merge = merge_edge | spatial_merge
    n_merged = 0
    for a, b in zip(edge_a[do_merge], edge_b[do_merge]):
        if uf.find(int(a)) != uf.find(int(b)):
            uf.union(int(a), int(b))
            n_merged += 1
    print(f"      merged {n_merged:,} edges → "
          f"{len(set(int(uf.find(i)) for i in range(S))):,} clusters")

    # Map supervoxel → cluster, then point → cluster
    root = np.array([uf.find(i) for i in range(S)], dtype=np.int32)
    # compact root ids
    _, comp_ids = np.unique(root, return_inverse=True)
    sv_to_cluster = comp_ids.astype(np.int32)
    point_cluster = np.full(N, -1, dtype=np.int32)
    point_cluster[sv_ids >= 0] = sv_to_cluster[sv_ids[sv_ids >= 0]]

    write_colored_ply(args.out / "merged.ply", xyz, point_cluster, seed=42)
    np.save(args.out / "instance_ids.npy", point_cluster)

    info = {
        "n_points": int(N),
        "n_supervoxels": int(S),
        "n_supervoxels_with_features": int(has_feat.sum()),
        "n_merged_edges": int(n_merged),
        "n_final_clusters": int(comp_ids.max() + 1),
        "voxel_size": float(args.voxel),
        "knn": int(args.knn),
        "max_merge_dist": float(args.max_merge_dist),
        "cos_thresh": float(args.cos_thresh),
        "feature_dim": int(D),
        "cluster_size_stats": {
            "min": int(np.bincount(point_cluster[point_cluster >= 0]).min()),
            "median": int(np.median(np.bincount(point_cluster[point_cluster >= 0]))),
            "p95": int(np.percentile(np.bincount(point_cluster[point_cluster >= 0]), 95)),
            "max": int(np.bincount(point_cluster[point_cluster >= 0]).max()),
        },
    }
    (args.out / "info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
