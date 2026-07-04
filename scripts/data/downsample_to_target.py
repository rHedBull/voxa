"""Voxel-downsample a PLY to a target point count.

Binary-searches voxel size until the result is within tolerance of the target.
Spatial uniformity preserved (better for RANSAC + SAM3 than random subsampling).

Usage:
    .venv/bin/python scripts/downsample_to_target.py <input.ply> <output.ply> <target_M>

Example:
    .venv/bin/python scripts/downsample_to_target.py \
        scan.raw_full_156M.ply scan.ply 3.0
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import open3d as o3d


def voxel_downsample_to(pcd: o3d.geometry.PointCloud, target: int,
                        tol: float = 0.03, max_iters: int = 20) -> tuple[o3d.geometry.PointCloud, float]:
    """Binary-search voxel size so |result - target| <= tol * target."""
    bb = pcd.get_axis_aligned_bounding_box()
    diag = float(np.linalg.norm(bb.max_bound - bb.min_bound))
    lo, hi = diag / 5000.0, diag / 20.0  # generous bracket
    best = None
    for i in range(max_iters):
        mid = (lo + hi) / 2.0
        ds = pcd.voxel_down_sample(voxel_size=mid)
        n = len(ds.points)
        rel = abs(n - target) / target
        print(f"  iter {i:2d}: voxel={mid:.5f}  n={n:>12,d}  rel={rel*100:5.2f}%")
        if best is None or rel < best[1]:
            best = (ds, rel, mid)
        if rel <= tol:
            return ds, mid
        if n > target:
            lo = mid  # bigger voxels -> fewer points
        else:
            hi = mid
    print(f"  binary search exhausted; using closest: rel={best[1]*100:.2f}%")
    return best[0], best[2]


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("inp", type=Path)
    p.add_argument("out", type=Path)
    p.add_argument("target_M", type=float, help="target point count in millions")
    args = p.parse_args()

    target = int(round(args.target_M * 1_000_000))
    print(f"input : {args.inp}")
    print(f"output: {args.out}")
    print(f"target: {target:,} points")

    t0 = time.time()
    print("loading…")
    pcd = o3d.io.read_point_cloud(str(args.inp))
    n_in = len(pcd.points)
    print(f"  loaded {n_in:,} points in {time.time()-t0:.1f}s")
    if n_in <= target:
        print(f"  input already <= target; copying unchanged")
        o3d.io.write_point_cloud(str(args.out), pcd, write_ascii=False, compressed=False)
        return 0

    print("voxel-downsampling…")
    t1 = time.time()
    ds, vox = voxel_downsample_to(pcd, target)
    n_out = len(ds.points)
    print(f"  result: {n_out:,} points (voxel={vox:.5f}) in {time.time()-t1:.1f}s")

    print("writing…")
    t2 = time.time()
    o3d.io.write_point_cloud(str(args.out), ds, write_ascii=False, compressed=False)
    print(f"  wrote {args.out} in {time.time()-t2:.1f}s")
    print(f"total: {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
