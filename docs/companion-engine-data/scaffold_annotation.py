#!/usr/bin/env python3
"""Scaffold an annotation directory under lidar/annotated/<scan_name>/.

Reads a LAZ source, voxel-downsamples to ~target points (uniform spatial
density — better for labeling than stride sampling), writes:

  source/scan.ply        binary PLY (xyz + rgb) at the sampled density
  prelabel/              empty directory (v2 preseg pipeline fills this)
  meta.json              provenance + class_map_version + schema_version: "2.0"
  README.md              minimal stub

The result conforms to lidar/SCHEMA.md (v2) and is immediately discoverable by
voxa v2. No labels/ directory is created — GT lives in sessions/<id>/output/
once a labeling session saves.

The preseg pipeline (scripts/preseg/presegment*.py) populates prelabel/ via
register_preseg() after scaffolding. Until that runs, voxa opens the scan in
blank-session mode.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import laspy
import open3d as o3d

CLASS_MAP_PATH = Path("lidar/classes.json")
SCHEMA_VERSION = "2.0"


def voxel_downsample_to_target(xyz: np.ndarray, rgb: np.ndarray, target: int,
                               start_voxel: float = 0.05) -> tuple[np.ndarray, np.ndarray, float]:
    """Voxel downsample, then random-thin to exact target.

    Returns (xyz_out, rgb_out, voxel_size_used).
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    pcd.colors = o3d.utility.Vector3dVector(rgb.astype(np.float64) / 65535.0)

    # Iterate voxel size until count is within [target, 4*target).
    voxel = start_voxel
    down = pcd.voxel_down_sample(voxel)
    for _ in range(8):
        n = len(down.points)
        if n < target:
            voxel *= 0.7
        elif n > 4 * target:
            voxel *= 1.4
        else:
            break
        down = pcd.voxel_down_sample(voxel)

    xyz_d = np.asarray(down.points)
    rgb_d = (np.asarray(down.colors) * 65535).astype(np.uint16)

    # Random subsample to exact target if still over.
    if len(xyz_d) > target:
        rng = np.random.default_rng(seed=42)
        idx = rng.choice(len(xyz_d), size=target, replace=False)
        idx.sort()
        xyz_d = xyz_d[idx]
        rgb_d = rgb_d[idx]

    return xyz_d, rgb_d, voxel


def write_ply_binary(path: Path, xyz: np.ndarray, rgb_u16: np.ndarray) -> None:
    """Binary little-endian PLY: float32 xyz + uchar rgb (LAS uint16 high byte)."""
    n = xyz.shape[0]
    rgb_u8 = (rgb_u16 >> 8).astype(np.uint8)
    rec = np.empty(n, dtype=np.dtype([
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("r", "u1"), ("g", "u1"), ("b", "u1"),
    ]))
    rec["x"] = xyz[:, 0].astype(np.float32)
    rec["y"] = xyz[:, 1].astype(np.float32)
    rec["z"] = xyz[:, 2].astype(np.float32)
    rec["r"] = rgb_u8[:, 0]
    rec["g"] = rgb_u8[:, 1]
    rec["b"] = rgb_u8[:, 2]
    with open(path, "wb") as f:
        f.write(
            f"ply\n"
            f"format binary_little_endian 1.0\n"
            f"element vertex {n}\n"
            f"property float x\n"
            f"property float y\n"
            f"property float z\n"
            f"property uchar red\n"
            f"property uchar green\n"
            f"property uchar blue\n"
            f"end_header\n".encode()
        )
        f.write(rec.tobytes())


def scaffold(laz_path: Path, scan_name: str, out_root: Path, target: int) -> None:
    out = out_root / scan_name
    if (out / "source/scan.ply").exists():
        print(f"skip (exists): {scan_name}", flush=True)
        return

    t0 = time.time()
    classes = json.loads(CLASS_MAP_PATH.read_text())
    class_map_version = classes["version"]

    print(f"[{scan_name}] reading {laz_path.name}", flush=True)
    las = laspy.read(str(laz_path))
    xyz_src = np.column_stack([np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)]).astype(np.float64)
    rgb_src = np.column_stack([np.asarray(las.red), np.asarray(las.green), np.asarray(las.blue)]).astype(np.uint16)
    n_src = xyz_src.shape[0]
    print(f"[{scan_name}] source pts: {n_src:,}", flush=True)

    xyz, rgb, voxel = voxel_downsample_to_target(xyz_src, rgb_src, target)
    n = xyz.shape[0]
    print(f"[{scan_name}] sampled to {n:,} pts (voxel={voxel:.3f} m)", flush=True)

    # Recenter to local frame if any axis lives far from the origin.
    # PLY xyz is float32; at UTM magnitudes (~1e6) precision drops to ~6 cm,
    # collapsing points and destroying detail. Round offset to integer meters
    # so it's trivial to reverse with coord_offset_m from meta.json.
    center = xyz.mean(axis=0)
    if np.max(np.abs(center)) > 1_000.0:
        coord_offset_m = np.round(center).astype(np.float64)
        xyz = xyz - coord_offset_m
        print(f"[{scan_name}] recentered by {coord_offset_m.tolist()} (world coords too large for float32 PLY)", flush=True)
    else:
        coord_offset_m = np.zeros(3, dtype=np.float64)

    (out / "source").mkdir(parents=True, exist_ok=True)
    # Empty prelabel/ dir — the preseg pipeline populates it via register_preseg().
    # Its presence (even empty) suppresses the "no prelabel" log noise in voxa.
    (out / "prelabel").mkdir(parents=True, exist_ok=True)

    write_ply_binary(out / "source/scan.ply", xyz, rgb)

    # No labels/ directory in v2 — GT lives in sessions/<id>/output/ once a
    # labeling session saves. Opening in voxa creates the first session.

    meta = {
        "schema_version": SCHEMA_VERSION,
        "scan_name": scan_name,
        "source_laz": f"lidar/laz/{laz_path.name}",
        "source_mesh": None,
        "n_points": int(n),
        "sample_method": "voxel",
        "sample_param": {
            "target_points": target,
            "voxel_size_m": float(voxel),
            "thinning": "random_seed_42 if voxel-result exceeded target",
            "tool": "open3d.PointCloud.voxel_down_sample",
        },
        "coords": "world_minus_offset" if np.any(coord_offset_m) else "world",
        "coord_offset_m": coord_offset_m.tolist(),
        "units": "meters",
        "class_map_version": class_map_version,
        "capture_date": None,
        "scanner": None,
        "notes": "Unlabeled stub. No labeling sessions yet. Run the preseg pipeline, then open in voxa.",
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    readme = (
        f"# {scan_name}\n\n"
        f"Scaffold from `{laz_path.name}` ({n_src:,} source pts) → "
        f"{n:,} pts at voxel {voxel:.3f} m.\n\n"
        f"**Status: unlabeled.** No labeling sessions exist yet. "
        f"Conforms to `lidar/SCHEMA.md` (v2).\n\n"
        f"## Next steps\n\n"
        f"1. Run auto-prelabeling (RANSAC etc.) → populates `prelabel/<preseg_id>/`.\n"
        f"2. Open in voxa → create a labeling session seeded from the preseg.\n"
        f"3. Label, then Ctrl+S → writes `sessions/<id>/output/`.\n"
    )
    (out / "README.md").write_text(readme)

    print(f"[{scan_name}] scaffolded in {time.time()-t0:.1f}s -> {out}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("laz", nargs="+", type=Path)
    ap.add_argument("--out-root", type=Path, default=Path("lidar/annotated"))
    ap.add_argument("--target", type=int, default=500_000)
    ap.add_argument("--name-from-stem", action="store_true",
                    help="Use LAZ stem as scan_name; default uses lowercased+underscored stem.")
    args = ap.parse_args()

    if not CLASS_MAP_PATH.exists():
        print(f"!! missing class map at {CLASS_MAP_PATH}", file=sys.stderr)
        return 1

    for laz in args.laz:
        if not laz.exists():
            print(f"!! missing: {laz}", flush=True)
            continue
        scan_name = laz.stem if args.name_from_stem else (
            laz.stem.lower().replace("-", "_").replace(" ", "_")
        )
        try:
            scaffold(laz, scan_name, args.out_root, args.target)
        except Exception as exc:
            print(f"!! {laz.name} failed: {exc}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
