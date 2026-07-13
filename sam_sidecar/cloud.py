"""Cloud loaders for the sidecar. Raw LAZ (cached .npz) + scan.ply."""
from __future__ import annotations
from pathlib import Path
import numpy as np

def load_raw(laz_path: str, cache_dir: str = ".") -> tuple[np.ndarray, np.ndarray]:
    """Full-res raw cloud (native frame). Returns (xyz float32[N,3], rgb uint8[N,3])."""
    cache = Path(cache_dir) / (Path(laz_path).stem + ".rawcache.npz")
    if cache.exists():
        d = np.load(cache); return d["xyz"], d["rgb"]
    import laspy
    xyz_parts, rgb_parts = [], []
    with laspy.open(laz_path) as fh:
        for ch in fh.chunk_iterator(6_000_000):
            xyz_parts.append(np.column_stack([ch.x, ch.y, ch.z]).astype(np.float32))
            r = (np.asarray(ch.red, np.uint32) >> 8).astype(np.uint8)
            g = (np.asarray(ch.green, np.uint32) >> 8).astype(np.uint8)
            b = (np.asarray(ch.blue, np.uint32) >> 8).astype(np.uint8)
            rgb_parts.append(np.stack([r, g, b], 1))
    xyz = np.concatenate(xyz_parts); rgb = np.concatenate(rgb_parts)
    np.savez(cache, xyz=xyz, rgb=rgb)
    return xyz, rgb

def load_scan_ply(ply_path: str) -> np.ndarray:
    """scan.ply xyz in native frame, in file order (indices must match voxa's session)."""
    from plyfile import PlyData
    v = PlyData.read(ply_path)["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
