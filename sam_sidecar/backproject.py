"""Mask + raw depth-buffer → visible scan-res point indices."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import minimum_filter
from reproject import project_points

def select_in_mask(scan_xyz, view, fov_y, W, H, mask, depth,
                   tol_rel=0.01, tol_abs=0.15, splat=2):
    u, v, z, infront = project_points(scan_xyz, view, fov_y, W, H)
    ui = u.astype(np.int64); vi = v.astype(np.int64)
    ok = infront & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    idx = np.where(ok)[0]
    ui, vi, zz = ui[ok], vi[ok], z[ok].astype(np.float32)
    zb = minimum_filter(depth, size=2 * splat + 1, mode="nearest") if splat else depth
    z_at = zb[vi, ui]
    tol = np.maximum(tol_abs, tol_rel * np.where(np.isinf(z_at), 0.0, z_at))
    visible = zz <= z_at + tol            # inf background → never visible (zz finite)
    in_mask = mask[vi, ui]
    return idx[visible & in_mask].astype(np.int32)
