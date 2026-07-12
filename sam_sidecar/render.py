"""Perspective point-splat render → RGB image + depth-buffer. Pure numpy.
Validated at full 188M density (design session persp_raw.py)."""
from __future__ import annotations
import numpy as np

def render_view(xyz, rgb, view, fov_y, W, H, splat=2, bg=(0, 0, 0)):
    """xyz float[N,3] (native frame), rgb uint8[N,3], view=world->camera 4x4.
    Returns (color uint8[H,W,3], depth float32[H,W] with inf where empty)."""
    R = view[:3, :3].astype(np.float32); t = view[:3, 3].astype(np.float32)
    f = np.float32((H / 2.0) / np.tan(np.deg2rad(fov_y) / 2.0))
    cam = xyz.astype(np.float32) @ R.T + t
    z = -cam[:, 2]
    u = (cam[:, 0] * f) / np.maximum(z, 1e-6) + W / 2.0
    v = (-cam[:, 1] * f) / np.maximum(z, 1e-6) + H / 2.0
    ix = u.astype(np.int64); iy = v.astype(np.int64)
    ok = (z > 0.05) & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix, iy, z, rgb = ix[ok], iy[ok], z[ok].astype(np.float32), rgb[ok]
    order = np.argsort(-z, kind="stable")     # far first → nearest wins on last write
    ix, iy, z, rgb = ix[order], iy[order], z[order], rgb[order]
    color = np.full((H, W, 3), bg, np.uint8)
    depth = np.full((H, W), np.inf, np.float32)
    for dv in range(-splat, splat + 1):
        for du in range(-splat, splat + 1):
            xx = np.clip(ix + du, 0, W - 1); yy = np.clip(iy + dv, 0, H - 1)
            closer = z < depth[yy, xx]
            yc, xc = yy[closer], xx[closer]
            depth[yc, xc] = z[closer]; color[yc, xc] = rgb[closer]
    return color, depth
