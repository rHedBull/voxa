"""Canonical point-cloud reprojection (Three.js perspective, Y-up).

The single home for the camera math used by BOTH the SAM3 feature pipeline and
the registration health-check (scan-schema v1.3 §6), so the check projects
exactly as the pipeline does. Pure numpy; no torch.

Convention: cameras look down -Z, up = +Y, square pixels, vertical FOV.
Callers that work with Z-up scans rotate the cloud first via
``euler_xyz_matrix(*ORIENTATION_PRESETS[orientation])`` (as ``extract_or_load``
does) before projecting.
"""
from __future__ import annotations

import numpy as np

ORIENTATION_PRESETS = {
    "Y+": (0.0, 0.0, 0.0),
    "Z+": (-np.pi / 2, 0.0, 0.0),
    "X+": (0.0, 0.0, np.pi / 2),
    "Y-": (np.pi, 0.0, 0.0),
    "Z-": (np.pi / 2, 0.0, 0.0),
    "X-": (0.0, 0.0, -np.pi / 2),
}


def euler_xyz_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    # Three.js Euler "XYZ" (intrinsic X→Y→Z) composes as Rx * Ry * Rz —
    # matching Matrix4.makeRotationFromEuler, which the viewer renders with.
    return Rx @ Ry @ Rz


def look_at_view(pos: np.ndarray, target: np.ndarray, up=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Three.js look-at: returns world->camera (camera looks down -Z)."""
    pos = np.asarray(pos, dtype=np.float64)
    f = np.asarray(target, dtype=np.float64) - pos
    f = f / (np.linalg.norm(f) + 1e-12)
    u = np.array(up, dtype=np.float64)
    s = np.cross(f, u)
    s /= (np.linalg.norm(s) + 1e-12)
    u2 = np.cross(s, f)
    R = np.stack([s, u2, -f], axis=0)  # 3x3
    t = -R @ pos
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


def project_points(pts_world: np.ndarray, view: np.ndarray, fov_y_deg: float,
                   W: int, H: int):
    """Returns (u, v, depth, in_front_mask). u,v in pixel coords (origin top-left)."""
    N = pts_world.shape[0]
    homo = np.concatenate([pts_world, np.ones((N, 1))], axis=1)
    cam = (view @ homo.T).T[:, :3]  # camera space, looks down -Z
    z = -cam[:, 2]  # depth in front of camera (positive)
    in_front = z > 0.05
    fy = (H / 2.0) / np.tan(np.deg2rad(fov_y_deg) / 2.0)
    fx = fy  # square pixels
    u = (cam[:, 0] * fx) / np.maximum(z, 1e-6) + W / 2.0
    v = (-cam[:, 1] * fy) / np.maximum(z, 1e-6) + H / 2.0
    return u, v, z, in_front


def depth_buffer_mask(u, v, z, in_front, W, H,
                      tol_rel=0.01, tol_abs=0.15,
                      splat_radius=2, max_depth=80.0):
    """Splatted z-buffer occlusion test.

    A point is 'visible' if its depth is within tol of the minimum depth observed
    in a small window around its pixel (splat fills gaps in sparse clouds).
    """
    valid = in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H) & (z < max_depth)
    ui = u[valid].astype(np.int32)
    vi = v[valid].astype(np.int32)
    zi = z[valid].astype(np.float32)
    idx = np.where(valid)[0]

    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    r = int(splat_radius)
    for dv in range(-r, r + 1):
        for du in range(-r, r + 1):
            uu = np.clip(ui + du, 0, W - 1)
            vv = np.clip(vi + dv, 0, H - 1)
            np.minimum.at(zbuf, (vv, uu), zi)

    z_at = zbuf[vi, ui]
    tol = np.maximum(tol_abs, tol_rel * z_at)
    visible = zi <= (z_at + tol)
    return idx[visible], ui[visible], vi[visible]
