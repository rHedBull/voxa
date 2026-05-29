"""SAM3 per-point image-feature extraction for presegmentation.

Given a point cloud (xyz, N×3) and one or more rendered-image directories
with a ``manifest.json`` describing per-frame camera pose, this module:

  1. Runs the SAM3 image encoder once per frame, grabbing the highest-
     resolution FPN feature map (C × H' × W').
  2. Projects every 3D point to image space using the manifest pose +
     Three.js perspective conventions (vertical FOV = 60°, Y-up).
  3. Uses a splatted z-buffer for occlusion, bilinear-samples the
     feature map at each visible point's pixel coords.
  4. Averages across frames, L2-normalizes, optional PCA to a smaller dim.

Results are cached under ``data/preseg_runs/<scene>/sam3_features.npz``
keyed by a content hash of the input render dirs + parameters.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from scenes.fingerprint import cloud_fingerprint


# Render discovery is per-scene-directory now: renders live under
# `<scan_dir>/renders/<run_id>/` per lidar/SCHEMA.md v1.2. The
# VOXA_RENDERS_ROOT env + DEFAULT_RENDERS_ROOT path are retained only as
# a transitional back-stop for callers that pass an explicit scene name
# (e.g. the /api/sam3/renders HTTP surface) — they're ignored when a
# scan_dir is provided directly. Remove once no caller relies on env.
RENDERS_ROOT_ENV = "VOXA_RENDERS_ROOT"
DEFAULT_RENDERS_ROOT = Path(
    "/home/hendrik/coding/engine/product/walker/robot-patrol-sim/renders"
)
ORIENTATION_PRESETS = {
    "Y+": (0.0, 0.0, 0.0),
    "Z+": (-np.pi / 2, 0.0, 0.0),
    "X+": (0.0, 0.0, np.pi / 2),
    "Y-": (np.pi, 0.0, 0.0),
    "Z-": (np.pi / 2, 0.0, 0.0),
    "X-": (0.0, 0.0, -np.pi / 2),
}


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

@dataclass
class RenderRun:
    path: Path
    name: str
    scene: str
    n_frames: int
    has_orbit_target: bool
    mtime: float


def renders_root() -> Path:
    env = os.environ.get(RENDERS_ROOT_ENV)
    return Path(env) if env else DEFAULT_RENDERS_ROOT


def discover_render_runs(scene_name: str,
                         root: Optional[Path] = None,
                         scan_dir: Optional[Path] = None) -> list[RenderRun]:
    """List render runs for a scene.

    Preferred: pass ``scan_dir`` — runs are discovered under
    ``<scan_dir>/renders/<run>/manifest.json`` (lidar/SCHEMA.md v1.2).
    Legacy: when ``scan_dir`` is None, falls back to
    ``<root or renders_root()>/<scene_name>/`` for callers that haven't
    moved to the per-scan layout yet.
    """
    if scan_dir is not None:
        base = Path(scan_dir) / "renders"
    else:
        base = (root or renders_root()) / scene_name
    if not base.exists():
        return []
    runs: list[RenderRun] = []
    for sub in base.iterdir():
        if not sub.is_dir():
            continue
        mani = sub / "manifest.json"
        if not mani.exists():
            continue
        try:
            m = json.loads(mani.read_text())
            frames = m.get("frames", [])
            has_target = bool(frames) and "target" in frames[0]
            runs.append(RenderRun(
                path=sub,
                name=sub.name,
                scene=m.get("scene", scene_name),
                n_frames=len(frames),
                has_orbit_target=has_target,
                mtime=mani.stat().st_mtime,
            ))
        except (json.JSONDecodeError, OSError):
            continue
    runs.sort(key=lambda r: -r.mtime)
    return runs


# ---------------------------------------------------------------------------
# Camera math (Three.js perspective, Y-up)
# ---------------------------------------------------------------------------

def _euler_xyz_matrix(rx: float, ry: float, rz: float) -> np.ndarray:
    cx, sx = np.cos(rx), np.sin(rx)
    cy, sy = np.cos(ry), np.sin(ry)
    cz, sz = np.cos(rz), np.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def _look_at_view(pos: np.ndarray, target: np.ndarray,
                  up=(0.0, 1.0, 0.0)) -> np.ndarray:
    f = target - pos
    f = f / (np.linalg.norm(f) + 1e-12)
    u = np.array(up, dtype=np.float64)
    s = np.cross(f, u)
    s /= (np.linalg.norm(s) + 1e-12)
    u2 = np.cross(s, f)
    R = np.stack([s, u2, -f], axis=0)
    M = np.eye(4)
    M[:3, :3] = R
    M[:3, 3] = -R @ pos
    return M


def _project_points(pts_world: np.ndarray, view: np.ndarray,
                    fov_y_deg: float, W: int, H: int):
    N = pts_world.shape[0]
    homo = np.concatenate([pts_world, np.ones((N, 1))], axis=1)
    cam = (view @ homo.T).T[:, :3]
    z = -cam[:, 2]
    in_front = z > 0.05
    fy = (H / 2.0) / np.tan(np.deg2rad(fov_y_deg) / 2.0)
    fx = fy
    u = (cam[:, 0] * fx) / np.maximum(z, 1e-6) + W / 2.0
    v = (-cam[:, 1] * fy) / np.maximum(z, 1e-6) + H / 2.0
    return u, v, z, in_front


def _depth_buffer_mask(u, v, z, in_front, W, H,
                       tol_rel=0.01, tol_abs=0.15,
                       splat_radius=2, max_depth=80.0):
    valid = (in_front & (u >= 0) & (u < W) & (v >= 0) & (v < H)
             & (z < max_depth))
    ui = u[valid].astype(np.int32)
    vi = v[valid].astype(np.int32)
    zi = z[valid].astype(np.float32)
    idx = np.where(valid)[0]
    zbuf = np.full((H, W), np.inf, dtype=np.float32)
    for dv in range(-splat_radius, splat_radius + 1):
        for du in range(-splat_radius, splat_radius + 1):
            uu = np.clip(ui + du, 0, W - 1)
            vv = np.clip(vi + dv, 0, H - 1)
            np.minimum.at(zbuf, (vv, uu), zi)
    z_at = zbuf[vi, ui]
    tol = np.maximum(tol_abs, tol_rel * z_at)
    visible = zi <= (z_at + tol)
    return idx[visible], ui[visible], vi[visible]


# ---------------------------------------------------------------------------
# Cache key
# ---------------------------------------------------------------------------

def _cache_key(render_dirs: list[Path], source_fingerprint: str, fpn_level: int,
               pca_dim: int, orientation: str, fov: float) -> str:
    h = hashlib.sha256()
    for rd in sorted(str(p.resolve()) for p in render_dirs):
        h.update(rd.encode())
        try:
            h.update(str((Path(rd) / "manifest.json").stat().st_mtime).encode())
        except OSError:
            pass
    # Key on cloud CONTENT (fingerprint), not point count — a recentered cloud
    # with the same n_points must not silently reuse a stale cache (v1.3 §4.5).
    h.update(f"fp={source_fingerprint}|fpn={fpn_level}|pca={pca_dim}|"
             f"orient={orientation}|fov={fov}".encode())
    return h.hexdigest()[:16]


def _scene_cache_path(scene_id: str, cache_dir: Path) -> Path:
    safe = scene_id.replace("/", "__").replace("\\", "__")
    return cache_dir / safe / "sam3_features.npz"


def load_cache(scene_id: str, key: str, cache_dir: Path):
    path = _scene_cache_path(scene_id, cache_dir)
    if not path.exists():
        return None
    try:
        z = np.load(path)
        meta = json.loads(str(z["meta_json"]))
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return None
    if meta.get("cache_key") != key:
        return None
    return z["features"], z["seen"], meta


def _save_cache(scene_id: str, cache_dir: Path,
                features: np.ndarray, seen: np.ndarray, meta: dict):
    path = _scene_cache_path(scene_id, cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        features=features,
        seen=seen,
        meta_json=json.dumps(meta),
    )


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def extract_or_load(
    xyz: np.ndarray,
    scene_id: str,
    *,
    render_dirs: list[Path],
    cache_dir: Path,
    fpn_level: int = 0,
    pca_dim: int = 64,
    orientation: str = "Z+",
    fov: float = 60.0,
    device: str = "cuda",
    force: bool = False,
    log: Callable[[str], None] = print,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not render_dirs:
        raise ValueError("render_dirs must be non-empty")
    n = int(xyz.shape[0])
    source_fp = cloud_fingerprint(xyz)
    key = _cache_key(render_dirs, source_fp, fpn_level, pca_dim, orientation, fov)

    if not force:
        cached = load_cache(scene_id, key, cache_dir)
        if cached is not None:
            log(f"SAM3 features: cache HIT (key={key})")
            features, seen, meta = cached
            return features, seen, meta
    log(f"SAM3 features: cache MISS (key={key}) — computing")

    rx, ry, rz = ORIENTATION_PRESETS[orientation]
    R = _euler_xyz_matrix(rx, ry, rz)
    pts_rot = xyz.astype(np.float64) @ R.T

    import torch
    import torch.nn.functional as F
    from PIL import Image
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    bpe = "/home/hendrik/anaconda3/lib/python3.12/site-packages/clip/bpe_simple_vocab_16e6.txt.gz"
    log("Loading SAM3 image model…")
    model = build_sam3_image_model(device=device, load_from_HF=True, bpe_path=bpe)
    proc = Sam3Processor(model, device=device)

    frames: list[tuple[Path, dict]] = []
    for rd in render_dirs:
        m = json.loads((rd / "manifest.json").read_text())
        for f in m.get("frames", []):
            p = rd / f["file"]
            if p.exists() and p.stat().st_size > 50_000:
                frames.append((rd, f))
    log(f"  {len(frames)} usable frames across {len(render_dirs)} run(s)")

    sum_feat: Optional[torch.Tensor] = None
    seen = np.zeros(n, dtype=np.int32)
    D = None
    feat_h = feat_w = None
    dev = torch.device(device)
    t0 = time.time()

    for fi, (rd, frame) in enumerate(frames):
        img = Image.open(rd / frame["file"]).convert("RGB")
        W, H = img.size
        pos = np.array(frame["position"], dtype=np.float64)
        if "target" in frame:
            tgt = np.array(frame["target"], dtype=np.float64)
        else:
            yaw = float(frame.get("yaw", 0.0))
            tgt = pos + np.array([np.cos(yaw), 0.0, np.sin(yaw)])
        view = _look_at_view(pos, tgt)
        u, v, z, in_front = _project_points(pts_rot, view, fov, W, H)
        vis_idx, vis_u, vis_v = _depth_buffer_mask(u, v, z, in_front, W, H)
        if vis_idx.size == 0:
            continue

        state = proc.set_image(img)
        feat = state["backbone_out"]["backbone_fpn"][fpn_level]
        if feat.dim() == 3:
            feat = feat.unsqueeze(0)
        _, C, h, w = feat.shape
        if sum_feat is None:
            D = int(C)
            feat_h, feat_w = int(h), int(w)
            sum_feat = torch.zeros((n, D), dtype=torch.float32, device=dev)

        gx = (vis_u.astype(np.float32) + 0.5) / W * 2.0 - 1.0
        gy = (vis_v.astype(np.float32) + 0.5) / H * 2.0 - 1.0
        grid = torch.from_numpy(np.stack([gx, gy], -1)[None, None, ...]).to(dev)
        sampled = F.grid_sample(
            feat.float(), grid, mode="bilinear", align_corners=False,
            padding_mode="border",
        )[0, :, 0, :].T
        sum_feat.index_add_(0, torch.from_numpy(vis_idx).to(dev).long(), sampled)
        seen[vis_idx] += 1
        if (fi + 1) % 10 == 0:
            log(f"  frame {fi+1}/{len(frames)}  elapsed={time.time()-t0:.1f}s")

    if sum_feat is None or D is None:
        raise RuntimeError("no usable frames")

    seen_t = torch.from_numpy(seen).to(dev).clamp(min=1).unsqueeze(1).float()
    mean = sum_feat / seen_t
    mean = torch.nn.functional.normalize(mean, dim=1, eps=1e-6)
    final = mean.cpu().numpy().astype(np.float32)

    pca_dim_used = D
    if pca_dim > 0 and pca_dim < D:
        seen_mask = seen > 0
        X = final[seen_mask]
        mu = X.mean(0, keepdims=True)
        Xc = X - mu
        if Xc.shape[0] > 60_000:
            sel = np.random.default_rng(0).choice(
                Xc.shape[0], 60_000, replace=False
            )
            _, _, Vt = np.linalg.svd(Xc[sel], full_matrices=False)
        else:
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Vt[:pca_dim]
        final = (final - mu) @ comps.T
        pca_dim_used = int(pca_dim)
        log(f"  PCA {D} → {pca_dim_used}")

    features = final.astype(np.float16)
    meta = {
        "cache_key": key,
        "source_fingerprint": source_fp,
        "n_points": int(n),
        "feature_dim": int(pca_dim_used),
        "encoder_dim": int(D),
        "feat_map_hw": [feat_h, feat_w],
        "fpn_level": int(fpn_level),
        "n_frames": int(len(frames)),
        "n_seen": int((seen > 0).sum()),
        "orientation": orientation,
        "fov": float(fov),
        "render_dirs": [str(p) for p in render_dirs],
        "computed_at": time.time(),
    }
    _save_cache(scene_id, cache_dir, features, seen, meta)
    log(f"SAM3 features done: {features.shape} in "
        f"{time.time()-t0:.1f}s, seen={meta['n_seen']:,}/{n:,}")
    return features, seen, meta
