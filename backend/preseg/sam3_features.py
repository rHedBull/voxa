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
# Camera/projection math is single-homed in scenes.reproject (the registration
# health-check and this pipeline MUST project identically); import, don't copy.
from scenes.reproject import (
    ORIENTATION_PRESETS, euler_xyz_matrix, look_at_view, project_points,
    depth_buffer_mask,
)


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
    cloud_frame=None,
    cloud_variant_id: Optional[str] = None,
    cloud_fingerprint_str: Optional[str] = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    if not render_dirs:
        raise ValueError("render_dirs must be non-empty")
    n = int(xyz.shape[0])
    source_fp = cloud_fingerprint(xyz)
    key = _cache_key(render_dirs, source_fp, fpn_level, pca_dim, orientation, fov)

    rx, ry, rz = ORIENTATION_PRESETS[orientation]
    R = euler_xyz_matrix(rx, ry, rz)

    # scan-schema v1.3 §5/§3b: if the caller passes the cloud's frame, resolve each
    # render run and remap the cloud into that run's pose frame before projecting.
    # Raises on a cross-scan/unpinned run (fail-closed). Folds the applied transforms
    # into the cache key so enabling remap invalidates a stale (no-remap) cache.
    dir_T: dict = {}
    if cloud_frame is not None:
        from preseg.resolver import dir_cloud_transforms
        dir_T = dir_cloud_transforms(render_dirs, cloud_frame,
                                     cloud_variant_id or scene_id,
                                     cloud_fingerprint_str or source_fp, R)
        applied = {rd: T for rd, T in dir_T.items() if T is not None}
        if applied:
            h = hashlib.sha256(key.encode())
            for rd in sorted(applied, key=lambda p: str(p)):
                h.update(np.asarray(applied[rd], dtype=np.float64).tobytes())
            key = h.hexdigest()[:16]
            log(f"  frame-aware: remapping {len(applied)}/{len(render_dirs)} render run(s)")

    if not force:
        cached = load_cache(scene_id, key, cache_dir)
        if cached is not None:
            log(f"SAM3 features: cache HIT (key={key})")
            features, seen, meta = cached
            return features, seen, meta
    log(f"SAM3 features: cache MISS (key={key}) — computing")

    pts_rot = xyz.astype(np.float64) @ R.T

    # Per-render-dir cloud, remapped into each run's pose frame (lazy + cached).
    _pts_by_dir: dict = {}

    def _pts_for(rd):
        T = dir_T.get(rd)
        if T is None:
            return pts_rot
        if rd not in _pts_by_dir:
            homo = np.concatenate([pts_rot, np.ones((pts_rot.shape[0], 1))], axis=1)
            _pts_by_dir[rd] = (T @ homo.T).T[:, :3]
        return _pts_by_dir[rd]

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
        view = look_at_view(pos, tgt)
        u, v, z, in_front = project_points(_pts_for(rd), view, fov, W, H)
        vis_idx, vis_u, vis_v = depth_buffer_mask(u, v, z, in_front, W, H)
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
