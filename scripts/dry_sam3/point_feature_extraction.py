"""Dry-run: per-3D-point SAM3 image features.

For each frame:
  1. Run SAM3 image encoder, grab the highest-resolution backbone feature
     map  F ∈ R^[C, H', W'].
  2. Project all 3D points to image space; depth-test visibility.
  3. Bilinear-sample F at each visible point's pixel → per-point feature
     contribution. Accumulate into a running mean.

Output:
  features.npy   (N, D) float16     — per-point mean feature (L2-normalized)
  seen.npy       (N,)   int32       — frame count each point was visible in
  meta.json                          — D, feature_scale, n_frames, prompt list
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from scenes.reproject import (  # noqa: E402
    ORIENTATION_PRESETS, look_at_view, project_points, depth_buffer_mask,
)
from sam3_common import load_ply, build_processor  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True, type=Path, action="append")
    ap.add_argument("--ply", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--orientation", default="Z+",
                    choices=list(ORIENTATION_PRESETS.keys()))
    ap.add_argument("--fpn-level", type=int, default=0,
                    help="which FPN level to sample (0 = highest-res)")
    ap.add_argument("--pca-dim", type=int, default=0,
                    help="if > 0, project final features to this dim with PCA")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading PLY {args.ply}")
    pts, _ = load_ply(args.ply, orientation=args.orientation)
    N = pts.shape[0]
    print(f"      {N:,} points")

    all_frames = []
    for rd in args.renders:
        m = json.loads((rd / "manifest.json").read_text())
        picked = m["frames"][::args.stride]
        if args.max_frames > 0:
            picked = picked[: args.max_frames]
        picked = [f for f in picked
                  if (rd / f["file"]).exists()
                  and (rd / f["file"]).stat().st_size > 50_000]
        print(f"  + {rd.name}: {len(picked)} frames")
        all_frames.extend((rd, f) for f in picked)
    print(f"[2/4] Using {len(all_frames)} frames")

    print(f"[3/4] Loading SAM3 on {args.device}…")
    proc = build_processor(args.device)

    sum_feat: torch.Tensor | None = None  # (N, D) float32 on GPU
    seen = np.zeros(N, dtype=np.int32)
    D = None
    feat_h = feat_w = None
    device = torch.device(args.device)

    for fi, (rd, frame) in enumerate(all_frames):
        t0 = time.time()
        img_path = rd / frame["file"]
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size
        pos = np.array(frame["position"], dtype=np.float64)
        tgt = (np.array(frame["target"], dtype=np.float64) if "target" in frame
               else pos + np.array([np.cos(frame["yaw"]), 0.0, np.sin(frame["yaw"])]))
        view = look_at_view(pos, tgt)
        u, v, z, in_front = project_points(pts, view, args.fov, W, H)
        vis_idx, vis_u, vis_v = depth_buffer_mask(u, v, z, in_front, W, H)
        if vis_idx.size == 0:
            print(f"  frame {fi+1}/{len(all_frames)}: 0 visible, skip")
            continue

        # Run encoder
        state = proc.set_image(pil)
        fpn = state["backbone_out"]["backbone_fpn"]
        feat = fpn[args.fpn_level]  # (1, C, h', w')
        if feat.dim() == 3:  # some FPNs return (C, h, w)
            feat = feat.unsqueeze(0)
        _, C, h, w = feat.shape

        if sum_feat is None:
            D = int(C)
            feat_h, feat_w = int(h), int(w)
            sum_feat = torch.zeros((N, D), dtype=torch.float32, device=device)
            print(f"      feature map: C={D}, {h}x{w} at level {args.fpn_level}")

        # Build grid_sample coords in [-1, 1] for the feature map's spatial dims.
        # Pixel coords are in the resized (1008x1008) input — but project_points (scenes.reproject)
        # projected to the *original* image WxH. The encoder resizes the input
        # to processor.resolution (square pad). To remain a dry test we assume
        # the SAM3 input is the same aspect/coverage as the PIL image; the
        # FPN spatial grid then corresponds directly to image (u, v) up to a
        # uniform scale → use (u/W, v/H) ∈ [0,1] regardless of resolution.
        gx = (vis_u.astype(np.float32) + 0.5) / W * 2.0 - 1.0
        gy = (vis_v.astype(np.float32) + 0.5) / H * 2.0 - 1.0
        grid = torch.from_numpy(np.stack([gx, gy], -1)[None, None, ...]).to(device)
        # grid shape: (1, 1, K, 2). Output: (1, C, 1, K)
        sampled = F.grid_sample(
            feat.float(), grid, mode="bilinear", align_corners=False,
            padding_mode="border",
        )
        sampled = sampled[0, :, 0, :].T  # (K, C)

        idx_t = torch.from_numpy(vis_idx).to(device).long()
        sum_feat.index_add_(0, idx_t, sampled)
        seen[vis_idx] += 1
        print(f"  frame {fi+1}/{len(all_frames)} {rd.name}/{frame['file']}: "
              f"{vis_idx.size:,} vis pts, {time.time()-t0:.1f}s")

    if sum_feat is None:
        raise SystemExit("no frames produced features")
    assert D is not None and feat_h is not None and feat_w is not None

    print("[4/4] Averaging + saving")
    seen_t = torch.from_numpy(seen).to(device).clamp(min=1).unsqueeze(1).float()
    mean_feat = sum_feat / seen_t
    # L2 normalize so that PCA / downstream cosine-sim is sensible.
    mean_feat = torch.nn.functional.normalize(mean_feat, dim=1, eps=1e-6)

    final = mean_feat.cpu().numpy().astype(np.float32)
    if args.pca_dim > 0 and args.pca_dim < D:
        # PCA over the *seen* points only.
        seen_mask = seen > 0
        X = final[seen_mask]
        mu = X.mean(0, keepdims=True)
        Xc = X - mu
        # SVD over a random subsample to keep memory bounded
        if Xc.shape[0] > 60000:
            sel = np.random.default_rng(0).choice(Xc.shape[0], 60000, replace=False)
            U, S, Vt = np.linalg.svd(Xc[sel], full_matrices=False)
        else:
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        comps = Vt[: args.pca_dim]  # (k, D)
        final = (final - mu) @ comps.T
        print(f"      PCA: {D} → {args.pca_dim}, var explained "
              f"≈ {(S[:args.pca_dim]**2).sum() / (S**2).sum():.3f}")
        D = args.pca_dim

    np.save(args.out / "features.npy", final.astype(np.float16))
    np.save(args.out / "seen.npy", seen.astype(np.int32))
    meta = {
        "n_points": int(N),
        "feature_dim": int(D),
        "feat_map_hw": [int(feat_h), int(feat_w)],
        "fpn_level": int(args.fpn_level),
        "n_frames": len(all_frames),
        "n_seen": int((seen > 0).sum()),
        "ply": str(args.ply),
        "orientation": args.orientation,
    }
    (args.out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    print("Wrote", args.out / "features.npy", "shape", final.shape, final.dtype)


if __name__ == "__main__":
    main()
