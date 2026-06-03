"""Dry-run: SAM3 with broad prompts → 3D instance map (no class taxonomy).

For each frame we ask SAM3 about a small set of broad noun phrases
("object", "thing", "part", ...) and keep *every* returned mask as its
own instance with a random color. The 3D PLY shows what SAM3 considers
distinct entities without us imposing a label vocabulary.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from scenes.reproject import (  # noqa: E402
    ORIENTATION_PRESETS, look_at_view, project_points, depth_buffer_mask,
)
from sam3_common import load_ply, build_processor, segment, gather_frames  # noqa: E402
from ply_viz import random_palette  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True, type=Path, action="append",
                    help="renders dir (can be passed multiple times)")
    ap.add_argument("--ply", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = all")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--orientation", default="Z+",
                    choices=list(ORIENTATION_PRESETS.keys()))
    ap.add_argument("--prompts", default="object,thing,part,component",
                    help="comma-separated broad prompts")
    ap.add_argument("--min-area-frac", type=float, default=0.001)
    ap.add_argument("--debug-overlays", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.out / "overlays"
    if args.debug_overlays:
        overlay_dir.mkdir(exist_ok=True)

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    print(f"[1/4] Loading PLY {args.ply}")
    pts, _ = load_ply(args.ply, orientation=args.orientation)
    N = pts.shape[0]
    print(f"      {N:,} points  prompts={prompts}")

    all_frames = gather_frames(args.renders, args.stride, args.max_frames)
    print(f"[2/4] Using {len(all_frames)} frames")

    print(f"[3/4] Loading SAM3 on {args.device}…")
    proc = build_processor(args.device)

    point_inst = np.full(N, -1, dtype=np.int32)
    point_depth = np.full(N, np.inf, dtype=np.float32)
    inst_meta: list[dict] = []  # one entry per instance: prompt, score, frame

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
        vis_z = z[vis_idx]

        # Collect all (mask, score, prompt) triples for this frame
        all_masks: list[np.ndarray] = []
        all_scores: list[float] = []
        all_prompts: list[str] = []
        min_area = int(args.min_area_frac * W * H)
        for prompt in prompts:
            m, s = segment(proc, pil, prompt)
            for k in range(m.shape[0]):
                if int(m[k].sum()) < min_area:
                    continue
                all_masks.append(m[k])
                all_scores.append(float(s[k]))
                all_prompts.append(prompt)

        palette = random_palette(len(all_masks), seed=fi * 1000 + 1)
        print(f"  frame {fi+1}/{len(all_frames)} {rd.name}/{frame['file']}: "
              f"{len(all_masks)} instances, {vis_idx.size:,} visible pts, "
              f"{time.time()-t0:.1f}s")

        for mi, mk in enumerate(all_masks):
            hit = mk[vis_v, vis_u]
            if not hit.any():
                continue
            cand_idx = vis_idx[hit]
            cand_z = vis_z[hit]
            take = cand_z < point_depth[cand_idx] - 0.05
            sel = cand_idx[take]
            new_id = len(inst_meta)
            point_inst[sel] = new_id
            point_depth[sel] = cand_z[take]
            inst_meta.append({
                "id": new_id, "frame": fi, "prompt": all_prompts[mi],
                "score": all_scores[mi], "rgb": palette[mi].tolist(),
                "n_3d_pts": int(sel.size),
            })

        if args.debug_overlays:
            ov = np.array(pil).astype(np.float32)
            for mi, mk in enumerate(all_masks):
                col = palette[mi].astype(np.float32)
                a = mk.astype(np.float32)[:, :, None] * 0.5
                ov = ov * (1 - a) + col[None, None, :] * a
            Image.fromarray(np.clip(ov, 0, 255).astype(np.uint8)).save(
                overlay_dir / f"{fi:03d}_{frame['file']}"
            )

    print("[4/4] Writing PLY")
    colors = np.full((N, 3), 60, dtype=np.uint8)
    if inst_meta:
        palette_arr = np.array([m["rgb"] for m in inst_meta], dtype=np.uint8)
        has = point_inst >= 0
        colors[has] = palette_arr[point_inst[has]]
    rec = np.empty(N, dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("instance", "i4"),
    ])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["red"], rec["green"], rec["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    rec["instance"] = point_inst
    out_ply = args.out / "labeled.ply"
    PlyData([PlyElement.describe(rec, "vertex")], text=False).write(str(out_ply))

    summary = {
        "n_points": int(N),
        "n_labeled": int((point_inst >= 0).sum()),
        "n_unlabeled": int((point_inst < 0).sum()),
        "n_instances": len(inst_meta),
        "frames_used": len(all_frames),
        "by_prompt": {
            p: sum(1 for m in inst_meta if m["prompt"] == p) for p in prompts
        },
        "top_instances_by_size": sorted(
            inst_meta, key=lambda m: -m["n_3d_pts"]
        )[:15],
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: summary[k] for k in
                      ("n_labeled", "n_instances", "by_prompt")}, indent=2))
    print("Wrote", out_ply)


if __name__ == "__main__":
    main()
