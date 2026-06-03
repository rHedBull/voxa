"""
Dry-run: SAM3 text-prompted segmentation → 3D point labels.

Pipeline per frame:
  1. Run SAM3 with text prompts (pipe, tank, structural, equipment).
  2. Build view+projection matrices from manifest pose (Three.js conventions).
  3. Project all 3D points to pixels, depth-test against a sparse z-buffer.
  4. For visible points, look up class mask → cast vote with score weight.
Across frames: argmax vote → per-point class label. Save colored PLY.

This is intentionally standalone (no voxa imports). Run from anaconda base
where sam3 is installed.

Usage:
  python scripts/dry_sam3/text_prompt_class_voting.py \
    --renders /home/hendrik/coding/engine/product/walker/robot-patrol-sim/renders/smart_ais/orbit01__orbit__ultra__20260512-084754 \
    --ply /home/hendrik/coding/engine/data/lidar/annotated/smart_ais/source/scan.ply \
    --out /tmp/sam3_dry \
    --max-frames 6 --stride 10
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "backend"))
from scenes.reproject import (  # noqa: E402
    ORIENTATION_PRESETS, look_at_view, project_points, depth_buffer_mask,
)
from sam3_common import build_processor, segment, union_mask, load_ply  # noqa: E402


CLASSES = [
    # (prompt, label_id, rgb)
    ("pipe", 1, (255, 80, 80)),
    ("storage tank", 2, (80, 200, 255)),
    ("steel beam", 3, (255, 220, 60)),
    ("industrial equipment", 4, (160, 255, 120)),
]
BG = (60, 60, 60)


def write_labeled_ply(path: Path, pts: np.ndarray, labels: np.ndarray, votes: np.ndarray):
    palette = {0: BG}
    for _, lid, c in CLASSES:
        palette[lid] = c
    colors = np.array([palette[int(l)] for l in labels], dtype=np.uint8)
    # Fade unconfident points toward gray
    conf = votes.max(axis=1) / (votes.sum(axis=1) + 1e-6)
    fade = np.clip(conf, 0.0, 1.0)[:, None]
    colors = (colors * fade + np.array(BG)[None, :] * (1 - fade)).astype(np.uint8)
    rec = np.empty(pts.shape[0], dtype=[
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ("label", "i2"),
    ])
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["red"], rec["green"], rec["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    rec["label"] = labels.astype(np.int16)
    PlyData([PlyElement.describe(rec, "vertex")], text=False).write(str(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--renders", required=True, type=Path, action="append",
                    help="renders dir (can be passed multiple times)")
    ap.add_argument("--ply", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-frames", type=int, default=0,
                    help="cap per renders dir (0 = all)")
    ap.add_argument("--stride", type=int, default=1, help="sample every Nth frame")
    ap.add_argument("--merge", default="",
                    help="comma-separated 'a=b' rules to merge class a into b after voting")
    ap.add_argument("--fov", type=float, default=60.0, help="vertical FOV (deg)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--orientation", default="Z+",
                    choices=list(ORIENTATION_PRESETS.keys()),
                    help="mesh orientation preset used at render time")
    ap.add_argument("--debug-overlays", action="store_true")
    ap.add_argument("--prompts", default=None,
                    help="comma-separated noun phrases overriding the default CLASSES list")
    args = ap.parse_args()

    global CLASSES
    if args.prompts:
        names = [p.strip() for p in args.prompts.split(",") if p.strip()]
        palette = [(255, 80, 80), (80, 200, 255), (255, 220, 60),
                   (160, 255, 120), (200, 120, 255), (255, 160, 60),
                   (120, 255, 220), (220, 100, 160)]
        CLASSES = [(n, i + 1, palette[i % len(palette)]) for i, n in enumerate(names)]
        print("Prompts:", [c[0] for c in CLASSES])

    args.out.mkdir(parents=True, exist_ok=True)
    overlay_dir = args.out / "overlays"
    if args.debug_overlays:
        overlay_dir.mkdir(exist_ok=True)

    print(f"[1/4] Loading PLY {args.ply}")
    pts, rgb = load_ply(args.ply, orientation=args.orientation)
    N = pts.shape[0]
    print(f"      {N:,} points  bbox=({pts.min(0)}) → ({pts.max(0)})")

    all_frames: list[tuple[Path, dict]] = []
    for renders_dir in args.renders:
        m = json.loads((renders_dir / "manifest.json").read_text())
        picked = m["frames"][::args.stride]
        if args.max_frames > 0:
            picked = picked[: args.max_frames]
        # drop stub frames (small dead PNGs)
        picked = [f for f in picked
                  if (renders_dir / f["file"]).exists()
                  and (renders_dir / f["file"]).stat().st_size > 50_000]
        print(f"  + {renders_dir.name}: {len(picked)} frames")
        all_frames.extend((renders_dir, f) for f in picked)
    print(f"[2/4] Using {len(all_frames)} frames total")

    print(f"[3/4] Loading SAM3 on {args.device}…")
    proc = build_processor(args.device)
    print("      ready")

    votes = np.zeros((N, len(CLASSES) + 1), dtype=np.float32)  # col 0 = bg
    seen = np.zeros(N, dtype=np.int32)

    for fi, (renders_dir, frame) in enumerate(all_frames):
        t0 = time.time()
        img_path = renders_dir / frame["file"]
        if not img_path.exists():
            print(f"  skip {frame['file']}: missing"); continue
        pil = Image.open(img_path).convert("RGB")
        W, H = pil.size

        # camera
        pos = np.array(frame["position"], dtype=np.float64)
        if "target" in frame:
            tgt = np.array(frame["target"], dtype=np.float64)
        else:
            # route: yaw around +Y, look forward in XZ
            yaw = float(frame["yaw"])
            fwd = np.array([np.cos(yaw), 0.0, np.sin(yaw)])
            tgt = pos + fwd
        view = look_at_view(pos, tgt)

        u, v, z, in_front = project_points(pts, view, args.fov, W, H)
        vis_idx, vis_u, vis_v = depth_buffer_mask(u, v, z, in_front, W, H)
        seen[vis_idx] += 1

        # SAM3 per class
        per_class_score = {}  # label_id -> [H,W] float32
        for prompt, lid, _ in CLASSES:
            masks, scores = segment(proc, pil, prompt)
            best = union_mask(masks, scores)
            if best is not None:
                per_class_score[lid] = best

        # cast votes
        for lid, score_map in per_class_score.items():
            s = score_map[vis_v, vis_u]
            votes[vis_idx, lid] += s

        if args.debug_overlays:
            ov = np.array(pil).astype(np.float32)
            for lid, score_map in per_class_score.items():
                color = np.array(dict((c[1], c[2]) for c in CLASSES)[lid], dtype=np.float32)
                a = (score_map > 0.3).astype(np.float32)[:, :, None] * 0.45
                ov = ov * (1 - a) + color[None, None, :] * a
            Image.fromarray(np.clip(ov, 0, 255).astype(np.uint8)).save(
                overlay_dir / f"{fi:03d}_{frame['file']}"
            )

        n_masks = sum(int((m > 0.3).any()) for m in per_class_score.values())
        print(f"  frame {fi+1}/{len(all_frames)} {renders_dir.name}/{frame['file']}: "
              f"{vis_idx.size:,} visible pts, {n_masks}/{len(CLASSES)} classes hit, "
              f"{time.time()-t0:.1f}s")

    print("[4/4] Aggregating votes")
    # Optional class merges: vote of `a` is added into `b` before argmax.
    name_to_id = {c[0]: c[1] for c in CLASSES}
    for rule in [r for r in args.merge.split(",") if r.strip()]:
        a, b = rule.split("=")
        a, b = a.strip(), b.strip()
        if a in name_to_id and b in name_to_id:
            votes[:, name_to_id[b]] += votes[:, name_to_id[a]]
            votes[:, name_to_id[a]] = 0.0
            print(f"  merged '{a}' → '{b}'")
    # background gets a small constant so points with no vote stay 0
    votes[:, 0] = 0.05
    labels = votes.argmax(axis=1).astype(np.int16)
    labels[seen == 0] = -1

    out_ply = args.out / "labeled.ply"
    write_labeled_ply(out_ply, pts, np.maximum(labels, 0), votes)
    summary = {
        "n_points": int(N),
        "n_seen": int((seen > 0).sum()),
        "per_class_counts": {
            CLASSES[i][0]: int((labels == CLASSES[i][1]).sum()) for i in range(len(CLASSES))
        },
        "n_bg": int((labels == 0).sum()),
        "n_unseen": int((labels == -1).sum()),
        "frames_used": len(all_frames),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print("Done:", out_ply)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
