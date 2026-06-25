"""SAM3 text-query presegmentation: one image mask = one cloud segment.

Queries SAM3 with a text prompt (default ``pipe``) on every render frame and
projects each per-image *instance* mask onto the cloud. Unlike the feature
pipeline (``presegment_sam3_features.py`` + ``presegment_sam3.py``) there is no
cross-view fusion: every 2D instance mask becomes its own 3D segment, and when a
point falls inside masks from several frames the **latest frame wins** (runs are
processed oldest -> newest, frames in manifest order; within one frame,
overlapping masks resolve by score — highest wins). ``--conflict first`` inverts
the policy: the first mask to claim a point keeps it and later masks only take
still-unassigned points. Points never hit by a mask stay unassigned (-1).

Publishes straight into ``prelabel/<preseg_id>/`` (scan-schema v2), so the
result is pickable when creating a labeling session. Single stage: no Open3D /
RANSAC involved, so the whole thing runs under the **anaconda base** python
(torch + sam3), NOT voxa's .venv:

    /home/hendrik/anaconda3/bin/python scripts/preseg/presegment_sam3_query.py \
        /home/hendrik/coding/engine/data/lidar/annotated/<scan>
"""
from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "scripts" / "dry_sam3"))

from scenes.reproject import (  # noqa: E402
    ORIENTATION_PRESETS, euler_xyz_matrix, look_at_view, project_points,
    depth_buffer_mask,
)
from app.constants import MAX_LABEL_POINTS  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import classes_from_yaml, ply_vertex_count, publish_preseg  # noqa: E402

# SAM3 helpers shared with the dry_sam3 experiments (torch imported lazily).
from sam3_common import build_processor, segment, gather_frames  # noqa: E402


def _sanitize(name: str) -> str:
    s = "".join(c if (c.isalnum() or c in "_-") else "_" for c in name.strip().lower())
    return s or "q"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", type=Path, help="annotated/<scan> directory")
    ap.add_argument("--query", default="pipe",
                    help="SAM3 text prompt; `a|b` unions synonym prompts (default: pipe)")
    ap.add_argument("--class-name", default="",
                    help="classes.yaml class for all segments (default: first query token)")
    ap.add_argument("--preseg-id", default="",
                    help="prelabel/<id>/ to publish (default: sam3_query_<query>)")
    ap.add_argument("--min-points", type=int, default=20,
                    help="drop image masks that hit fewer visible points (default: 20)")
    ap.add_argument("--conflict", choices=["latest", "first"], default="latest",
                    help="multi-view overlap policy: 'latest' = later frames overwrite "
                         "(default); 'first' = the first mask keeps the point, later "
                         "masks only claim still-unassigned points")
    ap.add_argument("--runs", default="",
                    help="comma list of substrings; only render runs whose dir name "
                         "matches one are used (default: all runs)")
    ap.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    ap.add_argument("--max-frames", type=int, default=0, help="cap total frames (0 = all)")
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--orientation", default="Z+", choices=list(ORIENTATION_PRESETS.keys()))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing prelabel/<preseg_id>/")
    ap.add_argument("--skip-registration-check", action="store_true",
                    help="bypass the cloud<->renders registration health-check (not recommended)")
    ap.add_argument("--config", type=Path, default=ROOT / "config" / "classes.yaml")
    args = ap.parse_args()

    scan_dir: Path = args.scan_dir.resolve()
    ply_path = scan_dir / "source" / "scan.ply"
    if not ply_path.exists():
        print(f"ERROR: no source/scan.ply in {scan_dir}", file=sys.stderr)
        return 2

    prompts = [p.strip() for p in args.query.split("|") if p.strip()]
    if not prompts:
        print("ERROR: empty --query", file=sys.stderr)
        return 2
    class_name = (args.class_name or prompts[0]).lower()
    preseg_id = args.preseg_id or f"sam3_query_{_sanitize(args.query)}"

    class_map = classes_from_yaml(args.config)
    if class_name not in class_map:
        print(f"ERROR: class '{class_name}' not in {args.config} "
              f"(have: {', '.join(class_map)})", file=sys.stderr)
        return 2
    class_id = class_map[class_name]

    from scan_schema.layout import ScanLayout
    preseg_dir = ScanLayout(scan_dir).preseg_dir(preseg_id)
    if preseg_dir.is_dir() and any(preseg_dir.iterdir()) and not args.force:
        print(f"ERROR: prelabel already exists in {preseg_dir} (use --force)",
              file=sys.stderr)
        return 1

    n_header = ply_vertex_count(ply_path)
    if n_header > MAX_LABEL_POINTS:
        print(f"ERROR: {ply_path} has {n_header:,} points > label cap "
              f"{MAX_LABEL_POINTS:,} — a prelabel would be unusable. Downsample "
              f"source/scan.ply first (see docs/point-cloud-sizing.md).", file=sys.stderr)
        return 4

    # Registration health-check (v1.3 §6): refuse to spend SAM3 compute when the
    # cloud doesn't project into the renders' poses (the navvis frame-mismatch bug).
    if not args.skip_registration_check:
        from preseg.registration import verify_scan_registration
        verdict = verify_scan_registration(scan_dir, orientation=args.orientation)
        if verdict["checked"]:
            for r in verdict["runs"]:
                ph = f"{r['photometric']:.1%}" if r["photometric"] is not None else "n/a"
                print(f"[reg-check] {r['run_id']}: coverage {r['coverage']:.1%}, "
                      f"photometric {ph}")
            if not verdict["ok"]:
                print("ERROR: cloud does not register to its renders:", file=sys.stderr)
                for reason in verdict["reasons"]:
                    print(f"  - {reason}", file=sys.stderr)
                print("  (re-run with --skip-registration-check to override)",
                      file=sys.stderr)
                return 6

    from scenes.point_cloud import load_ply
    print(f"[load] {ply_path}")
    pc, _ = load_ply(ply_path)
    xyz = np.asarray(pc.points, dtype=np.float64)
    n = int(xyz.shape[0])
    print(f"[load] {n:,} points | query={'|'.join(prompts)} -> class "
          f"'{class_name}' (id {class_id}) | preseg_id={preseg_id}")

    R = euler_xyz_matrix(*ORIENTATION_PRESETS[args.orientation])
    pts_rot = xyz @ R.T

    # v1.3 §5 frame-aware remap, fail-closed on cross-scan/unpinned runs.
    from preseg.resolver import dir_cloud_transforms
    from scan_schema.fingerprint import cloud_fingerprint
    from scan_schema.frame import apply_transform
    from scan_schema.metadata import read_scan_meta
    sm = read_scan_meta(scan_dir)
    from preseg.sam3_features import discover_render_runs
    runs = discover_render_runs(scan_dir.name, scan_dir=scan_dir)
    run_filters = [t.strip() for t in args.runs.split(",") if t.strip()]
    if run_filters:
        runs = [r for r in runs if any(t in r.name for t in run_filters)]
    # Oldest run first so the newest render run's masks win the overwrites.
    render_dirs = [r.path for r in sorted(runs, key=lambda r: r.mtime)]
    if not render_dirs:
        print(f"ERROR: no render runs with manifest.json under {scan_dir / 'renders'}"
              + (f" matching --runs {args.runs!r}" if run_filters else ""),
              file=sys.stderr)
        return 3
    dir_T = dir_cloud_transforms(render_dirs, sm["frame"],
                                 sm["derivation"]["variant_id"],
                                 cloud_fingerprint(xyz), R)
    n_remapped = sum(1 for t in dir_T.values() if t is not None)

    pts_cache: dict = {}

    def pts_for(rd):
        T = dir_T.get(rd)
        if T is None:
            return pts_rot
        if rd not in pts_cache:
            pts_cache[rd] = apply_transform(T, pts_rot)
        return pts_cache[rd]

    frames = gather_frames(render_dirs, args.stride, args.max_frames)
    if not frames:
        print("ERROR: no usable frames", file=sys.stderr)
        return 3
    print(f"[frames] {len(frames)} frames across {len(render_dirs)} run(s), "
          f"remapping {n_remapped}")

    print(f"[sam3] loading model on {args.device}…")
    proc = build_processor(args.device)

    from PIL import Image
    inst = np.full(n, -1, dtype=np.int32)
    segments: list[dict] = []
    next_id = 0
    t0 = time.time()
    for fi, (rd, frame) in enumerate(frames):
        pil = Image.open(rd / frame["file"]).convert("RGB")
        W, H = pil.size
        pos = np.array(frame["position"], dtype=np.float64)
        if "target" in frame:
            tgt = np.array(frame["target"], dtype=np.float64)
        else:
            yaw = float(frame.get("yaw", 0.0))
            tgt = pos + np.array([np.cos(yaw), 0.0, np.sin(yaw)])
        u, v, z, in_front = project_points(pts_for(rd), look_at_view(pos, tgt),
                                           args.fov, W, H)
        vis_idx, vis_u, vis_v = depth_buffer_mask(u, v, z, in_front, W, H)
        if vis_idx.size == 0:
            continue

        masks, scores = [], []
        for prompt in prompts:
            m, s = segment(proc, pil, prompt)
            masks.extend(m)
            scores.extend(s)
        # latest: ascending score — higher-scoring masks overwrite within the frame.
        # first: descending score — higher-scoring masks claim free points first.
        order = np.argsort(np.asarray(scores)) if scores else np.zeros(0, dtype=int)
        if args.conflict == "first":
            order = order[::-1]
        n_new = 0
        for k in order:
            sel = vis_idx[masks[k][vis_v, vis_u]]
            if args.conflict == "first":
                sel = sel[inst[sel] == -1]
            if sel.size < args.min_points:
                continue
            inst[sel] = next_id
            segments.append({"id": next_id, "class_id": class_id, "label": class_name,
                             "frame": f"{rd.name}/{frame['file']}",
                             "score": float(scores[k]), "n_at_assign": int(sel.size)})
            next_id += 1
            n_new += 1
        print(f"  frame {fi+1}/{len(frames)} {rd.name}/{frame['file']}: "
              f"{vis_idx.size:,} vis, {len(masks)} masks, {n_new} segments "
              f"({time.time()-t0:.0f}s)")

    # Later frames can fully overwrite earlier segments — drop the empty ones.
    counts = np.bincount(inst[inst >= 0], minlength=next_id) if next_id else np.zeros(0)
    segments = [s for s in segments if counts[s["id"]] > 0]
    publish_preseg(scan_dir, preseg_id, inst, segments,
                   generator="sam3_query",
                   params={"query": args.query, "class_name": class_name,
                           "min_points": args.min_points, "stride": args.stride,
                           "fov": args.fov, "orientation": args.orientation,
                           "frames_used": len(frames),
                           "conflict": f"{args.conflict}_frame_wins"})
    n_assigned = int((inst >= 0).sum())
    print(f"\n[done] wrote {preseg_dir}/instance_ids.npy")
    print(f"[done] {n_assigned:,}/{n:,} points assigned across "
          f"{len(segments)} segments ({next_id - len(segments)} fully overwritten)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
