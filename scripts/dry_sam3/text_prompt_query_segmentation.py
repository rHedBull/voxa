"""Dry-run: SAM3 prompt-driven segmentation → mask-membership PLY (query-independent).

Ask SAM3 for an arbitrary set of text queries (``floor``, ``wall``, ``pipe``, …)
and colour every 3D point whose projected pixel landed inside that query's mask.
Fast, re-runnable eyeball experiment. Generalises the old floor/wall-only script to
any number of queries; ``--queries floor,wall`` reproduces the original behaviour.

**Frame-aware (scan-schema v1.3):** if the scan has recorded frame metadata
(`meta.json` frame + `renders/<run>/meta.json`), each render run's poses are
resolved against the cloud and the cloud is **remapped into the run's pose frame**
before projecting — so points land on the real structures even when `scan.ply` is
in a different (e.g. unaligned) frame than the renders. `--no-remap` disables it.

Queries (`--queries`): comma-separated list. Each entry is ``name`` or
``name=prompt1|prompt2|…`` to union several SAM3 prompts (synonyms) into one class.
Colours auto-assign from a palette (floor→blue, wall→orange kept for back-compat);
override per class with ``--colors "pipe=#3cb44b,floor=#4287f5"``.

Assignment (`--assign`):
  - ``argmax`` (default): each visible pixel goes to whichever query's mask scores
    highest — mutually exclusive (fixes the case where one mask subsumes another).
  - ``union``: count every query independently (a pixel can belong to many).

Colour intensity encodes hit count (or ``--intensity fraction``); points hit by
multiple classes are resolved by ``--conflict`` (argmax-by-count / blend). Never-hit
grey, unseen darker.

Run from anaconda base (torch + sam3), NOT voxa's .venv.

Examples:
  python scripts/dry_sam3/text_prompt_query_segmentation.py \
    --scan /home/hendrik/coding/engine/data/lidar/annotated/navvis_vlx3_water_treatment \
    --queries floor,wall --out /tmp/sam3_floor_wall

  python scripts/dry_sam3/text_prompt_query_segmentation.py \
    --scan .../navvis_vlx3_water_treatment --queries pipe --out /tmp/sam3_pipe
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
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
from scenes.reproject import (  # noqa: E402
    ORIENTATION_PRESETS, look_at_view, project_points, depth_buffer_mask,
)
from sam3_common import load_ply, build_processor, segment, union_mask, gather_frames  # noqa: E402

# Frame-aware remap support (scan-schema v1.3).
try:
    from scenes.scan_meta import read_scan_meta  # noqa: E402
    from scenes.fingerprint import cloud_fingerprint  # noqa: E402
    from scenes.frame import apply_transform  # noqa: E402
    from scenes.reproject import euler_xyz_matrix  # noqa: E402
    from preseg.resolver import dir_cloud_transforms  # noqa: E402
    _V13 = True
except Exception:  # noqa: BLE001 — frame-awareness is optional
    _V13 = False

# Named colours kept for back-compat; everything else cycles the palette below.
NAMED_RGB = {
    "floor": (66, 135, 245),    # blue
    "wall": (245, 150, 56),     # orange
}
PALETTE = [
    (60, 200, 90),    # green
    (235, 70, 70),    # red
    (170, 110, 240),  # purple
    (70, 210, 210),   # cyan
    (240, 215, 60),   # yellow
    (240, 110, 200),  # pink
    (150, 180, 80),   # olive
    (120, 160, 250),  # periwinkle
]
BG_SEEN = np.array([45, 45, 45], dtype=np.float32)
BG_UNSEEN = np.array([22, 22, 22], dtype=np.float32)
MIN_BRIGHT = 0.25


def _sanitize(name: str) -> str:
    s = "".join(c if (c.isalnum() or c == "_") else "_" for c in name.strip().lower())
    return s or "q"


def _parse_hex(s: str):
    s = s.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError(f"bad hex colour '{s}'")
    return tuple(int(s[i:i + 2], 16) for i in (0, 2, 4))


def _parse_queries(spec: str, color_overrides: dict):
    """`floor,wall=wall|concrete wall,pipe` -> list of dicts {name, key, prompts, rgb}."""
    queries = []
    used_palette = 0
    seen_keys = set()
    for raw in spec.split(","):
        raw = raw.strip()
        if not raw:
            continue
        if "=" in raw:
            name, rest = raw.split("=", 1)
            prompts = [p.strip() for p in rest.split("|") if p.strip()]
        else:
            name, prompts = raw, [raw.strip()]
        name = name.strip()
        key = _sanitize(name)
        if key in seen_keys:
            raise ValueError(f"duplicate query name '{name}' (key '{key}')")
        seen_keys.add(key)
        if key in color_overrides:
            rgb = color_overrides[key]
        elif key in NAMED_RGB:
            rgb = NAMED_RGB[key]
        else:
            rgb = PALETTE[used_palette % len(PALETTE)]
            used_palette += 1
        queries.append({"name": name, "key": key,
                        "prompts": prompts, "rgb": np.array(rgb, dtype=np.float32)})
    if not queries:
        raise ValueError("no queries parsed from --queries")
    return queries


def _discover_runs(scan_dir: Path) -> list[Path]:
    base = scan_dir / "renders"
    if not base.exists():
        return []
    return sorted(d for d in base.iterdir()
                  if d.is_dir() and (d / "manifest.json").exists())


def _union_for_prompts(proc, pil, prompts):
    best = None
    for prompt in prompts:
        masks, scores = segment(proc, pil, prompt)
        u = union_mask(masks, scores)
        if u is None:
            continue
        best = u if best is None else np.maximum(best, u)
    return best


def _normalized_intensity(value, pct):
    pos = value[value > 0]
    if pos.size == 0:
        return np.zeros_like(value, dtype=np.float32)
    cap = np.percentile(pos, pct)
    if cap <= 0:
        cap = float(pos.max())
    return np.clip(value / cap, 0.0, 1.0).astype(np.float32)


def _colorize(hits, seen, queries, *, intensity, conflict, pct):
    """hits: [n, K] int hit counts (column k -> queries[k]). Returns uint8 [n,3]."""
    n, K = hits.shape
    if intensity == "fraction":
        seen_safe = np.maximum(seen, 1)[:, None]
        vals = hits / seen_safe
    else:
        vals = hits.astype(np.float32)
    # Per-class normalised intensity so each class uses the full brightness range.
    norm = np.stack([_normalized_intensity(vals[:, k], pct) for k in range(K)], axis=1)
    palette = np.stack([q["rgb"] for q in queries], axis=0)  # [K,3]

    any_hit = hits.max(axis=1) > 0
    winner = np.argmax(hits, axis=1)                          # argmax-by-count
    win_norm = norm[np.arange(n), winner]
    bright = (MIN_BRIGHT + (1.0 - MIN_BRIGHT) * win_norm)[:, None]

    colors = np.empty((n, 3), dtype=np.float32)
    colors[:] = BG_UNSEEN
    colors[seen > 0] = BG_SEEN

    if conflict == "blend":
        denom = np.maximum(norm.sum(axis=1, keepdims=True), 1e-6)
        weights = norm / denom                               # [n,K]
        mixed = weights @ palette                            # [n,3]
        max_norm = norm.max(axis=1)
        bbright = (MIN_BRIGHT + (1.0 - MIN_BRIGHT) * max_norm)[:, None]
        colors[any_hit] = (mixed * bbright)[any_hit]
    else:
        win_col = palette[winner] * bright
        colors[any_hit] = win_col[any_hit]
    return np.clip(colors, 0, 255).astype(np.uint8)


def _write_ply(path, pts, colors, hits, seen, queries):
    fields = [("x", "f4"), ("y", "f4"), ("z", "f4"),
              ("red", "u1"), ("green", "u1"), ("blue", "u1")]
    fields += [(f"{q['key']}_hits", "i4") for q in queries]
    fields += [("seen", "i4")]
    rec = np.empty(pts.shape[0], dtype=fields)
    rec["x"], rec["y"], rec["z"] = pts[:, 0], pts[:, 1], pts[:, 2]
    rec["red"], rec["green"], rec["blue"] = colors[:, 0], colors[:, 1], colors[:, 2]
    for k, q in enumerate(queries):
        rec[f"{q['key']}_hits"] = hits[:, k]
    rec["seen"] = seen
    PlyData([PlyElement.describe(rec, "vertex")], text=False).write(str(path))


def _hist(hits_col):
    pos = hits_col[hits_col >= 1]
    if pos.size == 0:
        return {}
    out = {f">={k}": int((pos >= k).sum()) for k in (1, 2, 3, 5, 10)}
    out["max"] = int(pos.max())
    return out


def _dir_transforms(render_dirs, ply_path, scan_dir, orientation):
    """Per render dir: 4x4 to apply to the oriented cloud → that run's pose frame
    (scan-schema v1.3). {} if frame metadata is unavailable (projects as-is)."""
    if not _V13 or scan_dir is None or not (scan_dir / "meta.json").exists():
        return {}
    try:
        sm = read_scan_meta(scan_dir)
        raw, _ = load_ply(ply_path, orientation="Y+")   # identity -> stored coords for fp
        fp = cloud_fingerprint(np.asarray(raw, dtype=np.float64))
        R3 = euler_xyz_matrix(*ORIENTATION_PRESETS[orientation])
        return dir_cloud_transforms(render_dirs, sm["frame"],
                                    sm["derivation"]["variant_id"], fp, R3)
    except Exception as e:  # noqa: BLE001
        print(f"  [remap] unavailable ({e}); projecting cloud as-is")
        return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", type=Path, help="annotated/<scan> dir (auto-discovers renders/ + scan.ply)")
    ap.add_argument("--renders", type=Path, action="append", default=[])
    ap.add_argument("--ply", type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--queries", default="floor,wall",
                    help="comma list; each `name` or `name=prompt1|prompt2` (synonym union)")
    ap.add_argument("--colors", default="",
                    help="optional per-query colour overrides, e.g. 'pipe=#3cb44b,floor=#4287f5'")
    ap.add_argument("--assign", choices=["argmax", "union"], default="argmax",
                    help="per-pixel exclusive winner (argmax) or independent per query (union)")
    ap.add_argument("--intensity", choices=["count", "fraction"], default="count")
    ap.add_argument("--conflict", choices=["argmax", "blend"], default="argmax")
    ap.add_argument("--pct", type=float, default=95.0)
    ap.add_argument("--mask-thresh", type=float, default=0.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--only-frame", default="")
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--orientation", default="Z+", choices=list(ORIENTATION_PRESETS.keys()))
    ap.add_argument("--no-remap", action="store_true", help="disable the v1.3 frame remap")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    color_overrides = {}
    for tok in args.colors.split(","):
        tok = tok.strip()
        if not tok:
            continue
        cname, chex = tok.split("=", 1)
        color_overrides[_sanitize(cname)] = _parse_hex(chex)
    queries = _parse_queries(args.queries, color_overrides)
    K = len(queries)

    render_dirs = list(args.renders)
    ply_path = args.ply
    scan_dir = args.scan.resolve() if args.scan else None
    if args.scan:
        if not render_dirs:
            render_dirs = _discover_runs(scan_dir)
        if ply_path is None:
            ply_path = scan_dir / "source" / "scan.ply"
    if ply_path is not None and scan_dir is None:
        scan_dir = ply_path.resolve().parent.parent   # source/scan.ply -> scan dir
    if not render_dirs:
        print("ERROR: no render dirs", file=sys.stderr); return 2
    if ply_path is None or not ply_path.exists():
        print(f"ERROR: scan.ply not found ({ply_path})", file=sys.stderr); return 2

    args.out.mkdir(parents=True, exist_ok=True)

    print(f"[1/4] Loading PLY {ply_path}")
    pts, _rgb = load_ply(ply_path, orientation=args.orientation)
    n = pts.shape[0]
    dir_T = {} if args.no_remap else _dir_transforms(render_dirs, ply_path, scan_dir, args.orientation)
    n_remapped = sum(1 for t in dir_T.values() if t is not None)
    qdesc = ", ".join(f"{q['name']}({'|'.join(q['prompts'])})" for q in queries)
    print(f"      {n:,} points  | queries=[{qdesc}] assign={args.assign}"
          f" | remapping {n_remapped}/{len(render_dirs)} run(s)")

    pts_cache: dict = {}

    def pts_for(rd):
        T = dir_T.get(rd)
        if T is None:
            return pts
        if rd not in pts_cache:
            pts_cache[rd] = apply_transform(T, pts)
        return pts_cache[rd]

    frames = gather_frames(render_dirs, args.stride, args.max_frames, args.only_frame)
    if not frames:
        print("ERROR: no usable frames", file=sys.stderr); return 3
    print(f"[2/4] {len(frames)} frames across {len(render_dirs)} run(s)")

    print(f"[3/4] Loading SAM3 on {args.device}…")
    proc = build_processor(args.device)
    print("      ready")

    hits = np.zeros((n, K), dtype=np.int32)
    seen = np.zeros(n, dtype=np.int32)

    for fi, (rd, frame) in enumerate(frames):
        t0 = time.time()
        pil = Image.open(rd / frame["file"]).convert("RGB")
        W, H = pil.size
        pos = np.array(frame["position"], dtype=np.float64)
        if "target" in frame:
            tgt = np.array(frame["target"], dtype=np.float64)
        else:
            yaw = float(frame.get("yaw", 0.0))
            tgt = pos + np.array([np.cos(yaw), 0.0, np.sin(yaw)])
        view = look_at_view(pos, tgt)
        u, v, z, in_front = project_points(pts_for(rd), view, args.fov, W, H)
        vis_idx, vis_u, vis_v = depth_buffer_mask(u, v, z, in_front, W, H)
        if vis_idx.size == 0:
            print(f"  frame {fi+1}/{len(frames)} {rd.name}/{frame['file']}: no visible pts")
            continue
        seen[vis_idx] += 1

        # One mask-score vector per query for the visible pixels.
        scores = np.zeros((vis_idx.size, K), dtype=np.float32)
        for k, q in enumerate(queries):
            qmap = _union_for_prompts(proc, pil, q["prompts"])
            if qmap is not None:
                scores[:, k] = qmap[vis_v, vis_u]

        if args.assign == "argmax":
            top = scores.argmax(axis=1)
            top_val = scores[np.arange(scores.shape[0]), top]
            hit = top_val > args.mask_thresh
            cols = top[hit]
            np.add.at(hits, (vis_idx[hit], cols), 1)
            per_q = np.bincount(cols, minlength=K)
        else:  # union
            mask = scores > args.mask_thresh
            for k in range(K):
                hits[vis_idx[mask[:, k]], k] += 1
            per_q = mask.sum(axis=0)
        summ = " ".join(f"{q['key']}+{int(per_q[k]):,}" for k, q in enumerate(queries))
        print(f"  frame {fi+1}/{len(frames)} {rd.name}/{frame['file']}: "
              f"{vis_idx.size:,} vis, {summ}, {time.time()-t0:.1f}s")

    print("[4/4] Colourizing")
    colors = _colorize(hits, seen, queries,
                       intensity=args.intensity, conflict=args.conflict, pct=args.pct)
    out_ply = args.out / "segmentation.ply"
    _write_ply(out_ply, pts, colors, hits, seen, queries)

    n_multi = int((((hits >= 1).sum(axis=1)) >= 2).sum())
    summary = {
        "n_points": int(n), "n_seen": int((seen > 0).sum()),
        "coverage_pct": round(100.0 * (seen > 0).sum() / n, 2),
        "queries": {q["key"]: {"name": q["name"], "prompts": q["prompts"],
                               "color": [int(c) for c in q["rgb"]],
                               "n_hit": int((hits[:, k] >= 1).sum()),
                               "hit_hist": _hist(hits[:, k])}
                    for k, q in enumerate(queries)},
        "n_multi_class": n_multi,
        "params": {"assign": args.assign, "remapped_runs": n_remapped,
                   "intensity": args.intensity, "conflict": args.conflict,
                   "frames_used": len(frames)},
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\nDone: {out_ply}")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
