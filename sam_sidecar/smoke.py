#!/usr/bin/env python3
"""Live smoke test for the SAM sidecar against the real SMART-AIS cloud.

Starts nothing — assumes the sidecar is already running (`bash run.sh`, port 8011).
Drives one /capture (box mode, centered box) + /project on the full 188M raw cloud,
prints the mask count + selected-point count, and saves the returned overlay PNG so
you can eyeball what SAM saw. This is Task 8 of the plan as a one-command harness.

    bash run.sh                       # terminal 1 (loads SAM; leave running)
    /home/hendrik/anaconda3/bin/python smoke.py   # terminal 2

Override the scan via --raw / --scan-ply / --url.
"""
from __future__ import annotations
import argparse, base64, sys
import numpy as np
import httpx
import laspy

RAW = "/home/hendrik/coding/engine/data/lidar/raw/Sample-Data-VLX3-ProcessIndustry-SMART-AIS.laz"
SCAN_PLY = "/home/hendrik/coding/engine/data/lidar/annotated/smart_ais_navvis/source/scan.ply"


def interior_pose(laz_path):
    """An eye-level interior pose in the cloud's NATIVE frame (mirrors the design
    session's persp_raw.py framing). Returns {pos, target, fov, W, H}."""
    with laspy.open(laz_path) as fh:
        lo = np.array(fh.header.mins, float)
        hi = np.array(fh.header.maxs, float)
    c = (lo + hi) / 2
    ext = hi - lo
    up_ax = int(np.argmin(ext))
    plane = [a for a in range(3) if a != up_ax]
    pos = c.copy()
    pos[plane[0]] = lo[plane[0]] + 0.15 * ext[plane[0]]
    pos[plane[1]] = c[plane[1]]
    pos[up_ax] = lo[up_ax] + 0.45 * ext[up_ax]
    tgt = c.copy()
    tgt[plane[0]] = hi[plane[0]]
    tgt[up_ax] = lo[up_ax] + 0.40 * ext[up_ax]
    return {"pos": pos.tolist(), "target": tgt.tolist(), "fov": 65.0, "W": 1400, "H": 1050}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8011")
    ap.add_argument("--raw", default=RAW)
    ap.add_argument("--scan-ply", default=SCAN_PLY)
    ap.add_argument("--text", default=None, help="optional SAM text refinement / concept prompt")
    ap.add_argument("--mode", default="box", choices=["box", "concept"])
    ap.add_argument("--out", default="smoke_overlay.png")
    args = ap.parse_args()

    cam = interior_pose(args.raw)
    print(f"[smoke] pose (native): pos={[round(x,1) for x in cam['pos']]} "
          f"target={[round(x,1) for x in cam['target']]}")

    cap_body = {
        "scan_id": "smoke", "source_fingerprint": "smoke-fp",
        "raw_laz_path": args.raw, "scan_ply_path": args.scan_ply,
        "camera": cam, "mode": args.mode,
        "box": None if args.mode == "concept" else [0.5, 0.5, 0.5, 0.5],
        "text": args.text,
    }
    print("[smoke] POST /capture (first call loads the 188M raw cloud — ~60s cold) ...")
    r = httpx.post(f"{args.url}/capture", json=cap_body, timeout=600.0)
    r.raise_for_status()
    cap = r.json()
    masks = cap["masks"]
    print(f"[smoke] capture_id={cap['capture_id']}  masks={len(masks)}  "
          f"scores={[round(m['score'], 3) for m in masks]}")
    if cap.get("overlay_png_b64"):
        b64 = cap["overlay_png_b64"].split(",", 1)[-1]
        with open(args.out, "wb") as f:
            f.write(base64.b64decode(b64))
        print(f"[smoke] wrote overlay → {args.out}  (open it to see what SAM segmented)")
    if not masks:
        print("[smoke] no masks — SAM found nothing; try --mode concept --text pipe")
        return 1

    pick = [m["mask_id"] for m in masks[:3]]
    print(f"[smoke] POST /project mask_ids={pick} ...")
    r = httpx.post(f"{args.url}/project", json={
        "scan_id": "smoke", "source_fingerprint": "smoke-fp",
        "capture_id": cap["capture_id"], "mask_ids": pick}, timeout=600.0)
    r.raise_for_status()
    for inst in r.json()["instances"]:
        sel = np.frombuffer(base64.b64decode(inst["scan_indices_b64"]), np.int32)
        print(f"[smoke]   mask #{inst['mask_id']} → {sel.size:,} scan points selected")
    print("[smoke] OK — render + SAM + back-projection all ran on the real cloud.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
