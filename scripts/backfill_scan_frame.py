"""Recover a scan's render-frame transform and backfill v1.3 frame metadata.

For a scan whose ``source/scan.ply`` is in a different (e.g. unaligned) frame than
the cloud its ``renders/`` were generated from, this:

  1. decodes the aligned source cloud from its Potree octree,
  2. ICP-registers scan.ply -> aligned cloud (both rotated by ``--orientation``),
  3. writes the scan ``meta.json`` v1.3 ``frame`` (identity = canonical) + ``derivation``,
  4. writes ``renders/<run>/meta.json`` pinning the aligned variant + the recovered
     ``transform_to_canonical`` (render -> canonical, in stored coords),
  5. proves the chain: resolver -> remap -> registration health-check.

Run with voxa's .venv (Open3D ICP; no torch):

    .venv/bin/python scripts/backfill_scan_frame.py <scan_dir> \
        --potree-dir <aligned potree cloud dir> --aligned-variant-id aligned15M

This is scan-schema v1.3 Phase 3a (recover + record the real frame).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from preseg.registration import check_registration, registration_score  # noqa: E402
from preseg.resolver import resolve_render_run  # noqa: E402
from scenes.fingerprint import cloud_fingerprint  # noqa: E402
from scenes.frame import Frame, apply_transform  # noqa: E402
from scenes.point_cloud import load_ply  # noqa: E402
from scenes.render_meta import read_render_meta, write_render_meta  # noqa: E402
from scenes.reproject import ORIENTATION_PRESETS, euler_xyz_matrix  # noqa: E402


def _decode_potree_xyz(potree_dir: Path) -> np.ndarray:
    """Decode all point positions from a Potree 2.0 DEFAULT-encoding octree."""
    meta = json.loads((potree_dir / "metadata.json").read_text())
    scale = np.array(meta["scale"]); offset = np.array(meta["offset"])
    bpp = sum(int(a["size"]) for a in meta["attributes"])  # bytes per point
    raw = np.fromfile(potree_dir / "octree.bin", dtype=np.uint8)
    n = raw.size // bpp
    pos = raw[: n * bpp].reshape(n, bpp)[:, 0:12].copy().view(np.int32).reshape(n, 3)
    return pos.astype(np.float64) * scale + offset


def _icp(src_xyz, tgt_xyz, n=300000):
    import open3d as o3d

    def pcd(x, seed):
        idx = np.random.default_rng(seed).choice(len(x), min(n, len(x)), replace=False)
        p = o3d.geometry.PointCloud(); p.points = o3d.utility.Vector3dVector(x[idx])
        p.estimate_normals(); return p

    src, tgt = pcd(src_xyz, 0), pcd(tgt_xyz, 1)
    init = np.eye(4)
    init[:3, 3] = np.asarray(tgt.points).mean(0) - np.asarray(src.points).mean(0)
    reg = o3d.pipelines.registration.registration_icp(
        src, tgt, 1.0, init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200))
    return reg.transformation, reg.fitness, reg.inlier_rmse


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", type=Path)
    ap.add_argument("--potree-dir", required=True, type=Path,
                    help="aligned source cloud as a Potree 2.0 octree dir")
    ap.add_argument("--aligned-variant-id", default="aligned")
    ap.add_argument("--orientation", default="Z+", choices=list(ORIENTATION_PRESETS))
    ap.add_argument("--variant-id", default=None, help="this scan's variant id (default: <name>)")
    ap.add_argument("--dry-run", action="store_true", help="recover + prove, but don't write metadata")
    args = ap.parse_args()

    scan_dir: Path = args.scan_dir.resolve()
    variant_id = args.variant_id or scan_dir.name
    R4 = np.eye(4); R4[:3, :3] = euler_xyz_matrix(*ORIENTATION_PRESETS[args.orientation])

    pcA, _ = load_ply(scan_dir / "source" / "scan.ply")
    A_stored = np.asarray(pcA.points, float)
    _cA = getattr(pcA, "colors", None)
    rgbA = np.asarray(_cA).astype(np.uint8) if _cA is not None and len(_cA) else None
    fp_scan = cloud_fingerprint(A_stored)

    B_native = _decode_potree_xyz(args.potree_dir)
    fp_aligned = cloud_fingerprint(B_native)
    print(f"[recover] scan.ply {len(A_stored):,} (fp {fp_scan[:19]}…), "
          f"aligned {len(B_native):,} (fp {fp_aligned[:19]}…)")

    A_yup = apply_transform(R4, A_stored)
    B_yup = apply_transform(R4, B_native)
    T_yup, fit, rmse = _icp(A_yup, B_yup)  # scanply_yup -> aligned_yup
    print(f"[recover] ICP fitness={fit:.3f} rmse={rmse:.3f}")
    if fit < 0.6:
        print("ERROR: ICP fitness too low — refusing to write a bad transform "
              "(check --potree-dir / --orientation)", file=sys.stderr)
        return 4
    # render_native -> scanply_stored (canonical), conjugated back from the Y-up frame
    M_render_to_canon = np.linalg.inv(R4) @ np.linalg.inv(T_yup) @ R4

    if not args.dry_run:
        sm = json.loads((scan_dir / "meta.json").read_text())
        sm["schema_version"] = "1.3"
        sm["frame"] = Frame(np.eye(4), f"{scan_dir.name}#local",
                            georef={"offset_m": sm["coord_offset_m"]} if sm.get("coord_offset_m") else None,
                            frame_uncertain=False).to_dict()
        sm["derivation"] = {"scan_id": scan_dir.name, "variant_id": variant_id,
                            "parent": "original", "op": "voxel_downsample",
                            "varies": ["density"], "source_fingerprint": fp_scan, "role": "labeling"}
        (scan_dir / "meta.json").write_text(json.dumps(sm, indent=2))
        print(f"[write] {scan_dir/'meta.json'}")

        render_frame = Frame(M_render_to_canon, f"{scan_dir.name}#local")
        for run in sorted(d for d in (scan_dir / "renders").iterdir() if (d / "manifest.json").exists()):
            m = json.loads((run / "manifest.json").read_text())
            intr = dict((m.get("frames") or [{}])[0].get("intrinsics",
                        {"fov_deg": 60, "aspect": 1.169, "width": 926, "height": 792}))
            intr.setdefault("fov_axis", "vertical")
            write_render_meta(run, run_id=run.name,
                              generated_from={"scan_id": scan_dir.name,
                                              "variant_id": args.aligned_variant_id,
                                              "source_fingerprint": fp_aligned,
                                              "n_points": int(len(B_native))},
                              frame=render_frame, intrinsics=intr,
                              generated_at=m.get("name", ""), n_frames=len(m.get("frames", [])))
            print(f"[write] {run/'meta.json'}")

    # prove
    print("[prove] resolving + remapping + health-check…")
    from PIL import Image
    cloud_frame = Frame(np.eye(4), f"{scan_dir.name}#local")
    frames = []
    for run in sorted(d for d in (scan_dir / "renders").iterdir() if (d / "manifest.json").exists()):
        rm = read_render_meta(run) if not args.dry_run else {
            "generated_from": {"variant_id": args.aligned_variant_id, "scan_id": scan_dir.name,
                               "source_fingerprint": fp_aligned},
            "frame": Frame(M_render_to_canon, f"{scan_dir.name}#local")}
        res = resolve_render_run(cloud_frame, variant_id, fp_scan, rm)
        print(f"  resolve({run.name}) -> {res.action}")
        m = json.loads((run / "manifest.json").read_text())
        for f in m.get("frames", []):
            if (run / f["file"]).exists():
                f["_run"] = str(run); frames.append(f)
    W, H = Image.open(Path(frames[0]["_run"]) / frames[0]["file"]).size
    A_render_yup = apply_transform(R4 @ np.linalg.inv(M_render_to_canon), A_stored)
    score = registration_score(A_render_yup, frames, fov_y_deg=60, W=W, H=H, rgb=rgbA,
                               image_loader=lambda f: np.array(
                                   Image.open(Path(f["_run"]) / f["file"]).convert("RGB")))
    ok, reasons = check_registration(score)
    ph = f"{score['photometric']:.1%}" if score["photometric"] is not None else "n/a"
    print(f"[prove] after remap: coverage {score['coverage']:.1%}, photometric {ph} "
          f"-> {'PASS' if ok else 'FAIL'}")
    for r in reasons:
        print("  -", r)
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
