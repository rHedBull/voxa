"""Registration health-check (scan-schema v1.3 §6): does a cloud actually
project into a render set's poses? Catches frame/version mismatch before any
expensive downstream work, independent of metadata correctness."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from scenes.reproject import depth_buffer_mask, look_at_view, project_points


def registration_score(xyz, frames, *, fov_y_deg, W, H,
                       rgb: Optional[np.ndarray] = None,
                       image_loader: Optional[Callable] = None,
                       color_tol: int = 40) -> dict:
    """Coverage (% of cloud visible in >=1 frame) and, if colours+images are
    given, photometric agreement (fraction of visible points whose cloud colour
    matches the render pixel within ``color_tol`` per channel).

    The cloud must already be in the renders' frame/orientation — rotate it
    (orientation preset) before calling, exactly as the SAM3 pipeline does.
    Each frame is a dict with ``position`` and ``target`` (or ``yaw``).
    """
    xyz = np.asarray(xyz, dtype=np.float64)
    seen = np.zeros(xyz.shape[0], dtype=bool)
    agree = total = 0
    for fr in frames:
        pos = np.asarray(fr["position"], float)
        if "target" in fr:
            tgt = np.asarray(fr["target"], float)
        else:
            yaw = float(fr.get("yaw", 0.0))
            tgt = pos + np.array([np.cos(yaw), 0.0, np.sin(yaw)])
        u, v, z, infront = project_points(xyz, look_at_view(pos, tgt), fov_y_deg, W, H)
        idx, ui, vi = depth_buffer_mask(u, v, z, infront, W, H)
        seen[idx] = True
        if rgb is not None and image_loader is not None and idx.size:
            img = np.asarray(image_loader(fr))
            diff = np.abs(rgb[idx].astype(int) - img[vi, ui].astype(int)).mean(1)
            agree += int((diff < color_tol).sum())
            total += idx.size
    return {
        "coverage": float(seen.mean()) if xyz.shape[0] else 0.0,
        "photometric": (agree / total) if total else None,
        "n_seen": int(seen.sum()),
        "n_points": int(xyz.shape[0]),
        "n_frames": len(frames),
    }


def check_registration(score: dict, *, min_coverage: float = 0.35,
                       min_photometric: float = 0.5,
                       coverage_floor: float = 0.05) -> tuple[bool, list[str]]:
    """Decide pass/fail from a score dict. Returns (ok, reasons).

    When colours are available, **photometric agreement is the primary signal**
    (it cleanly separates right ~90% from wrong ~40%); coverage is only a low
    floor to catch "nothing projected at all". A correctly-registered but sparsely
    / partially-viewed cloud legitimately has modest coverage, so we do NOT fail it
    on the coverage threshold when photometric confirms. Only when there are no
    colours to compare does coverage become the (stricter) primary test.
    """
    reasons: list[str] = []
    cov = score["coverage"]
    p = score.get("photometric")
    if p is not None:
        if p < min_photometric:
            reasons.append(
                f"photometric agreement {p:.1%} < {min_photometric:.0%} "
                f"— projected colours don't match the renders")
        if cov < coverage_floor:
            reasons.append(
                f"coverage {cov:.1%} < {coverage_floor:.0%} "
                f"— effectively nothing projects into the renders' frame")
    else:  # no colours — coverage is all we have, apply the stricter threshold
        if cov < min_coverage:
            reasons.append(
                f"coverage {cov:.1%} < {min_coverage:.0%} "
                f"— cloud likely not in the renders' frame (no colours to confirm)")
    return (not reasons), reasons


_VERDICT_CACHE: dict = {}


def _fov_y_from_intrinsics(intr: dict, W: int, H: int) -> float:
    """Vertical FOV (deg) for registration_score, honouring intrinsics.fov_axis.
    Horizontal FOV is converted to vertical via the render aspect; missing/vertical
    is used as-is. Default 60.0 when intrinsics absent."""
    intr = intr or {}
    fov = float(intr.get("fov_deg", 60.0))
    if intr.get("fov_axis") == "horizontal" and W and H:
        fov_x = math.radians(fov)
        return math.degrees(2.0 * math.atan(math.tan(fov_x / 2.0) / (W / H)))
    return fov


def verify_scan_registration(scan_dir, *, max_frames: int = 8, orientation: str = "Z+",
                             min_coverage: float = 0.35, min_photometric: float = 0.5,
                             coverage_floor: float = 0.05, color_tol: int = 40,
                             use_cache: bool = True) -> dict:
    """Whole-scan scan-schema v1.3 §6 health-check against the scan's render runs.

    Returns {checked, ok, runs:[{run_id, ok, coverage, photometric, n_seen,
    n_points, n_frames, reasons}], reasons}. checked=False (ok=True) when there is
    nothing verifiable (no renders / no images / resolves to legacy as-is with no
    images) — callers must NOT block in that case. A resolver cross-scan ValueError
    is a hard fail (checked=True, ok=False)."""
    from PIL import Image

    from preseg.resolver import dir_cloud_transforms
    from scenes.fingerprint import cloud_fingerprint
    from scenes.frame import apply_transform
    from scenes.point_cloud import load_ply
    from scenes.render_meta import read_render_meta
    from scenes.reproject import ORIENTATION_PRESETS, euler_xyz_matrix
    from scenes.scan_meta import read_scan_meta

    scan_dir = Path(scan_dir)
    renders_root = scan_dir / "renders"
    runs = (sorted(d for d in renders_root.iterdir()
                   if d.is_dir() and (d / "manifest.json").exists())
            if renders_root.is_dir() else [])
    skip = {"checked": False, "ok": True, "runs": [], "reasons": []}
    if not runs:
        return skip

    pc, _ = load_ply(scan_dir / "source" / "scan.ply")
    xyz_raw = np.asarray(pc.points, dtype=np.float64)
    rgb = (np.asarray(pc.colors).astype(np.uint8)
           if pc.colors is not None and len(pc.colors) else None)
    fp = cloud_fingerprint(xyz_raw)

    key = None
    if use_cache:
        run_fps = tuple(sorted(
            (r.name, ((read_render_meta(r) or {}).get("generated_from") or {}).get("source_fingerprint"))
            for r in runs))
        key = (fp, run_fps)
        if key in _VERDICT_CACHE:
            return _VERDICT_CACHE[key]

    R = euler_xyz_matrix(*ORIENTATION_PRESETS[orientation])
    xyz = xyz_raw @ R.T

    sm = read_scan_meta(scan_dir)
    try:
        dir_T = dir_cloud_transforms(runs, sm["frame"], sm["derivation"]["variant_id"], fp, R)
    except ValueError as e:
        verdict = {"checked": True, "ok": False, "runs": [], "reasons": [str(e)]}
        if key is not None:
            _VERDICT_CACHE[key] = verdict
        return verdict

    results = []
    for run in runs:
        manifest = json.loads((run / "manifest.json").read_text())
        frames = [f for f in manifest.get("frames", []) if (run / f["file"]).exists()]
        if not frames:
            continue
        step = max(1, len(frames) // max_frames)
        frames = frames[::step][:max_frames]
        T = dir_T.get(run)
        xyz_run = xyz if T is None else apply_transform(T, xyz)
        with Image.open(run / frames[0]["file"]) as _probe:
            W, H = _probe.size
        intr = (read_render_meta(run) or {}).get("intrinsics") or {}
        fov_y = _fov_y_from_intrinsics(intr, W, H)
        loader = lambda f, _run=run: np.array(Image.open(_run / f["file"]).convert("RGB"))
        s = registration_score(xyz_run, frames, fov_y_deg=fov_y, W=W, H=H,
                               rgb=rgb, image_loader=loader, color_tol=color_tol)
        ok, reasons = check_registration(s, min_coverage=min_coverage,
                                         min_photometric=min_photometric,
                                         coverage_floor=coverage_floor)
        results.append({"run_id": run.name, "ok": ok, "coverage": s["coverage"],
                        "photometric": s["photometric"], "n_seen": s["n_seen"],
                        "n_points": s["n_points"], "n_frames": s["n_frames"],
                        "reasons": reasons})

    if not results:
        verdict = dict(skip)
    else:
        verdict = {"checked": True, "ok": all(r["ok"] for r in results), "runs": results,
                   "reasons": [f"{r['run_id']}: {x}" for r in results for x in r["reasons"]]}
    if key is not None:
        _VERDICT_CACHE[key] = verdict
    return verdict
