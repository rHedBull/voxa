"""Registration health-check (scan-schema v1.3 §6): does a cloud actually
project into a render set's poses? Catches frame/version mismatch before any
expensive downstream work, independent of metadata correctness."""
from __future__ import annotations

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
                       min_photometric: float = 0.5) -> tuple[bool, list[str]]:
    """Decide pass/fail from a score dict. Returns (ok, reasons)."""
    reasons: list[str] = []
    if score["coverage"] < min_coverage:
        reasons.append(
            f"coverage {score['coverage']:.1%} < {min_coverage:.0%} "
            f"— cloud likely not in the renders' frame")
    p = score.get("photometric")
    if p is not None and p < min_photometric:
        reasons.append(
            f"photometric agreement {p:.1%} < {min_photometric:.0%} "
            f"— projected colours don't match the renders")
    return (not reasons), reasons
