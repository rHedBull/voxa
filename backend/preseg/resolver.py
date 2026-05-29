"""Resolve a cloud variant <-> render-run pairing (scan-schema v1.3 §5).

Returns how to consume a render run's poses with a given cloud variant:
``use_direct`` (same variant), ``remap`` (same scan, different frame -> apply the
returned transform to the poses), or ``fail`` (different scan / missing pin).

IMPORTANT: a ``remap`` result is the *metadata's* claim. Callers MUST still run the
Phase-1 ``preseg.registration`` health-check on the remapped result before trusting
it — metadata can be wrong, and the empirical check is the real guard.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from scenes.frame import Frame, compose_a_to_v


@dataclass
class Resolution:
    action: str                              # "use_direct" | "remap" | "fail"
    transform: Optional[np.ndarray] = None   # maps render poses INTO the cloud frame
    reasons: list = field(default_factory=list)


def resolve_render_run(cloud_frame: Frame, cloud_variant_id: str,
                       cloud_fingerprint: str, run_meta: dict) -> Resolution:
    gf = run_meta.get("generated_from") or {}
    src_variant = gf.get("variant_id")
    src_fp = gf.get("source_fingerprint")
    run_frame = run_meta.get("frame")
    if not src_variant or run_frame is None:
        return Resolution("fail", reasons=["render run has no source-variant/frame pin"])

    # Canonical id == scan identity (variant ids are only unique within a scan), so
    # a different canonical means a different scan -> refuse outright.
    if run_frame.canonical_id != cloud_frame.canonical_id:
        return Resolution("fail", reasons=[
            f"render run frame '{run_frame.canonical_id}' is a different scan than the "
            f"cloud frame '{cloud_frame.canonical_id}'"])

    # Same scan: exact variant (+fingerprint) -> use poses directly; else remap.
    if src_variant == cloud_variant_id and (src_fp is None or src_fp == cloud_fingerprint):
        return Resolution("use_direct", transform=np.eye(4))
    T = compose_a_to_v(run_frame, cloud_frame)
    return Resolution("remap", transform=T)


def dir_cloud_transforms(render_dirs, cloud_frame: Frame, cloud_variant_id: str,
                         cloud_fingerprint: str, orientation_R3: np.ndarray) -> dict:
    """Per render dir, the 4x4 transform to apply to the ORIENTED cloud
    (``xyz @ orientation_R3.T``) so it lands in that run's pose frame.

    Returns ``{render_dir: 4x4 | None}``; ``None`` means no render meta.json
    (legacy run — project the oriented cloud as-is, prior behaviour). Raises
    ``ValueError`` if any run resolves to ``fail`` (cross-scan / unpinned).

    The transform conjugates the canonical-frame bridge back into the oriented
    frame: ``R4 @ inv(M_run) @ M_cloud @ inv(R4)`` where ``M_*`` are the frames'
    ``transform_to_canonical`` (so a same-variant run yields identity, and the
    navvis pure-translation case yields the verified Y-up translation).
    """
    from scenes.render_meta import read_render_meta

    R4 = np.eye(4)
    R4[:3, :3] = orientation_R3
    R4inv = np.linalg.inv(R4)
    out: dict = {}
    for rd in render_dirs:
        rm = read_render_meta(rd)
        if rm is None:
            out[rd] = None
            continue
        res = resolve_render_run(cloud_frame, cloud_variant_id, cloud_fingerprint, rm)
        if res.action == "fail":
            raise ValueError(f"{getattr(rd, 'name', rd)}: " + "; ".join(res.reasons))
        core = np.linalg.inv(rm["frame"].transform_to_canonical) @ cloud_frame.transform_to_canonical
        out[rd] = R4 @ core @ R4inv
    return out
