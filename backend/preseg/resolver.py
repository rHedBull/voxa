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
