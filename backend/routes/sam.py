"""Proxy to the SAM sidecar. /capture forwards the pose (+recenter_offset, scan
identity + resolved cloud paths); /project forwards mask picks, then applies the
returned scan indices through the shared apply_reassign pipeline (protect_instances
= confirmed = locked)."""
from __future__ import annotations
import base64, os
import numpy as np
import httpx
from fastapi import APIRouter, HTTPException

from app.core import _state, _require_seg, _serialize_apply, _coerce_class_id, _resolve
from app.schemas import SamCaptureRequest, SamProjectRequest

router = APIRouter()
_TIMEOUT = 120.0

def _sidecar_url() -> str:
    url = os.environ.get("VOXA_SAM_SIDECAR_URL")
    if not url:
        raise HTTPException(503, "SAM sidecar not configured (VOXA_SAM_SIDECAR_URL)")
    return url.rstrip("/")

def _identity() -> dict:
    return {"scan_id": _state.get("scene"), "source_fingerprint": _state.get("source_fp")}

@router.post("/api/sam/capture")
def sam_capture(req: SamCaptureRequest):
    base = _sidecar_url()                       # fail fast on missing config (503)
    off = _state.get("recenter_offset") or [0.0, 0.0, 0.0]
    cam = dict(req.camera)
    cam["pos"] = [p + o for p, o in zip(cam["pos"], off)]        # recentered → native
    cam["target"] = [p + o for p, o in zip(cam["target"], off)]
    # voxa resolves the raw-LAZ + scan.ply paths (same source it uses for export)
    # and hands them to the sidecar, so the sidecar never re-resolves.
    src = _resolve(_state.get("scene"))
    raw_laz_path = src.extras.get("source_laz_path")
    if not raw_laz_path:
        raise HTTPException(409, {"diverged": "source", "detail": "no raw cloud for this scan"})
    body = {**_identity(), "raw_laz_path": raw_laz_path, "scan_ply_path": src.source_path,
            "camera": cam, "mode": req.mode, "box": req.box, "text": req.text}
    try:
        r = httpx.post(f"{base}/capture", json=body, timeout=_TIMEOUT)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 409:
            raise HTTPException(409, e.response.json().get("detail"))
        raise HTTPException(502, f"SAM sidecar error {e.response.status_code}: {e.response.text}")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"SAM sidecar unreachable: {e}")
    return r.json()

@router.post("/api/sam/project")
def sam_project(req: SamProjectRequest):
    base = _sidecar_url()
    seg = _require_seg()
    try:
        target_class = _coerce_class_id(req.target_class)
    except ValueError as e:
        raise HTTPException(400, str(e))
    body = {**_identity(), "capture_id": req.capture_id, "mask_ids": req.mask_ids}
    try:
        r = httpx.post(f"{base}/project", json=body, timeout=_TIMEOUT)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 409:
            raise HTTPException(409, e.response.json().get("detail"))
        raise HTTPException(502, f"SAM sidecar error {e.response.status_code}: {e.response.text}")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"SAM sidecar unreachable: {e}")
    results = []
    for inst in r.json()["instances"]:
        idx = np.frombuffer(base64.b64decode(inst["scan_indices_b64"]), np.int32)
        out = seg.apply_reassign(idx, target_inst=-1, target_class=target_class,
                                 protect_instances=req.protect_instances)
        results.append({"mask_id": inst["mask_id"], **_serialize_apply(out)})
    return {"instances": results}
