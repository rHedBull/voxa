"""Proxy to the SAM sidecar. /capture forwards the pose (+recenter_offset, scan
identity + resolved cloud paths); /project forwards mask picks, then materializes
the returned scan indices as SAM candidate segments (sam_ids layer) — NOT
classified instances. Classification happens later via the shared
apply_reassign pipeline, from a selection over the materialized candidates."""
from __future__ import annotations
import base64, os
import numpy as np
import httpx
from fastapi import APIRouter, HTTPException

from app.core import _state, _require_seg, _resolve, _y_up_to_z_up_xyz, _b64
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
    # voxa resolves the raw-LAZ + scan.ply paths (same source it uses for export)
    # and hands them to the sidecar, so the sidecar never re-resolves.
    src = _resolve(_state.get("scene"))
    recenter = _state.get("recenter_offset") or [0.0, 0.0, 0.0]      # already Y-up
    georef = _state.get("raw_georef_offset_m") or [0.0, 0.0, 0.0]    # native frame (Z-up if is_z_up)
    # georef is copied straight from the raw LAZ's native axis order; rotate it into
    # Y-up (matching recenter + the camera pose) only if this scan's scan.ply is Z-up.
    georef_yup = [georef[0], georef[2], -georef[1]] if src.extras.get("is_z_up") else georef
    off = [r + g for r, g in zip(recenter, georef_yup)]
    cam = dict(req.camera)
    pos_yup = [p + o for p, o in zip(cam["pos"], off)]            # recentered → native, still Y-up
    tgt_yup = [p + o for p, o in zip(cam["target"], off)]
    # The sidecar renders raw_xyz in its native frame (Z-up for a LAZ-derived scan) and
    # hardcodes up=(0,0,1) for that reason — voxa's camera is Y-up internally, so it must
    # be rotated back to Z-up here, not just offset, or pos/target land nowhere near the cloud.
    if src.extras.get("is_z_up"):
        pos_native = _y_up_to_z_up_xyz(np.array([pos_yup]))[0].tolist()
        tgt_native = _y_up_to_z_up_xyz(np.array([tgt_yup]))[0].tolist()
    else:
        pos_native, tgt_native = pos_yup, tgt_yup
    cam["pos"] = pos_native
    cam["target"] = tgt_native
    raw_laz_path = src.extras.get("source_laz_path")
    if not raw_laz_path:
        raise HTTPException(409, {"diverged": "source", "detail": "no raw cloud for this scan"})
    body = {**_identity(), "raw_laz_path": raw_laz_path, "scan_ply_path": str(src.source_path),
            "camera": cam, "mode": req.mode, "box": req.box, "text": req.text,
            "scan_ply_offset_m": georef}  # native/raw order; scan_xyz is loaded unrotated
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
    """Materialize each accepted mask as a SAM candidate segment — NOT a
    classified instance. Classification happens later, from the SAM segment
    list/viewport, through the same apply_reassign path every other tool
    uses (see mode-label.jsx::confirmSamSelection)."""
    base = _sidecar_url()
    seg = _require_seg()
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
        out = seg.materialize_sam_segment(idx, protect_instances=req.protect_instances)
        entry = {
            "mask_id": inst["mask_id"],
            "sam_seg_id": out["sam_seg_id"],
            "n_affected": out["n_affected"],
            "n_protected": out["n_protected"],
        }
        if out.get("indices") is not None:
            entry["scan_indices_b64"] = _b64(out["indices"].astype(np.int32))
        results.append(entry)
    return {"segments": results}
