"""FastAPI app tying the sidecar pieces together: render → SAM → back-project.

Endpoint contract (see CLAUDE.md task spec):
  POST /capture  — render the raw cloud from a given camera, run SAM (box or
                   concept mode), stash the result under a fresh capture_id.
  POST /project  — back-project one or more masks from the last capture onto
                   the (voxa-resolution) scan cloud, returning point indices.

voxa's proxy resolves scan paths and passes them straight through in the
/capture request body — this sidecar never resolves paths itself.
"""
from __future__ import annotations

import base64, colorsys, io, uuid
from typing import Literal
import numpy as np
from fastapi import FastAPI, HTTPException
from PIL import Image
from pydantic import BaseModel

import cloud
from render import render_view
from backproject import select_in_mask
from reproject import look_at_view
from scan_store import ScanStore, FingerprintMismatch

# Single-process, single-worker dev sidecar: these module globals are shared,
# unlocked state. Run under one uvicorn worker only. `CAPTURES` holds just the
# latest capture — a new /capture drops any prior one (last-capture-wins), so
# two concurrent clients would race and the loser's /project would 409.
STORE = ScanStore(loader=lambda sid, raw, ply: (*cloud.load_raw(raw), cloud.load_scan_ply(ply)))
CAPTURES: dict = {}          # at most one live capture_id (cleared each /capture)
_PROC = {"proc": None}


def _proc():
    if _PROC["proc"] is None:
        import sam_infer; _PROC["proc"] = sam_infer.build_processor()
    return _PROC["proc"]


def _sam_box(img, box, text):          # wrappers → monkeypatchable in tests
    import sam_infer; return sam_infer.segment_box(_proc(), img, box, text)


def _sam_concept(img, text):
    import sam_infer; return sam_infer.segment_concept(_proc(), img, text)


app = FastAPI()


class Camera(BaseModel):
    pos: list[float]; target: list[float]; fov: float; W: int; H: int


class CaptureReq(BaseModel):
    scan_id: str; source_fingerprint: str; raw_laz_path: str; scan_ply_path: str
    scan_ply_offset_m: list[float] = [0.0, 0.0, 0.0]
    camera: Camera
    mode: Literal["box", "concept"]; box: list[float] | None = None; text: str | None = None


class ProjectReq(BaseModel):
    scan_id: str; source_fingerprint: str; capture_id: str; mask_ids: list[int]


def _ensure(scan_id, fp, raw=None, ply=None, offset=None):
    try:
        STORE.ensure(scan_id, fp, raw, ply, offset)     # paths/offset used only when (re)loading
    except FingerprintMismatch as e:
        raise HTTPException(409, {"diverged": "source", "detail": str(e)})


def _palette(n: int):
    """n visually distinct RGB colors (golden-ratio hue spacing)."""
    out = []
    for i in range(n):
        r, g, b = colorsys.hsv_to_rgb((i * 0.61803398875) % 1.0, 0.65, 1.0)
        out.append((int(r * 255), int(g * 255), int(b * 255)))
    return out


def _resize(mask: np.ndarray, H: int, W: int) -> np.ndarray:
    """Align a SAM mask (possibly at the frame's own size) to the render's (H, W)."""
    if mask.shape == (H, W):
        return mask.astype(bool)
    resized = Image.fromarray(mask.astype(np.uint8) * 255).resize((W, H), Image.NEAREST)
    return np.array(resized) > 127


def _png_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _overlay_png(color: np.ndarray, masks: list[np.ndarray]) -> str:
    """Wash each mask over `color` in a distinct palette color, alpha-composited."""
    base = Image.fromarray(color, "RGB").convert("RGBA")
    H, W = color.shape[:2]
    overlay = np.zeros((H, W, 4), dtype=np.uint8)
    for i, col in zip(range(len(masks)), _palette(len(masks))):
        overlay[masks[i]] = (col[0], col[1], col[2], 130)
    over = Image.alpha_composite(base, Image.fromarray(overlay, "RGBA")).convert("RGB")
    return _png_data_uri(over)


def _index_png(H: int, W: int, masks: list[np.ndarray]) -> str:
    """Per-pixel mask index (1-based, 0 = none), lossless PNG. Same last-write-wins
    overlap order as _overlay_png, so a client hit-testing a click against this map
    always picks the mask that's visibly on top in the overlay — up to 254 masks."""
    idx = np.zeros((H, W), dtype=np.uint8)
    for i, m in enumerate(masks[:254]):
        idx[m] = i + 1
    return _png_data_uri(Image.fromarray(idx, "L"))


@app.get("/health")
def health(): return {"ok": True, "scan_id": STORE.scan_id}


@app.post("/capture")
def capture(req: CaptureReq):
    _ensure(req.scan_id, req.source_fingerprint, req.raw_laz_path, req.scan_ply_path,
            req.scan_ply_offset_m)
    cam = req.camera
    view = look_at_view(cam.pos, cam.target, up=(0.0, 0.0, 1.0))
    color, depth = render_view(STORE.raw_xyz, STORE.raw_rgb, view, cam.fov, cam.W, cam.H)
    frame = Image.fromarray(color, "RGB")
    if req.mode == "concept":
        if not req.text: raise HTTPException(400, "text required for concept mode")
        insts = _sam_concept(frame, req.text)
    else:
        m, sc = _sam_box(frame, req.box, req.text); insts = [(m, sc)] if m.any() else []
    masks = [_resize(m, cam.H, cam.W) for m, _ in insts]
    cid = uuid.uuid4().hex
    CAPTURES.clear()
    CAPTURES[cid] = {"depth": depth, "masks": masks, "camera": cam.model_dump()}
    return {"capture_id": cid, "overlay_png_b64": _overlay_png(color, masks),
            "mask_index_png_b64": _index_png(cam.H, cam.W, masks),
            "masks": [{"mask_id": i, "score": float(insts[i][1])} for i in range(len(masks))]}


@app.post("/project")
def project(req: ProjectReq):
    _ensure(req.scan_id, req.source_fingerprint)
    cap = CAPTURES.get(req.capture_id)
    if cap is None: raise HTTPException(409, "stale or unknown capture_id")
    cam = cap["camera"]; view = look_at_view(cam["pos"], cam["target"], up=(0.0, 0.0, 1.0))
    n_masks = len(cap["masks"])
    bad = [mid for mid in req.mask_ids if mid < 0 or mid >= n_masks]
    if bad:                                  # fail loud — never silently drop a pick
        raise HTTPException(400, f"mask_ids {bad} out of range for capture with {n_masks} masks")
    out = []
    for mid in req.mask_ids:
        sel = select_in_mask(STORE.scan_xyz, view, cam["fov"], cam["W"], cam["H"],
                             cap["masks"][mid], cap["depth"])
        out.append({"mask_id": mid, "scan_indices_b64": base64.b64encode(sel.tobytes()).decode()})
    return {"instances": out}
