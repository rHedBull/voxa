"""Voxa API routes: export."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.constants import *  # noqa: F401,F403
from app.schemas import *  # noqa: F401,F403
from app.core import *  # noqa: F401,F403

router = APIRouter()


@router.post("/api/edit/export-ply")
def edit_export_ply(req: ExportPlyRequest) -> Response:
    """Export the active slice as a PLY at native source density.

    Replays the slice's box-op chain (authored in the y-up + recentered
    display frame) against the source data at native resolution — re-
    reading the full source PLY or streaming the source LAZ chunk-by-
    chunk — so the saved file contains every source point that survives
    the slice, not just the in-memory viewer subsample. Output is written
    in the source-file frame (scan.ply / LAZ coordinates 1:1)."""
    scene = _state.get("scene")
    src = _state.get("source")
    if scene is None or src is None:
        raise HTTPException(409, "no scene loaded")
    if req.scene != scene:
        raise HTTPException(409, f"scene mismatch — server has '{scene}', request was '{req.scene}'")

    if req.ops is not None:
        ops = list(req.ops)
    elif req.boxes is not None:
        ops = [_ObbOp(op="keep", center=b.center, size=b.size, rotation=b.rotation)
               for b in req.boxes]
    else:
        ops = []

    offset = np.asarray(_state.get("recenter_offset") or [0.0, 0.0, 0.0], dtype=np.float64)
    scene_is_z_up = bool(_scene_is_z_up(src))

    if src.source_format == "laz":
        kept_xyz, kept_rgb = _stream_laz_keep(src.source_path, ops, scene_is_z_up, offset)
    elif src.source_format == "ply":
        from scenes.point_cloud import load_ply  # type: ignore
        full_pc, _ = load_ply(src.source_path)
        source_xyz = np.asarray(full_pc.points, dtype=np.float64)
        display = _to_display_frame(source_xyz, scene_is_z_up, offset)
        mask = _ops_chain_mask(display, ops)
        kept_xyz = source_xyz[mask]
        kept_rgb = (full_pc.colors[mask].astype(np.uint8, copy=False)
                    if full_pc.colors is not None else None)
    else:
        # GLB-sampled scenes: the sampled cloud _is_ the source — invert
        # the load-time transforms on the in-memory pc to land back in the
        # mesh frame.
        pc = _state.get("pc")
        if pc is None:
            raise HTTPException(409, "no scene loaded")
        display = np.asarray(pc.points, dtype=np.float64)
        mask = _ops_chain_mask(display, ops)
        kept = display[mask]
        if np.any(offset):
            kept = kept + offset
        if scene_is_z_up:
            kept = _y_up_to_z_up_xyz(kept)
        kept_xyz = kept
        kept_rgb = (pc.colors[mask].astype(np.uint8, copy=False)
                    if pc.colors is not None else None)

    return Response(content=_ply_response_bytes(kept_xyz, kept_rgb),
                    media_type="application/octet-stream")
