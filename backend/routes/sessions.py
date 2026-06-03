"""Voxa API routes: labeling sessions + preseg results (scan-schema v2)."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.constants import *  # noqa: F401,F403
from app.schemas import *  # noqa: F401,F403
from app.core import *  # noqa: F401,F403

router = APIRouter()


def _annotated_layout(scene_id: str):
    from scenes.scan_layout import ScanLayout
    src = _resolve(scene_id)
    if src.tier != "annotated":
        raise HTTPException(409, "sessions exist only on annotated/<scene>")
    return src, ScanLayout(Path(src.extras["scan_dir"]))


@router.get("/api/scenes/{tier}/{name}/sessions")
def sessions_list(tier: str, name: str):
    from labeling.session_store import list_sessions
    _, lay = _annotated_layout(f"{tier}/{name}")
    return {"sessions": [asdict(s) for s in list_sessions(lay)]}


@router.post("/api/scenes/{tier}/{name}/sessions")
def sessions_create(tier: str, name: str, req: CreateSessionRequest):
    from labeling.session_store import create_session
    src, lay = _annotated_layout(f"{tier}/{name}")
    # n_points + source_fp must describe the cloud as the loader sees it →
    # require the scene to be loaded (same single-cloud model as /segment/*).
    # source_fp was computed ONCE by /api/load and stashed in _state — the
    # pin definition ("fingerprint of the loaded, recentered cloud") lives
    # in exactly one place; never recompute it here.
    if _state.get("scene") != src.scene_id or _state.get("source_fp") is None:
        raise HTTPException(409, "load the scene before creating a session")
    try:
        info = create_session(lay, name=req.name, preseg_id=req.preseg_id,
                              n_points=len(_state["pc"]),
                              source_fp=_state["source_fp"],
                              min_segment_points=MIN_SEGMENT_POINTS)
    except (FileNotFoundError, ValueError, FileExistsError) as e:
        raise HTTPException(400, str(e))
    return asdict(info)


@router.patch("/api/scenes/{tier}/{name}/sessions/{sid}")
def sessions_rename(tier: str, name: str, sid: str, req: RenameSessionRequest):
    from labeling.session_store import rename_session
    _, lay = _annotated_layout(f"{tier}/{name}")
    try:
        rename_session(lay, sid, req.name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"ok": True}


@router.delete("/api/scenes/{tier}/{name}/sessions/{sid}")
def sessions_delete(tier: str, name: str, sid: str):
    # The UI's confirm dialog is the destructive-action guard; no backend
    # confirm handshake (single-user local tool, no other client).
    from labeling.session_store import delete_session
    _, lay = _annotated_layout(f"{tier}/{name}")
    try:
        delete_session(lay, sid)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    if _state.get("session_id") == sid:
        _state["seg"] = None
        _state["session_id"] = None
    return {"ok": True}


@router.get("/api/scenes/{tier}/{name}/presegs")
def presegs_list(tier: str, name: str):
    from preseg.preseg_store import list_presegs
    _, lay = _annotated_layout(f"{tier}/{name}")
    try:
        return {"presegs": [asdict(p) for p in list_presegs(lay)]}
    except ValueError as e:
        raise HTTPException(500, str(e))
