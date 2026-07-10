"""Voxa API routes: export."""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException, Response

from app.constants import *  # noqa: F401,F403
from app.schemas import *  # noqa: F401,F403
from app.core import *  # noqa: F401,F403
from labeling.materialize import MaterializeCtx, collect_volumes

router = APIRouter()


def _build_materialize_ctx(scene: str, session_id: str):
    """Assemble a MaterializeCtx (+ the instance doc / confirmed / class maps)
    from the active session's in-memory state + on-disk files.

    Snapshots the in-memory state up front so a concurrent /api/load can't
    swap `_state["seg"]`/`_state["pc"]` out from under this read.
    """
    if _state.get("scene") != scene:
        raise HTTPException(
            409, f"scene mismatch — server has '{_state.get('scene')}', request was '{scene}'")
    if _state.get("session_id") != session_id:
        raise HTTPException(
            409, f"session mismatch — server has '{_state.get('session_id')}', request was '{session_id}'")

    seg = _state["seg"]
    pc = _state["pc"]
    src = _state["source"]
    offset = np.asarray(_state.get("recenter_offset") or [0.0, 0.0, 0.0])
    if seg is None:
        raise HTTPException(409, "no active session")

    p = _annotation_path(scene, "gt", session_id)
    instances = json.loads(p.read_text()).get("instances", []) if p.exists() else []

    from labeling.centerline import load_centerlines
    centerlines = load_centerlines(seg.session_dir)
    volumes = collect_volumes(instances, centerlines)

    class_id_by_inst: dict[int, int] = {}
    seq_by_inst: dict[int, "int | None"] = {}
    confirmed_by_inst: dict[int, bool] = {}
    for i in instances:
        sid = i.get("segId")
        if sid is None:
            continue
        sid = int(sid)
        class_id_by_inst[sid] = _coerce_class_id(i["cls"])
        seq_by_inst[sid] = i.get("seq")
        confirmed_by_inst[sid] = bool(i.get("confirmed"))

    # Phase A's replay_labels indexes inst_class_id[winner_inst] for every
    # instance id it sees in seg.instance_ids -> must cover all of them, not
    # just the ones with an instance-doc entry (e.g. blank/legacy sessions).
    present = np.unique(seg.instance_ids)
    for iid in present:
        iid = int(iid)
        if iid < 0 or iid in class_id_by_inst:
            continue
        k = int(np.argmax(seg.instance_ids == iid))
        class_id_by_inst[iid] = int(seg.class_ids[k])

    ctx = MaterializeCtx(
        scan_pos=pc.points,
        colors=pc.colors,
        work_cls=seg.class_ids,
        work_inst=seg.instance_ids,
        volumes=volumes,
        seq_by_inst=seq_by_inst,
        inst_class_id=class_id_by_inst,
        raw_path=src.extras.get("source_laz_path"),
        scene_is_z_up=_scene_is_z_up(src),
        offset=offset,
    )
    return ctx, instances, confirmed_by_inst, class_id_by_inst


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
