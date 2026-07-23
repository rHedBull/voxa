"""Voxa API routes: export."""
from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import FileResponse
from starlette.background import BackgroundTask

from app.constants import *  # noqa: F401,F403
from app.schemas import *  # noqa: F401,F403
from app.core import *  # noqa: F401,F403
from labeling.materialize import (
    MaterializeCtx,
    collect_volumes,
    build_replay_index,
    materialize_downsample,
    materialize_raw,
    raw_reservoir_sample_spacing,
    raw_sample_spacing,
    loa_band,
)
from labeling.centerline import load_centerlines
from labeling.instance_meshes import build_instance_glbs
from labeling.export_pipeline import (
    apply_filters_remap,
    build_manifest,
    build_taxonomy,
    count_absent_instances,
    drop_unlabeled_rows,
    surviving_instance_ids,
    validate_export_request,
)

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
        try:
            class_id_by_inst[sid] = _coerce_class_id(i["cls"])
        except ValueError as e:
            # e.g. an instance saved under a class name since renamed/removed
            # in classes.yaml — diagnosable 400, not an unhandled 500.
            raise HTTPException(400, f"instance {i.get('id', sid)!r}: {e}")
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


@router.get("/api/labels/accuracy")
def labels_accuracy(scene: str, session_id: str) -> dict:
    """p50/p90 nearest-neighbor sample spacing — the labeling-density boundary
    uncertainty shown in the export wizard. Raw-backed (`is_raw: true`) when
    the scan has a registered raw source; otherwise a best-effort fallback
    against the loaded session cloud (`is_raw: false`) — this endpoint is
    informational, not a hard gate, so it degrades gracefully rather than
    refusing (contrast with the eval-grade region gate, which refuses)."""
    if _state.get("scene") != scene:
        raise HTTPException(409, f"scene mismatch — server has '{_state.get('scene')}', request was '{scene}'")
    if _state.get("session_id") != session_id:
        raise HTTPException(409, f"session mismatch — server has '{_state.get('session_id')}', request was '{session_id}'")
    pc = _state.get("pc")
    if pc is None:
        raise HTTPException(409, "no scene loaded")
    src = _state.get("source")
    raw_path = src.extras.get("source_laz_path") if src is not None else None
    if raw_path:
        offset = np.asarray(_state.get("recenter_offset") or [0.0, 0.0, 0.0], dtype=np.float64)
        p50, p90 = raw_reservoir_sample_spacing(raw_path, _scene_is_z_up(src), offset)
        is_raw = True
    else:
        p50, p90 = raw_sample_spacing(pc.points)
        is_raw = False
    return {"p50": p50, "p90": p90, "loa": loa_band(p90), "is_raw": is_raw}


@router.post("/api/labels/export")
def export_labels(req: ExportLabelsRequest) -> Response:
    """Export the active session's labels at any of scan/subsample/raw
    density as a zip of (scan_labeled.ply, manifest.json).

    Orchestrates the already-tested Phase B pieces (_build_materialize_ctx,
    validate_export_request, build_taxonomy/apply_filters_remap, the
    materialize_* helpers, and the _ply_labeled_* PLY writer) — this
    endpoint stays thin: no filtering/remap/materialize logic lives here.
    """
    from routes.meta import get_config

    config = get_config()
    palette = [{"class_id": c.class_id, "label": c.label, "color": c.color}
               for c in config.classes]
    palette_ids = {c.class_id for c in config.classes}

    ctx, instances, confirmed_by_inst, _class_id_by_inst = _build_materialize_ctx(
        req.scene, req.session_id)

    errs = validate_export_request(
        req, n_scan=len(ctx.scan_pos), palette_ids=palette_ids,
        raw_available=bool(ctx.raw_path))
    if errs:
        raise HTTPException(422, {"errors": errs})

    p50, p90 = raw_sample_spacing(ctx.scan_pos)
    taxonomy, src_to_tgt = build_taxonomy(palette, req)
    absent = count_absent_instances(ctx.work_inst, confirmed_by_inst)

    tmpdir = tempfile.mkdtemp(prefix="voxa_export_")
    try:
        ply_path = Path(tmpdir) / "scan_labeled.ply"

        if req.resolution.kind in ("scan", "subsample"):
            n = len(ctx.scan_pos) if req.resolution.kind == "scan" else req.resolution.n
            # Colorless sources load with colors=None; mirror the raw branch's
            # zeros fallback so subsample indexing and drop_unlabeled masking
            # never subscript None.
            colors = (ctx.colors if ctx.colors is not None
                      else np.zeros((len(ctx.scan_pos), 3), np.uint8))
            pos, col, cls, inst = materialize_downsample(
                ctx.scan_pos, colors, ctx.work_cls, ctx.work_inst, n)
            # materialize_downsample's identity branch (n >= N) returns the
            # live working arrays by reference — copy before mutating.
            cls = cls.copy()
            inst = inst.copy()
            out_cls, out_inst = apply_filters_remap(cls, inst, confirmed_by_inst, req, src_to_tgt)
            if req.drop_unlabeled:
                out_cls, pos, col, out_inst = drop_unlabeled_rows(out_cls, pos, col, out_inst)
            ply_path.write_bytes(_ply_labeled_bytes(pos, col, out_cls, out_inst))
            total = len(pos)

        elif req.resolution.kind == "raw":
            index = build_replay_index(
                ctx.scan_pos, ctx.work_inst, ctx.volumes, ctx.seq_by_inst, ctx.inst_class_id)
            body_path = Path(tmpdir) / "body.bin"
            total = 0
            with body_path.open("wb") as body:
                for xyz, rgb, cls, inst in materialize_raw(
                        index, ctx.raw_path, ctx.scene_is_z_up, ctx.offset):
                    oc, oi = apply_filters_remap(cls, inst, confirmed_by_inst, req, src_to_tgt)
                    rgb_use = rgb if rgb is not None else np.zeros((len(xyz), 3), np.uint8)
                    if req.drop_unlabeled:
                        oc, xyz, rgb_use, oi = drop_unlabeled_rows(oc, xyz, rgb_use, oi)
                    body.write(_ply_labeled_chunk_bytes(xyz, rgb_use, oc, oi))
                    total += len(xyz)
            with ply_path.open("wb") as out:
                out.write(_ply_labeled_header(total, has_color=True))
                with body_path.open("rb") as body:
                    shutil.copyfileobj(body, out)
            body_path.unlink()

        else:
            # Unreachable via normal flow (validate_export_request rejects
            # unknown kinds), but kept shape-consistent with the 422 above.
            raise HTTPException(422, {"errors": [f"unknown resolution kind: {req.resolution.kind!r}"]})

        manifest = build_manifest(
            taxonomy, p50, p90, scan=req.scene, session=req.session_id,
            resolution={"kind": req.resolution.kind}, points=total,
            confirmed_only=req.confirmed_only, include_classes=req.include_classes,
            drop_unlabeled=req.drop_unlabeled, absent_count=absent,
            exported_at=datetime.now(timezone.utc).isoformat(),
            labeling_points=len(ctx.scan_pos))

        zip_path = Path(tmpdir) / "export.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            zf.write(ply_path, "scan_labeled.ply")
            if req.include_meshes:
                surviving_ids = surviving_instance_ids(
                    ctx.work_cls, ctx.work_inst, confirmed_by_inst, req, src_to_tgt)
                glbs, skipped = build_instance_glbs(ctx.scan_pos, ctx.work_inst, surviving_ids)
                for iid, data in glbs.items():
                    zf.writestr(f"meshes/{iid}.glb", data)
                manifest["meshes"] = {
                    "written": len(glbs),
                    "skipped": [{"id": iid, "reason": reason} for iid, reason in skipped],
                }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
    except Exception:
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise

    return FileResponse(
        zip_path, media_type="application/zip",
        filename=f"scan_labeled_{req.resolution.kind}.zip",
        background=BackgroundTask(shutil.rmtree, tmpdir, ignore_errors=True))
