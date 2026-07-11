"""Voxa API routes: segment."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.constants import *  # noqa: F401,F403
from app.schemas import *  # noqa: F401,F403
from app.core import *  # noqa: F401,F403

router = APIRouter()


@router.post("/api/segment/apply")
def segment_apply(req: ApplyRequest):
    seg = _require_seg()
    try:
        if req.op == "set_class":
            idx = _decode_indices_or_400(req)
            out = seg.apply_set_class(idx, class_id=_coerce_class_id(req.payload["class_id"]))
        elif req.op == "merge":
            out = seg.apply_merge(
                source_inst=int(req.payload["source_inst"]),
                target_inst=int(req.payload["target_inst"]),
            )
        elif req.op == "reassign":
            idx = _decode_indices_or_400(req)
            out = seg.apply_reassign(
                idx,
                target_inst=req.payload.get("target_inst"),
                target_class=_coerce_class_id(req.payload.get("target_class")),
            )
        else:
            raise HTTPException(400, f"unknown op: {req.op}")
    except (KeyError, ValueError) as e:
        raise HTTPException(400, f"apply failed: {e}")
    return _serialize_apply(out)

@router.post("/api/segment/undo")
def segment_undo():
    seg = _require_seg()
    out = seg.undo()
    if out is None:
        return Response(status_code=204)
    return _serialize_delta(out)

@router.post("/api/segment/redo")
def segment_redo():
    seg = _require_seg()
    out = seg.redo()
    if out is None:
        return Response(status_code=204)
    return _serialize_delta(out)

@router.get("/api/segment/state", response_model=SegmentStateResponse)
def segment_state():
    """Return the active segment session if there is one (no-op otherwise).
    Used by the frontend to hydrate ``segState`` after a page reload."""
    seg = _state.get("seg")
    if seg is None:
        return SegmentStateResponse(has_state=False, has_seg=False, dirty=False)
    instance_ids = seg.instance_ids.astype(np.int32, copy=False)
    class_ids = seg.class_ids.astype(np.int8, copy=False)
    labeled = instance_ids >= 0
    box_ids, box_centers, box_sizes = _compute_segment_boxes(np.asarray(seg.positions), instance_ids)
    from labeling.segment_hulls import compute_hulls as _compute_hulls
    hull_v, hull_f, hull_seg = _compute_hulls(np.asarray(seg.positions), instance_ids)
    return SegmentStateResponse(
        has_state=True,
        has_seg=True,
        dirty=bool(seg.dirty),
        n_assigned=int(labeled.sum()),
        n_segments=int(np.unique(instance_ids[labeled]).size) if labeled.any() else 0,
        n_points=int(len(seg.instance_ids)),
        preseg_id=seg.preseg_id,
        preseg_fingerprint=seg.preseg_fingerprint,
        source_fingerprint=seg.source_fingerprint,
        is_from_prelabel=bool(seg.is_from_prelabel),
        full_class_ids=_b64(class_ids),
        full_instance_ids=_b64(instance_ids),
        seg_ids=_b64(box_ids),
        seg_centers=_b64(box_centers),
        seg_sizes=_b64(box_sizes),
        hull_vertices=_b64(hull_v),
        hull_faces=_b64(hull_f),
        hull_face_seg=_b64(hull_seg),
        session_id=_state.get("session_id"),
    )

def _apply_shape_core(seg, shape: dict, target_inst: int, target_class: int,
                      merged_from: list[int]) -> dict:
    """Shared core for /apply-shape and /centerline-apply: resolve a shape to
    full-res point indices, reassign them, and run the per-shape structure
    persist hook (tube -> update_centerlines; obb -> none)."""
    from labeling.shapes import shape_indices
    # Guard the tube session requirement BEFORE mutating working arrays, so a
    # session-less tube apply 409s cleanly instead of leaving partial state.
    if shape.get("type") == "tube" and seg.session_dir is None:
        raise HTTPException(409, "centerline labeling requires an active session")
    idx = shape_indices(np.asarray(seg.positions), shape)
    if idx.size == 0:
        # Same key-absence contract as _serialize_apply on an empty delta.
        return {"op": "apply-shape", "n_affected": 0, "dirty": bool(seg.dirty)}
    out = seg.apply_reassign(idx, target_inst=target_inst, target_class=target_class)
    # new_instance_id is only present on fresh allocation (target_inst < 0);
    # on re-apply the requested id is reused.
    instance_id = out.get("new_instance_id", target_inst)
    if shape.get("type") == "tube":
        from labeling.centerline import update_centerlines
        # merged_from re-capture correctness is the caller's contract: the spec
        # requires the request to carry the union of the absorbed instances'
        # paths, so their points land in idx above before we drop their entries.
        update_centerlines(seg.session_dir, instance_id, target_class,
                           shape["paths"], merged_from)
    body = _serialize_apply(out)
    body["instance_id"] = int(instance_id)
    return body


@router.post("/api/segment/apply-shape")
def apply_shape(req: ApplyShapeRequest):
    """Generic shape-based label apply: resolve `req.shape` (tube or obb) to
    full-res point indices and reassign them to (target_class, target_inst)."""
    seg = _require_seg()
    try:
        target_class = _coerce_class_id(req.target_class)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        return _apply_shape_core(seg, req.shape, req.target_inst,
                                 target_class, req.merged_from)
    except ValueError as e:            # unknown shape type
        raise HTTPException(400, str(e))


@router.post("/api/segment/centerline-apply")
def centerline_apply(req: CenterlineApplyRequest):
    """Label all full-res points within the tube(s) around the given
    centerline paths. Multiple paths in one call = one (merged) instance.
    See docs/superpowers/specs/2026-06-04-centerline-pipe-labeling-design.md."""
    seg = _require_seg()
    if seg.session_dir is None:
        raise HTTPException(409, "centerline labeling requires an active session")
    try:
        target_class = _coerce_class_id(req.target_class)
    except ValueError as e:
        raise HTTPException(400, str(e))
    paths = [p.model_dump() for p in req.paths]
    shape = {"type": "tube", "paths": paths}
    return _apply_shape_core(seg, shape, req.target_inst, target_class,
                             req.merged_from)

def _require_session_seg():
    """Active SegmentSession that has a session dir (409 otherwise) — shared
    by the centerline/structure persistence routes."""
    seg = _require_seg()
    if seg.session_dir is None:
        raise HTTPException(409, "no active session")
    return seg

@router.get("/api/segment/centerlines")
def get_centerlines():
    """Stored centerline paths for the active session (Draw sub-mode resume)."""
    from labeling.centerline import load_centerlines
    seg = _require_session_seg()
    return load_centerlines(seg.session_dir)

@router.get("/api/segment/structure")
def get_structure(session_id: str | None = None):
    """Stored Beam-tool graph for the active session (Beam sub-mode resume).
    `session_id` pins the read like the PUT pins the write: a remount racing
    a session switch must 409 loudly, never seed from the wrong session."""
    from labeling.beams import load_structure
    seg = _require_session_seg()
    if session_id is not None and session_id != _state.get("session_id"):
        raise HTTPException(
            409, f"session mismatch — server has '{_state.get('session_id')}', "
                 f"read was for '{session_id}'")
    return load_structure(seg.session_dir)

@router.put("/api/segment/structure")
def put_structure(doc: StructureDoc):
    """Replace the stored Beam-tool graph wholesale. The frontend owns graph
    geometry; point labels flow through apply-shape separately (not undoable
    here, matching centerlines.json)."""
    from labeling.beams import save_structure
    seg = _require_session_seg()
    if doc.session_id is not None and doc.session_id != _state.get("session_id"):
        raise HTTPException(
            409, f"session mismatch — server has '{_state.get('session_id')}', "
                 f"write was for '{doc.session_id}'")
    return save_structure(seg.session_dir, doc.model_dump(exclude={"session_id"}))

@router.put("/api/segment/save")
def segment_save():
    seg = _require_seg()
    session_id = _state.get("session_id")
    if session_id is None:
        raise HTTPException(409, "no active session — load a session before saving")
    src = _resolve(_state["scene"])
    if src.tier != "annotated":
        raise HTTPException(409, "Save is only supported on annotated/<scene> tier")
    scan_dir = Path(src.extras["scan_dir"])
    write_history = (
        os.environ.get("VOXA_DISABLE_ANNOTATION_HISTORY", "").strip().lower()
        not in ("1", "true", "yes", "on")
    )
    # Build a sanitized snapshot for the session output/ on the side; do NOT
    # touch in-memory state. SCHEMA invariant 3 (class==-1 ⟺ inst==-1, see
    # scan_schema.invariants.validate_invariants) requires stripping preseg-only
    # points (inst≥0, class=-1) on export because preseg is a suggestion,
    # not authoritative GT. The SegmentSession itself is the working
    # canvas with active preseg colors, so it MUST keep its full
    # instance_ids. The previous in-place mutation collapsed every
    # prelabel-derived segment into -1 on save AND leaked through the
    # session/working_*.npy autosave (which happens to run after this),
    # so reload couldn't recover preseg either.
    out_class = seg.class_ids
    out_inst = seg.instance_ids
    unclassified = (out_inst >= 0) & (out_class == -1)
    n_dropped = int(unclassified.sum())
    if n_dropped:
        out_inst = out_inst.copy()
        out_inst[unclassified] = np.int32(-1)
    # Autosave first so the recovery file reflects the unmutated working
    # canvas, independent of the output/ export.
    seg.flush_autosave()
    try:
        from labeling.segment_io import save_labels
        save_labels(
            scan_dir,
            session_id,
            class_ids=out_class,
            instance_ids=out_inst,
            positions=seg.positions,
            write_history=write_history,
            preseg_fingerprint=seg.preseg_fingerprint,
            source_fingerprint=seg.source_fingerprint,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    # The flush_autosave above persisted dirty:True; now that the export
    # succeeded, re-persist the cleared flag so the session list stops
    # reporting this session as unsaved after reload.
    seg.dirty = False
    seg.persist_aux()
    labeled = out_inst >= 0
    n_labeled_points = int(labeled.sum())
    n_segments = int(np.unique(out_inst[labeled]).size) if labeled.any() else 0
    return {
        "ok": True,
        "n_labeled_points": n_labeled_points,
        "n_segments": n_segments,
        "n_dropped_preseg": n_dropped,
    }
