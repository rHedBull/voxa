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
    from labeling.segment_io import sam_segments_to_list
    sam_segments = sam_segments_to_list(seg.sam_segments)
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
        full_sam_ids=_b64(seg.sam_ids.astype(np.int32, copy=False)),
        sam_segments=sam_segments,
        session_id=_state.get("session_id"),
    )

def _apply_shape_core(seg, shape: dict, target_inst: int, target_class: int,
                      merged_from: list[int],
                      protect_instances: list[int] | None = None) -> dict:
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
        return {"op": "apply-shape", "n_affected": 0, "n_protected": 0,
                "dirty": bool(seg.dirty)}
    out = seg.apply_reassign(idx, target_inst=target_inst, target_class=target_class,
                             protect_instances=protect_instances)
    if out["n_affected"] == 0:
        # Every point inside the shape belonged to a protected (confirmed)
        # instance — nothing written, no instance created. Surface n_protected
        # so the UI can say "skipped N locked points" instead of "empty box".
        return {"op": "apply-shape", "n_affected": 0,
                "n_protected": out.get("n_protected", 0), "dirty": bool(seg.dirty)}
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


EXCLUDE_CLASS_ID = 6   # config/classes.yaml -> unknown, "Exclude / Review"


def _denoise_core(seg, req: "DenoiseRequest") -> dict:
    """Shared core for /denoise: run global statistical-outlier detection
    over the whole cloud and materialize the flagged points as one
    unconfirmed Exclude pointset (Feature C)."""
    from labeling.outliers import statistical_outlier_indices
    # Backend-owned re-run replacement: erase the prior denoise instance's
    # points to unlabeled BEFORE recomputing (deleteInstance on the frontend
    # only drops the row, never the working-array labels).
    if req.replace_inst is not None:
        old = np.flatnonzero(seg.instance_ids == int(req.replace_inst)).astype(np.int32)
        if old.size:
            seg.apply_reassign(old, target_inst=None, target_class=None)
    all_idx = np.arange(seg.positions.shape[0], dtype=np.int64)
    outliers = statistical_outlier_indices(
        seg.positions, all_idx, k=int(req.k), std_ratio=float(req.std_ratio))
    if outliers.size == 0:
        return {"instance_id": None, "n_affected": 0, "n_protected": 0,
                "scan_indices_b64": None, "dirty": bool(seg.dirty)}
    out = seg.apply_reassign(
        outliers.astype(np.int32), target_inst=-1, target_class=EXCLUDE_CLASS_ID,
        protect_instances=req.protect_instances or None)
    if out["n_affected"] == 0:            # everything caught was locked/confirmed
        return {"instance_id": None, "n_affected": 0,
                "n_protected": out.get("n_protected", 0),
                "scan_indices_b64": None, "dirty": bool(seg.dirty)}
    return {"instance_id": int(out["new_instance_id"]),
            "n_affected": int(out["n_affected"]),
            "n_protected": out.get("n_protected", 0),
            "scan_indices_b64": _b64(out["indices"].astype(np.int32, copy=False)),
            "dirty": bool(seg.dirty)}


@router.post("/api/segment/denoise", response_model=DenoiseResponse)
def denoise(req: DenoiseRequest):
    """Global outlier detection: flag spatial outliers cloud-wide and
    materialize them as one unconfirmed Exclude pointset (Feature C).
    See docs/superpowers/specs/2026-07-20-outlier-detection-filtering-design.md."""
    seg = _require_seg()
    return _denoise_core(seg, req)


@router.post("/api/segment/denoise-selection", response_model=DenoiseSelectionResponse)
def denoise_selection(req: DenoiseSelectionRequest):
    """Per-selection "remove outliers" (Feature B): strip a selection's
    spatial outliers back to unlabeled. SAM candidate -> drop candidacy;
    unconfirmed instance -> erase to (-1,-1). Presegs are out of scope."""
    from labeling.outliers import statistical_outlier_indices
    seg = _require_seg()
    if req.source == "sam":
        membership = seg.sam_ids == int(req.id)
    else:
        membership = seg.instance_ids == int(req.id)
    subset = np.flatnonzero(membership).astype(np.int64)
    if subset.size == 0:
        raise HTTPException(404, f"{req.source} selection {req.id} is empty")
    outliers = statistical_outlier_indices(
        seg.positions, subset, k=int(req.k), std_ratio=float(req.std_ratio))
    n_kept = int(subset.size - outliers.size)
    if outliers.size == 0:
        return DenoiseSelectionResponse(source=req.source, id=req.id,
                                        n_removed=0, n_kept=n_kept, dirty=bool(seg.dirty))
    out_i32 = outliers.astype(np.int32)
    if req.source == "sam":
        seg.remove_sam_points(out_i32)
    else:
        seg.apply_reassign(out_i32, target_inst=None, target_class=None)
    return DenoiseSelectionResponse(
        source=req.source, id=req.id, n_removed=int(outliers.size), n_kept=n_kept,
        scan_indices_b64=_b64(out_i32), dirty=bool(seg.dirty))


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
                                 target_class, req.merged_from,
                                 req.protect_instances)
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
                             req.merged_from, req.protect_instances)

def _cut_shape_core(seg, shape: dict, sources: list, protect_instances: list[int] | None) -> dict:
    """Shared core for /cut-shape: resolve `shape` to full-res indices, then
    partition that index set per `sources` entry against that source's own
    full-res membership (preseg_ids / sam_ids / instance_ids), and
    materialize each non-empty partition — preseg/sam partitions become new
    sam-layer candidates, an instance partition becomes a freshly classified
    pointset that inherits the source instance's class. Sources never merge:
    each partition is materialized independently, one entry per non-empty
    source."""
    from labeling.shapes import shape_indices
    idx = shape_indices(np.asarray(seg.positions), shape)
    materialized: list[dict] = []
    instance_out: dict | None = None
    n_protected = 0
    if idx.size == 0:
        return {"materialized": materialized, "instance": instance_out, "n_protected": 0}
    for src in sources:
        if src.kind == "preseg":
            # instance_ids, not the immutable preseg_ids: PresegmentList
            # shows every id in segState.summary (i.e. instance_ids), not
            # only genuine preseg-sourced rows, and tags all of them
            # kind:'preseg' on cut — matches the frontend's own model
            # (buildCutCloud already tests instanceFull for 'preseg'
            # sources). A real, still-unclassified preseg segment has
            # preseg_ids == instance_ids anyway, so this is a no-op for the
            # common case and only changes behavior for orphaned/no-preseg
            # groups that would otherwise silently fail to cut.
            membership = seg.instance_ids == src.seg_id
        elif src.kind == "sam":
            membership = seg.sam_ids == src.seg_id
        elif src.kind == "instance":
            membership = seg.instance_ids == src.seg_id
        else:
            # Unreachable via HTTP — CutShapeSource.kind is a Pydantic Literal,
            # so FastAPI 422s an unknown kind before this route body runs.
            # Kept as a default-case guard for direct callers of this helper.
            raise ValueError(f"unknown source kind: {src.kind!r}")
        partition = idx[membership[idx]]
        if partition.size == 0:
            continue
        if src.kind in ("preseg", "sam"):
            out = seg.materialize_sam_segment(partition, source=src.kind,
                                              protect_instances=protect_instances)
            n_protected += out.get("n_protected", 0)
            if out["sam_seg_id"] is None:
                continue
            materialized.append({"sam_seg_id": out["sam_seg_id"], "source": src.kind,
                                 "n_points": out["n_affected"],
                                 "scan_indices_b64": _b64(out["indices"].astype(np.int32, copy=False))})
        else:  # instance
            src_class = int(seg.class_ids[np.flatnonzero(membership)[0]])
            out = seg.apply_reassign(partition, target_inst=-1, target_class=src_class,
                                     protect_instances=protect_instances)
            n_protected += out.get("n_protected", 0)
            if out["n_affected"] == 0:
                continue
            instance_out = {"instance_id": out["new_instance_id"], "n_points": out["n_affected"],
                            "scan_indices_b64": _b64(out["indices"].astype(np.int32, copy=False))}
    return {"materialized": materialized, "instance": instance_out, "n_protected": n_protected}


@router.post("/api/segment/cut-shape")
def cut_shape(req: CutShapeRequest):
    """Cut a client-drawn shape out of one or more source selections
    (presegments / SAM candidates / an instance), preserving the source
    boundary — points are partitioned per source before materializing, never
    merged across sources. See docs/superpowers/specs/2026-07-14-cut-selection-tool-design.md."""
    seg = _require_seg()
    try:
        return _cut_shape_core(seg, req.shape, req.sources, req.protect_instances)
    except ValueError as e:
        raise HTTPException(400, str(e))


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
