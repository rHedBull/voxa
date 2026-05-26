"""Voxa API routes: preseg."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.constants import *  # noqa: F401,F403
from app.schemas import *  # noqa: F401,F403
from app import constants
from app.core import *  # noqa: F401,F403

router = APIRouter()


@router.post("/api/segment/presegment", response_model=PresegmentResponse)
def segment_presegment(req: PresegmentRequest = PresegmentRequest()):
    """Run voxel presegmentation on the active scene's full-resolution
    points and return the new full arrays so the frontend can refresh
    its mirror.

    With ``preserve_labeled=True`` (default), only points whose
    ``class_id < 0`` are re-presegmented; points already assigned to a
    class keep their (class_id, instance_id) intact and the new
    supervoxel ids are renumbered to start above the highest existing
    instance id. With ``preserve_labeled=False`` the entire session is
    replaced.

    Bootstraps a fresh all-unlabeled session from the loaded cloud if no
    segment session exists yet. Slow on real clouds (~10–60 s for 1 M
    points); blocks until done. Clears undo/redo.
    """
    from preseg.presegment import presegment as _run_presegment

    seg = _state.get("seg")
    if seg is not None:
        positions = seg.positions
        existing_class = seg.class_ids.copy()
        existing_inst = seg.instance_ids.copy()
    else:
        pc = _state.get("pc")
        if pc is None:
            raise HTTPException(409, "No scene loaded — call /api/load first")
        if len(pc) > constants.MAX_LABEL_POINTS:
            raise HTTPException(
                413,
                f"Cloud has {len(pc)} points; presegmentation is capped at "
                f"VOXA_MAX_LABEL_POINTS={constants.MAX_LABEL_POINTS}",
            )
        positions = pc.points
        n_init = int(positions.shape[0])
        existing_class = np.full(n_init, -1, dtype=np.int8)
        existing_inst = np.full(n_init, -1, dtype=np.int32)

    n_points = int(positions.shape[0])
    name_to_id = _voxa_class_name_to_id()

    keep_mask = (existing_class >= 0) if req.preserve_labeled else np.zeros(n_points, dtype=bool)
    redo_mask = ~keep_mask

    if redo_mask.any():
        sub_positions = np.asarray(positions[redo_mask], dtype=np.float64)
        scene_id = _state.get("scene") or "_unknown"
        ransac_params = req.ransac.model_dump(exclude_none=True) if req.ransac else None

        sam3_features = None
        sam3_seen = None
        sam3_kwargs = {}
        if (req.mode == "ransac" and req.sam3 is not None
                and req.sam3.render_dirs):
            import preseg.sam3_features as _sam3
            # SCHEMA v1.2: renders live under <scan_dir>/renders/. Accept
            # any render_dir that is either inside the active scene's
            # renders/ tree OR under the legacy VOXA_RENDERS_ROOT fallback.
            src = _resolve(scene_id) if "/" in scene_id else None
            scan_dir_str = src.extras.get("scan_dir") if src and src.extras else None
            allowed_roots: list[Path] = []
            if scan_dir_str:
                allowed_roots.append((Path(scan_dir_str) / "renders").resolve())
            allowed_roots.append(_sam3.renders_root().resolve())
            resolved: list[Path] = []
            for p in req.sam3.render_dirs:
                rp = Path(p).resolve()
                if not any(_is_under(rp, root) for root in allowed_roots):
                    raise HTTPException(
                        400,
                        f"render_dir {p!r} is not under any allowed root: {allowed_roots}",
                    )
                if not (rp / "manifest.json").exists():
                    raise HTTPException(400, f"missing manifest.json in {p}")
                resolved.append(rp)
            # Cache features under <scan_dir>/sam3/ when we have one
            # (SCHEMA v1.2); else legacy PRESEG_RUNS_DIR fallback.
            cache_dir = (Path(scan_dir_str) / "sam3") if scan_dir_str else PRESEG_RUNS_DIR
            sam3_features, sam3_seen, _meta = _sam3.extract_or_load(
                sub_positions, scene_id,
                render_dirs=resolved,
                cache_dir=cache_dir,
                fpn_level=int(req.sam3.fpn_level),
                pca_dim=int(req.sam3.pca_dim),
                force=bool(req.sam3.force_recompute),
                log=print,
            )
            sam3_kwargs = {
                "feature_split_min_size": int(req.sam3.split_min_size),
                "feature_split_target_size": int(req.sam3.split_target_size),
                "feature_split_max_k": int(req.sam3.split_max_k),
            }
        else:
            sam3_kwargs = {}

        sub_inst, sub_summary = _run_presegment(
            sub_positions,
            mode=req.mode,
            class_map=name_to_id,
            log=lambda *_: None,
            resolution=float(req.resolution),
            ransac_params=req.ransac.model_dump(exclude_none=True) if req.ransac else None,
            labeler_strict=req.labeler_strict,
            features=sam3_features,
            feature_seen=sam3_seen,
            **sam3_kwargs,
        )
    else:
        sub_inst = np.empty(0, dtype=np.int32)
        sub_summary = []

    (
        sess,
        instance_ids,
        class_ids,
        box_ids,
        box_centers,
        box_sizes,
        hull_v,
        hull_f,
        hull_seg,
    ) = _apply_ransac_result_to_session(
        sub_inst=sub_inst,
        sub_summary=sub_summary,
        positions=positions,
        existing_class=existing_class,
        existing_inst=existing_inst,
        keep_mask=keep_mask,
        redo_mask=redo_mask,
        resolution=float(req.resolution),
    )
    _state["seg"] = sess
    _state["seg"].dirty = True

    labeled_mask = instance_ids >= 0
    n_assigned = int(labeled_mask.sum())
    n_segments = int(np.unique(instance_ids[labeled_mask]).size) if labeled_mask.any() else 0
    mean_seg_size = (n_assigned / n_segments) if n_segments > 0 else 0.0
    return PresegmentResponse(
        n_assigned=n_assigned,
        n_segments=n_segments,
        mean_seg_size=mean_seg_size,
        full_class_ids=_b64(class_ids.astype(np.int8)),
        full_instance_ids=_b64(instance_ids.astype(np.int32)),
        is_from_prelabel=True,
        seg_ids=_b64(box_ids),
        seg_centers=_b64(box_centers),
        seg_sizes=_b64(box_sizes),
        hull_vertices=_b64(hull_v),
        hull_faces=_b64(hull_f),
        hull_face_seg=_b64(hull_seg),
    )

@router.get("/api/sam3/renders", response_model=Sam3RendersResponse)
def sam3_list_renders(scene: Optional[str] = None):
    """List render runs for a scene from ``<scan_dir>/renders/`` (SCHEMA
    v1.2). When the active scene is annotated, ``scan_dir`` is resolved
    via the registry; otherwise we fall back to legacy
    ``VOXA_RENDERS_ROOT/<scene>/``. ``scene`` is the bare scene name; if
    omitted we use the active scene's basename.
    """
    import preseg.sam3_features as _sam3
    active_scene_id = _state.get("scene") or ""
    if not scene:
        scene = active_scene_id.split("/", 1)[-1] if active_scene_id else ""
    scan_dir: Optional[Path] = None
    if active_scene_id and "/" in active_scene_id:
        try:
            src = _resolve(active_scene_id)
            sd = src.extras.get("scan_dir") if src.extras else None
            if sd:
                scan_dir = Path(sd)
        except HTTPException:
            scan_dir = None
    runs = _sam3.discover_render_runs(scene or "", scan_dir=scan_dir) if (scene or scan_dir) else []
    root = scan_dir / "renders" if scan_dir is not None else _sam3.renders_root()
    return Sam3RendersResponse(
        scene=scene or "",
        root=str(root),
        root_exists=root.exists(),
        runs=[
            Sam3RenderRun(
                path=str(r.path), name=r.name, scene=r.scene,
                n_frames=r.n_frames, has_orbit_target=r.has_orbit_target,
                mtime=r.mtime,
            )
            for r in runs
        ],
    )

@router.post("/api/segment/presegment/optimize", response_model=PresegOptimizeStartResponse)
def segment_presegment_optimize(req: PresegOptimizeRequest = PresegOptimizeRequest()):
    existing = _state.get("preseg_opt_job")
    if existing and existing["status"] == "running":
        raise HTTPException(409, "An optimization is already running")

    seg = _state.get("seg")
    if seg is not None:
        positions = seg.positions
        existing_class = seg.class_ids.copy()
        existing_inst = seg.instance_ids.copy()
    else:
        pc = _state.get("pc")
        if pc is None:
            raise HTTPException(409, "No scene loaded — call /api/load first")
        positions = pc.points
        n_init = int(positions.shape[0])
        existing_class = np.full(n_init, -1, dtype=np.int8)
        existing_inst = np.full(n_init, -1, dtype=np.int32)

    n_points = int(positions.shape[0])
    keep_mask = (existing_class >= 0) if req.preserve_labeled else np.zeros(n_points, dtype=bool)
    redo_mask = ~keep_mask

    job = _new_job_state(req.n_trials)
    _state["preseg_opt_job"] = job
    t = _threading.Thread(
        target=_preseg_optimize_worker,
        kwargs=dict(
            job=job,
            positions=positions,
            existing_class=existing_class,
            existing_inst=existing_inst,
            keep_mask=keep_mask,
            redo_mask=redo_mask,
            subsample_n=req.subsample_n,
            n_trials=req.n_trials,
            class_map=_voxa_class_name_to_id(),
        ),
        daemon=True,
    )
    job["thread"] = t
    t.start()
    return PresegOptimizeStartResponse(job_id=job["id"], total=req.n_trials)

@router.get("/api/segment/presegment/optimize/status", response_model=PresegOptimizeStatusResponse)
def segment_presegment_optimize_status(job_id: str):
    job = _state.get("preseg_opt_job")
    if not job or job["id"] != job_id:
        raise HTTPException(404, "Unknown job_id")
    return PresegOptimizeStatusResponse(
        status=job["status"],
        trial=job["trial"],
        total=job["total"],
        best_score=job["best_score"],
        best_params=job["best_params"],
        error=job["error"],
    )

@router.post("/api/segment/presegment/optimize/abort")
def segment_presegment_optimize_abort(job_id: str):
    job = _state.get("preseg_opt_job")
    if not job or job["id"] != job_id:
        raise HTTPException(404, "Unknown job_id")
    job["cancel"].set()
    return {"status": "aborting"}
