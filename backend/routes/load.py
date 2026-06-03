"""Voxa API routes: load."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.constants import *  # noqa: F401,F403
from app.schemas import *  # noqa: F401,F403
from app import constants
from app.core import *  # noqa: F401,F403

router = APIRouter()

@router.post("/api/load", response_model=LoadResponse)
def load_scene(req: LoadRequest):
    src = _resolve(req.name)
    (pc, mesh, intensity, labels, palette, n_classes, n_instances, n_labeled,
     is_from_prelabel, n_source_total) = (
        _load_scene_source(src, req.max_points)
    )

    # LAS / lidar archive scans are Z-up; rotate into Three.js Y-up before
    # any further processing so bbox / recenter / subsample all operate in
    # the display frame. The decision is per-scene (annotated/<scan>/meta.json
    # tells us whether the PLY was sampled from a Y-up GLB or a Z-up LAZ).
    is_z_up = _scene_is_z_up(src)
    if is_z_up:
        pc = _z_up_to_y_up(pc)

    # Recenter for float32 stability (LAS UTM, etc).
    pc, offset = _recenter(pc)

    from labeling.segment_io import compute_fingerprint
    source_fp = compute_fingerprint(pc.points.astype(np.float32))

    # v2: resolve the active session (explicit session_id or last-worked
    # default) and rebuild its in-memory state. Per-point labels live in the
    # session, not in the cloud loader. Pin mismatches surface as 409.
    sessions_meta: list[dict] = []
    session_id = None
    seg = None
    if src.tier == "annotated":
        from dataclasses import asdict
        from labeling.session_store import PinMismatch, last_worked, list_sessions
        lay = ScanLayout(Path(src.extras["scan_dir"]))
        infos = list_sessions(lay)              # one pass over session.json files
        sessions_meta = [asdict(s) for s in infos]
        if req.session_id and req.session_id not in {s.session_id for s in infos}:
            raise HTTPException(404, f"session {req.session_id!r} not found")
        session_id = req.session_id or last_worked(infos)
        if session_id is not None:
            try:
                seg = _resume_session(lay, session_id, pc, source_fp)
            except PinMismatch as e:
                raise HTTPException(status_code=409, detail={
                    "error": "session_pin_mismatch",
                    "diverged": e.diverged,
                    "session_id": session_id,
                    "message": str(e)})

    if seg is not None:
        labels = LabelArrays(class_ids=seg.class_ids, instance_ids=seg.instance_ids)
        valid_c = seg.class_ids[seg.class_ids >= 0]
        valid_i = seg.instance_ids[seg.instance_ids >= 0]
        n_classes = int(valid_c.max()) + 1 if valid_c.size else 0
        n_instances = int(valid_i.max()) + 1 if valid_i.size else 0
        n_labeled = int((seg.class_ids >= 0).sum())

    sub, idx, sub_intensity, sub_labels = _safe_subsample(pc, req.max_points, intensity, labels)

    _state.update(
        scene=src.scene_id,
        source=src,
        pc=pc,
        mesh=mesh,
        subsample_idx=idx,
        intensity=intensity,
        labels=labels,
        recenter_offset=offset,
        seg=seg,
        session_id=session_id,
        source_fp=source_fp,   # single home; session creation reads this
    )

    positions = sub.points.astype(np.float32)
    colors = _normalize_colors(sub)
    bbox_min = positions.min(axis=0).tolist()
    bbox_max = positions.max(axis=0).tolist()

    intensity_b64 = _b64(sub_intensity.astype(np.float32)) if sub_intensity is not None else None
    class_ids_b64 = _b64(sub_labels.class_ids.astype(np.int8)) if sub_labels is not None else None
    instance_ids_b64 = _b64(sub_labels.instance_ids.astype(np.int32)) if sub_labels is not None else None

    full_payload: dict[str, Any] = {}
    if req.want_full_labels and labels is not None:
        full_payload["full_class_ids"] = _b64(labels.class_ids.astype(np.int8))
        full_payload["full_instance_ids"] = _b64(labels.instance_ids.astype(np.int32))
        full_payload["full_positions"] = _b64(pc.points.astype(np.float32))
        full_payload["full_n"] = int(len(pc))
        ii = labels.instance_ids
        ci = labels.class_ids
        m = ii >= 0
        if m.any():
            uids, idx0, counts = np.unique(ii[m], return_index=True, return_counts=True)
            summary = {
                str(int(uid)): {"class_id": int(ci[m][idx0[k]]), "n_points": int(counts[k])}
                for k, uid in enumerate(uids)
            }
        else:
            summary = {}
        full_payload["segment_summary"] = summary
        if (labels.instance_ids >= 0).any():
            box_ids, box_centers, box_sizes = _compute_segment_boxes(
                pc.points, labels.instance_ids)
            full_payload["seg_ids"] = _b64(box_ids)
            full_payload["seg_centers"] = _b64(box_centers)
            full_payload["seg_sizes"] = _b64(box_sizes)
    full_payload["is_from_prelabel"] = seg.is_from_prelabel if seg else False

    subsample_idx_b64 = _b64(idx.astype(np.int32)) if idx is not None else None

    # scan-schema v1.3: surface the cloud's frame/provenance (additive; {} for
    # non-annotated tiers). Never let it break a load.
    fsum: dict = {}
    _scan_dir = src.extras.get("scan_dir")
    if _scan_dir:
        try:
            from scenes.scan_meta import frame_summary
            fsum = frame_summary(_scan_dir)
        except Exception:  # noqa: BLE001 — frame metadata is best-effort surface info
            fsum = {}

    # scan-schema v1.3 §6: verify the cloud registers to its renders. Block the
    # load (409) on a real mismatch; never block a scan we cannot verify; a check
    # bug must never break loading a good scan.
    frame_check = None
    if _scan_dir and (Path(_scan_dir) / "renders").is_dir():
        from preseg.registration import verify_scan_registration
        try:
            _v = verify_scan_registration(Path(_scan_dir))
        except Exception:  # noqa: BLE001 — degrade to "unverified", never break a good load
            _v = {"checked": False, "ok": True, "runs": [], "reasons": []}
        if _v["checked"] and not _v["ok"]:
            raise HTTPException(status_code=409, detail={
                "error": "frame_registration_failed",
                "message": ("Scan does not register to its renders (scan-schema v1.3 §6); "
                            "the cloud and render poses appear to be in different frames."),
                "scan": src.scene_id,
                "frame_check": _v,
            })
        if _v["checked"]:
            frame_check = _v

    return LoadResponse(
        scene=src.scene_id,
        num_points=len(pc),
        num_points_total=n_source_total if (n_source_total is not None and n_source_total > len(pc)) else None,
        num_subsampled=len(sub),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        positions=_b64(positions),
        colors=_b64(colors.astype(np.float32)),
        intensity=intensity_b64,
        class_ids=class_ids_b64,
        instance_ids=instance_ids_b64,
        class_palette=palette,
        n_classes=n_classes,
        n_instances=n_instances,
        n_labeled_points=n_labeled,
        recenter_offset=offset,
        mesh_url=_mesh_url_for(src),
        mesh_is_z_up=is_z_up if src.has_mesh else False,
        scene_is_z_up=is_z_up,
        subsample_idx=subsample_idx_b64,
        schema_version=fsum.get("schema_version"),
        variant_id=fsum.get("variant_id"),
        frame_canonical_id=fsum.get("frame_canonical_id"),
        frame_uncertain=bool(fsum.get("frame_uncertain", False)),
        frame_check=frame_check,
        georef_offset=fsum.get("georef_offset"),
        session_id=session_id,
        sessions=sessions_meta,
        **full_payload,
    )

@router.post("/api/load-region", response_model=LoadRegionResponse)
def load_region(req: LoadRegionRequest):
    """Return points inside an AABB at *full* density (no stride).

    For LAZ scenes the file is re-streamed and filtered chunk-by-chunk in the
    loaded frame (z-up→y-up + recenter). For PLY/GLB scenes (already loaded
    full-density into _state) we just numpy-mask. Used by the viewer to pop
    extra detail inside a selected cuboid without re-loading the whole scene.
    """
    scene_id = _state.get("scene")
    if not scene_id:
        raise HTTPException(400, "No scene loaded")
    src = _resolve(scene_id)

    aabb_min = np.asarray(req.aabb_min, dtype=np.float32)
    aabb_max = np.asarray(req.aabb_max, dtype=np.float32)
    if (aabb_max <= aabb_min).any():
        raise HTTPException(400, "aabb_max must be > aabb_min componentwise")

    # Annotated scenes annotate against the 500k–1M PLY but their per-point
    # ML labels need to land on the *original* LAZ. When the user selects a
    # cuboid we pop full density from the source LAZ that the PLY was sampled
    # from — so labeling is in the dense substrate, not the navigation proxy.
    laz_path: Optional[Path] = None
    if src.tier in ("decimated", "raw") and src.source_format == "laz":
        laz_path = src.source_path
    elif src.tier == "annotated":
        slp = src.extras.get("source_laz_path")
        if slp:
            laz_path = Path(slp)

    if laz_path is not None:
        offset = np.asarray(_state.get("recenter_offset") or [0.0, 0.0, 0.0], dtype=np.float64)
        positions, colors = load_laz_region(
            laz_path, aabb_min, aabb_max,
            is_z_up=_scene_is_z_up(src), offset=offset,
        )
    else:
        pc = _state.get("pc")
        if pc is None:
            raise HTTPException(400, "Scene state missing")
        m = ((pc.points >= aabb_min) & (pc.points <= aabb_max)).all(axis=1)
        positions = pc.points[m].astype(np.float32)
        if pc.colors is not None:
            colors = (pc.colors[m].astype(np.float32) / 255.0).astype(np.float32)
        else:
            colors = None

    n_in_region = int(len(positions))
    if req.max_points is not None and n_in_region > req.max_points:
        rng = np.random.default_rng(7)
        idx = rng.choice(n_in_region, size=req.max_points, replace=False)
        idx.sort()
        positions = positions[idx]
        if colors is not None:
            colors = colors[idx]

    return LoadRegionResponse(
        num_points=int(len(positions)),
        num_in_region_total=n_in_region,
        positions=_b64(positions),
        colors=_b64(colors) if colors is not None else None,
    )

@router.get("/api/mesh/{tier}/{name}")
def get_mesh(tier: str, name: str):
    """Stream the canonical mesh.glb for a scene. The frontend applies the
    same Z-up → Y-up rotation the loader applies to points so mesh and
    cloud overlay correctly. 404 when no mesh is available."""
    src = _resolve(f"{tier}/{name}")
    mesh_path = src.extras.get("mesh_path")
    if not mesh_path:
        raise HTTPException(404, f"No mesh for scene: {tier}/{name}")
    return FileResponse(mesh_path, media_type="model/gltf-binary",
                        filename=f"{name}.glb")
