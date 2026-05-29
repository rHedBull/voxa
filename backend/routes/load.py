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
        _load_scene_source(src, req.max_points, prefer_prelabel=req.prefer_prelabel)
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

    # Drop segments below a sanity threshold. Some prelabel files in the
    # wild are essentially `np.arange(n_points)` (every point its own
    # instance) which floods the UI with hundreds of thousands of
    # one-point segments and tanks rendering. Anything smaller than
    # MIN_SEGMENT_POINTS is treated as unlabeled (class=-1, instance=-1)
    # so the user starts from a clean slate.
    if labels is not None:
        labels = _filter_tiny_segments(labels, MIN_SEGMENT_POINTS)

    sub, idx, sub_intensity, sub_labels = _safe_subsample(pc, req.max_points, intensity, labels)

    # Preserve any live seg session (e.g. RANSAC presegmentation) across
    # page reloads when the same scene is reloaded with a compatible point
    # count. Without this guard, the user's preseg work is silently
    # clobbered by the on-disk labels on every /api/load.
    prev_scene = _state.get("scene")
    prev_seg = _state.get("seg")
    keep_prev_seg = (
        prev_seg is not None
        and prev_scene == src.scene_id
        and len(prev_seg.positions) == len(pc)
        # Switching GT↔prelabel reseats the source of truth; carrying
        # over the prior session would silently keep the old mode.
        and bool(prev_seg.is_from_prelabel) == bool(is_from_prelabel)
    )

    _state.update(
        scene=src.scene_id,
        source=src,
        pc=pc,
        mesh=mesh,
        subsample_idx=idx,
        intensity=intensity,
        labels=labels,
        recenter_offset=offset,
    )
    from labeling.segment_state import SegmentSession
    from labeling.segment_io import (
        compute_fingerprint,
        load_session_aux,
        load_working_arrays,
    )

    source_fp = compute_fingerprint(pc.points.astype(np.float32))
    session_dir = src.session_dir

    # Try recovering an in-progress working session (commit-pointer gated).
    recovered = None
    if session_dir is not None and not keep_prev_seg:
        aux = load_session_aux(session_dir)
        if aux is not None and aux.get("source_fingerprint") == source_fp:
            wa = load_working_arrays(session_dir, n_points=len(pc))
            if wa is not None:
                recovered = (wa[0], wa[1], aux)

    if keep_prev_seg:
        # Carry over the existing session as-is.
        _state["seg"] = prev_seg
    elif recovered is not None:
        wc, wi, aux = recovered
        seg = SegmentSession(
            class_ids=wc,
            instance_ids=wi,
            positions=pc.points,
            is_from_prelabel=bool(aux.get("is_from_prelabel", False)),
            session_dir=session_dir,
        )
        seg.source_fingerprint = source_fp
        seg.preseg_run_id = aux.get("preseg_run_id")
        seg.preseg_fingerprint = aux.get("preseg_fingerprint")
        seg.hidden_inst_ids = set(int(x) for x in aux.get("hidden_inst_ids", []))
        seg.dirty = bool(aux.get("dirty", False))
        _state["seg"] = seg
    elif labels is not None and len(pc) <= constants.MAX_LABEL_POINTS:
        seg = SegmentSession(
            class_ids=labels.class_ids,
            instance_ids=labels.instance_ids,
            positions=pc.points,
            is_from_prelabel=is_from_prelabel,
            session_dir=session_dir,
        )
        seg.source_fingerprint = source_fp
        _state["seg"] = seg
    else:
        _state["seg"] = None

    # Detect stale prelabel: labels/gt_segment_metadata.json may carry a
    # prelabel_fingerprint recorded when labels were seeded. If prelabel/
    # has been re-run since (different content hash), surface this so the
    # frontend can warn the user that their on-disk labels may be stale.
    _seg = _state.get("seg")
    if _seg is not None:
        _seg.stale_prelabel = False
        scan_dir_str = src.extras.get("scan_dir") if src.extras else None
        if scan_dir_str is None and src.tier == "annotated":
            scan_dir_str = str(Path(src.source_path).parent.parent)
        if scan_dir_str is not None:
            scan_dir = Path(scan_dir_str)
            labels_meta_path = scan_dir / "labels" / "gt_segment_metadata.json"
            prelabel_path = scan_dir / "prelabel" / "ransac_instance_ids.npy"
            if labels_meta_path.exists() and prelabel_path.exists():
                try:
                    import json as _json
                    saved_fp = _json.loads(labels_meta_path.read_text()).get("prelabel_fingerprint")
                    current_fp = compute_fingerprint(np.load(prelabel_path).astype(np.int32))
                    if saved_fp and current_fp and saved_fp != current_fp:
                        _seg.stale_prelabel = True
                        logging.warning(
                            "scene %s: prelabel/ has been re-run since labels/ were saved "
                            "(was %s, now %s) — labels may be stale",
                            src.scene_id, saved_fp, current_fp,
                        )
                except (OSError, ValueError, json.JSONDecodeError):
                    pass

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
    seg_for_meta = _state.get("seg")
    full_payload["is_from_prelabel"] = bool(seg_for_meta.is_from_prelabel) if seg_for_meta is not None else False

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
        georef_offset=fsum.get("georef_offset"),
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
