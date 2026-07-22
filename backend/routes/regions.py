"""Voxa API routes: eval regions (scan-level, eval-labeling phase 1).

Regions are SCAN truth — the store lives at the scan root and is shared by
every session on the scan — but all routes still require an active session:
CRUD needs the scan dir (resolved from the session dir), and stats/gate need
the session's FULL-RES positions (the ``_state`` cloud is subsampled to
VOXA_MAX_POINTS, which would inflate a measured p90). The wire format is the
runtime (recentered) frame; the file is the stored frame.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import *  # noqa: F401,F403
from app.core import *  # noqa: F401,F403
from routes.segment import _require_session_seg

from labeling import regions as regstore

router = APIRouter()


def _ctx():
    seg = _require_session_seg()
    scan_dir = seg.session_dir.parent.parent      # sessions/<id>/ -> scan root
    offset = [float(v) for v in (_state.get("recenter_offset") or (0.0, 0.0, 0.0))]
    return seg, scan_dir, offset


def _to_runtime(region: dict, offset) -> dict:
    out = dict(region)
    out["prism"] = regstore.shift_prism(region["prism"], [-v for v in offset])
    return out


@router.get("/api/regions")
def list_regions():
    _seg, scan_dir, off = _ctx()
    doc = regstore.load_regions(scan_dir)
    return {"regions": [_to_runtime(r, off) for r in doc["regions"]]}


@router.post("/api/regions")
def create_region(req: CreateRegionRequest):
    _seg, scan_dir, off = _ctx()
    doc = regstore.load_regions(scan_dir)
    try:
        region = regstore.create_region(
            doc, regstore.shift_prism(req.prism.model_dump(), off), req.name)
    except regstore.RegionError as e:
        raise HTTPException(422, str(e))
    regstore.save_regions(scan_dir, doc)
    return _to_runtime(region, off)


@router.patch("/api/regions/{rid}")
def patch_region(rid: int, req: PatchRegionRequest):
    seg, scan_dir, off = _ctx()
    if req.name is None and req.prism is None and req.status is None:
        raise HTTPException(422, "empty patch — send name, prism, or status")
    doc = regstore.load_regions(scan_dir)
    # Field order is LOAD-BEARING: `prism` is applied BEFORE `status`, so a
    # combined {"status":"draft","prism":X} hits set_geometry's eval-grade lock
    # and 422s. Swapped, one request could unlock and redraw a benchmark
    # region — see test_patch_cannot_unlock_and_redraw_in_one_request.
    try:
        region = None
        if req.name is not None:
            region = regstore.rename_region(doc, rid, req.name)
        if req.prism is not None:
            region = regstore.set_geometry(
                doc, rid, regstore.shift_prism(req.prism.model_dump(), off))
        if req.status is not None:
            region = regstore.flip_status(doc, rid, req.status, seg.positions, off)
    except regstore.RegionNotFound as e:
        raise HTTPException(404, str(e))
    except regstore.RegionError as e:
        raise HTTPException(422, str(e))
    regstore.save_regions(scan_dir, doc)
    return _to_runtime(region, off)


@router.delete("/api/regions/{rid}")
def delete_region(rid: int):
    _seg, scan_dir, off = _ctx()
    doc = regstore.load_regions(scan_dir)
    try:
        regstore.delete_region(doc, rid)
    except regstore.RegionNotFound as e:
        raise HTTPException(404, str(e))
    except regstore.RegionError as e:
        raise HTTPException(422, str(e))
    regstore.save_regions(scan_dir, doc)
    return {"ok": True}


@router.get("/api/regions/stats")
def regions_stats():
    seg, scan_dir, off = _ctx()
    doc = regstore.load_regions(scan_dir)
    return {"regions": regstore.region_stats(
        doc, seg.positions, seg.class_ids, seg.instance_ids, off)}
