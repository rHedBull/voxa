"""Voxa API routes: compare."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.constants import *  # noqa: F401,F403
from app.schemas import *  # noqa: F401,F403
from app.core import *  # noqa: F401,F403

router = APIRouter()


@router.get("/api/annotations/{kind}/{scene:path}", response_model=AnnotationDoc)
def get_annotation(scene: str, kind: str, session_id: str | None = None):
    p = _annotation_path(scene, kind, session_id)
    if not p.exists():
        return AnnotationDoc(scene=scene, kind=kind, instances=[])
    with p.open() as f:
        data = json.load(f)
    return AnnotationDoc(
        scene=scene,
        kind=kind,
        instances=[Cuboid(**c) for c in data.get("instances", [])],
        meta=data.get("meta", {}),
    )

@router.put("/api/annotations/{kind}/{scene:path}")
def put_annotation(scene: str, kind: str, doc: SaveAnnotationRequest,
                   session_id: str | None = None):
    p = _annotation_path(scene, kind, session_id)
    p.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "scene": scene,
        "kind": kind,
        "instances": [c.model_dump() for c in doc.instances],
        "meta": doc.meta,
    }
    with p.open("w") as f:
        json.dump(body, f, indent=2)
    return {"saved": str(p), "count": len(doc.instances)}

@router.post("/api/compare-points/{tier}/{name}")
def compare_points(tier: str, name: str, req: ComparePointsRequest):
    """Compare two finished per-point labelings of one scan. Reads both
    sources from disk — no dependency on the in-memory loaded scene."""
    from labeling.compare_points import compare_class_arrays
    from scan_schema.layout import ScanLayout

    src = _resolve(f"{tier}/{name}")
    if src.tier != "annotated":
        raise HTTPException(409, "compare-points needs an annotated/<scene> scan")
    lay = ScanLayout(Path(src.extras["scan_dir"]))

    def load_source(ref: SourceRef):
        if ref.kind == "session":
            sp = lay.session(ref.id)
            if not sp.dir.is_dir():
                raise HTTPException(404, f"session {ref.id!r} not found")
            if not sp.output_gt_class_ids.exists():
                raise HTTPException(409, (
                    f"session {ref.id!r} has no saved output — save it "
                    f"(Ctrl+S) before comparing"))
            return np.load(sp.output_gt_class_ids).astype(np.int32)
        if ref.kind == "preseg":
            from preseg.preseg_store import load_preseg, read_preseg_meta
            try:
                # Validates the id (no path traversal) and 404s a missing
                # preseg before any path join below.
                read_preseg_meta(lay, ref.id)
                # n_points for load_preseg's shape check: the cloud's
                # declared count when meta carries it, else the preseg's own
                # length — a truncated OTHER source must hit the cross-source
                # 409 below, not a shape 400 in here, and meta-less scans
                # must still work.
                inst_path = lay.preseg_dir(ref.id) / "instance_ids.npy"
                n = int(src.n_points
                        or np.load(inst_path, mmap_mode="r").shape[0])
                class_ids, _ = load_preseg(lay, ref.id, n_points=n)
            except FileNotFoundError as e:
                raise HTTPException(404, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))
            return class_ids.astype(np.int32)
        raise HTTPException(400, f"unknown source kind {ref.kind!r}")

    a = load_source(req.a)
    b = load_source(req.b)
    if a.shape != b.shape:
        raise HTTPException(409, (
            f"sources cover different clouds: a has {a.shape[0]} points, "
            f"b has {b.shape[0]}"))

    from scenes.lidar_io import build_class_palette
    metrics = compare_class_arrays(a, b)
    palette = build_class_palette(constants.LIDAR_ROOT)
    return {
        "metrics": metrics,
        "a_class_ids": _b64(a.astype(np.int8)),
        "b_class_ids": _b64(b.astype(np.int8)),
        "palette": [p.__dict__ for p in palette],
    }

@router.post("/api/auto-fit", response_model=Cuboid)
def auto_fit(req: AutoFitRequest):
    """Snap a cuboid to the points inside the given AABB. If no PC is loaded
    or the box is empty, the request box is returned verbatim so the caller
    still gets a usable cuboid."""
    pc = _state.get("pc")
    cmin = np.array(req.bbox_min)
    cmax = np.array(req.bbox_max)

    fitted_center = (cmin + cmax) / 2
    fitted_size = (cmax - cmin)

    if pc is not None:
        pts = pc.points
        mask = np.all((pts >= cmin) & (pts <= cmax), axis=1)
        if mask.sum() >= 50:
            sel = pts[mask]
            lo = sel.min(axis=0)
            hi = sel.max(axis=0)
            fitted_center = ((lo + hi) / 2).tolist()
            fitted_size = (hi - lo + 0.005).tolist()

    return Cuboid(
        id=f"inst-{uuid.uuid4().hex[:8]}",
        cls=req.cls,
        label=req.label or req.cls.title(),
        color=req.color,
        center=list(fitted_center),
        size=list(fitted_size),
    )
