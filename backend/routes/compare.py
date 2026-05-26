"""Voxa API routes: compare."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Response

from app.constants import *  # noqa: F401,F403
from app.schemas import *  # noqa: F401,F403
from app.core import *  # noqa: F401,F403

router = APIRouter()


@router.get("/api/annotations/{kind}/{scene:path}", response_model=AnnotationDoc)
def get_annotation(scene: str, kind: str):
    p = _annotation_path(scene, kind)
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
def put_annotation(scene: str, kind: str, doc: SaveAnnotationRequest):
    p = _annotation_path(scene, kind)
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

@router.post("/api/compare/{scene:path}", response_model=CompareResponse)
def compare(scene: str, req: CompareRequest | None = None):
    iou_thr = (req.iou_threshold if req else 0.3)
    gt_doc = get_annotation(scene, "gt")
    pr_doc = get_annotation(scene, "pred")
    gt = gt_doc.instances
    pr = pr_doc.instances

    matched_pred: set[str] = set()
    matched_gt: set[str] = set()
    rows: list[DiffRow] = []
    ious: list[float] = []

    # Greedy match by best IoU within same class.
    for g in gt:
        best, best_iou = None, 0.0
        for p in pr:
            if p.cls != g.cls or p.id in matched_pred:
                continue
            iou = _iou_aabb(g, p)
            if iou > best_iou:
                best, best_iou = p, iou
        if best is not None and best_iou >= iou_thr:
            matched_pred.add(best.id)
            matched_gt.add(g.id)
            ious.append(best_iou)
            dpos = float(np.linalg.norm(np.array(g.center) - np.array(best.center)))
            ds = (np.array(best.size) - np.array(g.size)).sum() / max(np.array(g.size).sum(), 1e-6)
            rows.append(DiffRow(
                gt_id=g.id, pred_id=best.id, cls=g.cls, status="TP",
                iou=round(best_iou, 3), dpos=round(dpos, 3),
                dsize=round(float(ds), 3), conf=best.conf,
            ))
        else:
            rows.append(DiffRow(
                gt_id=g.id, pred_id=None, cls=g.cls, status="FN",
                iou=None, dpos=None, dsize=None, conf=None,
            ))

    for p in pr:
        if p.id not in matched_pred:
            rows.append(DiffRow(
                gt_id=None, pred_id=p.id, cls=p.cls, status="FP",
                iou=None, dpos=None, dsize=None, conf=p.conf,
            ))

    tp = sum(1 for r in rows if r.status == "TP")
    fp = sum(1 for r in rows if r.status == "FP")
    fn = sum(1 for r in rows if r.status == "FN")
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    iou_mean = float(np.mean(ious)) if ious else 0.0

    # Coverage / best-IoU: for each GT, best IoU against any pred regardless
    # of greedy 1:1 matching. Better fits the aided-labeling question of
    # "did the model find a usable starting point for this object?"
    best_per_gt: list[float] = []
    for g in gt:
        best = 0.0
        for p in pr:
            if p.cls != g.cls:
                continue
            iou = _iou_aabb(g, p)
            if iou > best:
                best = iou
        best_per_gt.append(best)
    if best_per_gt:
        coverage_loose = sum(1 for v in best_per_gt if v >= 0.1) / len(best_per_gt)
        coverage_strict = sum(1 for v in best_per_gt if v >= 0.3) / len(best_per_gt)
        best_iou_mean = float(np.mean(best_per_gt))
    else:
        coverage_loose = coverage_strict = best_iou_mean = 0.0

    return CompareResponse(
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        iou_mean=round(iou_mean, 3),
        coverage_loose=round(coverage_loose, 3),
        coverage_strict=round(coverage_strict, 3),
        best_iou_mean=round(best_iou_mean, 3),
        tp=tp, fp=fp, fn=fn,
        rows=rows, gt=gt, pred=pr,
    )

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
