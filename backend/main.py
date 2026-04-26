"""Voxa — unified 3D scan studio backend.

Endpoints serve the React/Three.js frontend with point clouds, cuboid
annotations, and GT-vs-prediction diffs. Scenes live under data/scenes/<name>/
with source.{ply,glb}, optional ground_truth.json (instance cuboids), and
optional predictions.json.
"""

from __future__ import annotations

import base64
import json
import os
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from point_cloud import PointCloud, load_glb, load_ply

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("VOXA_DATA_DIR", ROOT / "data"))
SCENES_DIR = DATA_DIR / "scenes"
ANNOT_DIR = DATA_DIR / "annotations"
CONFIG_PATH = Path(os.environ.get("VOXA_CONFIG", ROOT / "config" / "classes.yaml"))
FRONTEND_DIR = ROOT / "frontend"
MAX_POINTS_DEFAULT = int(os.environ.get("VOXA_MAX_POINTS", "300000"))

app = FastAPI(title="Voxa 3D scan studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── In-memory state ────────────────────────────────────────────────────────
# A single active point cloud per server instance keeps the surface area small;
# most workflows are scene-at-a-time anyway.
_state: dict[str, Any] = {
    "scene": None,
    "pc": None,           # PointCloud
    "mesh": None,         # trimesh.Trimesh or None
    "subsample_idx": None,  # indices into the original cloud (or None)
}


# ── Schemas ────────────────────────────────────────────────────────────────

class SceneInfo(BaseModel):
    name: str
    has_source: bool
    source_type: Optional[str]
    has_ground_truth: bool
    has_predictions: bool


class LoadRequest(BaseModel):
    name: str
    max_points: int = MAX_POINTS_DEFAULT


class LoadResponse(BaseModel):
    scene: str
    num_points: int
    num_subsampled: int
    bbox_min: list[float]
    bbox_max: list[float]
    positions: str   # base64 Float32Array (xyz)
    colors: str      # base64 Float32Array (rgb 0..1)


class Cuboid(BaseModel):
    id: str
    cls: str
    label: str = ""
    color: str = "#5b8def"
    center: list[float]   # [x,y,z]
    size: list[float]     # [w,h,d]
    rotation: list[float] = [0.0, 0.0, 0.0]   # euler xyz radians
    conf: float = 1.0
    source: str = "manual"   # 'manual' | 'auto' | 'fit'


class AnnotationDoc(BaseModel):
    scene: str
    kind: str   # 'gt' | 'pred'
    instances: list[Cuboid]
    meta: dict[str, Any] = {}


class SaveAnnotationRequest(AnnotationDoc):
    pass


class ClassDef(BaseModel):
    id: str
    label: str
    color: str
    hotkey: str = ""


class ConfigResponse(BaseModel):
    classes: list[ClassDef]


class CompareRequest(BaseModel):
    scene: str
    iou_threshold: float = 0.3


class DiffRow(BaseModel):
    gt_id: Optional[str]
    pred_id: Optional[str]
    cls: str
    status: str   # 'TP' | 'FP' | 'FN'
    iou: Optional[float]
    dpos: Optional[float]
    dsize: Optional[float]
    conf: Optional[float]


class CompareResponse(BaseModel):
    precision: float
    recall: float
    f1: float
    iou_mean: float
    tp: int
    fp: int
    fn: int
    rows: list[DiffRow]
    gt: list[Cuboid]
    pred: list[Cuboid]


# ── Helpers ────────────────────────────────────────────────────────────────

def _annotation_path(scene: str, kind: str) -> Path:
    if kind not in ("gt", "pred"):
        raise HTTPException(400, f"Invalid kind: {kind}")
    fname = "ground_truth.json" if kind == "gt" else "predictions.json"
    return ANNOT_DIR / scene / fname


def _scene_source(scene: str) -> tuple[Path, str]:
    sd = SCENES_DIR / scene
    if not sd.is_dir():
        raise HTTPException(404, f"Scene not found: {scene}")
    for ext in ("ply", "glb"):
        p = sd / f"source.{ext}"
        if p.exists():
            return p, ext
    raise HTTPException(404, f"No source.{{ply,glb}} in scene {scene}")


def _safe_subsample(pc: PointCloud, max_points: int) -> tuple[PointCloud, np.ndarray | None]:
    n = len(pc)
    if n <= max_points:
        return pc, None
    rng = np.random.default_rng(7)
    idx = rng.choice(n, size=max_points, replace=False)
    idx.sort()
    sub = PointCloud(
        points=pc.points[idx],
        colors=pc.colors[idx] if pc.colors is not None else None,
        labels=pc.labels[idx] if pc.labels is not None else None,
        instance_ids=pc.instance_ids[idx] if pc.instance_ids is not None else None,
    )
    return sub, idx


def _normalize_colors(pc: PointCloud) -> np.ndarray:
    """Return Nx3 float32 in 0..1 for the frontend."""
    if pc.colors is None:
        return np.full((len(pc), 3), 0.55, dtype=np.float32)
    c = pc.colors.astype(np.float32) / 255.0
    return c


def _b64f(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def _iou_aabb(a: Cuboid, b: Cuboid) -> float:
    """Axis-aligned IoU. Cuboid rotation is ignored — adequate for scoring
    industrial-pose annotations where rotation is usually small."""
    a_min = np.array(a.center) - np.array(a.size) / 2
    a_max = np.array(a.center) + np.array(a.size) / 2
    b_min = np.array(b.center) - np.array(b.size) / 2
    b_max = np.array(b.center) + np.array(b.size) / 2
    inter_min = np.maximum(a_min, b_min)
    inter_max = np.minimum(a_max, b_max)
    inter = np.maximum(0.0, inter_max - inter_min).prod()
    vol_a = np.array(a.size).prod()
    vol_b = np.array(b.size).prod()
    union = vol_a + vol_b - inter
    return float(inter / union) if union > 0 else 0.0


# ── Endpoints ──────────────────────────────────────────────────────────────

@app.get("/api/health")
def health():
    return {"status": "ok", "data_dir": str(DATA_DIR)}


@app.get("/api/config", response_model=ConfigResponse)
def get_config():
    if not CONFIG_PATH.exists():
        # Sensible defaults if no yaml is present.
        return ConfigResponse(classes=[
            ClassDef(id="boss",     label="Boss",       color="#5b8def", hotkey="1"),
            ClassDef(id="fastener", label="Fastener",   color="#f5a524", hotkey="2"),
            ClassDef(id="gasket",   label="Gasket",     color="#10b981", hotkey="3"),
            ClassDef(id="fitting",  label="Fitting",    color="#d4a017", hotkey="4"),
            ClassDef(id="rail",     label="Rail",       color="#a855f7", hotkey="5"),
            ClassDef(id="plate",    label="Base plate", color="#64748b", hotkey="6"),
            ClassDef(id="unknown",  label="Unknown",    color="#ef4444", hotkey="0"),
        ])
    with CONFIG_PATH.open() as f:
        raw = yaml.safe_load(f) or {}
    classes = []
    for i, (cid, body) in enumerate((raw.get("classes") or {}).items()):
        color = body.get("color", "#5b8def")
        if isinstance(color, list):
            r, g, b = (int(round(c * 255)) for c in color[:3])
            color = f"#{r:02x}{g:02x}{b:02x}"
        classes.append(ClassDef(
            id=cid,
            label=body.get("label", cid.title()),
            color=color,
            hotkey=str(body.get("key", body.get("hotkey", str(i + 1)))),
        ))
    return ConfigResponse(classes=classes)


@app.get("/api/scenes", response_model=list[SceneInfo])
def list_scenes():
    if not SCENES_DIR.exists():
        return []
    out: list[SceneInfo] = []
    for sd in sorted(SCENES_DIR.iterdir()):
        if not sd.is_dir():
            continue
        glb = sd / "source.glb"
        ply = sd / "source.ply"
        src_type = "glb" if glb.exists() else ("ply" if ply.exists() else None)
        gt = (ANNOT_DIR / sd.name / "ground_truth.json").exists()
        pr = (ANNOT_DIR / sd.name / "predictions.json").exists()
        out.append(SceneInfo(
            name=sd.name,
            has_source=src_type is not None,
            source_type=src_type,
            has_ground_truth=gt,
            has_predictions=pr,
        ))
    return out


@app.post("/api/load", response_model=LoadResponse)
def load_scene(req: LoadRequest):
    src, ext = _scene_source(req.name)
    if ext == "glb":
        pc, mesh = load_glb(src, num_samples=max(req.max_points, 50000))
    else:
        pc, mesh = load_ply(src)

    sub, idx = _safe_subsample(pc, req.max_points)
    _state.update(scene=req.name, pc=pc, mesh=mesh, subsample_idx=idx)

    positions = sub.points.astype(np.float32)
    colors = _normalize_colors(sub)
    bbox_min = positions.min(axis=0).tolist()
    bbox_max = positions.max(axis=0).tolist()

    return LoadResponse(
        scene=req.name,
        num_points=len(pc),
        num_subsampled=len(sub),
        bbox_min=bbox_min,
        bbox_max=bbox_max,
        positions=_b64f(positions),
        colors=_b64f(colors.astype(np.float32)),
    )


@app.get("/api/annotations/{scene}/{kind}", response_model=AnnotationDoc)
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


@app.put("/api/annotations/{scene}/{kind}")
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


@app.post("/api/compare/{scene}", response_model=CompareResponse)
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

    return CompareResponse(
        precision=round(precision, 3),
        recall=round(recall, 3),
        f1=round(f1, 3),
        iou_mean=round(iou_mean, 3),
        tp=tp, fp=fp, fn=fn,
        rows=rows, gt=gt, pred=pr,
    )


# ── Auto-fit helpers (re-exposing the legacy fitting hooks) ─────────────────

class AutoFitRequest(BaseModel):
    bbox_min: list[float]
    bbox_max: list[float]
    cls: str = "unknown"
    color: str = "#5b8def"
    label: str = ""


@app.post("/api/auto-fit", response_model=Cuboid)
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


# ── Static frontend ─────────────────────────────────────────────────────────
# Mounted last so /api/* takes precedence.
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")
