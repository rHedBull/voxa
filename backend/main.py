"""Voxa — unified 3D scan studio backend.

Endpoints serve the React/Three.js frontend with point clouds, cuboid
annotations, and GT-vs-prediction diffs. Scenes come from multiple roots:

  legacy     voxa/data/scenes/<name>/source.{ply,glb}
  annotated  $VOXA_LIDAR_ROOT/annotated/<name>/source/scan.ply (+ labels/, meta.json)
  decimated  $VOXA_LIDAR_ROOT/ply_viewer/<name>.ply
  raw        $VOXA_LIDAR_ROOT/laz/<name>.laz

Scene IDs in /api/load and /api/annotations are tier-prefixed
('annotated/munich_water_pump'); a bare legacy name still resolves for
backward compatibility.
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
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lidar_io import LabelArrays, load_annotated, load_laz
from point_cloud import PointCloud, load_glb, load_ply
from scene_registry import (
    SceneSource,
    discover,
    load_lidar_root_from_env,
    resolve as resolve_scene,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.environ.get("VOXA_DATA_DIR", ROOT / "data"))
SCENES_DIR = DATA_DIR / "scenes"
ANNOT_DIR = DATA_DIR / "annotations"
CONFIG_PATH = Path(os.environ.get("VOXA_CONFIG", ROOT / "config" / "classes.yaml"))
FRONTEND_DIST = ROOT / "dist"
MAX_POINTS_DEFAULT = int(os.environ.get("VOXA_MAX_POINTS", "300000"))
LIDAR_ROOT = load_lidar_root_from_env()

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
    "pc": None,             # PointCloud (recentered)
    "mesh": None,           # trimesh.Trimesh or None
    "subsample_idx": None,  # indices into the original cloud (or None)
    "intensity": None,      # full-resolution intensity array (or None)
    "labels": None,         # LabelArrays (or None)
    "recenter_offset": [0.0, 0.0, 0.0],
    "seg": None,            # SegmentSession | None
}


# ── Schemas ────────────────────────────────────────────────────────────────

class SceneInfo(BaseModel):
    id: str                      # tier-prefixed
    tier: str                    # 'legacy' | 'annotated' | 'decimated' | 'raw'
    name: str
    has_source: bool
    source_format: Optional[str]   # 'ply' | 'glb' | 'laz'
    has_labels: bool
    has_intensity: bool
    has_mesh: bool                 # canonical mesh.glb available (annotated only)
    has_ground_truth: bool         # cuboid GT (legacy)
    has_predictions: bool          # cuboid pred (legacy)
    n_points: Optional[int] = None


class LoadRequest(BaseModel):
    name: str                              # tier-prefixed id or bare legacy name
    max_points: int = MAX_POINTS_DEFAULT
    want_full_labels: bool = False


class LoadResponse(BaseModel):
    scene: str                             # tier-prefixed canonical id
    num_points: int
    num_subsampled: int
    bbox_min: list[float]
    bbox_max: list[float]
    positions: str                         # b64 Float32 (xyz, recentered)
    colors: str                            # b64 Float32 (rgb 0..1)
    intensity: Optional[str] = None        # b64 Float32 (0..1)
    class_ids: Optional[str] = None        # b64 Int8 (-1 = unlabeled)
    instance_ids: Optional[str] = None     # b64 Int32 (-1 = unlabeled)
    class_palette: Optional[list["ClassDef"]] = None
    n_classes: Optional[int] = None
    n_instances: Optional[int] = None
    n_labeled_points: Optional[int] = None
    recenter_offset: list[float] = [0.0, 0.0, 0.0]
    mesh_url: Optional[str] = None    # /api/mesh/<id> when a GLB exists
    mesh_is_z_up: bool = False        # frontend rotates the GLB only if true
    full_class_ids: Optional[str] = None     # b64 Int8, full-res
    full_instance_ids: Optional[str] = None  # b64 Int32, full-res
    full_positions: Optional[str] = None     # b64 Float32 (xyz, recentered), full-res
    full_n: Optional[int] = None
    is_from_prelabel: bool = False
    segment_summary: Optional[dict] = None   # { "<inst>": {class_id, n_points} }


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
    id: str | int
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


LoadResponse.model_rebuild()


# ── Helpers ────────────────────────────────────────────────────────────────

def _annotation_path(scene: str, kind: str) -> Path:
    if kind not in ("gt", "pred"):
        raise HTTPException(400, f"Invalid kind: {kind}")
    fname = "ground_truth.json" if kind == "gt" else "predictions.json"
    # Annotation files key off the scene name (without tier) so that legacy
    # bare-name lookups continue to work. Tier collisions in this dir are
    # acceptable for now — Compare mode is cuboid-only and only legacy scenes
    # produce cuboid GT/pred today.
    safe = scene.replace("/", "__")
    return ANNOT_DIR / safe / fname


def _resolve(scene_id: str) -> SceneSource:
    try:
        return resolve_scene(scene_id, DATA_DIR, LIDAR_ROOT)
    except KeyError:
        raise HTTPException(404, f"Scene not found: {scene_id}")


def _safe_subsample(
    pc: PointCloud,
    max_points: int,
    intensity: Optional[np.ndarray] = None,
    labels: Optional[LabelArrays] = None,
) -> tuple[PointCloud, np.ndarray | None, Optional[np.ndarray], Optional[LabelArrays]]:
    n = len(pc)
    if n <= max_points:
        return pc, None, intensity, labels
    rng = np.random.default_rng(7)
    idx = rng.choice(n, size=max_points, replace=False)
    idx.sort()
    sub = PointCloud(
        points=pc.points[idx],
        colors=pc.colors[idx] if pc.colors is not None else None,
        labels=pc.labels[idx] if pc.labels is not None else None,
        instance_ids=pc.instance_ids[idx] if pc.instance_ids is not None else None,
    )
    sub_intensity = intensity[idx] if intensity is not None else None
    sub_labels: Optional[LabelArrays] = None
    if labels is not None:
        sub_labels = LabelArrays(
            class_ids=labels.class_ids[idx],
            instance_ids=labels.instance_ids[idx],
        )
    return sub, idx, sub_intensity, sub_labels


def _normalize_colors(pc: PointCloud) -> np.ndarray:
    """Return Nx3 float32 in 0..1 for the frontend."""
    if pc.colors is None:
        return np.full((len(pc), 3), 0.55, dtype=np.float32)
    c = pc.colors.astype(np.float32) / 255.0
    return c


def _b64(arr: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode()


def _recenter(pc: PointCloud) -> tuple[PointCloud, list[float]]:
    """Subtract the bbox centroid so float32 stays precise in Three.js.

    LAS UTM coordinates can reach 7 digits, which is at the edge of float32
    precision and shows up as visible jitter. Returns the recentered cloud
    plus the offset that was subtracted so any future export can restore
    world coords."""
    if len(pc) == 0:
        return pc, [0.0, 0.0, 0.0]
    center = pc.points.mean(axis=0)
    # Only recenter if the magnitude is large enough to matter; small scenes
    # near the origin shouldn't have their coords mutated for nothing.
    if float(np.max(np.abs(center))) < 1e3:
        return pc, [0.0, 0.0, 0.0]
    new_points = (pc.points - center).astype(np.float32)
    return PointCloud(
        points=new_points,
        colors=pc.colors,
        labels=pc.labels,
        instance_ids=pc.instance_ids,
        face_indices=pc.face_indices,
    ), center.astype(np.float32).tolist()


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
    return {
        "status": "ok",
        "data_dir": str(DATA_DIR),
        "lidar_root": str(LIDAR_ROOT) if LIDAR_ROOT else None,
    }


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
    out: list[SceneInfo] = []
    for s in discover(DATA_DIR, LIDAR_ROOT):
        annot_key = s.name if s.tier == "legacy" else s.scene_id.replace("/", "__")
        gt = (ANNOT_DIR / annot_key / "ground_truth.json").exists()
        pr = (ANNOT_DIR / annot_key / "predictions.json").exists()
        out.append(SceneInfo(
            id=s.scene_id,
            tier=s.tier,
            name=s.name,
            has_source=True,
            source_format=s.source_format,
            has_labels=s.has_labels,
            has_intensity=s.has_intensity,
            has_mesh=s.has_mesh,
            has_ground_truth=gt,
            has_predictions=pr,
            n_points=s.n_points,
        ))
    return out


def _z_up_to_y_up(pc: PointCloud) -> PointCloud:
    """Rotate a Z-up point cloud (LAS / surveying convention) into the
    Three.js Y-up frame so the floor sits below the cloud as expected.

    Right-handed Z-up (X right, Y depth, Z up) → right-handed Y-up
    (X right, Y up, Z back). Mapping: (x, y, z) → (x, z, -y).
    Per-point label / instance arrays carry over unchanged.
    """
    pts = pc.points
    new_pts = np.empty_like(pts)
    new_pts[:, 0] = pts[:, 0]
    new_pts[:, 1] = pts[:, 2]
    new_pts[:, 2] = -pts[:, 1]
    return PointCloud(
        points=new_pts,
        colors=pc.colors,
        labels=pc.labels,
        instance_ids=pc.instance_ids,
        face_indices=pc.face_indices,
    )


def _scene_is_z_up(src: SceneSource) -> bool:
    """Decide whether the scene's source frame is Z-up (surveying / LAS).

    - legacy: author-defined, treated as Y-up.
    - decimated, raw: pulled from LAZ, always Z-up.
    - annotated: depends on what the PLY was sampled from.
      `meta.json::source_mesh` (a glTF) → Y-up.
      `meta.json::source_laz` → Z-up.
      Default Z-up if the meta is missing or ambiguous.
    """
    if src.tier == "legacy":
        return False
    if src.tier in ("decimated", "raw"):
        return True
    return bool(src.extras.get("is_z_up", True))


def _load_scene_source(src: SceneSource, max_points: int):
    """Dispatch to the right loader.

    Returns (pc, mesh, intensity, labels, palette, n_classes, n_instances,
             n_labeled_points, is_from_prelabel).
    """
    if src.tier == "annotated":
        a = load_annotated(src, LIDAR_ROOT)
        n_labeled = int((a.labels.class_ids >= 0).sum()) if a.labels is not None else 0
        palette = [ClassDef(id=p.id, label=p.label, color=p.color) for p in a.palette]
        return (a.pc, None, a.intensity, a.labels, palette, a.n_classes, a.n_instances,
                n_labeled, bool(a.is_from_prelabel))

    if src.tier == "raw":
        pc, intensity = load_laz(src.source_path, max_points=max(max_points, 50_000))
        return (pc, None, intensity, None, None, None, None, None, False)

    # legacy + decimated → reuse the existing loaders
    if src.source_format == "glb":
        pc, mesh = load_glb(src.source_path, num_samples=max(max_points, 50_000))
        return (pc, mesh, None, None, None, None, None, None, False)
    pc, _ = load_ply(src.source_path)
    return (pc, None, None, None, None, None, None, None, False)


@app.post("/api/load", response_model=LoadResponse)
def load_scene(req: LoadRequest):
    src = _resolve(req.name)
    pc, mesh, intensity, labels, palette, n_classes, n_instances, n_labeled, is_from_prelabel = (
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

    sub, idx, sub_intensity, sub_labels = _safe_subsample(pc, req.max_points, intensity, labels)
    _state.update(
        scene=src.scene_id,
        pc=pc,
        mesh=mesh,
        subsample_idx=idx,
        intensity=intensity,
        labels=labels,
        recenter_offset=offset,
    )
    from segment_state import SegmentSession
    _state["seg"] = (
        SegmentSession(
            class_ids=labels.class_ids,
            instance_ids=labels.instance_ids,
            positions=pc.points,
            is_from_prelabel=is_from_prelabel,
        )
        if labels is not None else None
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
    seg_for_meta = _state.get("seg")
    full_payload["is_from_prelabel"] = bool(seg_for_meta.is_from_prelabel) if seg_for_meta is not None else False

    return LoadResponse(
        scene=src.scene_id,
        num_points=len(pc),
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
        **full_payload,
    )


def _mesh_url_for(src: SceneSource) -> Optional[str]:
    """Cache-bust the mesh URL with the GLB's mtime, so a rebuilt mesh.glb
    is treated as a fresh resource by both HTTP caches and Three.js's
    GLTFLoader cache (which keys on URL)."""
    if not src.has_mesh:
        return None
    mesh_path = src.extras.get("mesh_path")
    try:
        mtime = int(os.path.getmtime(mesh_path)) if mesh_path else 0
    except OSError:
        mtime = 0
    return f"/api/mesh/{src.scene_id}?v={mtime}"


@app.get("/api/mesh/{tier}/{name}")
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
# Mounted last so /api/* takes precedence. In dev, Vite serves the frontend
# directly and proxies /api/* to this backend; in production, run `npm run
# build` and the built bundle ends up in `dist/`.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
