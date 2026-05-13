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
import threading as _threading
import time as _time
import uuid
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import yaml
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from lidar_io import LabelArrays, load_annotated, load_laz, load_laz_region
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
MAX_POINTS_DEFAULT = int(os.environ.get("VOXA_MAX_POINTS", "1000000"))
MAX_LABEL_POINTS = int(os.environ.get("VOXA_MAX_LABEL_POINTS", "5000000"))
# Drop segments smaller than this from on-disk labels. Defends against
# broken prelabel files (e.g. one-point-per-instance arrays) that would
# otherwise flood the UI with hundreds of thousands of micro-segments.
MIN_SEGMENT_POINTS = int(os.environ.get("VOXA_MIN_SEGMENT_POINTS", "10"))
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
    prefer_prelabel: bool = False          # if True, skip GT and surface model recommendation


class LoadResponse(BaseModel):
    scene: str                             # tier-prefixed canonical id
    num_points: int                        # in-memory full-resolution count
    num_points_total: Optional[int] = None # source file truth (e.g. LAZ header) when > num_points
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
    subsample_idx: Optional[str] = None      # b64 Int32, len==num_subsampled, maps sub row → full idx
    seg_ids: Optional[str] = None            # b64 Int32 — segment ids (full-res, for voxel box overlay)
    seg_centers: Optional[str] = None        # b64 Float32 (N×3) — bbox centres
    seg_sizes: Optional[str] = None          # b64 Float32 (N×3) — bbox extents


class LoadRegionRequest(BaseModel):
    aabb_min: list[float]    # in loaded frame (post recenter)
    aabb_max: list[float]
    max_points: Optional[int] = None   # None → no cap, return everything inside


class LoadRegionResponse(BaseModel):
    num_points: int                   # what we returned (post-cap)
    num_in_region_total: int          # before any cap
    positions: str                    # b64 Float32 (xyz, recentered frame)
    colors: Optional[str] = None      # b64 Float32 (rgb 0..1)


class Cuboid(BaseModel):
    # Despite the name, this model now covers both cuboid and pointset
    # instances. Pointsets carry `kind="pointset"` + `segId`, and have
    # null center/size. Compare-mode IoU skips pointset instances.
    id: str
    cls: str
    label: str = ""
    color: str = "#5b8def"
    center: Optional[list[float]] = None   # [x,y,z]; null for pointset
    size: Optional[list[float]] = None     # [w,h,d]; null for pointset
    rotation: list[float] = [0.0, 0.0, 0.0]   # euler xyz radians
    conf: float = 1.0
    source: str = "manual"   # 'manual' | 'auto' | 'fit' | 'preseg' | 'recommendation'
    confirmed: bool = False  # set true via Ctrl+Enter; hides interior points in main view
    kind: str = "cuboid"     # 'cuboid' | 'pointset'
    segId: Optional[int] = None  # set for pointset (and preseg-promoted) instances; per-point membership key in segState.instanceFull


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
    iou_mean: float                 # mean IoU over the 1:1 TPs (conditional on match)
    coverage_loose: float           # fraction of GT with best-pred IoU ≥ 0.1 (any pred)
    coverage_strict: float          # fraction of GT with best-pred IoU ≥ 0.3 (any pred)
    best_iou_mean: float            # mean of best-pred IoU per GT (overall recommendation tightness)
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
    industrial-pose annotations where rotation is usually small. Returns 0
    if either instance lacks a box (e.g. pointset)."""
    if a.center is None or a.size is None or b.center is None or b.size is None:
        return 0.0
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


def _filter_tiny_segments(labels, min_points: int):
    """Reset (class_id, instance_id) to (-1, -1) for any point belonging
    to an instance with fewer than ``min_points`` points. Keeps the
    arrays in-place-friendly: returns a fresh LabelArrays so the caller
    can swap without mutating the loader's outputs."""
    from lidar_io import LabelArrays
    inst = np.asarray(labels.instance_ids, dtype=np.int32)
    cls = np.asarray(labels.class_ids, dtype=np.int8)
    if inst.size == 0 or min_points <= 1:
        return LabelArrays(class_ids=cls.copy(), instance_ids=inst.copy())
    labeled = inst >= 0
    if not labeled.any():
        return LabelArrays(class_ids=cls.copy(), instance_ids=inst.copy())
    ids, counts = np.unique(inst[labeled], return_counts=True)
    drop_ids = ids[counts < int(min_points)]
    if drop_ids.size == 0:
        return LabelArrays(class_ids=cls.copy(), instance_ids=inst.copy())
    drop_mask = np.isin(inst, drop_ids)
    new_cls = cls.copy()
    new_inst = inst.copy()
    new_cls[drop_mask] = -1
    new_inst[drop_mask] = -1
    return LabelArrays(class_ids=new_cls, instance_ids=new_inst)


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


def _load_scene_source(src: SceneSource, max_points: int, *,
                       prefer_prelabel: bool = False):
    """Dispatch to the right loader.

    Returns (pc, mesh, intensity, labels, palette, n_classes, n_instances,
             n_labeled_points, is_from_prelabel, n_source_total).

    `n_source_total` is the source file's true point count when it differs
    from the loaded count (LAZ stride-samples at read time); None when the
    loaded count already matches the on-disk count.
    """
    if src.tier == "annotated":
        a = load_annotated(src, LIDAR_ROOT, prefer_prelabel=prefer_prelabel)
        n_labeled = int((a.labels.class_ids >= 0).sum()) if a.labels is not None else 0
        palette = [ClassDef(id=p.id, label=p.label, color=p.color) for p in a.palette]
        return (a.pc, None, a.intensity, a.labels, palette, a.n_classes, a.n_instances,
                n_labeled, bool(a.is_from_prelabel), None)

    if src.tier == "raw":
        pc, intensity, n_source_total = load_laz(src.source_path, max_points=max(max_points, 50_000))
        return (pc, None, intensity, None, None, None, None, None, False, n_source_total)

    # legacy + decimated → reuse the existing loaders
    if src.source_format == "glb":
        pc, mesh = load_glb(src.source_path, num_samples=max(max_points, 50_000))
        return (pc, mesh, None, None, None, None, None, None, False, None)
    pc, _ = load_ply(src.source_path)
    return (pc, None, None, None, None, None, None, None, False, None)


@app.post("/api/load", response_model=LoadResponse)
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
        pc=pc,
        mesh=mesh,
        subsample_idx=idx,
        intensity=intensity,
        labels=labels,
        recenter_offset=offset,
    )
    from segment_state import SegmentSession
    from segment_io import (
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
    elif labels is not None and len(pc) <= MAX_LABEL_POINTS:
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
        subsample_idx=subsample_idx_b64,
        **full_payload,
    )


@app.post("/api/load-region", response_model=LoadRegionResponse)
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


# `kind` comes first so `scene:path` can greedily match tier-prefixed ids
# like `annotated/smart_ais` (Starlette decodes `%2F` back to `/` during
# routing, so a `{scene}/{kind}` template would split such ids).
@app.get("/api/annotations/{kind}/{scene:path}", response_model=AnnotationDoc)
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


@app.put("/api/annotations/{kind}/{scene:path}")
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


@app.post("/api/compare/{scene:path}", response_model=CompareResponse)
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


# ── Segment endpoints ────────────────────────────────────────────────────────

class BrushQueryRequest(BaseModel):
    center: list[float]
    radius: float
    camera_ray: Optional[list[float]] = None
    depth_cull: Optional[float] = None


class BrushQueryResponse(BaseModel):
    indices: str        # b64 Int32
    n: int


class ApplyRequest(BaseModel):
    op: str
    indices: Optional[str] = None     # b64 Int32; required for set_class & reassign
    payload: dict


def _require_seg():
    seg = _state.get("seg")
    if seg is None:
        raise HTTPException(409, "No segment session loaded — load an annotated scene first")
    return seg


def _serialize_apply(out: dict) -> dict:
    body = {"op": out["op"], "n_affected": out["n_affected"], "dirty": True}
    if "new_instance_id" in out:
        body["new_instance_id"] = int(out["new_instance_id"])
    if "indices" in out:
        body["indices"] = _b64(out["indices"].astype(np.int32))
        body["after_class"] = _b64(out["after_class"].astype(np.int8))
        body["after_instance"] = _b64(out["after_instance"].astype(np.int32))
    return body


def _serialize_delta(out: dict) -> dict:
    return {
        "op": out["op"], "direction": out["direction"],
        "n_affected": out["n_affected"],
        "indices": _b64(out["indices"].astype(np.int32)),
        "after_class": _b64(out["after_class"].astype(np.int8)),
        "after_instance": _b64(out["after_instance"].astype(np.int32)),
    }


@app.post("/api/segment/brush-query", response_model=BrushQueryResponse)
def brush_query(req: BrushQueryRequest):
    seg = _require_seg()
    center = np.array(req.center, dtype=np.float32)
    cam = np.array(req.camera_ray, dtype=np.float32) if req.camera_ray else None
    idx = seg.brush_query(center, req.radius, camera_ray=cam, depth_cull=req.depth_cull)
    return BrushQueryResponse(indices=_b64(idx.astype(np.int32)), n=int(idx.size))


def _decode_indices_or_400(req: "ApplyRequest") -> np.ndarray:
    if req.indices is None:
        raise HTTPException(400, f"op '{req.op}' requires 'indices' (b64 Int32)")
    return np.frombuffer(base64.b64decode(req.indices), dtype=np.int32)


def _coerce_class_id(v):
    """Accept either int class id or string class name from the frontend.
    The labels palette uses string ids ('pipe', 'beam', ...); the seg
    state stores int8 class ids. Map names → ids via the configured
    classes.yaml so both wire formats work."""
    if v is None:
        return None
    if isinstance(v, (int, np.integer)):
        return int(v)
    name_to_id = _voxa_class_name_to_id()
    key = str(v).lower()
    if key not in name_to_id:
        raise ValueError(f"unknown class name: {v!r}")
    return name_to_id[key]


@app.post("/api/segment/apply")
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


@app.post("/api/segment/undo")
def segment_undo():
    seg = _require_seg()
    out = seg.undo()
    if out is None:
        return Response(status_code=204)
    return _serialize_delta(out)


@app.post("/api/segment/redo")
def segment_redo():
    seg = _require_seg()
    out = seg.redo()
    if out is None:
        return Response(status_code=204)
    return _serialize_delta(out)


def _voxa_class_name_to_id() -> dict[str, int]:
    """Build {class-name-lower: int-id} from the configured classes.yaml.

    Mirrors the enumeration order used by ``get_config()`` so id↔name
    stays consistent with the palette the frontend renders.
    """
    if not CONFIG_PATH.exists():
        return {}
    raw = yaml.safe_load(CONFIG_PATH.read_text()) or {}
    return {str(name).lower(): i
            for i, name in enumerate((raw.get("classes") or {}).keys())}


def _compute_segment_boxes(positions: np.ndarray, instance_ids: np.ndarray) -> tuple:
    """Compute per-segment bounding box as center + size (float32).

    Returns (seg_ids int32[N], centers float32[N,3], sizes float32[N,3]).
    Vectorised via argsort + reduceat: O(n log n) not O(n_seg × n_points).
    """
    pos = np.asarray(positions, dtype=np.float32)
    mask = instance_ids >= 0
    if not mask.any():
        empty3 = np.empty((0, 3), dtype=np.float32)
        return np.empty(0, dtype=np.int32), empty3, empty3
    pos_v = pos[mask]
    ids_v = instance_ids[mask]
    order = np.argsort(ids_v, kind='stable')
    ids_s = ids_v[order]
    pos_s = pos_v[order]
    unique_ids, starts = np.unique(ids_s, return_index=True)
    mins = np.minimum.reduceat(pos_s, starts)
    maxs = np.maximum.reduceat(pos_s, starts)
    centers = ((mins + maxs) / 2).astype(np.float32)
    sizes = np.maximum(maxs - mins, 0.01).astype(np.float32)
    return unique_ids.astype(np.int32), centers, sizes


class RansacParams(BaseModel):
    """Per-call overrides for ``presegment_ransac.RANSAC_DEFAULTS``. All
    fields optional — unset fields fall back to the hardcoded defaults."""
    plane_distance_threshold: Optional[float] = None
    plane_min_inliers: Optional[float] = None
    max_planes: Optional[float] = None
    plane_cluster_eps: Optional[float] = None
    leftover_cluster_eps: Optional[float] = None
    leftover_min_pts: Optional[float] = None
    flat_thresh: Optional[float] = None
    cylinder_ratio_thresh: Optional[float] = None
    cyl_search_radius: Optional[float] = None
    cyl_axis_thresh: Optional[float] = None
    cyl_radius_ratio: Optional[float] = None
    cyl_distance_threshold: Optional[float] = None
    merge_axis_dot: Optional[float] = None
    merge_radius_ratio: Optional[float] = None


class PresegmentRequest(BaseModel):
    mode: Literal["voxel", "ransac", "model"] = "voxel"
    resolution: float = 0.05      # voxel size in scene units (voxel mode only)
    preserve_labeled: bool = True  # only re-presegment points with class_id < 0
    ransac: Optional[RansacParams] = None  # ransac mode overrides
    labeler_strict: bool = False  # ransac mode: bit-for-bit industrial_point_labeler pipeline


class PresegmentResponse(BaseModel):
    n_assigned: int
    n_segments: int
    mean_seg_size: float = 0.0  # n_assigned / n_segments (0 if no segments)
    full_class_ids: str        # b64 Int8
    full_instance_ids: str     # b64 Int32
    is_from_prelabel: bool = True
    seg_ids: str = ""          # b64 Int32  — segment ids in order
    seg_centers: str = ""      # b64 Float32 (N×3) — bbox centres
    seg_sizes: str = ""        # b64 Float32 (N×3) — bbox extents
    # Per-segment convex hulls, packed into one merged geometry. Frontend
    # builds a single THREE.BufferGeometry from these (3d-labeler style).
    hull_vertices: str = ""    # b64 Float32 (V×3)
    hull_faces: str = ""       # b64 Int32   (F×3)  — global vertex indices
    hull_face_seg: str = ""    # b64 Int32   (F,)   — segment id per face


def _apply_ransac_result_to_session(
    *,
    sub_inst: np.ndarray,
    sub_summary: list[dict],
    positions: np.ndarray,
    existing_class: np.ndarray,
    existing_inst: np.ndarray,
    keep_mask: np.ndarray,
    redo_mask: np.ndarray,
    resolution: float = 0.05,
):
    """Renumber sub-instance ids above the existing max, assemble the new
    session arrays, compute boxes + hulls, and return everything. Caller
    is responsible for writing the session into ``_state["seg"]``."""
    from segment_state import SegmentSession
    from segment_hulls import compute_hulls as _compute_hulls

    id_offset = int(existing_inst.max(initial=-1)) + 1 if keep_mask.any() else 0
    if id_offset and sub_inst.size:
        sub_inst = np.where(sub_inst >= 0, sub_inst + id_offset, sub_inst)

    instance_ids = existing_inst.copy()
    instance_ids[redo_mask] = sub_inst

    class_ids = existing_class.copy()
    class_ids[redo_mask] = -1
    seg_to_class = {int(s["id"]) + id_offset: int(s.get("class_id", -1)) for s in sub_summary}
    for sid, cid in seg_to_class.items():
        if cid >= 0:
            class_ids[instance_ids == sid] = cid

    sess = SegmentSession(
        class_ids=class_ids,
        instance_ids=instance_ids.astype(np.int32),
        positions=positions,
        is_from_prelabel=True,
    )

    box_ids, box_centers, box_sizes = _compute_segment_boxes(
        np.asarray(positions), instance_ids
    )
    hull_v, hull_f, hull_seg = _compute_hulls(
        np.asarray(positions),
        instance_ids.astype(np.int32),
        resolution=float(resolution),
    )
    return (
        sess,
        instance_ids,
        class_ids,
        box_ids,
        box_centers,
        box_sizes,
        hull_v,
        hull_f,
        hull_seg,
    )


@app.post("/api/segment/presegment", response_model=PresegmentResponse)
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
    from presegment import presegment as _run_presegment

    seg = _state.get("seg")
    if seg is not None:
        positions = seg.positions
        existing_class = seg.class_ids.copy()
        existing_inst = seg.instance_ids.copy()
    else:
        pc = _state.get("pc")
        if pc is None:
            raise HTTPException(409, "No scene loaded — call /api/load first")
        if len(pc) > MAX_LABEL_POINTS:
            raise HTTPException(
                413,
                f"Cloud has {len(pc)} points; presegmentation is capped at "
                f"VOXA_MAX_LABEL_POINTS={MAX_LABEL_POINTS}",
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
        sub_inst, sub_summary = _run_presegment(
            sub_positions,
            mode=req.mode,
            class_map=name_to_id,
            log=lambda *_: None,
            resolution=float(req.resolution),
            ransac_params=req.ransac.model_dump(exclude_none=True) if req.ransac else None,
            labeler_strict=req.labeler_strict,
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


class PresegOptimizeRequest(BaseModel):
    n_trials: int = 20
    subsample_n: int = 200_000
    preserve_labeled: bool = True


class PresegOptimizeStartResponse(BaseModel):
    job_id: str
    total: int


class PresegOptimizeStatusResponse(BaseModel):
    status: Literal["running", "done", "aborted", "error"]
    trial: int
    total: int
    best_score: Optional[float] = None
    best_params: Optional[dict] = None
    error: Optional[str] = None


def _new_job_state(total: int) -> dict:
    return {
        "id": str(uuid.uuid4()),
        "thread": None,
        "cancel": _threading.Event(),
        "status": "running",
        "trial": 0,
        "total": total,
        "best_score": None,
        "best_params": None,
        "error": None,
        "started_at": _time.time(),
    }


def _preseg_optimize_worker(*, job, positions, existing_class, existing_inst,
                            keep_mask, redo_mask, subsample_n, n_trials, class_map):
    from preseg_optimize import run_study
    from presegment_ransac import presegment as _ransac
    try:
        candidate_idx = np.flatnonzero(redo_mask)
        if candidate_idx.size == 0:
            job["status"] = "error"
            job["error"] = "Nothing to optimize: all points are already labeled (preserve_labeled=true)"
            return
        if candidate_idx.size > subsample_n:
            rng = np.random.default_rng(0)
            sub_idx = rng.choice(candidate_idx, size=subsample_n, replace=False)
        else:
            sub_idx = candidate_idx
        xyz_sub = np.asarray(positions[sub_idx], dtype=np.float64)

        def cb(info):
            job["trial"] = info["trial"]
            job["best_score"] = info["best_score"]
            job["best_params"] = info["best_params"]

        result = run_study(
            xyz_sub,
            n_trials=n_trials,
            cancel_event=job["cancel"],
            progress_cb=cb,
            class_map=class_map,
        )
        job["best_score"] = result["best_score"]
        job["best_params"] = result["best_params"]

        if job["cancel"].is_set():
            job["status"] = "aborted"
            return

        sub_positions = np.asarray(positions[redo_mask], dtype=np.float64)
        sub_inst, sub_summary = _ransac(
            sub_positions,
            class_map=class_map,
            log=lambda *_: None,
            params=result["best_params"],
        )
        sess, *_unused = _apply_ransac_result_to_session(
            sub_inst=sub_inst,
            sub_summary=sub_summary,
            positions=positions,
            existing_class=existing_class,
            existing_inst=existing_inst,
            keep_mask=keep_mask,
            redo_mask=redo_mask,
            resolution=0.05,
        )
        _state["seg"] = sess
        _state["seg"].dirty = True
        job["status"] = "done"
    except Exception as e:  # noqa: BLE001
        job["status"] = "error"
        job["error"] = str(e)


@app.post("/api/segment/presegment/optimize", response_model=PresegOptimizeStartResponse)
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


@app.get("/api/segment/presegment/optimize/status", response_model=PresegOptimizeStatusResponse)
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


@app.post("/api/segment/presegment/optimize/abort")
def segment_presegment_optimize_abort(job_id: str):
    job = _state.get("preseg_opt_job")
    if not job or job["id"] != job_id:
        raise HTTPException(404, "Unknown job_id")
    job["cancel"].set()
    return {"status": "aborting"}


class SegmentStateResponse(BaseModel):
    """Snapshot of the in-memory segment session, returned to the frontend
    on page reload so the user doesn't have to re-run preseg every time
    they refresh the tab. Hulls are recomputed on demand (cheap relative
    to the original RANSAC run)."""
    has_state: bool
    has_seg: bool = False
    dirty: bool = False
    n_assigned: int = 0
    n_segments: int = 0
    full_class_ids: str = ""
    full_instance_ids: str = ""
    seg_ids: str = ""
    seg_centers: str = ""
    seg_sizes: str = ""
    hull_vertices: str = ""
    hull_faces: str = ""
    hull_face_seg: str = ""


@app.get("/api/segment/state", response_model=SegmentStateResponse)
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
    from segment_hulls import compute_hulls as _compute_hulls
    hull_v, hull_f, hull_seg = _compute_hulls(np.asarray(seg.positions), instance_ids)
    return SegmentStateResponse(
        has_state=True,
        has_seg=True,
        dirty=bool(seg.dirty),
        n_assigned=int(labeled.sum()),
        n_segments=int(np.unique(instance_ids[labeled]).size) if labeled.any() else 0,
        full_class_ids=_b64(class_ids),
        full_instance_ids=_b64(instance_ids),
        seg_ids=_b64(box_ids),
        seg_centers=_b64(box_centers),
        seg_sizes=_b64(box_sizes),
        hull_vertices=_b64(hull_v),
        hull_faces=_b64(hull_f),
        hull_face_seg=_b64(hull_seg),
    )


@app.put("/api/segment/save")
def segment_save():
    seg = _require_seg()
    src = _resolve(_state["scene"])
    if src.tier != "annotated":
        raise HTTPException(409, "Save is only supported on annotated/<scene> tier")
    scan_dir = Path(src.source_path).parent.parent
    write_history = (
        os.environ.get("VOXA_DISABLE_ANNOTATION_HISTORY", "").strip().lower()
        not in ("1", "true", "yes", "on")
    )
    # Drop unclassified preseg points (inst≥0, class=-1) so a save during
    # partial labeling succeeds: invariant 3 requires class==-1 ⟺ inst==-1,
    # and the preseg suggestion isn't authoritative until the user picks a
    # class for it. Mutating in place keeps in-memory state consistent with
    # what's about to land on disk.
    unclassified = (seg.instance_ids >= 0) & (seg.class_ids == -1)
    n_dropped = int(unclassified.sum())
    if n_dropped:
        seg.instance_ids[unclassified] = np.int32(-1)
    try:
        from segment_io import save_labels
        save_labels(
            scan_dir,
            class_ids=seg.class_ids,
            instance_ids=seg.instance_ids,
            positions=seg.positions,
            write_history=write_history,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    seg.dirty = False
    labeled = seg.instance_ids >= 0
    n_labeled_points = int(labeled.sum())
    n_segments = int(np.unique(seg.instance_ids[labeled]).size) if labeled.any() else 0
    return {
        "ok": True,
        "n_labeled_points": n_labeled_points,
        "n_segments": n_segments,
        "n_dropped_preseg": n_dropped,
    }


# ── Static frontend ─────────────────────────────────────────────────────────
# Mounted last so /api/* takes precedence. In dev, Vite serves the frontend
# directly and proxies /api/* to this backend; in production, run `npm run
# build` and the built bundle ends up in `dist/`.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
