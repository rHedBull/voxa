"""Pydantic request/response models for the Voxa API."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel

from app.constants import MAX_POINTS_DEFAULT


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
    scene_is_z_up: bool = False       # true if backend applied z-up→y-up on load;
                                      # frontend must invert when exporting a PLY
    full_class_ids: Optional[str] = None     # b64 Int8, full-res
    full_instance_ids: Optional[str] = None  # b64 Int32, full-res
    full_positions: Optional[str] = None     # b64 Float32 (xyz, recentered), full-res
    full_n: Optional[int] = None
    is_from_prelabel: bool = False
    segment_summary: Optional[dict] = None   # { "<inst>": {class_id, n_points} }
    subsample_idx: Optional[str] = None      # b64 Int32, len==num_subsampled, maps sub row → full idx
    # scan-schema v1.3 frame/provenance (annotated tier; None for legacy/laz tiers)
    schema_version: Optional[str] = None
    variant_id: Optional[str] = None
    frame_canonical_id: Optional[str] = None
    frame_uncertain: bool = False            # true ⇒ frame synthesized from legacy coords (unverified)
    georef_offset: Optional[list[float]] = None
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

class AutoFitRequest(BaseModel):
    bbox_min: list[float]
    bbox_max: list[float]
    cls: str = "unknown"
    color: str = "#5b8def"
    label: str = ""

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

class Sam3RenderRun(BaseModel):
    """One render directory found under VOXA_RENDERS_ROOT for a scene."""
    path: str
    name: str
    scene: str
    n_frames: int
    has_orbit_target: bool
    mtime: float

class Sam3RendersResponse(BaseModel):
    scene: str
    root: str
    root_exists: bool
    runs: list[Sam3RenderRun]

class Sam3Params(BaseModel):
    """SAM3 feature-aware sub-segmentation (ransac mode only).

    When ``render_dirs`` is non-empty the backend extracts per-point
    SAM3 features from the listed render directories, then post-
    processes large RANSAC instances by k-means in feature space so
    they split along semantic boundaries (e.g. a single ground plane
    splits into pipe/walkway/floor sub-regions).
    """
    render_dirs: list[str] = []     # absolute paths returned by /api/sam3/renders
    fpn_level: int = 0
    pca_dim: int = 64
    force_recompute: bool = False
    split_min_size: int = 3000      # only instances ≥ this many points are split
    split_target_size: int = 5000   # target pts per sub-cluster
    split_max_k: int = 8

class PresegmentRequest(BaseModel):
    mode: Literal["voxel", "ransac", "model"] = "voxel"
    resolution: float = 0.05      # voxel size in scene units (voxel mode only)
    preserve_labeled: bool = True  # only re-presegment points with class_id < 0
    ransac: Optional[RansacParams] = None  # ransac mode overrides
    labeler_strict: bool = False  # ransac mode: bit-for-bit industrial_point_labeler pipeline
    sam3: Optional[Sam3Params] = None  # ransac mode: feature-aware sub-segmentation

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
    n_points: Optional[int] = None
    preseg_run_id: Optional[str] = None
    preseg_fingerprint: Optional[str] = None
    source_fingerprint: Optional[str] = None
    hidden_inst_ids: list[int] = []
    is_from_prelabel: bool = False
    stale_prelabel: bool = False
    full_class_ids: str = ""
    full_instance_ids: str = ""
    seg_ids: str = ""
    seg_centers: str = ""
    seg_sizes: str = ""
    hull_vertices: str = ""
    hull_faces: str = ""
    hull_face_seg: str = ""

class HideRequest(BaseModel):
    inst_id: int

class SnapToPresegRequest(BaseModel):
    inst_ids: list[int]

class _ObbBox(BaseModel):
    center: list[float]    # [cx, cy, cz] in recentered scene units
    size: list[float]      # [sx, sy, sz]
    rotation: list[float]  # [rx, ry, rz] Euler XYZ (radians), Three.js convention

class _ObbOp(_ObbBox):
    op: str = "keep"       # "keep" intersects, "delete" subtracts

class ExportPlyRequest(BaseModel):
    scene: str
    # Applied root → active. Empty = whole cloud. `ops` is the canonical
    # field; `boxes` is accepted as a legacy alias treated as all-keep.
    ops: list[_ObbOp] | None = None
    boxes: list[_ObbBox] | None = None

__all__ = [n for n in list(globals()) if not n.startswith("__")]
