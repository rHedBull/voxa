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
    session_id: Optional[str] = None       # explicit session to resume; default = last-worked

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
    frame_check: Optional[dict] = None      # §6 verdict when the scan was verified; None otherwise
    georef_offset: Optional[list[float]] = None
    seg_ids: Optional[str] = None            # b64 Int32 — segment ids (full-res, for voxel box overlay)
    seg_centers: Optional[str] = None        # b64 Float32 (N×3) — bbox centres
    seg_sizes: Optional[str] = None          # b64 Float32 (N×3) — bbox extents
    session_id: Optional[str] = None         # active session resolved on load (annotated tier)
    sessions: list[dict] = []                # SessionInfo dicts for the session picker

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

class SourceRef(BaseModel):
    kind: str                 # "session" | "preseg"
    id: str

class ComparePointsRequest(BaseModel):
    a: SourceRef
    b: SourceRef

LoadResponse.model_rebuild()

class AutoFitRequest(BaseModel):
    bbox_min: list[float]
    bbox_max: list[float]
    cls: str = "unknown"
    color: str = "#5b8def"
    label: str = ""

class ApplyRequest(BaseModel):
    op: str
    indices: Optional[str] = None     # b64 Int32; required for set_class & reassign
    payload: dict

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
    preseg_id: Optional[str] = None
    preseg_fingerprint: Optional[str] = None
    source_fingerprint: Optional[str] = None
    is_from_prelabel: bool = False
    full_class_ids: str = ""
    full_instance_ids: str = ""
    seg_ids: str = ""
    seg_centers: str = ""
    seg_sizes: str = ""
    hull_vertices: str = ""
    hull_faces: str = ""
    hull_face_seg: str = ""
    session_id: Optional[str] = None

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

class CreateSessionRequest(BaseModel):
    name: str
    preseg_id: Optional[str] = None

class RenameSessionRequest(BaseModel):
    name: str

__all__ = [n for n in list(globals()) if not n.startswith("__")]
