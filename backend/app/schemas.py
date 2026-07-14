"""Pydantic request/response models for the Voxa API."""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.constants import MAX_POINTS_DEFAULT


class SceneInfo(BaseModel):
    id: str                      # tier-prefixed
    tier: str                    # 'legacy' | 'annotated'
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
    raw_source_available: bool = False       # true if a full-density raw cloud resolved (source_laz or lineage)

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
    source: str = "manual"   # 'manual' | 'auto' | 'fit' | 'preseg' | 'box' | 'beam' | 'draw' | 'recommendation'
    confirmed: bool = False  # set true via Ctrl+Enter; hides interior points in main view
    kind: str = "cuboid"     # 'cuboid' | 'pointset'
    segId: Optional[int] = None  # set for pointset (and preseg-promoted) instances; per-point membership key in segState.instanceFull
    seq: Optional[int] = None  # monotonic apply-order rank; stamped on save (resolution-independent-labels spec §2)

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
    # Canonical numeric class id (classes.yaml `id:`, matching
    # engine/data/lidar/classes.json). The frontend uses this — never the
    # array position — to map numeric class ids to palette entries.
    class_id: int = -1

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

class CenterlinePath(BaseModel):
    points: list[list[float]] = Field(min_length=2)
    radius: float = Field(gt=0)
    smooth: bool = False

    @field_validator("points")
    @classmethod
    def _points_are_3d(cls, v):
        if any(len(p) != 3 for p in v):
            raise ValueError("each point must be [x, y, z]")
        return v

class CenterlineApplyRequest(BaseModel):
    paths: list[CenterlinePath] = Field(min_length=1)
    target_class: int | str
    target_inst: int = -1
    merged_from: list[int] = []
    protect_instances: list[int] = []  # see ApplyShapeRequest

class ApplyShapeRequest(BaseModel):
    shape: dict            # {type:'tube'|'obb', ...} — validated in shape_indices
    target_class: int | str
    target_inst: int = -1
    merged_from: list[int] = []
    # Instance ids that must not be overwritten ("confirmed = locked"): points
    # inside the shape that belong to these instances are skipped, not stolen.
    protect_instances: list[int] = []

class SamCaptureRequest(BaseModel):
    camera: dict                      # {pos,target,fov,W,H} in the recentered frame
    mode: str                         # "box" | "concept"
    box: list[float] | None = None
    text: str | None = None

class SamProjectRequest(BaseModel):
    capture_id: str
    mask_ids: list[int]
    protect_instances: list[int] = []

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
    full_sam_ids: str = ""             # b64 Int32, full-res — SAM candidate layer
    sam_segments: list[dict] = []      # [{id, n_points, mask_score, created_at}]
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

class StructureNode(BaseModel):
    id: int
    pos: list[float]

    @field_validator("pos")
    @classmethod
    def _pos_is_3d(cls, v):
        if len(v) != 3:
            raise ValueError("pos must be [x, y, z]")
        return v

class StructureEdge(BaseModel):
    id: int
    a: int
    b: int
    width: float = Field(gt=0)
    class_id: int
    instance_id: Optional[int] = None
    dirty: bool = False   # edited since last apply (frontend re-apply bookkeeping)

class CommittedBeam(BaseModel):
    a: list[float]
    b: list[float]
    width: float = Field(gt=0)
    class_id: int
    instance_id: int

    @field_validator("a", "b")
    @classmethod
    def _endpoint_is_3d(cls, v):
        if len(v) != 3:
            raise ValueError("endpoint must be [x, y, z]")
        return v

class StructureDoc(BaseModel):
    # Written by the frontend after apply/commit/edits, debounced — session_id
    # pins the write to the session the graph was built in so a session switch
    # mid-debounce can't land the old graph in the new session's file.
    session_id: Optional[str] = None
    nodes: list[StructureNode] = []
    edges: list[StructureEdge] = []
    committed_beams: list[CommittedBeam] = []

class RemapTarget(BaseModel):
    id: int
    label: str
    color: str

class RemapRule(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    from_: list[int] = Field(alias="from")
    to: RemapTarget

class ExportResolution(BaseModel):
    kind: str            # "scan" | "subsample" | "raw"
    n: Optional[int] = None

class ExportLabelsRequest(BaseModel):
    scene: str
    session_id: str
    resolution: ExportResolution
    confirmed_only: bool = False
    include_classes: Optional[list[int]] = None
    remap: list[RemapRule] = []
    drop_unlabeled: bool = False

__all__ = [n for n in list(globals()) if not n.startswith("__")]
