"""Shared backend core: in-memory state + all request helpers.

Route modules `from app.core import *` to get state, helpers, and the
common toolkit (numpy, fastapi errors, scene loaders, ...)."""
from __future__ import annotations

import base64
import json
import os
import time as _time
import uuid  # noqa: F401  (re-exported via `from app.core import *` for routes/compare.py auto-fit)
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import yaml
from fastapi import HTTPException, Response
from fastapi.responses import FileResponse

from scenes.lidar_io import (
    LabelArrays, load_annotated, load_laz_region, z_up_to_y_up_xyz,
    _laz_rgb_to_uint8,
)
from scenes.point_cloud import PointCloud, load_glb, load_ply
from scenes.reproject import euler_xyz_matrix
from scan_schema.layout import ScanLayout
from scenes.scene_registry import (
   SceneSource, discover, load_lidar_root_from_env,
   resolve as resolve_scene,
)

from app import constants  # for live LIDAR_ROOT lookups (tests monkeypatch it)
from app.constants import *  # noqa: F401,F403
from app.schemas import *  # noqa: F401,F403


_state: dict[str, Any] = {
    "scene": None,
    "source": None,         # SceneSource (kept so /api/edit/export-ply can
                            # check is_z_up and reverse the load-time rotation)
    "pc": None,             # PointCloud (recentered)
    "mesh": None,           # trimesh.Trimesh or None
    "subsample_idx": None,  # indices into the original cloud (or None)
    "intensity": None,      # full-resolution intensity array (or None)
    "labels": None,         # LabelArrays (or None)
    "recenter_offset": [0.0, 0.0, 0.0],
    "raw_georef_offset_m": [0.0, 0.0, 0.0],
    "seg": None,            # SegmentSession | None
    "session_id": None,     # active session id (annotated tier) | None
    "source_fp": None,      # fingerprint of the loaded cloud (session creation reads it)
}

def _annotation_path(scene: str, kind: str, session_id: str | None = None) -> Path:
    if kind not in ("gt", "pred"):
        raise HTTPException(400, f"Invalid kind: {kind}")
    if session_id:
        # Session-scoped (annotated tier): the instance doc belongs to ONE
        # labeling session — a scene-global file would leak instances across
        # sessions (every session showing the last-saved session's panel).
        from labeling.session_store import _validate_session_id
        from scan_schema.layout import ScanLayout
        try:
            _validate_session_id(session_id)
        except ValueError as e:
            raise HTTPException(400, str(e))
        src = _resolve(scene)
        if src.tier != "annotated":
            raise HTTPException(409, "session_id is only valid for annotated scenes")
        sdir = ScanLayout(Path(src.extras["scan_dir"])).session(session_id).dir
        if not sdir.is_dir():
            raise HTTPException(404, f"session '{session_id}' not found")
        return sdir / f"instances_{kind}.json"
    fname = "ground_truth.json" if kind == "gt" else "predictions.json"
    # Scene-global file: legacy data-dir scenes only. Annotated scans must
    # pass session_id — their callers (App.jsx) thread the active session.
    safe = scene.replace("/", "__")
    return ANNOT_DIR / safe / fname

def _resolve(scene_id: str) -> SceneSource:
    try:
        return resolve_scene(scene_id, DATA_DIR, constants.LIDAR_ROOT)
    except KeyError:
        raise HTTPException(404, f"Scene not found: {scene_id}")

def _subsample_indices(n: int, max_points: int) -> np.ndarray:
    """Reservoir down-sample: seeded, ascending indices into ``range(n)``.

    Fixed seed so the full-load and region-pop paths pick the same points
    reproducibly; sorted so the result preserves on-disk point order."""
    rng = np.random.default_rng(7)
    idx = rng.choice(n, size=max_points, replace=False)
    idx.sort()
    return idx

def _safe_subsample(
    pc: PointCloud,
    max_points: int,
    intensity: Optional[np.ndarray] = None,
    labels: Optional[LabelArrays] = None,
) -> tuple[PointCloud, np.ndarray | None, Optional[np.ndarray], Optional[LabelArrays]]:
    n = len(pc)
    if n <= max_points:
        return pc, None, intensity, labels
    idx = _subsample_indices(n, max_points)
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

def _z_up_to_y_up(pc: PointCloud) -> PointCloud:
    """Rotate a Z-up point cloud (LAS / surveying convention) into the
    Three.js Y-up frame so the floor sits below the cloud as expected.

    Right-handed Z-up (X right, Y depth, Z up) → right-handed Y-up
    (X right, Y up, Z back). Mapping: (x, y, z) → (x, z, -y).
    Per-point label / instance arrays carry over unchanged.
    """
    return PointCloud(
        points=z_up_to_y_up_xyz(pc.points),
        colors=pc.colors,
        labels=pc.labels,
        instance_ids=pc.instance_ids,
        face_indices=pc.face_indices,
    )

def _y_up_to_z_up_xyz(pts: np.ndarray) -> np.ndarray:
    """Inverse of `_z_up_to_y_up`'s mapping applied to a raw (N, 3) array.

    (x, y, z)_y_up → (x, -z, y)_z_up. Used on export so saved PLYs come out
    in the source-file (LAS/scanner Z-up) frame instead of the in-memory
    Three.js display frame.
    """
    out = np.empty_like(pts)
    out[:, 0] = pts[:, 0]
    out[:, 1] = -pts[:, 2]
    out[:, 2] = pts[:, 1]
    return out

def _scene_is_z_up(src: SceneSource) -> bool:
    """Decide whether the scene's source frame is Z-up (surveying / LAS).

    - legacy: author-defined, treated as Y-up.
    - annotated: depends on what the PLY was sampled from.
      `meta.json::source_mesh` (a glTF) → Y-up.
      `meta.json::source_laz` → Z-up.
      Default Z-up if the meta is missing or ambiguous.
    """
    if src.tier == "legacy":
        return False
    return bool(src.extras.get("is_z_up", True))

def _load_scene_source(src: SceneSource, max_points: int):
    """Dispatch to the right loader.

    Returns (pc, mesh, intensity, labels, palette, n_classes, n_instances,
             n_labeled_points, is_from_prelabel, n_source_total).

    `n_source_total` is the source file's true point count when it differs
    from the loaded count (LAZ stride-samples at read time); None when the
    loaded count already matches the on-disk count.
    """
    if src.tier == "annotated":
        a = load_annotated(src, constants.LIDAR_ROOT)
        n_labeled = int((a.labels.class_ids >= 0).sum()) if a.labels is not None else 0
        palette = [ClassDef(id=p.id, label=p.label, color=p.color) for p in a.palette]
        return (a.pc, None, a.intensity, a.labels, palette, a.n_classes, a.n_instances,
                n_labeled, bool(a.is_from_prelabel), None)

    # legacy → reuse the existing PLY/GLB loaders
    if src.source_format == "glb":
        pc, mesh = load_glb(src.source_path, num_samples=max(max_points, 50_000))
        return (pc, mesh, None, None, None, None, None, None, False, None)
    pc, _ = load_ply(src.source_path)
    return (pc, None, None, None, None, None, None, None, False, None)

def _resume_session(lay: ScanLayout, session_id: str, pc, source_fp: str):
    """Resume one on-disk session: verify pins (PinMismatch propagates to the
    route → 409), load working arrays, rebuild the in-memory SegmentSession
    via SegmentSession.from_aux (the inverse of _aux_payload). Fails loudly
    on unreadable/misshapen arrays — never silently blank. The preseg array
    is read exactly once (for the snap-to layer); the pin check itself is a
    string compare against the preseg's meta.json."""
    from labeling.segment_state import SegmentSession
    from labeling.segment_io import load_working_arrays
    from labeling.session_store import verify_pins
    from preseg.preseg_store import load_preseg

    aux = verify_pins(lay, session_id, source_fp=source_fp)
    sp = lay.session(session_id)
    wa = load_working_arrays(sp.dir, n_points=len(pc))
    if wa is None:
        raise HTTPException(409, f"session {session_id}: working arrays "
                                 f"missing or wrong shape for this cloud")
    seg = SegmentSession.from_aux(aux, class_ids=wa[0], instance_ids=wa[1],
                                  positions=pc.points, session_dir=sp.dir)
    seg.source_fingerprint = source_fp
    if seg.preseg_id is not None:
        _, pre_ii = load_preseg(lay, seg.preseg_id, n_points=len(pc))
        seg.preseg_ids = pre_ii          # immutable preseg layer for snap-to
    return seg


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

def _require_seg():
    seg = _state.get("seg")
    if seg is None:
        raise HTTPException(409, "No segment session loaded — load an annotated scene first")
    return seg

def _serialize_apply(out: dict) -> dict:
    body = {"op": out["op"], "n_affected": out["n_affected"], "dirty": True}
    if "n_protected" in out:
        body["n_protected"] = int(out["n_protected"])
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

def _voxa_class_name_to_id(config_path: Optional[Path] = None) -> dict[str, int]:
    """Build {class-name-lower: int-id} from a classes.yaml (default: the
    configured CONFIG_PATH).

    Uses each class's explicit ``id:`` (the canonical id from
    engine/data/lidar/classes.json) and falls back to yaml position only
    when absent — positional ids silently corrupt labels if the yaml is
    ever reordered. Mirrors ``get_config()`` so id↔name stays consistent
    with the palette the frontend renders. Single home for this mapping:
    the preseg script CLIs delegate here too (scripts/preseg/_common.py).
    """
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    if not path.exists():
        return {}
    raw = yaml.safe_load(path.read_text()) or {}
    out: dict[str, int] = {}
    for i, (name, body) in enumerate((raw.get("classes") or {}).items()):
        explicit = body.get("id") if isinstance(body, dict) else None
        out[str(name).lower()] = int(explicit) if explicit is not None else i
    if len(set(out.values())) != len(out):
        raise ValueError(f"duplicate class ids in {path}: {out}")
    return out

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

def _obb_mask(points: np.ndarray, box: _ObbBox) -> np.ndarray:
    c = np.asarray(box.center, dtype=np.float64)
    s = np.asarray(box.size, dtype=np.float64)
    R = euler_xyz_matrix(*box.rotation)
    # local = R^T · (p - c)  → vectorized as (p - c) @ R
    local = (points.astype(np.float64) - c) @ R
    half = s / 2.0
    return np.all(np.abs(local) <= half + 1e-6, axis=1)

def _ops_chain_mask(display_xyz: np.ndarray, ops: list) -> np.ndarray:
    """Replay the box-select op chain on points in display (y-up + recentered)
    frame. Returns a boolean keep-mask."""
    mask = np.ones(len(display_xyz), dtype=bool)
    for op in ops:
        if not mask.any():
            break
        idx = np.flatnonzero(mask)
        inside = idx[_obb_mask(display_xyz[idx], op)]
        if op.op == "delete":
            mask[inside] = False
        else:
            mask = np.zeros_like(mask)
            mask[inside] = True
    return mask

def _to_display_frame(xyz: np.ndarray, scene_is_z_up: bool,
                      offset: np.ndarray) -> np.ndarray:
    """Recreate the load-time z-up→y-up rotation + recenter so source-frame
    points align with OBB ops authored in the display frame."""
    out = xyz.astype(np.float64, copy=True)
    if scene_is_z_up:
        # (x, y, z) → (x, z, -y)
        out = np.column_stack([out[:, 0], out[:, 2], -out[:, 1]])
    if np.any(offset):
        out = out - offset
    return out

def _ply_response_bytes(xyz: np.ndarray, rgb: Optional[np.ndarray]) -> bytes:
    """Encode (N, 3) xyz + optional (N, 3) uint8 rgb as binary PLY."""
    n = int(len(xyz))
    has_color = rgb is not None
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        + ("property uchar red\nproperty uchar green\nproperty uchar blue\n" if has_color else "")
        + "end_header\n"
    ).encode("ascii")
    xyz_f32 = xyz.astype("<f4", copy=False)
    if has_color:
        dt = np.dtype([("xyz", "<f4", 3), ("rgb", "u1", 3)])
        rec = np.empty(n, dtype=dt)
        rec["xyz"] = xyz_f32
        rec["rgb"] = rgb.astype(np.uint8, copy=False)
        body = rec.tobytes()
    else:
        body = xyz_f32.tobytes()
    return header + body

def _ply_labeled_header(n: int, has_color: bool) -> bytes:
    return (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        + ("property uchar red\nproperty uchar green\nproperty uchar blue\n" if has_color else "")
        + "property int class_id\n"
        "property int instance_id\n"
        "end_header\n"
    ).encode("ascii")

def _ply_labeled_chunk_bytes(xyz: np.ndarray, rgb: Optional[np.ndarray],
                              class_ids: np.ndarray, instance_ids: np.ndarray) -> bytes:
    """Encode one chunk's worth of xyz + optional rgb + class_id/instance_id as
    binary PLY body (no header) — reused by both the one-shot writer and the
    streaming exporter (which appends chunks then prepends the header once the
    total count is known)."""
    n = int(len(xyz))
    has_color = rgb is not None
    fields = [("xyz", "<f4", 3)]
    if has_color:
        fields.append(("rgb", "u1", 3))
    fields += [("class_id", "<i4"), ("instance_id", "<i4")]
    dt = np.dtype(fields)
    rec = np.empty(n, dtype=dt)
    rec["xyz"] = xyz.astype("<f4", copy=False)
    if has_color:
        rec["rgb"] = rgb.astype(np.uint8, copy=False)
    rec["class_id"] = class_ids.astype(np.int32, copy=False)
    rec["instance_id"] = instance_ids.astype(np.int32, copy=False)
    return rec.tobytes()

def _ply_labeled_bytes(xyz: np.ndarray, rgb: Optional[np.ndarray],
                        class_ids: np.ndarray, instance_ids: np.ndarray) -> bytes:
    """One-shot labeled binary PLY writer = header + single chunk body."""
    return (
        _ply_labeled_header(len(xyz), rgb is not None)
        + _ply_labeled_chunk_bytes(xyz, rgb, class_ids, instance_ids)
    )

def _stream_laz_keep(src_path: Path, ops: list, scene_is_z_up: bool,
                     offset: np.ndarray) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Walk the LAZ at native density, applying the box-op chain in display
    frame; accumulate kept points in source frame."""
    from scenes.lidar_io import _laz_chunk_iter  # type: ignore

    kept_xyz: list[np.ndarray] = []
    kept_rgb: list[np.ndarray] = []
    rgb_present = True
    for _hdr, chunk in _laz_chunk_iter(src_path, chunk_size=1_000_000):
        xyz_src = np.column_stack([
            np.asarray(chunk.x, dtype=np.float64),
            np.asarray(chunk.y, dtype=np.float64),
            np.asarray(chunk.z, dtype=np.float64),
        ])
        try:
            r = np.asarray(chunk.red, dtype=np.uint32)
            g = np.asarray(chunk.green, dtype=np.uint32)
            b = np.asarray(chunk.blue, dtype=np.uint32)
        except (AttributeError, ValueError):
            rgb_present = False
            r = g = b = None

        display = _to_display_frame(xyz_src, scene_is_z_up, offset)
        mask = _ops_chain_mask(display, ops)
        if not mask.any():
            continue
        kept_xyz.append(xyz_src[mask])
        if rgb_present and r is not None:
            rgb8 = _laz_rgb_to_uint8(np.column_stack([r, g, b]))
            kept_rgb.append(rgb8[mask])

    if not kept_xyz:
        return np.zeros((0, 3), dtype=np.float64), None
    xyz_out = np.concatenate(kept_xyz, axis=0)
    rgb_out = np.concatenate(kept_rgb, axis=0) if (rgb_present and kept_rgb) else None
    return xyz_out, rgb_out

__all__ = [n for n in list(globals()) if not n.startswith("__")]
