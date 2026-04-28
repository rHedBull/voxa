"""Loaders for the lidar archive layout (annotated/, raw LAZ).

Sits above point_cloud.py: thin functions that consume a SceneSource and
return either a vanilla PointCloud (raw/decimated) or a PointCloud + label
arrays + scene meta (annotated). Color resolution against lidar/classes.json
also lives here so the frontend gets a ready-to-use class palette.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from point_cloud import PointCloud, load_ply
from scene_registry import SceneSource


# A small palette used when classes.json carries no per-class color (it
# currently doesn't). Stable index → hex.
_FALLBACK_PALETTE = [
    "#5b8def",  # 0 pipe (blue)
    "#10b981",  # 1 tank (green)
    "#f5a524",  # 2 equipment (amber)
    "#a855f7",  # 3 structural (purple)
    "#ef4444",  # 4 double (red)
    "#06b6d4",  # 5 cyan
    "#facc15",  # 6 yellow
    "#84cc16",  # 7 lime
    "#f472b6",  # 8 pink
    "#22d3ee",  # 9 sky
]


@dataclass
class LabelArrays:
    """Per-point label arrays aligned to a PointCloud's points."""
    class_ids: np.ndarray      # int8, shape (N,), -1 = unlabeled
    instance_ids: np.ndarray   # int32, shape (N,), -1 = unlabeled


@dataclass
class ClassPaletteEntry:
    id: int
    label: str
    color: str   # hex


@dataclass
class AnnotatedScene:
    pc: PointCloud
    intensity: Optional[np.ndarray]   # always None for annotated PLYs (kept for symmetry)
    labels: Optional[LabelArrays]
    meta: dict
    palette: list[ClassPaletteEntry]
    n_classes: int
    n_instances: int
    is_from_prelabel: bool = False


def _read_classes_json(lidar_root: Optional[Path]) -> dict[int, str]:
    """Map class_id → label from <lidar_root>/classes.json. Empty if absent."""
    if not lidar_root:
        return {}
    p = lidar_root / "classes.json"
    if not p.exists():
        return {}
    try:
        with p.open() as f:
            data = json.load(f)
        return {int(c["id"]): str(c["name"]) for c in data.get("classes", [])}
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return {}


def _read_segment_metadata(path: Optional[Path]) -> dict:
    if not path or not path.exists():
        return {}
    try:
        with path.open() as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def _build_palette(class_id_to_name: dict[int, str],
                   segment_meta: dict) -> list[ClassPaletteEntry]:
    """Compose a palette from classes.json + per-scene segment metadata.

    Segment metadata sometimes carries its own class_map (label→id); we
    union it with the lidar/classes.json mapping so even an annotated scan
    that lives in a repo without classes.json still gets sensible names.
    """
    names: dict[int, str] = dict(class_id_to_name)
    cmap = segment_meta.get("class_map") or {}
    for label, cid in cmap.items():
        try:
            names.setdefault(int(cid), str(label))
        except (TypeError, ValueError):
            continue
    if not names:
        return []
    palette: list[ClassPaletteEntry] = []
    for cid, label in sorted(names.items()):
        color = _FALLBACK_PALETTE[cid % len(_FALLBACK_PALETTE)] if cid >= 0 else "#7c8088"
        palette.append(ClassPaletteEntry(id=cid, label=label, color=color))
    return palette


def load_annotated(src: SceneSource, lidar_root: Optional[Path]) -> AnnotatedScene:
    """Load an annotated/<scan>/ scene with its label arrays + class palette."""
    pc, _mesh = load_ply(src.source_path)

    labels: Optional[LabelArrays] = None
    n_classes = 0
    n_instances = 0
    if src.extras.get("gt_class_path") and src.extras.get("gt_segment_path"):
        class_path = Path(src.extras["gt_class_path"])
        segment_path = Path(src.extras["gt_segment_path"])
        try:
            class_ids = np.load(class_path)
            instance_ids = np.load(segment_path)
        except (OSError, ValueError):
            class_ids = instance_ids = None

        if (class_ids is not None and instance_ids is not None
                and len(class_ids) == len(pc) == len(instance_ids)):
            # Squash class IDs into int8 — voxa classes count is small.
            ci = class_ids.astype(np.int32)
            ii = instance_ids.astype(np.int32)
            if int(ci.max(initial=-1)) > 126:
                # Out of int8 range — keep int16 in memory but still ship as int8 truncated.
                # Practically classes.json caps below 127, so this branch is defensive.
                ci8 = np.clip(ci, -1, 126).astype(np.int8)
            else:
                ci8 = ci.astype(np.int8)
            labels = LabelArrays(class_ids=ci8, instance_ids=ii)
            valid_classes = ci[ci >= 0]
            valid_inst = ii[ii >= 0]
            n_classes = int(valid_classes.max()) + 1 if valid_classes.size else 0
            n_instances = int(valid_inst.max()) + 1 if valid_inst.size else 0

    from segment_io import load_prelabel  # noqa: PLC0415

    is_from_prelabel = False

    if labels is None:
        scan_dir = Path(src.source_path).parent.parent
        pre = load_prelabel(scan_dir, n_points=len(pc))
        if pre is not None:
            ci8, ii = pre
            labels = LabelArrays(class_ids=ci8, instance_ids=ii)
            valid_classes = ci8[ci8 >= 0]
            valid_inst = ii[ii >= 0]
            n_classes = int(valid_classes.max()) + 1 if valid_classes.size else 0
            n_instances = int(valid_inst.max()) + 1 if valid_inst.size else 0
            is_from_prelabel = True

    if labels is None:
        labels = LabelArrays(
            class_ids=np.full(len(pc), -1, dtype=np.int8),
            instance_ids=np.full(len(pc), -1, dtype=np.int32),
        )

    meta_path = src.extras.get("meta_path")
    meta = _read_segment_metadata(Path(meta_path)) if meta_path else {}

    seg_meta_path = src.extras.get("segment_metadata_path")
    seg_meta = _read_segment_metadata(Path(seg_meta_path)) if seg_meta_path else {}

    class_id_to_name = _read_classes_json(lidar_root)
    palette = _build_palette(class_id_to_name, seg_meta)

    return AnnotatedScene(
        pc=pc, intensity=None, labels=labels, meta=meta,
        palette=palette, n_classes=n_classes, n_instances=n_instances,
        is_from_prelabel=is_from_prelabel,
    )


def _laz_chunk_iter(path: Path, chunk_size: int = 1_000_000):
    """Yield laspy chunks. Imported lazily so the dep is optional at startup."""
    import laspy

    reader = laspy.open(str(path))
    try:
        for chunk in reader.chunk_iterator(chunk_size):
            yield reader.header, chunk
    finally:
        reader.close()


def load_laz(path: Path, max_points: int) -> tuple[PointCloud, np.ndarray]:
    """Stride-sample a LAS/LAZ file down to ~max_points.

    Returns (PointCloud, intensity[N]) where intensity is float32 in 0..1.
    The point count returned will be slightly less than max_points because
    stride is computed from the header total.
    """
    import laspy

    reader = laspy.open(str(path))
    n_total = int(reader.header.point_count)
    reader.close()

    stride = max(1, (n_total + max_points - 1) // max_points)

    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    zs: list[np.ndarray] = []
    rs: list[np.ndarray] = []
    gs: list[np.ndarray] = []
    bs: list[np.ndarray] = []
    its: list[np.ndarray] = []

    has_color = True
    offset = 0   # position within the global virtual stream
    for _hdr, chunk in _laz_chunk_iter(path, chunk_size=1_000_000):
        # Compute per-chunk indices that respect the global stride.
        first_global = offset
        # Local index of the first global stride hit inside this chunk:
        first_hit = (-first_global) % stride
        local_idx = np.arange(first_hit, len(chunk), stride, dtype=np.int64)
        offset += len(chunk)
        if local_idx.size == 0:
            continue
        xs.append(np.asarray(chunk.x[local_idx], dtype=np.float32))
        ys.append(np.asarray(chunk.y[local_idx], dtype=np.float32))
        zs.append(np.asarray(chunk.z[local_idx], dtype=np.float32))

        # LAS 1.2 / point format 3 has 16-bit RGB channels per the spec, but
        # some encoders write 8-bit. Detect by max value and rescale.
        try:
            r = np.asarray(chunk.red[local_idx], dtype=np.uint32)
            g = np.asarray(chunk.green[local_idx], dtype=np.uint32)
            b = np.asarray(chunk.blue[local_idx], dtype=np.uint32)
            rs.append(r); gs.append(g); bs.append(b)
        except (AttributeError, ValueError):
            has_color = False

        its.append(np.asarray(chunk.intensity[local_idx], dtype=np.float32))

    if not xs:
        raise ValueError(f"LAZ file produced no points: {path}")

    points = np.column_stack([np.concatenate(xs), np.concatenate(ys), np.concatenate(zs)])
    intensity = np.concatenate(its) if its else np.zeros(len(points), dtype=np.float32)

    if has_color and rs:
        r = np.concatenate(rs); g = np.concatenate(gs); b = np.concatenate(bs)
        # If the max channel value > 255 we treat colors as 16-bit, else 8-bit.
        cmax = int(max(r.max(initial=0), g.max(initial=0), b.max(initial=0)))
        if cmax > 255:
            r = (r >> 8).astype(np.uint8)
            g = (g >> 8).astype(np.uint8)
            b = (b >> 8).astype(np.uint8)
        else:
            r = r.astype(np.uint8); g = g.astype(np.uint8); b = b.astype(np.uint8)
        colors = np.column_stack([r, g, b])
    else:
        colors = None

    if intensity.size:
        # Per-file normalization. LAS intensity has no fixed range; the LAS
        # spec says it's vendor-defined. Normalizing to 0..1 by file max
        # keeps the visual contrast independent of scanner choice.
        imax = float(intensity.max())
        if imax > 0:
            intensity = intensity / imax
        intensity = intensity.astype(np.float32)

    return PointCloud(points=points, colors=colors), intensity
