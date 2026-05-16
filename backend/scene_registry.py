"""Multi-root scene discovery for Voxa.

Voxa scenes can come from four roots:

  legacy     voxa/data/scenes/<name>/source.{ply,glb}
  annotated  $VOXA_LIDAR_ROOT/annotated/<name>/source/scan.ply (+ labels/, meta.json)
  decimated  $VOXA_LIDAR_ROOT/ply_viewer/<name>.ply
  raw        $VOXA_LIDAR_ROOT/laz/<name>.laz

Scene IDs are tier-prefixed ('annotated/munich_water_pump') so that the same
filename living in two tiers (e.g. 'Factory-large' in laz/ and ply_viewer/)
doesn't collide. A bare legacy name still resolves for backward compatibility
with the v1 endpoints and existing tests.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


VALID_TIERS = ("legacy", "annotated", "decimated", "raw")
TIER_ORDER = {t: i for i, t in enumerate(VALID_TIERS)}


@dataclass(frozen=True)
class SceneSource:
    tier: str
    name: str
    source_path: Path
    source_format: str           # 'ply' | 'glb' | 'laz'
    has_labels: bool
    has_intensity: bool
    session_dir: Path            # per-scene dir for session/current.json
    n_points: Optional[int] = None
    extras: dict[str, Any] = field(default_factory=dict)

    @property
    def scene_id(self) -> str:
        return f"{self.tier}/{self.name}"

    @property
    def has_mesh(self) -> bool:
        return bool(self.extras.get("mesh_path"))


def _voxa_legacy_root(data_dir: Path) -> Path:
    return data_dir / "scenes"


def _session_dir_for(tier: str, name: str, scan_root: Optional[Path],
                     data_dir: Optional[Path]) -> Path:
    """Per-scene session dir for `session/current.json`.

    annotated tier lives inside the SCHEMA scan dir; other tiers live under
    `<data_dir>/sessions/<tier>__<name>/` (scene_id with '/' -> '__').
    """
    if tier == "annotated":
        if scan_root is None:
            raise ValueError("scene_registry: scan_root required for annotated tier")
        return scan_root / "session"
    if data_dir is None:
        raise ValueError("scene_registry: data_dir required for non-annotated tier")
    scene_id_safe = f"{tier}/{name}".replace("/", "__")
    return data_dir / "sessions" / scene_id_safe


def _discover_legacy(data_dir: Path) -> list[SceneSource]:
    root = _voxa_legacy_root(data_dir)
    if not root.is_dir():
        return []
    out: list[SceneSource] = []
    for sd in sorted(root.iterdir()):
        if not sd.is_dir():
            continue
        glb = sd / "source.glb"
        ply = sd / "source.ply"
        if glb.exists():
            src, fmt = glb, "glb"
        elif ply.exists():
            src, fmt = ply, "ply"
        else:
            continue
        out.append(SceneSource(
            tier="legacy", name=sd.name,
            source_path=src, source_format=fmt,
            has_labels=False, has_intensity=False,
            session_dir=_session_dir_for("legacy", sd.name, None, data_dir),
        ))
    return out


def _discover_annotated(lidar_root: Path) -> list[SceneSource]:
    root = lidar_root / "annotated"
    if not root.is_dir():
        return []
    out: list[SceneSource] = []
    for sd in sorted(root.iterdir()):
        if not sd.is_dir():
            continue
        source_dir = sd / "source"
        if not source_dir.is_dir():
            continue
        scan = source_dir / "scan.ply"
        if not scan.exists():
            plys = sorted(source_dir.glob("*.ply"))
            if not plys:
                continue
            scan = plys[0]
        labels_dir = sd / "labels"
        gt_class = labels_dir / "gt_class_ids.npy"
        gt_seg = labels_dir / "gt_segment_ids.npy"
        seg_meta = labels_dir / "gt_segment_metadata.json"
        meta_path = sd / "meta.json"
        mesh_path = sd / "source" / "mesh.glb"
        has_labels = gt_class.exists() and gt_seg.exists()
        n_points = None
        # is_z_up: PLYs sampled from a glTF mesh inherit the GLB's Y-up
        # frame, while PLYs sampled from a LAZ inherit the Z-up surveying
        # frame. Default: assume Z-up unless the meta.json explicitly says
        # otherwise (covers the unmaintained-meta case).
        is_z_up = True
        source_laz_path: Optional[str] = None
        if meta_path.exists():
            try:
                with meta_path.open() as f:
                    meta = json.load(f)
                n_points = int(meta.get("n_points") or 0) or None
                if meta.get("source_mesh") and not meta.get("source_laz"):
                    is_z_up = False
                # Resolve source LAZ (the cloud the PLY was subsampled from)
                # so the viewer can pop full-density points back from the
                # original file when a cuboid is selected. Path stored in
                # meta is canonical archive-relative (`lidar/laz/<file>.laz`);
                # fall back to basename lookup under `<lidar_root>/laz/` so
                # we're robust to small path format drift.
                src_laz_str = meta.get("source_laz")
                if src_laz_str:
                    cand1 = lidar_root / "laz" / Path(src_laz_str).name
                    cand2 = lidar_root.parent / src_laz_str
                    for cand in (cand1, cand2):
                        if cand.exists():
                            source_laz_path = str(cand)
                            break
            except (OSError, ValueError, json.JSONDecodeError):
                n_points = None
        out.append(SceneSource(
            tier="annotated", name=sd.name,
            source_path=scan, source_format="ply",
            has_labels=has_labels, has_intensity=False,
            session_dir=_session_dir_for("annotated", sd.name, sd, None),
            n_points=n_points,
            extras={
                "labels_dir": str(labels_dir),
                "gt_class_path": str(gt_class) if gt_class.exists() else None,
                "gt_segment_path": str(gt_seg) if gt_seg.exists() else None,
                "segment_metadata_path": str(seg_meta) if seg_meta.exists() else None,
                "meta_path": str(meta_path) if meta_path.exists() else None,
                "scan_dir": str(sd),
                "mesh_path": str(mesh_path) if mesh_path.exists() else None,
                "is_z_up": is_z_up,
                "source_laz_path": source_laz_path,
            },
        ))
    return out


def _discover_decimated(lidar_root: Path, data_dir: Path) -> list[SceneSource]:
    root = lidar_root / "ply_viewer"
    if not root.is_dir():
        return []
    out: list[SceneSource] = []
    for p in sorted(root.glob("*.ply")):
        out.append(SceneSource(
            tier="decimated", name=p.stem,
            source_path=p, source_format="ply",
            has_labels=False, has_intensity=False,
            session_dir=_session_dir_for("decimated", p.stem, None, data_dir),
        ))
    return out


def _discover_raw(lidar_root: Path, data_dir: Path) -> list[SceneSource]:
    root = lidar_root / "laz"
    if not root.is_dir():
        return []
    out: list[SceneSource] = []
    for p in sorted(root.glob("*.laz")):
        out.append(SceneSource(
            tier="raw", name=p.stem,
            source_path=p, source_format="laz",
            has_labels=False, has_intensity=True,
            session_dir=_session_dir_for("raw", p.stem, None, data_dir),
        ))
    # Some lidar archives also drop .las next to .laz.
    for p in sorted(root.glob("*.las")):
        out.append(SceneSource(
            tier="raw", name=p.stem,
            source_path=p, source_format="laz",
            has_labels=False, has_intensity=True,
            session_dir=_session_dir_for("raw", p.stem, None, data_dir),
        ))
    return out


def discover(data_dir: Path, lidar_root: Optional[Path]) -> list[SceneSource]:
    """Discover all scenes across all configured roots."""
    scenes: list[SceneSource] = []
    scenes.extend(_discover_legacy(data_dir))
    if lidar_root and lidar_root.is_dir():
        scenes.extend(_discover_annotated(lidar_root))
        scenes.extend(_discover_decimated(lidar_root, data_dir))
        scenes.extend(_discover_raw(lidar_root, data_dir))
    scenes.sort(key=lambda s: (TIER_ORDER.get(s.tier, 99), s.name.lower()))
    return scenes


def resolve(scene_id: str, data_dir: Path, lidar_root: Optional[Path]) -> SceneSource:
    """Look up a scene by tier-prefixed id, or by bare legacy name."""
    all_scenes = discover(data_dir, lidar_root)

    if "/" in scene_id:
        tier, _, name = scene_id.partition("/")
        for s in all_scenes:
            if s.tier == tier and s.name == name:
                return s
        raise KeyError(scene_id)

    # Bare name → legacy first, then anything else with that name.
    for s in all_scenes:
        if s.tier == "legacy" and s.name == scene_id:
            return s
    for s in all_scenes:
        if s.name == scene_id:
            return s
    raise KeyError(scene_id)


def load_lidar_root_from_env() -> Optional[Path]:
    raw = os.environ.get("VOXA_LIDAR_ROOT")
    if raw:
        return Path(raw)
    default = Path("/home/hendrik/coding/engine/data/lidar")
    return default if default.is_dir() else None
