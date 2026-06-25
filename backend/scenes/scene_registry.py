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
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from scenes.scan_meta import is_z_up_from_meta
from scan_schema.layout import ScanLayout
from scan_schema.metadata import check_meta


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
        lay = ScanLayout(sd)
        source_dir = lay.source_dir
        if not source_dir.is_dir():
            continue
        scan = lay.scan_ply
        if not scan.exists():
            plys = sorted(source_dir.glob("*.ply"))
            if not plys:
                continue
            scan = plys[0]
        meta_path = lay.meta_json
        # Discovery gate delegates to the shared schema: keep a scan iff
        # scan_schema.metadata.check_meta reports no errors. On a 2.x scan,
        # missing frame/derivation are warnings (grandfathered → kept); a
        # non-2.x/3.x version or missing required field is an error → skipped.
        try:
            with meta_path.open() as f:
                meta = json.load(f)
        except (OSError, ValueError, json.JSONDecodeError):
            meta = None
        if meta is None:
            logging.info("skipping %s: missing/unreadable meta.json", sd.name)
            continue
        errors, _ = check_meta(meta)
        if errors:
            logging.info(
                "skipping %s: scan-schema errors %r — run scripts/migrate_scan_v2.py",
                sd.name, errors)
            continue

        mesh_path = lay.mesh_glb
        has_labels = lay.sessions_root.is_dir()
        n_points = int(meta.get("n_points") or 0) or None
        # is_z_up: PLYs sampled from a glTF mesh inherit the GLB's Y-up frame,
        # while PLYs sampled from a LAZ inherit the Z-up surveying frame.
        is_z_up = is_z_up_from_meta(meta)
        # Resolve source LAZ (the cloud the PLY was subsampled from) so the
        # viewer can pop full-density points back from the original file when a
        # cuboid is selected. Path stored in meta is canonical archive-relative
        # (`lidar/raw/<file>.laz`); fall back to basename lookup under
        # `<lidar_root>/raw/` so we're robust to small path format drift.
        source_laz_path: Optional[str] = None
        src_laz_str = meta.get("source_laz")
        if src_laz_str:
            cand1 = lidar_root / "raw" / Path(src_laz_str).name
            cand2 = lidar_root.parent / src_laz_str
            for cand in (cand1, cand2):
                if cand.exists():
                    source_laz_path = str(cand)
                    break
        out.append(SceneSource(
            tier="annotated", name=sd.name,
            source_path=scan, source_format="ply",
            has_labels=has_labels, has_intensity=False,
            n_points=n_points,
            extras={
                "meta_path": str(meta_path),
                "scan_dir": str(sd),
                "mesh_path": str(mesh_path) if mesh_path.exists() else None,
                "is_z_up": is_z_up,
                "source_laz_path": source_laz_path,
            },
        ))
    return out


def _discover_decimated(lidar_root: Path) -> list[SceneSource]:
    root = lidar_root / "ply_viewer"
    if not root.is_dir():
        return []
    out: list[SceneSource] = []
    for p in sorted(root.glob("*.ply")):
        out.append(SceneSource(
            tier="decimated", name=p.stem,
            source_path=p, source_format="ply",
            has_labels=False, has_intensity=False,
        ))
    return out


def _discover_raw(lidar_root: Path) -> list[SceneSource]:
    root = lidar_root / "raw"
    if not root.is_dir():
        return []
    out: list[SceneSource] = []
    # Some lidar archives drop .las next to .laz; discover() re-sorts by name.
    for p in sorted([*root.glob("*.laz"), *root.glob("*.las")]):
        out.append(SceneSource(
            tier="raw", name=p.stem,
            source_path=p, source_format="laz",
            has_labels=False, has_intensity=True,
        ))
    return out


def discover(data_dir: Path, lidar_root: Optional[Path]) -> list[SceneSource]:
    """Discover all scenes across all configured roots."""
    scenes: list[SceneSource] = []
    scenes.extend(_discover_legacy(data_dir))
    if lidar_root and lidar_root.is_dir():
        scenes.extend(_discover_annotated(lidar_root))
        scenes.extend(_discover_decimated(lidar_root))
        scenes.extend(_discover_raw(lidar_root))
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
