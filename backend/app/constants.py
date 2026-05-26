"""Backend configuration constants and paths (no app/route logic)."""
from __future__ import annotations

import os
from pathlib import Path

from scenes.scene_registry import load_lidar_root_from_env


ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = Path(os.environ.get("VOXA_DATA_DIR", ROOT / "data"))

PRESEG_RUNS_DIR = DATA_DIR / "preseg_runs"

def _is_under(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False

SCENES_DIR = DATA_DIR / "scenes"

ANNOT_DIR = DATA_DIR / "annotations"

CONFIG_PATH = Path(os.environ.get("VOXA_CONFIG", ROOT / "config" / "classes.yaml"))

FRONTEND_DIST = ROOT / "dist"

MAX_POINTS_DEFAULT = int(os.environ.get("VOXA_MAX_POINTS", "1000000"))

MAX_LABEL_POINTS = int(os.environ.get("VOXA_MAX_LABEL_POINTS", "5000000"))

MIN_SEGMENT_POINTS = int(os.environ.get("VOXA_MIN_SEGMENT_POINTS", "10"))

LIDAR_ROOT = load_lidar_root_from_env()

__all__ = [n for n in list(globals()) if not n.startswith("__")]
