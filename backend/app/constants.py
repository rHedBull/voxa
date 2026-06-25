"""Backend configuration constants and paths (no app/route logic)."""
from __future__ import annotations

import os
from pathlib import Path

from scenes.scene_registry import load_lidar_root_from_env


ROOT = Path(__file__).resolve().parent.parent.parent

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

# Max points the backend will hold for a labeling session (per-point label
# arrays + cKDTree). Above this a scene loads view-only (no seg session).
MAX_LABEL_POINTS = int(os.environ.get("VOXA_MAX_LABEL_POINTS", "5000000"))

# Viewer subsample cap — how many points are sent to the browser to render.
# Defaults to the label cap so what's rendered == what's labelable (no
# viewer/label gap); only clouds above the label cap get subsampled for display.
# Override with VOXA_MAX_POINTS if rendering that many points is too heavy.
MAX_POINTS_DEFAULT = int(os.environ.get("VOXA_MAX_POINTS", str(MAX_LABEL_POINTS)))

MIN_SEGMENT_POINTS = int(os.environ.get("VOXA_MIN_SEGMENT_POINTS", "10"))

LIDAR_ROOT = load_lidar_root_from_env()

__all__ = [n for n in list(globals()) if not n.startswith("__")]
