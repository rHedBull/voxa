"""Voxa — unified 3D scan studio backend (app assembler).

Endpoints serve the React/Three.js frontend with point clouds, cuboid
annotations, and GT-vs-prediction diffs. Scenes come from multiple roots:

  annotated  $VOXA_LIDAR_ROOT/annotated/<name>/source/scan.ply (+ labels/, meta.json)
  decimated  $VOXA_LIDAR_ROOT/ply_viewer/<name>.ply
  raw        $VOXA_LIDAR_ROOT/laz/<name>.laz

Scene IDs in /api/load and /api/annotations are tier-prefixed
('annotated/munich_water_pump'); a bare legacy name still resolves for
backward compatibility.

This module only wires the app together: the request models live in
``app.schemas``, shared state + helpers in ``app.core``, configuration in
``app.constants``, and the endpoints in ``routes.*``.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.constants import FRONTEND_DIST, LIDAR_ROOT  # noqa: F401  (LIDAR_ROOT re-exported for tests)
# Re-exported so existing test imports (`main._state`, `from main import _resolve`,
# frame helpers) keep working after the split.
from app.core import (  # noqa: F401
    _state,
    _resolve,
    _z_up_to_y_up,
    _y_up_to_z_up_xyz,
    _to_display_frame,
)
from routes import compare, export, load, meta, sam, segment, sessions

app = FastAPI(title="Voxa 3D scan studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

for _module in (meta, load, compare, segment, export, sessions, sam):
    app.include_router(_module.router)


# ── Static frontend ─────────────────────────────────────────────────────────
# Mounted last so /api/* takes precedence. In dev, Vite serves the frontend
# directly and proxies /api/* to this backend; in production, run `npm run
# build` and the built bundle ends up in `dist/`.
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
