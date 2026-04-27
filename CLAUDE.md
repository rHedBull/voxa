# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Voxa is a unified 3D scan studio (viewer + cuboid labeler) for industrial LiDAR / point clouds. It replaces the older `3d-labeler` and `industrial-point-labeler` tools. The UI is organized around three modes: **Inspect** (fast review), **Label** (cuboid annotation, optional auto-fit), and **Compare** (GT vs prediction with server-computed precision/recall/F1/IoU + per-instance TP/FP/FN).

## Commands

```bash
npm install                         # first run only
npm run dev                         # FastAPI (backend) + Vite (frontend) via concurrently
npm run build                       # build frontend → ./dist
./scripts/run.sh                    # backend only (after build, single-port mode)
./scripts/import_scene.sh <name> <path-to-ply-or-glb>

npm test                            # frontend (vitest) + backend (pytest)
npm run test:frontend               # vitest only
npm run test:backend                # pytest only (auto-installs dev deps into .venv)
.venv/bin/pytest backend/tests/test_smoke.py::test_health   # single backend test
npx vitest run src/api.test.js      # single frontend test (run from ./frontend or with --root)
```

`npm run dev` opens the dev UI at `http://127.0.0.1:5173` and Vite proxies `/api/*` to the backend at `http://127.0.0.1:8765`. After `npm run build`, `./scripts/run.sh` alone serves both `/api/*` and the static frontend on port 8765.

`scripts/run.sh` auto-creates `voxa/.venv` and installs `backend/requirements.txt` on first run. `scripts/test.sh` does the same for `requirements-dev.txt`.

### Useful env vars

- `VOXA_PORT`, `VOXA_HOST` — backend bind (defaults `127.0.0.1:8765`)
- `VOXA_DATA_DIR` — overrides `./data` for scenes + annotations
- `VOXA_CONFIG` — path to a `classes.yaml` (default `config/classes.yaml`)
- `VOXA_MAX_POINTS` — server-side subsample cap (default `300000`)
- `VOXA_RELOAD=1` — enable uvicorn `--reload` (off by default to avoid hitting the system inotify limit)
- `VOXA_BACKEND`, `VOXA_FRONTEND_PORT` — Vite dev proxy target / port

## Architecture

**Backend** (`backend/`, FastAPI, served by `uvicorn main:app`):
- `main.py` — all HTTP endpoints, Pydantic schemas, in-memory `_state` holding the single active `PointCloud`. The IoU diff is axis-aligned (rotation ignored) and matched greedily within class.
- `point_cloud.py`, `supervoxels.py`, `clustering.py`, `fitting.py` — carried over from the old `3d-labeler`. Only the PLY/GLB loader and a small `auto-fit` (snap a cuboid to points inside an AABB) are wired into the current frontend; the supervoxel / cluster / RANSAC modules are present for future use but not exposed via routes.
- Static frontend is mounted at `/` **only if `dist/` exists**, so `/api/*` always wins. In dev mode (`dist/` absent) the FastAPI process is API-only and Vite owns the UI.

**Frontend** (`frontend/src/`, Vite + React 18 + Three.js, no TypeScript):
- `main.jsx` mounts `<App>` plus `<Agentation>` (in-app feedback toolbar from the `agentation` package).
- `App.jsx` is the shell: owns `scenes / activeScene / cloud / classes / gtInstances / predInstances` state, fetches config + scenes once, then re-loads cloud + GT + predictions whenever `activeScene` changes. It dispatches to one of three mode components and handles `⌘S/Ctrl+S` save.
- `mode-inspect.jsx`, `mode-label.jsx`, `mode-compare.jsx` — feature surfaces. They share `viewer.jsx` (the Three.js viewport with `attachOrbit` and `attachWalk` camera schemes; `navMode` is lifted to `App` so it persists across mode switches) and the small atoms in `viewport-atoms.jsx`.
- `api.js` is the only place that talks to `/api/*`. Point cloud `positions`/`colors` come over the wire as base64-encoded `Float32Array` payloads — decode via `b64ToFloat32`.
- `tweaks-panel.jsx` is a generic tweak-host (theme/mode controls). It implements an external "edit mode" host protocol via `postMessage` (`__activate_edit_mode` / `__edit_mode_set_keys` / etc.) — do not delete those listeners without checking the Agentation integration.

**Data layout** (gitignored):
```
data/
├── scenes/<name>/source.{ply,glb}              # input
└── annotations/<name>/{ground_truth,predictions}.json
```
Both annotation files share one schema (`AnnotationDoc` in `main.py`); a model's output dropped at `predictions.json` is enough to power Compare mode.

**Class config**: `config/classes.yaml`. Each entry has `label`, `color` (hex string or `[r,g,b]` floats 0-1), and `key` (hotkey). Restart the backend or re-fetch `/api/config` to pick up edits.

## Conventions and gotchas

- **EDITMODE markers**: `App.jsx` defines tweak defaults inside `/*EDITMODE-BEGIN*/ ... /*EDITMODE-END*/`. The Agentation/tweaks tooling rewrites this region, so preserve the markers verbatim when editing defaults.
- **Vite uses polling watchers** (`vite.config.js`) on purpose — to avoid blowing the system-wide inotify limit when many other projects are open. Keep it.
- **Backend has no autoreload by default** — after editing Python files, kill and restart `npm run dev` (or set `VOXA_RELOAD=1`). This matches the broader "restart server after code changes" rule from the global config.
- **Single in-memory point cloud**: `_state` in `main.py` holds one cloud at a time. Endpoints that need the points (e.g. `auto-fit`) implicitly depend on a prior `/api/load`. If you add multi-scene workflows, lift this state out first.
- **No per-point semantic labels yet** — Voxa is cuboid-instance-only. The PLY loader still reads `label` / `instance_id` channels, so a per-point mode is a small extension.
- **IoU is axis-aligned** in `_iou_aabb`; cuboid `rotation` is stored but ignored when scoring. Adequate for industrial poses where rotation is small; revisit if you start labeling rotated boxes.
- **Coordinate system**: Three.js Y-up. Cuboid `center`/`size` are in scene units; `rotation` is `[rx, ry, rz]` Euler XYZ in radians.

## Tests

- Backend: `pytest` (config in root `pyproject.toml`, `pythonpath = ["backend"]`). Tests live in `backend/tests/`. `conftest.py` sets `VOXA_DATA_DIR` to a tmp dir **before** importing `main`, because `main.py` reads that env at import time — preserve that ordering when adding fixtures.
- Frontend: `vitest` (config inlined in `vite.config.js`, environment `node`). Add `jsdom` + `@testing-library/react` if you start writing component tests; the current setup is pure-function only.
- Backend dev deps live in `backend/requirements-dev.txt` (separate from runtime `requirements.txt`).
