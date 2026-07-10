# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Voxa is a unified 3D scan studio for industrial LiDAR / point clouds. It replaces the older `3d-labeler` and `industrial-point-labeler` tools. The UI is organized around four modes: **Inspect** (fast review), **Label** (see below), **Compare** (two finished labelings — session outputs or presegs — diffed per-point: agreement coloring, per-class IoU/precision/recall, confusion matrix), and **Edit** (non-destructive OBB slicing → full-density PLY export).

**Label mode is unified around one principle: a tool is only a way to *select points*; everything downstream is one shared pipeline** (`select → apply+label → unconfirmed pointset → confirm`). A single `activeTool` state drives a 3-tool rail in the viewport top toolbar (`frontend/src/tool-rail.jsx`, list/gating in `label-tools.js`) and one contextual tool-options panel in the left rail (`frontend/src/tool-options.jsx`):
- **Presegment** — Ctrl/Shift-click precomputed segments (or the presegment list); a `rapid` toggle = the old fast-labeling queue (largest-first, one-keypress classify, see `frontend/src/fast-label.jsx`).
- **Box** — draw an oriented box, transform it (G/R/Y or the panel buttons), apply → the box vanishes and its enclosed points become a pointset. The box is a transient selection gizmo, not a persisted annotation.
- **Draw** — label pipes/tanks by drawing centerline paths (raw RGB cloud, walk nav, point anchors/branches, M merges; applied paths surface as pointset rows). The centerline graph persists per-session in `sessions/<id>/centerlines.json` — see `frontend/src/draw-mode.jsx`.

Every apply produces a `kind:'pointset'` instance (the geometric `cuboid` type is retired for new labels — legacy cuboids render display-only, no gizmo). Apply = Ctrl+Enter → class picker, or a class hotkey directly. A per-tool **auto-confirm on apply** toggle is the only deviation from unconfirmed-first (rapid preseg defaults on). Geometric tools (Box, Draw) reach the backend through **one generic endpoint** `POST /api/segment/apply-shape` (`backend/routes/segment.py` + `backend/labeling/shapes.py`): it resolves a shape (`obb` | `tube`; `beam` reserved) to the FULL-RES points inside it via `shape_indices`, then `apply_reassign`s them — this is why Box/Draw need the backend (the rendered cloud is subsampled but labels are full-res). `centerline-apply` is a thin `tube`-shaped alias of `apply-shape`.

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
- `VOXA_LIDAR_ROOT` — root of the canonical lidar archive (default `/home/hendrik/coding/engine/data/lidar`). Adds `annotated/` scenes to the picker; set to a missing path to disable.
- `VOXA_CONFIG` — path to a `classes.yaml` (default `config/classes.yaml`)
- `VOXA_MAX_POINTS` — server-side subsample cap (default `1000000`)
- `VOXA_RELOAD=1` — enable uvicorn `--reload` (off by default to avoid hitting the system inotify limit)
- `VOXA_BACKEND`, `VOXA_FRONTEND_PORT` — Vite dev proxy target / port

## Architecture

**Backend** (`backend/`, FastAPI, served by `uvicorn main:app`):
- `main.py` — assembles the FastAPI app and registers routers from `backend/routes/*.py`. HTTP endpoints live in `routes/{load,segment,sessions,compare,export,meta}.py`; Pydantic schemas in `app/schemas.py`; in-memory `_state` + helpers (`_recenter`, `_resume_session`, …) in `app/core.py`. Compare scoring is per-point (class agreement, per-class IoU/P/R, confusion matrix); there is no longer any cuboid-based IoU diff.
- `scene_registry.py` — multi-root scene discovery. Returns `SceneSource` for each tier (`legacy` / `annotated`). Scene IDs are tier-prefixed (`annotated/munich_water_pump`); bare legacy names still resolve.
- `lidar_io.py` — `load_annotated` (SCHEMA-conformant v2 scans; reads working arrays from `sessions/<id>/`) and `load_laz` (chunked, stride-sampled via `laspy[lazrs]`). Auto-recenter for float32 stability lives in `app/core.py::_recenter`.
- `point_cloud.py`, `supervoxels.py`, `clustering.py`, `fitting.py` — carried over from the old `3d-labeler`. Only the PLY/GLB loader and a small `auto-fit` (snap a cuboid to points inside an AABB) are wired into the current frontend; the supervoxel / cluster / RANSAC modules are present for future use but not exposed via routes.
- Static frontend is mounted at `/` **only if `dist/` exists**, so `/api/*` always wins. In dev mode (`dist/` absent) the FastAPI process is API-only and Vite owns the UI.

**Frontend** (`frontend/src/`, Vite + React 18 + Three.js, no TypeScript):
- `main.jsx` mounts `<App>` plus `<Agentation>` (in-app feedback toolbar from the `agentation` package).
- `App.jsx` is the shell: owns `scenes / activeScene / cloud / classes / gtInstances / predInstances` state, fetches config + scenes once, then re-loads cloud + GT + predictions whenever `activeScene` changes. It dispatches to one of three mode components and handles `⌘S/Ctrl+S` save. The `ScenePicker` surfaces **annotated scenes plus the `legacy/test_scene` sandbox** — all other legacy scenes are filtered out of the picker UI (the backend still discovers/resolves them for internal guards). When no scene is selected the default falls back to `legacy/test_scene`, then the first scene.
- `mode-inspect.jsx`, `mode-label.jsx`, `mode-compare.jsx` — feature surfaces. They share `viewer.jsx` (the Three.js viewport with `attachOrbit` and `attachWalk` camera schemes; `navMode` is lifted to `App` so it persists across mode switches) and the small atoms in `viewport-atoms.jsx`.
- `api.js` is the only place that talks to `/api/*`. Point cloud `positions`/`colors` come over the wire as base64-encoded `Float32Array` payloads — decode via `b64ToFloat32`.
- `tweaks-panel.jsx` is a generic tweak-host (theme/mode controls). It implements an external "edit mode" host protocol via `postMessage` (`__activate_edit_mode` / `__edit_mode_set_keys` / etc.) — do not delete those listeners without checking the Agentation integration.

**Data layout** (gitignored):
```
data/
├── scenes/<name>/source.{ply,glb}              # input
└── annotations/<name>/{ground_truth,predictions}.json
```
Both annotation files share one schema (`AnnotationDoc` in `app/schemas.py`); a model's output dropped at `predictions.json` is enough to power Compare mode.

**Annotated scan layout** (under `$VOXA_LIDAR_ROOT/annotated/<name>/`):
```
<name>/
├── meta.json              schema_version "2.x" (grandfathered) or "3.0"
├── source/scan.ply        point cloud (required)
├── prelabel/<preseg_id>/  each pipeline result: instance_ids.npy, segment_summary.json, meta.json
└── sessions/<session_id>/ each labeling session: session.json, working_*.npy, output/gt_*, history/
```
No `labels/`, `session/`, or `annotation_history/` — those are absorbed into `sessions/`.
The schema (layout, meta contract, fingerprints, invariants, validation, lineage, durable writes) is
defined once in the shared **`scan_schema`** package (`tools/scan-schema`, declared as a VCS dependency
in `backend/requirements.txt`) — voxa imports it rather than re-implementing it. See `docs/scan-schema.md`
for the voxa-facing subset and the cross-tool `lidar/SCHEMA.md` for the full contract.

**Class config**: `config/classes.yaml`. Each entry has `label`, `color` (hex string or `[r,g,b]` floats 0-1), and `key` (hotkey). Restart the backend or re-fetch `/api/config` to pick up edits.

## Conventions and gotchas

- **EDITMODE markers**: `App.jsx` defines tweak defaults inside `/*EDITMODE-BEGIN*/ ... /*EDITMODE-END*/`. The Agentation/tweaks tooling rewrites this region, so preserve the markers verbatim when editing defaults.
- **Vite uses polling watchers** (`vite.config.js`) on purpose — to avoid blowing the system-wide inotify limit when many other projects are open. Keep it.
- **Backend has no autoreload by default** — after editing Python files, kill and restart `npm run dev` (or set `VOXA_RELOAD=1`). This matches the broader "restart server after code changes" rule from the global config.
- **Single in-memory point cloud**: `_state` in `app/core.py` holds one cloud at a time. Endpoints that need the points (e.g. `auto-fit`) implicitly depend on a prior `/api/load`. If you add multi-scene workflows, lift this state out first.
- **Multi-session labeling model**: each annotated scan holds N labeling sessions under `sessions/<session_id>/`. The backend keeps ONE active `SegmentSession` at a time. Label mode (annotated tier) shows a session picker: list, create (name + preseg or blank), rename, delete, switch. Opening a scan auto-resumes the last-worked session (max `saved_at`). Save (Ctrl+S) writes to the active session's `output/` dir; downstream consumers enumerate `sessions/*/output/` — there is no single canonical GT slot. The cuboid/pointset instance doc (right Instances panel) is also session-scoped: `sessions/<id>/instances_gt.json` via `?session_id=` on `/api/annotations/*` (the scene-global `data/annotations/` path is legacy-tier only).
- **Session pinning + 409**: each session is pinned at creation to a preseg fingerprint (`prelabel/<id>/meta.json::fingerprint`) and a source fingerprint. On resume, voxa string-compares both pins. Any mismatch returns HTTP 409 `{detail, diverged: "preseg"|"source"}` and renders as a blocking banner in the UI. Re-registering the preseg (via `register_preseg()`) is the only supported way to update a preseg.
- **Per-point labels are read-only in Inspect** — annotated scans load working arrays from the active session (`sessions/<id>/working_*.npy`), surfaced through `LoadResponse.class_ids` / `instance_ids` / `class_palette`. Inspect's "Color by Class / Instance" pills consume them. Editing per-point segments is Label mode only.
- **Scan directory schema** is what voxa expects on disk for an annotated scan (source.ply, prelabel/<preseg_id>/, sessions/<session_id>/, renders/, sam3/, ...). See `docs/scan-schema.md`. Discovery delegates to `scan_schema.metadata.check_meta`: 2.x scans are grandfathered (missing frame/derivation are warnings) and 3.0 scans are fully validated — both are discovered. Legacy v1.3 scans (with `labels/`, `session/`, `annotation_history/`) are not discovered — run `scripts/migrate_scan_v2.py` first. Promote a 2.x scan to v3.0 lineage with `scripts/scan/promote_to_v3.py` (registers its raw root + writes the v3.0 nested derivation via `scan_schema.Registry.set_derivation`).
- **Auto-recenter on load** — `_recenter` in `app/core.py` subtracts the mean centroid (`points.mean(axis=0)`) when any coord exceeds 1e3 (LAS UTM scenes). The offset is in `LoadResponse.recenter_offset`. Cuboid endpoints operate in the recentered frame.
- **Cuboids are retired for new labels** — Label mode no longer creates `kind:'cuboid'` instances (every apply produces a `pointset`; the Box tool's OBB is a transient selection gizmo, not a persisted box). Legacy cuboid instances in already-saved sessions still render and are selectable/confirmable/deletable but display-only (no gizmo). The `Cuboid` schema + its `rotation` field remain for those legacy instances; `rotation` has no effect on Compare (which is per-point; `_iou_aabb` was removed).
- **Coordinate system**: Three.js Y-up. Cuboid `center`/`size` are in scene units; `rotation` is `[rx, ry, rz]` Euler XYZ in radians.

## Tests

- Backend: `pytest` (config in root `pyproject.toml`, `pythonpath = ["backend"]`). Tests live in `backend/tests/`. `conftest.py` sets `VOXA_DATA_DIR` to a tmp dir **before** importing `main`, because `main.py` reads that env at import time — preserve that ordering when adding fixtures.
- Frontend: `vitest` (config inlined in `vite.config.js`, environment `node`). Add `jsdom` + `@testing-library/react` if you start writing component tests; the current setup is pure-function only.
- Backend dev deps live in `backend/requirements-dev.txt` (separate from runtime `requirements.txt`).
