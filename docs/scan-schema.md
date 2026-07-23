# Scan directory schema (voxa-facing)

What voxa expects on disk for an annotated scan. The canonical, broader schema lives at `lidar/SCHEMA.md` in the data tree; this doc captures the subset voxa actually reads + writes so the layout is discoverable from inside the repo.

In code, this layout is encoded once in the shared `scan_schema` package (`scan_schema.layout::ScanLayout` and `SessionPaths`) — every module that needs a path inside a scan dir (`scene_registry`, `lidar_io`, `segment_io`, the load and save routes, `session_store`, `preseg_store`) resolves it there rather than hard-coding subpaths. The canonical source now lives in the separate `tools/scan-schema` repo; update `scan_schema.layout` and this doc together when the on-disk layout changes.

**Dependency pin discipline:** `backend/requirements.txt` pins `scan-schema` to a specific commit SHA, not `@main` — bump it explicitly (as its own dependency-bump commit) whenever `scan_schema` changes upstream. Tracking `@main` would mean any change there ships to voxa silently on the next `pip install`/CI run with no review step; a pinned commit makes every scan_schema upgrade an explicit, reviewable diff.

**Eval-labeling invariants (phase 3):** `scan_schema.eval_invariants` (9 checks, a distinct set from the SCHEMA invariants below — separately numbered to avoid collision) and `scan_schema.manifest::build_manifest` gate voxa's save-time write path (`backend/labeling/segment_io.py::save_labels`, via `_check_eval_invariants`) and are also wired into `scan_schema.validate_archive` for load-time auditing (`scan_schema validate --frozen-ids ...`). See `docs/superpowers/specs/2026-07-22-eval-invariants-manifest-design.md` for the full design and `scripts/migrate_eval_invariants.py` for the one-off migration that brought pre-phase-2 sessions into compliance (legacy frozen-class conversion + orphaned-presegment stripping).

## Where scans live

```
$VOXA_LIDAR_ROOT/annotated/<scan_name>/
```

`$VOXA_LIDAR_ROOT` defaults to `/home/hendrik/coding/engine/data/lidar`. Override via env. Every directory under `annotated/` with a `source/scan.ply` and a `meta.json` that passes `scan_schema.metadata.check_meta` (no errors) is a scene — i.e. both **2.x** scans (grandfathered: missing frame/derivation are warnings) and **3.0** scans (fully validated). Legacy v1.3 scans (or any unsupported `schema_version`) are skipped at discovery with a log hint to run `scripts/migrate_scan_v2.py`. Promote a 2.x scan to v3.0 lineage with `scripts/scan/promote_to_v3.py` (see Migration below).

Two additional tiers — `decimated/` (raw PLY previews under `<lidar_root>/ply_viewer/*.ply`) and `raw/` (LAZ source files under `<lidar_root>/raw/*.laz`, also the registry roots in `raw/sources.json`) — are surfaced by the scene picker as discovery shortcuts for unlabeled data. Neither has the per-scan substructure below; to label one, scaffold it into `annotated/` via `data/tools/scaffold_annotation.py`.

## Naming convention

Scan names and source IDs follow a structured slug:

```
scan_name  = <scene>_<vendor>[_<density>]
source_id  = <scene>_<vendor>
```

- **scene** — one or more lowercase alphanumeric tokens joined by underscores (`[a-z0-9]+(_[a-z0-9]+)*`).
- **vendor** — a known capture vendor from `scan_schema.KNOWN_VENDORS` (currently `navvis`, `matterport`). To admit a new vendor, extend that tuple in the `scan_schema` package.
- **density** — an approximate point-count suffix (`\d+[km]` or `full`, e.g. `500k`, `3m`). Include it **only** when more than one density of the same `<scene>_<vendor>` is materialized on disk; for a single-density scan, `scan_name` equals the `source_id` base.

The scheme is scene-first so names sort and grep cleanly by subject. Both IDs are stable slugs: rename only when the capture itself is being superseded.

**Enforcement**: warn-level only. `scan_schema.is_valid_scan_name` / `is_valid_source_id` flag non-conforming names during `validate_archive` / `python -m scan_schema <root>` — a warning, never an error, so discovery is never blocked.

**Examples** (real archive scans): `water_treatment_navvis`, `smart_ais_navvis`, `mechanical_room_matterport`, `water_pump_navvis_500k`, `water_pump_navvis_3m`.

## Per-scan layout (`annotated/<scan_name>/`)

```
<scan_name>/
├── meta.json                          (required)  provenance + frame/derivation; schema_version "2.x" or "3.0"
├── README.md                          (recognized, not enforced — every current scan has one)
├── source/
│   ├── scan.ply                       (required)  point cloud being labeled (xyz + rgb)
│   └── mesh.glb                       (optional)  textured mesh for the companion window
├── prelabel/                          (optional)  N presegment results, each independently addressable
│   └── <preseg_id>/                   e.g. ransac, sam3_orbit48  ([a-z0-9_-]+ only)
│       ├── instance_ids.npy           int32, shape (N_pts,)
│       ├── segment_summary.json       { "segments": [{"id": int, "class_id": int}, ...] }
│       └── meta.json                  { "preseg_id", "generator", "params",
│                                        "fingerprint", "n_segments", "created_at" }
├── sessions/                          (optional)  N labeling sessions, each self-contained
│   └── <session_id>/                  e.g. 20260603-143000_ransac  ([a-z0-9_-]+ only)
│       ├── session.json               pin + state (schema below)
│       ├── working_class_ids.npy      int8,  shape (N_pts,)  — autosave
│       ├── working_segment_ids.npy    int32, shape (N_pts,)  — autosave
│       ├── instances_gt.json          cuboid/pointset instance doc (right panel) — session-scoped, autosave
│       ├── centerlines.json           Draw sub-mode paths (optional; absent until first centerline apply)
│       ├── output/                    written by Save (Ctrl+S); absent until first explicit save
│       │   ├── gt_class_ids.npy       int32, shape (N_pts,)
│       │   ├── gt_segment_ids.npy     int32, shape (N_pts,)
│       │   └── gt_segment_metadata.json
│       └── history/<YYYYMMDD_HHMMSS>/ per-session save backups (10 most recent kept)
├── renders/<run_id>/
│   ├── manifest.json                  poses + intrinsics per frame
│   ├── meta.json                      render-run pin (generated_from, frame, intrinsics)
│   └── frame_NNN_*.png                color renders
└── sam3/                              (unchanged)  feature cache + pipeline workspace
    ├── sam3_features.npz
    └── <run_id>/                      per-render-set outputs (not promoted to prelabel/ automatically)
```

**Removed relative to v1.3**: top-level `labels/`, `session/`, `annotation_history/`
(absorbed into `sessions/<id>/`).

### What voxa REQUIRES vs what's nice-to-have

| Path | Voxa behavior if missing |
|---|---|
| `README.md` | no effect — recognized by the validator (in `ALLOWED_TOPLEVEL`) but never checked for presence |
| `source/scan.ply` | scene not discovered |
| `meta.json` passing `scan_schema.metadata.check_meta` (2.x or 3.0) | scene not discovered; legacy/unsupported `schema_version` logged with a migration hint |
| `sessions/` | UI shows an empty session list + create-session picker; no canvas until a session is created or selected |
| `prelabel/` | blank-start sessions only; preseg dropdown in session-create UI shows nothing |
| `sessions/<id>/output/` | session listed as unsaved (no `has_output` flag); Compare mode has no GT from that session |
| `source/mesh.glb` | "▦ Mesh window" button greyed out |
| `renders/<run>/manifest.json` | `/api/sam3/renders` returns empty; no SAM3 in-app preseg |
| `sam3/*` | recomputed on next SAM3 run (cache miss) |

**What writes what:**
- **Voxa** writes `sessions/*/` (working arrays, session.json, output/, history/). Nothing else under `sessions/` is written externally.
- **Offline pipelines** (scripts/preseg/presegment*.py, SAM3 post-process) write `prelabel/*/` via `register_preseg()` in `preseg/preseg_store.py`. Out-of-band edits to `instance_ids.npy` that skip `register_preseg` are not detected by pin checks — `meta.json::fingerprint` is the preseg's identity.
- **Nothing writes `labels/`** in v2. The v1.3 `labels/` slot is gone; GT lives under `sessions/<id>/output/`.
- **Downstream consumers** enumerate `sessions/*/output/` and choose which session's GT to use. There is no single canonical ground truth.

### File-level contracts

- **`source/scan.ply`** — binary little-endian PLY, properties `x y z` (float32) and `red green blue` (uchar). `N_pts = len(vertices)`; every per-point array in `prelabel/*/` and `sessions/*/` MUST have shape `(N_pts,)`.
- **`source/mesh.glb`** — single canonical filename. Build pipelines that emit variants MUST rename or hardlink one to `mesh.glb`. The viewer registers `MeshoptDecoder` so `EXT_meshopt_compression` GLBs load.
- **`prelabel/<id>/instance_ids.npy`** — `int32`, shape `(N_pts,)`. Read alongside `segment_summary.json` which maps each `id` to a `class_id` (use `-1` for class-agnostic preseg). Written by `register_preseg()` only. Model predictions should be registered the same way — `register_preseg(generator="model-…")` — which makes them selectable both as session seeds in Label mode and as Compare sources.
- **`prelabel/<id>/meta.json`** — the preseg's identity. Keys: `preseg_id`, `generator`, `params`, `fingerprint` (sha256 over dtype+shape+bytes of `instance_ids.npy`), `n_segments`, `created_at`. The fingerprint is computed once at register time; pin checks string-compare against it without re-hashing the array.
- **`sessions/<id>/session.json`** — see schema below.
- **`sessions/<id>/working_class_ids.npy`** — `int8`, shape `(N_pts,)`. Autosave; in-progress class state.
- **`sessions/<id>/working_segment_ids.npy`** — `int32`, shape `(N_pts,)`. Autosave; in-progress segment state. The `int8`/`int32` dtype asymmetry is intentional and carries over from v1.3 (autosave compactness vs export precision); do not unify.
- **`sessions/<id>/working_categories.npy`** — `int8`, shape `(N_pts,)`. Autosave; the point-category (annotation-status) axis added by eval-labeling phase 2: `0` none, `1` artifact, `2` transient, `3` excluded_review (`backend/labeling/categories.py`). Optional — absent means a session written before phase 2, which reads as all-`none`; a present file with the wrong shape is a hard error, not a silent reset.
- **`sessions/<id>/centerlines.json`** — Draw sub-mode persistence. Optional; absent until the first centerline apply. Flat shape `{ "paths": [{ "points": [[x,y,z],…], "radius", "smooth", "class_id", "instance_id" }] }` with coordinates in the recentered viewer frame. Write semantics are replace-by-`instance_id`: an apply targeting an existing instance replaces that instance's stored paths; a new instance appends; instance ids absorbed by a merge (`merged_from` in the apply request) have their entries deleted in the same write. Written by `labeling/centerline.py::update_centerlines` only; the per-point labels it produced live in the working arrays like any other edit (undo reverts labels, not this file).
- **`sessions/<id>/output/gt_class_ids.npy`** — `int32`, shape `(N_pts,)`. `-1` = unlabeled. Written by explicit Save (Ctrl+S).
- **`sessions/<id>/output/gt_segment_ids.npy`** — `int32`, shape `(N_pts,)`. `-1` = unlabeled.
- **`sessions/<id>/output/gt_point_category.npy`** — `int8`, shape `(N_pts,)`. Explicit-Save mirror of `working_categories.npy` (phase 2). Points marked `excluded_review` belong to a *review blob* — a class-less instance — and are therefore stripped from `gt_segment_ids` by the same rule that strips unclassified presegments; their blob ids live in the metadata's `review_blobs` until phase 3 moves voxa onto the upstream file set.
- **`sessions/<id>/output/gt_point_component_ids.npy`** — `int16`, shape `(N_pts,)`. Fragment index *within* each instance (0-based per instance), `-1` outside instances. Derived at save time by `backend/labeling/components.py::component_ids` (voxel-grid 26-connectivity at `component_link_radius_m`), never hand-labeled.
- **`sessions/<id>/output/gt_segment_metadata.json`** — `{ "n_points", "n_gt_segments", "n_labeled_points", "class_map_version", "segments": [...] }`, plus (phase 2) `"categories"` (per-name point histogram), `"review_blobs"` (`[{instance_id, n_points}]`), and `"component_link_radius_m"`. The `class_map_version` value here is an output mirror — it records which `classes.json::version` was active at save time. Invariant 6 is enforced by `scan_schema.invariants.validate_invariants` comparing `meta.json::class_map_version` (read via `segment_io._read_meta_class_map_version`) against `classes.json::version` at save time; if `meta.json` is missing the check is skipped.
- **`renders/<run>/manifest.json`** — `{ "scene", "frames": [{ "file", "position": [x,y,z], "target": [x,y,z], ... }] }`. SAM3 feature extraction reads `file`/`position`/`target` (or `yaw`).

### `session.json` schema

```json
{
  "schema_version": 2,
  "name": "first pass with sam3",
  "preseg_id": "sam3_orbit48",
  "preseg_fingerprint": "sha256:…",
  "source_fingerprint": "sha256:…",
  "created_at": "2026-06-03T14:30:00Z",
  "saved_at": "2026-06-03T15:10:42Z",
  "dirty": true,
  "hidden_inst_ids": []
}
```

- `preseg_id` / `preseg_fingerprint` are `null` for blank-start sessions.
- `is_from_prelabel` is **not stored** — it is derived as `preseg_id != null`.
- `saved_at` = last persisted edit (stamped by autosave and explicit save). Opening a session without editing does not update it. It is the sort key used by `last_worked()` to determine which session to auto-resume.
- `session_id` (the directory name) is generated at creation as `<YYYYMMDD-HHMMSS>_<preseg_id|blank>` and never renamed. `name` is the display name; rename is a metadata-only edit to `session.json`.

### Session pinning semantics

Pins are frozen at create time: `source_fingerprint` covers the recentered cloud positions, `preseg_fingerprint` covers `instance_ids.npy` — both computed via `scan_schema.fingerprint.array_fingerprint` (sha256 over dtype + shape + bytes; re-exported through `segment_io`).

**On resume** (`/api/load`): the backend calls `session_store.verify_pins()` which string-compares the session's `preseg_fingerprint` against `prelabel/<id>/meta.json::fingerprint` (no array load or re-hashing on the hot path) and the loaded cloud's current fingerprint against `source_fingerprint`.

**Any mismatch → HTTP 409** with body `{detail, diverged: "preseg" | "source"}`. The UI renders this as a blocking banner — the scene stays unloaded, no silent fallback, no auto-repair. Session data remains on disk; resumability is restored by restoring the preseg or cloud to the pinned state.

A deleted preseg makes its pinned sessions refuse to resume. Out-of-band edits to `instance_ids.npy` that bypass `register_preseg` are not detected (the meta.json fingerprint is the preseg's identity; re-registering is the only supported way a preseg changes).

### Invariants the save endpoint enforces

These v1.3 invariants carry over, applied **per session's `output/`**:

1. **Invariant 3**: `class_id == -1 ⟺ instance_id == -1` (per point in the output files). Preseg-only points are dropped to `-1` in export; the in-memory canvas and `working_*.npy` keep the preseg structure for reload.
2. **Invariant 4**: per-segment class consistency — every point sharing an `instance_id` must share the same `class_id`.
3. **Invariant 6**: enforced by `scan_schema.invariants.validate_invariants` comparing `meta.json::class_map_version` (read via `segment_io._read_meta_class_map_version`) against `classes.json::version` at save time; if `meta.json` is missing the check is skipped. The `class_map_version` written into `gt_segment_metadata.json` records which registry version the save used and is never read back for enforcement.

### Environment

- `VOXA_LIDAR_ROOT` — root of the canonical lidar archive (default `/home/hendrik/coding/engine/data/lidar`).
- `VOXA_DATA_DIR` — voxa-local data dir for legacy `data/scenes/*` + `data/annotations/*` (cuboid GT/predictions for the old workflow).
- `VOXA_RENDERS_ROOT` — **legacy fallback** for SAM3 render discovery. Only consulted when the active scene doesn't have its own `renders/` subdir. Prefer the per-scan layout for new scenes.
- `VOXA_CONFIG` — path to `classes.yaml` (default `config/classes.yaml`).
- `VOXA_MAX_POINTS` — viewer subsample cap (default `1000000`). Labels operate on the full N_pts.

## Adding a new scan

```bash
cd /home/hendrik/coding/engine/data
python3 tools/scaffold_annotation.py path/to/source.laz --out-root lidar/annotated --target 3000000
```

The scaffolder must be updated to emit the v2 skeleton (`schema_version: "2.0"`, no `labels/`, empty `prelabel/`). A prepared v2 scaffolder and companion SCHEMA.md live in `docs/companion-engine-data/` in the voxa repo — copy them into the `engine/data` tree at deploy time, then run `scripts/migrate_scan_v2.py` on existing scans (dry-run first).

Add `prelabel/<preseg_id>/` from your offline pipeline (via `register_preseg()`) and `renders/` before opening voxa to label.

## Migration from v1.3

`scripts/migrate_scan_v2.py` — one-shot, in-place, idempotent (skips scans whose `meta.json` already says `schema_version` starting with `"2"`):

```
labels/gt_*                  → sessions/legacy/output/
session/{current.json,*.npy} → sessions/legacy/  (session.json synthesized;
                               name "legacy"; created_at/saved_at from file mtimes)
annotation_history/*         → sessions/legacy/history/
prelabel/ransac_*            → prelabel/ransac/{instance_ids.npy,
                               segment_summary.json, meta.json}
meta.json                    → schema_version: "2.0"
```

Working-array coalescing rule: `working_* = session/working_*` if present, else the migrated GT cast to int8/int32, else all `-1`. The legacy session's pins are **recomputed** from the migrated `prelabel/ransac/instance_ids.npy` and the cloud (same path as `scan_schema.fingerprint.array_fingerprint` in the load route) — not copied from the old `prelabel_fingerprint` field — so the pin provably matches what is on disk after migration.

Usage:
```bash
python scripts/migrate_scan_v2.py --dry-run /home/hendrik/coding/engine/data/lidar
python scripts/migrate_scan_v2.py /home/hendrik/coding/engine/data/lidar
# Migrate only specific scans:
python scripts/migrate_scan_v2.py --scan munich_water_pump /home/hendrik/coding/engine/data/lidar
```

The script refuses loudly (per scan) on anything unexpected: a `sessions/` dir already present, unexpected files in `prelabel/`, shape mismatches. Recovery from a mid-migration crash is manual; the second run refuses rather than guessing.

## Promotion to v3.0

`migrate_scan_v2.py` lands a scan at `2.0` (no frame/derivation — grandfathered). To promote it to full **v3.0 lineage**, use `scripts/scan/promote_to_v3.py`, which routes the write through `scan_schema.Registry.set_derivation`:

```bash
python scripts/scan/promote_to_v3.py /home/hendrik/coding/engine/data/lidar --dry-run
python scripts/scan/promote_to_v3.py /home/hendrik/coding/engine/data/lidar
# or a subset:
python scripts/scan/promote_to_v3.py /home/hendrik/coding/engine/data/lidar --only smart_ais_clean
```

Per scan it: resolves the raw root (via `meta.source_laz` basename, else by scan name, from `raw/sources.json`), synthesizes an identity `<scan_id>#local` frame if absent (the stored cloud is its own canonical-local; `coord_offset_m` is preserved as georef), and calls `set_derivation(..., bump_to_3=True)` — writing the nested `root`/`parent` (`file_sha256`) derivation and flipping `schema_version` to `3.0`. `bump_to_3` makes frame/derivation **hard requirements**, so a scan with no resolvable root (e.g. a source whose `.laz` isn't registered) is **skipped** and stays 2.x. Regenerate `variants.json` afterward with `scripts/scan/scan_index.py`.

**Voxa reads both 2.x and 3.0** (discovery gate = `check_meta`, see *Where scans live*). Only legacy v1.3 (or unsupported versions) fail discovery and are skipped, logged at INFO with the found `schema_version` and the migrate script.
