# Scan directory schema v2.0 (voxa-facing)

What voxa expects on disk for an annotated scan. The canonical, broader schema lives at `lidar/SCHEMA.md` in the data tree; this doc captures the subset voxa actually reads + writes so the layout is discoverable from inside the repo.

In code, this layout is encoded once in `backend/scenes/scan_layout.py::ScanLayout` and `SessionPaths` — every module that needs a path inside a scan dir (`scene_registry`, `lidar_io`, `segment_io`, the load and save routes, `session_store`, `preseg_store`) resolves it there rather than hard-coding subpaths. Update `ScanLayout` and this doc together when the on-disk layout changes.

## Where scans live

```
$VOXA_LIDAR_ROOT/annotated/<scan_name>/
```

`$VOXA_LIDAR_ROOT` defaults to `/home/hendrik/coding/engine/data/lidar`. Override via env. Every directory under `annotated/` that contains a `source/scan.ply` **and** a `meta.json` with `schema_version` starting with `"2"` is a scene. v1.3 scans (any other `schema_version` or none) are skipped at discovery with a log hint to run `scripts/migrate_scan_v2.py`.

Two additional tiers — `decimated/` (raw PLY previews under `<lidar_root>/ply_viewer/*.ply`) and `raw/` (LAZ previews under `<lidar_root>/laz/*.laz`) — are surfaced by the scene picker as discovery shortcuts for unlabeled data. Neither has the per-scan substructure below; to label one, scaffold it into `annotated/` via `data/tools/scaffold_annotation.py`.

## Per-scan layout (`annotated/<scan_name>/`)

```
<scan_name>/
├── meta.json                          (required)  provenance + schema_version: "2.0"
├── README.md                          (recommended)
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
│       ├── output/                    written by Save (Ctrl+S); absent until first explicit save
│       │   ├── gt_class_ids.npy       int32, shape (N_pts,)
│       │   ├── gt_segment_ids.npy     int32, shape (N_pts,)
│       │   └── gt_segment_metadata.json
│       └── history/<YYYYMMDD_HHMMSS>/ per-session save backups (10 most recent kept)
├── renders/<run_id>/                  (unchanged)
│   ├── manifest.json                  poses + intrinsics per frame
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
| `source/scan.ply` | scene not discovered |
| `meta.json` with `schema_version` starting with `"2"` | scene not discovered; logged as v1.3 with migration hint |
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
- **`prelabel/<id>/instance_ids.npy`** — `int32`, shape `(N_pts,)`. Read alongside `segment_summary.json` which maps each `id` to a `class_id` (use `-1` for class-agnostic preseg). Written by `register_preseg()` only.
- **`prelabel/<id>/meta.json`** — the preseg's identity. Keys: `preseg_id`, `generator`, `params`, `fingerprint` (sha256 over dtype+shape+bytes of `instance_ids.npy`), `n_segments`, `created_at`. The fingerprint is computed once at register time; pin checks string-compare against it without re-hashing the array.
- **`sessions/<id>/session.json`** — see schema below.
- **`sessions/<id>/working_class_ids.npy`** — `int8`, shape `(N_pts,)`. Autosave; in-progress class state.
- **`sessions/<id>/working_segment_ids.npy`** — `int32`, shape `(N_pts,)`. Autosave; in-progress segment state. The `int8`/`int32` dtype asymmetry is intentional and carries over from v1.3 (autosave compactness vs export precision); do not unify.
- **`sessions/<id>/output/gt_class_ids.npy`** — `int32`, shape `(N_pts,)`. `-1` = unlabeled. Written by explicit Save (Ctrl+S).
- **`sessions/<id>/output/gt_segment_ids.npy`** — `int32`, shape `(N_pts,)`. `-1` = unlabeled.
- **`sessions/<id>/output/gt_segment_metadata.json`** — `{ "n_points", "n_gt_segments", "n_labeled_points", "class_map_version", "segments": [...] }`. The `class_map_version` value here is an output mirror — it records which `classes.json::version` was active at save time. Invariant 6 is enforced by `segment_io._validate_invariants` comparing `meta.json::class_map_version` (read via `_read_meta_class_map_version`) against `classes.json::version` at save time; if `meta.json` is missing the check is skipped.
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

Pins are frozen at create time: `source_fingerprint` covers the recentered cloud positions, `preseg_fingerprint` covers `instance_ids.npy` — both computed via `segment_io.compute_fingerprint()` (sha256 over dtype + shape + bytes).

**On resume** (`/api/load`): the backend calls `session_store.verify_pins()` which string-compares the session's `preseg_fingerprint` against `prelabel/<id>/meta.json::fingerprint` (no array load or re-hashing on the hot path) and the loaded cloud's current fingerprint against `source_fingerprint`.

**Any mismatch → HTTP 409** with body `{detail, diverged: "preseg" | "source"}`. The UI renders this as a blocking banner — the scene stays unloaded, no silent fallback, no auto-repair. Session data remains on disk; resumability is restored by restoring the preseg or cloud to the pinned state.

A deleted preseg makes its pinned sessions refuse to resume. Out-of-band edits to `instance_ids.npy` that bypass `register_preseg` are not detected (the meta.json fingerprint is the preseg's identity; re-registering is the only supported way a preseg changes).

### Invariants the save endpoint enforces

These v1.3 invariants carry over, applied **per session's `output/`**:

1. **Invariant 3**: `class_id == -1 ⟺ instance_id == -1` (per point in the output files). Preseg-only points are dropped to `-1` in export; the in-memory canvas and `working_*.npy` keep the preseg structure for reload.
2. **Invariant 4**: per-segment class consistency — every point sharing an `instance_id` must share the same `class_id`.
3. **Invariant 6**: enforced by `segment_io._validate_invariants` comparing `meta.json::class_map_version` (read via `_read_meta_class_map_version`) against `classes.json::version` at save time; if `meta.json` is missing the check is skipped. The `class_map_version` written into `gt_segment_metadata.json` records which registry version the save used and is never read back for enforcement.

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

Working-array coalescing rule: `working_* = session/working_*` if present, else the migrated GT cast to int8/int32, else all `-1`. The legacy session's pins are **recomputed** from the migrated `prelabel/ransac/instance_ids.npy` and the cloud (same path as `compute_fingerprint` in the load route) — not copied from the old `prelabel_fingerprint` field — so the pin provably matches what is on disk after migration.

Usage:
```bash
python scripts/migrate_scan_v2.py --dry-run /home/hendrik/coding/engine/data/lidar
python scripts/migrate_scan_v2.py /home/hendrik/coding/engine/data/lidar
# Migrate only specific scans:
python scripts/migrate_scan_v2.py --scan munich_water_pump /home/hendrik/coding/engine/data/lidar
```

The script refuses loudly (per scan) on anything unexpected: a `sessions/` dir already present, unexpected files in `prelabel/`, shape mismatches. Recovery from a mid-migration crash is manual; the second run refuses rather than guessing.

**Voxa v2 reads ONLY v2.** A non-v2 scan fails discovery and is skipped; the skip is logged at INFO level naming the found `schema_version` and the migrate script (`scripts/migrate_scan_v2.py`).
