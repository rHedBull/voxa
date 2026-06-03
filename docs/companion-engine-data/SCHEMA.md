# Labeled Lidar Scan Schema (v2.0)

Every directory under `lidar/annotated/<scan_name>/` follows this layout. A single PyTorch `Dataset` iterates over them without per-scan branching.

> **v2.0 introduces multi-session labeling and formalized preseg results**, replacing the
> single `labels/` GT slot with per-session `sessions/<id>/output/` dirs, and replacing the
> flat `prelabel/ransac_*` files with individually addressable `prelabel/<preseg_id>/` subdirs.
> The full design + rationale is in `voxa/docs/superpowers/specs/2026-06-03-multi-session-preseg-design.md`.
> **v2.0 is NOT backward-compatible**: voxa v2 reads only v2 scans. Run
> `voxa/scripts/migrate_scan_v2.py` on all existing scans before deploying voxa v2.

## Directory layout

```
<scan_name>/
├── README.md                          (required)  free-form scan description
├── meta.json                          (required)  provenance + schema_version: "2.0"
├── source/
│   ├── scan.ply                       (required)  point cloud being labeled
│   └── mesh.glb                       (optional)  canonical geometry, if sampled from a mesh
├── prelabel/                          (optional)  N presegment results
│   └── <preseg_id>/                   e.g. ransac, sam3_orbit48
│       ├── instance_ids.npy           int32, shape (N_pts,)
│       ├── segment_summary.json       { "segments": [{"id": int, "class_id": int}, ...] }
│       └── meta.json                  { "preseg_id", "generator", "params",
│                                        "fingerprint", "n_segments", "created_at" }
├── sessions/                          (optional)  N labeling sessions
│   └── <session_id>/                  e.g. 20260603-143000_ransac
│       ├── session.json               pin + state (schema below)
│       ├── working_class_ids.npy      int8,  shape (N_pts,)  — autosave
│       ├── working_segment_ids.npy    int32, shape (N_pts,)  — autosave
│       ├── output/                    written by Save; absent until first explicit save
│       │   ├── gt_class_ids.npy       int32, shape (N_pts,)
│       │   ├── gt_segment_ids.npy     int32, shape (N_pts,)
│       │   └── gt_segment_metadata.json
│       └── history/<YYYYMMDD_HHMMSS>/ per-session save backups (10 most recent kept)
├── renders/                           (optional)  per-view captures of the scan
│   └── <run_id>/
│       ├── manifest.json              poses + intrinsics per frame
│       └── frame_NNN_*.png
└── sam3/                              (optional)  SAM3 feature cache + offline-pipeline outputs
    ├── sam3_features.npz
    └── <run_id>/
```

**Removed relative to v1.3**: top-level `labels/`, `session/`, `annotation_history/`.
These are absorbed into `sessions/<id>/`.

## File specs

### `source/scan.ply`
Binary PLY (little-endian preferred). Required vertex properties:
- `x, y, z` (float32)
- `red, green, blue` (uchar) — recommended; if absent, log it in `meta.json`.

Let `N_pts = len(vertices)`. **Every per-point array in `prelabel/*/` and `sessions/*/` MUST have shape `(N_pts,)`.**

Coordinates may be world-frame or recentered; declare which in `meta.json::coords`.

### `source/mesh.glb` (optional)
Textured mesh, used as the canonical geometry source if `scan.ply` was sampled from a mesh.

### `prelabel/<preseg_id>/` (optional)
Each preseg result is a self-contained subdirectory:

- **`instance_ids.npy`** — `int32`, shape `(N_pts,)`. `-1` = unassigned. Auto-segmentation output before manual curation.
- **`segment_summary.json`** — per-auto-segment metadata: `{"segments": [{"id": int, "class_id": int}, ...]}`. Use `"class_id": -1` for class-agnostic preseg.
- **`meta.json`** — the preseg's identity: `{"preseg_id", "generator", "params", "fingerprint", "n_segments", "created_at"}`. The `fingerprint` is sha256 over dtype+shape+bytes of `instance_ids.npy`, computed once by `register_preseg()` and never re-hashed on the load path. **Only `register_preseg()` (in voxa's `preseg/preseg_store.py`) should write this dir.**

`preseg_id` must match `[a-z0-9_-]+`. Out-of-band edits to `instance_ids.npy` that bypass `register_preseg()` are not detected by voxa's pin checks.

### `sessions/<session_id>/session.json`
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
`preseg_id`/`preseg_fingerprint` are `null` for blank-start sessions. `saved_at` = last persisted edit (autosave or explicit save); opening a session without editing does not update it.

### `sessions/<session_id>/output/gt_segment_metadata.json`
```json
{
  "n_points": 500000,
  "n_gt_segments": 126,
  "n_labeled_points": 99940,
  "class_map_version": 1,
  "segments": [
    {
      "gt_id": 0,
      "label": "pipe",
      "class_id": 0,
      "n_points": 441,
      "bbox": [x0, y0, z0, x1, y1, z1]
    }
  ]
}
```
`class_map_version` must equal `lidar/classes.json::version` at write time (invariant 6).

### `meta.json` (required)
```json
{
  "schema_version": "2.0",
  "scan_name": "munich_water_pump",
  "source_laz": "lidar/laz/NavVis-MLX-Sample-Data.laz",
  "source_mesh": "source/mesh.glb",
  "n_points": 500000,
  "sample_method": "uniform",
  "sample_param": {"target_points": 500000},
  "coords": "world",
  "units": "meters",
  "class_map_version": 1,
  "capture_date": "2024-09-01",
  "scanner": "NavVis VLX-3",
  "notes": ""
}
```
`schema_version: "2.0"` is required for voxa v2 discovery. Any other value (or absent) causes the scan to be skipped with a migration hint.

### `README.md` (required)
Human-readable description: what the scan is, who captured it, annotation status, known issues.

## Invariants (enforced by voxa's save endpoint)

1. `len(scan.ply) == n_points` in `meta.json`.
2. All per-point arrays have shape `(n_points,)` matching the stated dtype.
3. **Per session output**: `gt_class_ids[i] == -1` ⟺ `gt_segment_ids[i] == -1`.
4. **Per session output**: for every point with `gt_segment_ids[i] = s` and `s != -1`: `gt_class_ids[i] == segment_metadata.segments[s].class_id`.
5. All non-`-1` class IDs are present in `lidar/classes.json`.
6. **Per session output**: `class_map_version` in `gt_segment_metadata.json` matches the active `lidar/classes.json`.

## GT model: no canonical output, enumerate sessions

v2 has **no single canonical ground truth**. Downstream consumers must enumerate `sessions/*/output/` and pick. The labeling tier (voxa) presents all sessions to the user; external training pipelines should apply their own selection policy (e.g. most-recently-saved, or a specific named session). There is no automatic "default" session promoted for external consumption.

## Class map (`lidar/classes.json`)

Append-only. Existing IDs are immutable. New classes get new IDs.

## Annotation status convention

Document in `meta.json::notes` (or `README.md`):
- `"unlabeled"` — no sessions yet, or all sessions have no `output/`.
- `"in-progress"` — at least one session has `output/` but `n_labeled_points < n_points`.
- `"complete"` — at least one session fully labeled to standard.

## Session pinning

Each session is pinned at creation to:
1. The source cloud fingerprint (`sha256` over the recentered xyz positions).
2. The preseg fingerprint (`prelabel/<id>/meta.json::fingerprint`) — `null` for blank sessions.

On every resume, voxa checks both pins. Any mismatch → HTTP 409. Session data is never silently discarded; re-registering the preseg or restoring the cloud restores resumability.

## Per-view renders + SAM3 (optional)

Same as v1.3: `renders/<run_id>/manifest.json` + `frame_*.png`; `sam3/` for feature cache. SAM3 pipeline outputs go into `sam3/<run_id>/` and are promoted to `prelabel/` by calling `register_preseg()` explicitly — no automatic promotion.

## Migration from v1.3

Run `voxa/scripts/migrate_scan_v2.py` on the annotated root:

```bash
python voxa/scripts/migrate_scan_v2.py --dry-run /path/to/lidar
python voxa/scripts/migrate_scan_v2.py /path/to/lidar
```

Migration is one-shot, in-place, and idempotent (skips v2 scans). Each v1.3 scan:
- `labels/gt_*` → `sessions/legacy/output/`
- `session/working_*.npy` → `sessions/legacy/working_*.npy`
- `prelabel/ransac_*` → `prelabel/ransac/{instance_ids.npy, segment_summary.json, meta.json}`
- `annotation_history/*` → `sessions/legacy/history/`
- `meta.json` → `schema_version: "2.0"`

The script refuses loudly on unexpected contents and supports `--scan NAME ...` for partial migration.

## Changelog

- v2.0 (2026-06-03) — multi-session model (`sessions/<id>/`), formalized preseg store (`prelabel/<id>/`), session pinning. Removes `labels/`, `session/`, `annotation_history/`. Not backward-compatible; requires migration.
- v1.3 (2026-05-29) — explicit `frame` + `derivation` in `meta.json`; multi-run labels; render meta. Backward-compatible.
- v1.2 (2026-05-16) — additive: `renders/<run_id>/` + `sam3/`.
- v1.1 (2026-05-13) — additive: `session/` + fingerprints in `gt_segment_metadata.json`.
- v1 — initial schema.
