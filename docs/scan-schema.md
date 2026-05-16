# Scan directory schema (voxa-facing)

What voxa expects on disk for an annotated scan. The canonical, broader schema lives at `lidar/SCHEMA.md` in the data tree; this doc captures the subset voxa actually reads + writes so the layout is discoverable from inside the repo.

## Where scans live

```
$VOXA_LIDAR_ROOT/annotated/<scan_name>/
```

`$VOXA_LIDAR_ROOT` defaults to `/home/hendrik/coding/engine/data/lidar`. Override via env. Every directory under `annotated/` that contains a `source/scan.ply` is a scene.

Two additional tiers — `decimated/` (raw PLY previews under `<lidar_root>/ply_viewer/*.ply`) and `raw/` (LAZ previews under `<lidar_root>/laz/*.laz`) — are surfaced by the scene picker as discovery shortcuts for unlabeled data. Neither has the per-scan substructure below; to label one, scaffold it into `annotated/` via `data/tools/scaffold_annotation.py`.

## Per-scan layout (`annotated/<scan_name>/`)

```
<scan_name>/
├── meta.json                       (required)  provenance + sampling params
├── README.md                       (recommended)
├── source/
│   ├── scan.ply                    (required)  point cloud being labeled (xyz + rgb)
│   └── mesh.glb                    (optional)  textured mesh for the companion window
├── labels/
│   ├── gt_class_ids.npy            (required)  int32, shape (N_pts,), -1 = unlabeled
│   ├── gt_segment_ids.npy          (required)  int32, shape (N_pts,), -1 = unlabeled
│   └── gt_segment_metadata.json    (required)  per-segment metadata + class_map_version
├── prelabel/                       (optional)  auto-suggestions surfaced when labels/ is empty
│   ├── ransac_instance_ids.npy            int32, shape (N_pts,)
│   └── ransac_segment_summary.json        { "segments": [{"id": int, "class_id": int}, ...] }
├── session/                        (optional)  autosave / page-reload recovery (voxa writes this)
│   ├── current.json                       commit pointer + flags (dirty, is_from_prelabel, fingerprints)
│   ├── working_class_ids.npy              int8, shape (N_pts,)
│   └── working_segment_ids.npy            int32, shape (N_pts,)
├── renders/                        (optional)  per-view captures of the scan (for SAM3 etc.)
│   └── <run_id>/                          one render pass (orbit, route, ...)
│       ├── manifest.json                  poses + intrinsics per frame
│       └── frame_NNN_*.png                color renders
├── sam3/                           (optional)  SAM3 feature cache + offline-pipeline outputs
│   ├── sam3_features.npz                  per-point image-encoder features (cache; render-set keyed)
│   └── <run_id>/                          per-render-set outputs (instance_ids.npy, info.json, ...)
└── annotation_history/             (optional)  timestamped backups voxa writes on save
    └── <YYYYMMDD_HHMMSS>/...
```

### What voxa REQUIRES vs what's nice-to-have

| Path | Voxa behavior if missing |
|---|---|
| `source/scan.ply` | scene not discovered |
| `meta.json` | scene listed but `n_points` shown as null |
| `labels/gt_class_ids.npy` + `labels/gt_segment_ids.npy` | falls through to `prelabel/`; if also missing, empty canvas |
| `labels/gt_segment_metadata.json` | save endpoint refuses (invariant 6: class_map_version mismatch detection) |
| `source/mesh.glb` | "▦ Mesh window" button greyed out |
| `prelabel/ransac_*` | no prelabel — labels/ is the only seed |
| `session/*` | clean load from labels/ (no in-progress recovery) |
| `renders/<run>/manifest.json` | `/api/sam3/renders` returns empty; no SAM3 in-app preseg available for this scene |
| `sam3/*` | recomputed on next SAM3 run (cache miss) |

### File-level contracts

- **`source/scan.ply`** — binary little-endian PLY, properties `x y z` (float32) and `red green blue` (uchar). `N_pts = len(vertices)`; every per-point array in `labels/` and `prelabel/` MUST have shape `(N_pts,)`.
- **`source/mesh.glb`** — single canonical filename. Build pipelines that emit variants (`mesh.optimized.glb`, `mesh.r05.glb`, etc.) MUST rename or hardlink one to `mesh.glb` to surface it. The viewer registers `MeshoptDecoder` so `EXT_meshopt_compression` GLBs load.
- **`labels/gt_class_ids.npy`** — `int32`, shape `(N_pts,)`. `-1` = unlabeled. Values must be IDs from `<lidar_root>/classes.json`.
- **`labels/gt_segment_ids.npy`** — `int32`, shape `(N_pts,)`. `-1` = unlabeled. IDs numbered per-scan; not globally unique.
- **`labels/gt_segment_metadata.json`** — `{ "n_points", "n_gt_segments", "n_labeled_points", "class_map_version", "segments": [...] }`. The save endpoint writes `prelabel_fingerprint` and `source_fingerprint` here so stale-prelabel detection on next load works.
- **`prelabel/ransac_instance_ids.npy`** — `int32`, shape `(N_pts,)`. Read with `prelabel/ransac_segment_summary.json` which maps each `id` to a `class_id` (use `-1` for class-agnostic preseg).
- **`renders/<run>/manifest.json`** — `{ "scene", "frames": [{ "file", "position": [x,y,z], "target": [x,y,z], ... }, ...] }`. SAM3 feature extraction reads only `file`/`position`/`target` (or `yaw` if `target` is absent).

### Invariants the save endpoint enforces

1. **Invariant 3**: `class_id == -1 ⟺ instance_id == -1` (per point, in the `labels/` file). Preseg-only points (instance assigned, class unassigned) are dropped to `-1` in the export — but voxa's in-memory canvas + `session/working_*.npy` keep the preseg structure so reload restores it.
2. **Invariant 4**: per-segment class consistency — every point sharing an `instance_id` must share the same `class_id`.
3. **Invariant 6**: `meta.json::class_map_version` must equal `classes.json::version`.

### Environment

- `VOXA_LIDAR_ROOT` — root of the canonical lidar archive (default `/home/hendrik/coding/engine/data/lidar`).
- `VOXA_DATA_DIR` — voxa-local data dir for legacy `data/scenes/*` + `data/annotations/*` (cuboid GT/predictions for the old workflow).
- `VOXA_RENDERS_ROOT` — **legacy fallback** for SAM3 render discovery (`<root>/<scene>/<run>/manifest.json`). Only consulted when the active scene doesn't have a `renders/` subdir of its own. Prefer the per-scan layout for new scenes.
- `VOXA_CONFIG` — path to `classes.yaml` (default `config/classes.yaml`).
- `VOXA_MAX_POINTS` — viewer subsample cap (default `1000000`). Labels operate on the full N_pts.

## Adding a new scan

```bash
cd /home/hendrik/coding/engine/data
python3 tools/scaffold_annotation.py path/to/source.laz --out-root lidar/annotated --target 3000000
```

Writes the SCHEMA-conformant `source/`, `labels/` (all -1), `prelabel/`, `meta.json`, `README.md`. Add `renders/` and a `prelabel/ransac_*` from your offline pipeline before opening voxa.
