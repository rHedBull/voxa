# Lidar multi-root loading — Inspect-mode view-only slice

**Status**: design approved 2026-04-27
**Scope**: Voxa Inspect mode reads from the canonical lidar archive at `/home/hendrik/coding/engine/data/lidar/`. View-only — no labeling, no editing.

## Goal

Make every scan in `engine/data/lidar/` openable in Voxa's Inspect mode, with per-point class/instance coloring for already-annotated scans.

## Non-goals (this slice)

- Editing per-point labels (waits for the SAM3 hybrid plan).
- Segment list panel, click-to-isolate-segment, connectivity-graph rendering.
- Compare-mode integration of per-point predictions.
- LAZ load caching (`~/.cache/voxa/...`).
- `pending_lgsx/` (needs Cyclone/ReCap to decode upstream).

## Mode placement: where do labels live?

Per-point labels are *part of the scan*, not a separate artifact. Coloring points by class is just another view of the cloud — same shape as Color-by-Height. Inspect already owns "any read-only view of the scan with whatever properties the cloud carries", so the label-coloring lives there.

When per-point editing arrives, it goes into Label (which expands to handle per-point segments alongside cuboids). A heavyweight "review with segment list / connectivity graph" experience may justify a new mode at that point, but does not now.

## Architecture: multi-root scene discovery

A new env var `VOXA_LIDAR_ROOT` (default `/home/hendrik/coding/engine/data/lidar`) is walked alongside the existing `VOXA_DATA_DIR`. Four scene tiers:

| Tier        | Path pattern                                                        | Reader                  | Has labels       | Has intensity |
|-------------|---------------------------------------------------------------------|-------------------------|------------------|---------------|
| `legacy`    | `voxa/data/scenes/<n>/source.{ply,glb}`                             | existing PLY / GLB      | no               | no            |
| `annotated` | `<lidar>/annotated/<n>/source/scan.ply` + `labels/*.npy`, `meta.json` | new (wraps PLY)         | yes (when present) | no          |
| `decimated` | `<lidar>/ply_viewer/<n>.ply`                                         | existing PLY            | no               | no            |
| `raw`       | `<lidar>/laz/<n>.laz`                                                | new (`laspy[lazrs]`)     | no               | yes           |

**Scene IDs are tier-prefixed** to avoid collisions (e.g. `Factory-large` lives in both `decimated` and `raw`):
- `legacy/test_scene`
- `annotated/munich_water_pump`
- `decimated/Factory-large`
- `raw/Factory-large`

The existing flat scene names (e.g. `test_scene`) keep working as a backwards-compat alias for `legacy/<name>`. Existing tests stay green.

## Backend

### New module: `backend/scene_registry.py`

```python
@dataclass(frozen=True)
class SceneSource:
    tier: str                 # 'legacy' | 'annotated' | 'decimated' | 'raw'
    name: str
    scene_id: str             # f"{tier}/{name}"
    source_path: Path
    source_format: str        # 'ply' | 'glb' | 'laz'
    has_labels: bool
    has_intensity: bool
    n_points: int | None      # if known cheaply (annotated/meta.json or LAS header)
    extras: dict              # tier-specific: annotated/ has {labels_dir, meta_path, segment_meta_path}

def discover() -> list[SceneSource]: ...
def resolve(scene_id: str) -> SceneSource: ...   # also accepts a bare legacy name
```

Discovery walks `VOXA_DATA_DIR/scenes/*` (legacy) and the four `<lidar_root>/{annotated,ply_viewer,laz}` roots. Sorting: by tier (legacy first, then annotated, decimated, raw), then by name.

### New module: `backend/lidar_io.py`

```python
def load_annotated(src: SceneSource) -> tuple[PointCloud, LabelArrays | None, dict meta]
def load_laz(path: Path, max_points: int) -> PointCloud
```

`load_annotated`:
- Reads `source/scan.ply` via existing `load_ply`.
- If `labels/gt_class_ids.npy` and `labels/gt_segment_ids.npy` exist, returns a `LabelArrays(class_ids, instance_ids)` aligned to the cloud (length-matched).
- Reads `meta.json` (best-effort).
- Reads `labels/gt_segment_metadata.json` for the class palette (segment names + class-id assignment).
- Resolves class hex colors against `<lidar_root>/classes.json` if present, else falls back to a built-in palette.

`load_laz`:
- `laspy.open(path)` to read the header without loading points.
- Compute `stride = max(1, ceil(n_points / max_points))`.
- Iterate `chunk_iterator(chunk_size=1_000_000)`; for each chunk take indices `[off::stride]` where `off` carries over between chunks so the stride is uniform.
- Extract `x, y, z` (laspy applies scale + offset), `red, green, blue` (uint16, scaled to 0..255), `intensity` (uint16, normalized to 0..1 via the chunk-max or header bounds when available).
- Returns a `PointCloud` with `.intensity` extra (a thin extension to `PointCloud` to carry per-point float32 intensity, optional).

### Auto-recenter on load

LAS UTM coords reach ~7 digits, breaking float32 precision in Three.js. Every loaded `PointCloud` is bbox-centered before transport: subtract the centroid, store the offset. The offset goes into `LoadResponse.recenter_offset`. `auto-fit` continues to operate on the recentered cloud (the existing endpoint receives a recentered AABB from the frontend; no behavior change because cuboid math is local).

### `/api/load` response — new optional fields

```python
class LoadResponse(BaseModel):
    scene: str                  # tier-prefixed id
    num_points: int
    num_subsampled: int
    bbox_min: list[float]
    bbox_max: list[float]
    positions: str              # b64 Float32 (xyz, recentered)
    colors: str                 # b64 Float32 (rgb 0..1)
    intensity: str | None       # b64 Float32 (0..1) — only for `raw` tier
    class_ids: str | None       # b64 Int8        — only when annotated has labels
    instance_ids: str | None    # b64 Int32
    class_palette: list[ClassDef] | None
    n_classes: int | None
    n_instances: int | None
    recenter_offset: list[float]
```

`_state` retains the full pre-subsample `PointCloud` so `auto-fit` keeps working. Subsample indices are applied to label arrays before transport.

### `/api/scenes` — new shape

```python
class SceneInfo(BaseModel):
    id: str                  # tier-prefixed
    tier: str
    name: str
    source_format: str       # 'ply' | 'glb' | 'laz'
    has_labels: bool
    has_intensity: bool
    n_points: int | None
```

Sorted: legacy → annotated → decimated → raw, then by name.

### Dependencies

Add to `backend/requirements.txt`:
- `laspy>=2.5`
- `lazrs` (pure-Rust LAZ codec, pip-installable, no system deps)

## Frontend

### `frontend/src/api.js`

- `b64ToInt8(b64)` and `b64ToInt32(b64)` decoder helpers.
- `load(id, maxPoints)` — pass-through of new optional fields; `id` already supports tier-prefixed strings.
- `scenes()` returns the new shape.

### `frontend/src/viewer.jsx`

Color-mode branches grow:
- `class` — per-point lookup `classPalette[class_ids[i]]`; `-1` → muted grey.
- `instance` — golden-ratio hue hash `hsl(((id * 137.508) % 360), 65%, 60%)`; `-1` → muted grey.
- `intensity` — currently a height-based placeholder; switch to using real `intensity` Float32Array when present.

### `frontend/src/mode-inspect.jsx`

Pills `RGB / Height / Intensity / Flat` grow to `RGB / Height / Intensity / Class / Instance / Flat`. Pills are `disabled` when the cloud lacks that channel; disabled state is styled grey with a tooltip ("no labels in this scene"). The Scene panel adds a label-summary line when applicable: `126 segments · 99,940 / 500K labeled`.

### `frontend/src/App.jsx` `ScenePicker`

Group scenes by tier with a small section header per tier. The existing flat list is preserved when only `legacy` scenes exist (zero-friction for anyone without the lidar archive).

## Performance

- `annotated/<scan>/source/scan.ply` (500K pts) — fast, no change.
- `decimated/<scan>.ply` (~5M pts) — read fully, subsample to 300K. Existing PLY reader handles it.
- `raw/<scan>.laz`:
  - SMART-AIS, 188M pts → stride ≈ 627 → ~30–60s cold-disk first load.
  - Smaller scans 30–40M pts → ~5–10s.
  - Subsequent loads hit OS page cache.
  - No caching layer this slice; documented as a known cost.

## Backward compatibility

- Existing `voxa/data/scenes/<name>/` continues to work as the `legacy` tier.
- `/api/scenes` returning the new shape is a breaking change for any external consumer, but the only consumer is `frontend/src/api.js`, updated in this slice.
- `/api/load` accepts both the new `tier/name` IDs and bare legacy names.
- Existing tests use legacy scenes; they stay green without changes.

## Testing

- `backend/tests/test_scene_registry.py` — discovery from a tmp dir mirroring all four tiers; resolve() with both legacy bare names and tier-prefixed IDs; collision handling.
- `backend/tests/test_lidar_io.py` — load_annotated reads a tiny synthesized PLY + label .npy fixtures; load_laz stride math via a tiny LAS file (or mocked chunk iterator).
- `backend/tests/test_load_endpoint.py` — `/api/load` returns label fields when labels present, omits them otherwise; recenter offset round-trips.
- Frontend (vitest) — `b64ToInt8` / `b64ToInt32` round-trip a known buffer; `scenes()` parser tolerates both shapes.

## Known follow-ups (out of scope)

- LAZ load caching to disk-backed decimations.
- Per-scene class config picked up from `lidar/classes.json`'s `classes` array dynamically (currently the palette is built from the segment metadata + a fallback hex set; a future slice should resolve full class colors from `lidar/classes.json` once that file gains color fields).
- Segment list panel and click-to-isolate-segment in a forthcoming Review/Label expansion.
- LAS coordinate-system handling beyond auto-recentering (CRS, units other than meters).
