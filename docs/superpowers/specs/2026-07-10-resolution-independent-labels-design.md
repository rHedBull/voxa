# Resolution-independent labels (volumetric saving + materialize-at-any-density)

**Date:** 2026-07-10
**Status:** Draft — under spec review

## Problem

Labels in voxa are stored per-point, 1:1 with the cloud that was loaded.
`load_annotated` reads all of `source/scan.ply` (a ~3M downsample of a much
larger raw cloud) and the session's `working_class_ids.npy` /
`working_segment_ids.npy` are shape `(len(scan.ply),)`. So the label store is
anchored to exactly one fixed point set — `scan.ply`.

We want to consume labels at **arbitrary resolution** — coarser for fast
visualization, denser (up to the full raw cloud) for training. But a per-point
array is just indices into `scan.ply`; index `i` is meaningless against any
other cloud. Producing labels at a different density needs a *transfer* step,
and today there is none.

Two facts discovered during brainstorming shape the design:

1. **The raw full-density clouds exist for (almost) every scan.** They live in
   `<lidar_root>/raw/*.laz` (7.6M–156M points), registered in
   `<lidar_root>/raw/sources.json` with fingerprints, and each annotated scan's
   `meta.json::derivation.root.source_id` points back to its raw source.
   `scan.ply` is a recentered downsample of that raw (the recenter offset is
   recorded in the scan's `derivation` / `sample_param`).
2. **Reconstructing an exact `scan.ply`↔raw index map by re-running the
   downsample is not viable and not needed.** The scans use three different
   sample methods (`random` seeded, `uniform`, `voxel_down_sample`); 7 of 10 use
   voxel downsampling, whose output points are voxel *centroids* that never
   existed in the raw cloud — there is no point identity to match. Nearest-
   neighbor transfer is the robust, method-agnostic mechanism instead.

## Goal

Make label resolution a **materialization choice at consume/export time**, not a
labeling decision. `scan.ply` stays the cloud you label on (fast on ~3M).
Given any target cloud (a subsample of `scan.ply`, or the raw LAZ, or a
subsample of raw), produce per-point `class_ids` + `instance_ids` for it.

Per-point labels + instances remain the **final output** — this is about
*generating* them at a requested density, not replacing them with volumes.

### Scope (decided during brainstorming)

- **Annotated tier only.** Legacy scans have no raw lineage.
- **Volume-based sources rasterize exactly; preseg transfers by NN.** Box and
  Draw are pure geometric membership (see below), so storing their volume and
  re-rasterizing at any density is lossless. Presegment is an arbitrary point
  cluster with no primitive → nearest-neighbor transfer, which we accept.
- **`scan.ply`'s working array is the precedence oracle** — no apply-order
  metadata is stored or replayed (see §2).
- **Degrade gracefully when no raw is linked.** Materialization above
  `scan.ply` density is simply unavailable for that scan (as with the existing
  Edit "Full density" checkbox); down-sampling from `scan.ply` always works.
- **Out of scope:** a live "volumes are the source of truth while editing"
  rewrite of the apply pipeline. The working array stays the live store; volumes
  are stored *alongside* it and consumed at materialize time only.

## Background: why Box/Draw are lossless as volumes, preseg is not

- **Box** — `shapes.py::obb_indices` resolves "every point inside the OBB," no
  per-point refinement. At `scan.ply` resolution the box is *already* a blunt
  volume-membership label. Storing the OBB and rasterizing it at 30M points is
  the identical semantics with more points — nothing to lose.
- **Draw (pipe/tank)** — `centerlines.json` already persists, per instance,
  `{class_id, instance_id, points:[[x,y,z]…], radius, smooth}`: a swept-sphere
  (capsule chain). Point-in-tube is a distance test; exact at any density.
- **Presegment** — `kind:"pointset", segId:<n>` rows are arbitrary clusters from
  a precomputed segmentation. No box/tube captures them without loss → NN only.

Box is the one tool that *could* be volumetric but currently discards its OBB on
apply (CLAUDE.md: "the box vanishes and its enclosed points become a pointset").
This design **re-persists the box OBB** — as the selection volume that *is* the
label, not as a display cuboid with a gizmo. The `Cuboid` schema still carries
`center/size/rotation` (currently `null` on pointsets), so storage is
low-friction. This is a conscious, narrow reversal of the "cuboids retired"
decision, limited to persisting the selection volume.

## Design

### 1. Storage — an optional `shape` on each instance

Extend the instance doc (`instances_gt.json`, `Cuboid`/instance schema in
`app/schemas.py`) so a volumetric instance carries a resolution-independent
primitive:

- **Box** → `shape:{type:"obb", center:[3], size:[3], rotation:[3]}` (reuse the
  existing `center/size/rotation` fields; the box apply stops discarding them).
- **Draw** → `shape:{type:"tube", …}` — already persisted in
  `centerlines.json`, keyed by `instance_id`. Keep that file as the pipe editing
  store; the materializer reads it. (No duplication: `centerlines.json` is the
  tube's home; `instances_gt.json` need only mark the instance `source:"draw"`.)
- **Presegment / any per-point source** → no `shape`. Materialized via NN.

Shapes are stored in the **display/recentered frame** (the frame the user drew
them in), consistent with `centerlines.json` today.

### 2. Precedence oracle — the `scan.ply` working array

By the time `working_*.npy` is saved, every apply has run in order and last-wins
is baked in: each `scan.ply` point has exactly one `(class, instance)`. So the
working array **already encodes resolved precedence**. We never store or replay
apply-order — we ask the oracle. This removes the only piece of ordering
metadata the naive design would have needed.

### 3. Materialize(target_cloud) → (class_ids, instance_ids)

Inputs: the active session's working arrays + `scan.ply` positions (oracle), the
instance shapes, and a `target_cloud` (N×3, in `scan.ply`'s display frame — see
§4 for frame alignment). Steps:

1. **NN baseline.** `cKDTree(scan.ply)` → for each target point, its nearest
   `scan.ply` point resolves `inst_nn` (and its class). One query; this is the
   layer that carries preseg + overall precedence.
2. **Volume refinement.** For each target point `p` with nearest-instance
   `inst_nn`, apply this deterministic rule (a volume only sharpens its *own*
   instance's extent; it never steals a point the oracle gave to another labeled
   instance):

   - `inst_nn` is a **volume** (box/tube) `i`:
     - `p` inside `i`'s shape → `i` *(confirm)*
     - `p` outside `i`'s shape → the other volume that contains `p` if any, else
       background *(exclusion-sharpen: kill NN leak past the true box edge)*
   - `inst_nn` is **unlabeled (−1)** but `p` is inside some volume `j` → `j`
     *(inclusion-sharpen: claim dense points the sparse `scan.ply` missed inside
     the box)*
   - `inst_nn` is a **preseg** instance → keep it *(oracle's identity stands)*

3. Emit `class_ids` (int8) + `instance_ids` (int32) for the target cloud.

**The one policy choice — locked to defer-to-`scan.ply`.** A point geometrically
inside a box whose oracle neighbor is a *preseg* instance (e.g. a tube crossing a
preseg wall) keeps the preseg label. Faithful to "what it actually was at label
resolution." The alternative ("volumes always beat preseg") is rejected — it
could override a deliberate later apply.

Complexity: one KD-tree build + query over `scan.ply` (~3M), plus O(N · shapes)
point-in-shape tests; shapes are few and each test is cheap (OBB inverse-
transform + bounds; tube segment-distance). Materialization is deterministic and
cacheable as one `int32` NN-index array per (scan, target variant).

### 4. Raw resolution + frame alignment (prerequisite)

To materialize *above* `scan.ply` density we need the raw cloud path and the
transform that maps a shape (display frame) into the raw's frame.

- **Raw path.** `scene_registry._discover_annotated` currently resolves the raw
  only from `meta.source_laz`, which is `null` on regenerated scans. Extend it to
  fall back to the scan-schema registry: `meta.derivation.root.source_id` →
  `raw/sources.json` → `raw/<file>.laz`. Prefer the existing package
  (`scan_schema.Registry`) over re-parsing JSON. Result still surfaced as
  `SceneSource.extras["source_laz_path"]`, so downstream code is unchanged.
- **Frame alignment.** Shapes live in the recentered display frame; the raw LAZ
  is in the source surveying frame (Z-up + UTM). Reuse the transforms the
  Edit-mode full-density export already relies on
  (`2026-06-08-edit-raw-full-density-export-design.md`):
  `_to_display_frame(xyz, scene_is_z_up, offset)` streams raw points into the
  display frame before the point-in-shape / NN test; the recenter `offset` is
  the constant shift recorded for the scan. No new transform math.
- **Streaming.** Reuse the chunked LAZ reader (`_stream_laz_keep` /
  `load_laz`-style chunk iteration) so a 150M raw never loads whole into memory;
  materialize per chunk, write incrementally.

### 5. API / consume surface

A single export entrypoint, mirroring the existing Edit full-density export:

`POST /api/labels/materialize` (name TBD in plan) with:
- `resolution`: `"scan"` (identity — return the working array as-is),
  `"raw"` (full raw LAZ), or a target point count `N` (subsample of raw when
  `N > len(scan.ply)`, else subsample of `scan.ply`).
- Writes a labeled cloud (PLY/NPY) to the session's `output/` as a **derived
  product** — the canonical `scan.ply` working array is never overwritten.
- `raw_source_available` (already on `LoadResponse` from the Edit-export work,
  once §4 broadens its resolution) gates the denser options in the UI.

Down-sampling (`N ≤ len(scan.ply)`) needs no raw and no NN: subsample `scan.ply`
and index the working array (`labels[idx]`) — reuse `_safe_subsample`.

## Testing

- **Materialize unit tests** (`backend/tests/`):
  - Identity: `materialize(scan.ply positions)` reproduces the working array
    exactly.
  - Down-sample: `N < len(scan.ply)` equals `labels[subsample_idx]`.
  - OBB exclusion/inclusion sharpen: synthetic cloud + one OBB instance; assert
    points just outside the box are dropped and dense points inside a sparsely-
    sampled box are claimed.
  - Tube rasterize: a capsule from `centerlines.json` labels exactly the points
    within `radius`.
  - Precedence: overlapping box + preseg region → shared points keep the
    oracle's instance (defer-to-`scan.ply`).
- **Raw resolution** (`backend/tests/`): an annotated scan with `source_laz:
  null` but a `derivation.root.source_id` present resolves `source_laz_path` via
  `sources.json`; a scan with neither resolves to `None` (denser materialize
  disabled, not an error).
- **Frame alignment**: reuse existing LAZ fixtures; assert a shape rasterized
  onto raw-streamed-into-display-frame selects the analogous region it selects on
  `scan.ply`.
- **Browser** (per project rule): materialize an annotated scan at raw density,
  load the result in Inspect, confirm box/tube edges are crisp and preseg regions
  are contiguous; zero console errors.

## Out of scope / follow-ups

- Live "volumes as source of truth during editing" (re-rasterize on every apply).
  Deferred; the working array stays the live store.
- Freehand/lasso point cleanup as a tool — would be another per-point (NN)
  source; none exists today.
- Caching the NN-index map on disk per (scan, target) — start by computing on
  demand; add caching only if materialize latency warrants it.
- `bim_industrial_mep_matterport` and any scan whose raw is genuinely absent:
  materialize is capped at `scan.ply` density until/unless the raw is restored.
