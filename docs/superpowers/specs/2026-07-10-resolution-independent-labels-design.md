# Resolution-independent labels (volumetric saving + materialize-at-any-density)

**Date:** 2026-07-10
**Status:** Approved (spec-review passed, 3 iterations). Phase 1 SHIPPED to main.
Phase 2 **materialize core (§3–§4) IMPLEMENTED** in
`backend/labeling/materialize.py` (see
`docs/superpowers/plans/2026-07-10-materialize-core-phase-a.md`); the export
endpoint (§5) is delivered by the export-wizard spec's Phase B, still pending.

**Implementation split into two phases:**
- **Phase 1 — saving (SHIPPED to main).** Capture the resolution-independent
  primitives *at labeling time*, since they are otherwise lost: persist the Box
  OBB (`source=='box'`), keep the Draw tube (already persisted), and stamp an
  explicit monotonic `seq` apply-order on every instance (+ backfill for existing
  sessions). Small, additive, low-risk; no consumer yet.
- **Phase 2 — export / materialize.** The materialize algorithm (§3) + raw-cloud
  resolution + frame alignment (§4) are now **built** as the callable
  `materialize()` core (`backend/labeling/materialize.py`, regime A index /
  regime B replay, + `raw_source_available` on `LoadResponse`). The export
  endpoint (§5) is Phase B of the export-wizard spec, not yet built.

The rest of this document is the full design; Phase-2 sections are tagged
**[Phase 2 — deferred]**.

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

1. **The raw full-density clouds exist for most scans.** They live in
   `<lidar_root>/raw/*.laz` (7.6M–156M points), registered in
   `<lidar_root>/raw/sources.json` with fingerprints, and most annotated scans'
   `meta.json::derivation.root.source_id` points back to their raw source.
   `scan.ply` is a recentered downsample of that raw.
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
- **Volume-based sources rasterize exactly; preseg + legacy transfer by NN.**
  Box and Draw are pure geometric membership (see below), so storing their volume
  and re-rasterizing at any density is lossless. Presegment is an arbitrary point
  cluster with no primitive, and legacy cuboids are display-only → both go by
  nearest-neighbor transfer.
- **We must persist apply-order** (revised — see §2). The earlier assumption that
  the working array alone encodes recoverable precedence is false for dense
  points in volume-overlap regions; a per-instance order rank is required.
- **Degrade gracefully when no raw is linked.** Materialization above
  `scan.ply` density is simply unavailable for that scan; down-sampling from
  `scan.ply` always works.
- **Out of scope:** a live "volumes are the source of truth while editing"
  rewrite of the apply pipeline. The working array stays the live store; volumes
  + order are stored *alongside* it and consumed at materialize time only.

## Background: why Box/Draw are lossless as volumes, preseg is not

- **Box** — `shapes.py::obb_indices` resolves "every point inside the OBB," no
  per-point refinement. At `scan.ply` resolution the box is *already* a blunt
  volume-membership label. Storing the OBB and rasterizing it at 30M points is
  the identical semantics with more points — nothing to lose.
- **Draw (pipe/tank)** — `centerlines.json` already persists, per `instance_id`,
  one or more paths `{class_id, instance_id, points:[[x,y,z]…], radius, smooth}`
  (branches/merges mean one instance can own several paths —
  `centerline.py::update_centerlines` appends per instance, and `tube_indices`
  takes the path *list*). Point-in-tube is a distance test; exact at any density.
- **Presegment** — `kind:"pointset", segId:<n>` rows are arbitrary clusters from
  a precomputed segmentation. No box/tube captures them without loss → NN.
- **Legacy `kind:"cuboid"`** — display-only per CLAUDE.md; the box need not
  tightly bound its labeled points, so it must **not** be treated as a volume →
  NN.

The volume test is therefore keyed on **`source`**, not on the presence of
`center/size`: an instance is a volume iff `source=='box'` (OBB from
`center/size/rotation`) or `source=='draw'` (tube from `centerlines.json`).
Everything else — `preseg`, `manual`, `fit`, legacy `cuboid` — materializes by NN.

Box is the one tool that *could* be volumetric but currently discards its OBB on
apply (CLAUDE.md: "the box vanishes and its enclosed points become a pointset").
This design **re-persists the box OBB** — as the selection volume that *is* the
label, not as a display cuboid with a gizmo. The `Cuboid` schema already carries
`center/size/rotation` (`schemas.py:90-92`, currently `null` on pointsets), so
storage is low-friction. This is a conscious, narrow reversal of "cuboids
retired," limited to persisting the selection volume for `source=='box'`.

## Design

### 1. Storage — `source`-typed shape + apply-order on each instance

Extend the instance doc (`instances_gt.json`, `Cuboid` in `app/schemas.py`):

- **Box** (`source=='box'`): the box apply stops discarding the OBB — persist
  `center/size/rotation` on the resulting instance. `kind` may stay `pointset`
  (the live working array still holds its scan.ply points); the OBB is the
  resolution-independent shape read at materialize time.
- **Draw** (`source=='draw'`): tube paths stay in `centerlines.json`, keyed by
  `instance_id` (unchanged). The materializer unions all paths for the instance.
- **`seq`: an integer apply-order rank** on every instance (all sources),
  assigned from an **explicit monotonic counter at apply time**. This is the
  piece the naive design wrongly omitted. Do *not* derive `seq` from
  `instance_id`: merges (Draw `M`), preseg-promoted instances, and legacy
  instances loaded from disk can have non-monotonic or reused ids. Persist the
  counter in the session. **Backfill:** already-saved sessions have instances with
  no `seq`; on first load under this feature, assign `seq` by existing instance
  order (a one-time migration) so the materializer always has an ordering.

Shapes are stored in the **display/recentered frame** (the frame the user drew
them in), consistent with `centerlines.json` today.

### 2. Why apply-order is required (corrected premise)

The working array encodes last-wins *only at `scan.ply` sample locations*. For a
**dense** point in the interior of two overlapping boxes A (earlier) then B
(later), `scan.ply` baked A∩B → B, but its nearest `scan.ply` sample may be an
A-only point → NN alone would label it A. The overlap interior would be speckled
A/B by sampling density, contradicting the last-wins result the user produced.
`obb_indices` (`shapes.py`) is pure membership with no ordering, so the only
place order ever lived was the apply sequence. To reproduce last-wins for dense
points we must persist it. Hence `seq` (§1).

### 3. Materialize(target_cloud) → (class_ids, instance_ids)  [Phase 2 — deferred]

*Designed here to validate that Phase-1 storage (box OBB + `seq` + tube) is
sufficient to materialize at any density. Not implemented in Phase 1.*

Two regimes, chosen by target density:

**A. Target ≤ `scan.ply` density (down-sample) — exact, no NN, no shapes.**
Subsample `scan.ply` (reuse `_safe_subsample`) and index the working array:
`labels[idx]`. This trivially reproduces the working array (identity when
`idx == arange`). This path serves Inspect/Compare and any coarser export.

**B. Target denser than `scan.ply` (super-resolution) — ordered replay.**
Ground truth does not exist between `scan.ply` samples, so this regime is
*defined* by the stored primitives, not reconstructed from a denser original:

Mark each `scan.ply` point **volumetric-owned** iff its working instance has
`source∈{box,draw}`. Build one `cKDTree` over **all** `scan.ply` points. For each
target point `p`, collect candidates:

- **Volumetric candidates:** every volumetric instance `V` with `p ∈ shape(V)`
  contributes `(V, seq_V)` (exact). OBB: inverse-transform + half-extent test;
  tube: min segment-distance ≤ radius over all of the instance's paths.
- **Baseline candidate** from the nearest sample's owner `O`:
  - `O` non-volumetric ⇒ `(O, seq_O)`.
  - `O` volumetric **and** `p ∈ shape(O)` ⇒ `(O, seq_O)` — *interior defended*:
    the box competes at full strength for its own interior.
  - `O` volumetric **and** `p ∉ shape(O)` ⇒ the **leak** case: discard `O` and
    re-query the nearest **non-volumetric** point for the baseline candidate.
    (Answering "nearest non-volumetric point" needs a *second* KD-tree over the
    non-volumetric subset, or k-NN-and-filter on the all-points tree — build both
    structures once.)
  - A `−1` (background) nearest owner yields **no** baseline candidate — it never
    competes, so it can never out-rank a real instance.
- **Winner = the max-`seq` candidate.** No candidate ⇒ background (−1).

The point of building the tree over *all* points and gating the baseline (rather
than pre-excluding volumetric points) is that a box must be able to *defend* its
interior: excluding volumetric points would let a far, higher-`seq` preseg that
never covered `p` win the box's interior, re-creating a regime-A/B disagreement.
This rule resolves every case by order, not heuristic:
- *Two overlapping boxes, dense interior* (`seq_B>seq_A`): both A and B are
  geometric candidates regardless of nearest sample ⇒ B. ✔
- *Box interior, an adjacent preseg has higher `seq`:* the far preseg is **not**
  admitted (it is neither `p`'s covering volume nor its nearest owner); the
  nearest sample is V-owned and covers `p` ⇒ V. ✔ (matches regime A)
- *Point inside a box later reassigned by a preseg that actually covers it:* that
  preseg is `p`'s nearest owner with higher `seq` ⇒ preseg. ✔ (matches regime A)
- *Point outside a box, inside a surrounding preseg wall, nearest sample is a box
  edge:* leak case — `O=V` discarded, re-query nearest non-volumetric ⇒ wall. ✔
  (no spurious drop-to-background)

**Guarantees (stated honestly):** volumetric-instance boundaries are **exact** at
any density; non-volumetric (preseg/legacy) boundaries are **NN-approximate** to
~`scan.ply` sample spacing (a few cm) — no ground truth exists to do better;
precedence among all instances follows the persisted `seq`. Regime A is exact.

Complexity: two KD-trees over `scan.ply` (all points, and the non-volumetric
subset for leak re-queries) + one query per target point, plus O(target ·
#volumes) cheap point-in-shape tests. Deterministic; the target→(labels) result
is cacheable per (scan, target variant).

### 4. Raw resolution + frame alignment (prerequisite)  [Phase 2 — deferred]

To materialize *above* `scan.ply` density we need the raw cloud path and the
transform mapping a shape (display frame) into the raw's frame.

- **Raw path.** `scene_registry._discover_annotated` resolves the raw only from
  `meta.source_laz` today (`scene_registry.py:129`). That is `null` on **two 3.0
  Matterport scans** (`mechanical_room_matterport`, `parkhouse_matterport`) which
  *do* have a `derivation.root.source_id` resolvable in `raw/sources.json`, and on
  **three lineage-less 2.0 scans** (`water_pump_navvis_3m`,
  `water_pump_navvis_500k`, `bim_industrial_mep_matterport`) which have neither
  and are permanently capped at `scan.ply` density. Extend the resolver to fall
  back to `meta.derivation.root.source_id` → `scan_schema.Registry` →
  `lidar_root / entry.path` (`Registry` exposes `root(source_id).path`; prefer a
  roots-only / `root_by_basename` load over the full `Registry.load`, which also
  walks all of `annotated/`). Result still surfaced as
  `SceneSource.extras["source_laz_path"]`, so downstream is unchanged.
- **Frame alignment.** Shapes live in the recentered display frame; the raw LAZ
  is in the source surveying frame (Z-up + UTM). Reuse the transforms the
  Edit-mode full-density export relies on
  (`2026-06-08-edit-raw-full-density-export-design.md`): `_to_display_frame(xyz,
  scene_is_z_up, offset)` (`core.py:371`) streams raw points into the display
  frame before the point-in-shape / NN test; `_stream_laz_keep` (`core.py:408`)
  chunks the LAZ so a 150M cloud never loads whole. A membership/NN test in the
  display frame is frame-identical to the geometric cutout those helpers were
  built for. **The recenter `offset` is not stored on disk** — `_recenter`
  (`core.py:129`) recomputes `points.mean(axis=0)` at load and stashes it in
  `_state["recenter_offset"]`. It is deterministic from `scan.ply`, so reuse is
  valid, but materialize therefore inherits the single-in-memory-cloud coupling
  (CLAUDE.md gotcha): it requires a prior `/api/load` of the same scan to have
  populated `_state`, exactly like `auto-fit` and `edit-export`.

### 5. API / consume surface  [Phase 2 — deferred]

A single export entrypoint, mirroring the Edit full-density export:

`POST /api/labels/materialize` (name TBD in plan) with:
- `resolution`: `"scan"` (regime A identity — return the working array),
  a target point count `N` (regime A when `N ≤ len(scan.ply)`; regime B subsample
  of raw when `N > len(scan.ply)`), or `"raw"` (regime B, full raw LAZ).
- Writes a labeled cloud (PLY/NPY) to the session's `output/` as a **derived
  product** — the canonical `scan.ply` working array is never overwritten.
- Gated in the UI by `raw_source_available` (already on `LoadResponse` from the
  Edit-export work) once §4 broadens its resolution; regime-B options disabled
  when no raw is linked.

## Testing

### Phase 1 — saving (this change)

- Box apply **persists** `center/size/rotation` on the resulting instance
  (`source=='box'`); a round-trip save/load preserves the OBB.
- Every apply stamps a strictly increasing `seq`; `seq` is monotonic across mixed
  tool applies, merges, and preseg promotions (never reused/derived from
  `instance_id`).
- Backfill: loading a pre-feature session assigns `seq` by existing instance
  order, deterministically.
- Draw tube persistence unchanged (regression guard on `centerlines.json`).

### Phase 2 — export / materialize (deferred, with the export work)

- **Regime A** (`backend/tests/`):
  - Identity: `resolution="scan"` returns the working array byte-for-byte.
  - Down-sample: `N < len(scan.ply)` equals `labels[subsample_idx]`.
- **Regime B precedence — these must actually exercise the blocker/majors:**
  - **Two overlapping boxes at dense resolution**, `seq_B>seq_A`: assert the
    entire A∩B interior is B (not sampling-speckled). *(guards the blocker)*
  - **Box interior defended vs a higher-`seq` adjacent preseg**: preseg `P`
    (`seq_P > seq_V`) drawn *next to* box `V` (P does not cover V's interior);
    assert V's interior materializes to `V`, not `P`. *(guards the rev2 defect —
    fails under the "exclude volumetric from baseline" rule, passes under the
    interior-defense gate)*
  - **Reassigned point inside a box**: a preseg that *does* cover interior points
    of box `V` with `seq_P > seq_V` materializes those points to `P`. (Erase-to
    −1 inside a retained box is not modeled — `obb_indices` labels all points in
    the OBB on apply and there is no unlabel-while-retaining-volume op, so
    background carries `seq −∞` and can never out-rank the enclosing volume.)
  - **Box inside a preseg wall**: dense points outside the box but inside the wall
    materialize to the wall, never dropped to background.
  - **Legacy `kind:"cuboid"`**: materializes by NN, is never sharpened.
  - **Multi-path tube**: an instance with two `centerlines.json` paths labels the
    union of both capsules.
- **Raw resolution** (`backend/tests/`): a scan with `source_laz:null` but a
  `derivation.root.source_id` resolves `source_laz_path` via `sources.json`; a
  lineage-less scan resolves to `None` (regime B disabled, not an error).
- **Frame alignment**: reuse existing LAZ fixtures; a shape rasterized onto
  raw-streamed-into-display-frame selects the region analogous to `scan.ply`.
- **Browser** (per project rule): materialize an annotated scan at raw density,
  load the result in Inspect — box/tube edges crisp, preseg regions contiguous,
  zero console errors.

## Out of scope / follow-ups

- Live "volumes as source of truth during editing" (re-rasterize on every apply).
  Deferred; the working array stays the live store.
- Freehand/lasso point cleanup as a tool — another per-point (NN) source; none
  exists today.
- Caching the target→labels result on disk per (scan, target); compute on demand
  first, add caching only if latency warrants it.
- **Permanently `scan.ply`-capped scans** (no raw lineage): `water_pump_navvis_3m`,
  `water_pump_navvis_500k`, `bim_industrial_mep_matterport`. Regime B is
  unavailable for these until/unless a raw source is registered.
- **Sub-cm preseg boundary accuracy** is a future *labeling-resolution* effort
  (raise the `scan.ply` / `MAX_LABEL_POINTS` density at annotation time), not a
  materialization change. The NN-approximate preseg seam (~one labeling-cloud
  sample spacing, ~±1 cm at 3M) is **accepted** for this design; the target
  export resolution does not affect it. Volumetric (box/tube) boundaries are
  already exact and are the intended lever where crisp edges matter now.
