# Region density gate: measure against the raw source, not the session cloud

Date: 2026-07-23

## Problem

Two consumers measure point-cloud density via `labeling.materialize.raw_sample_spacing`:

- **The eval-grade region gate** (`labeling/regions.py::flip_status`) refuses to flip a
  region to `eval_grade` if p90 nearest-neighbour spacing over the region's points exceeds
  10mm.
- **The export wizard's accuracy readout** (`GET /api/labels/accuracy`) surfaces the same
  p50/p90 spacing as a boundary-uncertainty line in the Review step.

Both currently measure against `SegmentSession.positions` / `_state["pc"].points` — the
**session's working cloud**, which is capped at `VOXA_MAX_POINTS` (today 3M for most
sessions). This makes eval-grade eligibility and the accuracy readout a function of
*viewer/session subsampling*, not the scan's actual density.

This is unnecessary. Volume-defined labels (Box/Prism/Beam/Draw) already replay exactly at
native raw resolution via `materialize.py`'s regime-B path — the labeling geometry is
resolution-independent. Only the *density gate itself* is anchored to the low-resolution
session cloud.

Measured consequence (2026-07-22, see `docs/superpowers/specs/` density notes): two real
scans fail the 10mm eval-grade bar today purely because of this — `bim_industrial_mep_
matterport` (3M session cloud, p90 31.9mm) and `water_treatment_navvis` (3M session cloud,
p90 17.3mm) — despite their raw sources measuring far below the bar (bim's E57 ≈9.0mm
extrapolated at full 37.8M density; water_treatment's LAZ 5.2–5.8mm at 38.6M). Raising the
session cloud's point cap ("phase 0b") was the originally proposed fix, but it only exists
to work around this gate — it does not change what's actually being labeled, and it costs
viewer FPS. This spec removes the need for it.

## Approach

Add two new helpers to `backend/labeling/materialize.py`, both sourced from the scan's
registered raw file rather than the working cloud:

### 1. `raw_region_sample_spacing(raw_path, prism, scene_is_z_up, offset)`

Takes the region's runtime prism (not a bare AABB) so it can filter exactly: computes an
AABB from the prism's own corners (`min`/`max` of the footprint polygon's x/z plus
`y0`/`y0+height` — no such helper exists today anywhere in `shapes.py`/`regions.py`/
`materialize.py`; this is a new ~3-line function, not a reuse of existing bounding logic),
calls the existing `scenes.lidar_io.load_laz_region()` with that AABB — already streams a
LAZ chunk-by-chunk (2M pts/chunk), filters to an AABB in the display (recentered) frame, and
returns only the in-region points at full native density, without loading the whole raw
cloud — then **re-filters the (small) AABB-filtered result through the exact prism** via the
already-imported `prism_indices` (`materialize.py:11`), before feeding the result into the
existing `raw_sample_spacing()` for the p50/p90 calculation (which already internally
subsamples to `sample=100_000` points, so no extra cap is needed here even for a
densely-populated region).

The exact-prism re-filter matters for non-rectangular or rotated footprints: an AABB-only
filter can pull in extra points from just outside the actual region (e.g. an adjacent wall
or surface at different density), which would skew the measured p50/p90 in either
direction. Re-filtering is nearly free here since it only runs over the already-small
AABB-filtered set, not the raw file.

Used by the eval-grade gate, scoped to the region's own prism — cheap because it only
returns points local to one region, not the whole scan.

### 2. `raw_reservoir_sample_spacing(raw_path, scene_is_z_up, offset, n_chunks=5, chunk=1_000_000, seed=0)`

**Correction from spec review round 2:** an earlier version of this design reservoir-sampled
individual *points* down to a flat 100k-point pool and fed that straight into
`raw_sample_spacing`. That's wrong — `raw_sample_spacing` builds its KD-tree over its
*entire* input array (`materialize.py:298`, `cKDTree(scan_pos)`) and only subsamples which
points to *query*; it does not know or care that its input is itself a subsample. Feeding it
a reservoir of individual points scattered across a ~156M-point file collapses local density
by the same factor as the reservoir ratio (e.g. ~11x sparser at a 100k reservoir over 156M
points) — the KD-tree's nearest-neighbour distances would reflect the *reservoir's* spacing,
not the scan's, and could easily report a **worse** number than today's session-cloud
measurement (which is a real, spatially-dense 3M-point cloud, not a scattered sample). That
would silently defeat this spec's entire goal for this endpoint.

**Fix: reservoir-sample whole *chunks*, not individual points**, following the same "local
chunks" methodology already used (and validated against ground truth) for the density
figures in the phase-0b memory notes. Stream the raw file via `scenes.lidar_io.
_laz_chunk_iter` as before, but instead of pulling points into a flat pool, run Algorithm R
over the **stream of chunks** to pick `n_chunks` (default 5) chunk indices uniformly at
random across the file — each retained chunk keeps its full native point density, on the
assumption that a `_laz_chunk_iter` chunk (sequential points in on-disk order) is spatially
coherent, which holds for typical LiDAR/NavVis scan-order exports; if a registered raw file
turns out to be pre-shuffled or globally tiled such that a "chunk" spans the whole scan
volume, this degrades to approximating whole-scan spacing from `n_chunks` scattered
mini-samples rather than true local density — acceptable for this informational endpoint,
but worth a code comment flagging the assumption. Concatenate the retained
chunks' positions (default: 5 × 1M = up to 5M points, still far below holding the full
~156M-point file, and each chunk internally is at true raw density) and feed that into the
existing `raw_sample_spacing()`, which then does its own internal `sample=100_000` query
subsample against this dense, spatially-real corpus — exactly as it already does for
`raw_region_sample_spacing`'s AABB-filtered result. This preserves true local density in
every point used for the KD-tree, at the cost of only sampling `n_chunks` spatial neighborhoods
of the scan rather than every neighborhood — an acceptable approximation for an
informational (non-gating) endpoint, and consistent with how p90 density was already
estimated by hand for bim/water_treatment.

Used by the export accuracy endpoint (and `export_labels`'s manifest stamping, see below),
which are scan-wide rather than region-scoped, so there's no AABB to filter by — a full-file
streaming pass is unavoidable if we want any raw measurement here, and chunk-level
reservoir sampling keeps it statistically representative of the scan's actual density rather
than of wherever the LAZ happens to be tiled/sorted on disk.

Both new helpers are thin: they each resolve a `positions` array (AABB+prism-filtered for
region 1, concatenated sampled chunks for region 2) and then call the existing
`raw_sample_spacing(positions)` unchanged — no refactor of `raw_sample_spacing` itself is
needed; its current per-call KD-tree-over-input behavior is exactly what both new callers
want.

## Behavior changes

### `labeling/regions.py::flip_status`

Gains `raw_path: str | None` and `scene_is_z_up: bool` parameters (positional/keyword,
inserted alongside the existing `positions` param). `positions` stays required, but its role
narrows to the `categories`/review-budget check only — that check has no raw-scoped
equivalent, since point categories are a session-resolution concept. The point-floor and
spacing measurement both move to the raw source.

- If `raw_path` is not `None`: **first** check the point floor against the raw region's own
  point count (a new `raw_region_point_count(raw_path, prism, scene_is_z_up, offset)`
  helper, sharing `raw_region_sample_spacing`'s AABB+prism filtering) — this must run
  *before* the spacing measurement, since an empty raw region measures spacing `(0.0, 0.0)`,
  which must not be misread as "coincident points." Only once the floor passes, measure
  spacing via `raw_region_sample_spacing(raw_path, region's runtime prism, scene_is_z_up,
  offset)`.
- If `raw_path` is `None` (no raw source registered for this scan — e.g. bim today):
  **refuse eval-grade outright** with `RegionError("no raw source registered for this scan
  — cannot verify true density; register a raw source before flipping regions to eval-
  grade")`. Do NOT fall back to measuring `positions` — that reintroduces exactly the
  ambiguity this spec removes, and would let regions silently pass at whatever density the
  session cloud happens to be.
- The near-zero-spacing (`MIN_GATE_P50_M`) check applies to the raw-measured spacing, same as
  the p90 bar. The `categories`/review-budget check alone stays scoped to `positions` (the
  session's own working array), since it has no raw-scoped equivalent.

### `backend/routes/regions.py`

`_ctx()` (or a call site in `patch_region`) resolves the raw source the same way
`load.py`/`export.py` already do:

```python
src = _resolve(_state.get("scene"))
raw_path = src.extras.get("source_laz_path")
scene_is_z_up = _scene_is_z_up(src)
```

and threads `raw_path`, `scene_is_z_up` into `flip_status(...)` alongside the existing
`positions`/`offset`/`categories` args.

### `GET /api/labels/accuracy` (`backend/routes/export.py`)

When `src.extras.get("source_laz_path")` is present, measure via
`raw_reservoir_sample_spacing` instead of `raw_sample_spacing(pc.points)`. When absent,
keep today's `pc.points`-based measurement as a best-effort fallback — this endpoint is
informational (the Review step's boundary-uncertainty line), not a hard gate, so degrading
gracefully rather than refusing is appropriate. Response gains an `is_raw: bool` field so
the frontend can visually distinguish an approximate (session-cloud) reading from a raw-
backed one — exact copy/UI treatment left to whoever implements the frontend side (not
speced further here; likely a small caption change, e.g. "measured against raw source" vs
"measured against loaded cloud (approximate)").

### `POST /api/labels/export` (`backend/routes/export.py::export_labels`)

There is a **third** call site of `raw_sample_spacing`, at `routes/export.py:214`, inside
`export_labels`: it stamps the export manifest's `accuracy` field with
`raw_sample_spacing(ctx.scan_pos)` — the session cloud — **even when the export itself is at
`resolution.kind == "raw"`** (i.e. `ctx.raw_path` is populated and the export is already
streaming/writing raw-density points). This is the same problem as `/api/labels/accuracy`,
for arguably the most important consumer: a raw-resolution export's own manifest currently
under-reports its own accuracy. Fix: when `resolution.kind == "raw"`, measure via
`raw_reservoir_sample_spacing(ctx.raw_path, ...)` (or, more precisely, an in-line reservoir
sample taken from the same `materialize_raw` chunks already being streamed to disk for this
export — avoiding a second full-file pass — if that's a cheap fold into the existing
streaming loop; otherwise a second `raw_reservoir_sample_spacing` pass is an acceptable v1).
For `scan`/`subsample` resolution kinds, keep measuring `ctx.scan_pos` as today — the
manifest should describe the accuracy of *what was actually exported*, not always the raw
ceiling.

## Data flow

```
Eval-grade flip (PATCH /api/regions/{rid} {status:"eval_grade"})
  routes/regions.py::patch_region
    -> _resolve(scene) -> raw_path, scene_is_z_up
    -> regstore.flip_status(doc, rid, "eval_grade", positions, offset,
                             categories=seg.categories,
                             raw_path=raw_path, scene_is_z_up=scene_is_z_up)
         if raw_path:
           aabb = bounds of region's runtime prism
           raw_region_sample_spacing(raw_path, aabb_min, aabb_max, scene_is_z_up, offset)
             -> load_laz_region(...) streams+filters raw LAZ to AABB
             -> raw_sample_spacing(...) -> p50, p90
         else:
           raise RegionError (refuse)

Export accuracy (GET /api/labels/accuracy)
  routes/export.py::labels_accuracy
    if src.extras["source_laz_path"]:
      raw_reservoir_sample_spacing(raw_path, scene_is_z_up, offset)
        -> stream whole LAZ via _laz_chunk_iter, reservoir-sample whole CHUNKS (not points)
        -> concatenate retained chunks -> raw_sample_spacing(...) -> p50, p90
      return {..., is_raw: true}
    else:
      raw_sample_spacing(pc.points) -> p50, p90   # today's behavior, unchanged
      return {..., is_raw: false}
```

## Error handling

- `raw_region_sample_spacing`/`raw_reservoir_sample_spacing` propagate any `laspy` read
  failure (corrupt/missing file) as a loud exception — no silent empty-result fallback. A
  scan whose registered raw path has gone stale should fail the gate/accuracy call visibly,
  not silently degrade to the session cloud.
- `flip_status`'s no-raw refusal is a `RegionError` → the existing `HTTPException(422, ...)`
  path in `routes/regions.py`, consistent with every other gate refusal (point floor, p90
  bar, review budget).
- Both new raw-measurement code paths must reuse `load_laz_region`'s existing
  frame-transform (`is_z_up` + `offset`) so the AABB filter and the eventual returned
  positions are in the same frame the caller's prism/region already is — a frame mismatch
  here would silently return zero or wrong points rather than erroring, so this is worth an
  explicit test (see below).

## Testing

- `materialize.py` unit tests:
  - `raw_region_sample_spacing` on a small synthetic LAZ: verify only in-AABB points are
    used (cross-check against `prism_indices`/manual masking) and spacing matches a direct
    KD-tree computation on the same filtered set.
  - `raw_reservoir_sample_spacing`: verify the number of *chunks* retained is bounded
    (`n_chunks`) regardless of input file size, that each retained chunk preserves its full
    native point density (no per-point thinning), and — the key regression test for the
    round-2 review bug — that measuring a uniformly-dense synthetic raw file via this
    function returns spacing close to the true known spacing (not inflated by ~the
    reservoir-ratio factor a flat point-reservoir would have introduced).
- `regions.py` tests:
  - Eval-grade flip **passes** for a region whose session-cloud (`positions`) spacing would
    fail the 10mm bar but whose raw-source spacing passes (the exact bim/water_treatment
    scenario) — construct a synthetic session cloud (sparse) + synthetic raw LAZ (dense,
    same geometry) fixture.
  - Eval-grade flip **refuses** with `RegionError` when `raw_path=None`, regardless of how
    dense `positions` is.
  - Point-floor and review-budget checks still fire correctly when sourced from raw region
    points.
- `export.py` test: `GET /api/labels/accuracy` returns `is_raw: true` + raw-backed numbers
  when a raw source is registered, `is_raw: false` + today's numbers when not.
- `export.py` test: `POST /api/labels/export` with `resolution.kind == "raw"` stamps the
  manifest's `accuracy` with raw-backed spacing, not session-cloud spacing; `scan`/
  `subsample` kinds keep measuring `ctx.scan_pos` as today.
- **Existing test migration**: `flip_status`'s signature change breaks every existing
  positional call site in `backend/tests/test_regions.py` (at minimum the calls around
  lines 87, 118, 133, 144, 155 as of this writing) — they call
  `flip_status(doc, id, "eval_grade", positions)` without `raw_path`/`scene_is_z_up`. All
  of these need updating to pass an explicit `raw_path` (a synthetic LAZ fixture, or
  `None` for tests that intend to exercise the new no-raw refusal path) as part of this
  work, not as a follow-up.

## Non-goals

- **Ingesting/registering bim's E57 raw source.** Separate, already-scoped problem (see
  `docs/superpowers/specs/` density notes); bim simply cannot reach eval-grade until that
  lands, which is the correct behavior under this spec's refusal rule.
- **Spatial indexing / performance optimization for `load_laz_region`.** It re-streams the
  whole raw file sequentially per gate flip with no spatial index (e.g. COPC). Confirmed
  acceptable: eval-grade flips are an infrequent, deliberate user action, not a hot path.
- **Preseg/SAM point-set labeling accuracy.** Those remain NN-approximate at any replay
  density regardless of this fix — untouched by this spec.
- **Phase 0b ("bigger labeling/session cloud").** This spec removes phase 0b's motivating
  problem. Phase 0b itself is superseded, not implemented — no session point-cap changes
  are part of this work.
- **Frontend accuracy-display copy/treatment for `is_raw`.** Noted as a follow-up in the
  Review-step UI; not speced here.
