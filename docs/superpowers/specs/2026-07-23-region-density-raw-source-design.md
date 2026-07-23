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

### 1. `raw_region_sample_spacing(raw_path, aabb_min, aabb_max, scene_is_z_up, offset)`

Thin wrapper: calls the existing `scenes.lidar_io.load_laz_region()` — already streams a
LAZ chunk-by-chunk (2M pts/chunk), filters to an AABB in the display (recentered) frame, and
returns only the in-region points at full native density, without loading the whole raw
cloud — then feeds the result into the existing `raw_sample_spacing()` for the p50/p90
calculation (which already internally subsamples to `sample=100_000` points, so no extra
cap is needed here even for a densely-populated region).

Used by the eval-grade gate, scoped to the region's own prism AABB — cheap because it only
returns points local to one region, not the whole scan.

### 2. `raw_reservoir_sample_spacing(raw_path, scene_is_z_up, offset, sample=100_000, chunk=1_000_000, seed=0)`

Streams the *whole* raw file via the same `scenes.lidar_io._laz_chunk_iter` chunk iterator
`materialize_raw` uses, transforming each chunk into the display frame, but — unlike
`materialize_raw` — does not replay labels or accumulate the whole cloud. Instead it
maintains a bounded reservoir sample of positions (Algorithm R, seeded, size `sample`) as
chunks stream past, so memory stays bounded regardless of file size (raw files run up to
~156M points). The resulting reservoir feeds the same spacing calculation.

Used by the export accuracy endpoint, which is scan-wide rather than region-scoped, so
there's no AABB to filter by — a full-file streaming pass is unavoidable if we want a raw
measurement at all, and reservoir sampling keeps it statistically unbiased over point
ordering rather than biased toward however the LAZ happens to be tiled/sorted on disk.

Both helpers should share `raw_sample_spacing`'s underlying KD-tree/subsample logic (factor
it out of `raw_sample_spacing` into a small internal `_spacing_from_positions(positions,
sample, seed)` used by all three public functions) rather than duplicating it.

## Behavior changes

### `labeling/regions.py::flip_status`

Gains `raw_path: str | None` and `scene_is_z_up: bool` parameters (positional/keyword,
inserted alongside the existing `positions` param — `positions` stays required for the
`categories`/review-budget check and the point-floor check, which remain scoped to the
session's working array as today; only the *spacing* measurement changes source).

- If `raw_path` is not `None`: resolve the region's runtime prism to an AABB (reuse the
  existing `shift_prism`/`prism_indices` bounding logic — or simply take the prism's own
  min/max corners, whichever is already exposed) and measure spacing via
  `raw_region_sample_spacing`.
- If `raw_path` is `None` (no raw source registered for this scan — e.g. bim today):
  **refuse eval-grade outright** with `RegionError("no raw source registered for this scan
  — cannot verify true density; register a raw source before flipping regions to eval-
  grade")`. Do NOT fall back to measuring `positions` — that reintroduces exactly the
  ambiguity this spec removes, and would let regions silently pass at whatever density the
  session cloud happens to be.
- The point-floor (`MIN_GATE_POINTS`) and near-zero-spacing (`MIN_GATE_P50_M`) checks apply
  to whichever point set was actually measured (raw region points when available).

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
        -> stream whole LAZ via _laz_chunk_iter, reservoir-sample positions
        -> raw_sample_spacing(...) -> p50, p90
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
  - `raw_reservoir_sample_spacing`: verify reservoir size is bounded regardless of input
    file size (test against a file larger than `sample`), and that sampling is
    order-independent (shuffling chunk order doesn't change the *statistical* result beyond
    sampling noise — e.g. compare mean spacing across a couple of seeds/orderings within a
    tolerance).
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
