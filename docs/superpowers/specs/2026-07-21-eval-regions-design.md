# Eval Regions (Eval-Labeling Phase 1) — Design

**Date:** 2026-07-21
**Status:** Implemented (2026-07-22, branch feat/eval-regions)
**Branch:** `feat/eval-regions`

## Problem

The eval-grade labeling spec
(`engine/research/fable/docs/superpowers/specs/2026-07-20-eval-labeling-design.md`,
"Eval regions: declared exhaustiveness" + "Tooling delta — Voxa", phase 1)
makes **eval regions** the unit the benchmark actually scores: explicit
spatial regions inside which every point is accounted for. Metrics run only
inside regions; outside them the scan is not part of the benchmark. A region
is **eval-grade** only if its measured local p90 nearest-neighbor spacing is
≤ 10 mm, and an eval-grade region's geometry is immutable — region placement
that can be silently redrawn is the biggest exclusion channel a benchmark
can have.

Voxa has no notion of a region today. Phase 1 adds the concept: per-scan
region geometry + status, a way to draw regions in the viewport, and enough
readout (coverage, contained instances, measured density) that a labeler can
work a region toward exhaustive. The full five-way category partition,
loader invariants, and manifest are phases 2–3.

## Decisions settled with the user (2026-07-21)

- **Phase 0b (canonical labeling-cloud import) is shelved.** A ~100M-point
  labeling cloud is a viewer/session-scale problem, not just an import
  script; regions are built and measured against the existing `scan.ply`
  clouds. If no region on a current cloud can meet the p90 ≤ 10 mm bar,
  that is an honest outcome (and the motivation to revisit 0b later).
- **Region geometry is a vertical prism** — XZ polygon footprint + aimed
  height, drawn with exactly the Prism tool's interaction (no full-height
  shortcut, no box variant).
- **The Region tool is a 7th rail tool.** Deliberate deviation from the
  "a tool selects points" rail principle: drawing a region selects and
  labels nothing.
- **Regions get a second tab in the right panel** (Instances | Regions),
  which also lists the confirmed instances belonging to each region.
- **Instance membership = majority-inside**: a confirmed instance is "part
  of" a region iff strictly more than 50 % of its points are inside the
  region prism.
- **Draft → eval-grade gates, records, and locks**: the flip measures the
  region's local p50/p90 spacing, refuses if p90 > 10 mm, records the
  measurement, and makes the geometry immutable until flipped back to
  draft.
- **Overlay = tool-active + per-region eye toggle**: region volumes render
  while the Region tool is active; each region row has an eye toggle to
  keep it visible while labeling with other tools.
- **Regions tab shows per-region unlabeled %** — the number a labeler
  drives down. Phase 2 upgrades it to the full category partition.
- **Backend approach A**: a new scan-level route module owns
  `eval_regions.json`; stats are computed server-side over the full-res
  working arrays; the backend enforces the eval-grade gate and lock.

## Non-goals

- No point categories, no `point_component_ids`, no loader invariants, no
  manifest (phases 2–3). The exhaustiveness invariant ("no unlabeled point
  in-region") is *displayed* (unlabeled %), not enforced.
- No benchmark-version machinery. Deleting an eval-grade region is possible
  by flipping it to draft first; version bumps and in-region-fraction
  tracking are phase 3+.
- No box/OBB region shape, no full-height shortcut, no multi-prism regions.
- No region support on legacy-tier scenes.
- No eval-regions entry in the `scan_schema` layout contract yet (phase 3
  adds it alongside the manifest); phase 1 just writes the file. Known
  consequence, accepted: `scan_schema`'s validator flags scan-root entries
  outside `ALLOWED_TOPLEVEL` as an "unexpected top-level entry" **warning**
  (warn-level only, nothing breaks). Allowing it now would mean a
  cross-repo `scan-schema` release for one line; phase 3 touches that
  contract anyway.

## Design

### 1. Data model: `<scan>/eval_regions.json`

Scan-level and session-independent — two sessions on one scan see the same
regions. Lives at the scan root, matching the parent spec's schema sketch.
Written atomically via the `scan_schema` durable-write helpers.

```json
{
  "version": 1,
  "next_region_id": 3,
  "regions": [
    {
      "id": 1,
      "name": "pump skid A",
      "status": "eval_grade",
      "prism": { "polygon": [[x, z], ...], "y0": 2.1, "height": 3.4 },
      "created_at": "2026-07-21T18:00:00Z",
      "accuracy": {
        "p50": 0.004, "p90": 0.008, "loa": "LOA40",
        "measured_at": "2026-07-21T18:30:00Z"
      }
    }
  ]
}
```

- `id` is monotonic per scan (`next_region_id`), never reused — the same
  stable-id contract instances follow.
- `status` ∈ `draft | eval_grade`.
- `accuracy` is present **iff** `status == "eval_grade"`: it is the recorded
  gate measurement (`p50`/`p90` in meters + the derived USIBD LOA band from
  `labeling.materialize.loa_band`). Flipping back to draft clears it;
  re-flipping re-measures.
- **Coordinates are in the scan's stored frame** (`scan.ply` coordinates).
  The routes add/subtract the load-time `recenter_offset` on the way
  out/in. For current annotated scans the offset is `[0,0,0]` (UTM offsets
  were already removed at scan level), but this keeps the file a portable
  benchmark artifact — phase 3's manifest generator reads it without a voxa
  load. One caveat: for z-up scans voxa's load applies the z-up → y-up
  rotation *before* recentering, so "stored frame" there means the y-up
  display frame plus offset — deterministic and consistent within voxa, but
  the phase-3 manifest generator must replay the same rotation for those
  scans (a vertical prism can't be expressed in the z-up frame anyway).

### 2. Backend: `backend/routes/regions.py`

New route module, registered like the existing ones in `main.py`. All
endpoints require the active scene to be an annotated-tier scan whose
directory is resolvable (else 409, mirroring the segment routes' pinning
style); stats **and the eval-grade flip** additionally require an active
`SegmentSession` — both need full-res positions, and the session's
`positions` array is the full-res cloud (`app.core._state`'s loaded cloud
is subsampled to `VOXA_MAX_POINTS` and would inflate the measured p90). Pure helpers
(region resolution, membership counting) live in
`backend/labeling/regions.py` so routes stay thin, matching the
`shapes.py`/`outliers.py` pattern.

- `GET /api/regions` — full region list, prism geometry converted to the
  runtime (recentered) frame.
- `POST /api/regions` `{name?, prism}` — create as `draft`. Server assigns
  `id` and a default name (`Region <id>`). The prism must satisfy the same
  constraints `prism_indices` enforces (≥ 3 vertices, height > 0), but with
  an explicit 422 instead of `prism_indices`'s silent empty selection —
  `apply-shape` itself has no such validation, so there is nothing to
  mirror there.
- `PATCH /api/regions/{id}` `{name? | prism? | status?}` — partial update.
  **The backend owns the locks** (same philosophy as
  `reject_frozen_class`):
  - `prism` on an eval-grade region → 422 ("eval-grade geometry is locked;
    flip to draft first").
  - `status: "eval_grade"` runs the gate: resolve the region's full-res
    point indices via the existing `shapes.py::prism_indices`, then compute
    `labeling.materialize.raw_sample_spacing` over those positions. Refuse
    with 422 if the region holds fewer than **100 points** or if
    p90 > 0.010 m. The floor is load-bearing, not cosmetic:
    `raw_sample_spacing` returns `(0.0, 0.0)` for n < 2, which would
    otherwise *pass* the p90 check on a near-empty region. On success,
    record `accuracy` and lock.
  - `status: "draft"` unlocks and drops `accuracy`.
  - `name` changes are always allowed — a name is not geometry.
- `DELETE /api/regions/{id}` — draft only; deleting an eval-grade region
  requires the explicit draft flip first (422 otherwise).
- `GET /api/regions/stats` — per region, computed against the active
  session's full-res working arrays:

  ```json
  {
    "regions": [
      {
        "id": 1,
        "n_points": 182034,
        "n_unlabeled": 12490,
        "instances": { "7": { "inside": 1420, "total": 1500 }, ... }
      }
    ]
  }
  ```

  `n_unlabeled` counts in-region points with working `class_ids == -1`.
  `instances` maps every instance id with ≥ 1 point in-region to its
  inside/total point counts. The **frontend** filters this to confirmed
  instances (it owns confirmed-status — the `protect_instances` pattern)
  and applies the majority rule (`inside / total > 0.5`). All vectorized
  numpy (`prism_indices` + `np.bincount`) over ~3M points per region — no
  caching.

Region edits are **not** on the undo stack — the same rule as Draw/Beam
geometry (`structure.json`, `centerlines.json`).

### 3. Frontend: Region tool + drawing

- `region` joins `frontend/src/label-tools.js` as the 7th rail tool.
  Enabled only for annotated-tier scenes with an active session; disabled
  with a tooltip otherwise (the SAM-without-sidecar precedent).
- **The prism draw machinery is extracted, not duplicated.** The
  footprint + height interaction currently inside `prism-mode.jsx`
  (`PrismOverlay`, rubber band, fill/wall mesh helpers) is parameterized by
  a commit handler:
  - The Prism tool's handler feeds the classify/apply pipeline, unchanged.
  - The Region tool's handler (`frontend/src/region-mode.jsx`) `POST`s the
    drawn `{polygon, y0, height}` to `/api/regions` and resets the draw
    state for the next region. No class picker, no apply, no point
    highlight — drawing a region labels nothing.
- `RegionOptions` in `tool-options.jsx` fills the left-rail slot with the
  usual interaction hints (click corners, double-click/Enter to close,
  click to commit height, Backspace/Esc), mirroring `PrismOptions`.

### 4. Frontend: Regions tab + overlay

- The right panel becomes tabbed: **Instances | Regions**. The Regions tab
  (`frontend/src/region-panel.jsx`) lists one row per region:
  - editable name, status badge (draft = amber, eval-grade = green),
  - **unlabeled %** from `/api/regions/stats`,
  - per-region eye toggle (overlay visibility while other tools are
    active),
  - delete (draft only),
  - a **"Mark eval-grade"** action — on success shows the recorded
    p50/p90/LOA inline; on failure surfaces the 422 detail (measured p90
    vs the 10 mm bar). Eval-grade rows offer "Back to draft" instead.
- Expanding a row lists the **confirmed instances majority-inside** the
  region (name, class swatch, inside-fraction); clicking one selects and
  focuses it, like an Instances-panel row.
- **Overlay**: translucent fill + outline per region (reusing the prism
  fill/wall mesh helpers), colored by status (draft amber, eval-grade
  green). Visible set = all regions while the Region tool is active, plus
  eye-toggled regions under any other tool. Eye state is UI-local (not
  persisted).
- Stats refetch after label-changing operations (apply, confirm, undo,
  redo, denoise, cut) — hooked into the same refresh path instance counts
  already use, debounced.

### 5. Edge cases

- Empty or too-sparse region → eval-grade flip refuses with a clear 422
  (point count + measured p90 in the detail).
- Degenerate drawn prism (self-intersecting footprint) is prevented by the
  coplanar draw interaction, same as the Prism tool; the backend still
  validates vertex count and height.
- Concurrent sessions: the file is scan-level with atomic writes;
  last-write-wins between sessions, acceptable for a single-labeler tool.
- Legacy-tier scenes: tool disabled; routes 409 on non-annotated scenes.
- A session resumed on a scan whose regions changed since: no pinning —
  regions are scan truth, the list simply reflects the file.

### 6. Testing

- **Backend** (`backend/tests/test_regions.py`): CRUD round-trip; id
  monotonicity across deletes; lock enforcement (422 on geometry edit /
  delete of eval-grade); gate behavior on synthetic dense (passes, records
  accuracy) and sparse (fails with p90 in detail) clouds; stats
  correctness on a synthetic cloud with known labels/instances (in-region
  counts, unlabeled count, per-instance inside/total); frame conversion
  with a nonzero `recenter_offset`.
- **Frontend** (vitest): majority-inside filter + unlabeled-% formatting
  as pure helpers; region-tool gating in `label-tools.test.js`; the prism
  draw extraction keeps existing `prism-geom` tests green; a jsdom test
  for Regions-tab basics (rows render, eval-grade action calls the API).
- **Browser verification** before merge (per the global rule): draw a
  region on a throwaway session, check overlay, tab readouts, eval-grade
  flip, and console/network cleanliness.

### 7. Documentation

- CLAUDE.md gains the Region tool bullet (rail list + panel + endpoints).
- This spec ships in the same PR as the implementation.
