# Export Wizard — labeled-dataset export from a session

**Date:** 2026-07-10
**Status:** Implemented (Phase B shipped on `feat/export-wizard`)

## Problem

There is no way to get a labeled point cloud *out* of voxa for training. A
session holds per-point labels on `scan.ply`, but a consumer wants a
self-contained labeled cloud at a chosen density, optionally with an
export-specific class taxonomy (e.g. collapse ceiling+floor+wall → "building"),
and needs to know how trustworthy the label boundaries are.

## Goal

An **Export** button in Label mode opens a wizard that produces a labeled cloud
from the **active session**: pick a resolution, filter/remap classes, and
download a zip of a labeled `.ply` + a `manifest.json` (taxonomy + accuracy +
provenance). Export never mutates the stored labels.

## Layering — this feature depends on the (unbuilt) materialize core

The wizard sits on **Phase 2 of the resolution-independent-labels design**
(`docs/superpowers/specs/2026-07-10-resolution-independent-labels-design.md`
§3–§4): `materialize(session, resolution) → (positions, colors, class_ids,
instance_ids)` for a target cloud — regime A (≤`scan.ply`: subsample+index,
exact) and regime B (raw: `derivation→sources.json` resolver, frame-align, NN +
shape replay). That core is **designed and spec-reviewed but not implemented**.

**Implementation is therefore two phases:**
- **Phase A — materialize core** (the prior spec's §3–§4). Prerequisite. Phase A
  must also **add `raw_source_available: bool` to `LoadResponse`** (populated by
  the §4 raw-path resolver: true iff `derivation.root.source_id` resolves to a
  `raw/*.laz`). This field does **not** exist today (`backend/app/schemas.py`
  `LoadResponse` has no such field; the earlier edit-export spec proposed it but
  it was never implemented) — the wizard's raw-availability gate depends on it, so
  it is part of Phase A's deliverables, not a pre-existing hook.
- **Phase B — this export layer + wizard.** Consumes `materialize()`.

This document specifies Phase B and treats `materialize()` + `raw_source_available`
as designed dependencies. The Phase-B plan must not start until Phase A lands.

### Scope (v1, decided during brainstorming)

- **In:** single active session; resolutions `scan` / `N` / `raw` (all three);
  PLY output (`x y z rgb class_id instance_id`); filters = confirmed-only +
  per-class include/exclude; **export-time class remap** (merge + rename);
  optional drop-unlabeled; browser **zip download**; expected-accuracy shown in
  the wizard and written to the zip.
- **Out:** multi-scene batch, train/val split, `.npy`/other formats,
  server-side output destination, per-instance manifest rows.

## Design

### 1. Trigger + wizard UX (frontend)

- **Export button** in the Label-mode header, near Save (session-scoped; label
  it "Export…"). Disabled when no active session.
- A **3-step modal** (`frontend/src/export-wizard.jsx`, new):
  1. **Resolution.** Radio: `scan.ply native` · `subsample to N` (number input)
     · `full raw`. `subsample to N` is a **down-sample only**: constrain the input
     to `1 ≤ N ≤ len(scan.ply)` (densities *between* `scan.ply` and raw are a
     follow-up — see Out of scope). Show a **point-count estimate** per choice
     (scan.ply count; `N`; raw `n_points` from `sources.json`). **Raw is disabled
     with a tooltip when `raw_source_available` is false.** Show a **multi-GB size
     warning** when raw is selected.
  2. **Classes.** A confirmed-only toggle; a per-class **include/exclude**
     checkbox list (from the session palette); and **merge/rename rows** — group
     N source classes into one target with an editable label (+ color). Live
     preview of the resulting target taxonomy. A class assigned to a merge row is
     **visually flagged/disabled in the exclude list** (and vice versa) so a user
     can't configure a class as both excluded and remapped (exclude wins — §2).
  3. **Review.** Summary: resolution + estimated points, an **estimated
     labeled-point count *after* filters** (materialize is not run here — use the
     working array's per-class/confirmed counts scaled to the target density as
     the estimate), the target taxonomy, and the **accuracy line** (§4). If the
     post-filter estimate is **~0**, disable Export with an explanatory message.
     A single **Export** action → POST → zip blob → `<a download>`
     (`scan_labeled_<resolution>.zip`).
- The wizard holds config in local state; nothing is persisted server-side.

### 2. Backend endpoint + pipeline

`POST /api/labels/export` (session-scoped; new handler in
`backend/routes/export.py`). Request (`app/schemas.py`):

```
ExportLabelsRequest:
  scene: str                          # tier-prefixed scene id (stale-state guard)
  session_id: str                     # active session (stale-state guard)
  resolution: {"kind": "scan" | "subsample" | "raw", "n": int | null}
  confirmed_only: bool = False
  include_classes: list[int] | null   # source class ids to keep; null = all
  remap: list[{ from: list[int], to: {id: int, label: str, color: str} }] = []
  drop_unlabeled: bool = False
```

**Stale-`_state` guard (first thing, before any work):** 409 if `req.scene`
!= `_state["scene"]` or `req.session_id` != `_state["session_id"]` — exactly like
`edit_export_ply`'s `req.scene != scene` check (`backend/routes/export.py:24`).
Then **snapshot** the fields the export needs (source path, `recenter_offset`,
`session_dir`, working arrays, `raw_source_available`) into locals at request
start, so a concurrent `/api/load` switching scene/session mid-export can't mutate
`_state` out from under a minutes-long raw export (CLAUDE.md single-in-memory
gotcha; this feature is the first to make that window minutes, not seconds).

Pipeline (operates on the recentered/display frame; requires a prior `/api/load`):

1. **materialize** → `positions, colors, class_ids, instance_ids` at the target
   resolution (Phase A). `scan` and `subsample` (`N ≤ len(scan.ply)`, validated)
   are regime A; `raw` is regime B.
2. **confirmed-only** (if set) → points whose `instance_id` maps to a
   *non-confirmed* instance become `class_id = -1`. The instance→confirmed map is
   built from the active session's `instances_gt.json` (`segId` == `instance_id`,
   `confirmed` flag). Instances absent from the doc are treated as confirmed
   (legacy/plain applies); **record the count of such absent-but-present
   instance ids** and write it to `manifest.json::filters.absent_instances` (don't
   only bury the assumption in prose).
3. **include/exclude** → points whose source `class_id ∉ include_classes` become
   `-1`.
4. **remap** → remaining source `class_id`s → target ids per `remap` (each `from`
   set collapses to its `to.id`; unmapped kept classes pass through with their
   original id/label/color). Build the **target taxonomy** for `manifest.json`.
   **Instances are untouched** — merging classes never merges `instance_id`s.
5. **drop_unlabeled** (if set) → drop `class_id == -1` points (and their
   positions/colors/instance_ids).
6. **write** binary PLY + `manifest.json`, **zip**, **stream** as the response
   (see §3, §5).

**Precedence — stated explicitly:** filters apply in the numbered order, so
**exclude/confirmed-only win over remap** (a class both excluded *and* in a remap
`from` set is already `-1` by step 3, so step 4 never sees it). The UI enforces
this by disabling the contradiction (§1 step 2).

**Validation (422 with a clear message) — reject:**
- a `from` id not in the session palette;
- **overlapping `from` sets** across rows (one source id in two rows → ambiguous);
- **two rows with the same `to.id`** but different `label`/`color`;
- a `to.id` that collides with a **kept-through** class id (a class that survives
  `include_classes` and is *not* consumed by any `from` set) — collision is
  checked against the post-filter kept set, not the full palette;
- `resolution.kind == "subsample"` with `n > len(scan.ply)` or `n < 1`;
- `resolution.kind == "raw"` when `raw_source_available` is false.

### 3. PLY + manifest.json (zip contents)

- **`scan_labeled.ply`** — binary little-endian PLY, vertex props: `x y z`
  (float32), `red green blue` (uchar), `class_id` (int32; `-1` = unlabeled),
  `instance_id` (int32; `-1` = none). Reuse the existing edit-export PLY writer
  patterns in `backend/routes/export.py` (extend to add the two label props).
- **`manifest.json`** (supersedes the bare `classes.json`):

```json
{
  "classes": { "0": {"label": "building", "color": "#8b5cf6"}, "1": {"label": "pipe", "color": "#22c55e"} },
  "accuracy": {
    "labeling_points": 2989215,
    "sample_spacing_p50_m": 0.021,
    "sample_spacing_p90_m": 0.058,
    "semantic_boundary_uncertainty_m": 0.058,
    "loa": "LOA10",
    "method": { "sample_spacing": "d_i = min_{j != i} ||x_i - x_j||_2 ...", "percentiles": "...", "boundary_uncertainty": "...", "loa": "..." },
    "note": "Semantic (preseg/legacy) boundaries are accurate to ~one labeling-cloud sample spacing (reported as p90 to reflect non-uniform LiDAR sampling) and are set by labeling density, NOT the export resolution. Box/pipe (volumetric) boundaries are exact at any density."
  },
  "source": { "scan": "smart_ais_navvis", "session": "<id>", "exported_at": "<iso8601>" },
  "resolution": { "kind": "raw", "points": 156519044 },
  "filters": { "confirmed_only": true, "include_classes": [...], "drop_unlabeled": false, "absent_instances": 0 }
}
```

### 4. Expected accuracy (from labeling density)

The reported accuracy is a property of the **labeling** cloud (`scan.ply`), not
the export resolution — surfaced both in the wizard Review step and in
`manifest.json::accuracy`.

- **Metric (exact):** per-point nearest-neighbor spacing
  `d_i = min_{j != i} ||x_i - x_j||_2` over the `scan.ply` positions;
  `p50`/`p90` are the 50th/90th percentiles of `{d_i}` (`numpy.percentile`,
  linear interpolation) over a deterministic random subsample of
  `min(N, 100000)` query points (`default_rng(0)`), with the KD-tree built
  over all `N` points. Report **both p50 and p90** and use **p90** as
  `semantic_boundary_uncertainty_m` — walkthrough LiDAR (NavVis/Matterport) is
  non-uniformly sampled (dense near the path, sparse far away), so a p50-only
  figure understates worst-case boundary trust in sparse regions. A drawn
  boundary snaps to the nearest sample, so its position error is at most ~one
  sample spacing.
- **LOA band:** `accuracy.loa` is the finest USIBD LOA Spec v3.0 band whose
  upper tolerance is ≥ p90 (LOA50 ≤ 1 mm < LOA40 ≤ 5 mm < LOA30 ≤ 15 mm <
  LOA20 ≤ 50 mm < LOA10) — the standard AEC vocabulary for the same claim,
  computed by `labeling.materialize.loa_band`. The full computation is
  embedded machine-readably in `accuracy.method` so the manifest is
  self-describing.
- **Compute:** a `cKDTree(scan.ply)` query for the nearest non-self neighbor over
  a bounded random subsample (e.g. 100k points) for speed. **Build it (and cache
  per-scan) independently of regime** — regime A ("scan"/"subsample") builds no
  KD-tree of its own, so the accuracy metric owns this small build; regime B may
  share the materialize KD-tree if convenient, but the metric must not *depend*
  on it. It never changes for a given `scan.ply`.
- **Framing (must be exact in the copy):** this uncertainty applies to
  **semantic (preseg/legacy) labels only**; **box/pipe (volumetric) labels are
  exact** at any density; and export resolution does **not** improve it. The
  wizard line: *"Semantic boundary accuracy LOA30 — ±~2 cm (p90 sample
  spacing; p50 0.9 cm; set by labeling density, unchanged by export
  resolution). Box/pipe boundaries: exact."* — the LOA band and "±~2 cm"
  are **illustrative**; render the actual computed values for the loaded
  scan, never hardcoded literals.

### 5. Zip streaming + large-file safety

Raw density is multi-GB (156M points → a ~3–4 GB PLY). The response must **not**
buffer the PLY or the zip in RAM:

- Materialize in chunks where regime B streams the raw LAZ (per the prior spec's
  chunked reader); write the PLY to a **temp file** as chunks arrive.
- Build the zip on a **temp file** (`zipfile` streamed to disk), then return a
  `FileResponse` with `Content-Disposition: attachment` + a `BackgroundTask` that
  unlinks the temp on successful send.
- **Failure cleanup:** `BackgroundTask` only runs on a *successful* response, so
  the materialize+write+zip build must live inside a `try/finally` (or
  `tempfile` context managers) that unlinks the temp PLY and temp zip on **any**
  exception — otherwise a failed multi-GB raw export (LAZ read error, disk full,
  OOM on a 156M cloud) leaks multi-GB temp files with no eviction. Write temps to
  a dedicated dir so a sweeper could evict orphans if a crash bypasses `finally`.
- The wizard's raw warning sets expectations; the browser download handles the
  size. (Compression: `manifest.json` is tiny; the PLY is near-incompressible
  floats — use `ZIP_STORED` or fast `ZIP_DEFLATED` level 1 to avoid CPU blowup
  on multi-GB.)

### 6. Frontend download flow

Reuse the edit-export download idiom (`mode-edit.jsx`: `fetch` → `blob` →
`URL.createObjectURL` → `<a download>`). POST the config, receive the zip blob,
trigger the download, revoke the URL. Show an in-wizard busy state; surface
backend errors (422 validation, 409 no-session/no-raw) inline, not `alert`.

## Testing

- **Backend** (`backend/tests/`):
  - Regime-A export (`scan`, small fixture): PLY has the right vertex count,
    `class_id`/`instance_id` props present; `manifest.json` classes match.
  - **confirmed-only**: points of an unconfirmed instance become `-1`.
  - **include/exclude**: excluded class's points become `-1`.
  - **remap merge**: two source classes → one target id/label in both the PLY
    `class_id` field and `manifest.classes`; `instance_id`s unchanged.
  - **drop_unlabeled**: `-1` points absent from the PLY; counts consistent.
  - **precedence**: a class both excluded and in a remap `from` set → its points
    are `-1` (exclude wins), remap is a no-op for it.
  - **validation** (each → 422): unknown `from` id; overlapping `from` sets;
    duplicate `to.id` with differing label/color; `to.id` colliding with a
    kept-through class; `subsample` with `n > len(scan.ply)` or `n < 1`; `raw`
    when `raw_source_available` is false.
  - **stale-state guard**: `req.scene`/`req.session_id` ≠ active `_state` → 409.
  - **absent instances**: an instance id present in the working array but missing
    from `instances_gt.json` is counted in `manifest.filters.absent_instances`.
  - **empty after filters**: filters removing all labeled points yield a valid
    (0-labeled or 0-vertex) response without a 500 (the wizard prevents this via
    the Review estimate; the endpoint must still not crash if called directly).
  - **accuracy**: `sample_spacing_p50_m`/`_p90_m` > 0, `p90 ≥ p50`, and match a
    direct NN compute on the fixture cloud within tolerance; present in manifest.
  - **zip**: response is a valid zip containing exactly `scan_labeled.ply` +
    `manifest.json`.
  - (Regime-B/raw path is exercised by Phase A's own tests; here, one test with a
    small linked-raw fixture confirms the export wires resolution="raw" through.)
- **Frontend**: pure-function tests for the remap→taxonomy reducer and the
  point-count estimate (vitest, no DOM). The wizard modal itself is
  browser-verified (load a scan, open Export, run a `scan`-resolution export,
  confirm the zip downloads with a valid PLY + manifest, zero console errors) —
  **use a throwaway session** (export is read-only, but keep the browser-verify
  discipline).

## Out of scope / follow-ups

- Multi-scene batch export, train/val split, `.npy`/`.pcd`/manifest-only formats,
  server-side output destination — all deferred (v1 is single-session, PLY zip,
  download).
- **Intermediate densities `scan.ply < N < raw`** (a regime-B subsample of the
  raw cloud): materialize supports it, but v1's `subsample` is down-sample only
  (`N ≤ len(scan.ply)`) and raw is all-or-nothing. Exposing a raw-subsample
  target is a natural follow-up.
- **Reusable remap presets** (named taxonomies applied across scans) — v1 is
  one-off per export; presets are a natural follow-up.
- Per-instance rows in `manifest.json` (id → class → n_points) — deferred; the
  taxonomy + accuracy block is enough for v1.
- **Phase A (materialize core) is a hard prerequisite** and is specified
  separately; this plan must not begin until it lands.
