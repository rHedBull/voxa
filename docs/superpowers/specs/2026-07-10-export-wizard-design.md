# Export Wizard — labeled-dataset export from a session

**Date:** 2026-07-10
**Status:** Draft — under spec review

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
- **Phase A — materialize core** (the prior spec's §3–§4). Prerequisite.
- **Phase B — this export layer + wizard.** Consumes `materialize()`.

This document specifies Phase B and treats `materialize()` as a designed
dependency. The Phase-B plan must not start until Phase A lands.

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
     · `full raw`. Show a **point-count estimate** per choice (scan.ply count;
     `N`; raw `n_points` from `sources.json`). **Raw is disabled with a tooltip
     when no raw source is linked** (reuse `raw_source_available`). Show a
     **multi-GB size warning** when raw is selected.
  2. **Classes.** A confirmed-only toggle; a per-class **include/exclude**
     checkbox list (from the session palette); and **merge/rename rows** — group
     N source classes into one target with an editable label (+ color). Live
     preview of the resulting target taxonomy.
  3. **Review.** Summary: resolution + estimated points, the target taxonomy,
     and the **accuracy line** (§4). A single **Export** action → POST → receive
     the zip blob → `<a download>` (filename `scan_labeled_<resolution>.zip`).
- The wizard holds config in local state; nothing is persisted server-side.

### 2. Backend endpoint + pipeline

`POST /api/labels/export` (session-scoped; new handler in
`backend/routes/export.py`). Request (`app/schemas.py`):

```
ExportLabelsRequest:
  resolution: {"kind": "scan" | "subsample" | "raw", "n": int | null}
  confirmed_only: bool = False
  include_classes: list[int] | null   # source class ids to keep; null = all
  remap: list[{ from: list[int], to: {id: int, label: str, color: str} }] = []
  drop_unlabeled: bool = False
```

Pipeline (operates on the recentered/display frame; requires a prior `/api/load`
of the scan — same single-in-memory-cloud coupling as auto-fit/edit-export):

1. **materialize** → `positions, colors, class_ids, instance_ids` at the target
   resolution (Phase A). Regime A for `scan`/`subsample`, regime B for `raw`.
2. **confirmed-only** (if set) → points whose `instance_id` maps to a
   *non-confirmed* instance become `class_id = -1`. The instance→confirmed map is
   built from the active session's `instances_gt.json` (`segId` == `instance_id`,
   `confirmed` flag). Instances absent from the doc are treated as confirmed
   (legacy/plain applies) — **surface this assumption**, don't silently drop.
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

Validation: reject a `remap` whose `to.id`s collide with kept-through class ids,
or `from` ids not in the palette (422 with a clear message).

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
    "sample_spacing_m": 0.021,
    "semantic_boundary_uncertainty_m": 0.021,
    "note": "Semantic (preseg/legacy) boundaries are accurate to ~one labeling-cloud sample spacing and are set by labeling density, NOT the export resolution. Box/pipe (volumetric) boundaries are exact at any density."
  },
  "source": { "scan": "smart_ais_navvis", "session": "<id>", "exported_at": "<iso8601>" },
  "resolution": { "kind": "raw", "points": 156519044 },
  "filters": { "confirmed_only": true, "include_classes": [...], "drop_unlabeled": false }
}
```

### 4. Expected accuracy (from labeling density)

The reported accuracy is a property of the **labeling** cloud (`scan.ply`), not
the export resolution — surfaced both in the wizard Review step and in
`manifest.json::accuracy`.

- **Metric:** `sample_spacing_m` = the median nearest-neighbor distance among
  `scan.ply` points (its true sampling pitch). `semantic_boundary_uncertainty_m`
  ≈ `sample_spacing_m`.
- **Compute:** one `cKDTree(scan.ply)` query for the nearest non-self neighbor,
  over a bounded random subsample (e.g. 100k points) for speed; **reuse the
  KD-tree the materialize core already builds** over `scan.ply` in regime B.
  Cache per (scan) — it never changes for a given `scan.ply`.
- **Framing (must be exact in the copy):** this uncertainty applies to
  **semantic (preseg/legacy) labels only**; **box/pipe (volumetric) labels are
  exact** at any density; and export resolution does **not** improve it. The
  wizard line: *"Semantic boundary accuracy ±~2 cm (set by labeling density;
  unchanged by export resolution). Box/pipe boundaries: exact."*

### 5. Zip streaming + large-file safety

Raw density is multi-GB (156M points → a ~3–4 GB PLY). The response must **not**
buffer the PLY or the zip in RAM:

- Materialize in chunks where regime B streams the raw LAZ (per the prior spec's
  chunked reader); write the PLY to a **temp file** as chunks arrive.
- Build the zip on a **temp file** (`zipfile` streamed to disk), then return a
  `FileResponse` with `Content-Disposition: attachment`, and delete the temp on
  completion (background task).
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
  - **validation**: colliding `to.id` / unknown `from` id → 422.
  - **accuracy**: `sample_spacing_m` > 0 and matches a direct median-NN compute
    on the fixture cloud within tolerance; present in `manifest.json`.
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
- **Reusable remap presets** (named taxonomies applied across scans) — v1 is
  one-off per export; presets are a natural follow-up.
- Per-instance rows in `manifest.json` (id → class → n_points) — deferred; the
  taxonomy + accuracy block is enough for v1.
- **Phase A (materialize core) is a hard prerequisite** and is specified
  separately; this plan must not begin until it lands.
