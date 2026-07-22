# Invariants + Manifest in scan-schema — Design (eval-labeling phase 3)

**Date:** 2026-07-22
**Status:** Proposed
**Upstream spec:** `engine/research/fable/docs/superpowers/specs/2026-07-20-eval-labeling-design.md`
(§ "Loader invariants (checkable)", § "Tooling delta — Voxa" item 3)
**Predecessors:** phase 0 (primitive vocabulary, PR #43), phase 1 (eval regions, PR #44),
phase 2 (point categories + components, PR #45)

## Problem

Phases 0–2 gave voxa the arrays and files the eval-labeling spec's loader
invariants need — `gt_class_ids`, `gt_segment_ids`, `gt_point_category`,
`gt_point_component_ids`, `eval_regions.json`, per-region measured accuracy —
but nothing checks them against the spec's own contract. Today's save-time gate
(`scan_schema.invariants.validate_invariants`, called from
`backend/labeling/segment_io.py::save_labels`) only covers the older,
lower-level SCHEMA invariants 3–6 (class↔instance unlabeled agreement,
per-segment class consistency, known class ids, `class_map_version` match).
Categories and components are written and never validated: nothing stops an
`excluded_review` point spray, a review-budget blowout, a component/instance
coverage mismatch, or a confirmed instance whose points quietly got
category-marked out from under it after confirmation.

The upstream spec's "eight loader invariants" are a separate, GT-semantics-level
set from the existing SCHEMA 1–6 (array shape/dtype, class/instance agreement)
and need their own module so the two don't collide in naming or numbering.
Scoping this phase also surfaced a ninth: `confirmed` (frontend-owned, in
`instances_gt.json`) and point-level category/class can diverge today with
nothing noticing.

The goal, per the upstream spec: voxa's save-time check and a future harness's
load-time check must be **the same code**, living in the shared `scan_schema`
package, so they cannot drift apart.

## Goals

- Implement all 8 upstream loader invariants plus the confirmed/category
  reconciliation discovered in scoping, as pure functions in `scan_schema`.
- A scan-level manifest (class histogram, review-budget check, LOA/p50-p90
  rollup, provenance) regenerated on every save.
- One code path: voxa's save gate and `scan_schema validate_archive` (the
  load-time / auditing path) call the identical functions.
- Migrate the three precious pre-phase-2 scans (munich, water_treatment_navvis,
  smart_ais_navvis) so the new gate applies uniformly, with no grandfathering
  exceptions to maintain going forward.
- Pin voxa's `scan_schema` dependency to a commit, closing the silent-drift risk
  the unpinned `@main` requirement currently carries.

## Non-goals

- Renaming voxa's `output/gt_*` files to the upstream spec's illustrative names
  (`point_instance_ids.npy`, `instances.json`, `manifest.json`). Voxa's existing
  names are the real contract this phase validates against; the spec's schema
  sketch was illustrative, not literal.
- Introducing a backend-owned `instances.json` as a new source of truth.
  `instances_gt.json` (frontend-owned) stays authoritative for instance
  metadata; the backend gate reads it at save time rather than replacing it.
- A downstream benchmark harness. This phase only makes the load-time check
  (`validate_archive`) capable of running the new invariants; nothing in this
  repo yet consumes them as a benchmark gate.
- Relations/hierarchy (generation 3, per the upstream spec) — instance-id
  stability (invariant 7) is the only forward contract this phase enforces.

## Design

### 1. Module layout

`scan_schema/eval_invariants.py` (new): the 9 checks below, one function each,
pure over parsed arrays/JSON — no file I/O — so they're unit-testable in
isolation and callable from both voxa and `validate_archive`. Named separately
from the existing `invariants.py` (SCHEMA invariants 1–6, a lower-level
array-shape/class-consistency set) to avoid numbering collisions; a docstring
cross-references both sets so a reader isn't left guessing which "invariant 3"
is meant.

`scan_schema/manifest.py` (new): `build_manifest(...) -> dict`, assembling the
fields in §3 from the same inputs the invariants consume. Called from both
voxa's save path and a new `scan_schema` CLI manifest subcommand (load-time
regeneration for auditing).

`backend/labeling/segment_io.py::save_labels` grows to:

1. Load `instances_gt.json` for the active session (already on disk
   per-session; no new file).
2. Call `eval_invariants.check_all(class_ids, segment_ids, categories,
   component_ids, instances_doc, eval_regions, prior_instances_doc)`.
3. On any failure, 422 with a structured `{invariant, detail}` list — the same
   shape as the existing eval-region gate 422 — and do not write anything.
4. On success, call `manifest.build_manifest(...)` and merge its output into
   `gt_segment_metadata.json` (extending the existing file, not a new doc).

### 2. The 9 invariants

| # | Rule | Checked against |
|---|---|---|
| 1 | Every in-region point has exactly one category | `gt_point_category.npy` × `eval_regions.json` region masks |
| 2 | `excluded_review` points per eval-grade region ≤ 3% budget | same, region status `eval-grade` only (draft regions are unchecked — the budget is an eval-grade admission bar, already enforced once at flip time in phase 2; this re-checks it stays true on every subsequent save) |
| 3 | Every instance id in `gt_segment_ids.npy` exists in `instances_gt.json` with a valid class, except review blobs (`class: null` iff every member point is `excluded_review`); every `excluded_review` point belongs to such a blob | `gt_segment_ids.npy` × `instances_gt.json` × `gt_point_category.npy` |
| 4 | No new GT contains class ids 4 (`double`) or 6 (`unknown`) | `gt_class_ids.npy` — defense-in-depth alongside the existing assign-time `reject_frozen_class` guard, since this phase's gate runs at a different point (whole-array save, not per-op assign) |
| 5 | Every point with an instance id has a component id and vice versa | `gt_segment_ids.npy` × `gt_point_component_ids.npy` |
| 6 | Manifest regenerated and consistent (drift is a hard error) | freshly computed `manifest.build_manifest(...)` vs. what's about to be written — this is inherently satisfied by construction (the gate always writes what it just computed) and exists as a check for callers that pass in a stale manifest to compare against, e.g. a future harness re-deriving one independently |
| 7 | Instance ids never reused or renumbered across saves | current `instances_gt.json` ids vs. the previous save's ids (read from the prior `gt_segment_metadata.json`, which already exists at the time of a second save) — only flags ids that vanished-and-reappeared-differently or were renumbered; new ids are always fine |
| 8 | An eval-grade region's declared accuracy is consistent with its measured p90 (≤10mm) | region's stored `accuracy.p90` (measured at phase-1 gate time) re-checked into the manifest; re-measured via the existing `labeling.materialize.raw_sample_spacing` if the region's points changed since that measurement |
| 9 (discovered in scoping) | A `confirmed` instance's points carry no non-`none` category, and a `confirmed` instance is never a review blob (review blobs have no real class by definition) | `instances_gt.json::confirmed` × `gt_point_category.npy` × `gt_class_ids.npy` |

All 9 are hard failures at save time — no warn-only mode for new saves. (See
§4 for why existing scans get migrated instead of grandfathered.)

### 3. Manifest fields

Merged into `gt_segment_metadata.json` on every successful save:

- class histogram (per-class point counts)
- per-region category histogram + review-budget check result
- `class_map_version`
- in-region fraction (labeled points inside any eval region ÷ total labeled)
- active thresholds used: review budget (0.03), size floor (5cm), edge-band
  width per scanner
- canonical spacing + native density
- per-region measured p50/p90 spacing + declared LOA band
- provenance chain (source scan lineage via `scan_schema.Registry`/derivation)

### 4. Legacy migration

Because invariants are hard failures everywhere (not warn-on-load), the three
precious pre-phase-2 scans need to pass before the gate ships, not after.
`scripts/migrate_eval_invariants.py` (mirrors `migrate_scan_v2.py` /
`promote_to_v3.py`), run once per scan:

- Backfills `gt_point_category.npy` with all `none` where absent.
- Backfills `gt_point_component_ids.npy` with component 0 per instance where
  absent (each pre-phase-2 instance becomes exactly one component — no
  fragment splitting is invented, since none was ever recorded).
- Regenerates `gt_segment_metadata.json` with the new manifest fields.
- Runs all 9 invariants against the result; a scan that still fails needs
  manual fixup (expected: none — the backfilled arrays are constructed to
  trivially satisfy invariants 1, 3, 5, 9; existing class/instance data is
  untouched, so 4 and 7 pass exactly as before).

Additive only: `working_*.npy` and every already-saved class/instance
assignment are byte-for-byte untouched, per the standing rule that labeled
scans are precious.

### 5. scan_schema dependency pin

Voxa's `requirements.txt` currently pins `scan-schema @
git+https://github.com/rHedBull/scan_schema.git@main` — no commit hash, so any
scan_schema change (this phase's included) lands on voxa's next
install/CI run with no explicit bump step. After this phase's scan_schema PR
merges, bump the pin to that merge commit's SHA. This becomes voxa's normal
dependency-bump discipline going forward, not a one-off exception — noted in
`docs/scan-schema.md` so it isn't silently dropped on the next scan_schema
change.

## Testing

`scan_schema` (pytest):

- `test_eval_invariants.py`: one test per invariant (pass case) + one per
  violation case (1–9 above), mirroring `test_invariants.py`'s style.
- `test_manifest.py`: `build_manifest` field-by-field on a synthetic session;
  drift detection (invariant 6) against a hand-edited stale manifest.

`voxa` backend (pytest):

- `segment_io`/save route: a session violating each of the 9 invariants gets
  422'd with that invariant identified; a clean session's save merges the new
  manifest fields into `gt_segment_metadata.json`; invariant 7 correctly
  ignores added ids and flags a renumbered one across two successive saves.
- `scripts/migrate_eval_invariants.py`: synthetic pre-phase-2 session fixture
  (no category/component arrays) migrates cleanly and passes all 9 invariants
  afterward; existing `gt_class_ids`/`gt_segment_ids` are unchanged byte-for-byte.

Manual verification: run the migration script against a copy (not the
original) of one precious scan's session directory and diff the untouched
arrays to confirm byte-for-byte preservation before running it for real.

## Risks / open points

- **Hard-fail-everywhere means the migration script is load-bearing before
  ship.** If it misses a case, save on a precious scan starts 422ing. Mitigated
  by running the migration against copies first and asserting exact invariant
  pass/fail before touching the real sessions.
- **Invariant 7 needs a prior save to compare against.** A scan's very first
  post-phase-3 save has nothing to diff — it trivially passes (no prior ids to
  have lost). This is correct but worth stating: the check only bites starting
  the *second* save after this phase ships.
- **Invariant 8's re-measurement cost.** Re-running `raw_sample_spacing` on
  every save if a region's points changed is a full nearest-neighbour pass;
  acceptable at today's ~3M-point test scans, revisit at phase 0b's ~100M
  canonical clouds (already flagged as a general phase-0b concern, not new
  here).
- **`instances_gt.json` as a save-time input, not just a display doc.** The
  backend save gate now has a correctness dependency on a frontend-owned file
  being present and current for the session being saved. If it's ever missing
  or stale relative to the point arrays, the gate must fail loudly (matching
  the existing `getAnnotation` "throw rather than empty-doc-fallback" rule)
  rather than silently skip invariants 3/7/9.
