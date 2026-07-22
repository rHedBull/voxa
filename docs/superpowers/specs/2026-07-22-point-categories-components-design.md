# Point Categories + Components — Design (eval-labeling phase 2)

**Date:** 2026-07-22
**Status:** Draft
**Upstream spec:** `engine/research/fable/docs/superpowers/specs/2026-07-20-eval-labeling-design.md`
(§ "Non-object taxonomy", § "Instance flags" — fragments, § "Tooling delta — Voxa" item 2)
**Predecessors:** phase 0 (primitive vocabulary, PR #43), phase 1 (eval regions, PR #44)

## Problem

Eval-grade labeling needs every in-region point accounted for on **three axes that
are never conflated**: semantic class, annotation status, instance identity. Voxa
today has two of them (`class_ids`, `instance_ids`). Annotation status is still
riding the class axis — the outlier-denoise feature (PR #41) writes class `unknown`
(id 6, "Exclude / Review") instances, which is exactly the failure mode the
upstream spec bans (`double` → `unknown` → back on the class axis), and is why the
phase-0 frozen-class guard carries a deliberate id-6 exemption.

Two per-point arrays are missing:

1. **`point_category`** — the non-object taxonomy: `artifact` (photons never hit a
   real surface), `transient` (person / self-mobile object), `excluded-review`
   (real, permanent, identity uncommittable). Confirmed-thing / confirmed-stuff
   stay *derived* from class+instance and are never stored.
2. **`point_component_ids`** — fragment membership *within* an instance, so
   instance metrics can score object-level (did the model link the pipe across the
   occluding column?) and fragment-level (did it segment each piece?) separately.

Without them the loader invariant "every in-region point is exactly one of
confirmed-thing ∨ confirmed-stuff ∨ artifact ∨ transient ∨ excluded-review" is
not expressible, and the 3% review budget per eval-grade region cannot be checked.

## Goals

- A third per-point axis in the working session, persisted, undoable, visible.
- Review blobs are **instance-shaped**, never a point spray (upstream invariant 3).
- Fragment components stored, derived deterministically — no new labeler work.
- Retire the class-id-6 write from denoise; drop the frozen-guard exemption.
- Wire the review budget into the phase-1 eval-grade gate.

## Non-goals

- The full loader-invariant suite + manifest generator — that is phase 3, in
  `scan-schema`, where voxa's save-gate and the harness's load-gate become one
  code path. This phase writes the arrays and records what phase 3 will check.
- Renaming voxa's output files to the upstream spec's names
  (`point_instance_ids.npy`, `instances.json`). Voxa's `output/gt_*` contract is
  consumed today; phase 3 owns that migration.
- Relabeling campaigns, edge-band derivation (eval-time geometry, not a label),
  manual component editing.
- Export/materialize changes. Category-marked points carry `class_id == -1`, so
  every export path already excludes them (verified below, not assumed).

## Design

### 1. The category array

`SegmentSession.categories`: `int8`, length == n_points, full-resolution, parallel
to `class_ids` / `instance_ids`.

| value | name | meaning |
|---|---|---|
| 0 | `none` | default — the point is unlabeled, or a confirmed thing/stuff (derivable from class+instance) |
| 1 | `artifact` | ghost/multipath/mixed-pixel — no real surface |
| 2 | `transient` | person or self-mobile object |
| 3 | `excluded_review` | real, permanent, identity uncommittable |

Constants live once in `backend/labeling/categories.py` (`CATEGORY_NONE`, …,
`CATEGORY_NAMES`, `parse_category`), mirrored by
`frontend/src/point-categories.js` — the same one-home-per-mapping pattern as
`class-chords.js` / `CLASS_GROUPS`.

`none` is the default so an untouched session, a legacy session with no
`working_categories.npy`, and a freshly loaded scan all agree without a migration.

### 2. One op, on the undo stack

New op `set_category` on `SegmentSession`, routed through the existing `_apply`
delta machinery so undo/redo works unchanged:

```
apply_category(indices, category, protect_instances=None) -> delta
```

Write rules — a category mark is a **statement about the points**, so it always
resolves the other two axes rather than layering over them:

| category | class_ids | instance_ids |
|---|---|---|
| `artifact` / `transient` | -1 | -1 |
| `excluded_review` | -1 | fresh instance id (one *review blob* per apply) |
| `none` (clear) | -1 | -1 |

Marking therefore *erases* whatever label those points carried — with
`protect_instances` honored exactly as on `apply_reassign` (confirmed = locked;
un-confirm before re-marking). A review blob gets its id from `_next_fresh_inst`,
so the stable-id contract the future relations layer depends on is preserved.

`_Delta` grows `before_cat` / `after_cat`; `undo()` / `redo()` restore all three
arrays; the wire delta grows `after_category` (b64 int8). Every other op
(`reassign`, `merge`, `set_class`, `snap_to_preseg`) **clears the category of the
points it writes** — labeling a point as a real object is by definition retracting
its non-object mark, and leaving a stale `artifact` on a labeled point would break
the "exactly one of five" invariant phase 3 checks. This mirrors the existing
unconditional `_retire_sam_ids(indices)` in `_apply`.

*Review blob = class-null instance.* Voxa's working arrays already tolerate
`instance_ids >= 0` with `class_ids == -1` (that is what an unclassified presegment
looks like), so no scan-schema change is needed for the working state. What
distinguishes a review blob from a preseg point is `categories == 3`.

### 3. Persistence

**Working (autosave, per-edit):** `sessions/<id>/working_categories.npy` (int8),
written by `save_session_aux` alongside `working_class_ids` / `working_segment_ids`
/ `working_sam_ids`, and loaded by a new `load_categories(session_dir, n_points)`
that returns `None` when absent (pre-phase-2 session → all `none`) and **raises**
on a shape mismatch — the `load_sam_ids` contract, for the same reason (a
misshapen file signals a foreign data dir, not an empty layer).

**Output (explicit save):** `PUT /api/segment/save` additionally writes

- `output/gt_point_category.npy` — int8, full-res, verbatim copy of the array
- `output/gt_point_component_ids.npy` — int16 (see §4)

and extends `gt_segment_metadata.json` with

```json
"categories": {"none": N, "artifact": N, "transient": N, "excluded_review": N},
"review_blobs": [{"instance_id": 12, "n_points": 431}],
"component_link_radius_m": 0.05
```

`gt_class_ids` / `gt_segment_ids` keep today's semantics: the existing save-path
strip of `(inst >= 0 & class == -1)` points also strips review blobs, so
`scan_schema.validate_invariants` (invariant 3: `class == -1 ⟺ inst == -1`) keeps
passing untouched. The blob ids are not lost — they are recorded in
`review_blobs` and their points carry category 3 in `gt_point_category.npy`.
Phase 3 replaces this pair with the upstream file set and enforces "every
`excluded-review` point belongs to a review blob" directly.

### 4. Components

`backend/labeling/components.py`:

```
component_ids(positions, instance_ids, link_radius=0.05) -> int16[n]
```

Component index **within** each instance (0-based per instance), `-1` for points
with `instance_id < 0`. Derived, never labeled: labelers link fragments by
*merging instances*, which is already a supported operation; geometry decides what
is one connected piece.

Algorithm: voxel-grid connectivity. Quantize each instance's points onto a grid of
cell size `link_radius`, union each occupied cell with its 26 neighbours
(union-find over a dict of occupied cells), then renumber roots per instance in
first-appearance order. O(n) in points and independent of local density — a
`cKDTree.query_pairs` at 5 cm over a million-point floor instance would generate
pair counts in the 10^8–10^9 range and is not viable at scan scale.

Two consequences, both accepted and documented: cell-quantization links points up
to `√3 · link_radius` apart along a cell diagonal (a coarser, deterministic link
rule than a metric ball), and points closer than `link_radius` can land in
non-adjacent cells only when farther than one cell apart — i.e. the rule is "same
or neighbouring 5 cm cell", stated plainly rather than approximated.

`link_radius` default **0.05 m** = the upstream 5 cm size floor; recorded in the
metadata so a later change is visible rather than silent. More than 32767
components in one instance raises `ValueError` (fail loudly — it means the radius
is wrong for the cloud, not that the data is exotic).

Computed **only on explicit save**, never on autosave: it is a GT artifact, and
paying a full-cloud pass on every brush stroke is not.

### 5. Denoise stops writing class 6

`_denoise_core` currently materializes its outliers as an unconfirmed pointset with
`EXCLUDE_CLASS_ID` (= `unknown`, archive id 6). It instead calls
`apply_category(outliers, CATEGORY_EXCLUDED_REVIEW, protect_instances=…)`,
producing a review blob. Consequences:

- `EXCLUDE_CLASS_ID` and the id-6 exemption in `app.core.reject_frozen_class`
  disappear; `unknown` becomes fully frozen, as phase 0 intended.
- The re-run replacement path is unchanged in shape (`replace_inst` still erases
  the previous blob's points first) but now also clears their category.
- The frontend's `runDenoise` row becomes a review-blob row (§6).

### 6. Frontend

**State.** `segState.categoryFull` (Int8Array), hydrated from `/api/load` +
`/api/segment/state` (`full_categories`, b64 int8) and updated by `applyDelta`
whenever the delta carries `after_category`. `initSegState` defaults it to zeros,
so an older backend or a legacy session degrades to "no categories" instead of
crashing.

**Marking.** `ClassPickerModal` — already the single "what is this selection?"
surface every tool reaches via Ctrl+Enter — grows a **Non-object** footer row:
`Artifact (a)`, `Transient (t)`, `Review (r)`, `Clear (x)`. These keys are live
only while no chord group is armed (armed member keys are letters, so there is no
collision), and `onPick` yields `{category:'artifact'}` instead of a class def.
Callers branch on the shape.

Reachable from: Presegment selection, SAM candidate selection, Box, Prism — the
four tools whose apply resolves to "these points". Draw and Beam stay class-only:
they are object-shaped tools (a tube, a beam OBB), and marking an artifact with
them is meaningless. `Clear (x)` exists so a mis-mark is fixable without relying
on undo, and so artifact/transient points (which have no Instances-panel row) are
never a dead end — re-select them with Box and clear.

**Backend calls.** Selection tools use `POST /api/segment/apply` with
`op:"set_category"` and `payload:{category}`; shape tools use
`POST /api/segment/apply-shape` with `target_category` in place of
`target_class` (mutually exclusive; supplying both is a 400).

**Viewport.** Category points recolor in Label mode's class color mode, exactly
like `SAM_CANDIDATE_COLOR` does for candidates: artifact `#ff4dd2`, transient
`#ff9f1c`, excluded-review `#9aa0a6`. Without this they would render as
unlabeled grey and be invisible as marks.

**Panels.** A one-line non-object counter in the left rail
(`Non-object: ✕ 1.2k · ⇢ 340 · ? 80`) from `categoryFull`. Review blobs appear as
ordinary Instances-panel rows with `cls: null` and a "Review" chip, so the
phase-0 `note` field, delete, and confirm all work on them unchanged.
`Cuboid.cls` becomes `Optional[str]` for exactly this case (null ⟺ review blob);
every consumer that reads `cls` for color/label falls back to the review grey and
the label "Review".

### 7. Review budget in the eval-grade gate

Phase 1's gate measures spacing only. It gains the upstream spec's second
condition: **`excluded_review` points ≤ 3% of the region's points**, else the flip
to `eval_grade` is refused (422, surfaced inline by the existing Regions-tab error
path). `region_stats` returns `n_review` per region so the tab can show the
budget before the user tries the flip. Constant `REVIEW_BUDGET_FRAC = 0.03` next
to `EVAL_GRADE_P90_M`; `flip_status` takes `categories` as a new argument.

The 100-point floor still applies first, so a tiny region cannot pass the budget
check vacuously.

## Testing

Backend (pytest):

- `categories.py`: name↔value round-trip, `parse_category` rejects garbage.
- `segment_state`: mark artifact erases class+instance; mark review allocates a
  fresh monotonic instance id; `protect_instances` blocks confirmed points;
  undo/redo restores all three arrays; a subsequent `reassign` over marked points
  clears their category.
- `components.py`: two clusters 1 m apart in one instance → 2 components;
  contiguous cluster → 1; points outside instances → -1; per-instance
  independence (two instances each get 0-based indices); >32767 components raises.
- `segment_io`: `working_categories.npy` round-trips; absent file → `None`;
  wrong shape → raises.
- save route: writes both output arrays, metadata carries the category histogram
  + review blobs + link radius, and `validate_invariants` still passes with review
  blobs present.
- denoise: produces a review blob (category 3, class -1, fresh instance id), never
  class 6; `reject_frozen_class` now refuses id 6 everywhere.
- regions: flip refuses over-budget review fraction; passes at exactly 3%;
  `region_stats` reports `n_review`.

Frontend (vitest):

- `point-categories.js` mirror test (values/names match the backend constants —
  pinned the same way `test_class_config.py` pins classes.yaml).
- `segment-state`: `applyDelta` writes `categoryFull` when present and leaves it
  untouched when absent; `initSegState` defaults to zeros.
- `class-picker` (jsdom): `a`/`t`/`r`/`x` yield `{category:…}` only when unarmed;
  armed group swallows them.
- `api.js`: `segApply('set_category')` and `applyShape({targetCategory})` shapes.

Browser verification (per the repo rule for UI changes, on a throwaway session):
mark a box selection artifact → points turn magenta, counter increments, undo
restores; mark review → blob row appears in the Instances panel; denoise produces
a review blob; region flip refused when the review budget is blown.

## Risks / open points

- **Erase-on-mark is destructive.** Marking a labeled point `artifact` drops its
  class. Mitigated by `protect_instances` (confirmed instances are locked) and by
  undo. The alternative — categories as an independent overlay — would let a point
  be both a confirmed pipe and an artifact, which is exactly the axis-conflation
  the upstream spec forbids.
- **Review blobs are stripped from `gt_segment_ids`** (§3). Information is
  preserved in the metadata + category array, but any consumer reading only
  `gt_segment_ids` sees review points as unlabeled. That is the correct reading
  today (they are not GT objects) and phase 3 formalizes it.
- **`Cuboid.cls` becoming optional** touches a widely-read field. Every read site
  gets an explicit review-blob branch rather than an implicit `|| ''`.
- Component ids are recomputed from scratch on every save (full-cloud pass). At
  ~3M points this is sub-second with the voxel approach; at the ~100M canonical
  labeling clouds phase 0b contemplates it needs re-measuring — noted, not
  pre-optimized.
