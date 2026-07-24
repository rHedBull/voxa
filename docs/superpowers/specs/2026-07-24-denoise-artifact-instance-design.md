# Detect-outliers → unconfirmed `artifact` instance

## Problem

The **✧ Detect outliers** tool (global statistical-outlier removal, Feature C in
`2026-07-20-outlier-detection-filtering-design.md`) materializes the strays it
catches as an **`excluded_review` review blob** — a grey, class-less instance on
the annotation-status axis (`backend/routes/segment.py::_denoise_core`,
`frontend/src/mode-label.jsx::runDenoise`).

This is the wrong category for sensor noise. The labeler decision guide
(`engine/research/fable/docs/superpowers/specs/2026-07-21-labeler-decision-guide.md`)
is explicit:

- **Node 1 — "Real surface? Did photons bounce off something physical here?"**
  → *no: ghost / multipath / range fuzz* → **`artifact` — label it, never delete it.**
- **`excluded_review`** is reserved for something else entirely: *real, ≥5 cm,
  a nameable unit you can't name* → blob + note, **max 3% of a region**. It is a
  budgeted escape valve for real geometry, not for noise.

Spatially-isolated strays (speckle / multipath / range fuzz) are exactly node 1's
"no real surface" case. They should be `artifact`, not `excluded_review`. Using
`excluded_review` also wrongly consumes the 3% review budget that the save-time
eval-invariant gate enforces.

## Goal

Detect-outliers produces an **unconfirmed `artifact` instance** as its outcome: a
magenta, confirmable Instances-panel row (parallel to today's grey Review blob but
on the `artifact` category). Confirming it hides its points, which then drop out of
the next run's population — leaving it unconfirmed keeps it visible and eligible.
At save it collapses to the guideline's canonical artifact representation
(`class −1, instance −1, category=artifact`); the instance id is a session-only
review handle.

Scope is deliberately narrow: **only the detect-outliers tool** mints an artifact
*instance*. Hand-marking `artifact` from the class picker's Non-object row keeps its
existing semantics (erase to `(−1, −1)`, no instance, no panel row).

## Non-goals

- No change to the per-selection **Remove outliers** feature (Feature B,
  `/denoise-selection`) — it strips strays back to unlabeled and never
  materializes a category. Untouched.
- No change to hand-marking `artifact`/`transient`/`excluded_review` from the class
  picker.
- No new SOR population model beyond what confirmed-hiding already gives. Re-running
  after a slider change still erases the prior instance by id (`replace_inst`);
  re-running after confirming excludes the confirmed points because they are hidden
  and already protected. We do **not** add an explicit "visible points" population
  parameter — it falls out of the existing confirmed-instance protection.

## Design

### 1. Backend — allocate an artifact blob (denoise only)

The blob-allocation mechanism already exists in
`SegmentSession.apply_category` (`backend/labeling/segment_state.py`). Today the
blob flag is hardwired to the review category:

```python
out = self._apply("set_category", indices, dict(
    category=cat, blob=(cat == CATEGORY_EXCLUDED_REVIEW)))
```

Generalize so a caller can request a blob for `artifact` too. Add an explicit
`allocate_instance: bool | None = None` parameter to `apply_category`:

- `None` (default) → today's behavior: `blob = (cat == CATEGORY_EXCLUDED_REVIEW)`.
  Every existing caller (class-picker marks) is unchanged.
- `True` → force a blob (used by denoise for `artifact`).
- `False` → force no blob.

`_denoise_core` (`backend/routes/segment.py`) changes its one materialize line from
`CATEGORY_EXCLUDED_REVIEW` to:

```python
from labeling.categories import CATEGORY_ARTIFACT
out = seg.apply_category(
    outliers.astype(np.int32), CATEGORY_ARTIFACT,
    allocate_instance=True, protect_instances=req.protect_instances)
```

The `replace_inst` re-run path (`backend/routes/segment.py::_denoise_core`, lines
~188-191) is **unchanged**: an artifact instance carries an id, so erase-by-id on a
strength re-run works exactly as before. The response contract
(`DenoiseResponse{instance_id, n_affected, n_protected, scan_indices_b64, dirty}`)
is unchanged — `instance_id` is now an artifact-blob id instead of a review-blob id.

Undo/redo: unchanged. `set_category` already rides the undo stack, and the
allocated instance id is session-monotonic (`_next_fresh_inst`) whether the category
is review or artifact.

### 2. Frontend — the denoise row is an "Artifact" instance

`frontend/src/mode-label.jsx::runDenoise` currently builds a `reviewBlobRow` and
writes `CATEGORY_EXCLUDED_REVIEW` into the `applyDelta` category array. Change both:

- The category delta fills `CATEGORY_ARTIFACT`.
- The panel row is an **artifact blob row** — class-less like the review row, but
  labelled "Artifact" and coloured magenta (`#ff4dd2`, the artifact category color
  from `point-categories.js`) instead of grey "Review".

Introduce a small helper alongside `reviewBlobRow` (e.g. `artifactBlobRow(segId,
source)`) rather than overloading the review row, so the two category-blob kinds stay
readable. Both are class-less rows (`cls: null`) that the Instances panel already
knows how to render; the only difference is label + color + which category they carry.

The row's `source` stays `'denoise'`. Confirm/reopen, focus, rename, delete, and the
re-run/replace wiring (`denoiseInstId` → `replace_inst`) are all inherited unchanged.

### 3. Confirmed-artifact-hide fix (bundled companion fix)

The "confirm → disappears → excluded next run" behavior depends on a pre-existing
bug: **category-overlay points do not hide when their instance is confirmed.**

Root cause (already diagnosed): category-marked points carry `class −1`, so they are
painted by a *separate* overlay geometry via `buildCategoryOverlay`
(`mode-label.jsx:275` → `viewer.jsx::setSelectedSegmentMask`), keyed purely on
`segState.categoryFull`. The confirmed-hide pass (`viewer.jsx:944`, driven by
`confirmedPointsetHideMask`) NaNs the *base cloud* positions but never touches the
overlay geometry — and the overlay-rebuild effect's deps
(`[segState, cloud, viewerRef, fastMode, activeTool]`) don't include `instances`, so
confirming doesn't even recompute it. Net effect today: confirming a review blob (or,
after this change, an artifact blob) hides its base-cloud RGB points but leaves the
category-overlay dots showing.

Fix in the category-overlay effect (`mode-label.jsx`, the effect ending at line 340):
when `hideConfirmed` is on, drop overlay `mask[p]` for any `p` where
`confirmedPointsetHideMask[p] === 1`, and add `hideConfirmed` +
`confirmedPointsetHideMask` to the effect's dependency array so it recomputes on
confirm. `confirmedPointsetHideMask` already excludes the *selected* instance
(`mode-label.jsx:826`), so re-selecting a confirmed blob brings its dots back —
consistent with how the base-cloud hide already behaves.

This fix is not artifact-specific; it also fixes the identical symptom on
`excluded_review` review blobs (the reported "Review #41 doesn't hide" case).

### 4. Save / GT semantics — strip the instance id to −1

The labeler guide requires artifact points to be `instance −1` in final GT. The
existing save path already handles this correctly *for free* via its class-less
strip: an artifact blob is `class −1`, so at save its instance id is stripped to −1
in `gt_segment_ids.npy`, leaving `(class −1, instance −1, category=artifact)`.

Concretely:

- `review_blob_summary` (`backend/labeling/segment_io.py:112`) records only
  `(category == EXCLUDED_REVIEW) & (inst >= 0)` into the `review_blobs` metadata
  table. Artifact blobs are **not** captured there — correct, they are not review
  blobs and must not survive as GT instances.
- Because the artifact blob's id is stripped from `gt_segment_ids` and is not in the
  `review_blobs` universe, **eval-invariant 3** (every id in `gt_segment_ids` has an
  `instances_gt.json`/`review_blobs` entry) never sees it → passes.
- `gt_point_category.npy` records `artifact` for those points (unchanged save code —
  it already persists the full category array).
- The `categories` histogram in `gt_segment_metadata.json` counts them under
  `artifact`; they do **not** count against the `excluded_review` 3% budget
  (eval-invariant 2).

Verify (do not assume) that the class-less strip actually nulls the artifact blob's
instance id at save — if the current strip only special-cases review blobs rather
than stripping *all* class-less instances, extend it to strip class-less artifact
instances to −1 as well. This is the one save-path spot to confirm during
implementation; the invariant behavior above is contingent on it.

Working-array persistence (`working_categories.npy`, `working_*.npy`) is unchanged:
the artifact blob's `(class −1, instance=id, category=artifact)` round-trips across
reload like any other instance, so the confirmable row survives a session resume.

## Data flow

```
run detect-outliers (σ)
  → POST /api/segment/denoise {std_ratio, k, replace_inst?, protect_instances}
  → _denoise_core: SOR over cloud (cached KD-tree)
      → apply_category(outliers, CATEGORY_ARTIFACT, allocate_instance=True,
                       protect_instances=…)   # blob id allocated, confirmed protected
  → DenoiseResponse{instance_id, indices, …}
  → runDenoise: applyDelta(category=ARTIFACT, instance=id, class=−1)
             + append artifactBlobRow(id) to Instances panel (magenta, unconfirmed)
  → viewport: points recolor magenta via category overlay

confirm the artifact row
  → confirmedPointsetHideMask picks up its segId
  → base cloud NaNs those points (viewer.jsx:944)
  → category overlay now ALSO drops them (fix §3)
  → points disappear

re-run at a new σ (unconfirmed row present)
  → replace_inst = prior blob id → erase-by-id → recompute → new blob
re-run after confirming (catch a new area)
  → confirmed points hidden + protected → excluded from the new catch

save (Ctrl+S)
  → gt_segment_ids: artifact blob id stripped to −1  (class-less)
  → gt_point_category.npy: artifact for those points
  → eval invariants 1–9 pass; not in review_blobs; no 3% budget hit
```

## Error handling

- Denoise catching only confirmed/locked points → `apply_category` drops them via
  `protect_instances`; if nothing remains, `_denoise_core` returns the existing empty
  response (`instance_id: null`), and `runDenoise` drops the prior row. Unchanged.
- `apply_category(allocate_instance=False)` on `excluded_review` (should not happen
  from any caller) is still well-defined: no blob, category written, points end
  class-less/instance-less. We do not add a guard; the parameter is internal.

## Testing

**Backend** (`backend/tests/`):
- `apply_category(idx, 'artifact', allocate_instance=True)` allocates a fresh
  session-monotonic instance id, sets `category=artifact`, `class=−1`,
  `instance=id`; `allocate_instance=None` keeps today's erase-to-`(−1,−1)` for
  artifact and blob-for-`excluded_review`.
- `_denoise_core` returns an artifact-blob `instance_id`; the flagged points carry
  `category=artifact`; `replace_inst` erases the prior artifact blob before
  recomputing.
- Save round-trip: after a denoise + save, `gt_segment_ids` has `−1` at the flagged
  points, `gt_point_category.npy` has `artifact` there, `review_blobs` is empty of
  them, and all 9 eval invariants pass (invariant 2 budget unaffected).

**Frontend** (`frontend/src/`):
- `runDenoise` appends a magenta, class-less "Artifact" row and writes
  `CATEGORY_ARTIFACT` into the delta (mirror the existing review-blob test).
- Category-overlay effect: with `hideConfirmed` on, a confirmed category blob's
  points are dropped from the overlay mask; re-selecting it restores them. (Covers
  both the artifact blob and the review-blob regression.)

## Files touched

- `backend/labeling/segment_state.py` — `apply_category` gains `allocate_instance`.
- `backend/routes/segment.py` — `_denoise_core` marks `CATEGORY_ARTIFACT` blob.
- `backend/labeling/segment_io.py` — confirm/extend the class-less strip for artifact
  instances (only if not already generic).
- `frontend/src/mode-label.jsx` — `runDenoise` writes artifact + `artifactBlobRow`;
  category-overlay effect suppresses confirmed-hidden points.
- Tests in `backend/tests/` and `frontend/src/`.
- Docs: update `CLAUDE.md`'s Outlier-filtering and Point-categories notes, and the
  `2026-07-20-outlier-detection-filtering-design.md` Feature C description.
