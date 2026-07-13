# SAM segments as a presegment-like candidate layer — design

Date: 2026-07-13
Status: designed (supersedes the persistence/apply decisions in
`2026-07-12-sam-labeling-tool-design.md`; everything else in that spec —
sidecar, capture/project endpoints, coordinates, scope boundaries — is
unchanged)

## Problem

The SAM tool (`docs/superpowers/specs/2026-07-12-sam-labeling-tool-design.md`)
is implemented: Shift-drag a box or type a concept prompt, SAM returns masks,
the review modal shows them, and accepting a mask calls `/api/sam/project` →
`apply_reassign(target_class=...)` directly — each accepted mask becomes an
**unconfirmed pointset instance immediately**, with a class already attached.

That collapses two decisions the other tools keep separate: *which points did
I select* and *what class are they*. Every other selection mechanism in Label
mode — most importantly Presegment — lets you accumulate a pool of candidate
segments, look at them, multi-select across several, and only then classify.
SAM masks should work the same way: a captured/accepted mask is a *candidate
segment*, an artifact of the SAM tool, not yet a label. It should sit in its
own list (shown only while the SAM tool is active, analogous to how the
presegment hulls/list are presegment-tool-only), be selectable and
multi-selectable there or by clicking the recolored points in the viewport,
and only become a real pointset instance when the user classifies the
selection — exactly the presegment flow, reusing the same
`select → apply+label → unconfirmed pointset → confirm` pipeline principle
the whole app is built on.

## Why SAM segments can't just reuse `instance_ids` directly

The obvious shortcut — give an accepted mask a fresh id directly in the live
`instance_ids` array, leaving `class_ids` at -1 until classified — is blocked
by **SCHEMA invariant 3: `class == -1 ⟺ instance == -1`**
(`backend/labeling/seg_inference.py:219`, enforced by
`scan_schema.invariants.validate_invariants` at save). A point with a real
instance id and no class would fail validation the moment the session
autosaves mid-review (SAM apply already autosaves today — see
`feedback_browser_verify_mutates_session`). `apply_reassign` itself already
guards this: a non-erase reassign without `target_class` raises
(`backend/labeling/segment_state.py:244-247`).

Presegments never hit this because prelabel's own pipeline assigns every
cluster a real (if rough) `class_id` before voxa ever sees it — segments start
pre-classified, and "applying" a presegment selection is really
*re*classifying. SAM masks have no such class at capture time, so they need a
data layer that can hold "these points form a candidate group" without ever
writing into `instance_ids`/`class_ids`.

Presegments already have the right shape for this: `SegmentSession.preseg_ids`
(`backend/labeling/segment_state.py:59`) is exactly such a layer — immutable,
separate from `instance_ids`/`class_ids`, used for snap-to-preseg. SAM
segments get an analogous, mutable, session-scoped layer.

## Data model

**Backend (`backend/labeling/segment_state.py`):**

- `SegmentSession.sam_ids: np.ndarray` — full-res `int32`, default `-1`,
  parallel to `preseg_ids` but **mutable** and **session-scoped** (accumulates
  over the session's life; not seeded at session creation).
- `SegmentSession.sam_segments: dict[int, dict]` — in-memory candidate
  metadata, `{sam_seg_id: {n_points, created_at, mask_score}}`, mirroring the
  shape of a preseg's `segment_summary.json` entry minus `class_id`.
- New method `materialize_sam_segment(indices, protect_instances=None) -> dict`:
  drops any already-confirmed points (`protect_instances`, same rule as
  `apply_reassign`), allocates a fresh `sam_seg_id` (own counter, independent
  of `_next_fresh_inst` — these ids never need to collide-check against real
  instance ids since they live in a different array), writes it into
  `sam_ids[indices]`, records the summary entry. Points already carrying an
  older `sam_ids` value are overwritten (**last materialize wins** — no
  dedup/exclusion logic, matching how `apply_reassign` already treats
  overlapping writes elsewhere). **Not** on the undo stack (matches
  `preseg_ids`: this is a selection aid, not an edit) and does not itself
  trigger autosave-of-arrays (only `sam_ids`/`sam_segments.json`, tracked
  separately from the `dirty` working-array flag).
- New method `resolve_sam_selection(sam_seg_ids) -> np.ndarray`: returns the
  union of point indices for the given candidate ids (mirrors how the
  frontend today derives indices from `instanceFull` for a preseg selection,
  but server-side isn't actually needed for this — see Frontend below; kept
  here only if the apply call ends up needing server-side resolution instead
  of client-computed indices. **Decision: frontend resolves indices from its
  own `samIds` array**, same as it already does for `instanceFull`, so no new
  backend endpoint is needed for selection resolution — `apply-shape`'s
  sibling `segApply('reassign', ...)` already accepts raw `indices`.)

**Persistence — new session file `sessions/<id>/sam_segments.json` +
`sam_ids.npy`:**

```json
{
  "segments": [
    {"id": 3, "n_points": 4211, "mask_score": 0.94, "created_at": "..."}
  ]
}
```

Written on the same autosave debounce as other session state (mirrors
`centerlines.json`/`structure.json` — geometry/candidate state persisted
separately from the working-array save). Loaded back into
`sam_ids`/`sam_segments` on session resume, so accumulated-but-unclassified
SAM candidates survive reload. Not part of `instances_gt.json` — candidates
aren't instances.

A materialized `sam_seg_id` whose points get fully classified (selection
applied) is removed from `sam_segments` and its `sam_ids` entries reset to
`-1` (mirrors presegments disappearing from `PresegmentList` via
`promotedSegIds` once absorbed into an instance).

## Backend routes (`backend/routes/sam.py`)

- `POST /api/sam/project` changes from *"apply_reassign per mask"* to
  *"materialize_sam_segment per accepted mask"*: same input
  (`{capture_id, mask_ids}`), same `protect_instances` threading, but the
  response becomes `{segments: [{sam_seg_id, mask_id, n_points}]}` instead of
  `{instances: [...]}`. No `apply_reassign` call, no undo entry, no instance
  row.
- No new endpoint is needed to *classify* a SAM selection — the frontend
  calls the same generic `VoxaAPI.segApply('reassign', {indices, payload})`
  presegments already use, with `indices` resolved client-side from `samIds`
  instead of `instanceFull`.
- Everything else in `2026-07-12-sam-labeling-tool-design.md` — `/capture`,
  sidecar identity/fingerprint handling, coordinate handling, error handling
  — is unchanged.

## Frontend

**State (`frontend/src/segment-state.js`):**

- `segState.samIds: Int32Array` — full-res, default `-1`, loaded/patched from
  `/project` responses and full reloads, parallel to `instanceFull` but never
  merged into it.
- `segState.samSegments: Map<samSegId, {nPoints, maskScore}>` — parallel to
  `summary`.
- `segState.samSelection: Set<samSegId>` — parallel to `selection`, entirely
  independent (a SAM candidate and a presegment can never appear in the same
  selection).
- `applySamDelta(state, {ids, segId})` — the SAM-layer analogue of
  `applyDelta`, patches `samIds` in place.
- `reconcileSamAfterApply(state, appliedSegIds)` — removes classified
  candidates from `samSegments`/clears their `samSelection` membership and
  resets their `samIds` entries to `-1` once the backend confirms the apply
  (mirrors `reconcilePointsetRows`'s dormant-on-apply handling, simpler since
  SAM candidates have no undo/redo lifecycle).

**UI (`frontend/src/mode-label.jsx`, `frontend/src/segment-tools.jsx` or a new
`sam-segment-list.jsx`):**

- A **SAM segment list** panel, structurally a sibling of `PresegmentList`
  (same row layout, same plain-click-selects / Ctrl-Shift-click-multi-selects
  semantics) but a **separate component and a separate list** — SAM
  candidates and presegments are never mixed in one panel or one selection
  set, per your explicit call. Shown in the bottom-left only while
  `activeTool === 'sam'`.
- Viewport: while the SAM tool is active, points whose `samIds` entry is
  live get a distinct-hue recolor (point recolor only — no hull mesh, per
  your call), computed and pushed to the viewer the same way
  `setSelectedSegmentMask`/presegment coloring already works today, gated on
  `activeTool === 'sam'` exactly like `segHulls`/`segBoxes` are gated on
  `isPreseg`.
- Ctrl/Shift-click on a recolored point in the viewport toggles
  `samSelection` via the same `viewer.onPointerPick` handler presegments use
  today, branching on which layer (`instanceFull` vs `samIds`) has a hit at
  that point.
- Ctrl+Enter / class hotkey, when `samSelection` is non-empty, resolves
  indices from `samIds` (same pattern as `confirmSegmentSelection` resolving
  from `instanceFull`) and calls the existing
  `VoxaAPI.segApply('reassign', {indices, payload:{target_inst:-1,
  target_class}})` → new instance row `{..., source:'sam', confirmed:
  autoConfirmFor('sam')}`, then `reconcileSamAfterApply`. If both a
  presegment selection and a SAM selection are somehow non-empty at once (tool
  switched mid-selection), only the active tool's selection applies — the
  inactive one is left untouched, not silently cleared.

**`SamReviewModal` (`frontend/src/sam-mode.jsx`) change:**

The modal's mask-picking UX (image + swatched list, click-to-select,
Ctrl/Cmd/Shift+click to multi-select) is unchanged — that's still *capture
review*, deciding which of SAM's proposed masks are worth keeping at all.
What changes is what "accept" does: instead of calling `/project` and
immediately getting classified instances back, accept calls `/project` and
the returned masks are added to the accumulated SAM segment list/layer above.
Classification is a separate, later step done from the list or viewport, same
as presegments. The modal closes on accept either way; there's no change to
when/how a capture is triggered (Box shift-drag / Concept text prompt).

## Data flow (revised)

**Box:** pick SAM tool → shift-drag box → `/capture` → review modal shows the
one mask → accept → `/project {mask_ids:[0]}` → `materialize_sam_segment` →
one new row in the SAM segment list, points recolored in the viewport → user
Ctrl-clicks it (or it's the only one, single-clicks) → class hotkey →
`apply_reassign` → unconfirmed pointset, row disappears from the SAM list.

**Concept:** pick SAM tool → type "pipe" → Segment all → `/capture` → review
modal shows N masks → user accepts several → `/project {mask_ids:[...]}` →
N new SAM-segment-list rows → user multi-selects across several (e.g. picks 3
of the 5 that are really pipes) → one class hotkey → `apply_reassign` for the
union → one unconfirmed pointset absorbing all selected candidates (matches
today's multi-select-presegments-then-classify-together behavior) → the 3
rows disappear, 2 remain as candidates for a future capture/selection.

**Across captures:** a second Shift-drag box that overlaps a still-unclassified
candidate from an earlier capture overwrites those points' `sam_ids` entry
(last materialize wins) — the old candidate's `n_points`/summary shrinks or
disappears accordingly; no explicit merge/dedup step.

**Invariant preserved:** `/capture` never mutates voxa state. `/project` now
mutates only the `sam_ids` candidate layer, not `instance_ids`/`class_ids`.
Only classifying a SAM selection (Ctrl+Enter/hotkey) touches the working
arrays, via the same `apply_reassign` path every other tool uses — no new
label-write code path.

## Scope boundaries (unchanged from the parent spec, plus)

- Still single-view v1: no multi-view merge, no "everything" mode, no video
  tracking, no click-point prompts (see parent spec).
- SAM candidates have **no OBB** — same as before, they NN-transfer to raw
  density on export like other point-set instances.
- No undo/redo for `materialize_sam_segment` or for selecting/deselecting SAM
  candidates (matches presegment selection, which also isn't on the undo
  stack — only the eventual `apply_reassign` is).
- Deleting/discarding an unwanted SAM candidate without classifying it: out
  of scope for this spec's first cut — candidates simply persist until either
  classified or overwritten by a later overlapping capture. (Flag: if this
  turns out to be needed in practice — e.g. a junk mask with no future
  overlap sitting in the list forever — a "discard" affordance on the list row
  is the natural follow-up, not designed here.)

## Error handling (fail loudly, unchanged principles)

- `materialize_sam_segment` with `protect_instances` dropping every point
  (fully inside confirmed geometry) → no candidate created, `n_protected`
  returned, same shape as `apply_reassign`'s all-protected case.
- Session resume with a `sam_segments.json`/`sam_ids.npy` size mismatch
  against the current point count → fail loudly (same posture as other
  working-array load mismatches), not a silent reset.

## Testing

- **Backend:** `materialize_sam_segment` unit tests (fresh id allocation,
  overlap/last-write-wins, `protect_instances` dropping, no undo-stack entry);
  `sam_segments.json`/`sam_ids.npy` round-trip on session save/resume;
  `/api/sam/project` route test (mocked sidecar) asserting it now returns
  `segments` not `instances` and does not call `apply_reassign`.
- **Frontend:** `applySamDelta`/`reconcileSamAfterApply` pure-fn tests; SAM
  segment list row click → `samSelection` toggling (mirrors existing
  `PresegmentList` row-click tests); classify-from-SAM-selection resolves
  indices from `samIds` and calls `segApply('reassign', ...)` with the right
  payload.
- **End-to-end:** browser-verification on a throwaway session — capture,
  accept 2+ masks across 2 captures (one overlapping), confirm both appear as
  separate candidates in the SAM list and recolor in the viewport, multi-select
  and classify, confirm the resulting instance and the list correctly
  emptying; reload the session and confirm unclassified candidates survive.

## Future (not this spec)

- A "discard candidate" affordance (see Scope boundaries).
- Multi-view merge, still deferred per the parent spec — this candidate layer
  is a natural home for it later (accumulate masks from several camera poses
  into the same `sam_ids` layer before classifying), but that's not designed
  here.
