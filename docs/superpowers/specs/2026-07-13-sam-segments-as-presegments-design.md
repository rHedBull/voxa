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
Selection resolution (candidate ids → point indices) happens **client-side**,
from the frontend's own `samIds` array, exactly like the frontend already
derives indices from `instanceFull` for a preseg selection — no new backend
endpoint is needed for this; `segApply('reassign', ...)` already accepts raw
`indices`. For the frontend to build `samIds` in the first place, `/project`
must return each candidate's point indices (see Backend routes below).

**Persistence — new session file `sessions/<id>/sam_segments.json` +
`sam_ids.npy`:**

```json
{
  "segments": [
    {"id": 3, "n_points": 4211, "mask_score": 0.94, "created_at": "..."}
  ]
}
```

`materialize_sam_segment` is invoked backend-side (inside `/api/sam/project`),
so — unlike `centerlines.json`/`structure.json`, which are frontend-driven
edits written synchronously via dedicated `PUT` routes on explicit user
action — it has a natural hook into the session's existing per-mutation
autosave path: `SegmentSession.schedule_autosave` → `_do_autosave`
(`backend/labeling/segment_state.py:318-343`), which today writes
`session.json` (aux payload) plus `class_ids.npy`/`instance_ids.npy`. That
path (and its aux-payload builder, `save_session_aux`) needs to be
**extended** to also serialize `sam_ids.npy` and the `sam_segments.json`
summary whenever `materialize_sam_segment` runs. Loaded back into
`sam_ids`/`sam_segments` on session resume, so accumulated-but-unclassified
SAM candidates survive reload. Not part of `instances_gt.json` — candidates
aren't instances.

A materialized `sam_seg_id` whose points get fully classified (selection
applied) is removed from `sam_segments` and its `sam_ids` entries reset to
`-1` (mirrors presegments disappearing from `PresegmentList` via
`promotedSegIds` once absorbed into an instance).

## Backend routes (`backend/routes/sam.py`)

- `POST /api/sam/project` changes from *"apply_reassign per mask"* to
  *"materialize_sam_segment per accepted mask"*, with two schema changes to
  `SamProjectRequest`/its response (`backend/app/schemas.py:179-183`):
  - **Request**: `target_class` is **removed** — it's currently a required
    field (used to classify immediately); classification no longer happens
    at accept time, so nothing to send. Request becomes
    `{capture_id, mask_ids}` only (`protect_instances` threading unchanged).
    `frontend/src/sam-mode.jsx`'s `doProject` (currently sends
    `targetClass: defaultClassId`, lines 248-253) drops that field and the
    class-picking it implies.
  - **Response**: becomes `{segments: [{sam_seg_id, mask_id, n_points,
    scan_indices_b64}]}` — note `scan_indices_b64` (int32, base64, same
    encoding the *current* response already uses for `scan_indices_b64` per
    instance, `backend/routes/sam.py:88`) is **kept**, just attached to a
    candidate instead of an instance. This is load-bearing: it's the only way
    the frontend can populate `samIds` (see Frontend below) — the response
    can name candidates but the frontend still needs the actual point indices
    to color/select them. No `apply_reassign` call, no undo entry, no
    instance row.
- No new endpoint is needed to *classify* a SAM selection — the frontend
  calls the same generic `VoxaAPI.segApply('reassign', {indices, payload})`
  presegments already use, with `indices` resolved client-side from `samIds`
  instead of `instanceFull`.
- Everything else in `2026-07-12-sam-labeling-tool-design.md` — `/capture`,
  sidecar identity/fingerprint handling, coordinate handling, error handling
  — is unchanged.

## Frontend

**State (`frontend/src/segment-state.js`):**

- `segState.samIds: Int32Array` — full-res, default `-1`, parallel to
  `instanceFull` but never merged into it. Patched incrementally from
  `/project` responses (decode each candidate's `scan_indices_b64`, write
  `sam_seg_id` at those positions — same decode-and-scatter the frontend
  already does for `applyDelta`). On full reload/session-resume, hydrated
  from a new `full_sam_ids` field added to `SegmentStateResponse`
  (`backend/app/schemas.py:185-208`) — the same base64-`Int32Array` encoding
  as the existing `full_instance_ids`, populated from
  `SegmentSession.sam_ids`. This does **not** go through
  `hydrateFromServerState` (`frontend/src/segment-state.js:47`) — that
  function only merges scalar metadata (`presegRunId`, `dirty`, etc.) onto an
  already-built `segState` and never touches point arrays. The actual
  decode/threading follows the existing `full_class_ids`/`full_instance_ids`
  path instead: `VoxaAPI.segState()` (`frontend/src/api.js:192-208`) decodes
  `full_sam_ids` alongside them, `initSegState` (`segment-state.js`) gets a
  new `samIds` param (defaulting `samSelection`/`samSegments` too), and both
  `App.jsx` call sites that build/rebuild `segState` (lines 262-272 and
  294-301) thread the decoded array through. `sam_segments` metadata for the
  list rows comes along in the same `SegmentStateResponse` (a `sam_segments`
  field mirroring the shape in `sessions/<id>/sam_segments.json`).
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
- New function `confirmSamSelection` (`mode-label.jsx`, sibling of
  `confirmSegmentSelection`, lines 641-691), same shape: resolves indices
  from `samIds` for every id in `samSelection`, calls
  `VoxaAPI.segApply('reassign', {indices, payload:{target_inst:-1,
  target_class}})` → new instance row `{..., source:'sam', confirmed:
  autoConfirmFor('sam')}`, then `reconcileSamAfterApply` (instead of
  `applyDelta`+clearing `selection`).
- Three existing call sites in `mode-label.jsx` decide/perform the
  apply-on-classify flow, and **all three** need a SAM-aware branch, keyed on
  `activeTool === 'sam' && segState.samSelection.size > 0` (checked first,
  same "gate on active tool, not just selection size" rule as below), or the
  feature is only half-wired:
  1. **Line 894** — the Ctrl+Enter gate deciding whether to open
     `ClassPickerModal` at all (currently
     `segState.selection.size > 0 || (activeTool === 'box' && selBox)`). Needs
     `|| (activeTool === 'sam' && segState.samSelection.size > 0)` or Ctrl+Enter
     on a SAM selection never opens the picker.
  2. **Line 913** — the direct class-hotkey path that calls
     `confirmSegmentSelection` immediately, bypassing the picker. Needs the
     equivalent SAM branch calling `confirmSamSelection` instead.
  3. **Lines 973-977** — `ClassPickerModal`'s `onPick` callback (currently
     `if (activeTool === 'box' && selBox) applyBox(cls); else
     confirmSegmentSelection(cls);`), which is where the Ctrl+Enter path's
     actual apply call happens. Needs a third case:
     `else if (activeTool === 'sam' && segState.samSelection.size > 0)
     confirmSamSelection(cls); else confirmSegmentSelection(cls);`.
  Gating each of these on `activeTool` (not just selection size) is what
  implements "only the active tool's selection applies" — if the user
  switches away from the SAM tool without clearing `samSelection`, none of
  the three sites route to the SAM path anymore, so a subsequent Ctrl+Enter
  on a leftover presegment `selection` behaves normally and the stale
  `samSelection` is left untouched (not cleared, not applied) until the user
  switches back to the SAM tool.

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
  `segments` (with `scan_indices_b64`, no `target_class` in the request) not
  `instances`, and does not call `apply_reassign`; `SegmentStateResponse`
  includes `full_sam_ids`/`sam_segments` and round-trips through
  `_resume_session`.
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
