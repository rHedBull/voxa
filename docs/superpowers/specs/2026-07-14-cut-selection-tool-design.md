# Cut-selection tool — design

Date: 2026-07-14
Status: designed

## Problem

Every Label-mode tool today is purely additive: you select points and they
either become a new group (presegment click, SAM mask) or get absorbed into
one (Box/Draw/Beam apply). There is no way to take *back* a subset of points
from an existing selection — to split a segment/instance into two because it
straddles two real-world objects, or to shave a handful of outlier points off
before classifying. The only "remove" affordance today is Set-level: Ctrl-click
again deselects a whole segment id from `segState.selection` — id-granularity,
no geometry.

This spec adds a **cut** operation: pick one or more existing segments
(presegments, SAM candidates, or — singly — one instance), open an isolated
view of just those points, box-select a sub-region, and split it out into a
new segment/instance of the same kind as its original source. Nothing about
the cloud itself changes — cutting only ever re-partitions an existing
selection into more, smaller selections (or, for the instance case, a new
classified-but-unconfirmed instance). It reuses the
`select → apply+label → unconfirmed pointset → confirm` pipeline principle the
whole app is built on: cutting is *another way to select points*, not a new
kind of edit.

## Why cut results can't just be one merged blob

If the input selection spans multiple source segments (e.g. presegment A +
presegment B, multi-selected together), a single box drawn across both in the
isolated view will enclose points that originally belonged to different
segments. The cut must **partition by original source**, not merge: cutting
across A and B produces two new segments — one holding the cut-out A points,
one holding the cut-out B points — never a single blob mixing provenance from
two different segments. This is a hard requirement, confirmed explicitly: "in
this case there would be 2 new segments created, new cut out segment points of
A and then B."

## Why cut results for presegments/SAM can't become instances directly

The obvious shortcut — give the cut-out points a fresh id directly in
`instance_ids` — is blocked by the same rule the SAM-candidate design hit:
**SCHEMA invariant 3: `class == -1 ⟺ instance == -1`**
(`backend/labeling/seg_inference.py:219`), enforced by
`apply_reassign`/`_apply` (`backend/labeling/segment_state.py`, "Non-erase
reassign must specify a class"). A cut-out chunk of an unclassified
presegment/SAM segment has no class yet, so it needs the same kind of
selection-only home SAM candidates already have — not a write into
`instance_ids`.

Cutting a single **instance**, by contrast, doesn't hit this: the source
instance already has a class in `class_ids`, so the cut-out chunk can be
handed straight to `apply_reassign(indices, target_inst=-1,
target_class=<source class>)` and land as a new unconfirmed pointset instance
with no picker interruption — the class is already known, cutting just
divides where it lives.

## Generalizing the mutable candidate layer

`sam_ids`/`sam_segments` (added in
`2026-07-13-sam-segments-as-presegments-design.md`) is already exactly "a
mutable, session-scoped layer that holds candidate point groups without ever
writing `instance_ids`/`class_ids`." `preseg_ids` itself cannot be extended —
it's immutable, sourced once from the prelabel pipeline on disk. There is no
existing generalized "segment" interface shared by preseg/SAM; they remain two
structurally separate arrays merged only for display.

Rather than invent a second parallel mutable layer, this spec **generalizes
the existing one**: `sam_segments` entries gain a `source: 'sam' | 'preseg'`
tag, and `mask_score` becomes optional/generic metadata (SAM-specific, not
structurally required). A cut-out chunk of a presegment materializes into this
same `sam_ids`/`sam_segments` layer tagged `source: 'preseg'`; a cut-out chunk
of a SAM candidate materializes tagged `source: 'sam'`. Display routing keys
off the tag: `source: 'preseg'` entries render as new rows in the
**Presegment list**; `source: 'sam'` entries render in the **SAM segment
list** exactly as today. The two lists/selections remain separate, matching
the existing "a SAM candidate and a presegment can never appear in the same
selection" rule — a cut only ever produces more entries in whichever list its
source already belonged to.

**No hull for cut results, regardless of tag.** Hull geometry is computed
server-side only from `instance_ids`/the static prelabel
`segment_summary.json` (`backend/labeling/segment_hulls.py`); nothing
computes a hull from `sam_ids`. Cut results follow the exact treatment SAM
candidates already get today — **point-recolor only, no hull mesh**
(`2026-07-13-sam-segments-as-presegments-design.md`'s explicit call) — even
when tagged `source: 'preseg'` and shown in the Presegment list. A
`source: 'preseg'` row is styled with a presegment-like hue and appears in
the Presegment list/selection, but renders as recolored points, not a hull.
Consequently, cutting never shrinks the *original* presegment's displayed
hull or point count — that data comes from the static, immutable prelabel
summary and is genuinely unaffected by a cut (the original segment continues
to include the cut-out points in its own display; only the new candidate's
points, separately recolored, visually distinguish the cut chunk while the
tool is active). This is expected, not a bug.

## Data model

**Backend (`backend/labeling/segment_state.py`):**

- `SegmentSession.sam_segments` entries change shape from
  `{n_points, created_at, mask_score}` to
  `{n_points, created_at, source: 'sam'|'preseg', mask_score: float|None}`.
  Existing SAM-created entries get `source: 'sam'`, `mask_score` unchanged.
  `materialize_sam_segment(indices, source, protect_instances=None,
  mask_score=None)` gains the `source` parameter (required) and the optional
  `mask_score` (only ever passed by the SAM `/project` route). Everything else
  about the method — fresh id allocation from its own counter, last-write-wins
  overlap, `protect_instances` guard, no undo-stack entry — is unchanged.
- No new array: cut-out presegment points and cut-out SAM points both write
  into the same `sam_ids` array, exactly like today's SAM candidates. A cut
  chunk that originated from a presegment is indistinguishable in storage from
  one that originated from SAM — only the `source` tag on its `sam_segments`
  entry says which list it belongs in.
- Persistence (`sessions/<id>/sam_segments.json`, `sam_ids.npy`): unchanged
  file names/shape, `source` added to each segment's JSON entry.

**Instance-cut path:** no new backend data — reuses `apply_reassign` exactly
as Box/Draw/Beam already do, called via the existing generic `reassign` op
with `target_inst=-1` and `target_class` set to the source instance's class
(read client-side from `class_ids`/`segState`).

## Backend routes

**New generic endpoint** `POST /api/segment/materialize-candidate`
(`backend/routes/segment.py`): accepts `{indices: int[], source: 'preseg'|'sam',
protect_instances?: int[]}`, calls `SegmentSession.materialize_sam_segment`
directly, returns `{sam_seg_id, n_points, n_protected}`. This is needed
because today `materialize_sam_segment` is only reachable from inside
`/api/sam/project`'s SAM-capture flow — cutting has no capture/mask step, it
already has raw point indices computed client-side (box-containment test
inside the isolated modal), so it needs a direct materialize path. One call
per partitioned source-group from a single cut (i.e. a cut spanning
presegments A and B issues two `materialize-candidate` calls).

**Instance-cut path:** no new route — reuses
`VoxaAPI.segApply('reassign', {indices, payload: {target_inst: -1,
target_class}})`, the same call presegment/SAM classification already makes.

**Existing `/api/sam/project`:** unchanged except its internal call to
`materialize_sam_segment` now passes `source: 'sam'` explicitly.

## Frontend

**Trigger — right-click "Edit selection…":**

Available only on the three list surfaces (`PresegmentList`, `SamSegmentList`,
the Instances panel) — **not** directly in the viewport. Right-click there is
already fully claimed by camera pan (`attachOrbit`/`attachWalk` both start a
pan-drag on right-mousedown and suppress the native context menu), so there is
no free gesture to repurpose without changing existing camera-control code,
which is out of scope here. Selecting points in the viewport (Ctrl/Shift-click,
as today) already drives the corresponding list's selection state; the user
right-clicks the populated row(s) in the list to open the menu. Behavior
depends on what's selected when triggered:

- **Presegment/SAM multi-select** (any mix of presegment and SAM rows, one or
  more): enabled. Opens the cut modal over the union of their points.
- **Single instance selected, nothing else:** enabled. Opens the cut modal
  over just that instance's points. (The Instances panel has no multi-select
  today — `mode-label.jsx`'s instance selection is a single `selectedId`
  scalar, not a Set — so "more than one instance" is not a reachable state;
  no changes to Instances-panel selection are in scope for this spec.)
- **Instance mixed with a presegment/SAM selection:** disabled — instance
  cuts are single-instance only, kept structurally separate from the
  multi-source preseg/SAM partitioning flow.
- **Confirmed instance:** disabled (existing "confirmed = locked" rule — must
  un-confirm first via existing UI).

**Isolated modal (new component, e.g. `frontend/src/cut-mode.jsx`):**

Mounts a second `Viewer` (`frontend/src/viewer.jsx` — a plain component, no
change needed) fed a `cloud` filtered to just the selected segments'/instance's
points, with its own camera/orbit state independent of the main viewport.
Each point in the filtered cloud carries its originating segment id (and
whether that source is `preseg`/`sam`/`instance`), computed client-side from
the already-loaded `segState.instanceFull`/`samIds`/`preseg_ids`-equivalent
arrays — no new fetch. Only the Box tool's draw/transform interaction
(reused from `tool-options.jsx`'s existing Box implementation) is available
inside. The modal supports multiple cut passes per session: draw a box,
confirm, the cut points disappear from the modal's remaining cloud, draw
another box on what's left, repeat; close when done.

**Cut-confirm flow:**

1. Box-containment test (`pointsInsideOBB`) runs scoped to the isolated
   subset, producing an index set.
2. Partition that index set by each point's tagged original source segment.
3. For each partition:
   - `source: preseg` or `source: sam` → `POST
     /api/segment/materialize-candidate` with that partition's indices and
     the matching `source` tag. Frontend patches `segState.samIds`
     (`applySamDelta`, already exists) and adds a `sam_segments` entry with
     the returned `sam_seg_id` — the new row shows up immediately in the
     Presegment list or SAM list per its tag.
   - `source: instance` (only ever one partition, since instance cuts are
     single-instance only) → `segApply('reassign', {indices, payload:
     {target_inst: -1, target_class: <source class>}})`, same handling as
     any other apply (new unconfirmed pointset row, `applyDelta`).
4. Remainder points (not enclosed by the box) are left untouched in whichever
   layer they already lived in — no write happens for them at all.

**Right-click context menu:** a small new shared component, invoked from the
three list row types only (see Trigger above — no viewport right-click).
Only one menu item for this spec: "Edit selection…", enabled/disabled per the
rules above.

## Scope boundaries

- Cutting a **Box/Draw/Beam-sourced pointset** that hasn't been confirmed yet:
  treated the same as any other unconfirmed instance (single-instance-only
  cut path, class inherited). No special-casing needed — the "instance"
  source case doesn't care how the instance was created.
- No lasso/freeform cut shape — Box only, matching what was asked for and
  reusing existing Box-tool code.
- No undo/redo for the cut operation itself beyond what already covers its
  underlying calls (`materialize_sam_segment` has none, matching presegment
  selection; `apply_reassign` already has undo).
- Cutting does not delete/discard points — every cut point ends up in exactly
  one place (a new candidate segment or a new instance); there is no "cut to
  trash" concept.
- A cut whose box encloses zero points from a given source produces no
  partition/call for that source (not an error, just a no-op for that group).
- If a cut's `source: preseg`/`source: sam` partition overlaps a still-open
  candidate from an earlier cut or capture, the same "last materialize wins"
  rule from `2026-07-13-sam-segments-as-presegments-design.md` applies
  unchanged — no new merge/dedup logic.
- A cut-out instance never gets its own persisted OBB (unlike a fresh
  Box-tool apply) — it's a plain `apply_reassign` over an explicit index set,
  not a shape apply. This resolves correctly under
  `backend/labeling/materialize.py`'s max-seq regime-B replay (a higher-seq
  non-volumetric claimant beats the source's stale full-extent OBB via the
  `baseline_inst` path) but is worth flagging since it's non-obvious.

## Testing

- **Backend:** `materialize_sam_segment` with the new `source` parameter
  (defaults, tag round-trips through `sam_segments.json`); new
  `/api/segment/materialize-candidate` route test (indices → candidate,
  `protect_instances` dropping, `source` validation rejects unknown values);
  existing `/api/sam/project` tests updated to assert `source: 'sam'` is set
  on the entries it creates.
- **Frontend:** cut-partitioning pure function (given tagged indices, splits
  correctly by source segment id) — this is the core correctness property
  from the A/B example and should be unit-tested directly; right-click menu
  enable/disable logic per the selection-scope rules above; modal cloud
  filtering (given a selection, produces the right point subset + tags).
- **End-to-end:** browser-verify on a throwaway session — multi-select two
  presegments, cut a box spanning both, confirm two new presegment-list rows
  appear (point-recolored, no hull) with the correct point counts, and that
  the two original presegments' own hulls/counts are unchanged (expected —
  see "No hull for cut results" above); repeat for a single unconfirmed
  instance, confirm the split-off piece
  appears as a new unconfirmed instance with the same class and no picker
  prompt; confirm a confirmed instance has "Edit selection…" disabled.

## Future (not this spec)

- Lasso/freeform cut shapes inside the modal (Box only for v1).
- A "discard" affordance for a cut candidate nobody ends up classifying
  (same open follow-up already flagged in the SAM-segments spec).
- Cutting across a mix of presegment/SAM *and* instance sources in one pass
  (explicitly out of scope — instance cuts stay single-instance only).
