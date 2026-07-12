# SAM-aided labeling tool — design

Date: 2026-07-12
Status: designed (not implemented)

## Problem

Labeling industrial point clouds by drawing volumes (Box), centerlines (Draw), or
beam graphs works well for geometric primitives, but a lot of real geometry —
irregular vessels, valve clusters, tanks, gratings, cable trays — isn't captured
cleanly by a box or a swept tube. A human looking at a rendered view can name such
an object instantly; SAM (Segment Anything, v3) can turn that visual judgment into
a 2D mask. If we can turn a SAM mask on a *rendered view* into a 3D point
selection, we get a fast "point at it and it's selected" labeling path that
complements the geometric tools.

The method was validated in a standalone PoC (`../quick-select-visual-test`):
navigate a high-res view → box (+ optional text) → SAM mask → depth-aware
back-projection onto the cloud → the covered 3D points light up. This spec brings
that method into voxa as a first-class Label-mode tool, single-view, for v1.

## Solution overview

A **SAM tool** — the 5th entry in the Label-mode tool rail, a sibling of
Presegment / Box / Draw / Beam. It follows voxa's unifying principle exactly: *a
tool is only a way to select points; everything downstream is one shared pipeline*
(`select → apply+label → unconfirmed pointset → confirm`). SAM produces a point
selection; `apply_reassign` and the Instances panel handle the rest, unchanged.

The one thing SAM needs that no other tool needs is a **rendered RGB image** of the
scene from the current camera, at full quality, to segment. voxa's own viewer only
renders a ~1M subsample, which is too sparse for good masks on large scans. So the
image is rendered **server-side from the raw cloud** (188M-class), decoupled from
voxa's viewer. This is what the PoC used Potree for; we replace Potree with a pure
numpy perspective point-splat render (validated at full resolution — photographic
on the 188M SMART-AIS plant). **voxa's viewer is unchanged; it is only a camera-pose
and box source.**

### Two selection modes (v1)

- **Box** — shift-drag a 2D box around one object (+ optional text prompt) → SAM's
  best mask → one selection.
- **Concept** — type a text prompt ("pipe"), click **Segment all** → SAM returns
  *all* instances of that concept in one forward pass → N masks → the user picks
  which to keep → N selections.

Explicitly **not** in v1: "everything" box-grid mode, multi-view merge, proximity
ranking, SAM video tracking, click-point prompts (SAM3's image API is box/text
only).

## Components

Three units, each with one clear responsibility and a well-defined interface.

### A. SAM sidecar (new standalone service)

A separate process in the GPU/anaconda env (torch + sam3) — essentially the PoC's
server, adapted. Runs outside voxa so voxa's lean `.venv` and `npm run dev` are
untouched, CUDA and the ~2.8GB raw cloud stay isolated, and SAM's slow model load
is paid once at sidecar startup.

Loads once at startup, for the active scan:
- the SAM3 image model + processor (`build_processor`, from the PoC),
- the **raw cloud** (188M xyz+rgb, resolved via `derivation→sources.json`; cached
  to `.npy` like the PoC) — used to render the image and the occlusion depth-buffer,
- the scan's **`scan.ply`** (the *same file* voxa's session labels against) — the
  projection/selection target, so returned indices align 1:1 with voxa's working
  arrays.

Endpoints (stateful across the two calls of one interaction, stateless across
interactions — one live `capture_id` at a time):

- `POST /capture` — body `{camera:{pos,target,fov,W,H} (native frame), mode:"box"|"concept",
  box?:[cx,cy,w,h], text?}`.
  1. Render the raw cloud through the camera → RGB image + per-pixel depth-buffer
     (numpy splat; distance-adaptive radius).
  2. Run SAM: box mode → highest-scoring mask; concept mode → all masks above a
     score threshold.
  3. Stash `{render, depth-buffer, masks, camera}` under a fresh `capture_id`
     (replaces any prior one).
  4. Return `{capture_id, overlay_png, masks:[{mask_id, score}]}` (overlay = the
     rendered image with masks washed over it, for review). **No projection, no
     voxa mutation.**

- `POST /project` — body `{capture_id, mask_ids:[...]}`.
  1. For each chosen mask: project `scan.ply` through the stored camera, keep the
     scan-res points that fall inside the mask **and** pass the raw depth-buffer
     occlusion test (nearest-wins, splatted).
  2. Return `{instances:[{mask_id, scan_indices}]}` (int32 indices into `scan.ply`
     order).
  Stale `capture_id` (a newer capture replaced it) → 409.

Occlusion is tested against the **raw** depth-buffer even though we select
scan-res points — so occlusion is dense-accurate while labels stay scan-res.

### B. voxa backend proxy (`backend/routes/sam.py`, new)

Thin. `POST /api/sam/capture` and `POST /api/sam/project` forward to the sidecar,
adding `recenter_offset` to the pose so the sidecar renders in native coordinates.
On `/project` return, each instance's `scan_indices` go through the **existing**
`apply_reassign(..., protect_instances=protectedSegIds)` (session-pinned), producing
one **unconfirmed** pointset per mask. Confirmed-lock, undo, save, and export are
all the existing machinery — no new label write path. SAM tool is disabled when
`raw_source_available` is false (mirrors export-wizard gating).

### C. voxa frontend (5th rail tool `'sam'`)

- `label-tools.js` — add the tool + its gating; `tool-rail.jsx` — 5th icon.
- `tool-options.jsx` → a SAM panel: mode toggle (Box / Concept), text-prompt input,
  **Segment all** button (concept), auto-confirm-on-apply toggle.
- **Box mode:** shift-drag a rubber-band `<div>` box on the canvas (Shift suppresses
  orbit); on mouseup read camera `pos`/`target`/`fov` + canvas `W,H`, normalize the
  box, POST `/capture`.
- **Concept mode:** type a prompt, click Segment all → POST `/capture` (no box).
- **Mask-review panel:** shows the returned `overlay_png` with the masks; user
  clicks which to keep (single accept in box mode, multi-select in concept mode) →
  POST `/project` with the chosen `mask_ids` → the kept masks become unconfirmed
  pointsets. voxa's renderer needs **no** `preserveDrawingBuffer` change — voxa
  never captures its own frame; it only sends the pose.

## Data flow

**Box:** pick Box → (optional prompt) → shift-drag box → `/capture` → review panel
shows the one mask overlay → accept → `/project {mask_ids:[0]}` → `apply_reassign`
→ one unconfirmed pointset (auto-confirmed if the toggle is on).

**Concept:** pick Concept → type "pipe" → Segment all → `/capture` → review panel
shows the multi-colored overlay + N masks → user clicks the real ones →
`/project {mask_ids:[chosen]}` → `apply_reassign` per mask → N unconfirmed
pointsets, each its own hue → user confirms/deletes via the Instances panel.

**Invariant:** `/capture` never mutates voxa state; only `/project` → `apply_reassign`
does, via the same path Box/Draw/Beam use.

## Coordinates (load-bearing invariant)

One pixel space everywhere = voxa's canvas `W,H`; the box is normalized
`[cx,cy,w,h]`; the sidecar renders at that exact `W,H` and voxa's vertical `fov`.
voxa's camera pose is in the **recentered** frame; the raw cloud + `scan.ply` are
**native UTM** — voxa sends `pose + recenter_offset`, the sidecar adds the offset
back and works in native coords with `up=(0,0,1)`. Point *indices* are
frame-independent, so they return to voxa directly. A ported projection sanity test
(center-point projects to image center; occluded point rejected) guards this.

## Scope boundaries (explicitly NOT in v1)

- No "everything" box-grid mode; no multi-view merge; no proximity ranking; no
  video tracking; no click-point prompts.
- **No new persistence.** No `sam_segments.json`. A projected mask is immediately a
  normal pointset in `instances_gt.json` + working arrays — nothing to stage across
  views yet. (The segment store arrives with multi-view merge later.)
- **No OBB / selection-volume** on SAM pointsets — a mask isn't an OBB. On export
  these NN-transfer to raw density like preseg/click-select pointsets (materialize
  regime B), acceptable at scan-res.
- Sidecar keeps only one live `capture_id`; no capture history.

## Error handling (fail loudly)

- Sidecar unreachable / SAM load failed → proxy returns a clear error; the SAM panel
  shows a blocking banner (like the 409 preseg-divergence banner). No silent empty
  selection.
- SAM returns 0 masks → panel says "no masks", applies nothing (never an empty
  pointset).
- Stale `capture_id` at `/project` → 409 → frontend re-captures.
- `raw_source_available` false → SAM tool disabled with a tooltip.

## Testing

- **Sidecar:** ported projection/occlusion unit test; render smoke test (non-empty,
  correct W×H); SAM box + concept wrappers behind a lazy import so non-CUDA CI skips.
- **voxa backend:** proxy route tests with a **mocked sidecar** — capture→project→
  `apply_reassign` yields an unconfirmed pointset; `protect_instances` respected;
  stale `capture_id` → 409; `raw_source_available` gating.
- **Frontend:** pure-fn tests for box normalization + pose-payload assembly (vitest,
  node env).
- **End-to-end:** browser-verification on `smart_ais_navvis` in a throwaway session
  (Label apply auto-saves), box + concept, zero console errors.

## Future (not this spec)

Multi-view merge (the PoC's designed next step): accumulate shells from many views
into a `sam_segments.json` store, merge same-object shells manually with proximity
as a ranked hint (region-overlap auto-merge was measured unreliable on thin
interwoven pipes), `apply_merge` into instances, export per-point `instance_id`.
"Everything" class-agnostic mode. SAM3 video tracking across a smooth orbit for
semi-automatic identity carry.
