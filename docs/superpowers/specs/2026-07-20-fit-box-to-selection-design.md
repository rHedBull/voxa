# Fit Box to Selection — Design

**Date:** 2026-07-20
**Branch:** `feat/fit-box-to-selection` (off `main`)
**Status:** Design approved, pending spec review

## Problem

A user has a set of selected points — a presegment selection, one or more SAM
candidates, or a single instance — and wants a Box-tool selection volume that
tightly encloses them. The point of doing this over the raw selection is that a
box **sweeps every point inside its volume on apply**, including points the
original selection missed (a preseg segment that only caught part of an object,
a SAM mask that back-projected to a near-facing surface only). Fitting a box
around the partial selection and applying it grabs the whole object.

Voxa already has a Box "Auto-fit" button, but it works the other way round: you
**draw** a box first, and it *shrinks* that box to the AABB of the points inside
it (`POST /api/auto-fit`, `mode-label.jsx::autoFitBox`). This feature inverts
that: **start from a point selection, produce the box.** The two are kept
separate (see Non-goals).

## Scope

- **Selection source:** whatever is currently selected, tool-agnostic — the same
  three list surfaces the Cut-selection tool operates on: `PresegmentList`,
  `SamSegmentList`, the Instances panel. No new freeform point-picking gesture
  (Voxa has none today; adding one is out of scope).
- **Box shape:** a **gravity-aligned (yaw-only) oriented box**. One axis is
  locked to world-up (Y); the footprint is the tightest rotated rectangle in the
  X/Z plane (2D PCA), extended to the full vertical extent. `rotation = [0, ry, 0]`.
- **Result is staged, not applied.** The fit produces an editable Box-tool
  `selBox`; the user then transforms (G/R/Y) and labels through the normal Box
  pipeline. Nothing is classified by the fit itself.

## Non-goals

- Not a new rail tool — it is a right-click affordance, like Cut-selection.
- Does not replace or modify the existing draw-then-shrink `/api/auto-fit` button.
- No fully-free 3D PCA (tilt on all axes) — yaw-only only. Rationale: Voxa's
  objects are mostly upright (tanks, pillars, walls) or run horizontally; a
  yaw-only box stays upright and is trivial to nudge afterward, and pure yaw
  sidesteps the Euler-parity hazard that the Beam tool's full OBB has.
- No multi-source fit mixing (e.g. preseg + instance in one box) beyond what a
  single list's selection naturally spans. Preseg and SAM selections are unioned
  within their own list; the Instances panel is single-instance only.

## Data flow

```
Select rows (Presegment / SAM / Instances list)
  → right-click "Fit box to selection…"
  → capture {source, ids} BEFORE any tool switch (selection clears on switch)
  → POST /api/segment/fit-box {source, ids}
  → backend: resolve source membership → full-res indices (union)
             → seg.positions[idx] → fit gravity-aligned OBB
             → {center, size, rotation:[0,ry,0]}
  → frontend: setSelBox(obb); setActiveTool('box')
  → box is STAGED (unconfirmed, editable via G/R/Y)
  → user labels via the normal Box pipeline (hotkey / Ctrl+Enter → apply-shape)
```

The fit is read-only on the source and labels nothing. Once handed off, the box
is an ordinary Box selection: apply resolves the OBB to full-res points via
`shape_indices('obb')` and `apply_reassign`s them, honoring `protect_instances`
(confirmed = locked). This is exactly the existing Box behavior — no new apply
path.

## Backend

### Endpoint: `POST /api/segment/fit-box`

Lives in `backend/routes/segment.py`, next to `_cut_shape_core`. Session-pinned
(`_require_seg()`; 409 if no active session). Request mirrors the source-list
shape used by cut:

```
FitBoxRequest {
  sources: [ { kind: 'preseg'|'sam'|'instance', seg_id: int }, ... ]
}
```

Core (`_fit_box_core(seg, sources)`):

1. Resolve each source to a full-res boolean membership the **same way**
   `_cut_shape_core` does — `preseg` and `instance` → `seg.instance_ids == seg_id`
   (preseg rows are the display ids in `instance_ids`, per the cut-core comment),
   `sam` → `seg.sam_ids == seg_id`. Union all matched indices.
2. If the union is empty → `HTTP 400` (fail loudly, no degenerate box).
3. `pts = seg.positions[idx]` → `fit_gravity_obb(pts)` →
   `{center, size, rotation}` (JSON floats). Return it.

Returns a plain OBB dict (not a `Cuboid` — this is a selection volume, not an
instance): `{ center: [x,y,z], size: [sx,sy,sz], rotation: [0, ry, 0] }`.

### Geometry: `backend/labeling/fit_box.py`

A pure, unit-tested `fit_gravity_obb(points: np.ndarray) -> (center, size, rotation)`:

- **Vertical (Y):** `y0, y1 = points[:,1].min(), max()`; `cy=(y0+y1)/2`, `sy=y1-y0`.
- **Footprint (X/Z):** 2D PCA on `points[:, [0,2]]`.
  - Covariance eigen-decomposition → principal direction; `ry = atan2(...)` of
    that direction, matching the **Rx·Ry·Rz** Euler-XYZ convention that
    `scenes/reproject.py::euler_xyz_matrix` and `shapes.py::_obb_mask` compose
    (with `rx=rz=0`, `Ry` is the only rotation, so parity reduces to getting the
    yaw sign right — locked by a round-trip test, see Testing).
  - Rotate the XZ points by `-ry` into box-local axes, take `min/max` along each
    → footprint center (`cx,cz` back in world) and extents (`sx,sz`).
- **Padding:** add a small epsilon to each size (match the existing `+0.005` in
  `/api/auto-fit`) so the box provably contains every source point despite float
  rounding.
- **Degenerate fallbacks:** fewer than 3 points, or a near-zero secondary
  eigenvalue (collinear XZ) → `ry = 0` (axis-aligned) and clamp each size to a
  small positive minimum. Never returns a zero-volume box.

Reuses `euler_xyz_matrix` / the existing OBB math rather than hand-rolling a new
rotation (per the coordinate-system gotcha in CLAUDE.md).

## Frontend

### `frontend/src/fit-eligibility.js`

Pure gate mirroring `cut-eligibility.js`:

- `list: 'preseg' | 'sam'` → eligible iff `selectionSize > 0`.
- `list: 'instance'` → eligible iff `isSelected`. **Confirmed instances are
  allowed** (unlike Cut): fitting is read-only on the source and the output is a
  new independent box, not a relabel of the confirmed instance. Documented
  divergence from `cutEligibility`.

### Context-menu item

"Fit box to selection…" added to the existing right-click menus on all three
list surfaces (same integration points as the Cut "Edit selection…" item). Uses
`fitEligibility` to enable/disable.

### `mode-label.jsx::fitBoxToSelection(source, ids)`

- Captures `{source, ids}` from the current selection **before** switching tools
  (tool switch clears `segState.selection` / `samSelection`).
- `await VoxaAPI.fitBox(sources)`.
- On success: `setSelBox(obb)` then `setActiveTool('box')`. The Box tool renders
  the staged OBB and owns it from here.
- On error (400/409): surface a banner/toast — no silent no-op.

### `api.js::fitBox(sources)`

Single call site: `POST /api/segment/fit-box`, returns the OBB dict.

## Error handling

- Empty / undersized selection → menu item disabled by `fitEligibility`;
  additionally the backend 400s if the resolved union is empty (defense in depth).
- No active session / unknown ids → backend 409/400, surfaced in the UI. No
  empty-object fallback (matches the "fail loudly" convention and the
  `getAnnotation` throw-don't-swallow rule).

## Testing

### Backend — `backend/tests/test_fit_box.py`

- **Shrinks to selection:** a box fit to a known point cluster has
  center/size matching the cluster's yaw-aligned bounds within epsilon.
- **Containment parity (critical):** feed the fitted OBB back through
  `shape_indices({type:'obb', ...})` and assert it selects **100%** of the
  source points. This is the round-trip that locks the yaw sign / Euler
  convention — a reversed yaw would drop points.
- **Yaw recovery:** synthesize a rectangle rotated by a known angle in XZ; assert
  the recovered `ry` matches (mod π/2 symmetry) and `sx/sz` are the true edge
  lengths, not the (larger) AABB.
- **Degenerate:** <3 points and collinear-XZ inputs return a finite,
  positive-volume, `ry=0` box (no crash, no NaN).
- **Multi-source union:** two preseg ids fit into one box spanning both.
- **No session → 409; empty union → 400.**

### Frontend

- `fit-eligibility.test.js` — preseg/sam threshold, instance selected/unselected,
  and the confirmed-instance-allowed divergence from cut.
- A jsdom context-menu test that "Fit box to selection…" appears and is
  gated by `fitEligibility`.
- `api.test.js` — `fitBox` posts the right body / parses the OBB.

## Files touched

**New**
- `backend/labeling/fit_box.py` — `fit_gravity_obb`.
- `backend/tests/test_fit_box.py`.
- `frontend/src/fit-eligibility.js`.
- `frontend/src/fit-eligibility.test.js`.
- `docs/superpowers/specs/2026-07-20-fit-box-to-selection-design.md` (this doc).

**Modified**
- `backend/routes/segment.py` — `FitBoxRequest`, `_fit_box_core`, route.
- `backend/app/schemas.py` — `FitBoxRequest` schema (or inline in segment route,
  matching where cut's request schema lives).
- `frontend/src/api.js` — `fitBox`.
- `frontend/src/mode-label.jsx` — `fitBoxToSelection` + wire the menu item.
- The context-menu host(s) for the three lists (same files the Cut item is in).
- `frontend/src/context-menu.test.jsx` / relevant jsdom test.
- `CLAUDE.md` — one bullet under the Box tool describing "Fit box to selection".

## Open questions / assumptions

- **Assumption:** fitting against `seg.positions[idx]` (full-res) is correct and
  affordable — the selection index set is small relative to the whole cloud, and
  this is the same array cut/apply already index. Verified: `seg.positions` is
  the full-res cloud held by the active `SegmentSession`.
- **Assumption:** the staged box does not need to persist anywhere until applied
  — it lives in transient `selBox` state, identical to a hand-drawn Box.
