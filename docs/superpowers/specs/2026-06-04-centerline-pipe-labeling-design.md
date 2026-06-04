# Centerline pipe/tank labeling — design

Date: 2026-06-04
Status: approved by user (brainstorming session)

## Problem

Labeling pipes and tanks point-by-point or via presegments is slow, and presegment
pipelines fragment long pipes. Most pipes/tanks are cylindrical, so a rough
centerline plus a radius describes one almost completely. This feature lets the
user draw that centerline directly and extracts the surrounding points as one
labeled instance. Right now only the `pipe` and `tank` classes matter, but the
tool is class-agnostic (any class from `classes.yaml` works).

This mode is designed for **blank sessions** — a fresh point cloud with no
presegments and no prior labels. It must not require a preseg to exist.

## Solution overview

A new **Draw** sub-mode of Label mode (sibling of the Fast-labeling sub-mode).
Paths move through three states: **drawing** (active path growing under
Ctrl+click) → **staged** (ended with Esc; editable, not yet labeled) →
**applied** (Enter; backend extracted + labeled the points).

### Workflow

1. Enter Draw sub-mode (button next to Fast-labeling). While active, the
   sub-mode owns the keyboard (capture-phase, like `FastLabelKeys`).
2. Press a class hotkey (e.g. `4` = pipe) — sets the class for the active /
   next path; tube preview renders in the class color.
3. **Ctrl+click** on the cloud places a control point at the raycast-picked
   point. The first Ctrl+click starts a path; each further Ctrl+click
   auto-connects to the previous point. A live semi-transparent tube preview
   follows at the current radius.
4. **Esc** ends the active path → it becomes a *staged* pipe segment.
5. Editing staged (or applied) paths:
   - Click a control point → select it; drag moves it on a screen-parallel
     plane (orbit is suppressed during the drag).
   - Click a path's tube → select the path. **Scroll now resizes its radius**
     (camera zoom resumes when no path is selected); `+`/`-` nudge in small
     steps; a numeric field in the panel gives exact entry. New paths default
     to the last used radius.
   - Number hotkey → set class of the selected path(s).
   - **C** → toggle Smooth (Catmull-Rom through the same control points) on
     the selected path(s); default is straight polyline.
6. **Merge:** click path A, **Shift+click** path B (and C…) to build a
   multi-selection, then press **M** to merge them into one group — on apply
   they share one instance ID (each path keeps its own radius). This handles
   pipes drawn in several runs (occlusion, junctions).
7. **Enter** applies the selected path(s) / merged group: backend extracts
   all full-resolution points within the tube(s) and assigns class +
   instance through the existing reassign flow. As a shortcut, Enter while
   still drawing ends the active path and applies it immediately as its own
   instance. Re-selecting an applied path, editing it, and pressing Enter
   re-applies it (same instance ID; stored paths replaced, see Persistence).
8. Undo (existing `segUndo`) reverts an apply.

### Key & mouse map (Draw sub-mode active)

| Input | While drawing | With selection | Nothing active |
|---|---|---|---|
| Ctrl+click cloud | add control point | start new path | start new path |
| Click control point | — | select + drag to move | select + drag to move |
| Click tube | — | select path (replaces selection) | select path |
| Shift+click tube | — | add/remove from selection | select path |
| Click empty space | — | clear selection | — |
| Scroll | resize active path's radius | resize selected paths' radius | camera zoom |
| `+` / `-` | nudge radius | nudge radius | — |
| class hotkeys | set active path's class | set selected paths' class | set default class |
| `C` | toggle smooth on active path | toggle smooth on selection | — |
| `M` | — | merge selected paths into one group | — |
| `Enter` | end path + apply as new instance | apply selected path(s)/group | — |
| `Esc` | end path → staged | clear selection | exit Draw sub-mode |
| `Backspace` | remove last placed point | delete selected point / staged path | — |

Deleting an **applied** path removes its stored centerline but does not
unlabel its points — use undo right after an apply, or re-label over it.

Extraction semantics: every point within `radius` of the path gets the new
class+instance — later applies overwrite earlier ones where tubes overlap
(latest wins). With the blank-session workflow this is the natural behavior,
and undo covers mistakes.

## Architecture

Frontend owns the drawing UX; backend owns extraction over the full-resolution
cloud (the viewport cloud may be subsampled, so frontend-side extraction would
leave holes at full resolution).

### Frontend (`frontend/src/`)

- `mode-label.jsx`: add the Draw sub-mode toggle next to Fast-labeling; host
  the draw panel (path list, radius field, smooth toggle, merge/confirm
  buttons) and keyboard wiring (class hotkeys, Enter/Esc/Backspace).
- New `draw-paths.js` (pure state, vitest-testable): path collection state
  machine — add/move/remove control point, active path, radius set, smooth
  toggle, multi-select, merge grouping, serialization to the API payload.
- Viewer integration (`viewer.jsx` / draw overlay component): render control
  points (small spheres), polyline, and a semi-transparent tube mesh per
  segment (THREE cylinders for straight segments, `TubeGeometry` on a
  Catmull-Rom curve for smooth paths) in the class color. Drag interaction
  moves a control point on the plane through the point, parallel to the
  screen. Reuses the existing raycast point-pick for placement.
- `api.js`: one new call, `centerlineApply(paths, targetClass, targetInst)`.

### Backend

- New endpoint `POST /api/segment/centerline-apply` (in
  `backend/routes/segment.py`):

  ```json
  {
    "paths": [
      {"points": [[x, y, z], ...], "radius": 0.15, "smooth": false}
    ],
    "target_class": 3,
    "target_inst": -1
  }
  ```

  - Multiple paths in one call = the merge case → one shared instance ID.
  - `target_inst: -1` allocates a new instance ID (same convention as
    `reassign`).
  - Coordinates are in the recentered frame (the frame the viewer and the
    cuboid endpoints already use).

- Extraction (new helper, e.g. `backend/labeling/centerline.py`):
  - Smooth paths are first sampled into short chords (Catmull-Rom, step
    ≈ radius/2), then treated identically to polylines.
  - For each polyline segment, vectorized NumPy point-to-segment distance over
    the full-res positions; a point is captured if its distance to any segment
    of any path in the call ≤ that path's radius. Output: unique full-res
    indices.
- Apply through the existing `segment_state.apply_reassign(indices,
  target_inst, target_class)` — undo stack, delta serialization, dirty state,
  and Ctrl+S save work unchanged.
- Response: same shape as the existing `segment_apply` response (delta with
  `n_affected`, `after_class`, `after_instance`, `new_instance_id`). Note:
  like `_serialize_apply` today, `after_class`/`after_instance` are absent
  when `n_affected` is 0 — the frontend empty-tube handler must not assume
  those keys exist.

### Persistence

Confirmed paths are stored in `sessions/<id>/centerlines.json`:

```json
{
  "paths": [
    {
      "points": [[x, y, z], ...],
      "radius": 0.15,
      "smooth": false,
      "class_id": 3,
      "instance_id": 17
    }
  ]
}
```

- Written by the backend on confirm and saved with the session. Stored paths
  are keyed by `instance_id`: a confirm that targets an existing instance ID
  (re-confirm after editing a path) **replaces** that instance's stored
  paths; a confirm that allocates a new instance ID appends. This keeps
  re-editing a pipe from duplicating its stored paths.
- Loaded when the Draw sub-mode opens, so previously confirmed paths render
  and a pipe's path/radius can be re-edited and re-applied (re-confirm =
  another apply with the same instance ID).

## Error handling

- Tube captures 0 points → HTTP 200 with `n_affected: 0`; frontend shows a
  "no points in tube" toast instead of silently confirming.
- Paths with < 2 points cannot be confirmed (client-side guard).
- Radius must be > 0 (client guard + backend 422 via Pydantic).
- No active session / wrong tier → existing 400/409 behavior of the segment
  routes applies untouched.

## Out of scope (explicitly deferred)

- Auto-centering / auto-radius via local cylinder fit (the
  `presegment_ransac.py` machinery could snap a rough path to the fitted axis
  later — the path data format already carries everything needed).
- Per-segment radius (reducers/tapers).
- Capped tube ends / spherical end caps — the tube is the union of segment
  cylinders plus spheres at joints implied by the distance-to-segment metric.

## Testing

- Backend (pytest, `backend/tests/`):
  - Tube math on a synthetic cylinder cloud: inside/outside radius, points
    near segment joints, multi-segment elbow, smooth (Catmull-Rom) path,
    multiple merged paths sharing one instance ID.
  - Route test: apply → working arrays updated → `segUndo` reverts;
    `n_affected: 0` case; validation errors (empty path, radius ≤ 0).
  - `centerlines.json` written and re-loadable.
- Frontend (vitest, pure-function): `draw-paths.js` state machine —
  add/drag/remove point, radius set, smooth toggle, merge selection,
  payload serialization.
- Manual browser verification of the drawing UX (placement, drag, scroll
  resize, confirm, undo) before calling it done.
