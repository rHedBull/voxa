# Prism Box labeling tool — design

Date: 2026-07-20
Status: designed

## Problem

The Box tool selects points with a single **oriented rectangle**. On any object
whose footprint is not rectangular — an L-shaped skid, a round tank, a bent wall
run, a floor patch bounded by other equipment — one box either **over-selects**
(the rotated rectangle grabs an empty corner and clips a neighbour) or
**under-covers**. The only workaround today is many small boxes or a presegment
that may not exist.

This spec adds a **Prism** tool: draw an arbitrary footprint **polygon** on a
horizontal plane, then set an **extrusion height**. The polygon×height volume
selects every enclosed point. It is shape-agnostic — the polygon can have any
number of vertices and be concave — so one prism can hug a footprint no single
box can.

Like every other Label-mode tool, **a prism is only a way to select points**;
everything downstream (`select → apply+label → unconfirmed pointset → confirm`)
is the shared pipeline. Every apply produces a `kind:'pointset'` instance.

## Non-goals (v1)

- **Vertex editing** — to change a footprint you Clear and redraw (mirrors Box's
  "Clear box"). No per-vertex drag, no gizmo.
- **Non-horizontal base planes** — the base plane is always world-horizontal
  (Y-up), extruding vertically. Snap-to-surface (arbitrary normal) and
  camera-aligned screen-space sweeps were considered and explicitly deferred.
- **Curve tools** — a round tank is approximated by a many-vertex polygon; there
  is no arc/spline primitive.
- **Self-intersecting polygons** — v1 assumes a simple polygon. Concave is fully
  supported (winding handles it); self-intersecting is undefined-but-safe (see
  Edge cases).

## Interaction model

The tool is the 6th entry in the viewport tool rail (`⬠ Prism`).

1. **Draw the footprint.** Click to place vertices. The **first click fixes the
   base plane**: a horizontal plane at that point's Y (`y0`). Every subsequent
   vertex ray-casts onto that same `y0` plane, so all vertices are coplanar. A
   rubber-band edge follows the cursor.
2. **Close.** Click the first vertex again, or press **Enter**. **Backspace**
   removes the last vertex; **Esc** cancels the whole in-progress polygon.
   Minimum 3 vertices.
3. **Set height.** After the polygon closes, **scroll** grows/shrinks the
   extrusion height, which grows **up** from the `y0` plane (`y0 → y0+height`).
   This reuses Beam's scroll-to-size gesture. A metres field in the tool-options
   panel shows the current height and accepts a typed exact value.
4. **Apply.** Ctrl+Enter → class picker, or a class hotkey directly (the shared
   apply pipeline). The prism vanishes; its enclosed points become a `pointset`
   instance. A per-tool **auto-confirm on apply** toggle is available, like the
   other tools.

No gizmo at any stage — consistent with the "no-gizmo v1" constraint and with
pointsets being display-only.

## Geometry & backend containment

The prism is a **shape descriptor** resolved through the existing generic
endpoint `POST /api/segment/apply-shape` — **no new endpoint** (same path Box,
Draw, and Beam use):

```
{ type: 'prism',
  polygon: [[x, z], ...],   # footprint vertices in the world horizontal (XZ) plane
  y0:      <float>,         # base-plane height
  height:  <float> }        # extrusion distance, upward
```

All coordinates are in the **recentered frame** (matching the OBB/tube shapes and
`_recenter` in `app/core.py`).

New resolver `prism_indices(positions, shape)` in `backend/labeling/shapes.py`:

- A point is inside iff its **XZ** projection is inside `polygon` (vectorised
  ray-casting point-in-polygon, concave-safe) **and** `y0 ≤ y ≤ y0 + height`.
- Exact and **resolution-independent by construction** — no nearest-neighbour
  approximation, unlike preseg transfer.

`shape_indices` gains a `prism` branch dispatching to `prism_indices`. The
`ApplyShapeRequest.shape` docstring/comment in `backend/app/schemas.py` is
updated from `{type:'tube'|'obb'}` to include `'prism'`. The apply path needs
**no per-shape structure hook** (like `obb`, prism persists nothing session-side
beyond the instance itself).

Because prism is a **volume sweep**, the apply carries `protect_instances` (the
confirmed instances' segIds), so **Confirmed = locked** holds: an overextended
prism can never overwrite already-confirmed points. `apply_reassign` drops those
points and returns `n_protected`, identical to Box/Draw/Beam.

## Persistence & resolution-independent export

Full parity with the shipped resolution-independent-labels principle (Box/Beam
persist their OBBs as inert selection volumes a future raw export rasterizes at
any density). The prism persists the same way:

- **Instance schema** (`backend/app/schemas.py`, the `Cuboid`/pointset model)
  gains an optional nested `prism: {polygon, y0, height}` field, alongside the
  existing `center/size/rotation` used by OBB pointsets. It is `null` for
  non-prism instances.
- The **frontend stamps** `{source:'prism', prism:{polygon,y0,height}}` onto the
  new pointset row in `instances_gt.json` on apply — exactly how Box stamps its
  OBB. The instance stays `kind:'pointset'`, so all gizmo/cuboid-edge/auto-fit
  paths that gate on `kind !== 'pointset'` continue to skip it.
- **Raw export** (`backend/labeling/materialize.py`): `_volumetric_instances`
  emits `{kind:'prism', polygon, y0, height, instance_id, seq}` for
  `source=='prism'`; the regime-B replay adds an `elif v['kind']=='prism'` branch
  calling `prism_indices`. Prism boundaries are therefore **exact** in raw
  exports and compete by apply-order `seq` like every other volume. This makes
  prism consistent with Box/Beam rather than the one geometric tool that
  degrades to NN boundaries at raw density.

## Frontend structure

- **`frontend/src/label-tools.js`** — add `{ id:'prism', icon:'⬠', label:'Prism' }`
  to `TOOLS`. Gating in `toolAvailable`: needs `segState` (same as Box — its
  geometry rides in `instances_gt.json`; it does **not** need the raw source or
  a preseg). The rail (`tool-rail.jsx`) renders from `TOOLS` and needs no change.
- **`frontend/src/prism-mode.jsx`** (new, mirrors `beam-mode.jsx`/`draw-mode.jsx`)
  — owns draw state (vertices, `y0`, `height`), the viewport interaction
  handlers (plane-click to place a vertex, scroll to set height, Enter/Backspace/
  Esc), and the in-progress **polygon + extrusion preview** rendered as a
  Viewport overlay using the same mechanism as the Box selection box and Beam
  graph.
- **`frontend/src/prism-geom.js`** (new) — `pointsInsidePrism(points, prism)` for
  the live in-viewport preview highlight, **mirroring** `prism_indices` so the
  applied label matches the on-screen selection exactly (the way
  `pointsInsideOBBLabel` in `mode-label.jsx` mirrors `obb_indices`). Parity is
  locked by a shared fixture (see Testing).
- **`frontend/src/tool-options.jsx`** — add `PrismOptions`: Draw / Clear buttons,
  a height metres field + scroll hint, and the shared `AutoConfirmToggle`.
- **`frontend/src/mode-label.jsx`** — wire the apply handler (build the `prism`
  descriptor, POST `apply-shape` threading `protectedSegIds`, stamp the instance
  row) and pass the prism preview props into the `Viewer`.

## Edge cases

- **< 3 vertices / collinear / zero-area footprint** → resolves to 0 indices; the
  apply is a benign no-op returning `n_affected: 0`, exactly like an empty box.
  Not an error.
- **Self-intersecting polygon** → point-in-polygon is evaluated by winding; the
  result is well-defined-but-unintuitive. v1 does not detect or block it
  (disclosed limitation); it cannot corrupt state.
- **Height 0 or negative** → the metres field clamps to a small positive minimum;
  scroll cannot drive height below it.
- **Overextended prism over confirmed points** → those points are protected and
  reported via `n_protected` (Confirmed = locked), never overwritten.

## Testing

**Backend** (`backend/tests/`):
- `prism_indices`: convex square, **concave** (L-shaped) footprint, points inside
  vs outside the Y-band, empty/degenerate footprint, point exactly on an edge.
- A **parity fixture** (polygon + points + expected inside-mask) shared with the
  frontend mirror test.
- `materialize` regime-B prism replay: a prism instance rasterizes onto a denser
  cloud with exact boundaries and correct `seq` competition.
- `apply-shape` route with a `prism` shape: `n_affected`/`n_protected` correct,
  unknown-field/validation behaviour unchanged.

**Frontend** (`frontend/src/*.test.js`):
- `pointsInsidePrism` parity against the shared fixture (same expected mask as the
  backend).
- `toolAvailable('prism', …)` gating (session required; no raw source needed).
- `PrismOptions` renders (Draw/Clear, height field, auto-confirm).

## Out of scope / future

- Snap-to-surface base plane (arbitrary normal) and camera-aligned sweep — the
  two alternatives rejected for v1.
- Per-vertex editing / a gizmo.
- Arc/curve primitives for round footprints.
- Multi-select or boolean combination of prisms.
