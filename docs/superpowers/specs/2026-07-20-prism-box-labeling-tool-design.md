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
- **Non-horizontal / arbitrary-normal base planes** — the extrusion axis is always
  world-vertical (Y-up) and the prism base is horizontal (at `y0`). v2 snaps each
  corner's *placement* to the cloud surface (so the footprint sits on the geometry
  and reads naturally), but the base-plane *orientation* stays horizontal — a
  fitted arbitrary-normal plane and camera-aligned screen-space sweeps remain out
  of scope.
- **Curve tools** — a round tank is approximated by a many-vertex polygon; there
  is no arc/spline primitive.
- **Self-intersecting polygons** — v1 assumes a simple polygon. Concave is fully
  supported (winding handles it); self-intersecting is undefined-but-safe (see
  Edge cases).

## Interaction model (v2 — mirrors the Measure Surface + Volume tools)

The tool is the 6th entry in the viewport tool rail (`⬠ Prism`). The interaction
mirrors voxa's sibling app `engine/product/demo`'s **Surface** measure tool (the
footprint) and its **Volume** measure tool (the height), because the original v1
model — corners dropped onto an invisible fixed plane at the first click's height,
height by scroll — read as disconnected from the geometry ("works kinda strange").
See `product/demo/src/use-measure-tool.js` for the reference implementation.

1. **Draw the footprint (on a horizontal base plane).** The **first** click
   snaps to the cloud surface via `viewerRef.firstHitUnderCursor` and its `world.y`
   fixes a **horizontal base plane** (`baseY`). Every **later** corner is the
   camera ray ∩ that plane, so all corners are **coplanar** at `baseY`. A
   **dashed rubber-band** previews on the *same* plane, updated imperatively on
   `pointermove` (never a per-move cloud raycast — that would hitch the main
   thread). A click is a `pointerdown`+`pointerup` within ~5 px (a drag is a
   camera move, not a placement).

   > **Why coplanar, not per-corner surface-snap** (superseded design): v2 first
   > snapped *every* corner to its own surface point (Surface-measure-tool style).
   > But a vertical prism needs a **planar XZ footprint**, and over depth-varying
   > geometry the per-corner XZ projection scrambled the click order into a
   > **self-intersecting (bowtie) polygon** — the even-odd point-in-polygon then
   > filled a bowtie region, so the selection did not match the drawn footprint.
   > Coplanar placement (the Measure **Volume** tool's model) keeps the polygon
   > simple: a projective map of a plane preserves a simple polygon.
2. **Close.** **Double-click** or **Enter** closes (minimum 3 corners; the
   double-click's own trailing corner is dropped — its two constituent clicks each
   fire the placement handler). **Backspace** drops the last corner; **Esc**
   cancels the in-progress footprint (empty → exits the tool).
3. **Set height (Volume-tool style).** On close, enter a **height stage**: the
   footprint's **camera-nearest edge** defines a vertical plane; moving the mouse
   up/down **grows/shrinks the height live** (raycast against that plane, height =
   `pt.y − baseY`); a **click commits** the height. Orbit is disabled for the
   whole in-progress interaction (footprint chain + height aim) and re-enabled on
   commit/Esc/tool-exit — the measure tool's exact pattern. A `suppressNextClick`
   guard swallows the closing double-click's trailing `click` so it can't commit a
   zero height.
4. **Classify.** After the height commits, Ctrl+Enter → class picker, or a class
   hotkey directly (the shared apply pipeline). The prism vanishes; its enclosed
   points become a `pointset` instance. A per-tool **auto-confirm on apply** toggle
   is available, like the other tools.

No gizmo at any stage — consistent with pointsets being display-only.

**Geometry mapping (frontend-only — the emitted shape and the whole backend are
unchanged).** The snapped corners give each a `(x, y, z)`. The prism is built as:
`polygon` = the corners' **XZ**; **base `y0` = the minimum corner Y** (lowest
snapped corner); the height-aim gives a top Y; `height = |topY − y0|`, with `y0`
normalized to the *lower* of base/top so the extrude works upward or downward. The
descriptor sent to `apply-shape` is the same `{type:'prism', polygon, y0, height}`
as before, so `prism_indices`, `materialize`, the `Cuboid.prism` schema, and the
`pointsInsidePrism` parity mirror are **all reused unchanged** — this redesign
touches only `frontend/src/prism-mode.jsx`.

**Superseded from v1:** the fixed horizontal base plane at the first click's height
(corners now snap to the surface individually), and scroll-to-set-height (now
aim-to-set-height). The tool-options metres field becomes a read-only display of
the committed height.

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
- **Raw export** (`backend/labeling/materialize.py`): `collect_volumes` (which
  today filters `src not in ("box","beam","draw")` and reads `center/size`) adds
  `"prism"` to its accepted sources and emits
  `{kind:'prism', polygon, y0, height, instance_id, seq}` reading the instance's
  persisted `prism` field; the regime-B `replay_labels` shape branch adds an
  `elif v['kind']=='prism'` calling `prism_indices`. Prism boundaries are
  therefore **exact** in raw
  exports and compete by apply-order `seq` like every other volume. This makes
  prism consistent with Box/Beam rather than the one geometric tool that
  degrades to NN boundaries at raw density.

## Frontend structure

- **`frontend/src/label-tools.js`** — add `{ id:'prism', icon:'⬠', label:'Prism' }`
  to `TOOLS`. Gating in `toolAvailable`: needs `segState` (same as Box — its
  geometry rides in `instances_gt.json`; it does **not** need the raw source or
  a preseg). The rail (`tool-rail.jsx`) renders from `TOOLS` and needs no change.
- **`frontend/src/prism-mode.jsx`** (the only file this v2 redesign rewrites)
  — owns the two-stage draw state (footprint corners as snapped `[x,y,z]`; then a
  height stage `{footprint, height, baseY, heightEdge}`) and the viewport
  interaction, ported from `product/demo/src/use-measure-tool.js`'s Surface (click
  to snap a corner via `firstHitUnderCursor`, imperative dashed rubber-band on
  `pointermove`, double-click/Enter to close) and Volume (camera-nearest-edge
  vertical plane, mouse-move height aim, click-to-commit with the `suppressNextClick`
  guard, orbit disabled during the interaction) handlers. Renders the in-progress
  footprint outline + corner markers + rubber-band, and after close the extrusion
  wireframe preview, as a Viewport overlay via `viewerRef.attachOverlayGroup()`.
  A small pure helper turns snapped corners + the aimed top Y into the emitted
  `{polygon, y0, height}` (XZ projection, `y0 = min corner Y`, signed-height
  normalization) — unit-tested independently of the Three.js interaction.
- **`frontend/src/prism-geom.js`** — `pointsInsidePrism(points, prism)`, the
  parity mirror of `prism_indices` (shared fixture, see Testing). Reused unchanged
  from v1; available for a future live point-highlight preview (the current
  overlay, like the measure tools, is wireframe-only).
- **`frontend/src/tool-options.jsx`** — `PrismOptions` mounts `PrismMode`; its
  panel shows the stage hint and a **read-only committed-height display** (no
  scroll hint, no editable metres field), plus the shared `AutoConfirmToggle`.
- **`frontend/src/mode-label.jsx`** — wire the apply handler (build the `prism`
  descriptor, POST `apply-shape` threading `protectedSegIds`, stamp the instance
  row) and pass the prism preview props into the `Viewer`.

## Edge cases

- **< 3 vertices / collinear / zero-area footprint** → resolves to 0 indices; the
  apply is a benign no-op returning `n_affected: 0`, exactly like an empty box.
  Not an error.
- **Self-intersecting polygon** → point-in-polygon is evaluated by winding; the
  result is well-defined-but-unintuitive. Not detected or blocked (disclosed
  limitation); it cannot corrupt state.
- **Corner click over empty space** (`firstHitUnderCursor` misses) → the corner is
  simply not placed (same as the measure tools). The first corner in particular
  requires a surface hit.
- **Degenerate height** → a height-commit whose aimed top is within a small epsilon
  of the base is treated as a stray click and ignored (mirrors the Volume tool's
  `MIN_FOOTPRINT`/near-zero-extent bail-out), so no zero-height prism is applied.
  Signed aim is normalized so extruding up or down both yield `height > 0`.
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
- **The v2 geometry-mapping helper** (snapped corners + aimed top Y → emitted
  `{polygon, y0, height}`): XZ projection preserves order; `y0 = min corner Y`;
  aiming above the base gives `height = topY − y0`; aiming *below* the base
  normalizes to `y0 = topY`, `height = baseY − topY` (both extrude directions →
  `height > 0`); a near-zero aim yields no shape. This pure helper is where the v2
  logic is unit-tested — the Three.js interaction itself is verified in-browser.

## Out of scope / future

- Snap-to-surface base plane (arbitrary normal) and camera-aligned sweep — the
  two alternatives rejected for v1.
- Per-vertex editing / a gizmo.
- Arc/curve primitives for round footprints.
- Multi-select or boolean combination of prisms.
