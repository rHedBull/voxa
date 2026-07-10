# Unified Label-mode tools — design

Date: 2026-07-10
Status: approved by user (brainstorming session)

## Problem

Label mode has grown several ways to label points — cuboids, presegment
selection, fast-labeling, and centerline draw — that were each bolted on
separately. They disagree on three axes:

- **Tool switching is two disjoint mechanisms.** `activeTool` is a hard-coded
  constant (`'cuboid'`, `mode-label.jsx:66`) while the others live in a separate
  `subMode` state (`null | 'fast' | 'draw'`, `:89`). Auto-fit is neither — it's an
  action button on a selected cuboid. There is no single tool selector.
- **Their controls are scattered.** The presegment list and draw panel sit in the
  left rail; the box gizmo (Move/Rotate/Scale, box-select, auto-fit) is a floating
  viewport toolbar. There is no single home for "the controls of the active tool."
- **They disagree on the output pipeline.** Presegment selection creates an
  *unconfirmed* `pointset` instance (`confirmSegmentSelection`, `:769`), but
  Fast-labeling and Draw both write `confirmed:true` immediately, and Box produces
  a geometric `cuboid` instance that is a different annotation type entirely.

The insight that unifies them: **a labeling tool is nothing but a way to select a
group of points.** Everything after selection — assign a class, land as an
unconfirmed instance, confirm it — can and should be one shared machine.

## Solution overview

Rebuild Label mode around a single principle: **tools differ only in how points
are selected; all downstream behavior is identical.**

- **One tool rail** of three selection tools: **Presegment**, **Box**, **Draw**
  (with a reserved 4th slot for **Beam**, spec-only, not built here).
- **One contextual tool-options panel** that swaps to show the active tool's
  sub-controls.
- **One shared pipeline:** `select points → apply+label → unconfirmed instance →
  confirm`, with a per-tool `auto-confirm on apply` option as the only deviation.
- **One output type:** every apply produces a `pointset` instance. The geometric
  `cuboid` annotation type is retired for new labels.
- **Structure is persisted** for tools that carry a graph (Draw, Beam), so the
  selection is re-editable — the graph is the source of truth, the point-group is
  its derived label.
- **One generic backend `apply-shape` endpoint.** The backend mirror of the
  frontend principle: a tool's selection is a *shape*, and the backend resolves any
  shape to the full-resolution points inside it, then reassigns them — identically
  for every shape.

This is mostly a **frontend + UX refactor**, with **one backend generalization**:
today's `centerline-apply` becomes a generic `apply-shape` endpoint (see below).
Everything else rides on the existing backend (`apply_reassign`, the `confirmed`
flag, `promotedSegIds`, and the per-session sidecar files).

## Backend: the generic `apply-shape` endpoint

The Draw and Box tools are the **same backend operation**: resolve a geometric
shape to the full-resolution point indices inside it, then `apply_reassign` them
to a label. They differ only in the shape's membership test. Presegment is the
degenerate case — its "shape" is an explicit index list, which `reassign` already
handles.

Crucially this **cannot** be done frontend-only for geometric shapes: the cloud
the frontend renders is **subsampled** (`_safe_subsample` in `routes/load.py`,
capped at `VOXA_MAX_POINTS`), while labeling reassigns points in the
**full-resolution** working arrays. Draw already handles this by sending its
geometry to the backend, where `centerline_apply` runs `tube_indices` on the
backend's full-res `seg.positions` (`routes/segment.py:103`). Box has the identical
requirement, so it needs backend membership computation too — running the OBB test
on the subsampled `cloud.positions` in the browser would label only the visible
subset and miss most in-box points.

So introduce **`POST /api/segment/apply-shape`** — a generalization of the existing
`centerline-apply`:

```
body: { shape, target_inst, target_class, merged_from? }
  shape = { type: 'tube', paths: [...] }              # Draw (migrated)
        | { type: 'obb',  center, size, rotation }    # Box (new)
        | { type: 'beam', ... }                       # later
```

The endpoint has two per-shape hooks and an otherwise-shared body:

1. **`shape → full-res indices`** (dispatch): `tube` → existing `tube_indices`;
   `obb` → a new `obb_indices(positions, box)` (the full-res analogue of
   `pointsInsideOBBLabel`, with the same AABB prefilter `tube_indices` uses).
2. **shared** `apply_reassign(idx, target_inst, target_class)` + serialize.
3. **persist structure sidecar** (per-shape): `tube` → `update_centerlines`
   (`centerlines.json`); `obb` → none; `beam` → `structure.json` (later).

**Migration:** `centerline_apply`'s body is refactored into `apply-shape` with
`shape.type='tube'`; the `/api/segment/centerline-apply` route is either kept as a
thin wrapper delegating to `apply-shape` or removed once the Draw frontend calls
`apply-shape`. Existing centerline tests are preserved (they must stay green — the
tube path is behavior-unchanged). Box adds the `obb` branch; Beam adds `beam`
later with no further endpoint work.

## Tool model

Replace the two disjoint mechanisms with one piece of state:

```
activeTool: 'presegment' | 'box' | 'draw'   // (later: | 'beam')
```

Every tool's job is to populate the shared **selection** (the set of points the
next apply will label). This deletes the `subMode` split and the hard-coded
`activeTool` constant.

- **Presegment** — select precomputed segments (Ctrl/Shift-click segments in the
  viewport or the presegment list). Its `manual`/`rapid` toggle and the presegment
  list move into its tool-options.
- **Box** — draw an oriented box and Move/Rotate/Scale it to enclose points. On
  apply, the enclosed points become the point-group and **the box outline
  vanishes** — it is a transient selection gizmo, not a persisted annotation.
  This is distinct from both existing "box" paths: it is **not** `addCuboid`
  (the `A` key, which creates a geometric `kind:'cuboid'` and encloses no points),
  and **not** `confirmBoxSelect` (the box-select toolbar, which toggles whole
  presegments by centroid). Box labels the **actual full-res points inside the
  OBB** by sending the box to the `apply-shape` endpoint (`shape.type='obb'`) — the
  browser-side `pointsInsideOBBLabel` operates on the subsampled cloud and is used
  only for the live selection *preview/highlight*, never for the label itself. Box
  works on raw / preseg-less clouds where there are no segments to select.
- **Draw** — draw centerline paths; `apply-shape` (`shape.type='tube'`) extracts
  full-res points within a per-path tube radius. Behavior otherwise as today.

**Tool gating.** The rail always shows all three tools, but tools unavailable for
the current scan are **disabled** (not hidden), with a tooltip: **Presegment** is
disabled when there is no `segState` / no presegments; **Draw** is disabled on
non-annotated (legacy-tier) scans, as today (`mode-label.jsx:1081`). **Box** is
always available — it needs neither presegments nor an annotated scan — so it is
the default tool on a raw/preseg-less cloud.

Because the tools now reach the backend by different routes — Presegment via
`reassign` (explicit indices) and Box/Draw via `apply-shape` (a shape) — the
frontend apply logic is refactored so all three land the *same* resulting
`pointset` instance in the right-side list (see The shared pipeline). Presegment
keeps using `confirmSegmentSelection` (indices from `segState.selection`); Box and
Draw call `apply-shape` and feed the returned instance id through the same
instance-creation path.

### Output type: pointset only

New labels always produce `kind:'pointset'` instances. The `kind:'cuboid'`
geometric annotation type is no longer *created*:

- **Legacy `cuboid` instances** in already-saved sessions still render and can be
  selected, confirmed, and deleted, but are **display-only legacy** — the Box tool
  no longer produces them. They render **read-only with no gizmo**; the
  Move/Rotate/Scale gizmo, per-cuboid Auto-fit, and densify controls (today gated
  on `activeTool === 'cuboid'` in the floating toolbar, `mode-label.jsx:1193–1222`)
  are **dropped for them** — those controls do not get re-homed. No auto-conversion
  (old boxes never stored their enclosed points, so there is nothing to migrate
  them into).
- Compare is already per-point, so retiring cuboids removes an annotation type
  that Compare never scored anyway.

## Layout

Three regions, mapping onto today's 3-column shell (`.side-l` / `.vp-stack` /
`.side-r`):

- **Viewport top toolbar** — the tool rail `[◱ Presegment] [▭ Box] [✎ Draw]`,
  rendered in the existing `.vp-hud-top` strip over the 3D view, always visible.
  A 4th **Beam** slot is reserved (rendered disabled/hidden until built).
- **Left rail** (`.side-l`) — `Session · Classes · Tool-options panel`. The
  tool-options panel is the single contextual home for the active tool's
  sub-controls; it swaps on tool change. It **absorbs** today's scattered panels:
  the presegment list, the draw panel, and the box gizmo controls.
- **Right rail** (`.side-r`) — Instances panel, same position, single flat list
  with filter options (see below).

Global viewport controls that are **not** tool-specific stay in the viewport HUD
as today: nav-mode toggle, camera presets, reset-cam, points-left chip, mesh
window. Only the **box-specific** gizmo buttons (Move/Rotate/Scale, Auto-fit,
box-select) move out of the floating toolbar and into Box's tool-options.

## Per-tool sub-controls (tool-options panel contents)

- **Presegment**
  - `manual ◦ rapid` selection toggle (rapid = today's fast-label queue behavior:
    step through unpromoted segments largest-first).
  - Segment-size filter.
  - Presegment list (Ctrl/Shift-click to multi-select), excluding already-promoted
    segments (`promotedSegIds`).
  - `auto-confirm on apply` toggle — **on by default in rapid**, off in manual.
- **Box**
  - Draw (`A`), Move / Rotate / Scale gizmo, Auto-fit, box-select.
  - `auto-confirm on apply` toggle — **off by default**.
- **Draw**
  - Today's DrawPanel verbatim: tube radius, point-size, path list, ◌/● applied
    toggle, anchors/branches.
  - `auto-confirm on apply` toggle — **off by default**.

## The shared pipeline

Identical for all tools:

```
select points
  → apply+label   (Ctrl+Enter → class picker, OR press a class hotkey directly)
  → UNCONFIRMED pointset instance
  → confirm       (per-row ✓, or class hotkey when auto-confirm is on)
```

- Apply resolves the selection to points (Presegment → `reassign` with explicit
  indices; Box/Draw → `apply-shape` with a shape) and creates a `pointset`
  instance with `confirmed: !!opts.autoConfirm`. The instance-creation half is
  shared across tools; only the resolve-to-points call differs.
- **`auto-confirm on apply`** is the *only* per-tool deviation. When on, apply
  lands directly in **confirmed** (preserving fast-labeling's one-keypress speed).
  When off, it lands **unconfirmed** and is confirmed as a separate step.
- Confirmed instances behave exactly as today: locked (read-only, no gizmo/rename/
  reclass/delete), and their points hidden when the "N done" toggle is on.
- **Draw** (and the **Fast**/rapid path) are re-routed from their current
  hard-coded `confirmed:true` (`mode-label.jsx:845`, `:911`) to this shared path,
  honoring their `auto-confirm` toggle. **Box** is re-routed from producing a
  geometric cuboid (`addCuboid`) to producing an unconfirmed pointset. All three
  default `auto-confirm` off (rapid preseg defaults on), so they now land
  unconfirmed unless the user opts in.
- **Applying via a class hotkey is new behavior.** Today a class hotkey with a
  selection only sets the active class / reclasses a selected cuboid
  (`mode-label.jsx:969–973`); it does not apply. That handler changes so that,
  when a tool selection is active, pressing a class hotkey applies+labels the
  current selection (honoring `auto-confirm`). Ctrl+Enter → class picker remains
  the explicit path.

## Persisted structure (Draw, and later Beam)

Every tool outputs a point-group into the shared pipeline. Tools whose selection
*is* a meaningful editable graph additionally persist that graph as a per-session
sidecar, linked to the resulting instance:

- **Draw** → centerline graph (control points/nodes, branches, junctions) →
  `sessions/<id>/centerlines.json` (already exists; now written by `apply-shape`'s
  per-shape persist hook for `tube`, behavior-unchanged; source of truth for the
  instance).
- **Beam** (later) → node/edge graph → `sessions/<id>/structure.json` (per the
  beam-structure spec).
- **Presegment** → structure is just the absorbed seg-ids (`promotedSegIds`,
  already tracked).
- **Box** → no structure to persist (the brush is transient).

The graph is the source of truth and re-editable: re-opening a pipe/beam instance
loads its graph back into the Draw/Beam tool so a node can be nudged and the
instance re-applied. Sidecars stay **per-tool** (`centerlines.json`,
`structure.json`) rather than one combined store, matching the existing pattern.

## Instances panel

Keep a **single flat list** (no unconfirmed/confirmed section split). Extend
today's text filter with an **unconfirmed / confirmed / all** filter option, so
the review queue is reachable without restructuring the panel. Rows and the inline
editor are otherwise unchanged; confirmed rows keep their locked styling and the
"N done" hide-confirmed toggle.

## Scope / non-goals

- **Backend:** one generalization only — `centerline-apply` → generic
  `apply-shape` (adds the `obb` shape for Box; `tube` migrated unchanged). No new
  annotation types and no new persistence formats: `obb` persists nothing,
  `tube` keeps writing `centerlines.json`. Reuses `apply_reassign`, the `confirmed`
  flag, and `promotedSegIds`.
- **Beam:** not implemented here — the rail reserves its slot, and `apply-shape`
  anticipates its `beam` shape + `structure.json` sidecar, per
  `2026-07-10-beam-structure-labeling-design.md`.
- **Legacy cuboids:** display-only; no migration/auto-conversion.

## Testing

- Backend tests (pytest):
  - `apply-shape` `tube` parity: same result as the old `centerline-apply` on the
    same input (migrate/extend the existing centerline tests — they must stay
    green).
  - `apply-shape` `obb`: `obb_indices` selects exactly the full-res points inside
    an oriented box on a known synthetic cloud (incl. a rotated box), and the
    endpoint reassigns them and returns a new instance id.
  - unknown `shape.type` → HTTP 400.
- Frontend unit tests (vitest, pure-function level as the repo does today):
  - tool-state reducer: switching `activeTool` clears/preserves selection
    correctly and swaps tool-options.
  - the shared apply path: `auto-confirm off → confirmed:false`,
    `auto-confirm on → confirmed:true`, for all three tools.
  - preseg `rapid` queue logic (existing `deriveFastQueue` tests carried over).
- Manual browser verification (per project convention for UI): switch between all
  three tools, apply from each, confirm unconfirmed→confirmed transition, verify
  box vanishes on apply and labels full-res points, verify a drawn pipe re-opens
  its centerline graph, verify legacy cuboids still render, zero console errors,
  API calls succeed.
