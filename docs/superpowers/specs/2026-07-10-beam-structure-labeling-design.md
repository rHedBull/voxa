# Beam / structure labeling — design

Date: 2026-07-10
Status: approved by user (brainstorming session)

## Problem

Steel structures (frames, trusses) are made of many straight prismatic members —
beams, pillars — that meet at shared joints. Labeling each one point-by-point or
via presegments is slow, and presegment pipelines fragment long members. A
straight member is described almost completely by its two endpoints plus a
cross-section size, and members share joints. This feature lets the user build a
lightweight **node/edge graph** over the structure: place joints once, connect
them into beams, and extract the surrounding points as labeled instances.

Like the pipe tool it descends from, this mode targets **blank sessions** — a
fresh cloud with no presegments and no prior labels — and must not require a
preseg to exist. It is class-agnostic (any class from `classes.yaml`); the
default class is `beam` (`config/classes.yaml`, hotkey `9`), and pillars/other
linear members use their own hotkeys per edge.

## Solution overview

A new **Beam** sub-mode of Label mode (sibling of Fast-labeling and Draw). It sits
directly on the pipe tool's proven infrastructure but models a graph rather than
ordered centerline polylines.

- **Nodes** = connection points placed on the cloud by raycast pick.
- **Edges** = connections between two nodes; each edge is one beam → a **square
  box** swept from node A to node B, captured on the backend. One edge = one
  instance.
- **Width** = the square cross-section size (full side length interpreted as a
  half-extent per side of the axis — see Extraction), set per edge by the mouse
  wheel. A new edge inherits the last-used width.

An isolated beam is the degenerate case (2 nodes, 1 edge), so this one model
covers everything. Members are assumed **straight** — no intermediate control
points, no curved paths.

### Workflow

1. Enter the Beam sub-mode (button next to Fast-labeling / Draw). While active,
   the sub-mode owns the keyboard (capture-phase, like `FastLabelKeys` /
   `DrawKeys`). The cloud renders as raw RGB with the point-size slider and walk
   nav, same as Draw.
2. Build the graph. **Ctrl is the "edit graph" modifier; plain mouse stays the
   camera** (orbit / zoom):
   - **Ctrl+click empty cloud** → create a node at the picked point. A new node
     is **not** auto-selected.
   - **Ctrl+click a node** → if a *different node* is already selected, create a
     beam edge between the two and select the new beam; otherwise select this
     node. Ctrl+click within a screen-space snap radius of an existing node
     reuses it (watertight joints).
   - **Ctrl+click a beam** → select that beam.
3. Size / classify the selected beam:
   - **Scroll** over a selected beam → resize its width; `+` / `-` nudge in small
     steps; a numeric field in the panel gives exact entry. With nothing selected,
     scroll is camera zoom.
   - **Class hotkey** (`9` beam, `q` pillar, …) → set the selected beam's class.
4. Edit geometry:
   - **Drag a selected node** → moves it on a screen-parallel plane; all beams
     incident to it follow; orbit suppressed mid-drag.
   - **Del / Backspace** → delete the selected node (cascades to its beams) or the
     selected beam.
   - **Esc** → clear selection; Esc with nothing selected exits the sub-mode.
5. **Enter** → apply the whole active graph: the backend extracts the points in
   every beam's box and assigns class + one instance per beam through the existing
   reassign flow. Each applied beam creates / refreshes an **unconfirmed pointset
   instance** in the right-hand Instances panel. Beams stay editable — drag a joint
   and press Enter again to re-apply (same instance IDs).
6. **Ctrl+Enter** → commit the batch. Pops a confirm dialog
   **"Commit N beams to unconfirmed instances?"**. On Yes:
   1. Any beam not yet applied is applied first (same as Enter).
   2. Every active beam's geometry is retired into a **committed layer** and
      removed from the active graph (canvas clears), so you continue building the
      next section fresh.
   3. Committed beams render as **faded, read-only boxes** behind a **◌/●**
      show/hide toggle (same affordance as the pipe sections). Re-editing a
      committed beam is done via the Instances panel, not the graph.
   The unconfirmed pointset instances created in step 5 remain in the panel and
   join the normal review/confirm workflow.
7. Undo (existing `segUndo`) reverts an apply's per-point labels.

### Key & mouse map (Beam sub-mode active)

| Input | Effect |
|---|---|
| Ctrl+click empty cloud | create a node at the pick (snaps to a nearby node) |
| Ctrl+click a node | different node selected → create beam + select it; else select this node |
| Ctrl+click a beam | select that beam |
| Drag selected node | move it (screen-parallel plane); incident beams follow; orbit suppressed |
| Scroll | selected beam → resize its width; else camera zoom |
| `+` / `-` | nudge selected beam's width |
| class hotkey | set selected beam's class |
| `Enter` | apply whole active graph (one instance per beam; refresh pointset rows) |
| `Ctrl+Enter` | commit batch: apply, retire geometry to committed layer, clear active graph |
| `Del` / `Backspace` | delete selected node (cascades to its beams) or selected beam |
| `Esc` | clear selection; with nothing selected, exit the sub-mode |

Raycast pick priority on Ctrl+click: **node sphere > beam box > cloud / empty**.

Extraction semantics: every point within the box of a beam gets that beam's
class + instance — later applies overwrite earlier ones where boxes overlap
(latest wins, e.g. at shared joints). With the blank-session workflow this is
the natural behavior, and undo covers mistakes.

## Architecture

Frontend owns the graph-building UX; backend owns extraction over the
full-resolution cloud (the viewport cloud may be subsampled, so frontend-side
extraction would leave holes at full resolution). This mirrors the pipe tool
exactly.

### Frontend (`frontend/src/`)

- **`beam-mode.jsx`** (new) — the sub-mode component, structured like
  `draw-mode.jsx`: `BeamKeys` (capture-phase key driver), `BeamHUD`,
  `BeamOverlay` (Three.js overlay + pointer interactions), `BeamPanel` (edge
  list, width field, point-size slider, ◌/● toggle, Apply / Commit buttons).
  Loads stored `structure.json` on open so the active graph + committed layer
  render. Hosts the Ctrl+Enter confirm popup (via `window.confirm`, decided on
  live state *outside* any React updater — same rule `draw-mode.jsx` follows for
  Esc/Backspace).
- **`beam-graph.js`** (new, pure / vitest-testable) — the graph state machine.
  No Three.js, no React (same testability contract as `draw-paths.js`). Shape:

  ```
  nodes:          [ { id, pos:[x,y,z] } ]            // recentered frame
  edges:          [ { id, a:nodeId, b:nodeId, width, classId, instanceId|null } ]
  committed:      [ { a:[x,y,z], b:[x,y,z], width, classId, instanceId } ]
  selection:      { kind:'node'|'edge', id } | null
  lastWidth:      number
  defaultClassId: number
  ```

  Pure operations: `addNode`, `snapOrAddNode(pos, snapNodeId?)`, `addEdge(aId,
  bId)`, `moveNode(id, pos)`, `deleteNode(id)` (cascades incident edges),
  `deleteEdge(id)`, `setWidth(id, w)` / `nudgeWidth(dir)`, `setClass(classId)`,
  `select(kind, id)` / `clearSelection`, `markApplied(edgeId, instanceId)`,
  `commitAll()` (move active edges → `committed`, clear active nodes/edges),
  `seedFromServer(doc)`, and `buildApplyPayload()` → the beam-apply request.
  `addEdge` de-dups: connecting an already-connected node pair is a no-op.
- **Viewer integration** (inside `BeamOverlay`, reusing `viewer.jsx` seams) —
  render nodes (small spheres, pick-priority targets), active beams (oriented
  `BoxGeometry` in the class color, selected one gets the white back-side rim
  shell borrowed from `draw-mode.jsx`), and committed beams (faded read-only
  boxes, `raycast = () => {}` so they never swallow picks) when the ◌/● toggle is
  on. Node drag moves the node on a camera-parallel plane; the box re-renders on
  release (sphere tracks live), same alloc-thrash avoidance as the pipe drag.
  The scroll-resize-vs-zoom gate reuses the capture-listener-on-parent trick.
- **`api.js`** — one new call, `beamApply(beams, targetClasses)` (see backend).
- **`mode-label.jsx`** — add the Beam sub-mode toggle; reuse the existing
  `onDrawApplied` pointset-row surfacing (rename to a shared `onGraphApplied` if
  cleaner, but the mechanism is unchanged) so applied/committed beams become
  unconfirmed pointset instances. Host `BeamPanel` in the left sidebar like
  `DrawPanel`.

### Backend

- **`backend/labeling/beams.py`** (new) — oriented-box extraction + persistence,
  analogous to `centerline.py`:
  - `box_indices(positions, a, b, width) -> np.ndarray[int32]`: unique full-res
    indices inside the oriented box for one beam. Build an orthonormal frame
    `(u, v, w)` where `u = normalize(b - a)` (guard the degenerate `|b-a| < eps`
    beam → skip), and `v, w` any stable pair perpendicular to `u` (e.g. cross
    with a reference axis, falling back to a second axis when `u` is parallel to
    the first). A point `p` is inside iff `0 ≤ (p-a)·u ≤ |b-a|` **and**
    `|(p-a)·v| ≤ width/2` **and** `|(p-a)·w| ≤ width/2`. AABB prefilter on the
    box's expanded bounds first (same pattern as `tube_indices`) so we don't
    allocate O(N) temporaries per beam. The box orientation about `u` (roll) is
    unspecified by design — the user chose a square section, so any stable frame
    is acceptable.
  - `load_structure(session_dir) -> dict` / `update_structure(...)` — read/write
    `structure.json` via `atomic_write_json` (same helper `centerline.py` uses).
    `update_structure` replaces the active graph and the committed list wholesale
    from the frontend's post-commit state (the frontend is the source of truth
    for graph geometry; the backend owns only point labels).
- **Endpoint `POST /api/segment/beam-apply`** (in `backend/routes/segment.py`,
  beside `centerline-apply`):

  ```json
  {
    "beams": [
      {"a": [x,y,z], "b": [x,y,z], "width": 0.2,
       "class_id": 9, "target_inst": -1}
    ]
  }
  ```

  - One call carries the whole active graph. Each beam is extracted and applied
    independently as its own instance via `apply_reassign(indices, target_inst,
    target_class)`. `target_inst: -1` allocates a new instance ID (existing
    `reassign` convention); a real ID re-applies that beam (drag-a-joint + Enter).
  - Coordinates are in the recentered frame (the frame the viewer and cuboid /
    centerline endpoints already use).
  - Response: a list of per-beam results, each the same shape as the
    `centerline-apply` / `segment_apply` delta (`n_affected`, `after_class`,
    `after_instance`, `new_instance_id`, `indices`). As in `_serialize_apply`,
    `after_class` / `after_instance` are **absent** when `n_affected == 0` — the
    frontend must not assume those keys exist and shows a "no points in box"
    toast for empty beams (still allocating no instance).
  - No active session / wrong tier → existing 400/409 behavior of the segment
    routes applies untouched.
- **Structure persistence endpoint** — `GET /api/segment/structure` returns the
  stored `structure.json` (frontend seeds the graph + committed layer on open),
  and `PUT /api/segment/structure` writes the current graph geometry after
  apply/commit/edit. Geometry writes are separate from point-label writes: labels
  go through `apply_reassign` (undo-covered, saved with the session), geometry
  goes to `structure.json` (not on the undo stack).

### Persistence

Per session, `sessions/<id>/structure.json`:

```json
{
  "nodes": [ {"id": 1, "pos": [x,y,z]} ],
  "edges": [ {"id": 5, "a": 1, "b": 2, "width": 0.2, "class_id": 9, "instance_id": null} ],
  "committed_beams": [
    {"a": [x,y,z], "b": [x,y,z], "width": 0.2, "class_id": 9, "instance_id": 42}
  ]
}
```

- Active `nodes` / `edges` survive reload (resume building where you left off);
  `committed_beams` is the geometry the ◌/● toggle draws. Committed beams bake in
  their endpoint positions (the graph structure is not needed after commit — only
  the box geometry, to render for show/hide).
- Written on apply, commit, and geometry edits; **not on the undo stack**. A
  wrong commit is fixed via the Instances panel or by re-labeling, matching how
  `centerlines.json` behaves. `segUndo` reverts only the per-point labels of an
  apply.
- Loaded when the Beam sub-mode opens.

## Error handling

- A beam's box captures 0 points → that beam's per-beam result has
  `n_affected: 0`; frontend shows a "no points in box" toast and allocates no
  instance for it. Other beams in the same call still apply.
- Degenerate beam (`|b - a| < eps`, e.g. both endpoints snapped to one node)
  → skipped server-side with `n_affected: 0`; client also guards `addEdge`
  against `a == b`.
- Width must be > 0 (client guard + backend 422 via Pydantic).
- Commit with an empty active graph → no-op (popup not shown, or shows "nothing
  to commit").
- No active session / wrong tier → existing segment-route 400/409 behavior.

## Reuse map (build on pipe infrastructure)

Reused **as-is**: `segment_state.apply_reassign` (undo stack, delta, dirty,
Ctrl+S save); `atomic_write_json`; the `onDrawApplied` unconfirmed-pointset-row
surfacing in `mode-label.jsx`; viewer seams (`attachOverlayGroup`,
`firstHitUnderCursor`, `evtToNdc`, `getCamera`, `setOrbitEnabled`); the
wheel-capture-on-parent scroll-resize trick; the ◌/● hide-toggle affordance and
CSS; LabelMode's raw-RGB cloud + point-size slider + walk nav + capture-phase key
driver pattern.

**New**, mirroring a pipe file without modifying it: `beam-graph.js` (vs.
`draw-paths.js`), `beam-mode.jsx` (vs. `draw-mode.jsx`), `backend/labeling/
beams.py` (vs. `centerline.py`), `POST /api/segment/beam-apply` +
`GET`/`PUT /api/segment/structure` (vs. `centerline-apply` /
`centerlines`). `draw-paths.js` and `centerline.py` are left untouched — a graph
and ordered polylines are different models and sharing one file would tangle them.

## Out of scope (explicitly deferred)

- Rectangular (width × height) cross-sections and roll orientation — the user
  chose a single square size; the data format carries one `width` per edge.
- Curved / multi-segment members (this tool is straight-only; curved members
  stay in the Draw / centerline tool).
- Auto-snap of a beam axis / width to a local fit of the points (a RANSAC line /
  box fit could snap a rough beam later; the beam data format already carries
  everything needed).
- Merging several beams into one instance — each edge is its own instance; a
  member occluded into pieces is labeled as separate beams (or re-labeled over).
- Re-editing a committed beam from the graph — it is retired to the Instances
  panel on commit; re-open via that panel or re-label.

## Testing

- **Backend** (pytest, `backend/tests/`):
  - `box_indices` on a synthetic beam cloud: points inside vs. outside the box,
    points just past each face (axis extent and both perpendicular half-extents),
    an axis not aligned to any world axis, the degenerate zero-length beam.
  - Route test: `beam-apply` with several beams → working arrays updated, one
    instance per beam, `segUndo` reverts; per-beam `n_affected: 0` case; validation
    errors (width ≤ 0, empty beams list).
  - `structure.json` written by `PUT /structure` and re-loadable by
    `GET /structure`.
- **Frontend** (vitest, pure-function): `beam-graph.js` state machine —
  add/snap node, add edge (incl. de-dup and `a == b` guard), move node, delete
  node cascades incident edges, delete edge, set / nudge width, set class,
  `markApplied`, `commitAll` moves edges to committed and clears the active
  graph, payload serialization, `seedFromServer` round-trip.
- **Manual browser verification** of the graph-building UX (node place, snap,
  connect, drag joint with beams following, scroll-resize, Enter apply,
  Ctrl+Enter commit + ◌/● toggle, undo) before calling it done.
