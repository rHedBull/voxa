# Unified Label-mode Tools Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Rebuild voxa's Label mode so its labeling tools are one unified system — three selection tools (Presegment, Box, Draw) in one rail, one contextual options panel, and one shared apply→unconfirmed→confirm pipeline where a tool differs only in how it selects points.

**Architecture:** Frontend/UX refactor of `mode-label.jsx` plus one backend generalization. Backend: `centerline-apply` becomes a generic `apply-shape` endpoint that resolves any *shape* (`tube` | `obb`; `beam` later) to full-resolution point indices, then reassigns them — with per-shape membership + structure-sidecar hooks. Frontend: replace the `activeTool` constant + `subMode` state with a single `activeTool` state, render a tool rail in the viewport top strip and a tool-options panel in the left rail, and route every tool's apply through one instance-creation path that produces a `pointset` with a per-tool `auto-confirm` flag.

**Tech Stack:** Backend FastAPI + NumPy (pytest). Frontend React 18 + Three.js, no TypeScript (vitest, pure-function tests only). Spec: `docs/superpowers/specs/2026-07-10-unified-label-tools-design.md`.

**Reference reading before starting:** the spec above; `backend/routes/segment.py` (`centerline_apply`, `segment_apply`); `backend/labeling/centerline.py` (`tube_indices`); `frontend/src/mode-label.jsx` (the whole file); `frontend/src/api.js` (`segApply`, `centerlineApply`); `frontend/src/draw-mode.jsx`.

**Conventions:** Backend has no autoreload — restart `npm run dev` after Python edits. Run backend tests with `.venv/bin/pytest`; frontend with `npx vitest run` from `frontend/`. Commit after every task.

---

## Phase A — Backend: the generic `apply-shape` endpoint

Build and land this first: it is low-risk (additive + a behavior-preserving refactor of `centerline_apply`) and unblocks Box on the frontend.

### Task 1: `obb_indices` — full-res points inside an oriented box

**Files:**
- Create: `backend/labeling/shapes.py`
- Test: `backend/tests/test_shapes.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_shapes.py
import numpy as np
from labeling.shapes import obb_indices


def test_obb_axis_aligned_selects_interior():
    # 3x3x3 grid of points at integer coords 0..2 on each axis (27 pts).
    xs = np.array([0, 1, 2], dtype=np.float32)
    pts = np.array([[x, y, z] for x in xs for y in xs for z in xs],
                   dtype=np.float32).reshape(-1)
    # Box centered at (1,1,1), size 1.0 → half-extent 0.5 → only the center point.
    box = {"center": [1.0, 1.0, 1.0], "size": [1.0, 1.0, 1.0],
           "rotation": [0.0, 0.0, 0.0]}
    idx = obb_indices(pts, box)
    assert idx.tolist() == [13]  # the (1,1,1) point is index 13 in the grid


def test_obb_rotated_matches_local_frame():
    # A point offset +0.9 along world X. A box rotated 45° about Z with
    # half-extent 0.5 in local x should NOT contain it (0.9 > 0.5 in any frame),
    # but a box half-extent 0.5 centered ON it should.
    pts = np.array([0.9, 0.0, 0.0], dtype=np.float32)
    inside = {"center": [0.9, 0.0, 0.0], "size": [1.0, 1.0, 1.0],
              "rotation": [0.0, 0.0, np.pi / 4]}
    outside = {"center": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0],
               "rotation": [0.0, 0.0, np.pi / 4]}
    assert obb_indices(pts, inside).tolist() == [0]
    assert obb_indices(pts, outside).tolist() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest backend/tests/test_shapes.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'labeling.shapes'`

- [ ] **Step 3: Write minimal implementation**

```python
# backend/labeling/shapes.py
"""Shape → full-resolution point-index resolvers for the apply-shape endpoint.

Each labeling tool selects points via a geometric *shape*; the backend resolves
any shape to the indices of the full-res points inside it, then reassigns them.
The OBB math mirrors the frontend preview `pointsInsideOBBLabel` (mode-label.jsx)
so the applied label matches the on-screen selection box exactly.
"""
import numpy as np


def obb_indices(positions: np.ndarray, box: dict) -> np.ndarray:
    """Int32 indices of points inside the oriented box.

    box = {center:[3], size:[3] (full side lengths), rotation:[rx,ry,rz]}
    Rotation is Euler XYZ in radians, matching Three.js `Euler(..., 'XYZ')`.
    local = R^T · (p - center); a point is inside iff |local| <= size/2 per axis.
    """
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    cx, cy, cz = (float(v) for v in box["center"])
    sx, sy, sz = (float(v) for v in box["size"])
    rx, ry, rz = (float(v) for v in box["rotation"])
    hx, hy, hz = sx / 2.0, sy / 2.0, sz / 2.0

    cxr, sxr = np.cos(rx), np.sin(rx)
    cyr, syr = np.cos(ry), np.sin(ry)
    czr, szr = np.cos(rz), np.sin(rz)
    # Columns of R (world = R·local); local = R^T·(p-c) picks these as the
    # projection axes — identical basis to pointsInsideOBBLabel.
    ax0 = (cyr * czr,               cyr * szr,               -syr)
    ax1 = (sxr * syr * czr - cxr * szr, sxr * syr * szr + cxr * czr, sxr * cyr)
    ax2 = (cxr * syr * czr + sxr * szr, cxr * syr * szr - sxr * czr, cxr * cyr)

    rel = positions.astype(np.float64) - (cx, cy, cz)
    lx = rel @ np.asarray(ax0)
    ly = rel @ np.asarray(ax1)
    lz = rel @ np.asarray(ax2)
    inside = (np.abs(lx) <= hx) & (np.abs(ly) <= hy) & (np.abs(lz) <= hz)
    return np.nonzero(inside)[0].astype(np.int32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest backend/tests/test_shapes.py -v`
Expected: PASS (both tests)

- [ ] **Step 5: Commit**

```bash
git add backend/labeling/shapes.py backend/tests/test_shapes.py
git commit -m "feat(backend): add obb_indices shape resolver"
```

---

### Task 2: `shape_indices` dispatch (tube + obb)

**Files:**
- Modify: `backend/labeling/shapes.py`
- Test: `backend/tests/test_shapes.py`

- [ ] **Step 1: Write the failing test** — append:

```python
from labeling.shapes import shape_indices


def test_shape_indices_obb_dispatch():
    pts = np.array([0.9, 0.0, 0.0], dtype=np.float32)
    shape = {"type": "obb", "center": [0.9, 0.0, 0.0],
             "size": [1.0, 1.0, 1.0], "rotation": [0.0, 0.0, 0.0]}
    assert shape_indices(pts, shape).tolist() == [0]


def test_shape_indices_tube_dispatch_matches_tube_indices():
    from labeling.centerline import tube_indices
    pts = np.array([[0, 0, 0], [10, 0, 0]], dtype=np.float32).reshape(-1)
    paths = [{"points": [[0, 0, 0], [1, 0, 0]], "radius": 0.5, "smooth": False}]
    shape = {"type": "tube", "paths": paths}
    np.testing.assert_array_equal(
        shape_indices(pts, shape), tube_indices(np.asarray(pts), paths))


def test_shape_indices_unknown_type_raises():
    import pytest
    with pytest.raises(ValueError):
        shape_indices(np.zeros(3, dtype=np.float32), {"type": "blob"})
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest backend/tests/test_shapes.py -v`
Expected: FAIL — `ImportError: cannot import name 'shape_indices'`

- [ ] **Step 3: Implement** — append to `shapes.py`:

```python
def shape_indices(positions: np.ndarray, shape: dict) -> np.ndarray:
    """Resolve any shape descriptor to the int32 indices of the full-res points
    it contains. Dispatches on shape['type']."""
    kind = shape.get("type")
    if kind == "obb":
        return obb_indices(positions, shape)
    if kind == "tube":
        from labeling.centerline import tube_indices
        return tube_indices(np.asarray(positions), shape["paths"])
    raise ValueError(f"unknown shape type: {kind!r}")
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/pytest backend/tests/test_shapes.py -v`
Expected: PASS (all)

- [ ] **Step 5: Commit**

```bash
git add backend/labeling/shapes.py backend/tests/test_shapes.py
git commit -m "feat(backend): add shape_indices dispatch (tube + obb)"
```

---

### Task 3: `apply-shape` route + refactor `centerline_apply` to delegate

**Files:**
- Modify: `backend/app/schemas.py` (add `ApplyShapeRequest`)
- Modify: `backend/routes/segment.py` (add route + shared core; delegate `centerline_apply`)
- Test: `backend/tests/test_apply_shape.py` (new); existing `test_centerline_endpoints.py` must stay green

**Design note:** Extract a shared helper `_apply_shape_core(seg, shape, target_inst, target_class, merged_from)` that: resolves indices via `shape_indices`, runs `apply_reassign`, and runs the per-shape structure-persist hook (`tube` → `update_centerlines`; `obb` → nothing). Both the new `/api/segment/apply-shape` route and the existing `/api/segment/centerline-apply` route call it, so the tube path is behavior-unchanged and its tests keep passing.

- [ ] **Step 1: Write the failing test**

Use the SAME fixtures as `test_centerline_endpoints.py`: `client_with_loaded_annotated_scene` (an active session with the synthetic `annotated/demo` cloud loaded) and `scan_dir_for_loaded_scene`. There is **no** `client_with_session` fixture — do not invent one. The scene's exact coords are unknown in-test, so (as the centerline tests do with a giant tube) use a **huge OBB that swallows the whole cloud** to guarantee a non-empty result, and a **far-away OBB** for the empty case.

```python
# backend/tests/test_apply_shape.py
"""Tests for the generic /api/segment/apply-shape endpoint."""
from __future__ import annotations


def _obb_body(**over):
    body = {
        "shape": {"type": "obb", "center": [0.0, 0.0, 0.0],
                  "size": [1e7, 1e7, 1e7], "rotation": [0.0, 0.0, 0.0]},
        "target_class": "pipe", "target_inst": -1, "merged_from": [],
    }
    body.update(over)
    return body


def test_apply_shape_obb_reassigns_and_allocates(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/apply-shape", json=_obb_body())
    assert r.status_code == 200
    j = r.json()
    assert j["n_affected"] > 0
    assert j["instance_id"] == j["new_instance_id"]


def test_apply_shape_obb_far_away_is_empty(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/apply-shape", json=_obb_body(
        shape={"type": "obb", "center": [9e6, 9e6, 9e6],
               "size": [0.01, 0.01, 0.01], "rotation": [0.0, 0.0, 0.0]}))
    assert r.status_code == 200
    assert r.json()["n_affected"] == 0


def test_apply_shape_tube_parity_with_centerline_apply(client_with_loaded_annotated_scene):
    # tube shape must match the legacy centerline-apply on identical input.
    client = client_with_loaded_annotated_scene
    paths = [{"points": [[-1e5, -1e5, -1e5], [1e5, 1e5, 1e5]],
              "radius": 1e6, "smooth": False}]
    r1 = client.post("/api/segment/apply-shape",
                     json={"shape": {"type": "tube", "paths": paths},
                           "target_class": "pipe", "target_inst": -1,
                           "merged_from": []})
    client.post("/api/segment/undo")  # revert before the second apply
    r2 = client.post("/api/segment/centerline-apply",
                     json={"paths": paths, "target_class": "pipe",
                           "target_inst": -1, "merged_from": []})
    assert r1.json()["n_affected"] == r2.json()["n_affected"]


def test_apply_shape_unknown_type_400(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/apply-shape",
                    json={"shape": {"type": "blob"}, "target_class": "pipe"})
    assert r.status_code == 400
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest backend/tests/test_apply_shape.py -v`
Expected: FAIL — 404 (route not registered) / schema import error

- [ ] **Step 3: Implement**

In `backend/app/schemas.py`, add (near `CenterlineApplyRequest`):

```python
class ApplyShapeRequest(BaseModel):
    shape: dict            # {type:'tube'|'obb', ...} — validated in shape_indices
    target_class: int | str
    target_inst: int = -1
    merged_from: list[int] = []
```

In `backend/routes/segment.py`, add the shared core + route, and rewrite `centerline_apply` to delegate:

```python
def _apply_shape_core(seg, shape, target_inst, target_class, merged_from):
    from labeling.shapes import shape_indices
    from labeling.centerline import update_centerlines
    idx = shape_indices(np.asarray(seg.positions), shape)
    if idx.size == 0:
        return {"op": "apply-shape", "n_affected": 0, "dirty": bool(seg.dirty)}
    out = seg.apply_reassign(idx, target_inst=target_inst, target_class=target_class)
    instance_id = out.get("new_instance_id", target_inst)
    # Per-shape structure sidecar hook.
    if shape.get("type") == "tube":
        if seg.session_dir is None:
            raise HTTPException(409, "centerline labeling requires an active session")
        update_centerlines(seg.session_dir, instance_id, target_class,
                           shape["paths"], merged_from)
    body = _serialize_apply(out)
    body["instance_id"] = int(instance_id)
    return body


@router.post("/api/segment/apply-shape")
def apply_shape(req: ApplyShapeRequest):
    """Resolve a shape to full-res point indices and reassign them to a label.
    Generalizes centerline-apply; see the unified-label-tools design spec."""
    seg = _require_seg()
    try:
        target_class = _coerce_class_id(req.target_class)
    except ValueError as e:
        raise HTTPException(400, str(e))
    try:
        return _apply_shape_core(seg, req.shape, req.target_inst,
                                 target_class, req.merged_from)
    except ValueError as e:            # unknown shape type
        raise HTTPException(400, str(e))


@router.post("/api/segment/centerline-apply")
def centerline_apply(req: CenterlineApplyRequest):
    """Backward-compatible alias: a tube-shaped apply-shape call."""
    seg = _require_seg()
    if seg.session_dir is None:
        raise HTTPException(409, "centerline labeling requires an active session")
    try:
        target_class = _coerce_class_id(req.target_class)
    except ValueError as e:
        raise HTTPException(400, str(e))
    shape = {"type": "tube", "paths": [p.model_dump() for p in req.paths]}
    return _apply_shape_core(seg, shape, req.target_inst, target_class,
                             req.merged_from)
```

Ensure `ApplyShapeRequest` is imported in `routes/segment.py` alongside `CenterlineApplyRequest`, and `numpy as np` is already imported there (it is, per centerline_apply).

- [ ] **Step 4: Run tests**

Run: `.venv/bin/pytest backend/tests/test_apply_shape.py backend/tests/test_centerline_endpoints.py backend/tests/test_centerline.py -v`
Expected: PASS — new apply-shape tests pass AND all pre-existing centerline tests stay green (parity).

- [ ] **Step 5: Commit**

```bash
git add backend/app/schemas.py backend/routes/segment.py backend/tests/test_apply_shape.py backend/tests/test_centerline_endpoints.py
git commit -m "feat(backend): generic apply-shape endpoint; centerline-apply delegates"
```

---

### Task 4: `api.js` — `applyShape()`; route `centerlineApply` through it

**Files:**
- Modify: `frontend/src/api.js` (`centerlineApply` ~235; add `applyShape`)

**Design note:** Add `applyShape`, and reimplement `centerlineApply` to call the new endpoint with a `tube` shape so Draw's UI code stays untouched while its traffic moves onto `apply-shape`.

- [ ] **Step 1: Implement** (this is API glue; no separate unit test — covered by backend tests + browser verification)

```js
// frontend/src/api.js — add to VoxaAPI
async applyShape({ shape, targetClass, targetInst = -1, mergedFrom = [] }) {
  const r = await fetch('/api/segment/apply-shape', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ shape, target_class: targetClass,
                           target_inst: targetInst, merged_from: mergedFrom }),
  });
  if (!r.ok) throw new Error(`applyShape failed: ${r.status} ${await r.text()}`);
  // CRITICAL: decode like segApply/centerlineApply do. The wire format is
  // base64 (_serialize_apply → indices/after_class/after_instance strings +
  // instance_id). _decodeApplyResponse (api.js:258) yields the camelCase
  // decoded arrays every consumer expects: indices (Int32Array), afterClass,
  // afterInstance, newInstanceId. Also surface the endpoint's instance_id as
  // instanceId (what DrawMode + applyBox read).
  const j = await r.json();
  return { ..._decodeApplyResponse(j), instanceId: j.instance_id ?? null };
},

// Reimplement the existing centerlineApply as a tube-shaped apply-shape call.
// (Note: this bypasses CenterlineApplyRequest's Pydantic validation — ≥2 pts,
// radius>0 — which the kept /centerline-apply route still enforces. tube_indices
// tolerates degenerate paths (empty result), so a malformed path now no-ops
// instead of 422. Acceptable because DrawMode guards its input; noted for parity.)
async centerlineApply({ paths, targetClass, targetInst = -1, mergedFrom = [] }) {
  return this.applyShape({
    shape: { type: 'tube', paths }, targetClass, targetInst, mergedFrom });
},
```

Confirm `_decodeApplyResponse` is defined/in-scope in `api.js` (it is, ~:258, used by `segApply`/`centerlineApply`).

- [ ] **Step 2: Verify build** — Run: `cd frontend && npx vite build` (or rely on `npm run dev`); Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api.js
git commit -m "feat(frontend): applyShape() api; centerlineApply routes through it"
```

---

## Phase B — Frontend: tool state, rail, and options panel

### Task 5: `label-tools.js` — pure tool-state helpers

**Files:**
- Create: `frontend/src/label-tools.js`
- Test: `frontend/src/label-tools.test.js`

- [ ] **Step 1: Write the failing test**

```js
// frontend/src/label-tools.test.js
import { describe, it, expect } from 'vitest';
import { TOOLS, toolAvailable, defaultTool } from './label-tools.js';

describe('label-tools', () => {
  it('lists the three selection tools in rail order', () => {
    expect(TOOLS.map((t) => t.id)).toEqual(['presegment', 'box', 'draw']);
  });
  it('gates presegment on segState and draw on annotated', () => {
    const raw = { segState: null, isAnnotated: false };
    expect(toolAvailable('box', raw)).toBe(true);
    expect(toolAvailable('presegment', raw)).toBe(false);
    expect(toolAvailable('draw', raw)).toBe(false);
    const ann = { segState: {}, isAnnotated: true };
    expect(toolAvailable('presegment', ann)).toBe(true);
    expect(toolAvailable('draw', ann)).toBe(true);
  });
  it('defaults to presegment when available, else box', () => {
    expect(defaultTool({ segState: {}, isAnnotated: true })).toBe('presegment');
    expect(defaultTool({ segState: null, isAnnotated: false })).toBe('box');
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run: `cd frontend && npx vitest run src/label-tools.test.js`
Expected: FAIL — cannot find `./label-tools.js`

- [ ] **Step 3: Implement**

```js
// frontend/src/label-tools.js
// Single source of truth for the Label-mode tool rail. A tool is only a way to
// select points; downstream behavior is shared (see the unified-label-tools spec).
export const TOOLS = [
  { id: 'presegment', icon: '◱', label: 'Presegment' },
  { id: 'box',        icon: '▭', label: 'Box' },
  { id: 'draw',       icon: '✎', label: 'Draw' },
  // { id: 'beam', ... } reserved — not built yet.
];

export function toolAvailable(id, { segState, isAnnotated }) {
  if (id === 'presegment') return !!segState;
  if (id === 'draw') return !!segState && !!isAnnotated;
  return true; // box always available (works on raw clouds)
}

export function defaultTool(ctx) {
  return toolAvailable('presegment', ctx) ? 'presegment' : 'box';
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `cd frontend && npx vitest run src/label-tools.test.js`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/label-tools.js frontend/src/label-tools.test.js
git commit -m "feat(frontend): label-tools tool list + gating helpers"
```

---

### Task 6: Replace `activeTool` const + `subMode` with `activeTool` state

**Files:**
- Modify: `frontend/src/mode-label.jsx` (`:66`, `:89–91`, and every `subMode`/`fastMode`/`drawMode` reference)

**Design note:** This is a mechanical state swap that must keep behavior identical (still compiles and runs the same). Introduce `activeTool` state; derive `fastMode`/`drawMode` from it plus a new `presegRapid` flag so the rest of the file changes minimally in this task.

- [ ] **Step 1: Make the edit**

Replace line 66 and 89–91:

```js
// was: const activeTool = 'cuboid';
const [activeTool, setActiveTool] = useStateLabel(() =>
  defaultTool({ segState, isAnnotated }));
// Presegment "rapid" = the old fast-labeling queue.
const [presegRapid, setPresegRapid] = useStateLabel(false);

// Derived legacy flags — keep the existing body working during the refactor.
const fastMode = activeTool === 'presegment' && presegRapid;
const drawMode = activeTool === 'draw';

// Per-tool auto-confirm (introduced here, before Task 8/9 reference it, to
// avoid a forward reference). Threaded into the apply paths in Task 10.
const [autoConfirm, setAutoConfirm] = useStateLabel({ box: false, draw: false, presegment: false });
const autoConfirmFor = (tool) =>
  tool === 'presegment' ? (presegRapid || autoConfirm.presegment) : !!autoConfirm[tool];
```

Add the import at top: `import { TOOLS, toolAvailable, defaultTool } from './label-tools.js';`

Keep an effect that resets `activeTool` to an available tool when the scene/gating changes:

```js
useEffectLabel(() => {
  if (!toolAvailable(activeTool, { segState, isAnnotated })) {
    setActiveTool(defaultTool({ segState, isAnnotated }));
  }
}, [segState, isAnnotated, activeTool]);
```

Replace the old `setSubMode(...)` calls (left-rail buttons at `:1076`, `:1086`; `onExit` at `:1010`, `:1096`) — for this task, temporarily wire them to the new state so nothing breaks:
- Fast button → `setActiveTool('presegment'); setPresegRapid((v) => !v);`
- Draw button → `setActiveTool((t) => t === 'draw' ? 'presegment' : 'draw')`
- `onExit` handlers → `setPresegRapid(false)` (fast) / `setActiveTool('presegment')` (draw)

The `activeTool === 'cuboid'` guards at `:962`, `:967`, `:1133`, `:1193` will now never match (there is no `'cuboid'` tool). That is expected — those cuboid-only paths are removed in Task 11. For THIS task, replace `activeTool === 'cuboid'` with a temporary `activeTool === 'box'` so the gizmo/hotkeys keep working until Box is properly built, keeping the app functional between commits.

(Known transient between Tasks 6–11: `F`-frame and class-hotkeys only fire in Box mode during this window. Presegment apply is unaffected — Ctrl+Enter is gated on `segState.selection.size` at `:957`, not `activeTool`. Task 11 rewrites the handler and closes the gap.)

- [ ] **Step 2: Verify build + existing tests**

Run: `cd frontend && npx vitest run` then `npx vite build`
Expected: builds; vitest passes (no component tests depend on this).

- [ ] **Step 3: Manual smoke** — `npm run dev`, open Label mode, confirm the app loads and fast/draw buttons still toggle. (Full UI verification happens in Task 12/13.)

- [ ] **Step 4: Commit**

```bash
git add frontend/src/mode-label.jsx
git commit -m "refactor(frontend): activeTool state replaces cuboid const + subMode"
```

---

### Task 7: Tool rail in the viewport top toolbar

**Files:**
- Create: `frontend/src/tool-rail.jsx`
- Modify: `frontend/src/mode-label.jsx` (render rail inside `.vp-hud-top`, `:1152`)

- [ ] **Step 1: Implement the component**

**Note:** `ToolButton` (viewport-atoms.jsx:9) has no `disabled` prop, and passing `onClick={undefined}` still renders a focusable, enabled-looking button. So add a `disabled` prop to `ToolButton` (small, additive: `disabled` → `disabled` attr + a `.disabled` class, and skip `onClick`) and use it here.

```jsx
// frontend/src/tool-rail.jsx
import { TOOLS, toolAvailable } from './label-tools.js';
import { ToolButton } from './viewport-atoms.jsx';

// The single home for Label-mode tools: a 3-icon strip over the viewport.
export default function ToolRail({ activeTool, onSelect, ctx }) {
  return (
    <div className="tool-rail">
      {TOOLS.map((t) => {
        const ok = toolAvailable(t.id, ctx);
        return (
          <ToolButton key={t.id} mini icon={t.icon}
            label={ok ? t.label : `${t.label} (unavailable for this scan)`}
            active={activeTool === t.id}
            disabled={!ok}
            onClick={() => onSelect(t.id)} />
        );
      })}
    </div>
  );
}
```

Render it in `.vp-hud-top` (left `hud-group`, before Points-left chip):

```jsx
<div className="hud-group">
  <ToolRail activeTool={activeTool}
    onSelect={setActiveTool}
    ctx={{ segState, isAnnotated }} />
</div>
```

Add CSS for `.tool-rail` (flex row, small gap) and `.tool-btn.disabled` (dimmed, `pointer-events:none`) to **`frontend/src/app.css`** (the stylesheet holding `.vp-hud-top` / `.tool-btn` / `.hud-group`). Match sibling `.hud-group` styling.

- [ ] **Step 2: Verify** — `npx vite build`; then `npm run dev` and confirm the rail renders with 3 icons and switching updates `activeTool` (Draw/Preseg disabled on a raw scan).

- [ ] **Step 3: Commit**

```bash
git add frontend/src/tool-rail.jsx frontend/src/mode-label.jsx
git commit -m "feat(frontend): tool rail in viewport top toolbar"
```

---

### Task 8: Tool-options panel in the left rail

**Files:**
- Create: `frontend/src/tool-options.jsx`
- Modify: `frontend/src/mode-label.jsx` (left rail `:1071–1114` — move preseg/draw/fast controls into the panel)

**Design note:** `ToolOptions` renders the active tool's sub-controls in one place. It hosts: Presegment → `manual/rapid` toggle + `PresegmentList` + `auto-confirm` toggle; Box → gizmo controls + `auto-confirm` (built in Task 9); Draw → `<DrawMode>` (already self-contained) + `auto-confirm`. This task moves the *existing* preseg/draw controls into it (box options wired in Task 9).

- [ ] **Step 1: Implement `ToolOptions`** hosting the existing pieces, e.g.:

```jsx
// frontend/src/tool-options.jsx
import { PresegmentList } from './segment-tools.jsx';
import DrawMode from './draw-mode.jsx';

export default function ToolOptions(props) {
  const { activeTool } = props;
  if (activeTool === 'presegment') return <PresegOptions {...props} />;
  if (activeTool === 'draw') return <DrawOptions {...props} />;
  if (activeTool === 'box') return <BoxOptions {...props} />;  // Task 9
  return null;
}
```

`PresegOptions` renders the `manual/rapid` toggle (drives `presegRapid`), the `auto-confirm` toggle, and `<PresegmentList ... excludeSegIds={promotedSegIds} />`. `DrawOptions` renders `<DrawMode .../>` (props exactly as passed today at `:1091–1105`) plus the `auto-confirm` toggle. Thread `autoConfirm`/`setAutoConfirm`/`autoConfirmFor` from `mode-label` (defined in Task 6; toggles wired in Task 10).

- [ ] **Step 2: Wire into left rail** — replace the Fast/Draw buttons + inline `<DrawMode>` + `<PresegmentList>` (`:1071–1114`) with `<ToolOptions .../>`. Remove the now-unused `⚡`/`✏` left-rail buttons (tool switching is the rail now).

- [ ] **Step 3: Verify** — `npx vite build`; `npm run dev`: selecting each tool shows its options in the left rail; Presegment shows the list + rapid toggle; Draw shows the draw panel.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/tool-options.jsx frontend/src/mode-label.jsx
git commit -m "feat(frontend): contextual tool-options panel in left rail"
```

---

## Phase C — Unified apply pipeline

### Task 9: Box tool — OBB gizmo → `apply-shape('obb')` → pointset

**Files:**
- Create: `frontend/src/box-tool.jsx` (BoxOptions + apply logic) — or add `BoxOptions` inside `tool-options.jsx`
- Modify: `frontend/src/mode-label.jsx` (reuse `selBox`/`onCuboidTransform` OBB gizmo; add `applyBox`)

**Design note:** Reuse the existing `selBox` OBB + gizmo infrastructure (`:97`, `toggleBoxSelect :197`, `onCuboidTransform :916`). When `activeTool === 'box'`, the gizmo edits `selBox`; Box's options panel shows Move/Rotate/Scale + Auto-fit + Apply. On Apply, call `applyShape({ shape: { type:'obb', center, size, rotation }, targetClass })`, create the `pointset` instance from the returned `instance_id`, then clear `selBox` (box vanishes).

- [ ] **Step 1: Add `applyBox` in `mode-label.jsx`** (near `confirmSegmentSelection`):

```jsx
const applyBox = useCallbackLabel(async (clsDef) => {
  const targetCls = clsDef || activeClassDef;
  if (!selBox || !targetCls) return;
  let r;
  try {
    r = await VoxaAPI.applyShape({
      shape: { type: 'obb', center: selBox.center,
               size: selBox.size, rotation: selBox.rotation },
      targetClass: targetCls.class_id ?? targetCls.id,
    });
  } catch (err) { console.error('box apply failed:', err); return; }
  const segId = Number.isFinite(r.instanceId) ? r.instanceId : -1;  // decoded field
  if (segId >= 0) {
    onChange([...instances, {
      id: newId(), segId, kind: 'pointset', cls: targetCls.id,
      label: `${targetCls.label} ${(counts[targetCls.id] || 0) + 1}`,
      color: targetCls.color, source: 'box',
      confirmed: !!autoConfirmFor('box'),
    }]);
  }
  // Refresh working arrays from the returned delta so hulls/hide update.
  setSegState((s) => s ? applyDelta(s, {
    indices: r.indices, after_class: r.afterClass,
    after_instance: r.afterInstance }) : s);
  setSelBox(null);  // box vanishes on apply
}, [selBox, activeClassDef, instances, counts, onChange, setSegState]);
```

(Confirm the `apply-shape` response includes `indices`/`afterClass`/`afterInstance` from `_serialize_apply`; it does — same shape as `segApply`. If field names differ, match `_serialize_apply`'s keys.)

- [ ] **Step 2: `BoxOptions`** panel: buttons for Move/Rotate/Scale (`setTransformMode`), Auto-fit (reuse `autoFitSelected`-style on `selBox`), a `Draw box`/`Clear` toggle (`toggleBoxSelect`), an `Apply (Ctrl+Enter)` button (`applyBox()`), and the `auto-confirm` toggle. When `activeTool==='box'` and no `selBox`, show a "Draw a box" hint that calls `toggleBoxSelect`.

- [ ] **Step 3: Gizmo wiring** — set the Viewer `transformMode` for box: replace the temporary `activeTool === 'box'` from Task 6 so the gizmo targets `selBox` when `activeTool==='box'` (it already does via `selBox ? ... : ...` at `:1133`; ensure `selBox` is auto-initialized when entering Box, or via the Draw-box button).

- [ ] **Step 4: Route Ctrl+Enter / class-hotkey for box** — in the keydown handler (Task 11 finalizes), when `activeTool==='box'` and `selBox` exists, Ctrl+Enter opens the class picker whose pick calls `applyBox(cls)`.

- [ ] **Step 5: Verify (browser)** — load a scan with a session, pick Box, draw + fit a box over a cluster, Apply, confirm: a new unconfirmed pointset appears, the labeled full-res points recolor, and the box disappears. Check the network call to `/api/segment/apply-shape` returns 200.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/box-tool.jsx frontend/src/tool-options.jsx frontend/src/mode-label.jsx
git commit -m "feat(frontend): Box tool applies OBB via apply-shape → pointset"
```

---

### Task 10: Per-tool `auto-confirm` state + unify Draw/Fast confirm flag

**Files:**
- Modify: `frontend/src/mode-label.jsx` (`confirmSegmentSelection :769`, `onDrawApplied :828`, `fastConfirm :907`)

**Design note:** The `autoConfirm` map + `autoConfirmFor` helper were added in Task 6. This task only *threads them into the apply paths* (replacing hard-coded `confirmed:true`) and wires the per-tool toggles in the options panels.

- [ ] **Step 1: Wire the per-tool toggles** — each tool's options panel (PresegOptions/DrawOptions/BoxOptions from Task 8/9) renders an `auto-confirm on apply` checkbox bound to `autoConfirm[tool]` via `setAutoConfirm((m) => ({ ...m, [tool]: v }))`.

- [ ] **Step 2: Thread into apply paths**
  - `confirmSegmentSelection`: default `opts.confirmed` from `autoConfirmFor('presegment')` when called from the preseg apply (keep the explicit `{confirmed:true}` only from `fastConfirm`, or better: `fastConfirm` passes nothing and relies on `presegRapid`).
  - `onDrawApplied` (`:845`): replace `confirmed: true` with `confirmed: autoConfirmFor('draw')`.
  - `applyBox`: already uses `autoConfirmFor('box')`.

- [ ] **Step 3: Verify** — with auto-confirm off, Draw/Box/Preseg apply land as unconfirmed (○); toggling a tool's auto-confirm on makes its next apply land confirmed (✓). Rapid preseg still one-keypress-confirms.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/mode-label.jsx
git commit -m "feat(frontend): per-tool auto-confirm; unify Draw/Fast confirm flag"
```

---

### Task 11: Class-hotkey-applies + retire cuboid creation/gizmo

**Files:**
- Modify: `frontend/src/mode-label.jsx` (keydown handler `:945–1001`; `addCuboid :665`; ViewportToolbar cuboid block `:1193–1222`; help text `:601–645`)

**Design note:** With no `'cuboid'` tool, remove cuboid *creation* and its hotkeys/toolbar. Legacy cuboid instances remain viewable/selectable/confirmable/deletable but get no gizmo. Add: pressing a class hotkey while a tool selection exists applies+labels it.

- [ ] **Step 1: Keydown handler rewrite**
  - Keep the `fastMode || drawMode` early-return (draw/rapid own the keyboard).
  - Ctrl+Enter: if `segState.selection.size > 0` → open class picker (preseg apply); else if `activeTool==='box' && selBox` → open class picker (box apply). Remove the `activeTool==='cuboid'` `toggleConfirmSelected` branch (no cuboid tool). Keep `toggleConfirmSelected` reachable via the row ✓ button.
  - Class hotkey: if a tool selection is active (`segState.selection.size>0` for preseg, or `selBox` for box), apply with that class honoring auto-confirm — `confirmSegmentSelection(cls)` / `applyBox(cls)`. Otherwise just `setActiveClass(cls.id)`. Remove the "reclass selected cuboid" behavior.
  - Remove the `A` (addCuboid), `G/R/Y` gizmo-mode, and `D` densify hotkeys (cuboid-only). Keep `F` (frame selection) — still useful for any selected instance.
  - Delete `addCuboid`, `deleteSelected`'s cuboid assumptions stay fine (works on any instance).

- [ ] **Step 2: Remove the cuboid ViewportToolbar block** (`:1193–1222`) — the Move/Rotate/Scale/Focus/Auto-fit/Delete floating buttons. Box's equivalents now live in BoxOptions; legacy cuboids are read-only. Keep the Diff (`Δ`) and Reset-cam buttons. Remove the `◫` box-select-segments button only if superseded — **keep** it if Presegment still offers centroid box-select in its options (move it into PresegOptions); otherwise remove.

- [ ] **Step 3: Update help text** (`helpSections`) — replace the "Cuboid" section with a "Tools" section (rail switching, apply = Ctrl+Enter / class hotkey, confirm). Keep Class assignment + Camera sections.

- [ ] **Step 4: Update the empty-state** (`:1297`) — "No instances yet. Press A to add." → "No instances yet. Pick a tool, select points, and apply."

- [ ] **Step 5: Verify (browser)** — legacy-cuboid scan: existing boxes render, are selectable and deletable, but show no gizmo. New scan: hotkey applies the current selection; A does nothing.

- [ ] **Step 6: Commit**

```bash
git add frontend/src/mode-label.jsx
git commit -m "feat(frontend): class-hotkey applies; retire cuboid creation + gizmo"
```

---

## Phase D — Instances filter + verification

### Task 12: Instances filter — unconfirmed / confirmed / all

**Files:**
- Modify: `frontend/src/mode-label.jsx` (`filteredInstances :588`, filter UI `:1284–1294`)

- [ ] **Step 1: Add a status filter** — a 3-way segmented control (`all | unconfirmed | confirmed`) beside the text filter. Extend `filteredInstances` to also filter on `inst.confirmed` per the selected status. Keep the single flat list (no section split).

- [ ] **Step 2: Verify** — filter narrows the list correctly; text + status filters compose.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/mode-label.jsx
git commit -m "feat(frontend): unconfirmed/confirmed/all filter on instances panel"
```

---

### Task 13: Full-stack verification + docs

**Files:**
- Modify: `CLAUDE.md` (Label-mode description: three tools, apply-shape), `docs/scan-schema.md` if it references centerline-apply
- Verify: whole flow

- [ ] **Step 1: Backend tests** — Run: `.venv/bin/pytest backend/` ; Expected: all pass (new shape/apply-shape + unchanged centerline).

- [ ] **Step 2: Frontend tests** — Run: `cd frontend && npx vitest run` ; Expected: all pass.

- [ ] **Step 3: Browser verification** (use the `browser-verification` skill) — for a scan with presegs + a session:
  - Rail switches between Presegment / Box / Draw; unavailable tools disabled on a raw scan.
  - Presegment: select segments → Ctrl+Enter / hotkey → unconfirmed pointset; rapid mode one-keypress confirms.
  - Box: draw + fit box → apply → full-res points labeled, box vanishes, unconfirmed pointset.
  - Draw: draw a pipe → apply → pointset; re-open loads its centerline graph; `/api/segment/apply-shape` (tube) 200.
  - Confirm toggles; instances status filter; legacy cuboids render read-only.
  - Zero console errors; all network calls 200.
  - Capture screenshots of each tool's options panel + a completed apply.

- [ ] **Step 4: Update docs** — `CLAUDE.md` Label-mode paragraph (three unified tools, `apply-shape`); note `apply-shape` in any doc that documented `centerline-apply`.

- [ ] **Step 5: Commit**

```bash
git add CLAUDE.md docs/
git commit -m "docs: unified Label-mode tools + apply-shape endpoint"
```

---

## Done criteria

- One rail switches Presegment / Box / Draw; one options panel per tool.
- All three apply through one pipeline to a `pointset`, unconfirmed by default, with per-tool auto-confirm.
- Box labels full-res points via `apply-shape('obb')` and vanishes on apply.
- Draw traffic flows through `apply-shape('tube')`; its centerline graph still persists + re-edits.
- Legacy cuboids display-only; no cuboid creation remains.
- Backend + frontend tests green; browser verification clean.
