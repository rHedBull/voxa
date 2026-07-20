# Prism Box Labeling Tool — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a "Prism" Label-mode tool that selects points by drawing a footprint polygon on a horizontal plane and extruding it to a height.

**Architecture:** A prism is a new geometric *shape* (`{type:'prism', polygon, y0, height}`) resolved to full-res point indices by the **existing** `POST /api/segment/apply-shape` endpoint — no new endpoint. Containment is exact: XZ point-in-polygon × Y-band. Backend `prism_indices` is mirrored by a frontend `pointsInsidePrism` (parity-locked by a shared fixture) for the live preview. The prism volume persists on the pointset instance so raw exports re-rasterize it exactly, matching Box/Beam. The frontend tool is a Beam-style sub-mode (`prism-mode.jsx`) that owns draw state, renders a Three.js overlay via `viewerRef.attachOverlayGroup()`, and applies through the shared pipeline.

**Tech Stack:** Python/FastAPI + numpy (backend), React 18 + Three.js (frontend, no TypeScript), pytest + vitest.

**Spec:** `docs/superpowers/specs/2026-07-20-prism-box-labeling-tool-design.md`

**Branch:** `feat/prism-box-tool` (already checked out — this is the dedicated branch; no worktree needed).

**Conventions:** All shape coordinates are in the **recentered frame** (like OBB/tube). The plane is **world-horizontal, Y-up**: the polygon lives in the XZ plane, `y0` is its height, extrusion grows **up** to `y0+height`. Restart `npm run dev` after backend edits (no autoreload). Run backend tests with `.venv/bin/pytest`, frontend with `npx vitest run` from `frontend/`.

---

## File Structure

**Backend (modify):**
- `backend/labeling/shapes.py` — add `prism_indices()`; add `prism` branch to `shape_indices()`.
- `backend/app/schemas.py` — add `prism` field to `Cuboid`; update `ApplyShapeRequest.shape` doc.
- `backend/labeling/materialize.py` — `collect_volumes()` emits prism volumes; `replay_labels()` resolves them.

**Frontend (create):**
- `frontend/src/prism-geom.js` — `pointsInsidePrism()` (mirror of backend `prism_indices`) + `pointInPolygonXZ()`.
- `frontend/src/prism-mode.jsx` — `PrismMode` (sub-mode: `PrismKeys` + `PrismOverlay` + `PrismPanel`).

**Frontend (modify):**
- `frontend/src/label-tools.js` — add the `prism` tool + gating.
- `frontend/src/tool-options.jsx` — add `PrismOptions`, wire into the `ToolOptions` dispatch.
- `frontend/src/mode-label.jsx` — add `prismMode` to `subModeOwnsInput`; extend `onToolApplied` with a `prism` volume param.

**Tests (create/modify):**
- `backend/tests/test_shapes.py` (modify), `backend/tests/test_apply_shape.py` (modify), `backend/tests/test_materialize.py` (modify).
- `frontend/src/prism-geom.test.js` (create), `frontend/src/label-tools.test.js` (modify).

---

## Task 1: Backend `prism_indices` resolver

**Files:**
- Modify: `backend/labeling/shapes.py`
- Test: `backend/tests/test_shapes.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_shapes.py`:

```python
from labeling.shapes import prism_indices


def _grid_xz(y_values):
    # Points on a 5x5 XZ grid (x,z in 0..4) at each given y. Row-major:
    # index = yi*25 + x*5 + z.
    pts = []
    for y in y_values:
        for x in range(5):
            for z in range(5):
                pts.append([float(x), float(y), float(z)])
    return np.asarray(pts, dtype=np.float32).reshape(-1)


def test_prism_square_footprint_and_y_band():
    # Square footprint covering x,z in [1,3]; band y in [0,1]. Two y-layers at
    # y=0 (inside band) and y=5 (outside band).
    pts = _grid_xz([0.0, 5.0])
    prism = {"polygon": [[1, 1], [3, 1], [3, 3], [1, 3]], "y0": 0.0, "height": 1.0}
    idx = prism_indices(pts, prism)
    got = set(idx.tolist())
    # Expect the 3x3 block x in {1,2,3}, z in {1,2,3} at y=0 only.
    expect = {0 * 25 + x * 5 + z for x in (1, 2, 3) for z in (1, 2, 3)}
    assert got == expect


def test_prism_concave_L_footprint():
    # Concave L: excludes the top-right quadrant. A box cannot express this.
    pts = _grid_xz([0.0])
    poly = [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]]
    prism = {"polygon": poly, "y0": -0.5, "height": 1.0}
    idx = set(prism_indices(pts, prism).tolist())
    # (3,3) is in the excluded notch -> out; (1,3) and (3,1) are in -> in.
    assert (0 * 25 + 3 * 5 + 3) not in idx     # x=3,z=3 excluded
    assert (0 * 25 + 1 * 5 + 3) in idx         # x=1,z=3 included
    assert (0 * 25 + 3 * 5 + 1) in idx         # x=3,z=1 included


def test_prism_empty_when_band_misses():
    pts = _grid_xz([0.0])
    prism = {"polygon": [[1, 1], [3, 1], [3, 3], [1, 3]], "y0": 10.0, "height": 1.0}
    assert prism_indices(pts, prism).tolist() == []


def test_prism_degenerate_polygon_selects_nothing():
    pts = _grid_xz([0.0])
    prism = {"polygon": [[1, 1], [2, 2]], "y0": 0.0, "height": 1.0}  # < 3 verts
    assert prism_indices(pts, prism).tolist() == []


def test_shape_indices_dispatches_prism():
    pts = _grid_xz([0.0])
    shape = {"type": "prism", "polygon": [[1, 1], [3, 1], [3, 3], [1, 3]],
             "y0": 0.0, "height": 1.0}
    assert len(shape_indices(pts, shape)) == 9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_shapes.py -k prism -v`
Expected: FAIL with `ImportError: cannot import name 'prism_indices'`.

- [ ] **Step 3: Implement `prism_indices`**

Add to `backend/labeling/shapes.py` (after `obb_indices`):

```python
def prism_indices(positions: np.ndarray, prism: dict) -> np.ndarray:
    """Int32 indices of points inside a vertical prism.

    prism = {polygon:[[x,z],...] (>=3 verts, XZ world plane),
             y0: base-plane height, height: upward extrusion (>0)}
    A point is inside iff its (x,z) is inside the polygon AND
    y0 <= y <= y0 + height. Concave polygons are supported (ray-cast rule);
    < 3 vertices or a zero/negative height select nothing.
    """
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    poly = np.asarray(prism["polygon"], dtype=np.float64)
    height = float(prism["height"])
    if poly.shape[0] < 3 or height <= 0.0:
        return np.empty(0, dtype=np.int32)
    y0 = float(prism["y0"])
    y = positions[:, 1].astype(np.float64)
    in_band = (y >= y0) & (y <= y0 + height)
    inside = in_band.copy()
    if in_band.any():
        px = positions[in_band, 0].astype(np.float64)
        pz = positions[in_band, 2].astype(np.float64)
        inside[in_band] = _point_in_polygon_xz(px, pz, poly)
    return np.nonzero(inside)[0].astype(np.int32)


def _point_in_polygon_xz(px: np.ndarray, pz: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Vectorized even-odd ray-cast point-in-polygon for arrays of (px,pz)
    against polygon vertices `poly` (M×2, columns = x,z). Points exactly on an
    edge are treated as inside-or-out per the standard crossing rule (not
    guaranteed either way — acceptable at LiDAR density)."""
    x1 = poly[:, 0]; z1 = poly[:, 1]
    x2 = np.roll(x1, -1); z2 = np.roll(z1, -1)
    inside = np.zeros(px.shape[0], dtype=bool)
    for i in range(poly.shape[0]):
        # Edge (x1[i],z1[i]) -> (x2[i],z2[i]). A horizontal ray in +x from the
        # point crosses this edge iff the edge straddles pz and the crossing x
        # is to the right of px.
        cond = ((z1[i] > pz) != (z2[i] > pz))
        # Avoid div-by-zero on horizontal edges (cond is already False there).
        denom = np.where(z2[i] != z1[i], z2[i] - z1[i], 1.0)
        xints = x1[i] + (pz - z1[i]) * (x2[i] - x1[i]) / denom
        inside ^= cond & (px < xints)
    return inside
```

- [ ] **Step 4: Add the `prism` dispatch branch**

In `shape_indices`, before the `raise ValueError`:

```python
    if kind == "prism":
        return prism_indices(positions, shape)
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest backend/tests/test_shapes.py -k prism -v`
Expected: PASS (5 tests).

- [ ] **Step 6: Commit**

```bash
git add backend/labeling/shapes.py backend/tests/test_shapes.py
git commit -m "feat(backend): prism_indices shape resolver (XZ polygon x Y-band)"
```

---

## Task 2: `apply-shape` accepts prism + schema doc

**Files:**
- Modify: `backend/app/schemas.py` (doc comment only)
- Test: `backend/tests/test_apply_shape.py`

The route already dispatches through `shape_indices`, so once Task 1 lands the endpoint resolves `prism` automatically. This task adds a route-level regression test and updates the descriptor docs.

- [ ] **Step 1: Write the failing test**

Read `backend/tests/test_apply_shape.py` first to reuse its session fixture (look for how existing tests build a `SegmentSession` and POST `/api/segment/apply-shape` — mirror that fixture exactly). Add a test alongside the OBB apply tests:

```python
def test_apply_shape_prism_labels_enclosed_points(client, seg_session):
    # seg_session: the existing fixture that loads a small cloud + active
    # session. Reuse whatever the OBB apply test uses; adapt coords to the
    # fixture's point extents.
    shape = {"type": "prism",
             "polygon": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
             "y0": -1.0, "height": 2.0}
    resp = client.post("/api/segment/apply-shape", json={
        "shape": shape, "target_class": 1, "protect_instances": []})
    assert resp.status_code == 200
    body = resp.json()
    assert body["op"] == "apply-shape"
    assert body["n_affected"] >= 0   # exact count asserted against the fixture cloud
```

(When writing the real test, assert an exact `n_affected` using the fixture's known point positions, the way the OBB apply test asserts exact counts.)

- [ ] **Step 2: Run to verify it fails or passes**

Run: `.venv/bin/pytest backend/tests/test_apply_shape.py -k prism -v`
Expected: PASS once the assertion matches the fixture (the code path already exists via Task 1). If it errors on an unknown shape, Task 1 was not applied.

- [ ] **Step 3: Update descriptor docs**

In `backend/app/schemas.py`, update the two comments referencing shape types:
- `ApplyShapeRequest.shape`: `# {type:'tube'|'obb'|'prism', ...} — validated in shape_indices`
- `CutShapeRequest.shape`: same string (cut can resolve any shape too).

- [ ] **Step 4: Run the full apply-shape suite**

Run: `.venv/bin/pytest backend/tests/test_apply_shape.py -v`
Expected: PASS (all existing + new).

- [ ] **Step 5: Commit**

```bash
git add backend/app/schemas.py backend/tests/test_apply_shape.py
git commit -m "test(backend): apply-shape resolves prism; doc prism shape type"
```

---

## Task 3: Persist prism volume on the instance schema

**Files:**
- Modify: `backend/app/schemas.py` (`Cuboid` model)

- [ ] **Step 1: Add the field**

In `Cuboid`, after `rotation`:

```python
    # Prism selection volume (source == 'prism'): {polygon:[[x,z],...],
    # y0, height}. Null for every other kind. Inert display-only data a raw
    # export rasterizes (materialize.collect_volumes), mirroring how box/beam
    # persist center/size/rotation. Spec 2026-07-20 §Persistence.
    prism: Optional[dict] = None
```

Add `'prism'` to the `source` comment enumeration:
```python
    source: str = "manual"   # 'manual' | 'auto' | 'fit' | 'preseg' | 'box' | 'beam' | 'draw' | 'prism' | 'recommendation'
```

- [ ] **Step 2: Verify the app still imports**

Run: `.venv/bin/pytest backend/tests/test_smoke.py -q`
Expected: PASS (schema change is additive/optional; nothing breaks).

- [ ] **Step 3: Commit**

```bash
git add backend/app/schemas.py
git commit -m "feat(backend): persist prism volume on the pointset instance schema"
```

---

## Task 4: Raw-export replay for prisms

**Files:**
- Modify: `backend/labeling/materialize.py` (`collect_volumes`, `replay_labels`)
- Test: `backend/tests/test_materialize.py`

- [ ] **Step 1: Write the failing test**

Read `backend/tests/test_materialize.py` to reuse how it exercises `collect_volumes` + `replay_labels` (find the OBB/box test and mirror its setup). Add:

```python
def test_collect_volumes_includes_prism():
    from labeling.materialize import collect_volumes
    instances = [{
        "source": "prism", "segId": 7, "seq": 3,
        "prism": {"polygon": [[0, 0], [2, 0], [2, 2], [0, 2]],
                  "y0": 0.0, "height": 1.0},
    }]
    vols = collect_volumes(instances, centerlines=None)
    assert len(vols) == 1
    v = vols[0]
    assert v["kind"] == "prism" and v["instance_id"] == 7 and v["seq"] == 3
    assert v["prism"]["height"] == 1.0


def test_replay_labels_rasterizes_prism_exactly():
    # A denser target than scan.ply: prism owns the exact XZ×Y region.
    # Mirror the existing OBB replay test's ReplayIndex construction, but with
    # a prism volume; assert points inside the polygon×band get instance 7.
    ...  # fill in mirroring the OBB replay test
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest backend/tests/test_materialize.py -k prism -v`
Expected: FAIL — `collect_volumes` filters prism out; `replay_labels` raises "unknown volume kind".

- [ ] **Step 3: Extend `collect_volumes`**

Change the source filter and add the prism branch:

```python
        if src not in ("box", "beam", "draw", "prism") or inst.get("segId") is None:
            continue
        ...
        if src in ("box", "beam") and inst.get("center") and inst.get("size"):
            out.append({"kind": "obb", ...})
        elif src == "prism" and inst.get("prism"):
            out.append({"kind": "prism", "instance_id": iid, "seq": seq,
                        "prism": inst["prism"]})
        elif src == "draw" and iid in paths_by_inst:
            out.append({"kind": "tube", ...})
```

- [ ] **Step 4: Extend `replay_labels`**

Add the import at the top of the file next to `from labeling.shapes import obb_indices`:
```python
from labeling.shapes import obb_indices, prism_indices
```
In the mask-building loop, add the branch:
```python
        elif v["kind"] == "prism":
            idx = prism_indices(target_pos, v["prism"])
```

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/pytest backend/tests/test_materialize.py -k prism -v`
Expected: PASS.

- [ ] **Step 6: Run the full materialize + shapes suite**

Run: `.venv/bin/pytest backend/tests/test_materialize.py backend/tests/test_shapes.py -q`
Expected: PASS.

- [ ] **Step 7: Commit**

```bash
git add backend/labeling/materialize.py backend/tests/test_materialize.py
git commit -m "feat(backend): raw-export replay rasterizes prism volumes exactly"
```

---

## Task 5: Frontend `pointsInsidePrism` (backend mirror)

**Files:**
- Create: `frontend/src/prism-geom.js`
- Test: `frontend/src/prism-geom.test.js`

This mirrors `prism_indices` so the live preview matches the applied label exactly (the way `pointsInsideOBB` in `mode-edit.jsx` mirrors `obb_indices`). Parity is locked by a fixture identical to the backend's.

- [ ] **Step 1: Write the failing test**

Create `frontend/src/prism-geom.test.js`:

```js
import { describe, it, expect } from 'vitest';
import { pointsInsidePrism, pointInPolygonXZ } from './prism-geom.js';

// Shared parity fixture — MUST match backend test_shapes.py exactly.
const SQUARE = { polygon: [[1, 1], [3, 1], [3, 3], [1, 3]], y0: 0, height: 1 };

function gridXZ(ys) {
  const out = [];
  for (const y of ys) for (let x = 0; x < 5; x++) for (let z = 0; z < 5; z++) out.push(x, y, z);
  return new Float32Array(out);
}

describe('pointsInsidePrism', () => {
  it('selects the 3x3 block inside a square footprint within the Y-band', () => {
    const pos = gridXZ([0, 5]);
    const idx = pointsInsidePrism(pos, null, SQUARE);
    const expect9 = [];
    for (const x of [1, 2, 3]) for (const z of [1, 2, 3]) expect9.push(x * 5 + z);
    expect([...idx].sort((a, b) => a - b)).toEqual(expect9);
  });

  it('supports concave (L) footprints', () => {
    const pos = gridXZ([0]);
    const L = { polygon: [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]], y0: -0.5, height: 1 };
    const idx = new Set(pointsInsidePrism(pos, null, L));
    expect(idx.has(3 * 5 + 3)).toBe(false); // notch excluded
    expect(idx.has(1 * 5 + 3)).toBe(true);
    expect(idx.has(3 * 5 + 1)).toBe(true);
  });

  it('returns nothing for < 3 vertices or zero height', () => {
    const pos = gridXZ([0]);
    expect(pointsInsidePrism(pos, null, { polygon: [[1, 1], [2, 2]], y0: 0, height: 1 })).toEqual([]);
    expect(pointsInsidePrism(pos, null, { ...SQUARE, height: 0 })).toEqual([]);
  });

  it('pointInPolygonXZ matches the ray-cast rule', () => {
    const sq = [[0, 0], [4, 0], [4, 4], [0, 4]];
    expect(pointInPolygonXZ(2, 2, sq)).toBe(true);
    expect(pointInPolygonXZ(5, 2, sq)).toBe(false);
  });
});
```

- [ ] **Step 2: Run to verify it fails**

Run (from `frontend/`): `npx vitest run src/prism-geom.test.js`
Expected: FAIL — module not found.

- [ ] **Step 3: Implement `prism-geom.js`**

Create `frontend/src/prism-geom.js`:

```js
// Frontend mirror of backend labeling/shapes.py::prism_indices — keeps the
// live in-viewport preview identical to the applied label. Parity is locked by
// prism-geom.test.js sharing test_shapes.py's fixture. XZ world plane, Y-up.

// Even-odd ray-cast: is (x,z) inside polygon [[x,z],...] (>=3 verts)?
export function pointInPolygonXZ(x, z, polygon) {
  let inside = false;
  const n = polygon.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i][0], zi = polygon[i][1];
    const xj = polygon[j][0], zj = polygon[j][1];
    const straddles = (zi > z) !== (zj > z);
    if (straddles && x < ((xj - xi) * (z - zi)) / (zj - zi) + xi) inside = !inside;
  }
  return inside;
}

// Indices into `positions` (Float32 xyz triples) inside the prism. `pool` is an
// optional index array to restrict the scan (null = all points).
export function pointsInsidePrism(positions, pool, prism) {
  const { polygon, y0, height } = prism;
  if (!polygon || polygon.length < 3 || !(height > 0)) return [];
  const yTop = y0 + height;
  const out = [];
  const N = pool ? pool.length : positions.length / 3;
  for (let k = 0; k < N; k++) {
    const i = pool ? pool[k] : k;
    const y = positions[3 * i + 1];
    if (y < y0 || y > yTop) continue;
    if (pointInPolygonXZ(positions[3 * i], positions[3 * i + 2], polygon)) out.push(i);
  }
  return out;
}
```

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run src/prism-geom.test.js`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/prism-geom.js frontend/src/prism-geom.test.js
git commit -m "feat(frontend): pointsInsidePrism mirror of backend prism_indices"
```

---

## Task 6: Register the Prism tool in the rail

**Files:**
- Modify: `frontend/src/label-tools.js`
- Test: `frontend/src/label-tools.test.js`

Prism gating = same as Box (`!!segState`; no raw source, no annotated requirement) — its geometry rides in `instances_gt.json`. Insert it **after Box** (its conceptual sibling), before Draw.

- [ ] **Step 1: Update the failing test**

In `frontend/src/label-tools.test.js`, update the rail-order assertion and add gating asserts:

```js
  it('lists the six selection tools in rail order', () => {
    expect(TOOLS.map((t) => t.id)).toEqual(['presegment', 'box', 'prism', 'draw', 'beam', 'sam']);
  });
```
And in the gating test, alongside the Box asserts:
```js
    expect(toolAvailable('prism', raw)).toBe(false);          // no session
    expect(toolAvailable('prism', sessionOnly)).toBe(true);   // session, non-annotated OK (like Box)
```

- [ ] **Step 2: Run to verify it fails**

Run: `npx vitest run src/label-tools.test.js`
Expected: FAIL on rail order + prism gating.

- [ ] **Step 3: Add the tool**

In `frontend/src/label-tools.js`, insert into `TOOLS` after the `box` entry:
```js
  { id: 'prism',      icon: '⬠', label: 'Prism' },
```
`toolAvailable` needs no change — `prism` falls through to the `return !!segState` default exactly like Box. (Confirm there is no earlier `if` that would catch it.)

- [ ] **Step 4: Run to verify it passes**

Run: `npx vitest run src/label-tools.test.js`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/label-tools.js frontend/src/label-tools.test.js
git commit -m "feat(frontend): register Prism tool in the label rail (Box gating)"
```

---

## Task 7: `prism-mode.jsx` sub-mode (overlay + input + apply)

**Files:**
- Create: `frontend/src/prism-mode.jsx`

**This is the largest task — mirror `frontend/src/beam-mode.jsx` structure closely.** Read `beam-mode.jsx` end-to-end first; `PrismMode` reuses its patterns: `viewerRef.attachOverlayGroup()` for a lifetime overlay group, `v.getCamera()` / `v.domElement()` / `v.setOrbitEnabled(false)` for interaction, a capture-phase `PrismKeys` component (mirror `BeamKeys`), and a sidebar `PrismPanel` (mirror `BeamPanel`). The apply mirrors `mode-label.jsx::applyBox` (lines 908–960): same `VoxaAPI.applyShape` call, same empty/`nProtected` guard, same `applyDelta` refresh via `setSegState`.

State shape (single object, like beam's graph):
```js
// prism = { vertices: [[x,z],...], y0: number|null, closed: bool, height: number }
const EMPTY_PRISM = { vertices: [], y0: null, closed: false, height: 2.0 };
```

- [ ] **Step 1: Scaffold `PrismMode` + `PrismPanel`**

Create `frontend/src/prism-mode.jsx`:

```jsx
import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { evtToNdc } from './viewer.jsx';       // exported helper
import { applyDelta } from './segment-state.js'; // confirm the export name/path
import { VoxaAPI } from './api.js';             // confirm the import style used elsewhere

const EMPTY_PRISM = { vertices: [], y0: null, closed: false, height: 2.0 };
const MIN_HEIGHT = 0.05;

export default function PrismMode({
  viewerRef, classes, setSegState, onExit, defaultClassId,
  onClassChange, onApplied, protectInstances,
}) {
  const [prism, setPrism] = useState(EMPTY_PRISM);
  const prismRef = useRef(prism); prismRef.current = prism;

  const reset = useCallback(() => setPrism(EMPTY_PRISM), []);

  const applyPrism = useCallback(async (classId) => {
    const p = prismRef.current;
    if (!p.closed || p.vertices.length < 3) return;
    const cls = classes.find((c) => c.class_id === classId);
    if (!cls) return;
    const shape = { type: 'prism', polygon: p.vertices, y0: p.y0, height: p.height };
    let r;
    try {
      r = await VoxaAPI.applyShape({
        shape, targetClass: cls.id, protectInstances,
      });
    } catch (err) { console.error('prism apply failed:', err); return; }
    if (!r.indices || r.nAffected === 0) {
      console.warn(r.nProtected > 0
        ? `prism apply: ${r.nProtected} point(s) skipped — inside a confirmed instance (un-confirm to re-label)`
        : 'prism apply: no points inside the prism');
      return;
    }
    const segId = Number.isFinite(r.instanceId) ? r.instanceId : -1;
    if (segId >= 0) {
      onApplied({
        instanceId: segId, classId: cls.class_id, source: 'prism',
        prism: { polygon: p.vertices.map((v) => [...v]), y0: p.y0, height: p.height },
      });
    }
    setSegState((s) => (s ? { ...applyDelta(s, {
      indices: r.indices, after_class: r.afterClass, after_instance: r.afterInstance,
    }), selection: new Set() } : s));
    reset();
  }, [classes, protectInstances, onApplied, setSegState, reset]);

  return (
    <>
      <PrismKeys prism={prism} setPrism={setPrism} classes={classes}
        defaultClassId={defaultClassId} onApply={applyPrism} onExit={onExit} />
      <PrismOverlay viewerRef={viewerRef} prism={prism} setPrism={setPrism} />
      <PrismPanel prism={prism} setPrism={setPrism} onClear={reset}
        onApply={() => applyPrism(defaultClassId)} />
    </>
  );
}
```

- [ ] **Step 2: Implement `PrismPanel` (sidebar height field + Draw/Clear/Apply)**

```jsx
function PrismPanel({ prism, setPrism, onClear, onApply }) {
  const drawing = prism.vertices.length > 0 && !prism.closed;
  return (
    <div className="prism-panel">
      {!prism.vertices.length && (
        <p className="tool-opt-hint">Click to place footprint vertices on the plane at the
          first point's height. Enter or click the first vertex to close.</p>
      )}
      {drawing && (
        <p className="tool-opt-hint">{prism.vertices.length} vertex(es) — Enter/first-vertex to
          close, Backspace to undo, Esc to cancel.</p>
      )}
      {prism.closed && (
        <>
          <label className="tool-opt-row">Height (m)
            <input type="number" min={MIN_HEIGHT} step="0.1" value={prism.height}
              onChange={(e) => setPrism((s) => ({
                ...s, height: Math.max(MIN_HEIGHT, Number(e.target.value) || MIN_HEIGHT) }))} />
          </label>
          <p className="tool-opt-hint">Scroll over the viewport to adjust height.</p>
          <div className="tool-opt-toggle">
            <button onClick={onClear}>Clear</button>
            <button className="active" onClick={onApply}>Apply (Ctrl+Enter)</button>
          </div>
        </>
      )}
    </div>
  );
}
```

- [ ] **Step 3: Implement `PrismKeys` (capture-phase key handling)**

Mirror `BeamKeys` (capture phase, so it beats `mode-label.jsx`'s handler which early-returns on `subModeOwnsInput`). Handle: Enter/Ctrl+Enter → close (if drawing) else apply with `defaultClassId`; class hotkeys → apply with that class (only when `closed`); Backspace → drop last vertex; Escape → clear or `onExit`. Map class hotkeys via `classes` (`c.hotkey`), calling `onApply(c.class_id)`.

- [ ] **Step 4: Implement `PrismOverlay` (rendering + pointer + scroll)**

Mirror `BeamOverlay`. Three `useEffect`s:

1. **Lifetime group:** `layerRef.current = v.attachOverlayGroup(); return () => layerRef.current?.remove()`.
2. **Rebuild children on `prism` change:** dispose group children, then draw:
   - in-progress / closed polygon as a `THREE.Line` (or `LineLoop` when closed) through `vertices` at `y0`;
   - small sphere per vertex;
   - when `closed`: the **top** polygon at `y0+height` and **vertical edges** (a `LineSegments`) — the extrusion preview.
   Guard against `y0 == null`.
3. **Pointer + scroll:** on the viewer DOM element (`v.domElement()`):
   - **click (button 0):** ray-cast to the horizontal plane `y = y0` (for the first click, intersect the plane at the first hit's Y — get the world point via a `THREE.Plane` at the clicked point, OR use `viewerRef.current.firstHitUnderCursor(evt)` to seed `y0` from the nearest cloud point, then a horizontal plane at that Y for subsequent vertices). Append `[x, z]`. If the click is near the first vertex (screen-space threshold) and `vertices.length >= 3`, set `closed`. Call `v.setOrbitEnabled(false)` on pointerdown over the overlay so drawing doesn't orbit; re-enable on tool exit.
   - **wheel (only when `closed`):** `setPrism((s) => ({ ...s, height: Math.max(MIN_HEIGHT, s.height + sign * step) }))`; `evt.preventDefault()` so it doesn't zoom.

   For seeding `y0`: on the **first** click, use `viewerRef.current.firstHitUnderCursor(evt)` (returns `{ position }` — verify its return shape) to get the clicked point's world Y → `y0`. All vertices (including the first) are the ray∩(horizontal plane at `y0`). This gives a stable coplanar footprint.

- [ ] **Step 5: Manual smoke (no automated test for the Three.js overlay)**

Overlay/interaction is verified in Task 9's browser check. `PrismMode`'s pure logic (`applyPrism` guards, geometry) is covered indirectly by `prism-geom` + the backend. Commit the scaffold.

```bash
git add frontend/src/prism-mode.jsx
git commit -m "feat(frontend): PrismMode sub-mode (overlay, keys, apply pipeline)"
```

---

## Task 8: Wire `PrismOptions` into the tool-options panel

**Files:**
- Modify: `frontend/src/tool-options.jsx`

- [ ] **Step 1: Import + PrismOptions**

At the top, import `PrismMode`:
```js
import PrismMode from './prism-mode.jsx';
```
Add the panel (mirror `BeamOptions`, keyed on `activeSessionId` so a session switch remounts):
```jsx
function PrismOptions({
  viewerRef, classes, setSegState, onExit,
  activeClass, setActiveClass, onToolApplied, autoConfirm, setAutoConfirm,
  activeSessionId, protectInstances,
}) {
  return (
    <div className="tool-options tool-options-prism">
      <PrismMode
        key={activeSessionId}
        viewerRef={viewerRef}
        classes={classes}
        setSegState={setSegState}
        onExit={onExit}
        defaultClassId={classes.find((c) => c.id === activeClass)?.class_id ?? classes[0]?.class_id ?? 0}
        onClassChange={(cid) => {
          const cls = classes.find((c) => c.class_id === cid);
          if (cls) setActiveClass(cls.id);
        }}
        onApplied={onToolApplied}
        protectInstances={protectInstances}
      />
      <AutoConfirmToggle tool="prism" autoConfirm={autoConfirm} setAutoConfirm={setAutoConfirm} />
    </div>
  );
}
```

- [ ] **Step 2: Add to the dispatch**

In `ToolOptions`, add before the `box` line:
```js
  if (activeTool === 'prism') return <PrismOptions {...props} />;
```

- [ ] **Step 3: Verify build**

Run (from `frontend/`): `npx vitest run` (full suite still passes) and `npm run build` (from repo root) to confirm no import/JSX errors.
Expected: build succeeds, tests pass.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/tool-options.jsx
git commit -m "feat(frontend): mount PrismMode via PrismOptions tool panel"
```

---

## Task 9: mode-label wiring + browser verification

**Files:**
- Modify: `frontend/src/mode-label.jsx`

- [ ] **Step 1: Include prism in `subModeOwnsInput`**

Around line 53–58:
```js
  const prismMode = activeTool === 'prism';
  ...
  const subModeOwnsInput = drawMode || beamMode || prismMode;
```
This makes `mode-label.jsx`'s key/click handlers early-return so `PrismKeys`/`PrismOverlay` own input (same as Beam/Draw).

- [ ] **Step 2: Extend `onToolApplied` with a prism volume**

In the `onToolApplied` callback (around line 971), add a `prism` param and fold it into `volume`:
```js
  const onToolApplied = useCallbackLabel(({
    instanceId, classId, mergedFrom = [], source = 'draw', obb = null, prism = null,
  }) => {
    ...
    const volume = obb
      ? { center: [...obb.center], size: [...obb.size], rotation: [...obb.rotation] }
      : prism
        ? { prism: { polygon: prism.polygon.map((v) => [...v]), y0: prism.y0, height: prism.height } }
        : {};
    ...
```
(The rest of the helper already surfaces/refreshes the pointset row; `...volume` now carries the prism.)

- [ ] **Step 3: Confirm no gating conflict**

Verify `mode-label.jsx`'s primary key handler (the one gated by `if (fastMode || subModeOwnsInput) return;` around line 1099) now short-circuits for prism, and that there is no `activeTool === 'prism'` branch needed in `applyBox`/class-picker paths (Prism applies entirely inside `PrismMode`).

- [ ] **Step 4: Browser verification** (REQUIRED — use the `browser-verification` skill)

Start a throwaway session (Label apply auto-saves to disk — use a scratch session on an annotated scan; restart any stale `:8765` backend first). Then:
1. Select the **Prism** tool (⬠) in the rail. Confirm the options panel shows the draw hint.
2. Click 4–6 points around an object's base footprint; confirm vertices + rubber-band render on the plane at the first click's height.
3. Press Enter (or click the first vertex) to close; confirm the extrusion preview (top polygon + vertical edges) appears.
4. Scroll over the viewport; confirm the height grows/shrinks live and the metres field updates.
5. Press a class hotkey (or Apply); confirm the prism vanishes, a new pointset instance appears in the Instances panel with the right class, and the enclosed points recolor.
6. Confirm the instance, re-open the Prism tool, draw an overlapping prism over the confirmed region, apply; confirm the console logs "skipped — inside a confirmed instance" and the confirmed points are **not** overwritten (Confirmed = locked).
7. Check **zero console errors** and that `POST /api/segment/apply-shape` returned 200.

Screenshot the closed-prism preview and the resulting labeled instance.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/mode-label.jsx
git commit -m "feat(frontend): wire Prism sub-mode input + persist prism volume on apply"
```

---

## Task 10: Docs

**Files:**
- Modify: `CLAUDE.md` (the Label-mode tool rail bullet list)

- [ ] **Step 1: Document the tool**

In `CLAUDE.md`, add a **Prism** bullet to the 5-tool rail list (now 6), mirroring the Box/Beam bullets: draw a footprint polygon on a horizontal plane at the first click's height, scroll to set upward extrusion, applies through the shared `apply-shape` endpoint as a new `prism` shape (XZ polygon × Y-band), persists `{polygon,y0,height}` on the pointset for exact raw export, no gizmo / Clear-and-redraw to edit. Note the tool count change ("5-tool rail" → "6-tool rail") wherever it appears.

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document the Prism Box labeling tool in CLAUDE.md"
```

---

## Final verification

- [ ] Backend: `.venv/bin/pytest backend/tests/test_shapes.py backend/tests/test_apply_shape.py backend/tests/test_materialize.py -q` → all pass.
- [ ] Frontend: `npx vitest run` (from `frontend/`) → all pass.
- [ ] Build: `npm run build` → succeeds.
- [ ] Browser check from Task 9 completed with screenshots + zero console errors.
- [ ] Then use `superpowers:finishing-a-development-branch` to decide merge/PR.
