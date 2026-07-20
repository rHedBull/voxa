# Fit Box to Selection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Right-click a preseg / SAM / instance selection → "Fit box to selection…" → the backend fits a gravity-aligned (yaw-only) oriented box tightly around the full-res selected points → it is handed to the Box tool as a staged, editable `selBox` that the user then labels through the normal Box pipeline.

**Architecture:** A pure geometry helper (`fit_gravity_obb`) computes a yaw-only OBB from a point array. A new session-pinned endpoint `POST /api/segment/fit-box` resolves the selection to full-res indices exactly like `_cut_shape_core`, unions them, and returns the OBB. The frontend adds a `fitEligibility` gate, a `fitBox` API call, a `fitBoxToSelection` handler in `mode-label.jsx` that sets `selBox` and switches to the Box tool, and a "Fit box to selection…" menu item on the three list surfaces. No new apply path — once staged, the box is an ordinary Box selection.

**Tech Stack:** Python / FastAPI / numpy (backend), React 18 / Three.js / Vitest (frontend). Full spec: `docs/superpowers/specs/2026-07-20-fit-box-to-selection-design.md`.

**Working directory:** the `feat/fit-box-to-selection` worktree. Backend tests: `npm run test:backend`. Frontend tests: `npm run test:frontend` (or `npx vitest run src/<file>` from `./frontend`).

---

## File Structure

**New**
- `backend/labeling/fit_box.py` — `fit_gravity_obb(points) -> (center, size, rotation)`. Pure, no FastAPI.
- `backend/tests/test_fit_box.py` — geometry + endpoint tests.
- `frontend/src/fit-eligibility.js` — `fitEligibility(params)` pure gate.
- `frontend/src/fit-eligibility.test.js` — gate tests.

**Modified**
- `backend/app/schemas.py` — add `FitBoxRequest` next to `CutShapeRequest`.
- `backend/routes/segment.py` — add `_fit_box_core` + `POST /api/segment/fit-box`.
- `frontend/src/api.js` — add `fitBox(sources)`.
- `frontend/src/mode-label.jsx` — add `fitBoxToSelection`; wire it into the three list menus.
- `frontend/src/segment-tools.jsx` — add "Fit box to selection…" menu item (preseg + sam blocks); new `onFitBox` prop.
- `frontend/src/sam-segment-list.jsx` — add "Fit box to selection…" menu item; new `onFitBox` prop.
- `frontend/src/api.test.js` — add a `fitBox` test.
- `CLAUDE.md` — one bullet under the Box tool.

---

## Task 1: Geometry — `fit_gravity_obb`

**Files:**
- Create: `backend/labeling/fit_box.py`
- Test: `backend/tests/test_fit_box.py`

> **As-built note:** the footprint step below was written as 2D PCA, but PCA is
> only a statistical orientation estimate and missed the axis-aligned-recovery
> tolerance in `test_axis_aligned_box_recovers_bounds`. It was replaced during
> implementation with the **exact minimum-area rectangle via rotating calipers
> over the convex hull** (`scipy.spatial.ConvexHull`, already a dependency) —
> the projection/center/padding/degenerate-fallback logic below is unchanged.
> The spec's Backend section is the authoritative description. The PCA snippet
> is retained here as the original plan of record.

The math (yaw-only OBB, Euler-XYZ `Rx·Ry·Rz` convention, so `rotation = [0, ry, 0]`):
- Vertical: `sy = y.max - y.min`, `cy = midpoint`.
- Footprint: 2D PCA on the X/Z columns → dominant direction `(dx, dz)`. Yaw `θ = atan2(-dz, dx)` so the box's local x-axis (world column `[cosθ, 0, -sinθ]` of `euler_xyz_matrix(0,θ,0)`) aligns with the principal direction. Project points onto `u=(cosθ,-sinθ)` and `v=(sinθ,cosθ)` in XZ, take min/max → extents `sx, sz` and center `(cx, cz)`.
- Padding: `+0.005` per size axis (matches `/api/auto-fit`), so the box provably contains every source point.
- Degenerate: `<3` points OR near-zero secondary eigenvalue (collinear XZ) → `θ = 0` (axis-aligned) and clamp each horizontal size to `MIN_SIZE = 0.01`. `0` points → `ValueError`.

- [ ] **Step 1: Write failing tests**

```python
# backend/tests/test_fit_box.py
"""Tests for fit_gravity_obb + the /api/segment/fit-box endpoint."""
import numpy as np
import pytest

from labeling.fit_box import fit_gravity_obb
from labeling.shapes import obb_indices


def _obb_dict(center, size, rotation):
    return {"center": list(center), "size": list(size), "rotation": list(rotation)}


def test_axis_aligned_box_recovers_bounds():
    # A filled axis-aligned box; fit should recover its center + extents (+pad).
    # NOTE: 2D PCA picks the higher-variance horizontal axis (z, extent 6) as
    # dominant, so ry ≈ ±π/2 and the footprint axes (size[0]/size[2]) may SWAP.
    # The eigenvector sign from eigh is also arbitrary (ry can come back as π).
    # So assert on the SORTED extents + containment, never positional size[i].
    rng = np.random.default_rng(0)
    pts = rng.uniform([-2, -1, -3], [2, 1, 3], size=(2000, 3)).astype(np.float32)
    center, size, rotation = fit_gravity_obb(pts)
    assert rotation[0] == 0.0 and rotation[2] == 0.0
    np.testing.assert_allclose(center, [0, 0, 0], atol=0.05)
    # Sorted extents ~ (2, 4, 6) + small pad, order-independent.
    np.testing.assert_allclose(sorted(size), [2, 4, 6], atol=0.05)
    # And the box actually contains every point (the real invariant).
    assert obb_indices(pts, _obb_dict(center, size, rotation)).size == pts.shape[0]


def test_containment_parity_after_yaw():
    # Rotate a thin rectangle by a known yaw; the fitted OBB fed back through
    # obb_indices must contain 100% of the points. This locks the yaw sign /
    # Euler convention — a reversed yaw drops points.
    rng = np.random.default_rng(1)
    theta = 0.6
    local = rng.uniform([-3, -0.5, -0.4], [3, 0.5, 0.4], size=(3000, 3))
    c, s = np.cos(theta), np.sin(theta)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    pts = (local @ Ry.T).astype(np.float32)  # rotate about world Y
    center, size, rotation = fit_gravity_obb(pts)
    inside = obb_indices(pts, _obb_dict(center, size, rotation))
    assert inside.size == pts.shape[0]  # every point contained


def test_yaw_gives_tight_footprint_not_aabb():
    # For a diagonal thin rectangle, the tight footprint is much smaller than
    # the axis-aligned bounding box of the same points.
    rng = np.random.default_rng(2)
    theta = np.pi / 4
    local = rng.uniform([-3, 0, -0.3], [3, 1, 0.3], size=(3000, 3))
    c, s = np.cos(theta), np.sin(theta)
    Ry = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    pts = (local @ Ry.T).astype(np.float32)
    _, size, _ = fit_gravity_obb(pts)
    aabb_x = pts[:, 0].max() - pts[:, 0].min()
    # Tight footprint x (~6) is well under the diagonal AABB (~4.6? -> actually
    # >4). Assert the SHORT side is tight (~0.6), which an AABB could never be.
    short = min(size[0], size[2])
    assert short < 1.0
    assert min(aabb_x, pts[:, 2].max() - pts[:, 2].min()) > 3.0


def test_collinear_xz_falls_back_to_axis_aligned():
    # All points share one XZ line (vary only along x and y). No NaN, ry=0,
    # positive volume.
    pts = np.zeros((100, 3), dtype=np.float32)
    pts[:, 0] = np.linspace(-1, 1, 100)
    pts[:, 1] = np.linspace(0, 2, 100)
    center, size, rotation = fit_gravity_obb(pts)
    assert rotation[1] == 0.0
    assert all(np.isfinite(v) for v in (*center, *size))
    assert size[0] > 0 and size[1] > 0 and size[2] > 0


def test_fewer_than_three_points():
    pts = np.array([[0, 0, 0], [1, 2, 1]], dtype=np.float32)
    center, size, rotation = fit_gravity_obb(pts)
    assert rotation == [0.0, 0.0, 0.0]
    assert all(v > 0 for v in size)


def test_zero_points_raises():
    with pytest.raises(ValueError):
        fit_gravity_obb(np.empty((0, 3), dtype=np.float32))
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `npm run test:backend -- backend/tests/test_fit_box.py`
Expected: FAIL — `ModuleNotFoundError: No module named 'labeling.fit_box'`.

- [ ] **Step 3: Implement `fit_gravity_obb`**

```python
# backend/labeling/fit_box.py
"""Fit a gravity-aligned (yaw-only) oriented box to a set of points.

The box's up axis is locked to world-Y; the footprint is the tightest rotated
rectangle in the X/Z plane (2D PCA). Rotation is returned as Euler-XYZ
[0, ry, 0], matching the Rx·Ry·Rz convention that shapes.py::obb_indices /
scenes/reproject.py::euler_xyz_matrix compose. See
docs/superpowers/specs/2026-07-20-fit-box-to-selection-design.md.
"""
import numpy as np

PAD = 0.005       # per-axis padding so the box provably contains all points
MIN_SIZE = 0.01   # floor for a degenerate (collinear / near-flat) footprint


def fit_gravity_obb(points):
    """points: (N,3) array. Returns (center, size, rotation) as plain float
    lists: center [x,y,z], size [sx,sy,sz] (full side lengths), rotation
    [0, ry, 0]. Raises ValueError on an empty input."""
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = pts.shape[0]
    if n == 0:
        raise ValueError("fit_gravity_obb: empty point set")

    y0, y1 = pts[:, 1].min(), pts[:, 1].max()
    cy = (y0 + y1) / 2.0
    sy = (y1 - y0) + PAD

    xz = pts[:, [0, 2]]
    theta = 0.0
    if n >= 3:
        centered = xz - xz.mean(axis=0)
        cov = centered.T @ centered / n
        evals, evecs = np.linalg.eigh(cov)  # ascending
        # Collinear / near-flat footprint → keep axis-aligned (theta=0).
        if evals[0] > 1e-9 and evals[1] > 1e-12:
            dx, dz = evecs[:, 1]  # dominant eigenvector (x,z)
            theta = float(np.arctan2(-dz, dx))

    c, s = np.cos(theta), np.sin(theta)
    # Project onto box-local horizontal axes u (local x) and v (local z).
    pu = xz[:, 0] * c - xz[:, 1] * s
    pv = xz[:, 0] * s + xz[:, 1] * c
    umin, umax = pu.min(), pu.max()
    vmin, vmax = pv.min(), pv.max()
    cu, cv = (umin + umax) / 2.0, (vmin + vmax) / 2.0
    sx = max(umax - umin, MIN_SIZE) + PAD
    sz = max(vmax - vmin, MIN_SIZE) + PAD
    # Center back to world XZ (u,v is an orthonormal rotation).
    cx = cu * c + cv * s
    cz = -cu * s + cv * c

    return ([cx, cy, cz], [sx, sy, sz], [0.0, theta, 0.0])
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `npm run test:backend -- backend/tests/test_fit_box.py`
Expected: PASS (6 passed). If `test_containment_parity_after_yaw` fails, the yaw sign is wrong — re-derive `theta` against `obb_indices`, do NOT loosen the assertion.

- [ ] **Step 5: Commit**

```bash
git add backend/labeling/fit_box.py backend/tests/test_fit_box.py
git commit -m "feat: fit_gravity_obb — yaw-only OBB fit to a point set"
```

---

## Task 2: Endpoint — `POST /api/segment/fit-box`

**Files:**
- Modify: `backend/app/schemas.py` (add `FitBoxRequest` after `CutShapeRequest`, ~line 187)
- Modify: `backend/routes/segment.py` (add `_fit_box_core` + route after `cut_shape`, ~line 238)
- Test: `backend/tests/test_fit_box.py` (append endpoint tests)

`_fit_box_core` reuses the exact source-membership resolution from `_cut_shape_core`
(`preseg`/`instance` → `instance_ids == seg_id`, `sam` → `sam_ids == seg_id`),
unions all matched indices, then fits.

- [ ] **Step 1: Add the request schema**

In `backend/app/schemas.py`, after `class CutShapeRequest` (reuse `CutShapeSource`):

```python
class FitBoxRequest(BaseModel):
    sources: list[CutShapeSource]
```

- [ ] **Step 2: Write failing endpoint tests**

Append to `backend/tests/test_fit_box.py`. Mirror the pattern in
`backend/tests/test_cut_shape.py`, which uses the shared fixture
**`client_with_loaded_annotated_scene`** (defined in `backend/tests/conftest.py`)
— it yields **just `client`** (not a tuple) with an active `SegmentSession`.
`build_annotated_root` fixes the instance array to `[-1,0,0,1,1,2,-1,3]`, so
preseg **`seg_id` 0 and 1 are always present** (id 0 → 2 points, id 1 → 2 points).
The `client` fixture (also in conftest, no loaded scene) drives the 409 test.

```python
# --- endpoint tests (append) ---

def test_fit_box_returns_containing_obb(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/fit-box",
                    json={"sources": [{"kind": "preseg", "seg_id": 0}]})
    assert r.status_code == 200
    obb = r.json()
    assert obb["rotation"][0] == 0.0 and obb["rotation"][2] == 0.0
    assert len(obb["center"]) == 3 and len(obb["size"]) == 3
    assert all(s > 0 for s in obb["size"])


def test_fit_box_union_contains_both_presegs(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/fit-box",
                    json={"sources": [{"kind": "preseg", "seg_id": 0},
                                      {"kind": "preseg", "seg_id": 1}]})
    assert r.status_code == 200
    # Fetch the active session's points, feed the returned OBB back through
    # obb_indices, and assert every point of BOTH presegs is contained.
    from app.core import _state
    seg = _state["seg"]
    inside = set(obb_indices(np.asarray(seg.positions), r.json()).tolist())
    for pid in (0, 1):
        members = np.nonzero(seg.instance_ids == pid)[0].tolist()
        assert members and set(members).issubset(inside)


def test_fit_box_empty_union_400(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/fit-box",
                    json={"sources": [{"kind": "preseg", "seg_id": 999999}]})
    assert r.status_code == 400


def test_fit_box_no_session_409(client):  # `client` = conftest fixture, no scene
    r = client.post("/api/segment/fit-box",
                    json={"sources": [{"kind": "preseg", "seg_id": 0}]})
    assert r.status_code == 409
```

Verify the active-session accessor before finalizing: the union test reads
`_state["seg"]` / `seg.positions` / `seg.instance_ids` — confirm those exact
names against `app/core.py` (grep `_state[` and the `SegmentSession` attrs) and
adjust if they differ.

- [ ] **Step 3: Run tests, verify they fail**

Run: `npm run test:backend -- backend/tests/test_fit_box.py`
Expected: FAIL — 404 (route missing) / import error for `FitBoxRequest`.

- [ ] **Step 4: Implement core + route**

In `backend/routes/segment.py`, after `cut_shape` (~line 238):

```python
def _fit_box_core(seg, sources: list) -> dict:
    """Resolve `sources` to a unioned full-res index set (same membership rules
    as _cut_shape_core) and fit a gravity-aligned OBB around those points.
    Raises ValueError (→ 400) if the union is empty."""
    from labeling.fit_box import fit_gravity_obb
    masks = []
    for src in sources:
        if src.kind in ("preseg", "instance"):
            masks.append(seg.instance_ids == src.seg_id)
        elif src.kind == "sam":
            masks.append(seg.sam_ids == src.seg_id)
        else:
            raise ValueError(f"unknown source kind: {src.kind!r}")
    if not masks:
        raise ValueError("no sources given")
    union = np.zeros(seg.instance_ids.shape[0], dtype=bool)
    for m in masks:
        union |= m
    idx = np.nonzero(union)[0]
    if idx.size == 0:
        raise ValueError("selection resolved to zero points")
    center, size, rotation = fit_gravity_obb(np.asarray(seg.positions)[idx])
    return {"center": center, "size": size, "rotation": rotation}


@router.post("/api/segment/fit-box")
def fit_box(req: FitBoxRequest):
    """Fit a gravity-aligned selection box around the currently-selected
    points (presegments / SAM candidates / an instance). Returns an OBB
    {center,size,rotation:[0,ry,0]} the frontend stages as the Box tool's
    selBox — it labels nothing. See
    docs/superpowers/specs/2026-07-20-fit-box-to-selection-design.md."""
    seg = _require_seg()
    try:
        return _fit_box_core(seg, req.sources)
    except ValueError as e:
        raise HTTPException(400, str(e))
```

Note: `_require_seg()` raises 409 when no session is active (verify against its
definition in `app/core.py`; the no-session test asserts 409).

- [ ] **Step 5: Run tests, verify they pass**

Run: `npm run test:backend -- backend/tests/test_fit_box.py`
Expected: PASS.

- [ ] **Step 6: Run the full backend suite (no regressions)**

Run: `npm run test:backend`
Expected: all pass (418 prior + the new fit-box tests).

- [ ] **Step 7: Commit**

```bash
git add backend/app/schemas.py backend/routes/segment.py backend/tests/test_fit_box.py
git commit -m "feat: POST /api/segment/fit-box — fit an OBB to the current selection"
```

---

## Task 3: Frontend gate — `fitEligibility`

**Files:**
- Create: `frontend/src/fit-eligibility.js`
- Test: `frontend/src/fit-eligibility.test.js`

Mirrors `cut-eligibility.js`, with ONE divergence: `instance` is eligible when
`isSelected` regardless of `confirmed` (fitting is read-only on the source).

- [ ] **Step 1: Write failing tests**

```js
// frontend/src/fit-eligibility.test.js
import { describe, it, expect } from 'vitest';
import { fitEligibility } from './fit-eligibility.js';

describe('fitEligibility', () => {
  it('preseg/sam eligible iff selection non-empty', () => {
    expect(fitEligibility({ list: 'preseg', selectionSize: 2 }).eligible).toBe(true);
    expect(fitEligibility({ list: 'preseg', selectionSize: 0 }).eligible).toBe(false);
    expect(fitEligibility({ list: 'sam', selectionSize: 1 }).eligible).toBe(true);
    expect(fitEligibility({ list: 'sam', selectionSize: 0 }).eligible).toBe(false);
  });
  it('instance eligible when selected, even if confirmed (diverges from cut)', () => {
    expect(fitEligibility({ list: 'instance', isSelected: true, confirmed: false }).eligible).toBe(true);
    expect(fitEligibility({ list: 'instance', isSelected: true, confirmed: true }).eligible).toBe(true);
    expect(fitEligibility({ list: 'instance', isSelected: false }).eligible).toBe(false);
  });
  it('throws on unknown list', () => {
    expect(() => fitEligibility({ list: 'nope' })).toThrow();
  });
});
```

- [ ] **Step 2: Run tests, verify they fail**

Run (from `./frontend`): `npx vitest run src/fit-eligibility.test.js`
Expected: FAIL — cannot resolve `./fit-eligibility.js`.

- [ ] **Step 3: Implement**

```js
// frontend/src/fit-eligibility.js
// Pure rule for the right-click "Fit box to selection…" menu item. Mirrors
// cut-eligibility.js, EXCEPT a confirmed instance is still fit-eligible: fitting
// only READS the source points to size a new, independent Box volume — it never
// relabels the confirmed instance (see the confirmed-instance note in
// docs/superpowers/specs/2026-07-20-fit-box-to-selection-design.md).
export function fitEligibility(params) {
  const { list } = params;
  if (list === 'preseg' || list === 'sam') {
    return params.selectionSize > 0
      ? { eligible: true } : { eligible: false, reason: 'empty' };
  }
  if (list === 'instance') {
    return params.isSelected
      ? { eligible: true } : { eligible: false, reason: 'not-selected' };
  }
  throw new Error(`fitEligibility: unknown list "${list}"`);
}
```

- [ ] **Step 4: Run tests, verify they pass**

Run (from `./frontend`): `npx vitest run src/fit-eligibility.test.js`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/fit-eligibility.js frontend/src/fit-eligibility.test.js
git commit -m "feat: fitEligibility gate for the Fit-box menu item"
```

---

## Task 4: API client — `VoxaAPI.fitBox`

**Files:**
- Modify: `frontend/src/api.js` (add `fitBox` next to `cutShape`, ~line 265)
- Test: `frontend/src/api.test.js` (add one test near the `cutShape` test)

- [ ] **Step 1: Write failing test**

Follow the existing `api.test.js` fetch-mock style (find the `cutShape` test and
mirror it). The test asserts `fitBox` POSTs to `/api/segment/fit-box` with body
`{ sources: [{ kind, seg_id }] }` (segId → seg_id mapping, same as `cutShape`)
and returns the parsed OBB.

```js
it('fitBox posts sources and returns the OBB', async () => {
  const obb = { center: [0, 0, 0], size: [1, 2, 3], rotation: [0, 0.5, 0] };
  global.fetch = vi.fn(async () => ({ ok: true, json: async () => obb }));
  const out = await VoxaAPI.fitBox([{ kind: 'preseg', segId: 7 }]);
  const [url, opts] = global.fetch.mock.calls[0];
  expect(url).toBe('/api/segment/fit-box');
  expect(JSON.parse(opts.body)).toEqual({ sources: [{ kind: 'preseg', seg_id: 7 }] });
  expect(out).toEqual(obb);
});
```

- [ ] **Step 2: Run test, verify it fails**

Run (from `./frontend`): `npx vitest run src/api.test.js`
Expected: FAIL — `VoxaAPI.fitBox is not a function`.

- [ ] **Step 3: Implement**

In `frontend/src/api.js`, add inside the `VoxaAPI` object (mirror `cutShape`'s
segId mapping + loud throw):

```js
  async fitBox(sources) {
    const r = await fetch('/api/segment/fit-box', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ sources: sources.map((s) => ({ kind: s.kind, seg_id: s.segId })) }),
    });
    if (!r.ok) throw new Error(`fitBox failed: ${r.status} ${await r.text()}`);
    return r.json();  // { center, size, rotation }
  },
```

- [ ] **Step 4: Run test, verify it passes**

Run (from `./frontend`): `npx vitest run src/api.test.js`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api.js frontend/src/api.test.js
git commit -m "feat: VoxaAPI.fitBox client for /api/segment/fit-box"
```

---

## Task 5: Wire the handler + menu items

**Files:**
- Modify: `frontend/src/mode-label.jsx` — add `fitBoxToSelection`; pass `onFitBox` to the lists; add the instance-panel menu item.
- Modify: `frontend/src/segment-tools.jsx` — add `onFitBox` prop + "Fit box to selection…" items (preseg block ~line 177, sam block ~line 219).
- Modify: `frontend/src/sam-segment-list.jsx` — add `onFitBox` prop + item (~line 79).

`fitBoxToSelection` captures the sources array, calls `VoxaAPI.fitBox`, then
`setSelBox(obb)` + `setActiveTool('box')`. The sources array is already in the
`[{ kind, segId }]` shape the list menus build for `onEditSelection`.

- [ ] **Step 1: Add the handler in `mode-label.jsx`**

Near `openCutModal` (~line 553):

```js
// Fit a gravity-aligned Box selection volume around the current selection's
// points, then hand it to the Box tool (staged, unconfirmed). Capture happens
// here — BEFORE setActiveTool('box'), which clears the preseg/SAM selection.
const fitBoxToSelection = useCallbackLabel(async (sources) => {
  try {
    const obb = await VoxaAPI.fitBox(sources);
    // MUST build the FULL selBox shape (see toggleBoxSelect, ~line 315). The
    // Viewer gates visibility + gizmo on id === LABEL_SEL_BOX_ID:
    // visibleInstanceIds / selectedId (~line 1296) both key off it. A box
    // missing that id renders invisible and can't be transformed by G/R/Y.
    setSelBox({
      id: LABEL_SEL_BOX_ID,
      label: 'box-select',
      cls: 0,
      color: LABEL_SEL_BOX_COLOR,
      center: obb.center, size: obb.size, rotation: obb.rotation,
    });
    setActiveTool('box');
  } catch (e) {
    console.error('fit-box failed', e);
    // Surface loudly — reuse whatever banner/toast mode-label already uses for
    // API errors (grep for existing setError/console patterns in this file and
    // match it; do NOT swallow).
  }
}, [setSelBox, setActiveTool]);
```

`LABEL_SEL_BOX_ID` / `LABEL_SEL_BOX_COLOR` are module constants at the top of
`mode-label.jsx` (~line 33-34) — already in scope, no import needed. The
`{ center, size, rotation }` fields come straight from the endpoint (rotation is
`[0, ry, 0]`).

- [ ] **Step 2: Pass `onFitBox` to the two list components**

Where `PresegmentList`/`segment-tools` and `SamSegmentList` are rendered (near
the existing `onEditSelection={openCutModal}` at ~line 1287), add
`onFitBox={fitBoxToSelection}`.

- [ ] **Step 3: Add the instance-panel menu item**

At the instance context-menu block (~line 1582, next to the existing
`cutEligibility({ list: 'instance', ... })` / `openCutModal([{ kind:'instance', segId: target.segId }], ...)`), add a sibling item:

```js
{
  label: 'Fit box to selection…',
  disabled: !fitEligibility({ list: 'instance', isSelected: true, confirmed: target.confirmed }).eligible,
  // ContextMenu items use onSelect, NOT onClick (see the sibling item at ~1597).
  // Mirror its guard so a row with no valid segId is a no-op, not a crash.
  onSelect: () => {
    if (!target || !Number.isFinite(target.segId)) return;
    fitBoxToSelection([{ kind: 'instance', segId: target.segId }]);
  },
},
```

Add `import { fitEligibility } from './fit-eligibility.js';` at the top of
`mode-label.jsx`.

- [ ] **Step 4: Add the menu items in the list components**

In `segment-tools.jsx`: add `onFitBox = null` to the props (~line 68) and, in
BOTH the preseg block (~line 177) and the sam block (~line 219), add a "Fit box
to selection…" item right after the "Edit selection…" item, gated by
`fitEligibility` (`{ list: 'preseg'|'sam', selectionSize }`) and calling
`onFitBox(Array.from(<selection>).map((segId) => ({ kind, segId })))` — mirror
the exact `onEditSelection` line above it. Import `fitEligibility`.

In `sam-segment-list.jsx`: same, add `onFitBox = null` prop (~line 16) and the
item after "Edit selection…" (~line 79). Import `fitEligibility`.

- [ ] **Step 5: Update the context-menu jsdom test**

In `frontend/src/segment-tools.jsdom.test.jsx` (and/or `context-menu.test.jsx`),
add a case: with a non-empty selection and an `onFitBox` spy, right-clicking a
row shows an enabled "Fit box to selection…" item, and clicking it calls
`onFitBox` with the `[{ kind, segId }]` array. Mirror the existing "Edit
selection…" test in that file.

- [ ] **Step 6: Run frontend tests**

Run (from `./frontend`): `npx vitest run`
Expected: all pass (201 prior + fit-eligibility + api + the new jsdom case).

- [ ] **Step 7: Commit**

```bash
git add frontend/src/mode-label.jsx frontend/src/segment-tools.jsx \
        frontend/src/sam-segment-list.jsx frontend/src/segment-tools.jsdom.test.jsx
git commit -m "feat: Fit-box menu item on preseg/SAM/instance lists → staged Box selection"
```

---

## Task 6: Browser verification + docs

**Files:**
- Modify: `CLAUDE.md` (one bullet under the Box tool description)

- [ ] **Step 1: Browser-verify the end-to-end flow**

REQUIRED SUB-SKILL: Use the `browser-verification` skill. Per the "Browser-verify
mutates session" memory: start a **fresh throwaway session** on an annotated scan
(restart any stale `:8765` backend first), because Label apply auto-saves to disk.

Steps to verify (screenshot each, confirm zero console errors + successful
network calls):
1. Presegment tool → select ≥1 segment → right-click → "Fit box to selection…".
2. Confirm the tool switches to **Box** and a box appears tightly around the
   selection. Per the "Verify selection tools visually" memory: do NOT trust the
   instance count — visually confirm the box encloses the selected points.
3. Apply a class (hotkey) → confirm the resulting pointset covers the box volume
   (including points the original selection missed).
4. Repeat once from the Instances panel (right-click an instance → "Fit box to
   selection…").

- [ ] **Step 2: Update CLAUDE.md**

Add a bullet under the **Box** tool entry (or a short "Fit box to selection"
note near Cut selection), e.g.:

> **Fit box to selection** — a right-click "Fit box to selection…" affordance on
> the same three list surfaces as Cut. It calls `POST /api/segment/fit-box`
> (`backend/routes/segment.py::_fit_box_core` → `backend/labeling/fit_box.py::fit_gravity_obb`),
> which resolves the selection to full-res points and fits a **gravity-aligned
> (yaw-only) OBB** (`rotation:[0,ry,0]`), then stages it as the Box tool's
> `selBox` (`mode-label.jsx::fitBoxToSelection`, gated by `fit-eligibility.js`).
> It labels nothing — the box is an ordinary editable Box selection from there.
> Fitting to a *confirmed* instance labels only the surrounding unconfirmed
> points into a new instance (confirmed = locked). See
> `docs/superpowers/specs/2026-07-20-fit-box-to-selection-design.md`.

- [ ] **Step 3: Run the full suite one last time**

Run: `npm test`
Expected: frontend + backend all green.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document Fit box to selection in CLAUDE.md"
```

---

## Definition of Done

- `npm test` green (backend + frontend), no regressions against the 201/418 baseline.
- Right-clicking a preseg/SAM/instance selection offers "Fit box to selection…",
  gated correctly; choosing it switches to the Box tool with a tight yaw-only box
  around the selection.
- Containment parity test proves the fitted OBB contains 100% of the source points.
- Browser-verified end-to-end on a throwaway session with zero console errors.
- Spec + CLAUDE.md updated in the same branch.
