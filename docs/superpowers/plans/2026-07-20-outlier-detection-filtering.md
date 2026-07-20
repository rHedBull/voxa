# Outlier Detection Filtering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add statistical-outlier-removal (SOR) filtering to Label mode: a global "Detect outliers → Exclude" action (Feature C) and a per-selection "Remove outliers" shrink on SAM candidates and unconfirmed instances (Feature B), both driven by one shared aggressiveness knob.

**Architecture:** One pure backend function `statistical_outlier_indices(positions, subset_idx, k, std_ratio)` (kNN mean-distance thresholding) serves both features — the only difference is the *population* (`subset_idx` = all points for C, the selection's members for B). Feature C materializes the outliers as an unconfirmed `pointset` instance with class `unknown` (id 6, "Exclude / Review") via the existing `apply_reassign`; re-runs are backend-owned (an optional `replace_inst` erases the prior instance first). Feature B strips outliers back to unlabeled — SAM candidates via `_retire_sam_ids`, instances via `apply_reassign` erase. No new labeling pipeline, no viewport code, nothing destructive on disk.

**Tech Stack:** Python / FastAPI / NumPy / scipy `cKDTree` (backend); React 18 (frontend); pytest + vitest.

**Spec:** `docs/superpowers/specs/2026-07-20-outlier-detection-filtering-design.md`

---

## File Structure

**Backend**
- `backend/labeling/outliers.py` (new) — `statistical_outlier_indices`, the one pure SOR function. One responsibility: given points + a subset, return the subset's spatial outliers.
- `backend/routes/segment.py` (modify) — two thin route handlers `/api/segment/denoise` and `/api/segment/denoise-selection`, following the `_apply_shape_core` / `_cut_shape_core` pattern already in this file.
- `backend/app/schemas.py` (modify) — request/response Pydantic models.
- `backend/labeling/segment_state.py` (modify) — one small public method `remove_sam_points(indices)` (persisted `_retire_sam_ids` wrapper) so the route never calls a private method.

**Frontend**
- `frontend/src/outlier-eligibility.js` (new) — pure rule for enabling "Remove outliers" (mirrors `cut-eligibility.js`).
- `frontend/src/api.js` (modify) — `denoise(...)` and `denoiseSelection(...)` clients (decode `scan_indices_b64` → `indices` inside, mirroring `cutShape`).
- `frontend/src/segment-state.js` (modify) — `export` the existing `retireSamIdsForIndices` (currently module-private, line 48) so Feature B's SAM shrink can reuse it.
- `frontend/src/sam-segment-list.jsx` (modify) — add an `onRemoveOutliers` prop and a second "Remove outliers" item to the SAM-row context menu (the SAM menu lives HERE, not in `mode-label.jsx`).
- `frontend/src/mode-label.jsx` (modify) — "Detect outliers" button + aggressiveness slider, the two handlers, re-run instance-id tracking, thread `onRemoveOutliers` into `<SamSegmentList>`, and add the "Remove outliers" item to the Instances-panel context menu.

**Tests**
- `backend/tests/test_outliers.py` (new) — pure-fn unit tests.
- `backend/tests/test_denoise_routes.py` (new) — endpoint tests.
- `frontend/src/outlier-eligibility.test.js` (new) — eligibility unit tests.

**Docs**
- `CLAUDE.md` (modify) — document the tool in the same PR.

---

## Constants

Exclude class id is `6` (`config/classes.yaml` → `unknown`, label "Exclude / Review").
SOR defaults: `k = 16`, `std_ratio = 2.0`.

---

### Task 1: Pure SOR function `statistical_outlier_indices`

**Files:**
- Create: `backend/labeling/outliers.py`
- Test: `backend/tests/test_outliers.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_outliers.py
import numpy as np
from labeling.outliers import statistical_outlier_indices


def _cluster(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.1, size=(n, 3)).astype(np.float32)


def test_flags_planted_specks_only():
    core = _cluster(500)
    specks = np.array([[10, 0, 0], [0, 12, 0], [-11, 0, 5]], dtype=np.float32)
    pts = np.vstack([core, specks])
    subset = np.arange(pts.shape[0])
    out = statistical_outlier_indices(pts, subset, k=16, std_ratio=2.0)
    # exactly the three planted specks (indices 500, 501, 502)
    assert set(out.tolist()) == {500, 501, 502}


def test_subset_scoping_returns_full_res_indices():
    # An outlier is judged only against the subset population, and returned
    # indices are into `positions`, not into the subset.
    core = _cluster(300)
    speck = np.array([[8, 8, 8]], dtype=np.float32)
    pts = np.vstack([core, speck])           # speck at index 300
    subset = np.array([250, 260, 270, 280, 290, 300], dtype=np.int64)
    out = statistical_outlier_indices(pts, subset, k=3, std_ratio=1.0)
    assert 300 in out.tolist()
    assert all(i in subset.tolist() for i in out.tolist())


def test_lower_std_ratio_flags_superset():
    core = _cluster(400)
    specks = (np.random.default_rng(1).normal(0, 3, size=(20, 3))
              + np.array([5, 5, 5])).astype(np.float32)
    pts = np.vstack([core, specks])
    subset = np.arange(pts.shape[0])
    greedy = set(statistical_outlier_indices(pts, subset, std_ratio=1.0).tolist())
    strict = set(statistical_outlier_indices(pts, subset, std_ratio=2.5).tolist())
    assert strict.issubset(greedy)


def test_degenerate_inputs_return_empty_not_crash():
    pts = _cluster(10)
    # empty subset
    assert statistical_outlier_indices(pts, np.array([], dtype=np.int64)).size == 0
    # subset smaller than k+1 -> not enough neighbors to judge -> empty
    assert statistical_outlier_indices(pts, np.array([0, 1, 2]), k=16).size == 0
    # all-identical points -> zero variance -> nothing flagged
    same = np.zeros((50, 3), dtype=np.float32)
    assert statistical_outlier_indices(same, np.arange(50)).size == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_outliers.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'labeling.outliers'`

- [ ] **Step 3: Write the implementation**

```python
# backend/labeling/outliers.py
"""Statistical Outlier Removal (SOR) over a point subset.

One pure function, shared by the global denoise and per-selection
"remove outliers" features. The population it judges against is the
caller-supplied `subset_idx`, so the same code flags a floating speck
against the whole cloud (global) or an edge-stray against a selection's
core (per-selection). See
docs/superpowers/specs/2026-07-20-outlier-detection-filtering-design.md.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def statistical_outlier_indices(
    positions: np.ndarray,
    subset_idx: np.ndarray,
    k: int = 16,
    std_ratio: float = 2.0,
) -> np.ndarray:
    """Return the indices (into `positions`) of the spatial outliers among
    `positions[subset_idx]`.

    For each subset point, compute the mean distance to its `k` nearest
    neighbours within the subset; flag points whose mean distance exceeds
    `mean + std_ratio * std` of that distribution. Density-adaptive: the
    threshold comes from the subset's own distances.
    """
    subset_idx = np.asarray(subset_idx, dtype=np.int64).ravel()
    n = subset_idx.size
    # Need at least k+1 points (self + k neighbours) to judge anything.
    if n < k + 1:
        return np.empty(0, dtype=np.int64)
    pts = positions[subset_idx]
    tree = cKDTree(pts)
    # query k+1 because the first neighbour is the point itself (dist 0).
    dists, _ = tree.query(pts, k=k + 1)
    mean_knn = dists[:, 1:].mean(axis=1)      # drop the self-distance column
    mu = float(mean_knn.mean())
    sigma = float(mean_knn.std())
    if sigma == 0.0:                           # zero variance -> no outliers
        return np.empty(0, dtype=np.int64)
    threshold = mu + std_ratio * sigma
    is_outlier = mean_knn > threshold
    return subset_idx[is_outlier]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest backend/tests/test_outliers.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/labeling/outliers.py backend/tests/test_outliers.py
git commit -m "feat(backend): statistical outlier removal (SOR) pure function"
```

---

### Task 2: `remove_sam_points` — persisted SAM-shrink helper

`_retire_sam_ids` is private and does not persist. Feature B's SAM path needs a public, autosaving shrink. Add a thin wrapper next to `materialize_sam_segment` in `segment_state.py`.

**Files:**
- Modify: `backend/labeling/segment_state.py` (add method after `materialize_sam_segment`, ~line 200)
- Test: `backend/tests/test_denoise_routes.py` (create now; add a unit test for this method)

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_denoise_routes.py
import numpy as np
from labeling.segment_state import SegmentSession


def _session(n=100):
    pos = np.random.default_rng(0).normal(size=(n, 3)).astype(np.float32)
    cls = np.full(n, -1, dtype=np.int8)
    inst = np.full(n, -1, dtype=np.int32)
    return SegmentSession(cls, inst, pos)


def test_remove_sam_points_shrinks_candidate():
    seg = _session()
    out = seg.materialize_sam_segment(np.arange(0, 40, dtype=np.int32), source="sam")
    sid = out["sam_seg_id"]
    assert seg.sam_segments[sid]["n_points"] == 40
    seg.remove_sam_points(np.arange(0, 10, dtype=np.int32))
    assert seg.sam_segments[sid]["n_points"] == 30
    assert int((seg.sam_ids == sid).sum()) == 30
    # removed points are back to no candidacy
    assert bool((seg.sam_ids[np.arange(0, 10)] == -1).all())
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `.venv/bin/pytest backend/tests/test_denoise_routes.py::test_remove_sam_points_shrinks_candidate -v`
Expected: FAIL with `AttributeError: 'SegmentSession' object has no attribute 'remove_sam_points'`

- [ ] **Step 3: Implement the method**

Add after `materialize_sam_segment` (after line ~199) in `backend/labeling/segment_state.py`:

```python
    def remove_sam_points(self, indices: np.ndarray) -> dict:
        """Drop SAM candidacy for these points and persist — used by
        per-selection "remove outliers" to shrink a candidate. Reuses the
        same _retire_sam_ids bookkeeping that a real label triggers, then
        schedules an autosave (unlike the private helper, which is only
        ever called mid-apply where the caller autosaves)."""
        indices = np.asarray(indices, dtype=np.int32)
        if indices.size == 0:
            return {"op": "remove_sam_points", "n_affected": 0}
        self._retire_sam_ids(indices)
        self.schedule_autosave(write_arrays=True)
        return {"op": "remove_sam_points", "n_affected": int(indices.size)}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `.venv/bin/pytest backend/tests/test_denoise_routes.py::test_remove_sam_points_shrinks_candidate -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/labeling/segment_state.py backend/tests/test_denoise_routes.py
git commit -m "feat(backend): remove_sam_points — persisted SAM-candidate shrink"
```

---

### Task 3: Schemas for `/denoise` and `/denoise-selection`

**Files:**
- Modify: `backend/app/schemas.py`

Find the existing `ApplyShapeRequest` / `CutShapeRequest` models and add these nearby, matching their style (Pydantic v2, `from pydantic import BaseModel`).

- [ ] **Step 1: Add the schemas**

```python
class DenoiseRequest(BaseModel):
    std_ratio: float = 2.0
    k: int = 16
    # When set, the prior denoise instance is erased to unlabeled before the
    # new outlier set is applied (backend-owned re-run replacement).
    replace_inst: Optional[int] = None
    # Confirmed instances that denoise must not overwrite ("confirmed = locked").
    protect_instances: list[int] = []


class DenoiseResponse(BaseModel):
    instance_id: Optional[int] = None
    n_affected: int = 0
    n_protected: int = 0
    scan_indices_b64: Optional[str] = None
    dirty: bool = False


class DenoiseSelectionRequest(BaseModel):
    source: Literal["sam", "instance"]
    id: int
    std_ratio: float = 2.0
    k: int = 16


class DenoiseSelectionResponse(BaseModel):
    source: str
    id: int
    n_removed: int = 0
    n_kept: int = 0
    scan_indices_b64: Optional[str] = None   # the removed (outlier) points
    dirty: bool = False
```

Confirm `Optional` and `Literal` are already imported at the top of `schemas.py` (they are used by existing models such as `CutShapeSource`); if not, add `from typing import Optional, Literal`.

- [ ] **Step 2: Verify import**

Run: `.venv/bin/python -c "from app.schemas import DenoiseRequest, DenoiseSelectionRequest; print('ok')"`
(Run from `backend/`, or `cd backend && ...`; matches how other schema-only checks run.)
Expected: `ok`

- [ ] **Step 3: Commit**

```bash
git add backend/app/schemas.py
git commit -m "feat(backend): request/response schemas for denoise endpoints"
```

---

### Task 4: `POST /api/segment/denoise` (Feature C)

**Files:**
- Modify: `backend/routes/segment.py` (add a `_denoise_core` helper + route, near `_apply_shape_core`)
- Test: `backend/tests/test_denoise_routes.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_denoise_routes.py`:

```python
from fastapi.testclient import TestClient


def _client_with_cloud():
    """Load a session with a planted-speck cloud into _state and return a
    TestClient. Mirrors how other route tests seed _state."""
    import main
    from app.core import _state
    core = np.random.default_rng(0).normal(0, 0.1, size=(400, 3)).astype(np.float32)
    specks = np.array([[9, 0, 0], [0, 9, 0], [0, 0, 9]], dtype=np.float32)
    pos = np.vstack([core, specks])
    seg = SegmentSession(np.full(pos.shape[0], -1, np.int8),
                         np.full(pos.shape[0], -1, np.int32), pos)
    _state["seg"] = seg
    return TestClient(main.app), seg


def test_denoise_materializes_exclude_instance():
    client, seg = _client_with_cloud()
    r = client.post("/api/segment/denoise", json={"std_ratio": 2.0})
    assert r.status_code == 200
    body = r.json()
    assert body["n_affected"] == 3            # the three specks
    inst = body["instance_id"]
    # the three speck points now carry the Exclude class (id 6)
    speck_idx = [400, 401, 402]
    assert bool((seg.class_ids[speck_idx] == 6).all())
    assert bool((seg.instance_ids[speck_idx] == inst).all())


def test_denoise_replace_inst_erases_prior():
    client, seg = _client_with_cloud()
    first = client.post("/api/segment/denoise", json={"std_ratio": 2.0}).json()
    inst1 = first["instance_id"]
    # Re-run replacing inst1: a stricter ratio flags fewer/equal points, and
    # the prior instance's points must be erased (no orphan Exclude labels
    # under a dead instance id).
    second = client.post("/api/segment/denoise",
                         json={"std_ratio": 2.0, "replace_inst": inst1}).json()
    inst2 = second["instance_id"]
    assert inst2 != inst1
    # No point still carries the dead inst1 id.
    assert int((seg.instance_ids == inst1).sum()) == 0


def test_denoise_empty_returns_null_instance():
    client, seg = _client_with_cloud()
    # Absurdly high ratio flags nothing.
    body = client.post("/api/segment/denoise", json={"std_ratio": 50.0}).json()
    assert body["instance_id"] is None
    assert body["n_affected"] == 0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_denoise_routes.py -k denoise_ -v`
Expected: FAIL (404 / route not found)

- [ ] **Step 3: Implement the route**

Add to `backend/routes/segment.py` (near `_apply_shape_core`). `_require_seg`, `_b64`, `_serialize_apply`, `HTTPException`, `np` are already available in this module via the wildcard imports.

```python
EXCLUDE_CLASS_ID = 6   # config/classes.yaml -> unknown, "Exclude / Review"


def _denoise_core(seg, req) -> dict:
    from labeling.outliers import statistical_outlier_indices
    # Backend-owned re-run replacement: erase the prior denoise instance's
    # points to unlabeled BEFORE recomputing (deleteInstance on the frontend
    # only drops the row, never the working-array labels).
    if req.replace_inst is not None:
        old = np.flatnonzero(seg.instance_ids == int(req.replace_inst)).astype(np.int32)
        if old.size:
            seg.apply_reassign(old, target_inst=None, target_class=None)
    all_idx = np.arange(seg.positions.shape[0], dtype=np.int64)
    outliers = statistical_outlier_indices(
        seg.positions, all_idx, k=int(req.k), std_ratio=float(req.std_ratio))
    if outliers.size == 0:
        return {"instance_id": None, "n_affected": 0, "n_protected": 0,
                "scan_indices_b64": None, "dirty": bool(seg.dirty)}
    out = seg.apply_reassign(
        outliers.astype(np.int32), target_inst=-1, target_class=EXCLUDE_CLASS_ID,
        protect_instances=req.protect_instances or None)
    if out["n_affected"] == 0:            # everything caught was locked/confirmed
        return {"instance_id": None, "n_affected": 0,
                "n_protected": out.get("n_protected", 0),
                "scan_indices_b64": None, "dirty": bool(seg.dirty)}
    return {"instance_id": int(out["new_instance_id"]),
            "n_affected": int(out["n_affected"]),
            "n_protected": out.get("n_protected", 0),
            "scan_indices_b64": _b64(out["indices"].astype(np.int32, copy=False)),
            "dirty": bool(seg.dirty)}


@router.post("/api/segment/denoise", response_model=DenoiseResponse)
def denoise(req: DenoiseRequest):
    """Global outlier detection: flag spatial outliers cloud-wide and
    materialize them as one unconfirmed Exclude pointset (Feature C).
    See docs/superpowers/specs/2026-07-20-outlier-detection-filtering-design.md."""
    seg = _require_seg()
    return _denoise_core(seg, req)
```

Note (accepted divergence from spec): `statistical_outlier_indices` builds a fresh
`cKDTree` over all points on each call rather than reusing `seg._ensure_tree()`. This
rebuilds the (~single-digit-million-point) tree on every slider re-run, but keeps the
SOR function pure and single-responsibility. Acceptable under the spec's
"synchronous + spinner" budget; do not prematurely optimize by threading the cached
tree into the pure function.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest backend/tests/test_denoise_routes.py -k denoise_ -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add backend/routes/segment.py backend/tests/test_denoise_routes.py
git commit -m "feat(backend): POST /api/segment/denoise — global outliers to Exclude"
```

---

### Task 5: `POST /api/segment/denoise-selection` (Feature B)

**Files:**
- Modify: `backend/routes/segment.py`
- Test: `backend/tests/test_denoise_routes.py`

- [ ] **Step 1: Write the failing tests**

Append to `backend/tests/test_denoise_routes.py`:

```python
def _client_with_selection(kind):
    """Seed a SAM candidate or an instance whose members = a tight core plus
    a couple of far strays, and return (client, seg, id, stray_full_indices)."""
    import main
    from app.core import _state
    core = np.random.default_rng(2).normal(0, 0.1, size=(120, 3)).astype(np.float32)
    strays = np.array([[7, 0, 0], [0, 7, 0]], dtype=np.float32)
    pos = np.vstack([core, strays])          # strays at 120, 121
    seg = SegmentSession(np.full(pos.shape[0], -1, np.int8),
                         np.full(pos.shape[0], -1, np.int32), pos)
    members = np.arange(0, 122, dtype=np.int32)   # whole cloud is the selection
    if kind == "sam":
        out = seg.materialize_sam_segment(members, source="sam")
        sid = out["sam_seg_id"]
    else:
        out = seg.apply_reassign(members, target_inst=-1, target_class=0)  # class pipe
        sid = out["new_instance_id"]
    _state["seg"] = seg
    from fastapi.testclient import TestClient
    return TestClient(main.app), seg, sid, [120, 121]


def test_denoise_selection_sam_retires_strays():
    client, seg, sid, strays = _client_with_selection("sam")
    r = client.post("/api/segment/denoise-selection",
                    json={"source": "sam", "id": sid, "std_ratio": 1.5})
    assert r.status_code == 200
    body = r.json()
    assert body["n_removed"] == 2
    assert bool((seg.sam_ids[strays] == -1).all())
    assert seg.sam_segments[sid]["n_points"] == 120


def test_denoise_selection_instance_erases_strays():
    client, seg, sid, strays = _client_with_selection("instance")
    r = client.post("/api/segment/denoise-selection",
                    json={"source": "instance", "id": sid, "std_ratio": 1.5})
    body = r.json()
    assert body["n_removed"] == 2
    # strays back to unlabeled; core still labelled with the instance
    assert bool((seg.instance_ids[strays] == -1).all())
    assert int((seg.instance_ids == sid).sum()) == 120
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_denoise_routes.py -k denoise_selection -v`
Expected: FAIL (404)

- [ ] **Step 3: Implement the route**

Add to `backend/routes/segment.py`:

```python
@router.post("/api/segment/denoise-selection", response_model=DenoiseSelectionResponse)
def denoise_selection(req: DenoiseSelectionRequest):
    """Per-selection "remove outliers" (Feature B): strip a selection's
    spatial outliers back to unlabeled. SAM candidate -> drop candidacy;
    unconfirmed instance -> erase to (-1,-1). Presegs are out of scope."""
    from labeling.outliers import statistical_outlier_indices
    seg = _require_seg()
    if req.source == "sam":
        membership = seg.sam_ids == int(req.id)
    else:
        membership = seg.instance_ids == int(req.id)
    subset = np.flatnonzero(membership).astype(np.int64)
    if subset.size == 0:
        raise HTTPException(404, f"{req.source} selection {req.id} is empty")
    outliers = statistical_outlier_indices(
        seg.positions, subset, k=int(req.k), std_ratio=float(req.std_ratio))
    n_kept = int(subset.size - outliers.size)
    if outliers.size == 0:
        return DenoiseSelectionResponse(source=req.source, id=req.id,
                                        n_removed=0, n_kept=n_kept, dirty=bool(seg.dirty))
    out_i32 = outliers.astype(np.int32)
    if req.source == "sam":
        seg.remove_sam_points(out_i32)
    else:
        seg.apply_reassign(out_i32, target_inst=None, target_class=None)
    return DenoiseSelectionResponse(
        source=req.source, id=req.id, n_removed=int(outliers.size), n_kept=n_kept,
        scan_indices_b64=_b64(out_i32), dirty=bool(seg.dirty))
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `.venv/bin/pytest backend/tests/test_denoise_routes.py -v`
Expected: PASS (all in file)

- [ ] **Step 5: Commit**

```bash
git add backend/routes/segment.py backend/tests/test_denoise_routes.py
git commit -m "feat(backend): POST /api/segment/denoise-selection — per-selection outlier shrink"
```

---

### Task 6: Frontend eligibility helper

**Files:**
- Create: `frontend/src/outlier-eligibility.js`
- Test: `frontend/src/outlier-eligibility.test.js`

- [ ] **Step 1: Write the failing tests**

```javascript
// frontend/src/outlier-eligibility.test.js
import { describe, it, expect } from 'vitest';
import { removeOutliersEligibility } from './outlier-eligibility.js';

describe('removeOutliersEligibility', () => {
  it('sam: eligible iff exactly one candidate is selected', () => {
    expect(removeOutliersEligibility({ list: 'sam', selectionSize: 1 }).eligible).toBe(true);
    expect(removeOutliersEligibility({ list: 'sam', selectionSize: 0 }).eligible).toBe(false);
    expect(removeOutliersEligibility({ list: 'sam', selectionSize: 3 }).eligible).toBe(false);
  });

  it('instance: eligible iff selected and not confirmed', () => {
    expect(removeOutliersEligibility({ list: 'instance', isSelected: true, confirmed: false }).eligible).toBe(true);
    expect(removeOutliersEligibility({ list: 'instance', isSelected: true, confirmed: true }).eligible).toBe(false);
    expect(removeOutliersEligibility({ list: 'instance', isSelected: false, confirmed: false }).eligible).toBe(false);
  });

  it('throws on unknown list', () => {
    expect(() => removeOutliersEligibility({ list: 'preseg' })).toThrow();
  });
});
```

Note: unlike cut, the SAM case requires **exactly one** candidate (`selectionSize === 1`) — `/denoise-selection` operates on a single `id`, not a multi-selection.

- [ ] **Step 2: Run to verify fail**

Run (from `frontend/`): `npx vitest run src/outlier-eligibility.test.js`
Expected: FAIL (module not found)

- [ ] **Step 3: Implement**

```javascript
// frontend/src/outlier-eligibility.js
// outlier-eligibility.js — pure rule for enabling the right-click "Remove
// outliers" menu item. Mirrors cut-eligibility.js, with one difference: the
// SAM case requires EXACTLY ONE selected candidate, because
// /api/segment/denoise-selection targets a single { source, id } — there is
// no multi-selection form. Presegs are out of scope (immutable layer), so
// only 'sam' and 'instance' lists are handled.
export function removeOutliersEligibility(params) {
  const { list } = params;
  if (list === 'sam') {
    const { selectionSize } = params;
    if (selectionSize === 1) return { eligible: true };
    return { eligible: false, reason: selectionSize > 1 ? 'multi' : 'empty' };
  }
  if (list === 'instance') {
    const { isSelected, confirmed } = params;
    if (!isSelected) return { eligible: false, reason: 'not-selected' };
    if (confirmed) return { eligible: false, reason: 'confirmed' };
    return { eligible: true };
  }
  throw new Error(`removeOutliersEligibility: unknown list "${list}"`);
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run src/outlier-eligibility.test.js`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/outlier-eligibility.js frontend/src/outlier-eligibility.test.js
git commit -m "feat(frontend): removeOutliersEligibility rule"
```

---

### Task 7: API client methods

**Files:**
- Modify: `frontend/src/api.js` (add after `cutShape`, ~line 277)

- [ ] **Step 1: Add the methods**

Mirror `cutShape` (~line 265): decode the base64 index payload **inside** the client with the existing `b64ToInt32` (`api.js:396`) and expose a ready `indices` field, so callers never touch base64. Confirm `b64ToInt32` is imported/in scope in `api.js` (it is — used at the top of the file).

```javascript
  async denoise({ stdRatio = 2.0, k = 16, replaceInst = null, protectInstances = [] }) {
    const r = await fetch('/api/segment/denoise', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        std_ratio: stdRatio, k,
        replace_inst: replaceInst, protect_instances: protectInstances,
      }),
    });
    if (!r.ok) throw new Error(`denoise failed: ${r.status} ${await r.text()}`);
    const j = await r.json();
    return { ...j, indices: j.scan_indices_b64 ? b64ToInt32(j.scan_indices_b64) : null };
  },

  async denoiseSelection({ source, id, stdRatio = 2.0, k = 16 }) {
    const r = await fetch('/api/segment/denoise-selection', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source, id, std_ratio: stdRatio, k }),
    });
    if (!r.ok) throw new Error(`denoiseSelection failed: ${r.status} ${await r.text()}`);
    const j = await r.json();
    return { ...j, indices: j.scan_indices_b64 ? b64ToInt32(j.scan_indices_b64) : null };
  },
```

Match the surrounding style exactly (these are methods on the exported `VoxaAPI` object — confirm whether entries use `async name()` inside an object literal or `VoxaAPI.name = async …`, and follow that; `cutShape` at ~line 265 is the template).

- [ ] **Step 2: Verify build parses**

Run (from `frontend/`): `npx vitest run src/api.test.js`
Expected: PASS (existing api tests still green — no regression)

- [ ] **Step 3: Commit**

```bash
git add frontend/src/api.js
git commit -m "feat(frontend): VoxaAPI.denoise + denoiseSelection clients"
```

---

### Task 8: Wire Feature C — "Detect outliers" button + slider

**Files:**
- Modify: `frontend/src/mode-label.jsx`

Read the existing "⬇ Export…" button block (~line 1240-1247) and the `onCutConfirmedHandler` pointset-row add (~line 586-595) first — this task mirrors both.

- [ ] **Step 1: Add denoise state + handler**

Near the other Label-mode state (e.g. by `const [exportOpen, setExportOpen] = useStateLabel(false);` at line 357), add:

```javascript
  const [denoiseRatio, setDenoiseRatio] = useStateLabel(2.0);
  const [denoiseInstId, setDenoiseInstId] = useStateLabel(null);  // backend instance id of the live denoise result
  const [denoiseBusy, setDenoiseBusy] = useStateLabel(false);
```

Add the handler near `onCutConfirmedHandler`:

```javascript
  const runDenoise = useCallbackLabel(async () => {
    if (!segState || denoiseBusy) return;
    setDenoiseBusy(true);
    try {
      const resp = await VoxaAPI.denoise({
        stdRatio: denoiseRatio,
        replaceInst: denoiseInstId,           // erase the prior result first
        protectInstances: protectedSegIds,
      });
      // Drop the previous denoise row (its points were just erased server-side).
      const cls = classes.find((c) => c.class_id === 6);   // Exclude / Review
      const kept = instances.filter((i) => i.segId !== denoiseInstId);
      if (resp.instance_id == null) {
        onChange(kept);
        setDenoiseInstId(null);
      } else {
        const idx = resp.indices;                            // Int32Array (decoded in api.js)
        const afterClass = new Int8Array(idx.length).fill(6);
        const afterInstance = new Int32Array(idx.length).fill(resp.instance_id);
        setSegState((s) => (s ? applyDelta(s, {
          indices: idx, after_class: afterClass, after_instance: afterInstance,
        }) : s));
        onChange([...kept, {
          id: newId(),
          segId: resp.instance_id,
          kind: 'pointset',
          cls: cls.id,
          label: `${cls.label} ${(counts[cls.id] || 0) + 1}`,
          color: cls.color,
          source: 'denoise',
          confirmed: false,
        }]);
        setDenoiseInstId(resp.instance_id);
      }
    } catch (e) {
      console.error('denoise failed', e);
    } finally {
      setDenoiseBusy(false);
    }
  }, [segState, denoiseBusy, denoiseRatio, denoiseInstId, protectedSegIds,
      classes, instances, counts, onChange, setSegState]);
```

Notes for the implementer:
- The API client already decodes indices (Task 7 exposes `resp.indices` as an `Int32Array`, mirroring `cutShape` which returns `resp.instance.indices` / `m.indices`). Do not hand-roll base64 decoding here.
- `applyDelta`, `newId`, `useStateLabel`, `useCallbackLabel`, `useMemoLabel` are already imported/defined in this file (used by `onCutConfirmedHandler`); reuse them.
- `classes.find(c => c.class_id === 6)` — confirm the class-object shape (`class_id` numeric vs `id` string) from how `onCutConfirmedHandler` reads `cls.class_id` and `cls.id`.

- [ ] **Step 2: Add the button + slider next to Export**

In the left-rail block by the Export button (~line 1240), add:

```jsx
        <div className="denoise-row" style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
          <button
            className="ghost-btn"
            disabled={!activeSessionId || denoiseBusy}
            title={activeSessionId ? 'Detect stray outlier points and stage them as Exclude'
                                   : 'Open a session first'}
            onClick={runDenoise}>
            {denoiseBusy ? '… Detecting' : '✧ Detect outliers'}
          </button>
          <input type="range" min="1" max="3" step="0.1"
            value={denoiseRatio}
            onChange={(e) => setDenoiseRatio(parseFloat(e.target.value))}
            title={`Aggressiveness (σ=${denoiseRatio.toFixed(1)}; lower = greedier)`} />
        </div>
```

- [ ] **Step 3: Verify build**

Run (from `frontend/`): `npx vitest run` (whole suite — no regressions)
Expected: PASS

- [ ] **Step 4: Browser-verify** — REQUIRED SUB-SKILL: `browser-verification`

Follow the memory `feedback_browser_verify_mutates_session`: restart the stale :8765 backend first and use a **throwaway session** (denoise auto-saves to disk). Load an annotated scan, open Label mode, click "Detect outliers", confirm:
- caught points highlight as an unconfirmed Exclude instance row appears in the Instances panel;
- moving the slider + re-clicking **replaces** (does not stack) the Exclude instance;
- zero console errors; the `/api/segment/denoise` network call returns 200.
Screenshot the caught points (per `feedback_verify_selection_visually`: confirm *which* points get selected, don't trust counts).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/mode-label.jsx
git commit -m "feat(frontend): Detect outliers button + aggressiveness slider (Feature C)"
```

---

### Task 9: Wire Feature B — "Remove outliers" context-menu item

**Files:**
- Modify: `frontend/src/segment-state.js` (export `retireSamIdsForIndices`)
- Modify: `frontend/src/sam-segment-list.jsx` (SAM-row menu — the SAM menu lives here, NOT in `mode-label.jsx`)
- Modify: `frontend/src/mode-label.jsx` (handler + Instances-panel menu + thread the new prop)

**Grounding (verified against the code):**
- Both context menus are built with `ContextMenu items={[{ label, disabled, onSelect }]}` — items use **`onSelect`** and a **`disabled`** boolean, NOT `onClick` and NOT conditional array-spread (`context-menu.jsx:35`; existing "Edit selection…" items at `sam-segment-list.jsx:78-83` and `mode-label.jsx:1592-1601`). Match that shape: render "Remove outliers" always, gate it with `disabled`.
- The SAM menu is inside `SamSegmentList` (`sam-segment-list.jsx`), which today takes `{ segState, setSegState, onEditSelection }`. Add an `onRemoveOutliers` prop and thread it from `mode-label.jsx:1287`.
- `retireSamIdsForIndices(state, indices)` (`segment-state.js:48`) is the exact SAM-shrink helper — it clears candidacy for those indices and shrinks/drops the `samSegments` count (returns state unchanged if none carry a live sam id). It is currently **module-private**; export it. Do NOT use `applySamDelta` with `samSegId: -1` — that unconditionally does `samSegments.set(-1, …)` (`segment-state.js:90`) and injects a phantom candidate.

- [ ] **Step 1: Export the SAM-shrink helper**

In `frontend/src/segment-state.js` change line 48 from `function retireSamIdsForIndices(` to `export function retireSamIdsForIndices(`. (No behavior change — `applyDelta` still calls it locally.) Run `npx vitest run src/` afterward to confirm nothing broke.

- [ ] **Step 2: Add the handler in `mode-label.jsx`**

Import at the top: `removeOutliersEligibility` from `./outlier-eligibility.js` and `retireSamIdsForIndices` from `./segment-state.js` (add to the existing `segment-state.js` import).

```javascript
  const removeOutliers = useCallbackLabel(async ({ source, id }) => {
    if (!segState) return;
    try {
      const resp = await VoxaAPI.denoiseSelection({ source, id, stdRatio: denoiseRatio });
      if (!resp.n_removed || !resp.indices) return;
      const idx = resp.indices;                       // Int32Array (decoded in api.js)
      setSegState((s) => {
        if (!s) return s;
        if (source === 'sam') {
          // Strays lose SAM candidacy; candidate shrinks (mirrors backend
          // remove_sam_points -> _retire_sam_ids).
          return retireSamIdsForIndices(s, idx);
        }
        // instance: strays back to unlabeled
        const afterClass = new Int8Array(idx.length).fill(-1);
        const afterInstance = new Int32Array(idx.length).fill(-1);
        return applyDelta(s, { indices: idx, after_class: afterClass, after_instance: afterInstance });
      });
    } catch (e) {
      console.error('removeOutliers failed', e);
    }
  }, [segState, denoiseRatio, setSegState]);
```

- [ ] **Step 3: Add the SAM-list menu item (`sam-segment-list.jsx`)**

Add `onRemoveOutliers = null` to the `SamSegmentList({ … })` destructured props, import `removeOutliersEligibility` from `./outlier-eligibility.js`, and make the `ContextMenu` `items` a two-entry array — keep the existing "Edit selection…" item and append:

```jsx
          {
            label: 'Remove outliers',
            disabled: !removeOutliersEligibility({ list: 'sam', selectionSize: segState.samSelection.size }).eligible,
            onSelect: () => {
              if (!onRemoveOutliers) return;
              onRemoveOutliers({ source: 'sam', id: [...segState.samSelection][0] });
            },
          },
```

(Eligibility requires exactly one selected candidate, so `[...segState.samSelection][0]` is the target id.)

- [ ] **Step 4: Thread the prop + add the Instances-panel item (`mode-label.jsx`)**

Pass the handler into the SAM list at ~line 1287:

```jsx
          onRemoveOutliers={removeOutliers}
```

In the Instances-panel context menu (`instCutMenu`, ~line 1592), make `items` a two-entry array — keep "Edit selection…" and append (reusing the `target`/`elig` already computed there; compute a second eligibility for this item):

```jsx
                {
                  label: 'Remove outliers',
                  disabled: !removeOutliersEligibility({
                    list: 'instance',
                    isSelected: instCutMenu.instId === selectedId,
                    confirmed: !!target?.confirmed,
                  }).eligible,
                  onSelect: () => {
                    if (!target || !Number.isFinite(target.segId)) return;
                    removeOutliers({ source: 'instance', id: target.segId });
                  },
                },
```

- [ ] **Step 5: Verify build**

Run (from `frontend/`): `npx vitest run`
Expected: PASS (whole suite green; existing `sam-segment-list.jsdom.test.jsx` / `context-menu.test.jsx` still pass)

- [ ] **Step 6: Browser-verify** — REQUIRED SUB-SKILL: `browser-verification`

Throwaway session; restart backend first. Create a SAM candidate (or use an existing one) with visible edge-strays, select exactly one, right-click its row → "Remove outliers" → confirm the strays drop from the candidate (screenshot before/after, per `feedback_verify_selection_visually`). Repeat on an unconfirmed instance. Zero console errors; `/api/segment/denoise-selection` returns 200. Confirm the item is **disabled** on a confirmed instance and when 0 or >1 SAM candidates are selected.

- [ ] **Step 7: Commit**

```bash
git add frontend/src/segment-state.js frontend/src/sam-segment-list.jsx frontend/src/mode-label.jsx
git commit -m "feat(frontend): Remove outliers context-menu item (Feature B)"
```

---

### Task 10: Run the simplify skill on the diff

- [ ] **Step 1:** Invoke the `simplify` skill on the full diff (`git diff main...HEAD`) to tidy reuse/duplication/efficiency before docs. Apply its suggestions; re-run the suites (`npm test`) after.

- [ ] **Step 2: Commit** any simplifications:

```bash
git add -A && git commit -m "refactor: simplify outlier-filtering diff"
```

---

### Task 11: Full test suite + docs

**Files:**
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run the full suite**

Run: `npm test`
Expected: all backend (pytest) + frontend (vitest) green. Record the counts.

- [ ] **Step 2: Update `CLAUDE.md`**

In the Label-mode tool description area (near the Cut selection bullet / the "Confirmed = locked" section), add a concise paragraph describing:
- **Global denoise (Feature C):** "Detect outliers" button + aggressiveness slider → SOR over the whole cloud → outliers materialize as one unconfirmed `pointset` instance, class `unknown` (Exclude / Review); `POST /api/segment/denoise` with backend-owned re-run replacement (`replace_inst`); protects confirmed instances.
- **Per-selection remove-outliers (Feature B):** "Remove outliers" right-click item on SAM candidates + unconfirmed instances (presegs out of scope); `POST /api/segment/denoise-selection` — SAM drops candidacy, instance erases strays to unlabeled.
- Shared pure fn `backend/labeling/outliers.py::statistical_outlier_indices`.
- Link the spec: `docs/superpowers/specs/2026-07-20-outlier-detection-filtering-design.md`.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document outlier detection filtering (Features B + C)"
```

---

## Definition of Done

- `statistical_outlier_indices` unit-tested (planted specks, subset scoping, monotonicity, degenerate inputs).
- `/api/segment/denoise` materializes an Exclude instance, replaces on re-run (`replace_inst`), protects confirmed, returns null-instance on empty.
- `/api/segment/denoise-selection` shrinks SAM candidates (`remove_sam_points`) and unconfirmed instances (erase), returns removed indices.
- Frontend: button + slider (C), context-menu item (B), eligibility helper unit-tested.
- Both features browser-verified (points highlighted/removed correctly, zero console errors, 200s) on a throwaway session.
- `npm test` fully green; `CLAUDE.md` updated in the same PR.
