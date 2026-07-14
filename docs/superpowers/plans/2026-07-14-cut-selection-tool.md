# Cut-selection tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let a user multi-select presegments/SAM candidates (or a single unconfirmed instance), box-cut a sub-region out of them in an isolated modal viewport, and have the cut-out points land as new segments/instance of the same kind as their original source — never merging provenance across sources.

**Architecture:** Backend gains a `source` tag on the existing `sam_ids`/`sam_segments` mutable candidate layer (generalizing it beyond SAM) and one new endpoint, `POST /api/segment/cut-shape`, that resolves a client-drawn OBB against full-resolution positions and partitions/materializes per source server-side (mirroring `/apply-shape`'s existing pattern). Frontend extracts the Box tool's draw/transform logic out of `mode-label.jsx` into a reusable module, builds a new isolated-viewport modal that reuses it, and adds a right-click "Edit selection…" menu to the three list surfaces (Presegment, SAM, Instances).

**Tech Stack:** FastAPI + Pydantic (backend), React + Three.js, Vitest (frontend), pytest (backend). No new dependencies.

**Spec:** `docs/superpowers/specs/2026-07-14-cut-selection-tool-design.md` — read this first; it is authoritative for every "why."

---

## File Structure

**Backend:**
- Modify `backend/labeling/segment_state.py` — `materialize_sam_segment` gains `source` param; `sam_segments` dict entries gain `source` key.
- Modify `backend/labeling/segment_io.py` — `sam_segments_to_list`/`save_sam_segments`/`load_sam_segments` round-trip `source`.
- Modify `backend/routes/sam.py` — `/api/sam/project` passes `source='sam'`.
- Modify `backend/app/schemas.py` — new `CutShapeSource`, `CutShapeRequest` models.
- Modify `backend/routes/segment.py` — new `POST /api/segment/cut-shape` route + a `_cut_shape_core` helper alongside the existing `_apply_shape_core`.
- Modify `backend/tests/test_segment_state.py` — `source` param coverage.
- Modify `backend/tests/test_sam_proxy.py` — assert `source='sam'` on `/project`-created candidates.
- Create `backend/tests/test_cut_shape.py` — new route's behavior.

**Frontend:**
- Modify `frontend/src/segment-state.js` — `samSegments` Map entries gain `source`; `applySamDelta` accepts/stores it.
- Modify `frontend/src/api.js` — new `cutShape(...)` client function.
- Modify `frontend/src/mode-label.jsx` — add right-click menu wiring for Presegment/SAM/Instance rows; add "Edit selection…" enable/disable logic; open/host the cut modal; handle `cut-shape` responses. (No `box-tool.js` extraction — see Task 8, superseded: `Viewer` is already the reusable unit.)
- Create `frontend/src/cut-mode.jsx` — the isolated modal: second `Viewer` instance (same `transformMode`/`onCuboidTransform` prop pattern `mode-label.jsx` already uses for its own gizmo), filtered/tagged cloud, cut-confirm flow.
- Create `frontend/src/context-menu.jsx` — small shared right-click menu component (one item for this feature: "Edit selection…").
- Modify `frontend/src/sam-segment-list.jsx` — `onContextMenu` per row.
- Modify `frontend/src/segment-tools.jsx` — `onContextMenu` per row (PresegmentList).
- Modify `frontend/src/segment-state.test.js` — `source`-tag coverage.
- Create `frontend/src/cut-mode.test.js` — partitioning/tagging pure functions.
- Modify `frontend/src/api.test.js` — `cutShape` request/response shape.

---

## Backend

### Task 1: `materialize_sam_segment` gains a required `source` tag

**Files:**
- Modify: `backend/labeling/segment_state.py:160-197`
- Test: `backend/tests/test_segment_state.py`

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_segment_state.py` near the existing `materialize_sam_segment` tests (around line 401):

```python
def test_materialize_sam_segment_requires_and_stores_source(seg_session):
    idx = np.array([0, 1, 2], dtype=np.int64)
    out = seg_session.materialize_sam_segment(idx, source="preseg")
    seg_id = out["sam_seg_id"]
    assert seg_session.sam_segments[seg_id]["source"] == "preseg"


def test_materialize_sam_segment_source_sam_default_behavior_unchanged(seg_session):
    idx = np.array([3, 4], dtype=np.int64)
    out = seg_session.materialize_sam_segment(idx, source="sam", mask_score=0.9)
    seg_id = out["sam_seg_id"]
    entry = seg_session.sam_segments[seg_id]
    assert entry["source"] == "sam"
    assert entry["mask_score"] == 0.9
```

(Match the existing fixture name used by neighboring tests in this file — check the top of `test_segment_state.py` for the actual fixture, e.g. `seg_session`, and reuse it verbatim rather than inventing a new one.)

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_segment_state.py -k source -v`
Expected: FAIL — `materialize_sam_segment() missing 1 required positional argument: 'source'` (or `TypeError` on the new kwarg once you attempt the call before implementing).

- [ ] **Step 3: Implement**

In `backend/labeling/segment_state.py`, change the signature at line 160 and the dict-building block at lines 189-193:

```python
def materialize_sam_segment(
    self, indices: np.ndarray,
    source: str,
    protect_instances: Optional[list[int]] = None,
    mask_score: Optional[float] = None,
) -> dict:
```

```python
self.sam_segments[sam_seg_id] = {
    "n_points": int(indices.size),
    "source": source,
    "mask_score": mask_score,
    "created_at": utc_now_iso(),
}
```

`source` is a plain required string (`"sam"` or `"preseg"`); no enum/validation needed at this layer — validation belongs at the Pydantic/route boundary (Task 5).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest backend/tests/test_segment_state.py -v`
Expected: all PASS, including the two new tests and every pre-existing `materialize_sam_segment` test (they'll now fail to compile/call unless you also update their call sites — see Step 5).

- [ ] **Step 5: Update existing call sites in the test file**

Every existing call to `seg_session.materialize_sam_segment(idx, ...)` in `test_segment_state.py` (lines ~401-462 per the earlier investigation: `test_materialize_sam_segment_allocates_fresh_id_and_writes_sam_ids`, `_ids_increment`, `_respects_protect_instances`, `_all_protected_creates_nothing`, `_overlap_is_last_write_wins`, `_full_overlap_drops_old_summary_entry`, and the autosave test at line 341/352) now needs `source="sam"` added as a positional or keyword arg (pick `source="sam"` — these all originated from SAM masks). Update each call site.

Run: `.venv/bin/pytest backend/tests/test_segment_state.py -v`
Expected: PASS, full file green.

- [ ] **Step 6: Commit**

```bash
git add backend/labeling/segment_state.py backend/tests/test_segment_state.py
git commit -m "feat(backend): tag sam_segments entries with their source (sam|preseg)"
```

### Task 2: Persist `source` through `segment_io.py`

**Files:**
- Modify: `backend/labeling/segment_io.py:275-291` (`sam_segments_to_list`/`save_sam_segments`/`load_sam_segments`)
- Test: find or create the existing round-trip test file covering `sam_segments.json` (search `backend/tests/` for `save_sam_segments`/`load_sam_segments` before creating a new file — likely already covered inside `test_segment_state.py`'s autosave test or a dedicated `test_segment_io.py`).

- [ ] **Step 1: Locate existing coverage**

Run: `grep -rn "save_sam_segments\|load_sam_segments\|sam_segments_to_list" backend/tests/`

Confirm which test file already exercises the round trip. Add your new assertions there rather than creating a duplicate test file.

- [ ] **Step 2: Write the failing test**

Extend the located test with an assertion that `source` survives a save/load round trip, e.g.:

```python
def test_sam_segments_round_trip_preserves_source(tmp_path, seg_session):
    idx = np.array([0, 1], dtype=np.int64)
    seg_session.materialize_sam_segment(idx, source="preseg")
    save_sam_segments(tmp_path, seg_session.sam_segments)
    loaded = load_sam_segments(tmp_path)
    assert loaded[next(iter(loaded))]["source"] == "preseg"
```

(Adapt to the actual return shape of `load_sam_segments` — check its real signature at `backend/labeling/segment_io.py:291` before finalizing; the investigation only confirmed it "reads it back" without the exact return type.)

- [ ] **Step 3: Run test to verify it fails or passes trivially**

Run: `.venv/bin/pytest <located_test_file> -k source -v`

Since `sam_segments_to_list`/`save_sam_segments`/`load_sam_segments` likely just serialize the dict as-is (no field allowlist), this may already pass once Task 1 lands — if so, this step confirms that rather than driving new production code. If it fails because of an explicit field allowlist somewhere in these functions, proceed to Step 4.

- [ ] **Step 4: Implement only if Step 3 failed**

Update `sam_segments_to_list`/`load_sam_segments` in `backend/labeling/segment_io.py` to include `source` in whatever explicit field list they build (per the investigation: `sam_segments_to_list` turns the dict into `[{id, **meta}, ...]`, which likely already spreads `source` automatically via `**meta` — verify before editing).

- [ ] **Step 5: Run full backend test suite for this file**

Run: `.venv/bin/pytest backend/tests/test_segment_state.py -v` (plus whichever file you extended)
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/labeling/segment_io.py backend/tests/
git commit -m "feat(backend): round-trip sam_segments source tag through session persistence"
```

### Task 3: `/api/sam/project` passes `source='sam'`

**Files:**
- Modify: `backend/routes/sam.py:90`
- Test: `backend/tests/test_sam_proxy.py`

- [ ] **Step 1: Write the failing test**

In `backend/tests/test_sam_proxy.py`, extend `test_capture_then_project` (line 22) or add a sibling test asserting the materialized segment's `source` is `"sam"`:

```python
def test_project_materializes_with_sam_source(...):
    # ... existing capture+project setup from test_capture_then_project ...
    resp = client.post("/api/sam/project", json={...})
    seg_id = resp.json()["segments"][0]["sam_seg_id"]
    assert seg.sam_segments[seg_id]["source"] == "sam"
```

Match whatever fixture/setup pattern `test_capture_then_project` already uses — read it first rather than inventing new mocking.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest backend/tests/test_sam_proxy.py -v`
Expected: FAIL — `materialize_sam_segment() missing 1 required positional argument: 'source'` (this route now breaks until updated, since Task 1 made `source` required).

- [ ] **Step 3: Implement**

In `backend/routes/sam.py:90`, change:

```python
out = seg.materialize_sam_segment(idx, protect_instances=req.protect_instances)
```
to:
```python
out = seg.materialize_sam_segment(idx, source="sam", protect_instances=req.protect_instances)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest backend/tests/test_sam_proxy.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/routes/sam.py backend/tests/test_sam_proxy.py
git commit -m "fix(backend): SAM project route tags materialized candidates source='sam'"
```

### Task 4: `CutShapeRequest`/response schemas

**Files:**
- Modify: `backend/app/schemas.py` (near `ApplyShapeRequest`, lines 164-171)

- [ ] **Step 1: Add the Pydantic models**

```python
class CutShapeSource(BaseModel):
    kind: Literal["preseg", "sam", "instance"]
    seg_id: int


class CutShapeRequest(BaseModel):
    shape: dict
    sources: list[CutShapeSource]
    protect_instances: list[int] = []
```

Match the existing import style at the top of `schemas.py` for `Literal` (check whether it's already imported from `typing`; add if missing).

No test for this step alone — it's exercised end-to-end in Task 5's tests.

- [ ] **Step 2: Commit**

```bash
git add backend/app/schemas.py
git commit -m "feat(backend): add CutShapeRequest/CutShapeSource schemas"
```

### Task 5: `POST /api/segment/cut-shape` route

**Files:**
- Modify: `backend/routes/segment.py` (add near `_apply_shape_core`, lines 93-146)
- Test: Create `backend/tests/test_cut_shape.py` (model structure on `backend/tests/test_apply_shape.py`)

This is the core of the backend work. Read `backend/routes/segment.py:93-146` (`_apply_shape_core` + `/apply-shape`) and `backend/labeling/shapes.py::shape_indices` in full before writing this task's code — the plan describes behavior, not a verbatim diff, and exact helper signatures must be confirmed against the live file.

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_cut_shape.py`, modeled on `backend/tests/test_apply_shape.py`'s structure (fixtures, client setup — copy its imports/setup pattern). Cover, at minimum:

```python
def test_cut_shape_partitions_two_presegments(client, seg_session_with_two_presegs):
    # OBB spanning points from preseg A and preseg B
    resp = client.post("/api/segment/cut-shape", json={
        "shape": {"type": "obb", "center": [...], "size": [...], "rotation": [0, 0, 0]},
        "sources": [{"kind": "preseg", "seg_id": A_ID}, {"kind": "preseg", "seg_id": B_ID}],
    })
    body = resp.json()
    assert len(body["materialized"]) == 2
    sources = {m["source"] for m in body["materialized"]}
    assert sources == {"preseg"}
    # each materialized entry's n_points matches full-res intersection, not render-cloud count


def test_cut_shape_instance_source_inherits_class_and_forwards_protect_instances(client, seg_session_with_instance):
    resp = client.post("/api/segment/cut-shape", json={
        "shape": {...},
        "sources": [{"kind": "instance", "seg_id": INST_ID}],
        "protect_instances": [OTHER_CONFIRMED_INST_ID],
    })
    body = resp.json()
    assert body["instance"]["inst_id"] != INST_ID  # fresh id
    # assert the new instance's class_ids match the source instance's class server-side


def test_cut_shape_empty_source_partition_produces_no_entry(client, seg_session_with_two_presegs):
    # OBB only covering preseg A's points
    resp = client.post("/api/segment/cut-shape", json={
        "shape": {...far from B...},
        "sources": [{"kind": "preseg", "seg_id": A_ID}, {"kind": "preseg", "seg_id": B_ID}],
    })
    body = resp.json()
    assert len(body["materialized"]) == 1


def test_cut_shape_unknown_source_kind_400(client, seg_session_with_two_presegs):
    resp = client.post("/api/segment/cut-shape", json={
        "shape": {...},
        "sources": [{"kind": "bogus", "seg_id": 1}],
    })
    assert resp.status_code == 422  # Pydantic Literal validation, not a hand-rolled 400
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_cut_shape.py -v`
Expected: FAIL — `404 Not Found` (route doesn't exist yet).

- [ ] **Step 3: Implement `_cut_shape_core` and the route**

In `backend/routes/segment.py`, add alongside `_apply_shape_core`:

```python
@router.post("/api/segment/cut-shape")
def cut_shape(req: CutShapeRequest):
    seg = _require_session()  # match whatever session-fetch helper apply-shape uses
    positions = np.asarray(seg.positions)
    idx = shape_indices(positions, req.shape)  # full-res, same call apply-shape makes

    materialized = []
    instance_result = None
    total_protected = 0

    for src in req.sources:
        if src.kind == "preseg":
            membership = (seg.preseg_ids == src.seg_id)
        elif src.kind == "sam":
            membership = (seg.sam_ids == src.seg_id)
        else:  # "instance"
            membership = (seg.instance_ids == src.seg_id)

        partition = idx[membership[idx]]
        if partition.size == 0:
            continue

        if src.kind in ("preseg", "sam"):
            out = seg.materialize_sam_segment(
                partition, source=src.kind, protect_instances=req.protect_instances,
            )
            total_protected += out["n_protected"]
            materialized.append({
                "sam_seg_id": out["sam_seg_id"],
                "source": src.kind,
                "n_points": out["n_affected"],
            })
        else:
            source_class = int(seg.class_ids[np.where(membership)[0][0]])
            out = seg.apply_reassign(
                partition, target_inst=-1, target_class=source_class,
                protect_instances=req.protect_instances,
            )
            total_protected += out["n_protected"]
            instance_result = {"inst_id": out["indices"]... , "n_points": out["n_affected"]}
            # confirm the exact key apply_reassign's return dict uses for the fresh instance id
            # before finalizing — check _apply's return shape in segment_state.py

    return {"materialized": materialized, "instance": instance_result, "n_protected": total_protected}
```

Before finalizing, confirm against the live `segment_state.py`:
- The exact key name `_apply`/`apply_reassign` uses to report the newly-allocated instance id in its return dict (the plan's placeholder `out["indices"]...` above is a stand-in — replace it with the real field).
- Whether `shape_indices` needs `np.asarray(seg.positions)` or already accepts the raw type stored on `seg.positions`.
- The correct session-fetch helper name (`_require_session` above is a placeholder matching the convention other routes in this file use — confirm the actual helper name before writing).

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest backend/tests/test_cut_shape.py -v`
Expected: PASS

- [ ] **Step 5: Run the full backend suite**

Run: `.venv/bin/pytest backend/tests -v`
Expected: PASS, no regressions in `test_apply_shape.py`, `test_segment_state.py`, `test_sam_proxy.py`, `test_segment_endpoints.py`.

- [ ] **Step 6: Commit**

```bash
git add backend/routes/segment.py backend/tests/test_cut_shape.py
git commit -m "feat(backend): add POST /api/segment/cut-shape endpoint"
```

---

## Frontend

### Task 6: `segment-state.js` — `source` on `samSegments` entries

**Files:**
- Modify: `frontend/src/segment-state.js:1-30` (`initSegState`), `74-92` (`applySamDelta`)
- Test: `frontend/src/segment-state.test.js` (extend the "SAM candidate layer" describe block starting line 209)

- [ ] **Step 1: Write the failing test**

```js
test('applySamDelta stores the source tag on the samSegments entry', () => {
  const state = initSegState({ /* ...minimal fixture matching neighboring tests... */ });
  const next = applySamDelta(state, { indices: [0, 1], samSegId: 5, source: 'preseg' });
  expect(next.samSegments.get(5).source).toBe('preseg');
});
```

Match the exact fixture shape used by the existing tests immediately above/below line 231/240/251 in `segment-state.test.js` — copy their `initSegState(...)` call verbatim and adapt only the assertion.

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/segment-state.test.js -t "source tag"`
Expected: FAIL — `undefined` is not `'preseg'`.

- [ ] **Step 3: Implement**

In `frontend/src/segment-state.js`:
- Line ~26-27 (`initSegState`'s `samSegments` construction): thread a `source` field from the incoming `samSegments` list items (`s.source`) into the Map value alongside `nPoints`/`maskScore`.
- `applySamDelta` signature (line 74): add `source` to its destructured options, and line 90's `samSegments.set(...)` call to include `source`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/segment-state.test.js`
Expected: PASS, no regressions.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/segment-state.js frontend/src/segment-state.test.js
git commit -m "feat(frontend): thread source tag through samSegments state"
```

### Task 7: `api.js` — `cutShape` client function

**Files:**
- Modify: `frontend/src/api.js` (near `applyShape`, lines 248-264)
- Test: `frontend/src/api.test.js`

- [ ] **Step 1: Write the failing test**

Add near wherever `api.test.js` would naturally test a POST-based client function (check if `applyShape`/`segApply` have existing tests first — the investigation found none visible in the grepped range; if so, this is the first such test and should establish the pattern, using `fetch` mocking consistent with the rest of the file):

```js
test('cutShape posts shape+sources and decodes materialized/instance response', async () => {
  global.fetch = vi.fn().mockResolvedValue({
    ok: true,
    json: async () => ({
      materialized: [{ sam_seg_id: 7, source: 'preseg', n_points: 12 }],
      instance: null,
      n_protected: 0,
    }),
  });
  const result = await VoxaAPI.cutShape({
    shape: { type: 'obb', center: [0,0,0], size: [1,1,1], rotation: [0,0,0] },
    sources: [{ kind: 'preseg', segId: 3 }],
  });
  expect(result.materialized[0]).toEqual({ samSegId: 7, source: 'preseg', nPoints: 12 });
  expect(result.instance).toBeNull();
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd frontend && npx vitest run src/api.test.js -t cutShape`
Expected: FAIL — `VoxaAPI.cutShape is not a function`.

- [ ] **Step 3: Implement**

In `frontend/src/api.js`, add a new function mirroring `applyShape`'s shape (lines 248-264):

```js
async cutShape({ shape, sources, protectInstances = [] }) {
  const res = await fetch('/api/segment/cut-shape', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      shape,
      sources: sources.map((s) => ({ kind: s.kind, seg_id: s.segId })),
      protect_instances: protectInstances,
    }),
  });
  const j = await res.json();
  return {
    materialized: (j.materialized || []).map((m) => ({
      samSegId: m.sam_seg_id, source: m.source, nPoints: m.n_points,
    })),
    instance: j.instance ? { instId: j.instance.inst_id, nPoints: j.instance.n_points } : null,
    nProtected: j.n_protected,
  };
},
```

Match the exact surrounding object/class shape `applyShape` lives in (confirm whether `VoxaAPI` is a plain object literal or class before finalizing the snippet's indentation/context).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/api.test.js`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api.js frontend/src/api.test.js
git commit -m "feat(frontend): add VoxaAPI.cutShape client function"
```

### Task 8: SUPERSEDED — no box-transform extraction needed

**Original plan:** extract Box tool draw/transform logic into a new `frontend/src/box-tool.js` module, on the assumption that G/R/Y transform math was pure, hand-rolled logic living inline in `mode-label.jsx` that could be moved verbatim.

**What research found (implementer correctly escalated rather than forcing a bad extraction):** that assumption was wrong. `mode-label.jsx`'s G/R/Y handlers (around line 1057-1065) are three one-line `setTransformMode('translate'|'rotate'|'scale')` calls — no math at all. The actual box-transform math is owned by Three.js's own `TransformControls` gizmo, instantiated and driven entirely inside `frontend/src/viewer.jsx` (`new TransformControls(camera, renderer.domElement)`, an `objectChange` handler reading the live `Object3D` anchor's position/rotation/scale and reporting a `{center, size, rotation}` patch up through an `onCuboidTransform` callback prop). `mode-label.jsx`'s own `onCuboidTransform` is just a two-line dispatcher that merges whatever patch the gizmo already computed into `selBox` state — there is no freestanding pure function anywhere to extract. Hand-rolling one in a new `box-tool.js` would not be an extraction — it would be an untested reimplementation the app doesn't actually call, risking silent divergence from Three's real transform semantics (axis order, gizmo-space vs. world-space composition, snapping).

**Resolution:** no extraction needed. `Viewer` (`frontend/src/viewer.jsx`) is already the reusable unit — it's a plain component that accepts `transformMode`/`onCuboidTransform` as props and owns its own gizmo internally. Task 11's isolated modal reuses the Box tool by mounting a **second `Viewer` instance** with the same `transformMode`/`onCuboidTransform` prop pattern `mode-label.jsx` already uses, plus its own small local G/R/Y key-dispatch (the same trivial three-line pattern, duplicated rather than shared — not worth a module for three lines). See Task 11's updated description below.

No files changed, no commit for this task — it is a no-op given what Task 11 actually needs.

### Task 9: `context-menu.jsx` — shared right-click menu component

**Files:**
- Create: `frontend/src/context-menu.jsx`
- Test: Create `frontend/src/context-menu.test.js` (only if the project has any existing component-level tests to model against — check `frontend/vite.config.js`'s vitest `environment` setting per CLAUDE.md, which notes the current test setup is "pure-function only." If no `jsdom`/`@testing-library/react` is configured, add them as dev dependencies first, matching CLAUDE.md's guidance: "Add `jsdom` + `@testing-library/react` if you start writing component tests.")

- [ ] **Step 1: Check current test environment**

Run: `grep -n "environment" frontend/vite.config.js`

If it's `'node'` (per CLAUDE.md), this is the first component test in the repo — install `jsdom` and `@testing-library/react` as dev dependencies (`cd frontend && npm install -D jsdom @testing-library/react`) and add a `test.environment: 'jsdom'` override scoped to this test file only (Vitest supports per-file environment via a `// @vitest-environment jsdom` comment at the top of the test file — prefer that over a global config change, to avoid slowing down every other test).

- [ ] **Step 2: Write the failing test**

```jsx
// @vitest-environment jsdom
import { render, screen, fireEvent } from '@testing-library/react';
import { ContextMenu } from './context-menu.jsx';

test('renders items and calls onSelect, closes on outside click', () => {
  const onSelect = vi.fn();
  const onClose = vi.fn();
  render(
    <ContextMenu x={10} y={10} items={[{ label: 'Edit selection…', onSelect, disabled: false }]} onClose={onClose} />
  );
  fireEvent.click(screen.getByText('Edit selection…'));
  expect(onSelect).toHaveBeenCalled();
});

test('disabled item does not call onSelect', () => {
  const onSelect = vi.fn();
  render(
    <ContextMenu x={10} y={10} items={[{ label: 'Edit selection…', onSelect, disabled: true }]} onClose={() => {}} />
  );
  fireEvent.click(screen.getByText('Edit selection…'));
  expect(onSelect).not.toHaveBeenCalled();
});
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/context-menu.test.js`
Expected: FAIL — module doesn't exist.

- [ ] **Step 4: Implement**

```jsx
export function ContextMenu({ x, y, items, onClose }) {
  return (
    <div
      style={{ position: 'fixed', left: x, top: y, zIndex: 1000 }}
      className="context-menu"
      onMouseLeave={onClose}
    >
      {items.map((item) => (
        <div
          key={item.label}
          className={`context-menu-item${item.disabled ? ' disabled' : ''}`}
          onClick={() => { if (!item.disabled) { item.onSelect(); onClose(); } }}
        >
          {item.label}
        </div>
      ))}
    </div>
  );
}
```

Match existing CSS conventions in the repo (check `frontend/src/*.css` for how other small floating UI, e.g. `ClassPickerModal`, is styled, and follow that pattern rather than inventing new class names ad hoc).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/context-menu.test.js`
Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add frontend/src/context-menu.jsx frontend/src/context-menu.test.js frontend/package.json frontend/package-lock.json
git commit -m "feat(frontend): add shared ContextMenu component"
```

### Task 10: Wire right-click "Edit selection…" into the three list surfaces

**Files:**
- Modify: `frontend/src/sam-segment-list.jsx` (row div at line 46)
- Modify: `frontend/src/segment-tools.jsx` (row div at line 113)
- Modify: `frontend/src/mode-label.jsx` (Instances panel row — locate via `grep -n selectedId frontend/src/mode-label.jsx` first; the earlier investigation did not pin an exact line for the instance row's own render, only for `selectedId` state)
- Test: extend `frontend/src/sam-segment-list.test.js` (currently only covers `toggleSamSelection`) and add an equivalent for `segment-tools.jsx` if it has a test file (check `frontend/src/segment-tools.test.js` exists before assuming; create if absent, matching neighboring row-click test patterns).

- [ ] **Step 1: Write failing tests for enable/disable logic**

The enable/disable rules live in the spec's "Trigger" section. Write pure-function tests first for whatever function decides "can Edit selection… fire for this selection state" — e.g. in a new small module `frontend/src/cut-eligibility.js`:

```js
test('same-list preseg multi-select is eligible', () => {
  expect(cutEligibility({ list: 'preseg', selection: new Set([1,2]) })).toEqual({ eligible: true });
});

test('single unconfirmed instance is eligible', () => {
  expect(cutEligibility({ list: 'instance', selectedId: 5, confirmed: false })).toEqual({ eligible: true });
});

test('confirmed instance is not eligible', () => {
  expect(cutEligibility({ list: 'instance', selectedId: 5, confirmed: true })).toEqual({ eligible: false, reason: 'confirmed' });
});

test('empty selection is not eligible', () => {
  expect(cutEligibility({ list: 'preseg', selection: new Set() })).toEqual({ eligible: false, reason: 'empty' });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/cut-eligibility.test.js`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `cut-eligibility.js`**

Pure function encoding exactly the rules from the spec's "Trigger" section (same-list multi-select for preseg/SAM, single-unconfirmed-instance for Instances, no viewport trigger). No React, no API calls — a plain function so it's trivially testable and reusable from all three row components.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/cut-eligibility.test.js`
Expected: PASS

- [ ] **Step 5: Wire into the three row components**

Add `onContextMenu={(e) => { e.preventDefault(); openCutMenu(e, seg.id); }}` to the row divs in `sam-segment-list.jsx:46` and `segment-tools.jsx:113`, and the equivalent instance row in `mode-label.jsx`. `openCutMenu` (new local state/handler in the parent that renders `ContextMenu` at the click coordinates) uses `cut-eligibility.js` to decide whether the single "Edit selection…" item is disabled, and its `onSelect` opens the Task 11 modal.

- [ ] **Step 6: Run the full frontend test suite**

Run: `cd frontend && npx vitest run`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add frontend/src/cut-eligibility.js frontend/src/cut-eligibility.test.js frontend/src/sam-segment-list.jsx frontend/src/segment-tools.jsx frontend/src/mode-label.jsx
git commit -m "feat(frontend): wire right-click Edit selection menu into list rows"
```

### Task 11: `cut-mode.jsx` — isolated modal, filtered cloud, source tagging

**Files:**
- Create: `frontend/src/cut-mode.jsx`
- Test: Create `frontend/src/cut-mode.test.js` (pure filtering/partitioning logic only — the modal's Three.js mount itself is not unit-testable the way `Viewer` already isn't elsewhere in this codebase; follow existing precedent of testing pure helpers, not the mounted component).

- [ ] **Step 1: Write failing tests for the pure cloud-filtering function**

```js
test('buildCutCloud filters points to the selected sources and tags each with its origin', () => {
  const segState = {
    instanceFull: new Int32Array([1, -1, 2]),
    samIds: new Int32Array([-1, 7, -1]),
    /* ...minimal positions/colors fixture... */
  };
  const result = buildCutCloud(segState, { sources: [{ kind: 'sam', segId: 7 }] });
  expect(result.indices).toEqual([1]);
  expect(result.tags[0]).toEqual({ kind: 'sam', segId: 7 });
});
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd frontend && npx vitest run src/cut-mode.test.js`
Expected: FAIL — module doesn't exist.

- [ ] **Step 3: Implement `buildCutCloud` and the `CutModal` component**

`buildCutCloud(segState, { sources })`: for each source, find matching full-array indices (`instanceFull === segId` for instance/preseg — check whether presegment membership is actually tracked in `instanceFull` or a separate array before finalizing; the investigation flagged `preseg_ids`-equivalent as a frontend concept without pinning its exact `segState` field name, so confirm against `segment-state.js` directly), union them, and return `{indices, tags}` where `tags[i]` records which source each returned point came from.

`CutModal({ segState, sources, onClose, onCutConfirmed })`: mounts a second `Viewer` fed `{positions: filtered, colors: filtered}` (using `buildCutCloud`'s indices to slice `segState`'s full arrays), reuses the Box tool's draw/transform interaction by mounting `Viewer` with the same `transformMode`/`onCuboidTransform` prop pattern `mode-label.jsx` already uses for its own gizmo (own local `selBox`-equivalent state + a small local G/R/Y key-dispatch, duplicated rather than shared — see Task 8, superseded), and on "confirm cut" calls `VoxaAPI.cutShape({shape, sources, protectInstances})` (Task 7), then removes the cut points from its own local filtered view (so the next box draws against the remainder) and calls `onCutConfirmed(response)` so the parent (`mode-label.jsx`) can patch `segState`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd frontend && npx vitest run src/cut-mode.test.js`
Expected: PASS

- [ ] **Step 5: Wire `CutModal` into `mode-label.jsx`**

Add modal-open state (`cutModalSources`, set by Task 10's `openCutMenu`/menu-item `onSelect`), render `<CutModal .../>` conditionally, and on `onCutConfirmed`, apply the response to `segState`:
- Each `materialized` entry → `applySamDelta(segState, {indices: <server needs to also return indices, or the frontend re-derives via a follow-up state fetch>, samSegId, source})`. **Flag for implementer:** the `cut-shape` response in Task 5 does not currently include per-materialized-entry point indices (only counts) — check whether `applySamDelta` truly needs indices to patch state correctly (it does, per Task 6/existing usage) and if so, add `scan_indices_b64` to the `cut-shape` response in Task 5 (mirroring how `/api/sam/project`'s response already includes `scan_indices_b64` per the SAM spec) before this step, rather than leaving it as a silent gap.
- The `instance` entry (if present) → same `applyDelta`-shaped patch `confirmSamSelection`/`confirmSegmentSelection` already perform after a successful apply (mode-label.jsx:703-809), including pushing a new `instances` row.

- [ ] **Step 6: Run the full frontend test suite**

Run: `cd frontend && npx vitest run`
Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add frontend/src/cut-mode.jsx frontend/src/cut-mode.test.js frontend/src/mode-label.jsx backend/routes/segment.py backend/app/schemas.py backend/labeling/segment_state.py
git commit -m "feat(frontend): isolated cut modal wired to cut-shape endpoint"
```

*(Note: this task's Step 5 may require reopening Task 5's backend work to add indices to the response — if so, add a corresponding backend test update there and note the amendment in the commit message.)*

### Task 12: End-to-end browser verification

**Files:** none (verification only)

- [ ] **Step 1: Start the dev server**

Run: `npm run dev` (from repo root)

- [ ] **Step 2: Use the `browser-verification` skill**

Invoke the `browser-verification` skill (per this repo's CLAUDE.md/global instructions: UI changes must be verified in a browser, not just "tests pass"). On a throwaway session (per the `feedback_browser_verify_mutates_session` memory — voxa's Label apply auto-saves to disk):

1. Multi-select two presegments in the Presegment list, right-click one, confirm "Edit selection…" is enabled, click it.
2. In the modal, draw a box spanning points from both presegments, confirm the cut.
3. Confirm two new rows appear in the Presegment list (point-recolored, no hull), with correct point counts; confirm the two original presegments' displayed hulls/counts are unchanged (per spec, this is expected).
4. Close the modal. Select a single unconfirmed instance, right-click, confirm "Edit selection…" is enabled, cut a sub-region.
5. Confirm the split-off piece appears as a new unconfirmed instance with the same class as the source, with no class-picker prompt.
6. Confirm a confirmed instance, verify "Edit selection…" is disabled/absent on its row.
7. Screenshot each key state. Check the browser console for zero errors and confirm `/api/segment/cut-shape` network calls return 200 with the expected `materialized`/`instance` shape.

- [ ] **Step 3: Report findings**

If any step fails, treat it as a bug against the relevant task above (fix there, don't patch around it here) — do not mark this task complete until all steps pass live.

- [ ] **Step 4: Commit any fixes discovered during verification**

```bash
git add -A
git commit -m "fix(frontend|backend): <specific fix found during browser verification>"
```

---

## Docs

### Task 13: Update CLAUDE.md

**Files:**
- Modify: `/home/hendrik/coding/engine/tools/labeling/voxa/CLAUDE.md`

Per this repo's own CLAUDE.md workflow rule ("Docs ship with the code... goes in the same PR as the implementation"), add a short paragraph to the Label-mode tool list (alongside the existing Presegment/Box/Draw/Beam/SAM bullets) describing the cut-selection feature: right-click "Edit selection…" on Presegment/SAM/Instance rows, the isolated modal, source-partitioned results, and the generalized `sam_segments.source` tag — cross-reference `docs/superpowers/specs/2026-07-14-cut-selection-tool-design.md`.

- [ ] **Step 1: Write the CLAUDE.md addition** (a few sentences, matching the density/style of the existing SAM paragraph)
- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: document the cut-selection tool in CLAUDE.md"
```

---

## Notes for the implementing agent

- Tasks 1-5 (backend) are fully TDD-specified and should be low-ambiguity.
- Tasks 8, 10, 11 involve real codebase archaeology (`grep`/`Read` before writing) — the plan flags every place an exact line number or field name wasn't confirmed by the research pass. Do not guess; read the file first.
- Task 11 Step 5 has a known open gap (indices missing from the `cut-shape` response) — resolve it by amending Task 5, not by working around it client-side.
- Run `npm test` (both suites) after every task, not just the task's own suite, to catch cross-cutting regressions early.
