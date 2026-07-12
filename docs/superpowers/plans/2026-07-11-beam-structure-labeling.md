# Beam / Structure Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the Beam tool — a node/edge graph over steel structures where each edge is a square-section box applied as one pointset instance — per `docs/superpowers/specs/2026-07-10-beam-structure-labeling-design.md`, reconciled with the shipped unified-label-tools and export-wizard architecture.

**Architecture:** Beam becomes the 4th entry in the Label-mode tool rail (`label-tools.js` / `tool-rail.jsx` / `tool-options.jsx`). The frontend owns the graph UX (`beam-graph.js` pure state machine + `beam-mode.jsx` component, mirroring `draw-paths.js` / `draw-mode.jsx`). **Each beam converts to an OBB on the frontend** (center = midpoint, size = `[|b-a|, width, width]`, rotation = Euler XYZ from the beam frame) and applies through the existing `POST /api/segment/apply-shape` `{type:'obb'}` — one call per beam, sequential, exactly like Draw loops `centerlineApply`. Applied beams surface as unconfirmed pointset rows that **persist their OBB** (like Box rows), so `materialize.py::collect_volumes` needs only a 1-line change to make beams replay exactly into raw-density exports. The backend gains only `structure.json` persistence (`backend/labeling/beams.py` + `GET`/`PUT /api/segment/structure`).

**Tech Stack:** FastAPI + Pydantic + numpy (backend), React 18 + Three.js (frontend), pytest + vitest.

---

## Reconciliation with the spec (deviations, all reviewed against shipped code)

The spec was written 2026-07-10, the same day the unified-label-tools spec was written; unified tools have since **shipped**. These deviations reconcile the spec with what exists now:

1. **No `POST /api/segment/beam-apply`, no `beams.py::box_indices`.** The spec predates the generic `apply-shape` endpoint. A beam's swept square box *is* an OBB, and `labeling/shapes.py::obb_indices` already does exact oriented-box extraction with the codebase-wide Rx·Ry·Rz Euler convention (CLAUDE.md "Coordinate system"; don't hand-roll a new containment test). The frontend computes `{center, size, rotation}` per edge and calls `applyShape({type:'obb'})` per beam — sequential calls in a loop, the exact pattern `draw-mode.jsx::applySelection` already uses. `target_inst` re-apply, `n_affected == 0` handling, undo, and the `instance_id` response field all come for free.
2. **Beam pointset rows persist their OBB** (`center/size/rotation` on the row, like `source:'box'` rows do per the resolution-independent-labels spec). `collect_volumes` in `materialize.py` extends its source gate from `('box','draw')` to `('box','beam','draw')` → beams replay **exactly** (not NN-approximate) into raw exports. This matters: confirmed-label accuracy through export is a stated user priority. A re-apply after a joint drag must overwrite the stored OBB on the existing row.
3. **`backend/labeling/beams.py` holds persistence only** (`load_structure` / `save_structure` for `sessions/<id>/structure.json`), mirroring `centerline.py`'s load/update half.
4. **`PUT /api/segment/structure` carries a `session_id` pin.** Geometry writes are client-driven and debounced (800 ms), so a session switch mid-debounce could write the old graph into the new session's file. The PUT body carries the session id it was built for; the backend 409s on mismatch instead of writing to the wrong session.
5. **Per-edge `dirty` flag** (persisted in `structure.json`): Enter applies only never-applied or edited-since-apply edges, so re-pressing Enter after building 10 beams and dragging one joint sends 1 call, not 10 (and pushes 1 undo entry, not 10). Spec's "apply the whole active graph" is preserved semantically — the skipped edges are already applied and unchanged.
6. **The `mode-label.jsx` row-creation helper the spec asks to parameterize already takes auto-confirm into account** (`onDrawApplied` + `autoConfirmFor(tool)` shipped with unified tools). We generalize it to `onToolApplied({instanceId, classId, mergedFrom, source, obb})` — Draw call sites unchanged (`source` defaults `'draw'`), Beam passes `source:'beam'` + the OBB.
7. **Commit keeps failed beams active.** A beam whose apply returned `n_affected: 0` (or errored) has no instance; `commitAll` retires only applied edges, the rest stay in the active graph with a toast.

Everything else follows the spec: Ctrl-click graph building with plain-mouse camera, node snap via raycast pick priority (node sphere > beam box > cloud), scroll-resize on selected beam, Enter apply / Ctrl+Enter commit with confirm dialog, committed layer behind a ◌/● toggle, `structure.json` not on the undo stack.

**Cleanup opportunities noted, deliberately NOT folded in** (each would touch draw/fast-label code outside this feature's scope — candidates for a separate refactor): a shared capture-phase key-driver component (`FastLabelKeys` / `DrawKeys` / `BeamKeys` are now three copies of one trick), a shared `useToast` hook, and a shared bottom-HUD shell (Draw's and Beam's HUDs duplicate the same fixed-position style block).

**Relevant skills:** superpowers:test-driven-development (every task), superpowers:verification-before-completion (Tasks 10–11), browser-verification (Task 10 — **use a throwaway session**: Label apply auto-saves to disk; restart any stale :8765 backend first).

---

### Task 0: Branch setup

**Files:** none (git only)

- [ ] **Step 0.1:** `git status` (must be clean), then:

```bash
git fetch origin
git checkout main && git pull
git checkout -b feat/beam-tool
git add docs/superpowers/plans/2026-07-11-beam-structure-labeling.md
git commit -m "docs(beam): implementation plan"
```

PR #26 (`fix/data-label-integrity`) is already merged into `origin/main`; the new branch must include it (beam rows depend on its seq/undo-reconciliation work). The plan file currently sits untracked in the working tree — carry it onto the branch with the commit above (untracked files survive the `checkout`).

---

### Task 1: Backend — `structure.json` persistence module

**Files:**
- Create: `backend/labeling/beams.py`
- Test: `backend/tests/test_beam_structure.py`

- [ ] **Step 1.1: Write the failing tests**

```python
"""Tests for labeling/beams.py — structure.json persistence."""
from __future__ import annotations

import json

import pytest

from labeling.beams import STRUCTURE_FILENAME, load_structure, save_structure


DOC = {
    "nodes": [{"id": 1, "pos": [0.0, 0.0, 0.0]}, {"id": 2, "pos": [2.0, 0.0, 0.0]}],
    "edges": [{"id": 3, "a": 1, "b": 2, "width": 0.2, "class_id": 10,
               "instance_id": None, "dirty": True}],
    "committed_beams": [{"a": [0.0, 0.0, 0.0], "b": [0.0, 2.0, 0.0], "width": 0.3,
                         "class_id": 11, "instance_id": 42}],
}


def test_load_missing_returns_empty(tmp_path):
    assert load_structure(tmp_path) == {"nodes": [], "edges": [], "committed_beams": []}


def test_save_load_round_trip(tmp_path):
    save_structure(tmp_path, DOC)
    assert (tmp_path / STRUCTURE_FILENAME).exists()
    assert load_structure(tmp_path) == DOC


def test_load_malformed_raises(tmp_path):
    (tmp_path / STRUCTURE_FILENAME).write_text(json.dumps({"nodes": []}))
    with pytest.raises(ValueError, match="malformed structure.json"):
        load_structure(tmp_path)
```

- [ ] **Step 1.2:** Run: `.venv/bin/pytest backend/tests/test_beam_structure.py -v` — expect FAIL (`ModuleNotFoundError: labeling.beams`).

- [ ] **Step 1.3: Implement `backend/labeling/beams.py`**

```python
"""Per-session Beam-tool structure persistence.

`structure.json` stores the node/edge graph (active, resumable) plus committed
beams (baked geometry for the read-only committed layer). Point extraction
needs no beam-specific backend code: the frontend converts each beam to an
OBB and applies it through /api/segment/apply-shape (see the beam spec's
as-built notes). Mirrors centerline.py's persistence half.
"""
from __future__ import annotations

import json
from pathlib import Path

STRUCTURE_FILENAME = "structure.json"

_KEYS = {"nodes", "edges", "committed_beams"}


def load_structure(session_dir: Path) -> dict:
    f = Path(session_dir) / STRUCTURE_FILENAME
    if not f.exists():
        return {"nodes": [], "edges": [], "committed_beams": []}
    data = json.loads(f.read_text())
    missing = _KEYS - data.keys()
    if missing:
        raise ValueError(
            f"malformed structure.json in {session_dir}: missing {sorted(missing)}"
        )
    return data


def save_structure(session_dir: Path, doc: dict) -> dict:
    from labeling.segment_io import atomic_write_json
    atomic_write_json(Path(session_dir) / STRUCTURE_FILENAME, doc)
    return doc
```

- [ ] **Step 1.4:** Run: `.venv/bin/pytest backend/tests/test_beam_structure.py -v` — expect 3 PASS.

- [ ] **Step 1.5: Commit**

```bash
git add backend/labeling/beams.py backend/tests/test_beam_structure.py
git commit -m "feat(beam): structure.json persistence module"
```

---

### Task 2: Backend — schemas + `GET`/`PUT /api/segment/structure`

**Files:**
- Modify: `backend/app/schemas.py` (append before `__all__`)
- Modify: `backend/routes/segment.py` (after `get_centerlines`, ~line 160)
- Test: `backend/tests/test_beam_structure_endpoints.py`

- [ ] **Step 2.1: Write the failing route tests** (fixtures `client`, `client_with_loaded_annotated_scene`, `scan_dir_for_loaded_scene` already exist in `conftest.py` — see `test_centerline_endpoints.py` for usage)

```python
"""Tests for GET/PUT /api/segment/structure (Beam tool persistence)."""
from __future__ import annotations

import json

EMPTY = {"nodes": [], "edges": [], "committed_beams": []}

DOC = {
    "nodes": [{"id": 1, "pos": [0.0, 0.0, 0.0]}, {"id": 2, "pos": [2.0, 0.0, 0.0]}],
    "edges": [{"id": 3, "a": 1, "b": 2, "width": 0.2, "class_id": 10,
               "instance_id": None, "dirty": True}],
    "committed_beams": [{"a": [0.0, 0.0, 0.0], "b": [0.0, 2.0, 0.0], "width": 0.3,
                         "class_id": 11, "instance_id": 42}],
}


def test_structure_empty_then_populated(
        client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    assert client.get("/api/segment/structure").json() == EMPTY
    r = client.put("/api/segment/structure", json=DOC)
    assert r.status_code == 200
    assert client.get("/api/segment/structure").json() == DOC
    # Written to the active session's structure.json on disk.
    sessions = list((scan_dir_for_loaded_scene / "sessions").iterdir())
    docs = [d / "structure.json" for d in sessions if (d / "structure.json").exists()]
    assert len(docs) == 1
    assert json.loads(docs[0].read_text()) == DOC


def test_structure_put_replaces_wholesale(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    client.put("/api/segment/structure", json=DOC)
    client.put("/api/segment/structure", json=EMPTY)
    assert client.get("/api/segment/structure").json() == EMPTY


def test_structure_409_without_session(client):
    assert client.get("/api/segment/structure").status_code in (400, 409)
    assert client.put("/api/segment/structure", json=EMPTY).status_code in (400, 409)


def test_structure_put_validation(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    bad_width = {**EMPTY, "edges": [{"id": 1, "a": 1, "b": 2, "width": 0.0, "class_id": 10}]}
    assert client.put("/api/segment/structure", json=bad_width).status_code == 422
    bad_pos = {**EMPTY, "nodes": [{"id": 1, "pos": [0.0, 0.0]}]}
    assert client.put("/api/segment/structure", json=bad_pos).status_code == 422


def test_structure_put_session_pin_mismatch(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.put("/api/segment/structure",
                   json={**DOC, "session_id": "not-the-active-session"})
    assert r.status_code == 409
```

- [ ] **Step 2.2:** Run: `.venv/bin/pytest backend/tests/test_beam_structure_endpoints.py -v` — expect FAIL (404 / validation).

- [ ] **Step 2.3: Add schemas** to `backend/app/schemas.py`, after `RenameSessionRequest` (mirror `CenterlinePath`'s len-3 validator style):

```python
class StructureNode(BaseModel):
    id: int
    pos: list[float]

    @field_validator("pos")
    @classmethod
    def _pos_is_3d(cls, v):
        if len(v) != 3:
            raise ValueError("pos must be [x, y, z]")
        return v

class StructureEdge(BaseModel):
    id: int
    a: int
    b: int
    width: float = Field(gt=0)
    class_id: int
    instance_id: Optional[int] = None
    dirty: bool = False   # edited since last apply (frontend re-apply bookkeeping)

class CommittedBeam(BaseModel):
    a: list[float]
    b: list[float]
    width: float = Field(gt=0)
    class_id: int
    instance_id: int

    @field_validator("a", "b")
    @classmethod
    def _endpoint_is_3d(cls, v):
        if len(v) != 3:
            raise ValueError("endpoint must be [x, y, z]")
        return v

class StructureDoc(BaseModel):
    # Written by the frontend after apply/commit/edits, debounced — session_id
    # pins the write to the session the graph was built in so a session switch
    # mid-debounce can't land the old graph in the new session's file.
    session_id: Optional[str] = None
    nodes: list[StructureNode] = []
    edges: list[StructureEdge] = []
    committed_beams: list[CommittedBeam] = []
```

- [ ] **Step 2.4: Add routes** to `backend/routes/segment.py` after `get_centerlines`. The active-session-with-dir guard already exists twice in this file (`centerline_apply`, `get_centerlines`) and these routes would make it four — extract it once and refactor `get_centerlines` to use it (same file, 2-line change):

```python
def _require_session_seg():
    """Active SegmentSession that has a session dir (409 otherwise) — shared
    by the centerline/structure persistence routes."""
    seg = _require_seg()
    if seg.session_dir is None:
        raise HTTPException(409, "no active session")
    return seg

@router.get("/api/segment/structure")
def get_structure():
    """Stored Beam-tool graph for the active session (Beam sub-mode resume)."""
    from labeling.beams import load_structure
    return load_structure(_require_session_seg().session_dir)

@router.put("/api/segment/structure")
def put_structure(doc: StructureDoc):
    """Replace the stored Beam-tool graph wholesale. The frontend owns graph
    geometry; point labels flow through apply-shape separately (not undoable
    here, matching centerlines.json)."""
    from labeling.beams import save_structure
    seg = _require_session_seg()
    if doc.session_id is not None and doc.session_id != _state.get("session_id"):
        raise HTTPException(
            409, f"session mismatch — server has '{_state.get('session_id')}', "
                 f"write was for '{doc.session_id}'")
    return save_structure(seg.session_dir, doc.model_dump(exclude={"session_id"}))
```

In `get_centerlines`, replace its two guard lines with `seg = _require_session_seg()` (behavior identical; its existing tests stay green).

- [ ] **Step 2.5:** Run: `.venv/bin/pytest backend/tests/test_beam_structure_endpoints.py backend/tests/test_beam_structure.py -v` — expect all PASS. Note: the GET response echoes exactly what was PUT (`model_dump` of the doc minus `session_id`), so `== DOC` holds — if a test fails on a missing `dirty` key, the PUT round-trips defaults in, which is fine: adjust `DOC` to always carry explicit `dirty`.

- [ ] **Step 2.6: Commit**

```bash
git add backend/app/schemas.py backend/routes/segment.py backend/tests/test_beam_structure_endpoints.py
git commit -m "feat(beam): GET/PUT /api/segment/structure with session pin"
```

---

### Task 3: Backend — beams replay exactly in exports (`collect_volumes`)

**Files:**
- Modify: `backend/labeling/materialize.py:25-47` (`collect_volumes`)
- Modify: `backend/app/schemas.py:95` (source comment)
- Test: `backend/tests/test_materialize.py` (extend)

- [ ] **Step 3.1: Write the failing test** (append to `test_materialize.py`, next to `test_collect_volumes_box_and_tube`):

```python
def test_collect_volumes_beam_source_is_obb():
    instances = [
        {"source": "beam", "segId": 7, "seq": 3,
         "center": [1.0, 0.0, 0.0], "size": [2.0, 0.2, 0.2],
         "rotation": [0.0, 0.0, 0.5]},
        # Beam row without a persisted OBB (defensive): skipped, not crashed.
        {"source": "beam", "segId": 8, "seq": 4},
    ]
    vols = collect_volumes(instances, {"paths": []})
    assert len(vols) == 1
    assert vols[0] == {"kind": "obb", "instance_id": 7, "seq": 3,
                       "shape": {"center": [1.0, 0.0, 0.0], "size": [2.0, 0.2, 0.2],
                                 "rotation": [0.0, 0.0, 0.5]}}
```

- [ ] **Step 3.2:** Run: `.venv/bin/pytest backend/tests/test_materialize.py -v` — new test FAILS (empty list).

- [ ] **Step 3.3: Implement.** In `collect_volumes`:
  - Docstring: `source in {'box','beam','draw'}` — box/beam → obb, draw → tube.
  - `if src not in ("box", "draw")` → `if src not in ("box", "beam", "draw")`
  - `if src == "box" and inst.get("center")...` → `if src in ("box", "beam") and inst.get("center")...`

  In `schemas.py` update the `Cuboid.source` comment to include `'beam'`.

- [ ] **Step 3.4:** Run: `.venv/bin/pytest backend/tests/test_materialize.py backend/tests/test_export_labels.py -v` — all PASS.

- [ ] **Step 3.5: Commit**

```bash
git add backend/labeling/materialize.py backend/app/schemas.py backend/tests/test_materialize.py
git commit -m "feat(beam): beam-source pointsets replay as exact OBB volumes in exports"
```

---

### Task 4: Backend — parity test for the frontend beam→OBB math

The frontend derives Euler XYZ angles from the beam frame (Task 5). This test replicates that algorithm in numpy and proves `obb_indices` selects exactly the points a beam's box should — axis extent and both cross-section half-extents, on a **non-axis-aligned** axis. If the frontend algorithm and this test ever disagree with `obb_indices`, labels diverge from the on-screen box — this is the single most important correctness gate in the feature.

**Files:**
- Test: `backend/tests/test_shapes.py` (extend)

- [ ] **Step 4.1: Write the test** (append to `test_shapes.py`):

```python
def _beam_frame(a, b):
    """Replicates frontend beam-graph.js beamFrame(): u along the axis,
    (v, w) from the world axis least aligned with u."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    d = b - a
    u = d / np.linalg.norm(d)
    ref = np.eye(3)[int(np.argmin(np.abs(u)))]
    v = np.cross(ref, u)
    v /= np.linalg.norm(v)
    w = np.cross(u, v)
    return u, v, w


def _euler_xyz_from_basis(u, v, w):
    """Replicates frontend eulerXYZFromBasis() (THREE.Euler 'XYZ' algorithm):
    angles whose Rx.Ry.Rz composition equals the matrix with columns u,v,w."""
    m13, m23, m33 = w[0], w[1], w[2]
    m11, m12 = u[0], v[0]
    m22, m32 = v[1], v[2]
    y = np.arcsin(np.clip(m13, -1.0, 1.0))
    if abs(m13) < 0.9999999:
        x = np.arctan2(-m23, m33)
        z = np.arctan2(-m12, m11)
    else:
        x = np.arctan2(m32, m22)
        z = 0.0
    return [float(x), float(y), float(z)]


def test_beam_style_obb_matches_frame_membership():
    """A beam OBB built the frontend's way (frame -> euler) must select exactly
    the points inside the swept square box, on a skew (non-world-aligned) axis."""
    rng = np.random.default_rng(7)
    a = np.array([0.5, -0.2, 1.0])
    b = np.array([2.0, 1.3, -0.4])
    width = 0.3
    u, v, w = _beam_frame(a, b)
    length = np.linalg.norm(b - a)

    pts = rng.uniform(-1.5, 3.0, size=(5000, 3)).astype(np.float32)
    # Ground truth straight from the frame (the spec's containment definition).
    rel = pts.astype(np.float64) - a
    du, dv, dw = rel @ u, rel @ v, rel @ w
    inside = (du >= 0) & (du <= length) & (np.abs(dv) <= width / 2) & (np.abs(dw) <= width / 2)
    assert inside.sum() > 10          # sanity: the box actually contains points

    box = {
        "center": ((a + b) / 2).tolist(),
        "size": [float(length), width, width],
        "rotation": _euler_xyz_from_basis(u, v, w),
    }
    got = np.zeros(len(pts), dtype=bool)
    got[obb_indices(pts, box)] = True
    # float32 points on float64 boundaries: allow disagreement only within an
    # epsilon shell of the box faces.
    margin = 1e-5
    strict = (du >= margin) & (du <= length - margin) \
        & (np.abs(dv) <= width / 2 - margin) & (np.abs(dw) <= width / 2 - margin)
    outside = (du < -margin) | (du > length + margin) \
        | (np.abs(dv) > width / 2 + margin) | (np.abs(dw) > width / 2 + margin)
    assert got[strict].all()
    assert not got[outside].any()
```

- [ ] **Step 4.2:** Run: `.venv/bin/pytest backend/tests/test_shapes.py -v` — expect PASS (this validates existing `obb_indices` against the new construction; failure means the euler derivation above is wrong — fix the test helper, and mirror the fix in Task 5's JS).

- [ ] **Step 4.3: Commit**

```bash
git add backend/tests/test_shapes.py
git commit -m "test(beam): frame->euler OBB parity against obb_indices"
```

---

### Task 5: Frontend — `beam-graph.js` pure state machine + tests

**Files:**
- Create: `frontend/src/beam-graph.js`
- Test: `frontend/src/beam-graph.test.js`

- [ ] **Step 5.1: Write the failing tests**

```javascript
import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  initBeamState, addNode, clickNode, selectNode, selectEdge, clearSelection,
  moveNode, deleteSelected, setWidth, nudgeWidth, setClass,
  applyTargets, markApplied, commitAll, beamFrame, eulerXYZFromBasis,
  obbForEdge, toStructureDoc, seedFromServer, MIN_WIDTH,
} from './beam-graph.js';

const CLS = 10;

function twoNodes() {
  let s = initBeamState();
  s = addNode(s, [0, 0, 0]);
  s = addNode(s, [2, 0, 0]);
  return s;
}

function oneEdge() {
  let s = twoNodes();
  s = clickNode(s, 1, CLS);          // select node 1
  s = clickNode(s, 2, CLS);          // connect 1-2 -> edge selected
  return s;
}

describe('beam-graph: nodes & edges', () => {
  it('addNode appends without selecting (spec: new node not auto-selected)', () => {
    const s = addNode(initBeamState(), [1, 2, 3]);
    expect(s.nodes).toEqual([{ id: 1, pos: [1, 2, 3] }]);
    expect(s.selection).toBeNull();
  });

  it('clickNode selects; clicking a different node connects and selects the edge', () => {
    let s = twoNodes();
    s = clickNode(s, 1, CLS);
    expect(s.selection).toEqual({ kind: 'node', id: 1 });
    s = clickNode(s, 2, CLS);
    expect(s.edges).toHaveLength(1);
    expect(s.edges[0]).toMatchObject({ a: 1, b: 2, classId: CLS, instanceId: null, dirty: true });
    expect(s.edges[0].width).toBe(s.lastWidth);
    expect(s.selection).toEqual({ kind: 'edge', id: s.edges[0].id });
  });

  it('clicking the selected node again keeps it selected (no self-edge)', () => {
    let s = clickNode(twoNodes(), 1, CLS);
    s = clickNode(s, 1, CLS);
    expect(s.edges).toHaveLength(0);
    expect(s.selection).toEqual({ kind: 'node', id: 1 });
  });

  it('connecting an already-connected pair selects the existing edge (de-dup, both orders)', () => {
    let s = oneEdge();
    const edgeId = s.edges[0].id;
    s = clickNode(s, 2, CLS);          // select node 2
    s = clickNode(s, 1, CLS);          // reverse order
    expect(s.edges).toHaveLength(1);
    expect(s.selection).toEqual({ kind: 'edge', id: edgeId });
  });

  it('moveNode updates pos and dirties incident edges only', () => {
    let s = oneEdge();
    // Nodes and edges share one nextId counter: after oneEdge() (nodes 1, 2 +
    // edge 3) the next node is id 4.
    s = addNode(s, [0, 5, 0]);
    s = selectNode(s, 4);
    s = clickNode(s, 1, CLS);          // second edge 4-1
    s = markApplied(markApplied(s, s.edges[0].id, 100), s.edges[1].id, 101);
    s = moveNode(s, 2, [3, 0, 0]);
    expect(s.nodes.find((n) => n.id === 2).pos).toEqual([3, 0, 0]);
    expect(s.edges.find((e) => e.a === 1 && e.b === 2).dirty).toBe(true);
    expect(s.edges.find((e) => e.a === 4).dirty).toBe(false);
  });

  it('deleteSelected on a node cascades incident edges; on an edge removes just it', () => {
    let s = oneEdge();
    s = selectNode(s, 1);
    s = deleteSelected(s);
    expect(s.nodes.map((n) => n.id)).toEqual([2]);
    expect(s.edges).toHaveLength(0);
    expect(s.selection).toBeNull();

    let t = oneEdge();                 // edge already selected
    t = deleteSelected(t);
    expect(t.edges).toHaveLength(0);
    expect(t.nodes).toHaveLength(2);   // nodes survive edge deletion
  });
});

describe('beam-graph: width & class', () => {
  it('setWidth targets the selected edge, updates lastWidth, clamps to MIN_WIDTH', () => {
    let s = setWidth(oneEdge(), 0.5);
    expect(s.edges[0].width).toBe(0.5);
    expect(s.lastWidth).toBe(0.5);
    s = setWidth(s, 0);
    expect(s.edges[0].width).toBe(MIN_WIDTH);
  });

  it('setWidth with no edge selected only updates lastWidth (next-edge default)', () => {
    let s = setWidth(clearSelection(oneEdge()), 0.7);
    expect(s.edges[0].width).not.toBe(0.7);
    expect(s.lastWidth).toBe(0.7);
    s = addNode(s, [0, 3, 0]);                   // node 4 (shared id counter)
    s = clickNode(selectNode(s, 1), 4, CLS);
    expect(s.edges.at(-1).width).toBe(0.7);
  });

  it('nudgeWidth is multiplicative on the selected edge, no-op otherwise', () => {
    let s = oneEdge();
    const w0 = s.edges[0].width;
    s = nudgeWidth(s, +1);
    expect(s.edges[0].width).toBeCloseTo(w0 * 1.08);
    const t = nudgeWidth(clearSelection(s), +1);
    expect(t.edges[0].width).toBe(s.edges[0].width);
  });

  it('setClass retargets the selected edge and dirties it', () => {
    let s = markApplied(oneEdge(), oneEdge().edges[0].id, 100);
    s = setClass(s, 11);
    expect(s.edges[0]).toMatchObject({ classId: 11, dirty: true });
  });
});

describe('beam-graph: apply & commit', () => {
  it('applyTargets returns unapplied or dirty edges, skipping degenerate ones', () => {
    let s = oneEdge();
    expect(applyTargets(s).map((e) => e.id)).toEqual([s.edges[0].id]);
    s = markApplied(s, s.edges[0].id, 100);
    expect(applyTargets(s)).toEqual([]);
    s = moveNode(s, 2, [4, 0, 0]);
    expect(applyTargets(s)).toHaveLength(1);
    // Degenerate: both endpoints coincide.
    s = moveNode(s, 2, [0, 0, 0]);
    expect(applyTargets(s)).toEqual([]);
  });

  it('markApplied stores the instance id and clears dirty', () => {
    let s = oneEdge();
    s = markApplied(s, s.edges[0].id, 42);
    expect(s.edges[0]).toMatchObject({ instanceId: 42, dirty: false });
  });

  it('commitAll retires applied edges (baked endpoints), keeps failed ones + their nodes', () => {
    let s = oneEdge();
    s = addNode(s, [0, 5, 0]);                   // node 4 (shared id counter)
    s = clickNode(selectNode(s, 1), 4, CLS);     // second edge 1-4, unapplied
    s = markApplied(s, s.edges[0].id, 100);
    s = commitAll(s);
    expect(s.committed).toEqual([{
      a: [0, 0, 0], b: [2, 0, 0], width: s.lastWidth, classId: CLS, instanceId: 100,
    }]);
    expect(s.edges).toHaveLength(1);              // the unapplied one stays
    expect(s.nodes.map((n) => n.id).sort()).toEqual([1, 4]);  // node 2 dropped
    expect(s.selection).toBeNull();
  });
});

describe('beam-graph: OBB math', () => {
  it('beamFrame is orthonormal with u along the axis, for skew and axis-aligned axes', () => {
    for (const [a, b] of [
      [[0.5, -0.2, 1.0], [2.0, 1.3, -0.4]],
      [[0, 0, 0], [1, 0, 0]],
      [[0, 0, 0], [0, 3, 0]],
      [[1, 1, 1], [1, 1, 5]],
    ]) {
      const { u, v, w, len } = beamFrame(a, b);
      const dot = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
      expect(len).toBeCloseTo(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]));
      for (const vec of [u, v, w]) expect(dot(vec, vec)).toBeCloseTo(1);
      expect(dot(u, v)).toBeCloseTo(0);
      expect(dot(u, w)).toBeCloseTo(0);
      expect(dot(v, w)).toBeCloseTo(0);
      const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      expect(dot(u, d)).toBeCloseTo(len);
    }
  });

  it('eulerXYZFromBasis reconstructs the basis via THREE Euler XYZ (Rx·Ry·Rz parity)', () => {
    for (const [a, b] of [
      [[0.5, -0.2, 1.0], [2.0, 1.3, -0.4]],
      [[0, 0, 0], [1, 0, 0]],
      [[0, 0, 0], [0, 0, -2]],       // near the m13 = ±1 gimbal branch
      [[-1, 2, 0.3], [0.7, 2.1, 4]],
    ]) {
      const { u, v, w } = beamFrame(a, b);
      const [rx, ry, rz] = eulerXYZFromBasis(u, v, w);
      const rebuilt = new THREE.Matrix4().makeRotationFromEuler(
        new THREE.Euler(rx, ry, rz, 'XYZ'));
      const direct = new THREE.Matrix4().makeBasis(
        new THREE.Vector3(...u), new THREE.Vector3(...v), new THREE.Vector3(...w));
      rebuilt.elements.forEach((el, i) => expect(el).toBeCloseTo(direct.elements[i], 6));
    }
  });

  it('obbForEdge: center is the midpoint, size is [len, width, width]', () => {
    const s = setWidth(oneEdge(), 0.3);
    const obb = obbForEdge(s, s.edges[0]);
    expect(obb.center).toEqual([1, 0, 0]);
    expect(obb.size[0]).toBeCloseTo(2);
    expect(obb.size[1]).toBe(0.3);
    expect(obb.size[2]).toBe(0.3);
    expect(obb.rotation).toHaveLength(3);
  });
});

describe('beam-graph: serialization', () => {
  it('toStructureDoc <-> seedFromServer round-trip preserves graph + committed + dirty', () => {
    let s = oneEdge();
    s = addNode(s, [0, 5, 0]);
    s = markApplied(s, s.edges[0].id, 100);
    s = moveNode(s, 1, [0, 1, 0]);               // applied edge now dirty
    s = commitAll(s);                            // applied edge → committed; isolated nodes dropped
    s = addNode(s, [7, 7, 7]);
    const doc = toStructureDoc(s);
    expect(doc.committed_beams).toHaveLength(1);
    expect(doc.committed_beams[0].class_id).toBe(CLS);

    const seeded = seedFromServer(initBeamState(), doc);
    expect(toStructureDoc(seeded)).toEqual(doc);
    // nextId continues past every seeded id (no collisions).
    const ids = [...seeded.nodes.map((n) => n.id), ...seeded.edges.map((e) => e.id)];
    expect(seeded.nextId).toBeGreaterThan(Math.max(0, ...ids));
  });

  it('seedFromServer defaults dirty from instance_id when absent', () => {
    const doc = {
      nodes: [{ id: 1, pos: [0, 0, 0] }, { id: 2, pos: [1, 0, 0] }],
      edges: [
        { id: 3, a: 1, b: 2, width: 0.2, class_id: CLS, instance_id: 9 },
        { id: 4, a: 1, b: 2, width: 0.2, class_id: CLS, instance_id: null },
      ],
      committed_beams: [],
    };
    const s = seedFromServer(initBeamState(), doc);
    expect(s.edges[0].dirty).toBe(false);
    expect(s.edges[1].dirty).toBe(true);
  });
});
```

- [ ] **Step 5.2:** Run: `cd frontend && npx vitest run src/beam-graph.test.js` — expect FAIL (module missing).

- [ ] **Step 5.3: Implement `frontend/src/beam-graph.js`**

```javascript
// beam-graph.js — pure state machine for the Beam sub-mode of Label mode.
// No Three.js, no React (same testability contract as draw-paths.js).
// Spec: docs/superpowers/specs/2026-07-10-beam-structure-labeling-design.md.
//
// State shape:
// {
//   nodes:     [{ id, pos: [x,y,z] }],                        // recentered frame
//   edges:     [{ id, a, b, width, classId, instanceId|null, dirty }],
//   committed: [{ a:[x,y,z], b:[x,y,z], width, classId, instanceId }],
//   selection: { kind: 'node'|'edge', id } | null,
//   lastWidth: number,             // a new edge inherits the last-used width
//   nextId:    number,             // monotonic id source (deterministic, testable)
// }
// classId is the canonical numeric class id (ClassDef.class_id), never the
// array position. `dirty` marks an edge whose geometry/width/class changed
// since its last apply — Enter re-applies only dirty or never-applied edges.

export const MIN_WIDTH = 0.01;
const DEGENERATE_EPS = 1e-6;

export function initBeamState({ defaultWidth = 0.2 } = {}) {
  return {
    nodes: [], edges: [], committed: [],
    selection: null, lastWidth: defaultWidth, nextId: 1,
  };
}

export function nodePos(state, id) {
  const n = state.nodes.find((x) => x.id === id);
  if (!n) throw new Error(`beam-graph: no node with id ${id}`);
  return n.pos;
}

export function addNode(state, pos) {
  // A new node is NOT auto-selected (spec: workflow step 2) — placing a run
  // of joints must not chain accidental beams.
  const node = { id: state.nextId, pos };
  return { ...state, nodes: [...state.nodes, node], nextId: state.nextId + 1 };
}

export function selectNode(state, id) {
  return { ...state, selection: { kind: 'node', id } };
}

export function selectEdge(state, id) {
  return { ...state, selection: { kind: 'edge', id } };
}

export function clearSelection(state) {
  return { ...state, selection: null };
}

// Ctrl+click on an existing node: with a *different* node selected, connect
// them into a beam (selecting it); otherwise select the clicked node.
export function clickNode(state, nodeId, defaultClassId) {
  const sel = state.selection;
  if (sel?.kind === 'node' && sel.id !== nodeId) {
    return addEdge(state, sel.id, nodeId, defaultClassId);
  }
  return selectNode(state, nodeId);
}

function addEdge(state, aId, bId, classId) {
  // Connecting an already-connected pair is a select, not a duplicate.
  const existing = state.edges.find(
    (e) => (e.a === aId && e.b === bId) || (e.a === bId && e.b === aId));
  if (existing) return selectEdge(state, existing.id);
  const edge = {
    id: state.nextId, a: aId, b: bId, width: state.lastWidth,
    classId, instanceId: null, dirty: true,
  };
  return {
    ...state, edges: [...state.edges, edge],
    selection: { kind: 'edge', id: edge.id }, nextId: state.nextId + 1,
  };
}

export function moveNode(state, nodeId, pos) {
  const nodes = state.nodes.map((n) => (n.id === nodeId ? { ...n, pos } : n));
  // Incident beams follow the joint; they need a re-apply to re-extract.
  const edges = state.edges.map((e) =>
    e.a === nodeId || e.b === nodeId ? { ...e, dirty: true } : e);
  return { ...state, nodes, edges };
}

function deleteNode(state, nodeId) {
  return {
    ...state,
    nodes: state.nodes.filter((n) => n.id !== nodeId),
    edges: state.edges.filter((e) => e.a !== nodeId && e.b !== nodeId),
    selection: null,
  };
}

function deleteEdge(state, edgeId) {
  return {
    ...state,
    edges: state.edges.filter((e) => e.id !== edgeId),
    selection: null,
  };
}

export function deleteSelected(state) {
  const sel = state.selection;
  if (!sel) return state;
  return sel.kind === 'node' ? deleteNode(state, sel.id) : deleteEdge(state, sel.id);
}

export function setWidth(state, w) {
  const width = Math.max(w, MIN_WIDTH);
  if (state.selection?.kind !== 'edge') {
    // No beam selected: the field still sets the width new edges inherit.
    return { ...state, lastWidth: width };
  }
  const edges = state.edges.map((e) =>
    e.id === state.selection.id ? { ...e, width, dirty: true } : e);
  return { ...state, edges, lastWidth: width };
}

export function nudgeWidth(state, dir) {
  if (state.selection?.kind !== 'edge') return state;
  const e = state.edges.find((x) => x.id === state.selection.id);
  // Multiplicative steps feel uniform across member sizes (8%, like Draw).
  return setWidth(state, e.width * (1 + Math.sign(dir) * 0.08));
}

export function setClass(state, classId) {
  if (state.selection?.kind !== 'edge') return state;
  const edges = state.edges.map((e) =>
    e.id === state.selection.id ? { ...e, classId, dirty: true } : e);
  return { ...state, edges };
}

function isDegenerate(state, edge) {
  const a = nodePos(state, edge.a), b = nodePos(state, edge.b);
  return Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) < DEGENERATE_EPS;
}

// Edges Enter should apply: never applied, or edited since the last apply.
// Degenerate (coincident endpoints) edges are skipped — the client-side guard
// from the spec's error handling.
export function applyTargets(state) {
  return state.edges.filter(
    (e) => (e.dirty || e.instanceId == null) && !isDegenerate(state, e));
}

export function markApplied(state, edgeId, instanceId) {
  const edges = state.edges.map((e) =>
    e.id === edgeId ? { ...e, instanceId, dirty: false } : e);
  return { ...state, edges };
}

// Ctrl+Enter: retire every applied edge into the committed layer (bake its
// endpoint positions — the graph structure is not needed after commit) and
// drop it from the active graph. Edges that never got an instance (apply
// failed / 0 points) stay active so the user can fix them; nodes with no
// remaining edges are dropped (canvas clears).
export function commitAll(state) {
  const committed = [...state.committed];
  const remaining = [];
  for (const e of state.edges) {
    if (e.instanceId == null) { remaining.push(e); continue; }
    committed.push({
      a: [...nodePos(state, e.a)], b: [...nodePos(state, e.b)],
      width: e.width, classId: e.classId, instanceId: e.instanceId,
    });
  }
  const used = new Set(remaining.flatMap((e) => [e.a, e.b]));
  return {
    ...state,
    nodes: state.nodes.filter((n) => used.has(n.id)),
    edges: remaining, committed, selection: null,
  };
}

// ── OBB math ────────────────────────────────────────────────────────────────
// A beam's swept square box is expressed as the codebase-standard OBB
// {center, size, rotation} so the existing apply-shape endpoint, viewer
// preview conventions, and export replay (materialize.collect_volumes) all
// consume it unchanged. Rotation is Euler XYZ whose Rx·Ry·Rz composition is
// the frame matrix — the one convention every containment test shares (see
// CLAUDE.md "Coordinate system"). Roll about the axis is unspecified by
// design (square section): any stable (v, w) pair is acceptable.

const sub = (p, q) => [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
const cross = (p, q) => [
  p[1] * q[2] - p[2] * q[1],
  p[2] * q[0] - p[0] * q[2],
  p[0] * q[1] - p[1] * q[0],
];
const norm = (p) => Math.hypot(p[0], p[1], p[2]);
const scale = (p, s) => [p[0] * s, p[1] * s, p[2] * s];

export function beamFrame(a, b) {
  const d = sub(b, a);
  const len = norm(d);
  const u = scale(d, 1 / len);
  // Reference: the world axis least aligned with u — the cross product can't
  // degenerate. Mirrored by _beam_frame in backend/tests/test_shapes.py.
  const abs = [Math.abs(u[0]), Math.abs(u[1]), Math.abs(u[2])];
  const ref = abs[0] <= abs[1] && abs[0] <= abs[2] ? [1, 0, 0]
    : abs[1] <= abs[2] ? [0, 1, 0] : [0, 0, 1];
  const v0 = cross(ref, u);
  const v = scale(v0, 1 / norm(v0));
  const w = cross(u, v);
  return { u, v, w, len };
}

// Euler XYZ (radians) whose Rx·Ry·Rz composition equals the rotation matrix
// with COLUMNS (u, v, w). Same algorithm as
// THREE.Euler.setFromRotationMatrix(m, 'XYZ') — parity-tested against THREE
// in beam-graph.test.js and against obb_indices in test_shapes.py.
export function eulerXYZFromBasis(u, v, w) {
  const m11 = u[0], m12 = v[0], m13 = w[0];
  const m22 = v[1], m23 = w[1];
  const m32 = v[2], m33 = w[2];
  const y = Math.asin(Math.min(1, Math.max(-1, m13)));
  if (Math.abs(m13) < 0.9999999) {
    return [Math.atan2(-m23, m33), y, Math.atan2(-m12, m11)];
  }
  return [Math.atan2(m32, m22), y, 0];
}

export function obbForEdge(state, edge) {
  const a = nodePos(state, edge.a);
  const b = nodePos(state, edge.b);
  const { u, v, w, len } = beamFrame(a, b);
  return {
    center: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2],
    size: [len, edge.width, edge.width],
    rotation: eulerXYZFromBasis(u, v, w),
  };
}

// ── Persistence (structure.json) ────────────────────────────────────────────

export function toStructureDoc(state) {
  return {
    nodes: state.nodes.map((n) => ({ id: n.id, pos: n.pos })),
    edges: state.edges.map((e) => ({
      id: e.id, a: e.a, b: e.b, width: e.width,
      class_id: e.classId, instance_id: e.instanceId, dirty: e.dirty,
    })),
    committed_beams: state.committed.map((c) => ({
      a: c.a, b: c.b, width: c.width,
      class_id: c.classId, instance_id: c.instanceId,
    })),
  };
}

export function seedFromServer(state, doc) {
  let maxId = 0;
  const nodes = (doc.nodes ?? []).map((n) => {
    maxId = Math.max(maxId, n.id);
    return { id: n.id, pos: n.pos };
  });
  const edges = (doc.edges ?? []).map((e) => {
    maxId = Math.max(maxId, e.id);
    return {
      id: e.id, a: e.a, b: e.b, width: e.width, classId: e.class_id,
      instanceId: e.instance_id ?? null,
      // Older docs without a dirty flag: an unapplied edge needs an apply.
      dirty: e.dirty ?? e.instance_id == null,
    };
  });
  const committed = (doc.committed_beams ?? []).map((c) => ({
    a: c.a, b: c.b, width: c.width, classId: c.class_id, instanceId: c.instance_id,
  }));
  return { ...state, nodes, edges, committed, nextId: Math.max(state.nextId, maxId + 1) };
}
```

- [ ] **Step 5.4:** Run: `cd frontend && npx vitest run src/beam-graph.test.js` — all PASS. (If the euler parity test fails, fix `eulerXYZFromBasis` to match THREE, then re-check Task 4's Python mirror stays identical.)

- [ ] **Step 5.5: Commit**

```bash
git add frontend/src/beam-graph.js frontend/src/beam-graph.test.js
git commit -m "feat(beam): pure graph state machine with OBB conversion"
```

---

### Task 6: Frontend — API client + tool registration

**Files:**
- Modify: `frontend/src/api.js` (after `getCenterlines`, ~line 269)
- Modify: `frontend/src/label-tools.js`
- Test: `frontend/src/label-tools.test.js`

- [ ] **Step 6.1: Update `label-tools.test.js`** — rail order gains `'beam'`; beam gates like draw:

```javascript
  it('lists the four selection tools in rail order', () => {
    expect(TOOLS.map((t) => t.id)).toEqual(['presegment', 'box', 'draw', 'beam']);
  });
```

and in the gating test add alongside the draw assertions:

```javascript
    expect(toolAvailable('beam', raw)).toBe(false);
    // ...
    expect(toolAvailable('beam', sessionOnly)).toBe(false);
    // ...
    expect(toolAvailable('beam', ann)).toBe(true);
```

- [ ] **Step 6.2:** Run: `cd frontend && npx vitest run src/label-tools.test.js` — FAIL.

- [ ] **Step 6.3: Implement.** `label-tools.js`: replace the reserved comment with the real entry and widen the draw gate:

```javascript
export const TOOLS = [
  { id: 'presegment', icon: '◱', label: 'Presegment' },
  { id: 'box',        icon: '▭', label: 'Box' },
  { id: 'draw',       icon: '✎', label: 'Draw' },
  { id: 'beam',       icon: '⌶', label: 'Beam' },
];

export function toolAvailable(id, { segState, isAnnotated }) {
  // Draw persists centerlines.json, Beam persists structure.json — both need
  // a session dir, which only annotated-tier scans have.
  if (id === 'draw' || id === 'beam') return !!segState && !!isAnnotated;
  ...
```

`api.js`: add after `getCenterlines`:

```javascript
  async getStructure() {
    const r = await fetch('/api/segment/structure');
    if (!r.ok) throw new Error(`getStructure failed: ${r.status} ${await r.text()}`);
    return r.json();
  },
  // sessionId pins the write: the backend 409s if the active session changed
  // between the edit and this (debounced) write — never write cross-session.
  async putStructure(doc, sessionId) {
    const r = await fetch('/api/segment/structure', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...doc, session_id: sessionId }),
    });
    if (!r.ok) throw new Error(`putStructure failed: ${r.status} ${await r.text()}`);
    return r.json();
  },
```

- [ ] **Step 6.4:** Run: `cd frontend && npx vitest run` — all PASS.

- [ ] **Step 6.5: Commit**

```bash
git add frontend/src/api.js frontend/src/label-tools.js frontend/src/label-tools.test.js
git commit -m "feat(beam): register Beam tool + structure API client"
```

---

### Task 7: Frontend — `beam-mode.jsx` component

**Files:**
- Create: `frontend/src/beam-mode.jsx`

No unit tests here — all logic already lives (tested) in `beam-graph.js`; this file is the Three.js/React shell, verified in Task 10's browser pass. Structure and idioms mirror `draw-mode.jsx` deliberately — read it side by side while implementing.

- [ ] **Step 7.1: Implement `frontend/src/beam-mode.jsx`**

```jsx
// beam-mode.jsx — Beam sub-mode of Label mode. Steel members (beams/pillars)
// are labeled by building a node/edge graph: joints are placed on the cloud,
// edges between joints are square-section boxes applied through the shared
// apply-shape pipeline, one instance per beam. State machine in beam-graph.js;
// spec: docs/superpowers/specs/2026-07-10-beam-structure-labeling-design.md.

import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { evtToNdc } from './viewer.jsx';
import { VoxaAPI } from './api.js';
import { applyDelta } from './segment-state.js';
import {
  initBeamState, addNode, clickNode, selectNode, selectEdge, clearSelection,
  moveNode, deleteSelected, setWidth, nudgeWidth, setClass,
  applyTargets, markApplied, commitAll, obbForEdge, toStructureDoc,
  seedFromServer, nodePos,
} from './beam-graph.js';

// Capture-phase keyboard driver (same trick as DrawKeys / FastLabelKeys).
export function BeamKeys({ active, classes, onKey }) {
  useEffect(() => {
    if (!active) return undefined;
    const handler = (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      // Ctrl+Enter commits the batch; every other Ctrl/Meta/Alt combo
      // (Ctrl+S/Z, Ctrl+click…) passes through untouched.
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault(); e.stopPropagation();
        onKey({ type: 'commit' });
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const cls = classes.find((c) => c.hotkey === e.key);
      let handled = true;
      if (cls) onKey({ type: 'class', classId: cls.class_id });
      else if (e.key === 'Enter') onKey({ type: 'apply' });
      else if (e.key === 'Escape') onKey({ type: 'escape' });
      else if (e.key === 'Backspace' || e.key === 'Delete') onKey({ type: 'delete' });
      else if (e.key === '+' || e.key === '=') onKey({ type: 'width', dir: +1 });
      else if (e.key === '-' || e.key === '_') onKey({ type: 'width', dir: -1 });
      else handled = false;
      if (handled) { e.preventDefault(); e.stopPropagation(); }
    };
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [active, classes, onKey]);
  return null;
}

function BeamHUD({ state, toast }) {
  const sel = state.selection;
  return (
    <div style={{
      position: 'fixed', bottom: 16, left: '50%', transform: 'translateX(-50%)',
      background: 'rgba(17, 24, 39, 0.92)', color: '#e5e7eb', borderRadius: 8,
      padding: '8px 14px', fontSize: 12, display: 'flex', gap: 14,
      alignItems: 'center', pointerEvents: 'none', zIndex: 30,
      border: '1px solid rgba(96,165,250,0.5)',
    }}>
      {toast ? <b style={{ color: '#fbbf24' }}>{toast}</b> : (
        <>
          <b style={{ color: sel?.kind === 'node' ? '#fb923c' : '#60a5fa' }}>
            {sel?.kind === 'node' ? 'Joint selected'
              : sel?.kind === 'edge' ? 'Beam selected'
              : 'Build beam graph'}
          </b>
          <span style={{ opacity: 0.65 }}>
            {sel?.kind === 'node'
              ? 'Ctrl+click another joint to connect · drag move · ⌫ delete · Esc deselect'
              : sel?.kind === 'edge'
              ? 'scroll/± width · class hotkey · ⌫ delete · Enter apply · Esc deselect'
              : 'Ctrl+click cloud place joint · Ctrl+click joint select · Enter apply · Ctrl+Enter commit · Esc exit'}
          </span>
        </>
      )}
    </div>
  );
}

// Oriented box mesh for a beam a→b (shared by active edges + committed layer).
function makeBeamBox(a, b, width, matCfg) {
  const av = new THREE.Vector3(...a), bv = new THREE.Vector3(...b);
  const len = av.distanceTo(bv);
  if (len < 1e-6) return null;
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(len, width, width),
    new THREE.MeshBasicMaterial(matCfg));
  mesh.position.copy(av).lerp(bv, 0.5);
  // Box local x is the beam axis; any roll about it is fine (square section).
  mesh.quaternion.setFromUnitVectors(
    new THREE.Vector3(1, 0, 0), bv.clone().sub(av).normalize());
  return mesh;
}

function BeamOverlay({ viewerRef, beam, setBeam, classes, defaultClassId, showCommitted }) {
  const layerRef = useRef(null);        // { group, remove }
  const dragRef = useRef(null);         // { nodeId, plane, mesh, last }
  const beamRef = useRef(beam);
  beamRef.current = beam;
  const defaultClassIdRef = useRef(defaultClassId);
  defaultClassIdRef.current = defaultClassId;

  // One overlay group for the lifetime of the sub-mode.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v?.attachOverlayGroup) return undefined;
    layerRef.current = v.attachOverlayGroup();
    return () => { layerRef.current?.remove(); layerRef.current = null; };
  }, [viewerRef]);

  // Rebuild overlay children whenever the graph/selection changes. Graphs are
  // tiny (dozens of members), so dispose-and-rebuild beats bookkeeping.
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer?.group) return;
    const group = layer.group;
    while (group.children.length) {
      const c = group.children.pop();
      c.geometry?.dispose?.(); c.material?.dispose?.();
      group.remove(c);
    }
    const addRimShell = (mesh, grow, color = 0xffffff, opacity = 0.55) => {
      // White back-side shell = selection rim (borrowed from draw-mode).
      // Never swallows picks: raycast is a no-op.
      const p = mesh.geometry.parameters;
      const shell = new THREE.Mesh(
        new THREE.BoxGeometry(p.width + grow, p.height + grow, p.depth + grow),
        new THREE.MeshBasicMaterial({
          color, side: THREE.BackSide, transparent: true, opacity, depthWrite: false,
        }));
      shell.raycast = () => {};
      shell.position.copy(mesh.position);
      shell.quaternion.copy(mesh.quaternion);
      group.add(shell);
    };
    // Committed layer: faded, read-only, unpickable.
    if (showCommitted) {
      for (const c of beam.committed) {
        const cls = classes.find((k) => k.class_id === c.classId);
        const mesh = makeBeamBox(c.a, c.b, c.width, {
          color: new THREE.Color(cls?.color || '#9ca3af'),
          transparent: true, opacity: 0.08, depthWrite: false,
        });
        if (!mesh) continue;
        mesh.raycast = () => {};
        group.add(mesh);
      }
    }
    // Active edges.
    for (const e of beam.edges) {
      const cls = classes.find((k) => k.class_id === e.classId);
      const isSel = beam.selection?.kind === 'edge' && beam.selection.id === e.id;
      const mesh = makeBeamBox(
        nodePos(beam, e.a), nodePos(beam, e.b), e.width, {
          color: new THREE.Color(cls?.color || '#60a5fa'),
          transparent: true, depthWrite: false,
          opacity: isSel ? 0.40 : 0.25,
        });
      if (!mesh) continue;
      mesh.userData.beamEdge = e.id;
      group.add(mesh);
      if (isSel) addRimShell(mesh, Math.max(0.02, e.width * 0.15));
    }
    // Nodes last — pick priority: node sphere > beam box > cloud.
    const sphereR = Math.max(0.03, beam.lastWidth * 0.3);
    for (const n of beam.nodes) {
      const isSel = beam.selection?.kind === 'node' && beam.selection.id === n.id;
      const sph = new THREE.Mesh(
        new THREE.SphereGeometry(sphereR, 12, 8),
        new THREE.MeshBasicMaterial({ color: 0x60a5fa }));
      sph.position.set(n.pos[0], n.pos[1], n.pos[2]);
      sph.userData.beamNode = n.id;
      group.add(sph);
      if (isSel) {
        // Opaque orange ring — the connect anchor, same affordance as Draw's
        // extend anchor.
        const shell = new THREE.Mesh(
          new THREE.SphereGeometry(sphereR * 1.5, 12, 8),
          new THREE.MeshBasicMaterial({
            color: 0xfb923c, side: THREE.BackSide, transparent: true,
            opacity: 1, depthWrite: false,
          }));
        shell.raycast = () => {};
        shell.position.copy(sph.position);
        group.add(shell);
      }
    }
  }, [beam, classes, showCommitted]);

  // Pointer interactions.
  useEffect(() => {
    const v = viewerRef.current;
    const dom = v?.domElement?.();
    if (!dom) return undefined;
    const raycaster = new THREE.Raycaster();

    const castOverlay = (evt) => {
      const camera = v.getCamera();
      const group = layerRef.current?.group;
      if (!camera || !group) return [];
      const rect = dom.getBoundingClientRect();
      raycaster.setFromCamera(evtToNdc(evt, rect), camera);
      return raycaster.intersectObjects(group.children, false);
    };

    const onPointerDown = (evt) => {
      if (evt.button !== 0) return;
      const hits = castOverlay(evt);
      const nodeHit = hits.find((h) => h.object.userData.beamNode != null);
      if (nodeHit && !evt.ctrlKey && !evt.metaKey) {
        // Drag start: move the joint on a camera-parallel plane through it.
        const camera = v.getCamera();
        const normal = new THREE.Vector3();
        camera.getWorldDirection(normal);
        const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(
          normal, nodeHit.object.position.clone());
        dragRef.current = {
          nodeId: nodeHit.object.userData.beamNode, plane,
          mesh: nodeHit.object, last: null,
        };
        v.setOrbitEnabled(false);
        evt.stopPropagation();
        return;
      }
      if (evt.ctrlKey || evt.metaKey) {
        // Ctrl is the "edit graph" modifier; pick priority node > beam > cloud.
        if (nodeHit) {
          setBeam((s) => clickNode(s, nodeHit.object.userData.beamNode,
            defaultClassIdRef.current));
          evt.stopPropagation();
          return;
        }
        const edgeHit = hits.find((h) => h.object.userData.beamEdge != null);
        if (edgeHit) {
          setBeam((s) => selectEdge(s, edgeHit.object.userData.beamEdge));
          evt.stopPropagation();
          return;
        }
        const hit = v.firstHitUnderCursor(evt);
        if (hit) setBeam((s) => addNode(s, [hit.world.x, hit.world.y, hit.world.z]));
        evt.stopPropagation();
        return;
      }
      // Plain click: select a beam under the cursor (doesn't block orbit,
      // same feel as Draw's tube click); empty click clears the selection.
      const edgeHit = hits.find((h) => h.object.userData.beamEdge != null);
      if (edgeHit) {
        setBeam((s) => selectEdge(s, edgeHit.object.userData.beamEdge));
        return;
      }
      if (beamRef.current.selection) setBeam((s) => clearSelection(s));
    };

    const onPointerMove = (evt) => {
      const drag = dragRef.current;
      if (!drag) return;
      const camera = v.getCamera();
      if (!camera) return;
      const rect = dom.getBoundingClientRect();
      raycaster.setFromCamera(evtToNdc(evt, rect), camera);
      const pt = new THREE.Vector3();
      if (raycaster.ray.intersectPlane(drag.plane, pt)) {
        // Sphere tracks live; incident boxes re-render once on release —
        // full rebuilds at pointer rate would thrash geometry alloc/dispose.
        drag.mesh.position.copy(pt);
        drag.last = [pt.x, pt.y, pt.z];
      }
    };

    const onPointerUp = () => {
      const drag = dragRef.current;
      if (!drag) return;
      if (drag.last) setBeam((cur) => moveNode(cur, drag.nodeId, drag.last));
      // No movement → it was a click: select the joint (connect anchor).
      // Race-tolerant: the node may have been deleted mid-drag.
      else setBeam((cur) => cur.nodes.some((n) => n.id === drag.nodeId)
        ? selectNode(cur, drag.nodeId) : cur);
      dragRef.current = null;
      v.setOrbitEnabled(true);
    };

    // Wheel-resize beats orbit-zoom via a CAPTURE listener on the PARENT
    // (the same trick draw-mode documents). Falls through to zoom unless a
    // beam is selected.
    const wheelHost = dom.parentElement || dom;
    const onWheel = (evt) => {
      if (beamRef.current.selection?.kind !== 'edge') return;
      evt.preventDefault();
      evt.stopPropagation();
      setBeam((cur) => nudgeWidth(cur, -Math.sign(evt.deltaY)));
    };

    dom.addEventListener('pointerdown', onPointerDown, true);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    wheelHost.addEventListener('wheel', onWheel, { capture: true, passive: false });
    return () => {
      dom.removeEventListener('pointerdown', onPointerDown, true);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      wheelHost.removeEventListener('wheel', onWheel, { capture: true });
      v.setOrbitEnabled(true);
    };
  }, [viewerRef, setBeam]);

  return null;
}

export default function BeamMode({
  viewerRef, classes, setSegState, onExit, pointSize, setPointSize,
  defaultClassId, onClassChange, onApplied, sessionId,
}) {
  const [beam, setBeam] = useState(() => initBeamState());
  const beamLiveRef = useRef(beam);
  beamLiveRef.current = beam;
  const [showCommitted, setShowCommitted] = useState(true);
  const [toast, setToast] = useState(null);
  const toastTimer = useRef(null);
  const showToast = useCallback((msg) => {
    clearTimeout(toastTimer.current);
    setToast(msg);
    toastTimer.current = setTimeout(() => setToast(null), 2500);
  }, []);
  const seededRef = useRef(false);

  // Load stored structure once on open (active graph + committed layer).
  // On failure seededRef stays false → we never PUT → a load hiccup can't
  // wipe the server-side doc with an empty graph.
  useEffect(() => {
    let gone = false;
    VoxaAPI.getStructure()
      .then((doc) => {
        if (gone) return;
        setBeam((s) => seedFromServer(s, doc));
        seededRef.current = true;
      })
      .catch((err) => { if (!gone) showToast(`structure load failed: ${err.message}`); });
    return () => { gone = true; };
  }, [showToast]);

  // Persist graph geometry (debounced) after every graph change post-seed.
  // Covers apply/commit/edits per the spec; the session pin makes a session
  // switch mid-debounce a loud 409, never a cross-session write. Not on the
  // undo stack (matching centerlines.json). Depends on the three persisted
  // arrays — the beam-graph ops preserve their identity when untouched — so
  // selection clicks don't trigger writes.
  useEffect(() => {
    if (!seededRef.current) return undefined;
    const t = setTimeout(() => {
      VoxaAPI.putStructure(toStructureDoc(beamLiveRef.current), sessionId)
        .catch((err) => showToast(`structure save failed: ${err.message}`));
    }, 800);
    return () => clearTimeout(t);
  }, [beam.nodes, beam.edges, beam.committed, sessionId, showToast]);

  // Apply every never-applied/edited beam: one apply-shape call per beam,
  // sequential (same loop pattern as draw-mode's applySelection). Press-time
  // snapshot decides the calls; all state WRITES go through functional
  // updaters so user edits during the round-trips survive.
  const applyAll = useCallback(async () => {
    const snapshot = beamLiveRef.current;
    const targets = applyTargets(snapshot);
    if (targets.length === 0) return true;
    let allOk = true;
    for (const edge of targets) {
      const obb = obbForEdge(snapshot, edge);
      let r;
      try {
        r = await VoxaAPI.applyShape({
          shape: { type: 'obb', ...obb },
          targetClass: edge.classId,
          targetInst: edge.instanceId ?? -1,
        });
      } catch (err) {
        showToast(`apply failed: ${err.message}`);
        allOk = false;
        continue;                     // surface and move on; edge stays dirty
      }
      if (r.nAffected === 0) {
        showToast('no points in beam box');
        allOk = false;
        continue;                     // no instance allocated for empty beams
      }
      setBeam((cur) => markApplied(cur, edge.id, r.instanceId));
      setSegState((st) => st ? applyDelta(st, {
        indices: r.indices,
        after_class: r.afterClass,
        after_instance: r.afterInstance,
      }) : st);
      onApplied?.({ instanceId: r.instanceId, classId: edge.classId,
                    source: 'beam', obb });
    }
    return allOk;
  }, [setSegState, showToast, onApplied]);

  const commitBatch = useCallback(async () => {
    // Decide on live state OUTSIDE any updater — window.confirm must never
    // run inside a React updater (same rule as draw-mode Esc/Backspace).
    const s = beamLiveRef.current;
    if (s.edges.length === 0) { showToast('nothing to commit'); return; }
    const n = s.edges.length;
    if (!window.confirm(`Commit ${n} beam${n > 1 ? 's' : ''} to unconfirmed instances?`)) return;
    const ok = await applyAll();
    if (!ok) showToast('some beams failed to apply — they stay in the active graph');
    setBeam((cur) => commitAll(cur));
  }, [applyAll, showToast]);

  const onKey = useCallback((action) => {
    switch (action.type) {
      case 'class':
        onClassChange(action.classId);
        setBeam((s) => setClass(s, action.classId));
        break;
      case 'apply':
        applyAll();
        break;
      case 'commit':
        commitBatch();
        break;
      case 'width':
        setBeam((s) => nudgeWidth(s, action.dir));
        break;
      case 'delete': {
        const s = beamLiveRef.current;
        if (s.selection) setBeam((cur) => deleteSelected(cur));
        break;
      }
      case 'escape': {
        // Live-state decision outside the updater (onExit is a LabelMode
        // setState — calling it inside setBeam would run during render).
        const s = beamLiveRef.current;
        if (s.selection) setBeam((cur) => clearSelection(cur));
        else onExit();
        break;
      }
      default:
    }
  }, [applyAll, commitBatch, onExit, onClassChange]);

  // The sidebar class list is the single source of the beam class — clicking
  // a class row (or its hotkey) re-targets the selected beam.
  useEffect(() => {
    setBeam((s) => (s.selection?.kind === 'edge') ? setClass(s, defaultClassId) : s);
  }, [defaultClassId]);

  return (
    <>
      <BeamKeys active classes={classes} onKey={onKey} />
      <BeamHUD state={beam} toast={toast} />
      <BeamOverlay
        viewerRef={viewerRef}
        beam={beam}
        setBeam={setBeam}
        classes={classes}
        defaultClassId={defaultClassId}
        showCommitted={showCommitted}
      />
      <BeamPanel
        beam={beam}
        setBeam={setBeam}
        classes={classes}
        onApply={applyAll}
        onCommit={commitBatch}
        pointSize={pointSize}
        setPointSize={setPointSize}
        showCommitted={showCommitted}
        setShowCommitted={setShowCommitted}
      />
    </>
  );
}

// Side-panel section: beam list + width field + actions (rendered by
// LabelMode inside the left sidebar, like DrawPanel).
function BeamPanel({
  beam, setBeam, classes, onApply, onCommit, pointSize, setPointSize,
  showCommitted, setShowCommitted,
}) {
  const selEdge = beam.selection?.kind === 'edge'
    ? beam.edges.find((e) => e.id === beam.selection.id) : null;
  const widthValue = selEdge?.width ?? beam.lastWidth;
  const nCommitted = beam.committed.length;
  return (
    <div className="beam-panel" style={{ marginTop: 10 }}>
      <div className="side-hd"><span>Beams</span>
        <div className="side-hd-actions">
          {nCommitted > 0 && (
            <button className="hide-labeled-btn"
              onClick={() => setShowCommitted((v) => !v)}
              title={showCommitted
                ? `Hide ${nCommitted} committed beam${nCommitted === 1 ? '' : 's'}`
                : `Show ${nCommitted} committed beam${nCommitted === 1 ? '' : 's'}`}>
              {showCommitted ? '●' : '◌'} {nCommitted} committed
            </button>
          )}
          <span className="badge-soft">{beam.edges.length}</span>
        </div></div>
      <div className="ins-row">
        <label>Width</label>
        <input className="ins-input" type="number" step="0.01" min="0.01"
          value={Number(widthValue.toFixed(4))}
          onChange={(e) => {
            const v = parseFloat(e.target.value);
            if (Number.isFinite(v) && v > 0) setBeam((s) => setWidth(s, v));
          }} />
      </div>
      <div className="ctrl" style={{ margin: '6px 0' }}>
        <label>Point size <span className="mono">{pointSize.toFixed(3)}</span></label>
        <input type="range" min={0.002} max={1.5} step={0.005}
          value={pointSize} className="slider"
          onChange={(e) => setPointSize(Number(e.target.value))} />
      </div>
      <div style={{ maxHeight: 180, overflowY: 'auto' }}>
        {beam.edges.map((e) => {
          const cls = classes.find((c) => c.class_id === e.classId);
          const isSel = beam.selection?.kind === 'edge' && beam.selection.id === e.id;
          const status = e.instanceId == null ? '(staged)'
            : e.dirty ? `#${e.instanceId} (edited)` : `#${e.instanceId}`;
          return (
            <div key={e.id}
              className={'inst-row' + (isSel ? ' selected' : '')}
              onClick={() => setBeam((s) => selectEdge(s, e.id))}>
              <span className="inst-dot" style={{ background: cls?.color }} />
              <div className="inst-text">
                <b>{cls?.label || '?'} {status}</b>
                <em>w={e.width.toFixed(3)}</em>
              </div>
            </div>
          );
        })}
      </div>
      <div className="ins-actions">
        <button className="ghost-btn" disabled={beam.edges.length === 0}
          onClick={onApply}>↵ Apply</button>
        <button className="ghost-btn" disabled={beam.edges.length === 0}
          onClick={onCommit}>⌃↵ Commit</button>
      </div>
    </div>
  );
}
```

- [ ] **Step 7.2:** Run: `cd frontend && npx vitest run && npm run build` (from repo root: `npm run build`) — build must succeed with the new file imported nowhere yet; this catches syntax errors early.

- [ ] **Step 7.3: Commit**

```bash
git add frontend/src/beam-mode.jsx
git commit -m "feat(beam): beam-mode component (keys, HUD, overlay, panel)"
```

---

### Task 8: Frontend — wire Beam into `tool-options.jsx` + `mode-label.jsx`

**Files:**
- Modify: `frontend/src/tool-options.jsx`
- Modify: `frontend/src/mode-label.jsx`

- [ ] **Step 8.1: `tool-options.jsx`** — import `BeamMode from './beam-mode.jsx'`; add `BeamOptions` next to `DrawOptions` and dispatch it:

```jsx
function BeamOptions({
  viewerRef, classes, setSegState, onExit, pointSize, setPointSize,
  activeClass, setActiveClass, onToolApplied, autoConfirm, setAutoConfirm,
  activeSessionId,
}) {
  return (
    <div className="tool-options tool-options-beam">
      <BeamMode
        viewerRef={viewerRef}
        classes={classes}
        setSegState={setSegState}
        onExit={onExit}
        pointSize={pointSize}
        setPointSize={setPointSize}
        defaultClassId={classes.find((c) => c.id === activeClass)?.class_id ?? classes[0]?.class_id ?? 0}
        onClassChange={(cid) => {
          const cls = classes.find((c) => c.class_id === cid);
          if (cls) setActiveClass(cls.id);
        }}
        onApplied={onToolApplied}
        sessionId={activeSessionId}
      />
      <AutoConfirmToggle tool="beam" autoConfirm={autoConfirm} setAutoConfirm={setAutoConfirm} />
    </div>
  );
}
```

In the same file: rename the `onDrawApplied` prop consumption to `onToolApplied` (DrawOptions destructures `onToolApplied` and passes `onApplied={onToolApplied}`), and add to the dispatcher:

```jsx
  if (activeTool === 'beam') return <BeamOptions {...props} />;
```

- [ ] **Step 8.2: `mode-label.jsx`** — five edits:

1. Next to the `drawMode` derivation (line ~42):

```jsx
  const beamMode = activeTool === 'beam';
  // Sub-modes whose overlay owns viewport input (capture-phase keys +
  // pointer): global pick/hotkey handlers stand down. A future 5th tool
  // adds one term here instead of touching every gate.
  const subModeOwnsInput = drawMode || beamMode;
```

2. The two selection-pick effects that bail on `drawMode` (`onPointerPick` ~line 152, `onHullPick` ~line 172): bail on `subModeOwnsInput`, and swap `drawMode` for `subModeOwnsInput` in their dep arrays.
3. `autoConfirm` initial state gains `beam: false` (line ~50).
4. The global hotkey effect (~line 861): `if (fastMode || subModeOwnsInput) return;` and swap `drawMode` for `subModeOwnsInput` in the dep array (~line 920).
5. Generalize `onDrawApplied` → `onToolApplied` (replace the whole block at lines ~732-760: the lead comment, the existing `const instancesRef = useRefLabel(instances); instancesRef.current = instances;` pair, AND the callback — the snippet below re-declares `instancesRef`, so leaving the old declaration in place would be a `const` redeclaration SyntaxError). A re-apply must refresh the persisted OBB, not just the class — a beam re-applied after a joint drag replays into raw exports with its stored box:

```jsx
  // Surface each applied Draw/Beam instance in the right Instances panel as a
  // pointset row. Re-applies refresh the class AND (for beams) the persisted
  // OBB selection volume — a stale box would replay into raw exports;
  // instances absorbed by a Draw merge drop their row. Reads the latest
  // instances through a ref — one Enter can apply several groups back to
  // back, faster than the prop re-renders.
  const instancesRef = useRefLabel(instances);
  instancesRef.current = instances;
  const onToolApplied = useCallbackLabel(({
    instanceId, classId, mergedFrom = [], source = 'draw', obb = null,
  }) => {
    const cls = classes.find((c) => c.class_id === classId);
    if (!cls) return;
    const absorbed = new Set(mergedFrom);
    const kept = instancesRef.current.filter(
      (i) => !(i.kind === 'pointset' && absorbed.has(i.segId)));
    const existing = kept.find((i) => i.kind === 'pointset' && i.segId === instanceId);
    const volume = obb ? {
      center: [...obb.center], size: [...obb.size], rotation: [...obb.rotation],
    } : {};
    const next = existing
      ? kept.map((i) => i === existing ? { ...i, cls: cls.id, color: cls.color, ...volume } : i)
      : [...kept, {
        id: newId(),
        segId: instanceId,
        kind: 'pointset',
        cls: cls.id,
        label: `${cls.label} #${instanceId}`,
        color: cls.color,
        source,
        confirmed: autoConfirmFor(source),
        ...volume,
      }];
    instancesRef.current = next;
    onChange(next);
  }, [classes, onChange, autoConfirm, presegRapid]);
```

Then update the `<ToolOptions>` invocation (~line 1019): `onDrawApplied={onDrawApplied}` → `onToolApplied={onToolApplied}`, and add `activeSessionId={activeSessionId}`. Update the help section text (~line 495): `'Switch Presegment / Box / Draw / Beam'`.

- [ ] **Step 8.3:** Run: `npm test` (root — vitest + pytest) — all PASS. `grep -rn "onDrawApplied" frontend/src/` must return nothing.

- [ ] **Step 8.4: Commit**

```bash
git add frontend/src/tool-options.jsx frontend/src/mode-label.jsx
git commit -m "feat(beam): wire Beam tool into Label mode (rail, options, pointset rows)"
```

---

### Task 9: End-to-end backend test — beam apply → export replay

Proves the whole label path: an OBB applied via apply-shape + a beam pointset row with that OBB → raw-regime materialization labels exactly the points in the box. Mirrors the existing e2e OBB replay test added on `fix/data-label-integrity` — find it with `grep -rn "replay" backend/tests/test_materialize.py backend/tests/test_export_labels.py` and add a beam variant beside it.

**Files:**
- Test: `backend/tests/test_materialize.py` (extend)

- [ ] **Step 9.1: Write the test** (adapt to the existing e2e test's fixtures/helpers — reuse its scan/session setup verbatim, changing only `source: "box"` → `source: "beam"` and building the OBB from an `(a, b, width)` triple via the Task 4 helpers; import/duplicate `_beam_frame` / `_euler_xyz_from_basis` from `test_shapes.py`).

Assertions:
- `collect_volumes` picks the beam row up as an `obb` volume.
- `build_replay_index` + `replay_labels` on a denser synthetic target cloud: every target point inside the beam box (by the frame ground-truth test) gets the beam's class/instance; points outside don't.

- [ ] **Step 9.2:** Run: `.venv/bin/pytest backend/tests/test_materialize.py -v` — PASS.

- [ ] **Step 9.3: Commit**

```bash
git add backend/tests/test_materialize.py
git commit -m "test(beam): e2e beam OBB replay into raw materialization"
```

---

### Task 10: Full verification

- [ ] **Step 10.1:** Run the full suites and report actual output:

```bash
npm test
```

Expected: all backend (pytest) + frontend (vitest) tests green.

- [ ] **Step 10.2: Browser verification** (REQUIRED SUB-SKILL: browser-verification). **Safety rails from memory:** Label apply auto-saves to disk — create a **throwaway session** on a test scan (`legacy/test_scene` has no sessions — use an annotated scan + new blank session, delete it afterwards); kill any stale backend on :8765 and restart `npm run dev` first (Python changes don't hot-reload).

Verify, with screenshots and zero console errors:
1. Beam tool appears in the rail (annotated scan + session only; disabled otherwise).
2. Ctrl+click cloud places joints; Ctrl+click joint→joint creates a beam box between them; Ctrl+click near an existing joint reuses it (watertight).
3. Scroll resizes the selected beam; `+`/`-` nudge; panel width field exact-entry; class hotkey recolors.
4. Drag a joint — incident beams follow on release; orbit suppressed mid-drag.
5. Enter → beams apply; unconfirmed pointset rows appear in the Instances panel; the labeled points recolor; empty beam → "no points in beam box" toast.
6. Drag a joint, Enter again → same instance id re-applies (row count unchanged, points move).
7. Ctrl+Enter → confirm dialog → committed beams render faded; ◌/● toggle hides/shows them; active canvas cleared.
8. Ctrl+Z undoes an apply (points revert; row goes dormant via the existing reconciliation).
9. Reload the page → active graph + committed layer resume from `structure.json`.
10. Export wizard (scan resolution) on the session → beam-labeled points present in the output.
11. Delete the throwaway session.

- [ ] **Step 10.3:** Fix anything found (each fix: failing test where feasible → fix → green → commit).

---

### Task 11: Docs + finish

**Files:**
- Modify: `CLAUDE.md` (project)
- Modify: `docs/superpowers/specs/2026-07-10-beam-structure-labeling-design.md` (status header + as-built note)

- [ ] **Step 11.1: Update `CLAUDE.md`:**
  - "3-tool rail" → "4-tool rail"; add a **Beam** bullet next to the Draw bullet: node/edge graph, each edge → OBB via `apply-shape`, persists `sessions/<id>/structure.json` (`frontend/src/beam-mode.jsx`, `beam-graph.js`, `backend/labeling/beams.py`).
  - `materialize.py` bullet: box/tube → box/tube/**beam** boundaries exact (beam = OBB).
  - Remove/adjust the "`beam` reserved" phrasing in the apply-shape sentence.

- [ ] **Step 11.2: Update the spec** — set `Status: implemented 2026-07-11` and append a short "As built" section listing deviations 1–7 from this plan's reconciliation header.

- [ ] **Step 11.3:** Commit docs:

```bash
git add CLAUDE.md docs/superpowers/specs/2026-07-10-beam-structure-labeling-design.md
git commit -m "docs(beam): CLAUDE.md + spec status for the Beam tool"
```

- [ ] **Step 11.4:** REQUIRED SUB-SKILL: superpowers:finishing-a-development-branch — push, open PR to `main`.
