# Centerline Pipe/Tank Labeling Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A "Draw" sub-mode of Label mode where the user draws centerline paths along pipes/tanks and the backend labels all full-resolution points within a tube radius — per the approved spec `docs/superpowers/specs/2026-06-04-centerline-pipe-labeling-design.md`.

**Architecture:** Frontend owns the drawing UX (control points, tube preview, key/mouse map) via a pure-function state machine (`draw-paths.js`) plus a thin React component (`draw-mode.jsx`). Backend owns extraction: a new `POST /api/segment/centerline-apply` computes points-within-tube over the full-res cloud (vectorized NumPy) and applies through the existing `SegmentSession.apply_reassign` (undo/dirty/autosave free). Confirmed paths persist per-session in `sessions/<id>/centerlines.json`.

**Tech Stack:** FastAPI + NumPy (backend), React 18 + Three.js (frontend), pytest + vitest.

**Branch:** `centerline-labeling` (already created; spec committed). Worktree: `/home/hendrik/coding/engine/tools/labeling/voxa/.claude/worktrees/preseg-query-v2`. **Do not touch** the pre-existing unrelated dirty files (`backend/app/constants.py`, `backend/preseg/resolver.py`, `backend/preseg/sam3_features.py`, `backend/routes/meta.py`, `config/classes.yaml`) — never `git add -A`; always add files explicitly.

**Test commands:**
- Backend single file: `.venv/bin/pytest backend/tests/test_centerline.py -v` (run from repo root)
- Backend all: `npm run test:backend`
- Frontend: `cd frontend && npx vitest run src/draw-paths.test.js` / all: `npm run test:frontend`

**Conventions you must follow (from the codebase):**
- Backend routes use `from app.constants import *` / `app.schemas` / `app.core` star-imports; helpers like `_require_seg()`, `_serialize_apply()`, `_coerce_class_id()` come from `app/core.py`.
- `target_class` arrives as a class **name string** (`'pipe'`) or int palette index; always pass through `_coerce_class_id`.
- Class palette index == position in the configured class list. Frontend `classes[i]` ↔ backend int8 class id `i`.
- Frontend: no TypeScript, no jsdom — vitest tests are pure-function only. Comments explain *why*, matching file density.
- Coordinates from the viewer are in the recentered frame, which is exactly the frame of `seg.positions` — no conversion needed.

## File structure

| File | Status | Responsibility |
|---|---|---|
| `backend/labeling/centerline.py` | create | Tube math (point-to-segment distance, Catmull-Rom sampling) + `centerlines.json` store |
| `backend/app/schemas.py` | modify | `CenterlinePath`, `CenterlineApplyRequest` |
| `backend/routes/segment.py` | modify | `POST /api/segment/centerline-apply`, `GET /api/segment/centerlines` |
| `backend/tests/test_centerline.py` | create | Tube math + sampling + store unit tests |
| `backend/tests/test_centerline_endpoints.py` | create | Route tests via existing fixtures |
| `frontend/src/draw-paths.js` | create | Pure path-state machine: draw/edit/select/merge/apply-call building |
| `frontend/src/draw-paths.test.js` | create | State machine tests |
| `frontend/src/draw-mode.jsx` | create | DrawKeys (capture-phase), HUD, side-panel section, THREE overlay + pointer interactions, apply wiring |
| `frontend/src/viewer.jsx` | modify | Three small imperative additions: `setOrbitEnabled`, `attachOverlayGroup`, `getCamera` |
| `frontend/src/api.js` | modify | `centerlineApply`, `getCenterlines` |
| `frontend/src/api.test.js` | modify | Payload-shape tests for the two new calls |
| `frontend/src/mode-label.jsx` | modify | Draw toggle button, mutual exclusion with fastMode, hotkey suppression, `<DrawMode>` mount |
| `CLAUDE.md`, `docs/scan-schema.md` | modify | Document sub-mode + `centerlines.json` (final task) |

---

### Task 1: Backend tube math (`points_in_tube` / `tube_indices`)

**Files:**
- Create: `backend/labeling/centerline.py`
- Create: `backend/tests/test_centerline.py`

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for labeling.centerline — tube extraction + path sampling + store."""
from __future__ import annotations

import numpy as np
import pytest

from labeling.centerline import tube_indices


def _cylinder_cloud(axis_len=2.0, radius=0.1, n=500, seed=0):
    """Synthetic pipe along +X: points on a cylinder surface around the X axis."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(0.0, axis_len, n)
    theta = rng.uniform(0, 2 * np.pi, n)
    y = radius * np.cos(theta)
    z = radius * np.sin(theta)
    return np.column_stack([x, y, z]).astype(np.float32)


def test_tube_captures_cylinder_points():
    pts = _cylinder_cloud(radius=0.1)
    # Outlier far away must not be captured.
    cloud = np.vstack([pts, [[5.0, 5.0, 5.0]]]).astype(np.float32)
    paths = [{"points": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], "radius": 0.15, "smooth": False}]
    idx = tube_indices(cloud, paths)
    assert set(idx.tolist()) == set(range(len(pts)))


def test_tube_excludes_points_outside_radius():
    pts = _cylinder_cloud(radius=0.3)   # all at distance 0.3 from the axis
    paths = [{"points": [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], "radius": 0.15, "smooth": False}]
    idx = tube_indices(pts, paths)
    assert idx.size == 0


def test_multi_segment_elbow():
    # L-shaped path: along +X then turning to +Y. One point near each leg,
    # one near the joint, one in the "inside corner" beyond the radius.
    cloud = np.array([
        [1.0, 0.05, 0.0],    # near leg 1
        [2.0, 1.0, 0.05],    # near leg 2
        [2.0, 0.02, 0.0],    # at the joint
        [1.0, 1.0, 0.0],     # inside corner, far from both segments
    ], dtype=np.float32)
    paths = [{"points": [[0, 0, 0], [2, 0, 0], [2, 2, 0]], "radius": 0.1, "smooth": False}]
    idx = tube_indices(cloud, paths)
    assert set(idx.tolist()) == {0, 1, 2}


def test_multiple_paths_union_unique():
    cloud = np.array([[0.5, 0, 0], [0.5, 1, 0], [0.5, 0.02, 0]], dtype=np.float32)
    paths = [
        {"points": [[0, 0, 0], [1, 0, 0]], "radius": 0.1, "smooth": False},
        {"points": [[0, 1, 0], [1, 1, 0]], "radius": 0.1, "smooth": False},
        # Overlapping with path 1 — union must stay unique.
        {"points": [[0, 0, 0], [1, 0, 0]], "radius": 0.05, "smooth": False},
    ]
    idx = tube_indices(cloud, paths)
    assert sorted(idx.tolist()) == [0, 1, 2]
    assert idx.dtype == np.int32


def test_per_path_radius_respected():
    cloud = np.array([[0.5, 0.2, 0.0]], dtype=np.float32)
    thin = [{"points": [[0, 0, 0], [1, 0, 0]], "radius": 0.1, "smooth": False}]
    thick = [{"points": [[0, 0, 0], [1, 0, 0]], "radius": 0.3, "smooth": False}]
    assert tube_indices(cloud, thin).size == 0
    assert tube_indices(cloud, thick).size == 1


def test_degenerate_zero_length_segment():
    # Two identical control points → segment degenerates to a sphere test.
    cloud = np.array([[0.0, 0.05, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    paths = [{"points": [[0, 0, 0], [0, 0, 0]], "radius": 0.1, "smooth": False}]
    idx = tube_indices(cloud, paths)
    assert idx.tolist() == [0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_centerline.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'labeling.centerline'`

- [ ] **Step 3: Write the implementation**

```python
"""Centerline tube extraction + per-session centerline persistence.

A "path" is dict {points: [[x,y,z],...], radius: float, smooth: bool} in the
recentered frame (same frame as SegmentSession.positions). Extraction is the
union over paths of all points within `radius` of the polyline — the implied
tube is segment cylinders plus spheres at the joints (distance-to-segment
metric), per the design spec.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CENTERLINES_FILENAME = "centerlines.json"


def _segment_mask(positions: np.ndarray, a: np.ndarray, b: np.ndarray,
                  r2: float) -> np.ndarray:
    """Bool mask: squared distance from each point to segment a→b ≤ r2."""
    d = b - a
    l2 = float(d @ d)
    if l2 < 1e-12:                      # degenerate segment → sphere test
        diff = positions - a
        return np.einsum("ij,ij->i", diff, diff) <= r2
    t = np.clip((positions - a) @ d / l2, 0.0, 1.0)
    closest = a + t[:, None] * d
    diff = positions - closest
    return np.einsum("ij,ij->i", diff, diff) <= r2


def tube_indices(positions: np.ndarray, paths: list[dict]) -> np.ndarray:
    """Unique int32 indices of points within any path's tube."""
    positions = np.asarray(positions, dtype=np.float32)
    mask = np.zeros(positions.shape[0], dtype=bool)
    for p in paths:
        pts = np.asarray(sample_path(p), dtype=np.float32)
        r2 = float(p["radius"]) ** 2
        for i in range(len(pts) - 1):
            mask |= _segment_mask(positions, pts[i], pts[i + 1], r2)
    return np.flatnonzero(mask).astype(np.int32)
```

For Step 3 only, add a temporary passthrough so Task 1 is self-contained (replaced in Task 2):

```python
def sample_path(path: dict) -> np.ndarray:
    return np.asarray(path["points"], dtype=np.float32)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest backend/tests/test_centerline.py -v`
Expected: 6 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/labeling/centerline.py backend/tests/test_centerline.py
git commit -m "feat(backend): centerline tube extraction (point-to-segment union)"
```

---

### Task 2: Catmull-Rom sampling for smooth paths

**Files:**
- Modify: `backend/labeling/centerline.py` (replace the `sample_path` stub)
- Modify: `backend/tests/test_centerline.py` (append tests)

- [ ] **Step 1: Write the failing tests** (append to `test_centerline.py`)

```python
from labeling.centerline import sample_path


def test_sample_path_straight_passthrough():
    p = {"points": [[0, 0, 0], [1, 0, 0], [1, 1, 0]], "radius": 0.1, "smooth": False}
    out = sample_path(p)
    assert np.allclose(out, p["points"])


def test_sample_path_smooth_interpolates_through_controls():
    p = {"points": [[0, 0, 0], [1, 1, 0], [2, 0, 0]], "radius": 0.2, "smooth": True}
    out = sample_path(p)
    # Control points must lie on the sampled curve.
    for cp in p["points"]:
        d = np.linalg.norm(out - np.asarray(cp, dtype=np.float32), axis=1)
        assert d.min() < 1e-5
    # Densified: more samples than control points. Chords must stay below
    # the tube radius so the segment-distance test can't visibly cut corners.
    # (Uniform-parameter sampling overshoots step≈radius/2 near the apex —
    # the guarantee that matters is "< radius", not "< step".)
    assert len(out) > 3
    chords = np.linalg.norm(np.diff(out, axis=0), axis=1)
    assert chords.max() <= 0.2


def test_sample_path_smooth_two_points_is_segment():
    p = {"points": [[0, 0, 0], [1, 0, 0]], "radius": 0.1, "smooth": True}
    out = sample_path(p)
    # Degenerates to a straight chordal run along the segment.
    assert np.allclose(out[0], [0, 0, 0]) and np.allclose(out[-1], [1, 0, 0])
    assert np.allclose(out[:, 1:], 0, atol=1e-6)


def test_smooth_tube_captures_curve_apex():
    # Point at the apex of the arc through (0,0),(1,1),(2,0) — outside the
    # straight-polyline corner-cutting chord, inside the smooth tube.
    cloud = np.array([[1.0, 1.05, 0.0]], dtype=np.float32)
    smooth = [{"points": [[0, 0, 0], [1, 1, 0], [2, 0, 0]], "radius": 0.15, "smooth": True}]
    assert tube_indices(cloud, smooth).size == 1
```

- [ ] **Step 2: Run tests to verify the new ones fail**

Run: `.venv/bin/pytest backend/tests/test_centerline.py -v -k smooth`
Expected: smooth tests FAIL (stub ignores `smooth`); straight-passthrough passes.

- [ ] **Step 3: Replace the `sample_path` stub**

```python
def sample_path(path: dict) -> np.ndarray:
    """Control points → polyline chords. Straight paths pass through
    unchanged; smooth paths get Catmull-Rom sampling with target step
    ≈ radius/2 (worst-case chord stays < radius near apexes), so the tube
    test on chords can't visibly cut corners."""
    pts = np.asarray(path["points"], dtype=np.float32)
    if not path.get("smooth") or len(pts) < 3:
        return pts
    step = max(float(path["radius"]) / 2.0, 1e-4)
    # Endpoint-duplicated control polygon so the curve spans all controls.
    ctrl = np.vstack([pts[0], pts, pts[-1]])
    out = [pts[0]]
    for i in range(1, len(ctrl) - 2):
        p0, p1, p2, p3 = ctrl[i - 1], ctrl[i], ctrl[i + 1], ctrl[i + 2]
        seg_len = float(np.linalg.norm(p2 - p1))
        n = max(int(np.ceil(seg_len / step)), 1)
        for t in np.linspace(0, 1, n + 1)[1:]:
            t2, t3 = t * t, t * t * t
            v = (0.5 * ((2 * p1) + (-p0 + p2) * t
                 + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t2
                 + (-p0 + 3 * p1 - 3 * p2 + p3) * t3))
            out.append(v.astype(np.float32))
    return np.asarray(out, dtype=np.float32)
```

- [ ] **Step 4: Run the full file**

Run: `.venv/bin/pytest backend/tests/test_centerline.py -v`
Expected: 10 PASS. (If `test_smooth_tube_captures_curve_apex` fails on the apex margin, the uniform Catmull-Rom apex for these controls is y=1.0 at t=0.5 — distance 0.05 from the test point — well inside radius 0.15; debug rather than widen the radius.)

- [ ] **Step 5: Commit**

```bash
git add backend/labeling/centerline.py backend/tests/test_centerline.py
git commit -m "feat(backend): Catmull-Rom sampling for smooth centerline paths"
```

---

### Task 3: `centerlines.json` store (load / replace / append / delete)

**Files:**
- Modify: `backend/labeling/centerline.py`
- Modify: `backend/tests/test_centerline.py` (append)

- [ ] **Step 1: Write the failing tests** (append)

```python
from labeling.centerline import load_centerlines, update_centerlines


def _path(x=0.0, radius=0.15):
    return {"points": [[x, 0, 0], [x + 1, 0, 0]], "radius": radius, "smooth": False}


def test_store_roundtrip_append_and_load(tmp_path):
    assert load_centerlines(tmp_path) == {"paths": []}
    update_centerlines(tmp_path, instance_id=7, class_id=0, paths=[_path()], merged_from=[])
    doc = load_centerlines(tmp_path)
    assert len(doc["paths"]) == 1
    assert doc["paths"][0]["instance_id"] == 7
    assert doc["paths"][0]["class_id"] == 0
    assert doc["paths"][0]["radius"] == 0.15


def test_store_reapply_replaces_by_instance_id(tmp_path):
    update_centerlines(tmp_path, 7, 0, [_path(0.0), _path(5.0)], [])
    # Re-apply instance 7 with ONE edited path → both old entries replaced.
    update_centerlines(tmp_path, 7, 2, [_path(0.0, radius=0.3)], [])
    doc = load_centerlines(tmp_path)
    assert len(doc["paths"]) == 1
    assert doc["paths"][0]["radius"] == 0.3
    assert doc["paths"][0]["class_id"] == 2


def test_store_merged_from_deletes_absorbed_entries(tmp_path):
    update_centerlines(tmp_path, 7, 0, [_path(0.0)], [])
    update_centerlines(tmp_path, 9, 0, [_path(5.0)], [])
    # Merge 9 into 7: union of paths applied under 7, 9 absorbed.
    update_centerlines(tmp_path, 7, 0, [_path(0.0), _path(5.0)], merged_from=[9])
    doc = load_centerlines(tmp_path)
    assert sorted(p["instance_id"] for p in doc["paths"]) == [7, 7]


def test_store_distinct_instances_append(tmp_path):
    update_centerlines(tmp_path, 7, 0, [_path(0.0)], [])
    update_centerlines(tmp_path, 9, 1, [_path(5.0)], [])
    doc = load_centerlines(tmp_path)
    assert sorted(p["instance_id"] for p in doc["paths"]) == [7, 9]
```

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest backend/tests/test_centerline.py -v -k store`
Expected: FAIL — `ImportError: cannot import name 'load_centerlines'`

- [ ] **Step 3: Implement** (append to `centerline.py`)

```python
def load_centerlines(session_dir: Path) -> dict:
    f = Path(session_dir) / CENTERLINES_FILENAME
    if not f.exists():
        return {"paths": []}
    return json.loads(f.read_text())


def update_centerlines(session_dir: Path, instance_id: int, class_id: int,
                       paths: list[dict], merged_from: list[int]) -> dict:
    """Replace-by-instance_id write: drop stored paths for `instance_id` and
    any id in `merged_from`, then append the new ones. Keeps re-editing a
    pipe from duplicating stored paths (spec: Persistence)."""
    doc = load_centerlines(session_dir)
    dead = {int(instance_id), *(int(m) for m in merged_from)}
    kept = [p for p in doc["paths"] if p.get("instance_id") not in dead]
    for p in paths:
        kept.append({
            "points": [[float(c) for c in pt] for pt in p["points"]],
            "radius": float(p["radius"]),
            "smooth": bool(p.get("smooth", False)),
            "class_id": int(class_id),
            "instance_id": int(instance_id),
        })
    doc = {"paths": kept}
    f = Path(session_dir) / CENTERLINES_FILENAME
    f.write_text(json.dumps(doc, indent=1))
    return doc
```

- [ ] **Step 4: Run full file**

Run: `.venv/bin/pytest backend/tests/test_centerline.py -v`
Expected: 14 PASS

- [ ] **Step 5: Commit**

```bash
git add backend/labeling/centerline.py backend/tests/test_centerline.py
git commit -m "feat(backend): per-session centerlines.json store with replace/merge semantics"
```

---

### Task 4: Schemas + `POST /api/segment/centerline-apply`

**Files:**
- Modify: `backend/app/schemas.py` (after `ApplyRequest`, ~line 138)
- Modify: `backend/routes/segment.py`
- Create: `backend/tests/test_centerline_endpoints.py`

Existing fixtures to reuse (see `backend/tests/conftest.py`): `client_with_loaded_annotated_scene` gives a TestClient with an annotated scene loaded and an active session (so `_require_seg()` passes and `seg.session_dir` is set), and `scan_dir_for_loaded_scene` gives the scan dir. **Read `conftest.py` first** to confirm fixture names/cloud size before writing tests; adapt the capture-everything trick below to the fixture cloud's actual bounds (a huge radius makes it bound-independent).

- [ ] **Step 1: Write the failing tests**

```python
"""Tests for /api/segment/centerline-* endpoints."""
from __future__ import annotations

import json


def _apply_body(**over):
    body = {
        "paths": [{"points": [[-1e5, -1e5, -1e5], [1e5, 1e5, 1e5]],
                   "radius": 1e6, "smooth": False}],   # tube swallows the whole cloud
        "target_class": "pipe",
        "target_inst": -1,
        "merged_from": [],
    }
    body.update(over)
    return body


def test_centerline_apply_labels_points_and_persists(
        client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/centerline-apply", json=_apply_body())
    assert r.status_code == 200
    j = r.json()
    assert j["n_affected"] > 0
    assert j["instance_id"] == j["new_instance_id"]
    # Stored in the active session's centerlines.json.
    sessions = list((scan_dir_for_loaded_scene / "sessions").iterdir())
    docs = [d / "centerlines.json" for d in sessions if (d / "centerlines.json").exists()]
    assert len(docs) == 1
    doc = json.loads(docs[0].read_text())
    assert len(doc["paths"]) == 1
    assert doc["paths"][0]["instance_id"] == j["instance_id"]


def test_centerline_apply_empty_tube_returns_zero_no_store(
        client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    body = _apply_body(paths=[{"points": [[9e6, 9e6, 9e6], [9e6 + 1, 9e6, 9e6]],
                               "radius": 0.01, "smooth": False}])
    r = client.post("/api/segment/centerline-apply", json=body)
    assert r.status_code == 200
    j = r.json()
    assert j["n_affected"] == 0
    assert "after_class" not in j          # spec: keys absent on empty delta
    assert "instance_id" not in j
    files = list((scan_dir_for_loaded_scene / "sessions").rglob("centerlines.json"))
    assert files == []                     # nothing persisted


def test_centerline_apply_undo_reverts_labels(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    before = client.get("/api/segment/state").json()
    r = client.post("/api/segment/centerline-apply", json=_apply_body())
    assert r.json()["n_affected"] > 0
    u = client.post("/api/segment/undo")
    assert u.status_code == 200
    after = client.get("/api/segment/state").json()
    assert after["n_assigned"] == before["n_assigned"]


def test_centerline_apply_reapply_same_instance_replaces(
        client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    j1 = client.post("/api/segment/centerline-apply", json=_apply_body()).json()
    inst = j1["instance_id"]
    j2 = client.post("/api/segment/centerline-apply",
                     json=_apply_body(target_inst=inst)).json()
    assert "new_instance_id" not in j2     # reused, not allocated
    assert j2["instance_id"] == inst
    doc = json.loads(next((scan_dir_for_loaded_scene / "sessions")
                          .rglob("centerlines.json")).read_text())
    assert len(doc["paths"]) == 1          # replaced, not appended


def test_centerline_apply_validation_errors(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    # < 2 points
    r = client.post("/api/segment/centerline-apply", json=_apply_body(
        paths=[{"points": [[0, 0, 0]], "radius": 0.1, "smooth": False}]))
    assert r.status_code == 422
    # radius ≤ 0
    r = client.post("/api/segment/centerline-apply", json=_apply_body(
        paths=[{"points": [[0, 0, 0], [1, 0, 0]], "radius": 0, "smooth": False}]))
    assert r.status_code == 422
    # unknown class name
    r = client.post("/api/segment/centerline-apply", json=_apply_body(
        target_class="not-a-class"))
    assert r.status_code == 400


def test_centerline_apply_409_without_session(client):
    # Plain client: nothing loaded.
    r = client.post("/api/segment/centerline-apply", json=_apply_body())
    assert r.status_code == 409
```

(If conftest has no bare `client` fixture, check what `test_segment_endpoints.py`-adjacent tests use for the no-session case and reuse that; if none exists, drop the last test's fixture down to constructing a TestClient the same way conftest does.)

- [ ] **Step 2: Run to verify failure**

Run: `.venv/bin/pytest backend/tests/test_centerline_endpoints.py -v`
Expected: FAIL — 404s (route doesn't exist)

- [ ] **Step 3: Add schemas** (in `backend/app/schemas.py`, after `ApplyRequest`)

```python
class CenterlinePath(BaseModel):
    points: list[list[float]] = Field(min_length=2)
    radius: float = Field(gt=0)
    smooth: bool = False

    @field_validator("points")
    @classmethod
    def _points_are_3d(cls, v):
        if any(len(p) != 3 for p in v):
            raise ValueError("each point must be [x, y, z]")
        return v

class CenterlineApplyRequest(BaseModel):
    paths: list[CenterlinePath] = Field(min_length=1)
    target_class: int | str
    target_inst: int = -1
    merged_from: list[int] = []
```

Check the top of `schemas.py` for existing imports — add `Field`, `field_validator` to the `pydantic` import if missing.

- [ ] **Step 4: Add the route** (in `backend/routes/segment.py`)

```python
@router.post("/api/segment/centerline-apply")
def centerline_apply(req: CenterlineApplyRequest):
    """Label all full-res points within the tube(s) around the given
    centerline paths. Multiple paths in one call = one (merged) instance.
    See docs/superpowers/specs/2026-06-04-centerline-pipe-labeling-design.md."""
    from labeling.centerline import tube_indices, update_centerlines
    seg = _require_seg()
    if seg.session_dir is None:
        raise HTTPException(409, "centerline labeling requires an active session")
    try:
        target_class = _coerce_class_id(req.target_class)
    except ValueError as e:
        raise HTTPException(400, str(e))
    paths = [p.model_dump() for p in req.paths]
    idx = tube_indices(np.asarray(seg.positions), paths)
    if idx.size == 0:
        # Same key-absence contract as _serialize_apply on an empty delta.
        return {"op": "centerline", "n_affected": 0, "dirty": bool(seg.dirty)}
    out = seg.apply_reassign(idx, target_inst=req.target_inst, target_class=target_class)
    instance_id = out.get("new_instance_id", req.target_inst)
    update_centerlines(seg.session_dir, instance_id, target_class, paths,
                       req.merged_from)
    body = _serialize_apply(out)
    body["instance_id"] = int(instance_id)
    return body
```

- [ ] **Step 5: Run tests**

Run: `.venv/bin/pytest backend/tests/test_centerline_endpoints.py backend/tests/test_segment_endpoints.py -v`
Expected: all PASS (segment endpoint tests prove no regression)

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas.py backend/routes/segment.py backend/tests/test_centerline_endpoints.py
git commit -m "feat(backend): POST /api/segment/centerline-apply"
```

---

### Task 5: `GET /api/segment/centerlines`

**Files:**
- Modify: `backend/routes/segment.py`
- Modify: `backend/tests/test_centerline_endpoints.py` (append)

- [ ] **Step 1: Failing tests** (append)

```python
def test_get_centerlines_empty_then_populated(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.get("/api/segment/centerlines")
    assert r.status_code == 200
    assert r.json() == {"paths": []}
    client.post("/api/segment/centerline-apply", json=_apply_body())
    j = client.get("/api/segment/centerlines").json()
    assert len(j["paths"]) == 1
    assert {"points", "radius", "smooth", "class_id", "instance_id"} <= set(j["paths"][0])


def test_get_centerlines_409_without_session(client):
    r = client.get("/api/segment/centerlines")
    assert r.status_code == 409
```

- [ ] **Step 2: Run** — expected 404 FAIL.

- [ ] **Step 3: Implement** (in `routes/segment.py`)

```python
@router.get("/api/segment/centerlines")
def get_centerlines():
    """Stored centerline paths for the active session (Draw sub-mode resume)."""
    from labeling.centerline import load_centerlines
    seg = _require_seg()
    if seg.session_dir is None:
        raise HTTPException(409, "no active session")
    return load_centerlines(seg.session_dir)
```

- [ ] **Step 4: Run** `.venv/bin/pytest backend/tests/test_centerline_endpoints.py -v` — all PASS.

- [ ] **Step 5: Run the whole backend suite**

Run: `npm run test:backend`
Expected: PASS (no regressions)

- [ ] **Step 6: Commit**

```bash
git add backend/routes/segment.py backend/tests/test_centerline_endpoints.py
git commit -m "feat(backend): GET /api/segment/centerlines"
```

---

### Task 6: `api.js` client calls

**Files:**
- Modify: `frontend/src/api.js` (inside `VoxaAPI`, after `segState`)
- Modify: `frontend/src/api.test.js`

- [ ] **Step 1: Failing tests** — `api.test.js` mocks `fetch` via `vi.stubGlobal('fetch', vi.fn(...))`; follow that existing pattern (the `vi.spyOn` form below works too, but prefer matching the file — adapt the snippets accordingly). Append:

```javascript
describe('centerline API', () => {
  afterEach(() => vi.restoreAllMocks());

  it('centerlineApply posts snake_case payload and decodes the delta', async () => {
    const fetchSpy = vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({ op: 'reassign', n_affected: 0, dirty: true }),
    });
    await VoxaAPI.centerlineApply({
      paths: [{ points: [[0, 0, 0], [1, 0, 0]], radius: 0.15, smooth: false }],
      targetClass: 'pipe', targetInst: -1, mergedFrom: [4],
    });
    const [url, opts] = fetchSpy.mock.calls[0];
    expect(url).toBe('/api/segment/centerline-apply');
    const body = JSON.parse(opts.body);
    expect(body.target_class).toBe('pipe');
    expect(body.target_inst).toBe(-1);
    expect(body.merged_from).toEqual([4]);
    expect(body.paths[0].radius).toBe(0.15);
  });

  it('centerlineApply surfaces instance_id on the decoded response', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true,
      json: async () => ({ op: 'reassign', n_affected: 3, dirty: true,
                           new_instance_id: 9, instance_id: 9 }),
    });
    const r = await VoxaAPI.centerlineApply({ paths: [], targetClass: 0 });
    expect(r.instanceId).toBe(9);
    expect(r.nAffected).toBe(3);
  });

  it('getCenterlines returns the stored paths', async () => {
    vi.spyOn(globalThis, 'fetch').mockResolvedValue({
      ok: true, json: async () => ({ paths: [{ instance_id: 7 }] }),
    });
    const j = await VoxaAPI.getCenterlines();
    expect(j.paths).toHaveLength(1);
  });
});
```

- [ ] **Step 2: Run** `cd frontend && npx vitest run src/api.test.js` — new tests FAIL.

- [ ] **Step 3: Implement** (in `VoxaAPI`)

```javascript
  async centerlineApply({ paths, targetClass, targetInst = -1, mergedFrom = [] }) {
    const body = {
      paths,
      target_class: targetClass,
      target_inst: targetInst,
      merged_from: mergedFrom,
    };
    const r = await fetch('/api/segment/centerline-apply', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`centerlineApply failed: ${r.status} ${await r.text()}`);
    const j = await r.json();
    return { ..._decodeApplyResponse(j), instanceId: j.instance_id ?? null };
  },
  async getCenterlines() {
    const r = await fetch('/api/segment/centerlines');
    if (!r.ok) throw new Error(`getCenterlines failed: ${r.status} ${await r.text()}`);
    return r.json();
  },
```

- [ ] **Step 4: Run** `npx vitest run src/api.test.js` — PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api.js frontend/src/api.test.js
git commit -m "feat(frontend): centerlineApply + getCenterlines API calls"
```

---

### Task 7: `draw-paths.js` — drawing core

The state machine is pure and immutable-ish (top-level object spread, arrays copied on write) so React `useState` can hold it.

**State shape (document at the top of the file):**

```javascript
// {
//   paths: [{ key, points: [[x,y,z],...], radius, smooth, classId, instKey }],
//   active: string|null,          // key of the path being drawn
//   selection: Set<string>,       // selected path keys
//   instanceIds: { [instKey]: number },        // backend ids, applied groups only
//   pendingMergedFrom: { [instKey]: number[] },// absorbed backend ids awaiting next apply
//   lastRadius: number,
//   nextKey: number,              // monotonic key source (deterministic, testable)
// }
// classId is the int palette index (== position in the classes array).
// instKey groups paths into one instance; starts unique per path, M merges.
```

**Files:**
- Create: `frontend/src/draw-paths.js`
- Create: `frontend/src/draw-paths.test.js`

- [ ] **Step 1: Failing tests**

```javascript
import { describe, expect, it } from 'vitest';
import {
  initDrawState, addPoint, movePoint, removeLastPoint, endActive,
} from './draw-paths.js';

const P = (x = 0) => [x, 0, 0];

describe('drawing core', () => {
  it('first addPoint starts a path; subsequent ones chain', () => {
    let s = initDrawState({ defaultRadius: 0.2 });
    s = addPoint(s, P(0), 1);
    expect(s.active).not.toBeNull();
    expect(s.paths).toHaveLength(1);
    expect(s.paths[0].classId).toBe(1);
    expect(s.paths[0].radius).toBe(0.2);
    s = addPoint(s, P(1), 1);
    expect(s.paths).toHaveLength(1);
    expect(s.paths[0].points).toEqual([P(0), P(1)]);
  });

  it('movePoint replaces a control point', () => {
    let s = addPoint(initDrawState(), P(0), 0);
    s = addPoint(s, P(1), 0);
    s = movePoint(s, s.paths[0].key, 1, [1, 2, 3]);
    expect(s.paths[0].points[1]).toEqual([1, 2, 3]);
  });

  it('removeLastPoint pops; removing the only point discards the path', () => {
    let s = addPoint(initDrawState(), P(0), 0);
    s = addPoint(s, P(1), 0);
    s = removeLastPoint(s);
    expect(s.paths[0].points).toEqual([P(0)]);
    s = removeLastPoint(s);
    expect(s.paths).toHaveLength(0);
    expect(s.active).toBeNull();
  });

  it('endActive stages a valid path, discards a 1-point path', () => {
    let s = addPoint(initDrawState(), P(0), 0);
    s = endActive(s);                       // 1 point → discard
    expect(s.paths).toHaveLength(0);
    s = addPoint(s, P(0), 0);
    s = addPoint(s, P(1), 0);
    s = endActive(s);
    expect(s.active).toBeNull();
    expect(s.paths).toHaveLength(1);
  });

  it('new path after endActive reuses lastRadius and gets a fresh instKey', () => {
    let s = addPoint(initDrawState({ defaultRadius: 0.1 }), P(0), 0);
    s = addPoint(s, P(1), 0);
    s = endActive(s);
    s = addPoint(s, P(5), 0);
    expect(s.paths[1].radius).toBe(0.1);
    expect(s.paths[1].instKey).not.toBe(s.paths[0].instKey);
  });
});
```

- [ ] **Step 2: Run** `npx vitest run src/draw-paths.test.js` — FAIL (module missing).

- [ ] **Step 3: Implement**

```javascript
// draw-paths.js — pure state machine for the Draw (centerline) sub-mode.
// All functions take state and return a new state; React owns the object.
// See docs/superpowers/specs/2026-06-04-centerline-pipe-labeling-design.md.

export function initDrawState({ defaultRadius = 0.15 } = {}) {
  return {
    paths: [],
    active: null,
    selection: new Set(),
    instanceIds: {},
    pendingMergedFrom: {},
    lastRadius: defaultRadius,
    nextKey: 1,
  };
}

function freshKey(s) {
  return [`p${s.nextKey}`, { ...s, nextKey: s.nextKey + 1 }];
}

export function addPoint(state, xyz, classId) {
  if (state.active) {
    const paths = state.paths.map((p) =>
      p.key === state.active ? { ...p, points: [...p.points, xyz] } : p);
    return { ...state, paths };
  }
  const [key, s] = freshKey(state);
  const path = {
    key,
    points: [xyz],
    radius: s.lastRadius,
    smooth: false,
    classId,
    instKey: key,        // unique until merged
  };
  return { ...s, paths: [...s.paths, path], active: key, selection: new Set() };
}

export function movePoint(state, pathKey, pointIdx, xyz) {
  const paths = state.paths.map((p) => {
    if (p.key !== pathKey) return p;
    const points = p.points.slice();
    points[pointIdx] = xyz;
    return { ...p, points };
  });
  return { ...state, paths };
}

export function removeLastPoint(state) {
  if (!state.active) return state;
  const p = state.paths.find((x) => x.key === state.active);
  if (p.points.length <= 1) {
    return {
      ...state,
      paths: state.paths.filter((x) => x.key !== state.active),
      active: null,
    };
  }
  return movePointCount(state, p);
}

function movePointCount(state, p) {
  const paths = state.paths.map((x) =>
    x.key === p.key ? { ...x, points: x.points.slice(0, -1) } : x);
  return { ...state, paths };
}

export function endActive(state) {
  if (!state.active) return state;
  const p = state.paths.find((x) => x.key === state.active);
  if (p.points.length < 2) {
    // Can't confirm < 2 points (spec: Error handling) — discard.
    return { ...state, paths: state.paths.filter((x) => x.key !== p.key), active: null };
  }
  return { ...state, active: null };
}
```

- [ ] **Step 4: Run** — PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/draw-paths.js frontend/src/draw-paths.test.js
git commit -m "feat(frontend): draw-paths state machine — drawing core"
```

---

### Task 8: `draw-paths.js` — selection, radius, class, smooth, delete

**Files:** modify both files from Task 7.

- [ ] **Step 1: Failing tests** (append; build a helper that stages two paths)

```javascript
import {
  selectPath, clearSelection, setRadius, nudgeRadius, setClass,
  toggleSmooth, deleteSelected,
} from './draw-paths.js';

function staged2() {
  let s = initDrawState();
  s = addPoint(s, P(0), 0); s = addPoint(s, P(1), 0); s = endActive(s);
  s = addPoint(s, P(5), 1); s = addPoint(s, P(6), 1); s = endActive(s);
  return s;
}

describe('selection + path edits', () => {
  it('selectPath replaces; additive=true toggles', () => {
    let s = staged2();
    const [a, b] = s.paths.map((p) => p.key);
    s = selectPath(s, a);
    expect([...s.selection]).toEqual([a]);
    s = selectPath(s, b, { additive: true });
    expect(s.selection.size).toBe(2);
    s = selectPath(s, b, { additive: true });   // shift-click again removes
    expect([...s.selection]).toEqual([a]);
    s = selectPath(s, b);                       // plain click replaces
    expect([...s.selection]).toEqual([b]);
    expect(clearSelection(s).selection.size).toBe(0);
  });

  it('setRadius / nudgeRadius hit selected paths (or active) and update lastRadius', () => {
    let s = staged2();
    const a = s.paths[0].key;
    s = selectPath(s, a);
    s = setRadius(s, 0.5);
    expect(s.paths[0].radius).toBe(0.5);
    expect(s.paths[1].radius).not.toBe(0.5);
    expect(s.lastRadius).toBe(0.5);
    s = nudgeRadius(s, +1);
    expect(s.paths[0].radius).toBeGreaterThan(0.5);
    // While drawing, the active path is implicitly targeted.
    let d = addPoint(initDrawState({ defaultRadius: 0.2 }), P(0), 0);
    d = nudgeRadius(d, -1);
    expect(d.paths[0].radius).toBeLessThan(0.2);
    expect(d.paths[0].radius).toBeGreaterThan(0);   // floor > 0
  });

  it('setClass and toggleSmooth target the selection', () => {
    let s = staged2();
    s = selectPath(s, s.paths[0].key);
    s = setClass(s, 3);
    expect(s.paths[0].classId).toBe(3);
    expect(s.paths[1].classId).toBe(1);
    s = toggleSmooth(s);
    expect(s.paths[0].smooth).toBe(true);
  });

  it('deleteSelected removes staged paths and clears selection', () => {
    let s = staged2();
    s = selectPath(s, s.paths[0].key);
    s = deleteSelected(s);
    expect(s.paths).toHaveLength(1);
    expect(s.selection.size).toBe(0);
  });
});
```

- [ ] **Step 2: Run** — FAIL (missing exports).

- [ ] **Step 3: Implement** (append)

```javascript
export function selectPath(state, pathKey, { additive = false } = {}) {
  const selection = additive ? new Set(state.selection) : new Set();
  if (additive && selection.has(pathKey)) selection.delete(pathKey);
  else selection.add(pathKey);
  return { ...state, selection };
}

export function clearSelection(state) {
  return { ...state, selection: new Set() };
}

// Radius/class/smooth target the active path while drawing, else the selection.
function targetKeys(state) {
  if (state.active) return new Set([state.active]);
  return state.selection;
}

const MIN_RADIUS = 0.005;

export function setRadius(state, radius) {
  const r = Math.max(radius, MIN_RADIUS);
  const keys = targetKeys(state);
  if (keys.size === 0) return state;
  const paths = state.paths.map((p) => keys.has(p.key) ? { ...p, radius: r } : p);
  return { ...state, paths, lastRadius: r };
}

export function nudgeRadius(state, dir) {
  const keys = targetKeys(state);
  const first = state.paths.find((p) => keys.has(p.key));
  if (!first) return state;
  // Multiplicative steps feel uniform across pipe sizes (8% like orbit zoom).
  return setRadius(state, first.radius * (1 + Math.sign(dir) * 0.08));
}

export function setClass(state, classId) {
  const keys = targetKeys(state);
  if (keys.size === 0) return state;
  const paths = state.paths.map((p) => keys.has(p.key) ? { ...p, classId } : p);
  return { ...state, paths };
}

export function toggleSmooth(state) {
  const keys = targetKeys(state);
  if (keys.size === 0) return state;
  const anyOff = state.paths.some((p) => keys.has(p.key) && !p.smooth);
  const paths = state.paths.map((p) => keys.has(p.key) ? { ...p, smooth: anyOff } : p);
  return { ...state, paths };
}

export function deleteSelected(state) {
  const paths = state.paths.filter((p) => !state.selection.has(p.key));
  return { ...state, paths, selection: new Set() };
}
```

- [ ] **Step 4: Run** — PASS. **Step 5: Commit**

```bash
git add frontend/src/draw-paths.js frontend/src/draw-paths.test.js
git commit -m "feat(frontend): draw-paths selection + radius/class/smooth edits"
```

---

### Task 9: `draw-paths.js` — merge, apply-call building, server seeding

This encodes the spec's hardest semantics. Rules:
- **M (mergeSelection):** all selected paths adopt one survivor `instKey` and the survivor's `classId`. Survivor = the selected group with the **lowest applied instance id**, else the first selected path's group. Absorbed groups that were applied contribute their backend ids to `pendingMergedFrom[survivor]`.
- **Enter (buildApplyCalls):** expands the selection to whole `instKey` groups; one call per group: `{paths, targetClass, targetInst: instanceIds[instKey] ?? -1, mergedFrom: pendingMergedFrom[instKey] ?? []}`.
- **markApplied:** records the backend instance id for the group, clears its `pendingMergedFrom`.
- **seedFromServer:** rebuilds applied paths from `GET /api/segment/centerlines` (one `instKey` per `instance_id`).

- [ ] **Step 1: Failing tests** (append)

```javascript
import {
  mergeSelection, buildApplyCalls, markApplied, seedFromServer,
} from './draw-paths.js';

function applied(state, instKeyToId) {
  let s = state;
  for (const [k, id] of Object.entries(instKeyToId)) s = markApplied(s, k, id);
  return s;
}

describe('merge + apply calls', () => {
  it('buildApplyCalls: one call per selected instance group, whole-instance expansion', () => {
    let s = staged2();
    const [a, b] = s.paths.map((p) => p.key);
    s = selectPath(s, a);
    s = selectPath(s, b, { additive: true });
    // Unmerged → two independent calls (spec: M is the only merge trigger).
    const calls = buildApplyCalls(s);
    expect(calls).toHaveLength(2);
    expect(calls[0].targetInst).toBe(-1);
    expect(calls[0].mergedFrom).toEqual([]);
    expect(calls[0].paths[0].points).toEqual([P(0), P(1)]);
  });

  it('merge of two staged paths → one call, one shared class', () => {
    let s = staged2();
    const [a, b] = s.paths.map((p) => p.key);
    s = selectPath(s, a); s = selectPath(s, b, { additive: true });
    s = mergeSelection(s);
    expect(s.paths[0].instKey).toBe(s.paths[1].instKey);
    expect(s.paths[1].classId).toBe(s.paths[0].classId);   // survivor's class
    const calls = buildApplyCalls(s);
    expect(calls).toHaveLength(1);
    expect(calls[0].paths).toHaveLength(2);
  });

  it('selecting one path of a multi-path instance expands to all its paths', () => {
    let s = staged2();
    const [a, b] = s.paths.map((p) => p.key);
    s = selectPath(s, a); s = selectPath(s, b, { additive: true });
    s = mergeSelection(s);
    s = clearSelection(s);
    s = selectPath(s, a);                       // only one sibling selected
    const calls = buildApplyCalls(s);
    expect(calls[0].paths).toHaveLength(2);     // whole instance anyway
  });

  it('applied-applied merge: lowest id survives, others go to mergedFrom', () => {
    let s = staged2();
    const [a, b] = s.paths.map((p) => p.key);
    s = applied(s, { [a]: 9, [b]: 4 });
    s = selectPath(s, a); s = selectPath(s, b, { additive: true });
    s = mergeSelection(s);
    const calls = buildApplyCalls(s);
    expect(calls).toHaveLength(1);
    expect(calls[0].targetInst).toBe(4);        // lowest survives
    expect(calls[0].mergedFrom).toEqual([9]);
  });

  it('markApplied records id and clears pendingMergedFrom', () => {
    let s = staged2();
    const [a, b] = s.paths.map((p) => p.key);
    s = applied(s, { [a]: 9, [b]: 4 });
    s = selectPath(s, a); s = selectPath(s, b, { additive: true });
    s = mergeSelection(s);
    const survivor = s.paths[0].instKey;
    s = markApplied(s, survivor, 4);
    expect(s.instanceIds[survivor]).toBe(4);
    expect(s.pendingMergedFrom[survivor] ?? []).toEqual([]);
    // Re-apply after edit: same target, no mergedFrom this time.
    s = selectPath(clearSelection(s), s.paths[0].key);
    const again = buildApplyCalls(s);
    expect(again[0].targetInst).toBe(4);
    expect(again[0].mergedFrom).toEqual([]);
  });

  it('staged + applied merge adopts the applied id', () => {
    let s = staged2();
    const [a, b] = s.paths.map((p) => p.key);
    s = applied(s, { [a]: 7 });
    s = selectPath(s, a); s = selectPath(s, b, { additive: true });
    s = mergeSelection(s);
    const calls = buildApplyCalls(s);
    expect(calls[0].targetInst).toBe(7);
    expect(calls[0].mergedFrom).toEqual([]);
  });

  it('seedFromServer rebuilds applied groups keyed by instance_id', () => {
    const doc = { paths: [
      { points: [[0, 0, 0], [1, 0, 0]], radius: 0.2, smooth: false, class_id: 0, instance_id: 7 },
      { points: [[2, 0, 0], [3, 0, 0]], radius: 0.2, smooth: true,  class_id: 0, instance_id: 7 },
      { points: [[9, 0, 0], [9, 1, 0]], radius: 0.4, smooth: false, class_id: 2, instance_id: 8 },
    ] };
    const s = seedFromServer(initDrawState(), doc);
    expect(s.paths).toHaveLength(3);
    const groups = new Set(s.paths.map((p) => p.instKey));
    expect(groups.size).toBe(2);
    const g7 = s.paths.filter((p) => s.instanceIds[p.instKey] === 7);
    expect(g7).toHaveLength(2);
    expect(g7[1].smooth).toBe(true);
  });
});
```

- [ ] **Step 2: Run** — FAIL (missing exports).

- [ ] **Step 3: Implement** (append)

```javascript
export function mergeSelection(state) {
  if (state.selection.size < 2) return state;
  const selectedGroups = [];          // ordered, unique instKeys of the selection
  for (const p of state.paths) {
    if (state.selection.has(p.key) && !selectedGroups.includes(p.instKey)) {
      selectedGroups.push(p.instKey);
    }
  }
  if (selectedGroups.length < 2) return state;
  // Survivor: lowest applied backend id wins (spec); else first selected group.
  const appliedGroups = selectedGroups.filter((g) => state.instanceIds[g] != null);
  const survivor = appliedGroups.length
    ? appliedGroups.reduce((m, g) => state.instanceIds[g] < state.instanceIds[m] ? g : m)
    : selectedGroups[0];
  const survivorClass = state.paths.find((p) => p.instKey === survivor).classId;
  const absorbed = selectedGroups.filter((g) => g !== survivor);
  const absorbedIds = absorbed
    .map((g) => state.instanceIds[g])
    .filter((id) => id != null);
  const paths = state.paths.map((p) =>
    absorbed.includes(p.instKey)
      ? { ...p, instKey: survivor, classId: survivorClass }
      : p);
  const instanceIds = { ...state.instanceIds };
  const pendingMergedFrom = { ...state.pendingMergedFrom };
  const carried = absorbed.flatMap((g) => pendingMergedFrom[g] ?? []);
  for (const g of absorbed) { delete instanceIds[g]; delete pendingMergedFrom[g]; }
  if (absorbedIds.length || carried.length) {
    pendingMergedFrom[survivor] = [
      ...(pendingMergedFrom[survivor] ?? []), ...absorbedIds, ...carried,
    ];
  }
  return { ...state, paths, instanceIds, pendingMergedFrom };
}

export function buildApplyCalls(state) {
  // Enter applies whole instances only (spec, workflow item 7).
  const groups = [];
  for (const p of state.paths) {
    if (state.selection.has(p.key) && !groups.includes(p.instKey)) groups.push(p.instKey);
  }
  return groups.map((g) => {
    const paths = state.paths
      .filter((p) => p.instKey === g)
      .map((p) => ({ points: p.points, radius: p.radius, smooth: p.smooth }));
    return {
      instKey: g,
      paths,
      classId: state.paths.find((p) => p.instKey === g).classId,
      targetInst: state.instanceIds[g] ?? -1,
      mergedFrom: state.pendingMergedFrom[g] ?? [],
    };
  });
}

export function markApplied(state, instKey, instanceId) {
  const pendingMergedFrom = { ...state.pendingMergedFrom };
  delete pendingMergedFrom[instKey];
  return {
    ...state,
    instanceIds: { ...state.instanceIds, [instKey]: instanceId },
    pendingMergedFrom,
  };
}

export function seedFromServer(state, doc) {
  let s = state;
  const groupByInstance = {};
  for (const sp of doc.paths ?? []) {
    let instKey = groupByInstance[sp.instance_id];
    const [key, next] = freshKey(s);
    s = next;
    if (!instKey) {
      instKey = key;
      groupByInstance[sp.instance_id] = instKey;
      s = { ...s, instanceIds: { ...s.instanceIds, [instKey]: sp.instance_id } };
    }
    s = {
      ...s,
      paths: [...s.paths, {
        key, points: sp.points, radius: sp.radius,
        smooth: !!sp.smooth, classId: sp.class_id, instKey,
      }],
    };
  }
  return s;
}
```

- [ ] **Step 4: Run the whole file** `npx vitest run src/draw-paths.test.js` — all PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/draw-paths.js frontend/src/draw-paths.test.js
git commit -m "feat(frontend): draw-paths merge semantics + apply-call building + server seeding"
```

---

### Task 10: Viewer imperative additions

**Files:**
- Modify: `frontend/src/viewer.jsx` — add three methods inside `useImperativeHandle` (after `attachBrushGizmo`, ~line 1584).

No unit test (Three.js + DOM; the repo has no jsdom). Verified by the browser pass in Task 13.

- [ ] **Step 1: Add the methods**

```javascript
    // Draw sub-mode hooks. setOrbitEnabled is the explicit seam the spec
    // calls out: pointer-drag of a control point and wheel-resize of a tube
    // must win over camera orbit/zoom, and the controller already has an
    // enabled flag for exactly this (the gizmo uses it too).
    setOrbitEnabled(on) {
      stateRef.current.controller?.setEnabled?.(!!on);
    },
    getCamera() {
      return stateRef.current.camera ?? null;
    },
    attachOverlayGroup() {
      const s = stateRef.current;
      if (!s.scene) return { group: null, remove: () => {} };
      const group = new THREE.Group();
      s.scene.add(group);
      return {
        group,
        remove() {
          s.scene.remove(group);
          group.traverse((n) => {
            n.geometry?.dispose?.();
            n.material?.dispose?.();
          });
        },
      };
    },
```

Note: both controllers (`attachOrbit` viewer.jsx:158, `attachWalk` viewer.jsx:318) implement `setEnabled`, and Draw mode forces orbit nav anyway (Task 13 mirrors fast-label's `onNavModeChange('orbit')`).

- [ ] **Step 2: Verify the frontend still builds**

Run: `cd frontend && npx vitest run` (exercises imports) and `npm run build`
Expected: PASS / build OK

- [ ] **Step 3: Commit**

```bash
git add frontend/src/viewer.jsx
git commit -m "feat(frontend): viewer hooks for draw overlay (orbit gate, camera, overlay group)"
```

---

### Task 11: `draw-mode.jsx` — keys, HUD, panel, apply wiring

**Files:**
- Create: `frontend/src/draw-mode.jsx`

This component owns the draw state (`useState(initDrawState)`), the capture-phase keyboard, the HUD, the panel section, and the apply round-trip. The 3D overlay + pointer handling land in Task 12 — keep `DrawOverlay` a stub here so the component tree mounts.

- [ ] **Step 1: Write the component**

```javascript
// draw-mode.jsx — Draw (centerline) sub-mode of Label mode. Pipes/tanks are
// labeled by drawing centerline paths; the backend extracts points within a
// tube radius. State machine in draw-paths.js; spec in
// docs/superpowers/specs/2026-06-04-centerline-pipe-labeling-design.md.

import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { VoxaAPI } from './api.js';
import { applyDelta } from './segment-state.js';
import {
  initDrawState, addPoint, movePoint, removeLastPoint, endActive,
  selectPath, clearSelection, setRadius, nudgeRadius, setClass, toggleSmooth,
  deleteSelected, mergeSelection, buildApplyCalls, markApplied, seedFromServer,
} from './draw-paths.js';

// Capture-phase keyboard driver (same trick as FastLabelKeys: beat the
// LabelMode global keydown). classes[i] ↔ palette index i.
export function DrawKeys({ active, classes, onKey }) {
  useEffect(() => {
    if (!active) return undefined;
    const handler = (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;   // leave Ctrl+S/Z/click alone
      const clsIdx = classes.findIndex((c) => c.hotkey === e.key);
      let handled = true;
      if (clsIdx >= 0) onKey({ type: 'class', clsIdx });
      else if (e.key === 'Enter') onKey({ type: 'apply' });
      else if (e.key === 'Escape') onKey({ type: 'escape' });
      else if (e.key === 'Backspace' || e.key === 'Delete') onKey({ type: 'backspace' });
      else if (e.key === 'm' || e.key === 'M') onKey({ type: 'merge' });
      else if (e.key === 'c' || e.key === 'C') onKey({ type: 'smooth' });
      else if (e.key === '+' || e.key === '=') onKey({ type: 'radius', dir: +1 });
      else if (e.key === '-' || e.key === '_') onKey({ type: 'radius', dir: -1 });
      else handled = false;
      if (handled) { e.preventDefault(); e.stopPropagation(); }
    };
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [active, classes, onKey]);
  return null;
}

function DrawHUD({ state, classes, toast }) {
  const drawing = !!state.active;
  const nSel = state.selection.size;
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
          <b style={{ color: '#60a5fa' }}>
            {drawing ? 'Drawing path…' : nSel ? `${nSel} path${nSel > 1 ? 's' : ''} selected` : 'Draw centerlines'}
          </b>
          <span style={{ opacity: 0.65 }}>
            {drawing
              ? 'Ctrl+click add · ⌫ undo pt · Esc end · Enter end+apply'
              : 'Ctrl+click start · click tube select · scroll/± radius · C smooth · M merge · Enter apply · Esc exit'}
          </span>
        </>
      )}
    </div>
  );
}

// Stub replaced in the next task.
function DrawOverlay() { return null; }

export default function DrawMode({
  viewerRef, classes, segState, setSegState, onExit,
}) {
  const [draw, setDraw] = useState(() => initDrawState());
  const [defaultClsIdx, setDefaultClsIdx] = useState(0);
  const [toast, setToast] = useState(null);
  const toastTimer = useRef(null);
  const showToast = useCallback((msg) => {
    clearTimeout(toastTimer.current);
    setToast(msg);
    toastTimer.current = setTimeout(() => setToast(null), 2500);
  }, []);

  // Load stored centerlines once on open so applied paths render + re-edit.
  useEffect(() => {
    let gone = false;
    VoxaAPI.getCenterlines()
      .then((doc) => { if (!gone) setDraw((s) => seedFromServer(s, doc)); })
      .catch((err) => { if (!gone) showToast(`centerlines load failed: ${err.message}`); });
    return () => { gone = true; };
  }, [showToast]);

  const applySelection = useCallback(async () => {
    // Enter while drawing = end + apply that path (spec shortcut).
    let s = draw;
    if (s.active) {
      const key = s.active;
      s = endActive(s);
      if (s.paths.some((p) => p.key === key)) s = selectPath(s, key);
    }
    const calls = buildApplyCalls(s);
    if (calls.length === 0) { setDraw(s); return; }
    for (const call of calls) {
      let r;
      try {
        r = await VoxaAPI.centerlineApply({
          paths: call.paths,
          targetClass: call.classId,
          targetInst: call.targetInst,
          mergedFrom: call.mergedFrom,
        });
      } catch (err) {
        showToast(`apply failed: ${err.message}`);
        continue;                       // surface and move on; state unchanged for this group
      }
      if (r.nAffected === 0) {
        showToast('no points in tube');
        continue;
      }
      s = markApplied(s, call.instKey, r.instanceId);
      setSegState((st) => st ? applyDelta(st, {
        indices: r.indices,
        after_class: r.afterClass,
        after_instance: r.afterInstance,
      }) : st);
    }
    setDraw(clearSelection(s));
  }, [draw, setSegState, showToast]);

  const onKey = useCallback((action) => {
    switch (action.type) {
      case 'class':
        setDefaultClsIdx(action.clsIdx);
        setDraw((s) => setClass(s, action.clsIdx));
        break;
      case 'apply':
        applySelection();
        break;
      case 'escape':
        setDraw((s) => {
          if (s.active) return endActive(s);
          if (s.selection.size) return clearSelection(s);
          onExit();
          return s;
        });
        break;
      case 'backspace':
        setDraw((s) => s.active ? removeLastPoint(s) : deleteSelected(s));
        break;
      case 'merge':
        setDraw((s) => mergeSelection(s));
        break;
      case 'smooth':
        setDraw((s) => toggleSmooth(s));
        break;
      case 'radius':
        setDraw((s) => nudgeRadius(s, action.dir));
        break;
      default:
    }
  }, [applySelection, onExit]);

  return (
    <>
      <DrawKeys active classes={classes} onKey={onKey} />
      <DrawHUD state={draw} classes={classes} toast={toast} />
      <DrawOverlay
        viewerRef={viewerRef}
        draw={draw}
        setDraw={setDraw}
        classes={classes}
        defaultClsIdx={defaultClsIdx}
      />
      <DrawPanel
        draw={draw}
        setDraw={setDraw}
        classes={classes}
        onApply={applySelection}
      />
    </>
  );
}

// Side-panel section: path list + radius field + actions. Rendered by
// LabelMode inside the left sidebar (portal-free: this component returns
// plain divs; LabelMode places it).
export function DrawPanel({ draw, setDraw, classes, onApply }) {
  const selected = draw.paths.filter((p) => draw.selection.has(p.key));
  const radiusValue = selected[0]?.radius
    ?? draw.paths.find((p) => p.key === draw.active)?.radius
    ?? draw.lastRadius;
  return (
    <div className="draw-panel" style={{ marginTop: 10 }}>
      <div className="side-hd"><span>Centerline paths</span>
        <span className="badge-soft">{draw.paths.length}</span></div>
      <div className="ins-row">
        <label>Radius</label>
        <input className="ins-input" type="number" step="0.01" min="0.005"
          value={Number(radiusValue.toFixed(4))}
          onChange={(e) => {
            const v = parseFloat(e.target.value);
            if (Number.isFinite(v) && v > 0) setDraw((s) => setRadius(s, v));
          }} />
      </div>
      <div style={{ maxHeight: 180, overflowY: 'auto' }}>
        {draw.paths.map((p) => {
          const cls = classes[p.classId];
          const applied = draw.instanceIds[p.instKey] != null;
          const isSel = draw.selection.has(p.key);
          return (
            <div key={p.key}
              className={'inst-row' + (isSel ? ' selected' : '')}
              onClick={(e) => setDraw((s) =>
                selectPath(s, p.key, { additive: e.shiftKey }))}>
              <span className="inst-dot" style={{ background: cls?.color }} />
              <div className="inst-text">
                <b>{cls?.label || '?'} {applied ? `#${draw.instanceIds[p.instKey]}` : '(staged)'}</b>
                <em>{p.points.length} pts · r={p.radius.toFixed(3)}{p.smooth ? ' · smooth' : ''}</em>
              </div>
            </div>
          );
        })}
      </div>
      <div className="ins-actions">
        <button className="ghost-btn" disabled={draw.selection.size < 2}
          onClick={() => setDraw((s) => mergeSelection(s))}>M Merge</button>
        <button className="ghost-btn" disabled={!draw.selection.size && !draw.active}
          onClick={onApply}>↵ Apply</button>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify build/tests**

Run: `cd frontend && npx vitest run && npm run build`
Expected: PASS (existing tests untouched; new file compiles)

- [ ] **Step 3: Commit**

```bash
git add frontend/src/draw-mode.jsx
git commit -m "feat(frontend): DrawMode component — keys, HUD, panel, apply wiring"
```

---

### Task 12: `draw-mode.jsx` — THREE overlay + pointer interactions

Replace the `DrawOverlay` stub. Responsibilities:
1. Render paths into a viewer overlay group: control-point spheres, polyline, semi-transparent tube (cylinders per straight segment; `TubeGeometry` over `CatmullRomCurve3` for smooth). Staged tubes opacity 0.25, applied 0.10, selected paths get opacity +0.15 and a brighter color.
2. Pointer handling with the spec's pick priority (control point > tube > cloud):
   - `pointerdown` (capture, on viewer DOM element): raycast spheres → start drag (disable orbit); else raycast tubes → select (`shiftKey` additive, stop), `evt.ctrlKey` + cloud hit (`viewerRef.current.firstHitUnderCursor(evt)`) → `addPoint`; else (no ctrl, no hit) → `clearSelection`.
   - `pointermove`/`pointerup` (window): drag the grabbed control point on a camera-parallel plane through its position; re-enable orbit on release.
   - `wheel` (capture, on the viewer DOM element's **parent**, so it fires before the orbit's bubble-phase canvas listener): when a path is active/selected → `preventDefault` + `stopPropagation` + `nudgeRadius`. *(Capture vs bubble on the same node does not reorder listeners — the parent hop is what guarantees winning; keep this comment in the code.)*

- [ ] **Step 1: Implement `DrawOverlay`** (replace the stub)

```javascript
function DrawOverlay({ viewerRef, draw, setDraw, classes, defaultClsIdx }) {
  const layerRef = useRef(null);        // { group, remove }
  const dragRef = useRef(null);         // { pathKey, pointIdx, plane }
  const drawRef = useRef(draw);
  drawRef.current = draw;

  // One overlay group for the lifetime of the sub-mode.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v?.attachOverlayGroup) return undefined;
    layerRef.current = v.attachOverlayGroup();
    return () => { layerRef.current?.remove(); layerRef.current = null; };
  }, [viewerRef]);

  // Rebuild overlay children whenever paths/selection change. Path counts
  // are tiny (dozens), so dispose-and-rebuild beats incremental bookkeeping.
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer?.group) return;
    const group = layer.group;
    while (group.children.length) {
      const c = group.children.pop();
      c.geometry?.dispose?.(); c.material?.dispose?.();
      group.remove(c);
    }
    for (const p of draw.paths) {
      const cls = classes[p.classId];
      const color = new THREE.Color(cls?.color || '#60a5fa');
      const isSel = draw.selection.has(p.key) || draw.active === p.key;
      const applied = draw.instanceIds[p.instKey] != null;
      const baseOpacity = applied ? 0.10 : 0.25;
      const tubeMat = new THREE.MeshBasicMaterial({
        color, transparent: true, depthWrite: false,
        opacity: isSel ? baseOpacity + 0.15 : baseOpacity,
      });
      // Tube: smooth → one TubeGeometry; straight → cylinder per segment.
      const pts3 = p.points.map(([x, y, z]) => new THREE.Vector3(x, y, z));
      if (p.smooth && pts3.length >= 3) {
        const curve = new THREE.CatmullRomCurve3(pts3);
        const tube = new THREE.Mesh(
          new THREE.TubeGeometry(curve, pts3.length * 8, p.radius, 12, false), tubeMat);
        tube.userData.drawPath = p.key;
        group.add(tube);
      } else {
        for (let i = 0; i < pts3.length - 1; i++) {
          const a = pts3[i], b = pts3[i + 1];
          const len = a.distanceTo(b);
          if (len < 1e-6) continue;
          const cyl = new THREE.Mesh(
            new THREE.CylinderGeometry(p.radius, p.radius, len, 12, 1, true),
            tubeMat.clone());
          cyl.position.copy(a).lerp(b, 0.5);
          cyl.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
          cyl.userData.drawPath = p.key;
          group.add(cyl);
        }
      }
      // Control points — pick priority targets (spec: point > tube > cloud).
      const sphereR = Math.max(0.02, p.radius * 0.3);
      p.points.forEach((pt, i) => {
        const sph = new THREE.Mesh(
          new THREE.SphereGeometry(sphereR, 12, 8),
          new THREE.MeshBasicMaterial({ color: isSel ? 0xffffff : color }));
        sph.position.set(pt[0], pt[1], pt[2]);
        sph.userData.drawPoint = { pathKey: p.key, pointIdx: i };
        group.add(sph);
      });
    }
  }, [draw, classes]);

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
      raycaster.setFromCamera({
        x: ((evt.clientX - rect.left) / rect.width) * 2 - 1,
        y: -((evt.clientY - rect.top) / rect.height) * 2 + 1,
      }, camera);
      return raycaster.intersectObjects(group.children, false);
    };

    const onPointerDown = (evt) => {
      if (evt.button !== 0) return;
      const hits = castOverlay(evt);
      const sphereHit = hits.find((h) => h.object.userData.drawPoint);
      if (sphereHit && !evt.ctrlKey) {
        // Drag start: move the point on a camera-parallel plane through it.
        const camera = v.getCamera();
        const normal = new THREE.Vector3();
        camera.getWorldDirection(normal);
        const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(
          normal, sphereHit.object.position.clone());
        dragRef.current = { ...sphereHit.object.userData.drawPoint, plane };
        v.setOrbitEnabled(false);
        evt.stopPropagation();
        return;
      }
      if (evt.ctrlKey || evt.metaKey) {
        // Place a control point at the picked cloud point.
        const hit = v.firstHitUnderCursor(evt);
        if (hit) {
          setDraw((s) => addPoint(
            s, [hit.world.x, hit.world.y, hit.world.z],
            s.active ? s.paths.find((p) => p.key === s.active).classId : defaultClsIdx));
        }
        evt.stopPropagation();
        return;
      }
      const tubeHit = hits.find((h) => h.object.userData.drawPath);
      if (tubeHit) {
        setDraw((s) => selectPath(s, tubeHit.object.userData.drawPath,
          { additive: evt.shiftKey }));
        return;     // don't stop: orbit-from-tube is harmless and feels natural
      }
      // Plain click on empty space / cloud clears the selection (keymap).
      if (drawRef.current.selection.size) setDraw((s) => clearSelection(s));
    };

    const onPointerMove = (evt) => {
      const drag = dragRef.current;
      if (!drag) return;
      const camera = v.getCamera();
      if (!camera) return;
      const rect = dom.getBoundingClientRect();
      raycaster.setFromCamera({
        x: ((evt.clientX - rect.left) / rect.width) * 2 - 1,
        y: -((evt.clientY - rect.top) / rect.height) * 2 + 1,
      }, camera);
      const pt = new THREE.Vector3();
      if (raycaster.ray.intersectPlane(drag.plane, pt)) {
        setDraw((s) => movePoint(s, drag.pathKey, drag.pointIdx, [pt.x, pt.y, pt.z]));
      }
    };

    const onPointerUp = () => {
      if (!dragRef.current) return;
      dragRef.current = null;
      v.setOrbitEnabled(true);
    };

    // Wheel-resize beats orbit-zoom: orbit's wheel listener bubbles on the
    // canvas, so a CAPTURE listener on the PARENT runs first and can stop
    // propagation before the canvas ever sees it. (Capture vs bubble on the
    // same node would NOT reorder — the parent hop is the trick.)
    const wheelHost = dom.parentElement || dom;
    const onWheel = (evt) => {
      const s = drawRef.current;
      if (!s.active && s.selection.size === 0) return;   // fall through to zoom
      evt.preventDefault();
      evt.stopPropagation();
      setDraw((cur) => nudgeRadius(cur, -Math.sign(evt.deltaY)));
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
  }, [viewerRef, setDraw, defaultClsIdx]);

  return null;
}
```

Also import the extra state functions used here at the top of `draw-mode.jsx` (already listed in Task 11's import block).

- [ ] **Step 2: Verify build/tests** — `npx vitest run && npm run build` from `frontend/`. Expected: PASS.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/draw-mode.jsx
git commit -m "feat(frontend): Draw overlay — tube/point rendering + pointer interactions"
```

---

### Task 13: `mode-label.jsx` integration + browser verification

**Files:**
- Modify: `frontend/src/mode-label.jsx`

- [ ] **Step 1: Wire the sub-mode**

1. Import: `import DrawMode, { DrawPanel } from './draw-mode.jsx';` → actually only `DrawMode` (it renders its own panel; see step 3).
2. State, next to `fastMode` (~line 84): `const [drawMode, setDrawMode] = useStateLabel(false);`
3. Mutual exclusion + suppression:
   - In the toggle handlers: enabling one disables the other (`setFastMode(false)` when enabling draw, and vice versa).
   - In the main hotkey effect (~line 910): change `if (fastMode) return;` → `if (fastMode || drawMode) return;` and add `drawMode` to the dep array.
   - Force orbit nav like fast mode does (~line 858): extend the effect to `if ((fastMode || drawMode) && navMode === 'walk') onNavModeChange?.('orbit');` with `drawMode` in deps.
   - The Ctrl/Cmd-click preseg-toggle pick effect (~line 146) must not fight Ctrl+click point placement: add `if (drawMode) return;` as the first line of that effect's callback wiring (i.e., make the effect bail when `drawMode` is on) and add `drawMode` to its deps.
4. Toggle button, under the fast-labeling button (~line 1039), shown for annotated scenes with a session (works on blank sessions — `segState` exists whenever a session is active):

```javascript
        {segState && isAnnotated && (
          <button
            className={'tool-btn' + (drawMode ? ' active' : '')}
            style={{ margin: '6px 0 0', width: '100%', justifyContent: 'center',
                     borderColor: drawMode ? '#60a5fa' : undefined }}
            onClick={() => { setFastMode(false); setDrawMode((d) => !d); }}
            title="Draw centerlines along pipes/tanks; points within the tube radius get labeled">
            ⤳ {drawMode ? 'Exit draw mode' : 'Draw centerlines'}
          </button>
        )}
```

5. Mount the sub-mode (next to the FastLabel components at the top of the returned JSX):

```javascript
      {drawMode && (
        <DrawMode
          viewerRef={viewerRef}
          classes={classes}
          segState={segState}
          setSegState={setSegState}
          onExit={() => setDrawMode(false)}
        />
      )}
```

`DrawPanel` placement: `DrawMode` renders the HUD + overlay as fixed/null elements; move the `<DrawPanel>` render out of `DrawMode`'s fragment into the left sidebar — simplest is to have `DrawMode` accept the panel placement as-is (it returns the panel in its fragment, which renders inside `mode-root` — fixed-position HUD is fine, but the panel needs to sit in the sidebar). **Do this instead:** render `<DrawMode>` inside the left `<aside className="side-l">` right after the toggle button. Its HUD/overlay are `position: fixed`/null so visual placement is unaffected, and the panel flows naturally into the sidebar.

6. Add `drawMode` to the help sections if trivial (optional, skip if it bloats the diff).

- [ ] **Step 2: Run all frontend tests + build**

Run: `cd frontend && npx vitest run && npm run build`
Expected: PASS

- [ ] **Step 3: Run the full test suite**

Run: `npm test` (from repo root)
Expected: backend + frontend all PASS

- [ ] **Step 4: Commit**

```bash
git add frontend/src/mode-label.jsx
git commit -m "feat(frontend): wire Draw sub-mode into Label mode"
```

- [ ] **Step 5: Browser verification** (use the `browser-verification` skill; per the user's global rules this is REQUIRED before calling the feature done)

Start the dev stack (`npm run dev`, backend on :8765, UI on :5173), open an annotated scene with a **blank session** in Label mode, then verify:
1. "Draw centerlines" button appears; clicking enters the sub-mode (HUD strip visible).
2. Press `1` (pipe class), Ctrl+click 3 points along a pipe → tube preview renders in class color, chains the points.
3. Scroll → tube radius changes (camera must NOT zoom); click empty space then scroll → camera zooms again.
4. Drag a control point → it follows the cursor on a screen-parallel plane; camera does not orbit during the drag.
5. Esc → path staged; click its tube → selected (brighter); `C` toggles smooth on a ≥3-point path (tube re-renders curved).
6. Enter → points inside the tube recolor to the class (apply round-trip works); panel shows the instance id.
7. Draw a second path, select both (click + Shift+click), `M`, Enter → one shared instance id in the panel.
8. Ctrl+Z (or the undo route via UI if wired) reverts the labels.
9. Reload the page, re-enter Draw mode → applied paths reappear (seeded from `GET /api/segment/centerlines`).
10. Zero console errors; screenshot the tube preview + an applied pipe.

---

### Task 14: Docs

**Files:**
- Modify: `CLAUDE.md` (Project modes blurb + conventions)
- Modify: `docs/scan-schema.md` (session dir contents: `centerlines.json`)

- [ ] **Step 1: Update `CLAUDE.md`**
- In the Project paragraph where Label mode's sub-modes are described, add the Draw sub-mode one-liner: draws centerline paths for cylindrical objects (pipes/tanks); backend labels points within a tube radius; per-session persistence in `sessions/<id>/centerlines.json`; see `frontend/src/draw-mode.jsx`.

- [ ] **Step 2: Update `docs/scan-schema.md`**
- Add `centerlines.json` to the `sessions/<session_id>/` listing with a one-paragraph contract: the flat `{"paths": [...]}` schema (points/radius/smooth/class_id/instance_id), replace-by-instance_id write semantics, optional file (absent until the first centerline apply).

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/scan-schema.md
git commit -m "docs: Draw sub-mode + centerlines.json contract"
```

---

## Execution notes

- **Restart the backend** after Python edits (no autoreload — kill and restart `npm run dev`).
- Pre-existing dirty files in the worktree are unrelated work — never commit them.
- After all tasks: run the `simplify` skill on the final diff (user's global workflow), then `superpowers:requesting-code-review` before opening the PR onto `main`.
