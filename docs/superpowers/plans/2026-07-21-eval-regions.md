# Eval Regions (Phase 1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Scan-level eval regions in voxa — prism-drawn geometry + draft/eval-grade status persisted in `eval_regions.json`, a 7th Region rail tool, a Regions tab with per-region unlabeled % and majority-inside confirmed instances, and a status-colored viewer overlay.

**Architecture:** New pure backend module `backend/labeling/regions.py` (store + gate + stats) behind a new route module `backend/routes/regions.py`; the backend owns the eval-grade gate (p90 ≤ 10 mm via `raw_sample_spacing`) and the geometry lock. Frontend extracts the prism footprint+height draw machinery into a shared `prism-draw.jsx`, adds `region-mode.jsx` (draw + overlay) and `region-panel.jsx` (right-panel tab). Spec: `docs/superpowers/specs/2026-07-21-eval-regions-design.md` — read it before starting.

**Tech Stack:** FastAPI + numpy + scan_schema (backend), React 18 + Three.js (frontend), pytest + vitest.

**Branch:** `feat/eval-regions` (already exists, spec committed).

**Conventions that bind every task** (from CLAUDE.md):
- Backend has no autoreload — irrelevant for tests, but restart any running dev server before manual checks.
- Run backend tests with `.venv/bin/pytest`, frontend with `npx vitest run --root frontend`.
- Commit after each task with a conventional message.
- Coordinates: Three.js Y-up; prism = `{polygon:[[x,z],...], y0, height}`.

---

## File map

| File | Action | Responsibility |
|---|---|---|
| `backend/labeling/regions.py` | Create | Store (load/save/CRUD), frame shift, eval-grade gate, stats — pure functions |
| `backend/routes/regions.py` | Create | HTTP surface; session requirement; frame conversion; error mapping |
| `backend/app/schemas.py` | Modify | `RegionPrism`, `CreateRegionRequest`, `PatchRegionRequest` |
| `backend/main.py` | Modify | Register `regions` router |
| `backend/tests/conftest.py` | Modify | `pts=` param on scene builders + dense-scene fixture |
| `backend/tests/test_regions.py` | Create | Pure-fn tests (store, gate, stats, frame shift) |
| `backend/tests/test_regions_endpoints.py` | Create | Route tests (CRUD, locks, gate, stats, frame round-trip) |
| `frontend/src/api.js` | Modify | `regionsList/regionCreate/regionPatch/regionDelete/regionStats` |
| `frontend/src/prism-draw.jsx` | Create | Extracted footprint+height draw machinery (from `prism-mode.jsx`) |
| `frontend/src/prism-mode.jsx` | Modify | Import the extracted machinery (pure refactor) |
| `frontend/src/region-utils.js` (+ test) | Create | `majorityInstances`, `unlabeledPct`, status colors — pure |
| `frontend/src/label-tools.js` (+ test) | Modify | `region` tool entry + availability |
| `frontend/src/region-mode.jsx` | Create | `RegionMode` (draw → POST create) + `RegionsOverlay` (persisted volumes) |
| `frontend/src/tool-options.jsx` | Modify | `RegionOptions` case |
| `frontend/src/region-panel.jsx` (+ jsdom test) | Create | Regions tab: rows, rename, status flip, eye, delete, instance list |
| `frontend/src/mode-label.jsx` | Modify | Tool wiring, regions/stats state, tabs in `side-r`, overlay mount |
| `CLAUDE.md` | Modify | Region tool bullet |

---

### Task 1: Backend region store, gate, and stats (`backend/labeling/regions.py`)

**Files:**
- Create: `backend/labeling/regions.py`
- Test: `backend/tests/test_regions.py`

- [ ] **Step 1: Write the failing tests**

Create `backend/tests/test_regions.py`:

```python
"""Pure-function tests for the eval-region store, gate, and stats
(backend/labeling/regions.py). Endpoint tests live in
test_regions_endpoints.py."""
from __future__ import annotations

import json

import numpy as np
import pytest

from labeling import regions as reg

PRISM = {"polygon": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
         "y0": -0.5, "height": 2.0}


def grid_positions(spacing, n=20):
    """n×n XZ grid at y=0 with the given spacing — nn distance == spacing."""
    g = np.arange(n) * spacing
    xs, zs = np.meshgrid(g, g)
    return np.stack([xs.ravel(), np.zeros(n * n), zs.ravel()], axis=1).astype(np.float32)


# ---- store ----------------------------------------------------------------

def test_load_missing_file_returns_empty_doc(tmp_path):
    doc = reg.load_regions(tmp_path)
    assert doc == {"version": 1, "next_region_id": 1, "regions": []}


def test_save_then_load_roundtrip(tmp_path):
    doc = reg.load_regions(tmp_path)
    reg.create_region(doc, PRISM, name="skid A")
    reg.save_regions(tmp_path, doc)
    assert (tmp_path / "eval_regions.json").exists()
    loaded = reg.load_regions(tmp_path)
    assert loaded["regions"][0]["name"] == "skid A"
    assert loaded["next_region_id"] == 2


def test_create_assigns_monotonic_ids_never_reused(tmp_path):
    doc = reg.load_regions(tmp_path)
    r1 = reg.create_region(doc, PRISM)
    r2 = reg.create_region(doc, PRISM)
    reg.delete_region(doc, r2["id"])
    r3 = reg.create_region(doc, PRISM)
    assert (r1["id"], r2["id"], r3["id"]) == (1, 2, 3)


def test_create_defaults(tmp_path):
    doc = reg.load_regions(tmp_path)
    r = reg.create_region(doc, PRISM)
    assert r["name"] == "Region 1"
    assert r["status"] == "draft"
    assert "accuracy" not in r
    assert "created_at" in r


def test_create_validates_prism():
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    with pytest.raises(reg.RegionError):
        reg.create_region(doc, {"polygon": [[0, 0], [1, 0]], "y0": 0, "height": 1})
    with pytest.raises(reg.RegionError):
        reg.create_region(doc, {**PRISM, "height": 0.0})


def test_rename_rejects_empty():
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, PRISM)
    reg.rename_region(doc, r["id"], "new name")
    assert doc["regions"][0]["name"] == "new name"
    with pytest.raises(reg.RegionError):
        reg.rename_region(doc, r["id"], "  ")


def test_unknown_id_raises_not_found():
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    with pytest.raises(reg.RegionNotFound):
        reg.rename_region(doc, 99, "x")


# ---- locks ----------------------------------------------------------------

def _eval_grade_doc():
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, {"polygon": [[-0.01, -0.01], [0.2, -0.01], [0.2, 0.2], [-0.01, 0.2]],
                                "y0": -0.5, "height": 1.0})
    reg.flip_status(doc, r["id"], "eval_grade", grid_positions(0.005))
    return doc, r["id"]


def test_eval_grade_locks_geometry_and_delete():
    doc, rid = _eval_grade_doc()
    with pytest.raises(reg.RegionError):
        reg.set_geometry(doc, rid, PRISM)
    with pytest.raises(reg.RegionError):
        reg.delete_region(doc, rid)
    # renames stay allowed — a name is not geometry
    reg.rename_region(doc, rid, "still fine")


def test_draft_flip_unlocks_and_clears_accuracy():
    doc, rid = _eval_grade_doc()
    reg.flip_status(doc, rid, "draft", grid_positions(0.005))
    r = doc["regions"][0]
    assert r["status"] == "draft"
    assert "accuracy" not in r
    reg.set_geometry(doc, rid, PRISM)   # unlocked now
    reg.delete_region(doc, rid)
    assert doc["regions"] == []


# ---- gate -----------------------------------------------------------------

def test_gate_passes_on_5mm_grid_and_records_accuracy():
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, {"polygon": [[-0.01, -0.01], [0.2, -0.01], [0.2, 0.2], [-0.01, 0.2]],
                                "y0": -0.5, "height": 1.0})
    out = reg.flip_status(doc, r["id"], "eval_grade", grid_positions(0.005))
    assert out["status"] == "eval_grade"
    acc = out["accuracy"]
    assert acc["p90"] == pytest.approx(0.005, abs=1e-4)
    assert acc["loa"] == "LOA40"
    assert "measured_at" in acc


def test_gate_refuses_20mm_grid():
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, {"polygon": [[-0.1, -0.1], [0.5, -0.1], [0.5, 0.5], [-0.1, 0.5]],
                                "y0": -0.5, "height": 1.0})
    with pytest.raises(reg.RegionError, match="p90"):
        reg.flip_status(doc, r["id"], "eval_grade", grid_positions(0.020))
    assert doc["regions"][0]["status"] == "draft"


def test_gate_refuses_below_point_floor():
    # 100-point floor is load-bearing: raw_sample_spacing returns (0,0) for
    # n<2, which would falsely PASS the p90 check on a near-empty region.
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, PRISM)
    few = grid_positions(0.005, n=5)          # 25 pts inside
    with pytest.raises(reg.RegionError, match="100"):
        reg.flip_status(doc, r["id"], "eval_grade", few)


def test_gate_unknown_status_rejected():
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, PRISM)
    with pytest.raises(reg.RegionError):
        reg.flip_status(doc, r["id"], "frozen", grid_positions(0.005))


# ---- frame shift ----------------------------------------------------------

def test_shift_prism_roundtrip():
    off = [574184.0, 49.0, 6220868.0]        # (dx, dy, dz)
    stored = reg.shift_prism(PRISM, off)
    assert stored["polygon"][1] == [1.0 + 574184.0, 0.0 + 6220868.0]
    assert stored["y0"] == pytest.approx(-0.5 + 49.0)
    assert stored["height"] == PRISM["height"]
    back = reg.shift_prism(stored, [-v for v in off])
    assert np.allclose(back["polygon"], PRISM["polygon"])
    assert back["y0"] == pytest.approx(PRISM["y0"])


# ---- stats ----------------------------------------------------------------

def test_region_stats_counts_unlabeled_and_instances():
    # 6 points: 4 inside the unit prism, 2 outside.
    pos = np.array([[0.5, 0.0, 0.5], [0.6, 0.0, 0.6], [0.7, 0.0, 0.7],
                    [0.8, 0.0, 0.8], [5.0, 0.0, 5.0], [6.0, 0.0, 6.0]],
                   dtype=np.float32)
    class_ids = np.array([0, 0, -1, -1, 0, -1], dtype=np.int8)
    inst_ids = np.array([7, 7, -1, -1, 7, -1], dtype=np.int32)
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    reg.create_region(doc, PRISM)
    stats = reg.region_stats(doc, pos, class_ids, inst_ids)
    assert len(stats) == 1
    s = stats[0]
    assert s["id"] == 1
    assert s["n_points"] == 4
    assert s["n_unlabeled"] == 2
    # instance 7: 2 of its 3 points inside
    assert s["instances"] == {7: {"inside": 2, "total": 3}}


def test_region_stats_applies_offset():
    # Stored-frame region at x,z ∈ [10,11]; runtime positions near origin with
    # offset (10,0,10) — the offset shift must bring them together.
    pos = np.array([[0.5, 0.0, 0.5]], dtype=np.float32)
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    reg.create_region(doc, {"polygon": [[10.0, 10.0], [11.0, 10.0], [11.0, 11.0], [10.0, 11.0]],
                            "y0": -1.0, "height": 4.0})
    empty = reg.region_stats(doc, pos, np.array([-1], dtype=np.int8),
                             np.array([-1], dtype=np.int32))
    assert empty[0]["n_points"] == 0
    hit = reg.region_stats(doc, pos, np.array([-1], dtype=np.int8),
                           np.array([-1], dtype=np.int32), offset=[10.0, 0.0, 10.0])
    assert hit[0]["n_points"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_regions.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'labeling.regions'` (collection error is fine).

- [ ] **Step 3: Implement `backend/labeling/regions.py`**

```python
"""Eval-region store, gate, and stats (eval-labeling phase 1).

``eval_regions.json`` lives at the SCAN root — scan-level truth shared by
every session on the scan. Geometry in the file is in the scan's STORED
frame; the routes convert to/from the runtime (recentered) frame via
``shift_prism`` + the load-time ``recenter_offset``. Caveat for z-up scans:
load also applies the z-up -> y-up rotation BEFORE recentering, so for those
scans "stored frame" means the y-up display frame plus offset, not literal
scan.ply coordinates — consistent within voxa (load is deterministic), but
the phase-3 manifest generator must replay the same rotation to read it. The backend owns the
eval-grade gate (p90 <= 10 mm over the region's full-res points) and the
geometry lock — same server-side-invariant philosophy as
``reject_frozen_class``. See
docs/superpowers/specs/2026-07-21-eval-regions-design.md.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from scan_schema import atomic_write_json

from labeling.materialize import loa_band, raw_sample_spacing
from labeling.shapes import prism_indices

REGIONS_FILE = "eval_regions.json"
MIN_GATE_POINTS = 100          # raw_sample_spacing returns (0,0) for n<2 —
EVAL_GRADE_P90_M = 0.010       # without a floor an empty region would PASS.


class RegionError(ValueError):
    """Invalid input or a locked/gated operation — routes map this to 422."""


class RegionNotFound(KeyError):
    """Unknown region id — routes map this to 404."""


def empty_doc() -> dict:
    return {"version": 1, "next_region_id": 1, "regions": []}


def load_regions(scan_dir) -> dict:
    p = Path(scan_dir) / REGIONS_FILE
    if not p.exists():
        return empty_doc()
    return json.loads(p.read_text())


def save_regions(scan_dir, doc: dict) -> None:
    atomic_write_json(Path(scan_dir) / REGIONS_FILE, doc)


def validate_prism(prism: dict) -> None:
    if len(prism.get("polygon") or []) < 3:
        raise RegionError("footprint needs at least 3 vertices")
    if not float(prism.get("height") or 0) > 0:
        raise RegionError("height must be > 0")


def shift_prism(prism: dict, delta) -> dict:
    """Translate a prism by (dx, dy, dz). stored = runtime + recenter_offset;
    pass the negated offset to go the other way."""
    dx, dy, dz = (float(v) for v in delta)
    return {"polygon": [[float(x) + dx, float(z) + dz] for x, z in prism["polygon"]],
            "y0": float(prism["y0"]) + dy, "height": float(prism["height"])}


def _get(doc: dict, rid: int) -> dict:
    for r in doc["regions"]:
        if r["id"] == rid:
            return r
    raise RegionNotFound(f"no region with id {rid}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_region(doc: dict, prism: dict, name: str | None = None) -> dict:
    validate_prism(prism)
    rid = int(doc["next_region_id"])
    doc["next_region_id"] = rid + 1
    region = {"id": rid, "name": name or f"Region {rid}", "status": "draft",
              "prism": shift_prism(prism, (0, 0, 0)),   # deep-copies + floats
              "created_at": _now()}
    doc["regions"].append(region)
    return region


def rename_region(doc: dict, rid: int, name: str) -> dict:
    name = (name or "").strip()
    if not name:
        raise RegionError("name must be non-empty")
    r = _get(doc, rid)
    r["name"] = name
    return r


def set_geometry(doc: dict, rid: int, prism: dict) -> dict:
    r = _get(doc, rid)
    if r["status"] == "eval_grade":
        raise RegionError("eval-grade geometry is locked — flip to draft first")
    validate_prism(prism)
    r["prism"] = shift_prism(prism, (0, 0, 0))
    return r


def delete_region(doc: dict, rid: int) -> None:
    r = _get(doc, rid)
    if r["status"] == "eval_grade":
        raise RegionError("eval-grade region is locked — flip to draft first")
    doc["regions"].remove(r)


def flip_status(doc: dict, rid: int, status: str, positions,
                offset=(0.0, 0.0, 0.0)) -> dict:
    """draft <-> eval_grade. The eval_grade flip is the gate: measure the
    region's local p50/p90 spacing over its full-res points and refuse
    (RegionError) below the point floor or above the 10 mm bar."""
    r = _get(doc, rid)
    if status == "draft":
        r["status"] = "draft"
        r.pop("accuracy", None)
        return r
    if status != "eval_grade":
        raise RegionError(f"unknown status {status!r}")
    if r["status"] == "eval_grade":
        return r
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    runtime = shift_prism(r["prism"], [-v for v in offset])
    idx = prism_indices(positions, runtime)
    if len(idx) < MIN_GATE_POINTS:
        raise RegionError(
            f"region holds {len(idx)} points — at least {MIN_GATE_POINTS} "
            "needed to measure spacing")
    p50, p90 = raw_sample_spacing(positions[idx])
    if p90 > EVAL_GRADE_P90_M:
        raise RegionError(
            f"measured p90 spacing {p90 * 1000:.1f} mm exceeds the "
            f"{EVAL_GRADE_P90_M * 1000:.0f} mm eval-grade bar")
    r["status"] = "eval_grade"
    r["accuracy"] = {"p50": p50, "p90": p90, "loa": loa_band(p90),
                     "measured_at": _now()}
    return r


def region_stats(doc: dict, positions, class_ids, instance_ids,
                 offset=(0.0, 0.0, 0.0)) -> list[dict]:
    """Per-region point/unlabeled/instance-overlap counts over the full-res
    working arrays. The caller (frontend) filters instances to confirmed and
    applies the majority rule — confirmed-status lives client-side, exactly
    like the protect_instances pattern."""
    positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
    class_ids = np.asarray(class_ids)
    instance_ids = np.asarray(instance_ids)
    labeled_all = instance_ids >= 0
    n_inst = int(instance_ids[labeled_all].max()) + 1 if labeled_all.any() else 0
    totals = np.bincount(instance_ids[labeled_all], minlength=n_inst)
    out = []
    for r in doc["regions"]:
        runtime = shift_prism(r["prism"], [-v for v in offset])
        idx = prism_indices(positions, runtime)
        inst_in = instance_ids[idx]
        counts = (np.bincount(inst_in[inst_in >= 0], minlength=n_inst)
                  if n_inst else np.zeros(0, dtype=int))
        out.append({
            "id": r["id"],
            "n_points": int(len(idx)),
            "n_unlabeled": int((class_ids[idx] < 0).sum()),
            "instances": {int(i): {"inside": int(counts[i]), "total": int(totals[i])}
                          for i in np.nonzero(counts)[0]},
        })
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest backend/tests/test_regions.py -q`
Expected: all PASS.

- [ ] **Step 5: Run the full backend suite (no regressions)**

Run: `.venv/bin/pytest backend/tests -q`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/labeling/regions.py backend/tests/test_regions.py
git commit -m "feat: eval-region store, gate, and stats (labeling/regions.py)"
```

---

### Task 2: Region routes + schemas + dense-scene fixture

**Files:**
- Create: `backend/routes/regions.py`
- Modify: `backend/app/schemas.py` (append), `backend/main.py:35,48`, `backend/tests/conftest.py`
- Test: `backend/tests/test_regions_endpoints.py`

- [ ] **Step 1: Extend conftest with custom-points scene support**

In `backend/tests/conftest.py`, change `write_scene_ply` and `build_annotated_root` to accept optional explicit points, and add a dense-scene fixture. Modify `write_scene_ply`:

```python
def write_scene_ply(path: Path, n: int = 8, pts: np.ndarray | None = None) -> None:
    if pts is None:
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((n, 3)).astype(np.float32)
    n = len(pts)
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    arr = np.zeros(n, dtype=dtype)
    arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr['red'] = arr['green'] = arr['blue'] = 200
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(str(path))
```

In `build_annotated_root`, add a `pts: np.ndarray | None = None` keyword. When `pts` is None everything stays byte-identical to today (n=8 random points, the existing `inst` array). When `pts` is given:
- call `write_scene_ply(scan_dir / "source" / "scan.ply", pts=pts)`,
- write `"n_points": len(pts)` into `meta.json`,
- use `inst = np.full(len(pts), -1, dtype=np.int32); inst[:2] = 0` and a one-segment summary `{"segments": [{"id": 0, "class_id": 0}]}` for `register_preseg`,
- pass `n_points=len(pts)` to `create_session`.

Then append two fixtures after `client_with_loaded_annotated_scene`:

```python
def dense_grid_pts(spacing=0.005, n=20, offset=(0.0, 0.0, 0.0)):
    """n×n XZ grid at the given spacing (nn distance == spacing), optionally
    translated — offsets > 1e3 trigger the load-time auto-recenter."""
    g = np.arange(n, dtype=np.float32) * spacing
    xs, zs = np.meshgrid(g, g)
    pts = np.stack([xs.ravel(), np.zeros(n * n, dtype=np.float32), zs.ravel()], axis=1)
    return (pts + np.asarray(offset, dtype=np.float32)).astype(np.float32)


@pytest.fixture
def client_with_dense_annotated_scene(monkeypatch, tmp_path):
    """Loaded annotated scene whose 400 points sit on a 5 mm grid — dense
    enough to pass the eval-grade gate (p90 = 5 mm <= 10 mm)."""
    import main
    from fastapi.testclient import TestClient

    root, _sid = build_annotated_root(tmp_path, pts=dense_grid_pts())
    monkeypatch.setattr("app.constants.LIDAR_ROOT", root, raising=False)
    client = TestClient(main.app)
    r = client.post("/api/load", json={"name": "annotated/demo", "max_points": 100})
    assert r.status_code == 200
    return client
```

Run: `.venv/bin/pytest backend/tests -q` — the existing suite must stay green after this refactor before moving on.

- [ ] **Step 2: Write the failing endpoint tests**

Create `backend/tests/test_regions_endpoints.py`:

```python
"""Tests for the /api/regions routes (eval-labeling phase 1)."""
from __future__ import annotations

import json

import numpy as np
import pytest

PRISM = {"polygon": [[-0.01, -0.01], [0.2, -0.01], [0.2, 0.2], [-0.01, 0.2]],
         "y0": -0.5, "height": 1.0}


def test_409_without_session(client):
    assert client.get("/api/regions").status_code == 409
    assert client.post("/api/regions", json={"prism": PRISM}).status_code == 409
    assert client.get("/api/regions/stats").status_code == 409


def test_crud_roundtrip(client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    assert client.get("/api/regions").json() == {"regions": []}

    r = client.post("/api/regions", json={"prism": PRISM})
    assert r.status_code == 200
    region = r.json()
    assert region["id"] == 1
    assert region["name"] == "Region 1"
    assert region["status"] == "draft"

    # persisted at the SCAN root, not the session dir
    on_disk = json.loads((scan_dir_for_loaded_scene / "eval_regions.json").read_text())
    assert on_disk["next_region_id"] == 2

    r = client.patch("/api/regions/1", json={"name": "skid A"})
    assert r.status_code == 200 and r.json()["name"] == "skid A"

    r = client.patch("/api/regions/1", json={"prism": {**PRISM, "height": 2.0}})
    assert r.status_code == 200 and r.json()["prism"]["height"] == 2.0

    assert client.delete("/api/regions/1").status_code == 200
    assert client.get("/api/regions").json() == {"regions": []}


def test_validation_and_404(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    bad = {"polygon": [[0, 0], [1, 0]], "y0": 0.0, "height": 1.0}
    assert client.post("/api/regions", json={"prism": bad}).status_code == 422
    assert client.patch("/api/regions/99", json={"name": "x"}).status_code == 404
    assert client.delete("/api/regions/99").status_code == 404
    assert client.patch("/api/regions/1", json={}).status_code == 422  # empty patch (no region needed — 404 checked first? create one)


def test_gate_refuses_sparse_fixture(client_with_loaded_annotated_scene):
    # The default fixture has 8 points — under the 100-point floor.
    client = client_with_loaded_annotated_scene
    client.post("/api/regions", json={"prism": {"polygon": [[-10, -10], [10, -10], [10, 10], [-10, 10]],
                                                "y0": -10.0, "height": 20.0}})
    r = client.patch("/api/regions/1", json={"status": "eval_grade"})
    assert r.status_code == 422
    assert "100" in r.json()["detail"]


def test_gate_passes_on_dense_scene_and_locks(client_with_dense_annotated_scene):
    client = client_with_dense_annotated_scene
    client.post("/api/regions", json={"prism": PRISM})
    r = client.patch("/api/regions/1", json={"status": "eval_grade"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "eval_grade"
    assert body["accuracy"]["p90"] == pytest.approx(0.005, abs=1e-4)
    assert body["accuracy"]["loa"] == "LOA40"

    # locked: geometry edits and delete refuse; rename still fine
    assert client.patch("/api/regions/1", json={"prism": PRISM}).status_code == 422
    assert client.delete("/api/regions/1").status_code == 422
    assert client.patch("/api/regions/1", json={"name": "ok"}).status_code == 200

    # draft flip unlocks + clears accuracy
    r = client.patch("/api/regions/1", json={"status": "draft"})
    assert r.status_code == 200 and "accuracy" not in r.json()
    assert client.delete("/api/regions/1").status_code == 200


def test_stats(client_with_dense_annotated_scene):
    client = client_with_dense_annotated_scene
    client.post("/api/regions", json={"prism": PRISM})
    r = client.get("/api/regions/stats")
    assert r.status_code == 200
    s = r.json()["regions"][0]
    assert s["id"] == 1
    assert s["n_points"] == 400          # whole grid inside
    # dense fixture labels inst 0 on its first 2 points (conftest)
    assert s["n_unlabeled"] == 398
    assert s["instances"] == {"0": {"inside": 2, "total": 2}}


def test_frame_roundtrip_with_recenter(monkeypatch, tmp_path):
    """A scene whose coords exceed 1e3 triggers _recenter; the stored file
    must hold stored-frame geometry while the API speaks the runtime frame."""
    import main
    from fastapi.testclient import TestClient
    from tests.conftest import build_annotated_root, dense_grid_pts

    off = (5000.0, 0.0, 3000.0)
    root, _sid = build_annotated_root(tmp_path, pts=dense_grid_pts(offset=off))
    monkeypatch.setattr("app.constants.LIDAR_ROOT", root, raising=False)
    client = TestClient(main.app)
    r = client.post("/api/load", json={"name": "annotated/demo", "max_points": 100})
    assert r.status_code == 200
    recenter = r.json()["recenter_offset"]
    assert abs(recenter[0]) > 1e3        # recenter actually triggered

    # Create in the RUNTIME frame (near origin, like the viewer sees it).
    client.post("/api/regions", json={"prism": PRISM})
    stored = json.loads((tmp_path / "lidar" / "annotated" / "demo" /
                         "eval_regions.json").read_text())
    sx, sz = stored["regions"][0]["prism"]["polygon"][0]
    # stored = runtime + recenter_offset
    assert sx == pytest.approx(PRISM["polygon"][0][0] + recenter[0], abs=1e-3)
    assert sz == pytest.approx(PRISM["polygon"][0][1] + recenter[2], abs=1e-3)
    # ...and it comes back out in the runtime frame.
    out = client.get("/api/regions").json()["regions"][0]["prism"]["polygon"][0]
    assert out[0] == pytest.approx(PRISM["polygon"][0][0], abs=1e-3)
    # The gate works across the frame shift too (grid is 5 mm).
    assert client.patch("/api/regions/1", json={"status": "eval_grade"}).status_code == 200
```

Also fix `test_validation_and_404`: create a region first, then check `client.patch("/api/regions/1", json={})` → 422.

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_regions_endpoints.py -q`
Expected: FAIL — 404s (routes don't exist yet).

- [ ] **Step 4: Add schemas**

Append to `backend/app/schemas.py`:

```python
class RegionPrism(BaseModel):
    polygon: list[tuple[float, float]]
    y0: float
    height: float


class CreateRegionRequest(BaseModel):
    prism: RegionPrism
    name: str | None = None


class PatchRegionRequest(BaseModel):
    name: str | None = None
    prism: RegionPrism | None = None
    status: str | None = None   # 'draft' | 'eval_grade' (validated in labeling.regions)
```

- [ ] **Step 5: Implement `backend/routes/regions.py`**

```python
"""Voxa API routes: eval regions (scan-level, eval-labeling phase 1).

Regions are SCAN truth — the store lives at the scan root and is shared by
every session on the scan — but all routes still require an active session:
CRUD needs the scan dir (resolved from the session dir), and stats/gate need
the session's FULL-RES positions (the ``_state`` cloud is subsampled to
VOXA_MAX_POINTS, which would inflate a measured p90). The wire format is the
runtime (recentered) frame; the file is the stored frame.
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.schemas import *  # noqa: F401,F403
from app.core import *  # noqa: F401,F403
from routes.segment import _require_session_seg

from labeling import regions as regstore

router = APIRouter()


def _ctx():
    seg = _require_session_seg()
    scan_dir = seg.session_dir.parent.parent      # sessions/<id>/ -> scan root
    offset = [float(v) for v in (_state.get("recenter_offset") or (0.0, 0.0, 0.0))]
    return seg, scan_dir, offset


def _to_runtime(region: dict, offset) -> dict:
    out = dict(region)
    out["prism"] = regstore.shift_prism(region["prism"], [-v for v in offset])
    return out


@router.get("/api/regions")
def list_regions():
    _seg, scan_dir, off = _ctx()
    doc = regstore.load_regions(scan_dir)
    return {"regions": [_to_runtime(r, off) for r in doc["regions"]]}


@router.post("/api/regions")
def create_region(req: CreateRegionRequest):
    _seg, scan_dir, off = _ctx()
    doc = regstore.load_regions(scan_dir)
    try:
        region = regstore.create_region(
            doc, regstore.shift_prism(req.prism.model_dump(), off), req.name)
    except regstore.RegionError as e:
        raise HTTPException(422, str(e))
    regstore.save_regions(scan_dir, doc)
    return _to_runtime(region, off)


@router.patch("/api/regions/{rid}")
def patch_region(rid: int, req: PatchRegionRequest):
    seg, scan_dir, off = _ctx()
    if req.name is None and req.prism is None and req.status is None:
        raise HTTPException(422, "empty patch — send name, prism, or status")
    doc = regstore.load_regions(scan_dir)
    try:
        region = None
        if req.name is not None:
            region = regstore.rename_region(doc, rid, req.name)
        if req.prism is not None:
            region = regstore.set_geometry(
                doc, rid, regstore.shift_prism(req.prism.model_dump(), off))
        if req.status is not None:
            region = regstore.flip_status(doc, rid, req.status, seg.positions, off)
    except regstore.RegionNotFound as e:
        raise HTTPException(404, str(e))
    except regstore.RegionError as e:
        raise HTTPException(422, str(e))
    regstore.save_regions(scan_dir, doc)
    return _to_runtime(region, off)


@router.delete("/api/regions/{rid}")
def delete_region(rid: int):
    _seg, scan_dir, off = _ctx()
    doc = regstore.load_regions(scan_dir)
    try:
        regstore.delete_region(doc, rid)
    except regstore.RegionNotFound as e:
        raise HTTPException(404, str(e))
    except regstore.RegionError as e:
        raise HTTPException(422, str(e))
    regstore.save_regions(scan_dir, doc)
    return {"ok": True}


@router.get("/api/regions/stats")
def regions_stats():
    seg, scan_dir, off = _ctx()
    doc = regstore.load_regions(scan_dir)
    return {"regions": regstore.region_stats(
        doc, seg.positions, seg.class_ids, seg.instance_ids, off)}
```

**Route-ordering caveat:** register `GET /api/regions/stats` — FastAPI matches `/api/regions/{rid}` only for PATCH/DELETE (different methods), so there is no collision with `stats`; no reordering needed.

- [ ] **Step 6: Register the router**

In `backend/main.py` line 35, add `regions` to the import: `from routes import compare, export, load, meta, regions, sam, segment, sessions` — and confirm the registration loop at ~line 48 iterates that tuple (if the modules are listed explicitly, add `regions` there too).

- [ ] **Step 7: Run tests to verify they pass**

Run: `.venv/bin/pytest backend/tests/test_regions_endpoints.py backend/tests/test_regions.py -q`
Expected: all PASS.

- [ ] **Step 8: Full backend suite**

Run: `.venv/bin/pytest backend/tests -q`
Expected: all PASS (conftest refactor + new routes, no regressions).

- [ ] **Step 9: Commit**

```bash
git add backend/routes/regions.py backend/app/schemas.py backend/main.py \
        backend/tests/conftest.py backend/tests/test_regions_endpoints.py
git commit -m "feat: /api/regions routes — scan-level CRUD, eval-grade gate, stats"
```

---

### Task 3: API client methods (`frontend/src/api.js`)

**Files:**
- Modify: `frontend/src/api.js` (append inside `VoxaAPI`, after `fitBox`)
- Test: `frontend/src/api.test.js` (append)

- [ ] **Step 1: Write the failing test**

Append to `frontend/src/api.test.js` (it already imports `vi`, `afterEach`; note the existing `afterEach(() => vi.unstubAllGlobals())` pattern — add one in this describe block if not global):

```js
describe('VoxaAPI regions', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('regionPatch surfaces the backend detail on 422', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify({ detail: 'measured p90 spacing 23.0 mm exceeds the 10 mm eval-grade bar' }),
      { status: 422 },
    )));
    await expect(VoxaAPI.regionPatch(1, { status: 'eval_grade' }))
      .rejects.toMatchObject({ status: 422, detail: expect.stringContaining('p90') });
  });

  it('regionsList unwraps the regions array', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify({ regions: [{ id: 1, name: 'Region 1', status: 'draft' }] }),
      { status: 200 },
    )));
    expect(await VoxaAPI.regionsList()).toEqual([{ id: 1, name: 'Region 1', status: 'draft' }]);
  });
});
```

- [ ] **Step 2: Run test to verify it fails**

Run: `npx vitest run src/api.test.js --root frontend`
Expected: FAIL — `VoxaAPI.regionPatch is not a function`.

- [ ] **Step 3: Implement the client methods**

Append inside `VoxaAPI` (after `fitBox`), using the shared `throwApiError` so the gate's 422 detail reaches the panel:

```js
  // --- eval regions (scan-level; see 2026-07-21-eval-regions-design.md) ---
  async regionsList() {
    const r = await fetch('/api/regions');
    if (!r.ok) await throwApiError(r, 'regionsList');
    return (await r.json()).regions;
  },
  async regionCreate({ prism, name = null }) {
    const r = await fetch('/api/regions', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ prism, name }),
    });
    if (!r.ok) await throwApiError(r, 'regionCreate');
    return r.json();
  },
  async regionPatch(id, patch) {
    const r = await fetch(`/api/regions/${id}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(patch),
    });
    if (!r.ok) await throwApiError(r, 'regionPatch');
    return r.json();
  },
  async regionDelete(id) {
    const r = await fetch(`/api/regions/${id}`, { method: 'DELETE' });
    if (!r.ok) await throwApiError(r, 'regionDelete');
    return r.json();
  },
  async regionStats() {
    const r = await fetch('/api/regions/stats');
    if (!r.ok) await throwApiError(r, 'regionStats');
    return (await r.json()).regions;
  },
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `npx vitest run src/api.test.js --root frontend`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api.js frontend/src/api.test.js
git commit -m "feat: VoxaAPI region client methods"
```

---

### Task 4: Extract the prism draw machinery (`frontend/src/prism-draw.jsx`)

Pure refactor — no behavior change. The Region tool reuses the footprint+height interaction; the classify-specific parts (chords, class picker, apply) stay in `prism-mode.jsx`.

**Files:**
- Create: `frontend/src/prism-draw.jsx`
- Modify: `frontend/src/prism-mode.jsx`

- [ ] **Step 1: Move the shared machinery**

Create `frontend/src/prism-draw.jsx` and MOVE (cut, don't copy) these from `prism-mode.jsx`, exporting each: `EMPTY_PRISM`, `MIN_HEIGHT`, `CLICK_PX`, `ACCENT`, `liveHeight`, `edgePlane`, `nearestEdge`, `fanTris`, `wallTris`, `fillMesh`, `markerRadius`, `PrismRubberBand`, `PrismOverlay`. Keep the imports they need (`react`, `three`, `evtToNdc` from `./viewer.jsx`, `prismShapeFromCorners` from `./prism-geom.js`). Add one parameter while moving — `fillMesh(positions, opacity, color = ACCENT)` (pass `color` into the material) — the region overlay needs non-accent fills; all existing callers are unchanged by the default.

Give the file a header comment:

```js
// prism-draw.jsx — the shared footprint+height draw machinery (state shape,
// geometry helpers, viewport overlay + rubber band), extracted from
// prism-mode.jsx so the Region tool can reuse the exact Prism interaction
// with a different commit handler (eval-regions spec §3). Classify-specific
// logic (chords, class picker, apply) stays in prism-mode.jsx.
```

- [ ] **Step 2: Rewire `prism-mode.jsx`**

Replace the moved definitions with:

```js
import {
  EMPTY_PRISM, MIN_HEIGHT, CLICK_PX, ACCENT, liveHeight, nearestEdge,
  PrismOverlay, PrismRubberBand,
} from './prism-draw.jsx';
```

(Drop whichever of `MIN_HEIGHT`/`CLICK_PX`/`ACCENT` are no longer referenced in `prism-mode.jsx` after the move — check with a grep — and remove the now-unused `three`/`evtToNdc` imports if nothing else in the file uses them. `fanTris`/`wallTris`/`fillMesh`/`markerRadius`/`edgePlane` are only used by the moved components, so they should not need re-importing.)

- [ ] **Step 3: Run the full frontend suite**

Run: `npx vitest run --root frontend`
Expected: all PASS (this is a move; `prism-geom` tests and any component tests must stay green).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/prism-draw.jsx frontend/src/prism-mode.jsx
git commit -m "refactor: extract shared prism draw machinery to prism-draw.jsx"
```

---### Task 5: Region pure helpers (`frontend/src/region-utils.js`)

**Files:**
- Create: `frontend/src/region-utils.js`
- Test: `frontend/src/region-utils.test.js`

- [ ] **Step 1: Write the failing tests**

```js
import { describe, expect, it } from 'vitest';
import { majorityInstances, unlabeledPct, regionCssColor, REGION_COLORS } from './region-utils.js';

const stat = {
  id: 1, n_points: 200, n_unlabeled: 30,
  instances: { 7: { inside: 90, total: 100 }, 8: { inside: 40, total: 100 }, 9: { inside: 60, total: 100 } },
};
const instances = [
  { id: 'a', segId: 7, confirmed: true, label: 'pump' },
  { id: 'b', segId: 8, confirmed: true, label: 'pipe' },   // 40% — not majority
  { id: 'c', segId: 9, confirmed: false, label: 'tank' },  // unconfirmed — excluded
];

describe('majorityInstances', () => {
  it('keeps only confirmed instances with >50% of points inside, sorted by fraction', () => {
    const out = majorityInstances(stat, instances);
    expect(out.map((o) => o.inst.segId)).toEqual([7]);
    expect(out[0].frac).toBeCloseTo(0.9);
  });
  it('handles missing stats and empty inputs', () => {
    expect(majorityInstances(null, instances)).toEqual([]);
    expect(majorityInstances(stat, [])).toEqual([]);
    expect(majorityInstances({ ...stat, instances: {} }, instances)).toEqual([]);
  });
  it('ignores instances without a finite segId (legacy cuboids)', () => {
    expect(majorityInstances(stat, [{ id: 'x', segId: null, confirmed: true }])).toEqual([]);
  });
});

describe('unlabeledPct', () => {
  it('returns the unlabeled percentage', () => {
    expect(unlabeledPct(stat)).toBeCloseTo(15);
  });
  it('returns null for empty or missing regions', () => {
    expect(unlabeledPct(null)).toBeNull();
    expect(unlabeledPct({ n_points: 0, n_unlabeled: 0 })).toBeNull();
  });
});

describe('regionCssColor', () => {
  it('maps statuses to css colors matching REGION_COLORS', () => {
    expect(regionCssColor('draft')).toBe('#f59e0b');
    expect(regionCssColor('eval_grade')).toBe('#22c55e');
    expect(REGION_COLORS.draft).toBe(0xf59e0b);
    expect(REGION_COLORS.eval_grade).toBe(0x22c55e);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/region-utils.test.js --root frontend`
Expected: FAIL (module missing).

- [ ] **Step 3: Implement**

```js
// region-utils.js — pure helpers for the eval-regions feature (phase 1).
// Membership rule: an instance is "part of" a region iff >50% of its points
// are inside (settled 2026-07-21; boundary-crossers below the bar simply
// don't list). Confirmed-status lives frontend-side, so the majority filter
// runs here over the backend's raw {inside,total} counts.

export const REGION_COLORS = { draft: 0xf59e0b, eval_grade: 0x22c55e }; // amber-500 / green-500

export function regionCssColor(status) {
  return status === 'eval_grade' ? '#22c55e' : '#f59e0b';
}

export function majorityInstances(statRegion, instances) {
  if (!statRegion?.instances) return [];
  const out = [];
  for (const inst of instances) {
    if (!inst.confirmed || !Number.isFinite(inst.segId)) continue;
    const s = statRegion.instances[inst.segId];
    if (!s || !s.total) continue;
    const frac = s.inside / s.total;
    if (frac > 0.5) out.push({ inst, frac });
  }
  return out.sort((a, b) => b.frac - a.frac);
}

export function unlabeledPct(statRegion) {
  if (!statRegion || !statRegion.n_points) return null;
  return (100 * statRegion.n_unlabeled) / statRegion.n_points;
}
```

- [ ] **Step 4: Run to verify pass, then commit**

Run: `npx vitest run src/region-utils.test.js --root frontend` → PASS.

```bash
git add frontend/src/region-utils.js frontend/src/region-utils.test.js
git commit -m "feat: region pure helpers (majority membership, unlabeled %, colors)"
```

---

### Task 6: `region` tool in the rail (`frontend/src/label-tools.js`)

**Files:**
- Modify: `frontend/src/label-tools.js`, test `frontend/src/label-tools.test.js`

- [ ] **Step 1: Write the failing test**

Append to `frontend/src/label-tools.test.js` (mirror its existing style — read it first):

```js
it('region tool is annotated-tier + session gated, like draw/beam', () => {
  expect(TOOLS.some((t) => t.id === 'region')).toBe(true);
  expect(toolAvailable('region', { segState: {}, isAnnotated: true })).toBe(true);
  expect(toolAvailable('region', { segState: {}, isAnnotated: false })).toBe(false);
  expect(toolAvailable('region', { segState: null, isAnnotated: true })).toBe(false);
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/label-tools.test.js --root frontend` → FAIL.

- [ ] **Step 3: Implement**

In `TOOLS`, append after `sam`:

```js
  { id: 'region',     icon: '▦', label: 'Region' },
```

In `toolAvailable`, extend the draw/beam branch (regions persist `eval_regions.json` at the scan root — annotated tier only, and the tool's panel/stats need the session):

```js
  if (id === 'draw' || id === 'beam' || id === 'region') return !!segState && !!isAnnotated;
```

- [ ] **Step 4: Run to verify pass, then commit**

Run: `npx vitest run src/label-tools.test.js --root frontend` → PASS.

```bash
git add frontend/src/label-tools.js frontend/src/label-tools.test.js
git commit -m "feat: Region as 7th rail tool (annotated + session gated)"
```

---

### Task 7: RegionMode drawing + RegionsOverlay (`frontend/src/region-mode.jsx`) and wiring

**Files:**
- Create: `frontend/src/region-mode.jsx`
- Modify: `frontend/src/tool-options.jsx`, `frontend/src/mode-label.jsx`

No new unit tests in this task (viewport/Three.js components follow the untested-component precedent of `prism-mode.jsx`/`beam-mode.jsx`; the pure logic was tested in Tasks 5–6, and Task 10 browser-verifies the interaction). The full suite still must stay green.

- [ ] **Step 1: Implement `region-mode.jsx`**

```jsx
// region-mode.jsx — Region sub-mode of Label mode (eval-labeling phase 1).
// Drawing reuses the shared prism-draw machinery: trace a footprint, aim a
// height, and the commit click immediately POSTs a draft region — no class,
// no point selection, nothing on the undo stack (geometry, like Draw/Beam).
// RegionsOverlay renders the PERSISTED regions (status-colored translucent
// volumes); visibility = all while the Region tool is active, else the
// per-region eye set from the Regions tab.

import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { VoxaAPI } from './api.js';
import { prismShapeFromCorners, footprintBaseY } from './prism-geom.js';
import {
  EMPTY_PRISM, liveHeight, nearestEdge, fanTris, wallTris, fillMesh,
  PrismOverlay, PrismRubberBand,
} from './prism-draw.jsx';
import { REGION_COLORS } from './region-utils.js';

// Capture-phase keys like PrismKeys, minus chords/classify: Enter closes the
// footprint, Backspace steps back, Esc clears-then-exits.
function RegionKeys({ prism, setPrism, onExit, onClose }) {
  const prismRef = useRef(prism);
  prismRef.current = prism;
  useEffect(() => {
    const handler = (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const p = prismRef.current;
      const canClose = p.phase === 'footprint' && p.corners.length >= 3;
      let handled = true;
      if (e.key === 'Enter') {
        if (canClose) onClose(); else handled = false;
      } else if (e.key === 'Escape') {
        if (p.corners.length > 0) setPrism(EMPTY_PRISM); else onExit();
      } else if (e.key === 'Backspace' || e.key === 'Delete') {
        if (p.phase === 'height') {
          setPrism((s) => ({ ...EMPTY_PRISM, phase: 'footprint', corners: s.corners }));
        } else if (p.corners.length > 0) {
          setPrism((s) => ({ ...s, corners: s.corners.slice(0, -1) }));
        } else handled = false;
      } else handled = false;
      if (handled) { e.preventDefault(); e.stopPropagation(); }
    };
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [setPrism, onExit, onClose]);
  return null;
}

function RegionDrawPanel({ prism, onClear }) {
  const placing = prism.phase === 'footprint';
  const aiming = prism.phase === 'height' && !prism.committed;
  const h = liveHeight(prism);
  return (
    <div className="tool-options tool-options-region">
      {placing && prism.corners.length === 0 && (
        <p className="tool-opt-hint">Click points on the surface to trace a region footprint. Double-click or Enter to close (min 3).</p>
      )}
      {placing && prism.corners.length > 0 && (
        <p className="tool-opt-hint">
          {prism.corners.length} corner{prism.corners.length === 1 ? '' : 's'} · double-click or Enter to close · Backspace undo · Esc cancel
        </p>
      )}
      {aiming && (
        <p className="tool-opt-hint">
          Move the mouse to set the height, then click to create the region. Height: {h.toFixed(2)} m · Backspace to edit footprint
        </p>
      )}
      {prism.corners.length > 0 && (
        <div className="tool-opt-toggle"><button onClick={onClear}>Clear</button></div>
      )}
      <p className="tool-opt-hint">Regions appear in the Regions tab (right panel). Drawing labels nothing.</p>
    </div>
  );
}

export default function RegionMode({ viewerRef, onExit, onCreated }) {
  const [prism, setPrism] = useState(EMPTY_PRISM);
  const busyRef = useRef(false);

  const closeFootprint = useCallback((dropLast = false) => {
    const camera = viewerRef.current?.getCamera?.();
    if (!camera) return;
    setPrism((s) => {
      if (s.phase !== 'footprint') return s;
      const corners = dropLast ? s.corners.slice(0, -1) : s.corners;
      if (corners.length < 3) return { ...s, corners };
      const baseY = footprintBaseY(corners);
      return { phase: 'height', corners, baseY, heightEdge: nearestEdge(corners, baseY, camera), topY: baseY, committed: false };
    });
  }, [viewerRef]);
  const closeNow = useCallback(() => closeFootprint(false), [closeFootprint]);

  // The commit click (PrismOverlay sets committed:true) IS the create action.
  useEffect(() => {
    if (!(prism.phase === 'height' && prism.committed) || busyRef.current) return;
    const shape = prismShapeFromCorners(prism.corners, prism.topY);
    if (!shape) { setPrism(EMPTY_PRISM); return; }
    busyRef.current = true;
    VoxaAPI.regionCreate({ prism: shape })
      .then((region) => { onCreated?.(region); })
      .catch((err) => console.error('region create failed:', err))
      .finally(() => { busyRef.current = false; setPrism(EMPTY_PRISM); });
  }, [prism, onCreated]);

  return (
    <>
      <RegionKeys prism={prism} setPrism={setPrism} onExit={onExit} onClose={closeNow} />
      <PrismOverlay viewerRef={viewerRef} prism={prism} setPrism={setPrism} onClose={closeFootprint} />
      <PrismRubberBand viewerRef={viewerRef} prism={prism} />
      <RegionDrawPanel prism={prism} onClear={() => setPrism(EMPTY_PRISM)} />
    </>
  );
}

// Persisted-region volumes: translucent fill + outline per visible region,
// colored by status. Dispose-and-rebuild on change, like PrismOverlay.
export function RegionsOverlay({ viewerRef, regions, visibleIds }) {
  const layerRef = useRef(null);
  useEffect(() => {
    const v = viewerRef.current;
    if (!v?.attachOverlayGroup) return undefined;
    layerRef.current = v.attachOverlayGroup();
    return () => { layerRef.current?.remove(); layerRef.current = null; };
  }, [viewerRef]);

  useEffect(() => {
    const group = layerRef.current?.group;
    if (!group) return;
    while (group.children.length) {
      const c = group.children.pop();
      c.geometry?.dispose?.(); c.material?.dispose?.();
    }
    const noRay = (o) => { o.raycast = () => {}; return o; };
    for (const region of regions) {
      if (!visibleIds.has(region.id)) continue;
      const color = REGION_COLORS[region.status] ?? REGION_COLORS.draft;
      const { polygon, y0, height } = region.prism;
      const ring = (y) => polygon.map(([x, z]) => new THREE.Vector3(x, y, z));
      const bot = ring(y0), top = ring(y0 + height);
      group.add(fillMesh([...fanTris(bot), ...fanTris(top), ...wallTris(bot, top)], 0.08, color));
      const mat = () => new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9, depthWrite: false });
      group.add(noRay(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(bot), mat())));
      group.add(noRay(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(top), mat())));
      const seg = [];
      for (let i = 0; i < polygon.length; i++) seg.push(bot[i], top[i]);
      group.add(noRay(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(seg), mat())));
    }
  }, [regions, visibleIds]);
  return null;
}
```

- [ ] **Step 2: Add `RegionOptions` to `tool-options.jsx`**

Import `RegionMode` at the top; add before the dispatcher:

```jsx
function RegionOptions({ viewerRef, onExit, onRegionCreated }) {
  // No wrapper div: RegionMode's RegionDrawPanel already renders the
  // `tool-options` container (same as PrismMode's PrismPanel).
  return <RegionMode viewerRef={viewerRef} onExit={onExit} onCreated={onRegionCreated} />;
}
```

And in `ToolOptions`: `if (activeTool === 'region') return <RegionOptions {...props} />;`

- [ ] **Step 3: Wire `mode-label.jsx`**

All anchors verified against current `main`:

1. Near `const prismMode = activeTool === 'prism';` (line ~63): add `const regionMode = activeTool === 'region';` and extend `const subModeOwnsInput = drawMode || beamMode || prismMode || regionMode;` (line ~67) — this keeps the global class-hotkey handler inert while drawing a region.
2. Add state near the other Label-mode state hooks:

```jsx
// Eval regions (scan-level; phase 1). regions = runtime-frame list from
// GET /api/regions; regionStats keyed by region id; regionEyes = per-region
// overlay visibility while OTHER tools are active (UI-local, not persisted).
const [regions, setRegions] = useState([]);
const [regionStats, setRegionStats] = useState({});
const [regionEyes, setRegionEyes] = useState(() => new Set());
```

3. Fetch the list when a session becomes active (find the existing effect keyed on the active session / `segState` presence; add a sibling):

```jsx
useEffect(() => {
  if (!segState || !isAnnotated) { setRegions([]); setRegionStats({}); return; }
  let dead = false;
  VoxaAPI.regionsList()
    .then((rs) => { if (!dead) setRegions(rs); })
    .catch((err) => console.error('regions load failed:', err));
  return () => { dead = true; };
}, [!!segState, isAnnotated, activeSessionId]);
```

(Use the file's actual variable names for the annotated flag + session id — check how `label-tools.js`'s `ctx` is built in this file and reuse those exact names.)

4. Mount the overlay next to the other always-mounted viewport overlays (e.g. right after the Viewer / near where beam's committed layer renders):

```jsx
<RegionsOverlay viewerRef={viewerRef} regions={regions}
  visibleIds={activeTool === 'region' ? new Set(regions.map((r) => r.id)) : regionEyes} />
```

5. Thread `onRegionCreated` into the `ToolOptions` props: `onRegionCreated={(region) => setRegions((rs) => [...rs, region])}`.

- [ ] **Step 4: Full frontend suite + build**

Run: `npx vitest run --root frontend` → all PASS.
Run: `npm run build` → succeeds (catches JSX/import errors in the wired files).

- [ ] **Step 5: Commit**

```bash
git add frontend/src/region-mode.jsx frontend/src/tool-options.jsx frontend/src/mode-label.jsx
git commit -m "feat: Region tool — prism-draw regions + status-colored overlay"
```

---

### Task 8: Regions tab (`frontend/src/region-panel.jsx`) + right-panel tabs + stats refresh

**Files:**
- Create: `frontend/src/region-panel.jsx`
- Test: `frontend/src/region-panel.jsdom.test.jsx`
- Modify: `frontend/src/mode-label.jsx`

- [ ] **Step 1: Write the failing jsdom test**

Model on `sam-segment-list.jsdom.test.jsx` (`@vitest-environment jsdom` pragma + `@testing-library/react`):

```jsx
// @vitest-environment jsdom
import { describe, expect, it, vi } from 'vitest';
import { fireEvent, render, screen } from '@testing-library/react';
import RegionPanel from './region-panel.jsx';

const regions = [
  { id: 1, name: 'skid A', status: 'draft', prism: { polygon: [[0, 0], [1, 0], [1, 1]], y0: 0, height: 2 } },
  { id: 2, name: 'row B', status: 'eval_grade', prism: { polygon: [[0, 0], [1, 0], [1, 1]], y0: 0, height: 2 },
    accuracy: { p50: 0.004, p90: 0.008, loa: 'LOA40', measured_at: 'x' } },
];
const stats = {
  1: { id: 1, n_points: 200, n_unlabeled: 50, instances: { 7: { inside: 90, total: 100 } } },
  2: { id: 2, n_points: 100, n_unlabeled: 0, instances: {} },
};
const instances = [{ id: 'a', segId: 7, confirmed: true, label: 'pump', cls: 'tank' }];

function renderPanel(over = {}) {
  const props = {
    regions, stats, instances, classes: [{ id: 'tank', label: 'Tank', color: '#123456' }],
    eyes: new Set(), onToggleEye: vi.fn(), onRename: vi.fn(), onDelete: vi.fn(),
    onFlipStatus: vi.fn().mockResolvedValue(undefined), onSelectInstance: vi.fn(),
    ...over,
  };
  render(<RegionPanel {...props} />);
  return props;
}

describe('RegionPanel', () => {
  it('renders rows with status badge and unlabeled %', () => {
    renderPanel();
    expect(screen.getByText('skid A')).toBeTruthy();
    expect(screen.getByText(/25% unlabeled/)).toBeTruthy();     // 50/200
    expect(screen.getByText(/eval-grade/)).toBeTruthy();
  });
  it('delete is offered on draft rows only', () => {
    const p = renderPanel();
    const dels = screen.getAllByTitle('Delete region');
    expect(dels).toHaveLength(1);
    fireEvent.click(dels[0]);
    expect(p.onDelete).toHaveBeenCalledWith(1);
  });
  it('mark eval-grade calls onFlipStatus; eval-grade rows offer back-to-draft', () => {
    const p = renderPanel();
    fireEvent.click(screen.getByText('Mark eval-grade'));
    expect(p.onFlipStatus).toHaveBeenCalledWith(1, 'eval_grade');
    fireEvent.click(screen.getByText('Back to draft'));
    expect(p.onFlipStatus).toHaveBeenCalledWith(2, 'draft');
  });
  it('expanding a row lists majority-inside confirmed instances and selects on click', () => {
    const p = renderPanel();
    fireEvent.click(screen.getByText('skid A'));               // expand
    const row = screen.getByText(/pump/);
    expect(row.textContent).toMatch(/90\s*%/);
    fireEvent.click(row);
    expect(p.onSelectInstance).toHaveBeenCalledWith(instances[0]);
  });
  it('eye toggle reports the region id', () => {
    const p = renderPanel();
    fireEvent.click(screen.getAllByTitle(/overlay/i)[0]);
    expect(p.onToggleEye).toHaveBeenCalledWith(1);
  });
});
```

- [ ] **Step 2: Run to verify failure**

Run: `npx vitest run src/region-panel.jsdom.test.jsx --root frontend` → FAIL (module missing).

- [ ] **Step 3: Implement `region-panel.jsx`**

Self-contained presentational component (all data + callbacks via props, like `SamSegmentList`). Requirements the test pins:

```jsx
// region-panel.jsx — the Regions tab of Label mode's right panel (eval-
// labeling phase 1). Presentational: regions + stats + instances come in as
// props; every mutation goes out through a callback so mode-label.jsx owns
// the API calls and state. Membership = majority-inside (region-utils.js).

import { useState } from 'react';
import { majorityInstances, unlabeledPct, regionCssColor } from './region-utils.js';

export default function RegionPanel({
  regions, stats, instances, classes, eyes,
  onToggleEye, onRename, onDelete, onFlipStatus, onSelectInstance,
}) {
  const [expanded, setExpanded] = useState(null);   // region id
  const [editingName, setEditingName] = useState(null); // {id, value}
  const [flipError, setFlipError] = useState(null);     // {id, message}

  if (!regions.length) {
    return <div className="sugg-empty">No regions yet. Pick the Region tool and trace a footprint.</div>;
  }
  return (
    <div className="inst-list region-list">
      {regions.map((region) => {
        const stat = stats[region.id];
        const pct = unlabeledPct(stat);
        const isOpen = expanded === region.id;
        const evalGrade = region.status === 'eval_grade';
        const members = isOpen ? majorityInstances(stat, instances) : [];
        return (
          <div key={region.id} className={'inst-row region-row' + (isOpen ? ' open' : '')}>
            <div className="inst-row-hd" onClick={() => setExpanded(isOpen ? null : region.id)}>
              <span className="class-swatch" style={{ background: regionCssColor(region.status) }} />
              {editingName?.id === region.id ? (
                <input className="ins-input" autoFocus value={editingName.value}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => setEditingName({ id: region.id, value: e.target.value })}
                  onBlur={() => { onRename(region.id, editingName.value); setEditingName(null); }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') e.target.blur();
                    if (e.key === 'Escape') setEditingName(null);
                  }} />
              ) : (
                <span onDoubleClick={(e) => { e.stopPropagation(); setEditingName({ id: region.id, value: region.name }); }}>
                  {region.name}
                </span>
              )}
              <span className="badge-soft">{evalGrade ? 'eval-grade' : 'draft'}</span>
              {pct != null && <span className="badge-soft">{Math.round(pct)}% unlabeled</span>}
              <button className="ghost-btn" title={eyes.has(region.id) ? 'Hide overlay' : 'Show overlay while other tools are active'}
                onClick={(e) => { e.stopPropagation(); onToggleEye(region.id); }}>
                {eyes.has(region.id) ? '●' : '◌'}
              </button>
              {!evalGrade && (
                <button className="ghost-btn danger" title="Delete region"
                  onClick={(e) => { e.stopPropagation(); onDelete(region.id); }}>×</button>
              )}
            </div>
            {/* Flip actions are ALWAYS visible (not behind expansion) — the
                jsdom test clicks them on two different rows in one test. */}
            <div className="region-actions">
              {evalGrade ? (
                <button className="ghost-btn" onClick={() => onFlipStatus(region.id, 'draft')}>Back to draft</button>
              ) : (
                <button className="ghost-btn" onClick={() => {
                  setFlipError(null);
                  Promise.resolve(onFlipStatus(region.id, 'eval_grade'))
                    .catch((err) => setFlipError({ id: region.id, message: err.detail || err.message }));
                }}>Mark eval-grade</button>
              )}
              {flipError?.id === region.id && <p className="tool-opt-hint danger">{flipError.message}</p>}
            </div>
            {isOpen && (
              <div className="region-detail">
                {evalGrade && (
                  <p className="tool-opt-hint">
                    p50 {(region.accuracy.p50 * 1000).toFixed(1)} mm · p90 {(region.accuracy.p90 * 1000).toFixed(1)} mm · {region.accuracy.loa} — geometry locked
                  </p>
                )}
                <div className="region-members">
                  {members.length === 0 && <p className="tool-opt-hint">No confirmed instance is majority-inside this region.</p>}
                  {members.map(({ inst, frac }) => {
                    const cls = classes.find((c) => c.id === inst.cls);
                    return (
                      <div key={inst.id} className="inst-row region-member" onClick={() => onSelectInstance(inst)}>
                        <span className="class-swatch" style={{ background: cls?.color }} />
                        <span>{inst.label || cls?.label || inst.cls}</span>
                        <span className="badge-soft">{Math.round(frac * 100)} %</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
```

Adjust to the test until green; keep CSS classnames from the existing panel vocabulary (`inst-row`, `badge-soft`, `ghost-btn`, `sugg-empty`, `class-swatch`, `tool-opt-hint`) so it inherits styling; add any `region-*` rules needed to `frontend/src/app.css` (flex row, small gaps — match neighboring styles).

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run src/region-panel.jsdom.test.jsx --root frontend` → PASS.

- [ ] **Step 5: Mount as a right-panel tab in `mode-label.jsx`**

1. Add `const [sideRTab, setSideRTab] = useState('instances');` near `sideRCollapsed`.
2. In the `side-hd` (line ~1665), replace the static `<span>Instances</span>` with two tab buttons (reuse the `pill-group`/`pill` or `tool-opt-toggle` pattern already in the file):

```jsx
<div className="tool-opt-toggle side-tabs">
  <button className={sideRTab === 'instances' ? 'active' : ''} onClick={() => setSideRTab('instances')}>Instances</button>
  <button className={sideRTab === 'regions' ? 'active' : ''} onClick={() => setSideRTab('regions')}>
    Regions{regions.length ? ` (${regions.length})` : ''}
  </button>
</div>
```

3. Wrap the existing instances content (filter box, status toggle, `inst-list`, context menu) in `{sideRTab === 'instances' && (...)}` and add:

```jsx
{sideRTab === 'regions' && (
  <RegionPanel regions={regions} stats={regionStats} instances={instances} classes={classes}
    eyes={regionEyes}
    onToggleEye={(id) => setRegionEyes((s) => {
      const next = new Set(s); next.has(id) ? next.delete(id) : next.add(id); return next;
    })}
    onRename={(id, name) => VoxaAPI.regionPatch(id, { name })
      .then((r) => setRegions((rs) => rs.map((x) => (x.id === id ? r : x))))
      .catch((err) => console.error('region rename failed:', err))}
    onDelete={(id) => VoxaAPI.regionDelete(id)
      .then(() => setRegions((rs) => rs.filter((x) => x.id !== id)))
      .catch((err) => console.error('region delete failed:', err))}
    onFlipStatus={(id, status) => VoxaAPI.regionPatch(id, { status })
      .then((r) => { setRegions((rs) => rs.map((x) => (x.id === id ? r : x))); })}
    onSelectInstance={(inst) => { setSelectedId(inst.id); focusInstance(inst); }} />
)}
```

(`onFlipStatus` deliberately does NOT catch — RegionPanel catches and shows the 422 detail inline. Use the file's real setters: `setSelectedId`, `focusInstance` exist; verify names before wiring.)

- [ ] **Step 6: Stats fetch + refresh after label ops**

Add a debounced stats effect in `mode-label.jsx` — `segState` identity changes on every apply/undo/confirm delta, which is exactly the refresh signal the spec asks for:

```jsx
// Region stats (unlabeled %, instance overlap) refresh whenever labels
// change — segState is replaced on every apply/undo/redo delta. Debounced:
// prism containment over the full-res arrays is cheap but not free.
useEffect(() => {
  if (!regions.length || !segState) { setRegionStats({}); return; }
  let dead = false;
  const t = setTimeout(() => {
    VoxaAPI.regionStats()
      .then((list) => { if (!dead) setRegionStats(Object.fromEntries(list.map((s) => [s.id, s]))); })
      .catch((err) => console.error('region stats failed:', err));
  }, 400);
  return () => { dead = true; clearTimeout(t); };
}, [regions, segState]);
```

- [ ] **Step 7: Full suite + build, then commit**

Run: `npx vitest run --root frontend` → all PASS.
Run: `npm run build` → succeeds.

```bash
git add frontend/src/region-panel.jsx frontend/src/region-panel.jsdom.test.jsx \
        frontend/src/mode-label.jsx frontend/src/app.css
git commit -m "feat: Regions tab — status flip, unlabeled %, majority-inside instances"
```

---

### Task 9: Docs

**Files:**
- Modify: `CLAUDE.md`, `docs/superpowers/specs/2026-07-21-eval-regions-design.md`

- [ ] **Step 1: CLAUDE.md**

Add a Region bullet to the Label-mode tool list (after SAM, before "Cut selection"), covering: 7th rail tool, prism-drawn scan-level regions in `<scan>/eval_regions.json` (stored frame, runtime conversion via `recenter_offset`), `backend/routes/regions.py` + `backend/labeling/regions.py`, server-owned eval-grade gate (p90 ≤ 10 mm over ≥ 100 full-res points via `raw_sample_spacing`) + geometry lock, Regions tab (majority-inside confirmed instances, unlabeled %), overlay visibility rule (tool-active + eye toggles), not on the undo stack, and the spec pointer. Match the density/voice of the neighboring bullets.

- [ ] **Step 2: Spec status**

Flip the spec header to `**Status:** Implemented (2026-07-XX, branch feat/eval-regions)`.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md docs/superpowers/specs/2026-07-21-eval-regions-design.md
git commit -m "docs: Region tool in CLAUDE.md; spec status -> implemented"
```

---

### Task 10: Full verification

- [ ] **Step 1: Full test suite**

Run: `npm test`
Expected: all backend + frontend tests PASS. Paste the tail of the output into the task report.

- [ ] **Step 2: Browser verification** (REQUIRED — use the `browser-verification` skill)

⚠️ Per project memory: Label-mode applies auto-save to disk — use a **throwaway session** on a real annotated scan, and restart any stale backend on :8765 first (`npm run dev` after a kill; backend has no autoreload).

Verify, with screenshots:
1. Region tool visible + enabled on an annotated scan (disabled on `legacy/test_scene`).
2. Draw a region: footprint trace → Enter/double-click close → aim height → click → region appears in the Regions tab and as a translucent amber volume.
3. Switch to the Box tool: overlay disappears; toggle the region's eye in the Regions tab: overlay reappears while Box is active.
4. Regions tab: unlabeled % shows; apply a Box label inside the region and watch the % drop after the debounce.
5. Confirm an instance mostly inside the region → it lists under the expanded region row with its fraction; clicking it selects/focuses.
6. "Mark eval-grade" on a sparse region → inline error with the measured p90 / point-floor detail (422 surfaced, not swallowed). On a dense scan (if available) → badge flips green, p50/p90/LOA shown, delete/geometry-edit refused until "Back to draft".
7. Reload the page → regions still there (scan-level persistence); zero console errors; all `/api/regions*` calls 2xx (except the deliberate 422).
8. Delete the throwaway session when done.

- [ ] **Step 3: Commit any verification fixes, then hand off**

Follow `superpowers:finishing-a-development-branch` — the PR should carry the spec, plan, implementation, and docs together.
