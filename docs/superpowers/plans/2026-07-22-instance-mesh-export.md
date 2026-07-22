# Instance Mesh Export Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move per-instance convex-hull `.glb` mesh generation out of the standalone `scripts/build_instance_meshes.py` and into the voxa backend's `/api/labels/export` endpoint, as an `include_meshes` option, then delete the old script.

**Architecture:** A new pure helper (`backend/labeling/instance_meshes.py`) computes a convex-hull `.glb` per surviving instance id from arrays already in memory. `export.py`'s `export_labels` endpoint gains an `include_meshes` request flag: when set, it computes the same confirmed/include-classes-filtered instance set the point export uses, calls the helper, and writes `meshes/<id>.glb` files plus a `meshes` summary into the existing export zip. The export wizard UI gets one checkbox to opt in.

**Tech Stack:** Python (FastAPI backend), `scipy.spatial.ConvexHull`, `trimesh`, `numpy`, `pytest`; React (frontend wizard).

**Spec:** `docs/superpowers/specs/2026-07-22-instance-mesh-export-design.md`

---

### Task 1: `build_instance_glbs` helper

**Files:**
- Create: `backend/labeling/instance_meshes.py`
- Test: `backend/tests/test_instance_meshes.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_instance_meshes.py
"""TDD for build_instance_glbs (instance-mesh export)."""
import numpy as np
import trimesh

from labeling.instance_meshes import MIN_POINTS_FOR_MESH, build_instance_glbs


def _cube_points(n, center=(0.0, 0.0, 0.0), scale=1.0, seed=0):
    """n points scattered on/near a cube's surface — always ≥4 non-coplanar,
    so ConvexHull succeeds regardless of n."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-scale, scale, size=(n, 3)).astype(np.float32)
    return pts + np.asarray(center, dtype=np.float32)


def test_min_points_threshold_is_100():
    assert MIN_POINTS_FOR_MESH == 100


def test_happy_path_writes_glb_for_surviving_instance():
    pts_a = _cube_points(MIN_POINTS_FOR_MESH, center=(0, 0, 0), seed=1)
    pts_b = _cube_points(5, center=(10, 0, 0), seed=2)  # below threshold
    points = np.concatenate([pts_a, pts_b])
    instance_ids = np.concatenate([
        np.zeros(len(pts_a), dtype=np.int32),
        np.ones(len(pts_b), dtype=np.int32),
    ])

    glbs, skipped = build_instance_glbs(points, instance_ids, {0, 1})

    assert set(glbs.keys()) == {0}
    mesh = trimesh.load(trimesh.util.wrap_as_stream(glbs[0]), file_type="glb")
    assert len(mesh.vertices) >= 4
    assert skipped == [(1, f"only {len(pts_b)} points")]


def test_below_threshold_is_skipped():
    pts = _cube_points(MIN_POINTS_FOR_MESH - 1, seed=3)
    instance_ids = np.zeros(len(pts), dtype=np.int32)

    glbs, skipped = build_instance_glbs(pts, instance_ids, {0})

    assert glbs == {}
    assert skipped == [(0, f"only {MIN_POINTS_FOR_MESH - 1} points")]


def test_coplanar_points_skipped_with_qhullerror_reason():
    # All points on the z=0 plane, ≥ MIN_POINTS_FOR_MESH of them — enough
    # points, but degenerate (no volume), so ConvexHull raises QhullError.
    rng = np.random.default_rng(4)
    xy = rng.uniform(-1, 1, size=(MIN_POINTS_FOR_MESH, 2)).astype(np.float32)
    pts = np.concatenate([xy, np.zeros((MIN_POINTS_FOR_MESH, 1), dtype=np.float32)], axis=1)
    instance_ids = np.zeros(len(pts), dtype=np.int32)

    glbs, skipped = build_instance_glbs(pts, instance_ids, {0})

    assert glbs == {}
    assert len(skipped) == 1
    assert skipped[0][0] == 0
    assert "coplanar" in skipped[0][1] or "QhullError" in skipped[0][1]


def test_only_requested_ids_are_considered():
    pts = _cube_points(MIN_POINTS_FOR_MESH, seed=5)
    instance_ids = np.zeros(len(pts), dtype=np.int32)

    glbs, skipped = build_instance_glbs(pts, instance_ids, set())

    assert glbs == {} and skipped == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ../.venv/bin/python -m pytest tests/test_instance_meshes.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'labeling.instance_meshes'`

- [ ] **Step 3: Write the implementation**

```python
# backend/labeling/instance_meshes.py
"""Per-instance convex-hull .glb generation for the labels-export endpoint.

Ported from the retired scripts/build_instance_meshes.py: each mesh is the
convex hull of that instance's own labeled points (not a reconstruction from
an external mesh), so it can never be in a different coordinate frame from
the point cloud that produced it. Convex, so it won't tightly hug a
bent/branching pipe, but it's real geometry, cheap, and always well-formed.
"""
from __future__ import annotations

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, QhullError

# Quality bar, not just the geometric floor (a hull needs >= 4 non-coplanar
# points to exist at all) -- a technically-valid hull from a handful of
# points is too noisy for collision detection to trust.
MIN_POINTS_FOR_MESH = 100


def build_instance_glbs(
    points: np.ndarray,
    instance_ids: np.ndarray,
    surviving_ids: set[int],
) -> tuple[dict[int, bytes], list[tuple[int, str]]]:
    """Convex-hull .glb per id in `surviving_ids`.

    Returns (glbs, skipped): glbs maps instance_id -> glb bytes; skipped
    lists (instance_id, reason) for ids with < MIN_POINTS_FOR_MESH points or
    a degenerate/coplanar hull (QhullError).
    """
    points = np.asarray(points, dtype=np.float64)
    instance_ids = np.asarray(instance_ids)

    glbs: dict[int, bytes] = {}
    skipped: list[tuple[int, str]] = []

    for asset_id in sorted(surviving_ids):
        pts = points[instance_ids == asset_id]
        if len(pts) < MIN_POINTS_FOR_MESH:
            skipped.append((asset_id, f"only {len(pts)} points"))
            continue
        try:
            hull = ConvexHull(pts)
        except QhullError as e:
            skipped.append((asset_id, f"degenerate/coplanar points ({e.__class__.__name__})"))
            continue
        mesh = trimesh.Trimesh(vertices=pts, faces=hull.simplices, process=True)
        glbs[asset_id] = mesh.export(file_type="glb")

    return glbs, skipped
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ../.venv/bin/python -m pytest tests/test_instance_meshes.py -v`
Expected: PASS (5 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/labeling/instance_meshes.py backend/tests/test_instance_meshes.py
git commit -m "feat: add build_instance_glbs helper for per-instance mesh export"
```

---

### Task 2: `include_meshes` request field

**Files:**
- Modify: `backend/app/schemas.py:362-369` (`ExportLabelsRequest`)
- Test: `backend/tests/test_export_labels.py` (schema section, near the top with the other `ExportLabelsRequest` parse tests)

- [ ] **Step 1: Write the failing test**

Add near the top of `backend/tests/test_export_labels.py` (alongside `test_parses_from_alias` etc.):

```python
def test_include_meshes_defaults_false():
    req = _base_req()
    assert req.include_meshes is False


def test_include_meshes_can_be_set_true():
    req = _base_req(include_meshes=True)
    assert req.include_meshes is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && ../.venv/bin/python -m pytest tests/test_export_labels.py -k include_meshes -v`
Expected: FAIL — `ExportLabelsRequest` has no field `include_meshes` (pydantic ignores unknown extra kwargs by default here, so `req.include_meshes` raises `AttributeError`).

- [ ] **Step 3: Add the field**

In `backend/app/schemas.py`, `ExportLabelsRequest` (currently lines 362-369):

```python
class ExportLabelsRequest(BaseModel):
    scene: str
    session_id: str
    resolution: ExportResolution
    confirmed_only: bool = False
    include_classes: Optional[list[int]] = None
    remap: list[RemapRule] = []
    drop_unlabeled: bool = False
    include_meshes: bool = False
```

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && ../.venv/bin/python -m pytest tests/test_export_labels.py -k include_meshes -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add backend/app/schemas.py backend/tests/test_export_labels.py
git commit -m "feat: add include_meshes field to ExportLabelsRequest"
```

---

### Task 3: Wire mesh generation into `export_labels`

**Files:**
- Modify: `backend/routes/export.py:195-283` (`export_labels`)
- Test: `backend/tests/test_export_labels.py` (near `test_export_labels_confirmed_only_zeros_unconfirmed`, line ~604)

**Context:** `zf` (the `zipfile.ZipFile`) only exists inside the `with zipfile.ZipFile(...) as zf:` block (currently `export.py:273-275`), which opens *after* `manifest = build_manifest(...)` (line 264) and is immediately followed by writing `scan_labeled.ply` then `manifest.json`. The mesh block must run between those two `zf` calls, mutating `manifest` before it's serialized — see the spec's "Placement matters" note.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_export_labels.py`, near `test_export_labels_confirmed_only_zeros_unconfirmed`:

```python
def test_export_labels_include_meshes_false_by_default(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    _load_demo(client, scene_id, session_id)

    r = client.post("/api/labels/export", json={
        "scene": scene_id, "session_id": session_id,
        "resolution": {"kind": "scan"},
    })
    assert r.status_code == 200, r.text
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert zf.namelist() == ["scan_labeled.ply", "manifest.json"]
    manifest = json.loads(zf.read("manifest.json"))
    assert "meshes" not in manifest


def test_export_labels_include_meshes_true_reports_skips(client_with_annotated_scene):
    # Demo fixture's 4 instances have 1-2 points each -- all below
    # MIN_POINTS_FOR_MESH (100), so every confirmed/included instance is
    # skipped, but the wiring (filters -> surviving ids -> manifest) is
    # still fully exercised.
    client, scene_id, session_id = client_with_annotated_scene
    _load_demo(client, scene_id, session_id)

    instances = [
        {"id": "i0", "cls": "pipe", "kind": "pointset", "segId": 0, "confirmed": True},
        {"id": "i1", "cls": "tank", "kind": "pointset", "segId": 1, "confirmed": False},
        {"id": "i2", "cls": "equipment", "kind": "pointset", "segId": 2, "confirmed": True},
        {"id": "i3", "cls": "equipment", "kind": "pointset", "segId": 3, "confirmed": True},
    ]
    r = client.put(
        f"/api/annotations/gt/{scene_id}?session_id={session_id}",
        json={"scene": scene_id, "kind": "gt", "instances": instances, "meta": {}},
    )
    assert r.status_code == 200, r.text

    r = client.post("/api/labels/export", json={
        "scene": scene_id, "session_id": session_id,
        "resolution": {"kind": "scan"},
        "confirmed_only": True,
        "include_meshes": True,
    })
    assert r.status_code == 200, r.text
    zf = zipfile.ZipFile(io.BytesIO(r.content))
    assert zf.namelist() == ["scan_labeled.ply", "manifest.json"]  # nothing written, all skipped
    manifest = json.loads(zf.read("manifest.json"))
    assert manifest["meshes"]["written"] == 0
    skipped_ids = {s["id"] for s in manifest["meshes"]["skipped"]}
    # instance 1 is unconfirmed -> excluded from the surviving set entirely,
    # so it must NOT appear in the meshes skip list (it never got the chance
    # to be considered -- it was filtered out upstream, not skipped for size).
    assert 1 not in skipped_ids
    assert skipped_ids == {0, 2, 3}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && ../.venv/bin/python -m pytest tests/test_export_labels.py -k include_meshes -v`
Expected: `test_export_labels_include_meshes_false_by_default` PASSES already (no code path exists yet, so no `meshes` key — this one just documents the baseline); `test_export_labels_include_meshes_true_reports_skips` FAILS with `KeyError: 'meshes'`.

- [ ] **Step 3: Wire the endpoint**

In `backend/routes/export.py`, add the import at the top (alongside the other `labeling.*` imports):

```python
from labeling.instance_meshes import build_instance_glbs
```

Change the body of `export_labels` (currently lines 264-275) from:

```python
        manifest = build_manifest(
            taxonomy, p50, p90, scan=req.scene, session=req.session_id,
            resolution={"kind": req.resolution.kind}, points=total,
            confirmed_only=req.confirmed_only, include_classes=req.include_classes,
            drop_unlabeled=req.drop_unlabeled, absent_count=absent,
            exported_at=datetime.now(timezone.utc).isoformat(),
            labeling_points=len(ctx.scan_pos))

        zip_path = Path(tmpdir) / "export.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            zf.write(ply_path, "scan_labeled.ply")
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
```

to:

```python
        manifest = build_manifest(
            taxonomy, p50, p90, scan=req.scene, session=req.session_id,
            resolution={"kind": req.resolution.kind}, points=total,
            confirmed_only=req.confirmed_only, include_classes=req.include_classes,
            drop_unlabeled=req.drop_unlabeled, absent_count=absent,
            exported_at=datetime.now(timezone.utc).isoformat(),
            labeling_points=len(ctx.scan_pos))

        zip_path = Path(tmpdir) / "export.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
            zf.write(ply_path, "scan_labeled.ply")
            if req.include_meshes:
                mesh_cls, _mesh_inst = apply_filters_remap(
                    ctx.work_cls, ctx.work_inst, confirmed_by_inst, req, src_to_tgt)
                surviving_ids = {int(i) for i in np.unique(ctx.work_inst[mesh_cls >= 0]) if i >= 0}
                glbs, skipped = build_instance_glbs(ctx.scan_pos, ctx.work_inst, surviving_ids)
                for iid, data in glbs.items():
                    zf.writestr(f"meshes/{iid}.glb", data)
                manifest["meshes"] = {
                    "written": len(glbs),
                    "skipped": [{"id": iid, "reason": reason} for iid, reason in skipped],
                }
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
```

Note `apply_filters_remap` returns `instance_ids` unchanged (by reference), so `ctx.work_inst` is passed directly to `build_instance_glbs` rather than the discarded `_mesh_inst` — both are the same array.

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && ../.venv/bin/python -m pytest tests/test_export_labels.py -v`
Expected: PASS (all tests in the file, including the two new ones)

- [ ] **Step 5: Commit**

```bash
git add backend/routes/export.py backend/tests/test_export_labels.py
git commit -m "feat: wire include_meshes into /api/labels/export"
```

---

### Task 4: Export wizard checkbox

**Files:**
- Modify: `frontend/src/export-wizard.jsx`

- [ ] **Step 1: Add state**

Near the other Classes-step state (`export-wizard.jsx:43`, `const [confirmedOnly, setConfirmedOnly] = useState(false);`):

```jsx
const [includeMeshes, setIncludeMeshes] = useState(false);
```

- [ ] **Step 2: Add the checkbox to step 2's pane**

In the `step === 2` block (`export-wizard.jsx:212-218`), directly after the existing "Confirmed instances only" `<label className="ew-check">`:

```jsx
              <label className="ew-check">
                <input type="checkbox" checked={includeMeshes}
                  onChange={(e) => setIncludeMeshes(e.target.checked)} />
                <span>Include instance meshes <em>(.glb per instance, for collision detection)</em></span>
              </label>
```

- [ ] **Step 3: Send it in the export payload**

In `doExport` (`export-wizard.jsx:143-164`), add `include_meshes` to `cfg`:

```jsx
      const cfg = {
        scene,
        session_id: sessionId,
        resolution: { kind, ...(kind === 'subsample' ? { n: subN } : {}) },
        confirmed_only: confirmedOnly,
        include_classes: includeArr,
        remap: previewRows.map((r) => ({ from: r.from, to: r.to })),
        drop_unlabeled: false,
        include_meshes: includeMeshes,
      };
```

- [ ] **Step 4: Manual verification**

Run: `cd /home/hendrik/coding/engine/tools/labeling/voxa && npm run dev` (or the project's usual dev-server command — check `package.json`/`README.md` if `npm run dev` isn't it), open the labeler, load any annotated scene, open the export wizard, confirm the new checkbox appears on step 2 and toggles without console errors. This is a UI change — per the browser-verification skill, screenshot the checkbox state and confirm no console errors before moving on. (An end-to-end zip-download check isn't necessary here since Task 3's backend tests already cover the payload wiring; this step is purely about the checkbox rendering and toggling correctly.)

- [ ] **Step 5: Commit**

```bash
git add frontend/src/export-wizard.jsx
git commit -m "feat: add include-meshes checkbox to export wizard"
```

---

### Task 5: Retire the standalone script

**Files:**
- Delete: `scripts/build_instance_meshes.py`

- [ ] **Step 1: Confirm it's untracked (safe to remove without git history loss)**

Run: `cd /home/hendrik/coding/engine/tools/labeling/voxa && git status --short scripts/build_instance_meshes.py`
Expected: `?? scripts/build_instance_meshes.py` (untracked)

- [ ] **Step 2: Delete it**

Run: `rm /home/hendrik/coding/engine/tools/labeling/voxa/scripts/build_instance_meshes.py`

- [ ] **Step 3: Confirm no other file references it**

Run: `grep -rn "build_instance_meshes" /home/hendrik/coding/engine/tools/labeling/voxa --include="*.md" --include="*.py" --include="*.jsx" --include="*.json" | grep -v ".worktrees\|node_modules\|docs/superpowers"`
Expected: no output (only the spec/plan docs under `docs/superpowers/` mention it, which is fine — they're historical records, not live references).

- [ ] **Step 4: Run the full backend test suite once more**

Run: `cd backend && ../.venv/bin/python -m pytest -q`
Expected: PASS, same count as before minus nothing (the deleted file had no tests referencing it).

No commit needed for this step alone — since the file was never tracked, `git status` won't show it as a change. Confirm with:

```bash
git status --short
```

Expected: no entry for `scripts/build_instance_meshes.py` (it simply no longer exists on disk, and git never knew about it).

---

### Task 6: Full regression pass

- [ ] **Step 1: Run the whole backend suite**

Run: `cd /home/hendrik/coding/engine/tools/labeling/voxa/backend && ../.venv/bin/python -m pytest -q`
Expected: all tests PASS.

- [ ] **Step 2: Run the simplify skill on the diff**

Invoke `simplify` (per this project's usual workflow) over the full diff from Tasks 1-5 before opening a PR — check for reuse/simplification opportunities, e.g. whether `build_instance_glbs`'s per-id loop could share more with `segment_hulls.py`'s grouping approach (note: they intentionally differ on the degenerate-case fallback, per the spec, so don't unify that part — just check for accidental duplication elsewhere).

- [ ] **Step 3: Final commit if simplify made changes**

```bash
git add -A
git commit -m "refactor: simplify instance-mesh export per simplify pass"
```

(Skip this step if `simplify` found nothing to change.)
