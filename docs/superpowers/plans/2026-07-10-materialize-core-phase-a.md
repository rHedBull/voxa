# Materialize Core (Phase A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the backend *materialize core* — a library function that produces per-point `(positions, colors, class_ids, instance_ids)` for a session at any target density (≤`scan.ply` exact; up to full-raw via NN + volumetric-shape replay) — plus the raw-cloud resolver and the `raw_source_available` flag the export wizard depends on. No HTTP endpoint, no UI in this phase.

**Architecture:** A new pure-ish module `backend/labeling/materialize.py`. Regime A (target ≤ `scan.ply`) is `subsample + index`. Regime B (denser, up to raw) streams the raw LAZ into the display frame and, per target point, resolves its label by a **max-`seq` rule**: volumetric instances (Box OBB, Draw tube) rasterize exactly and compete by apply-order `seq`; non-volumetric (preseg/legacy) come from nearest-`scan.ply`-neighbor; a box defends its own interior. Reuses `shapes.obb_indices`, `centerline.tube_indices`, `core._to_display_frame`, `lidar_io.load_laz`, and `scan_schema.Registry`.

**Tech Stack:** Python, NumPy, SciPy `cKDTree`, laspy (via existing `load_laz`), pytest.

**Spec:** `docs/superpowers/specs/2026-07-10-resolution-independent-labels-design.md` §3–§4 (the materialize algorithm + guarantees) and `docs/superpowers/specs/2026-07-10-export-wizard-design.md` (Phase A is this doc's prerequisite; it names `raw_source_available` as a Phase-A deliverable).

**Prerequisite already shipped (Phase 1):** every instance carries `seq` (apply-order) and Box pointsets persist their OBB (`center/size/rotation`). Draw tubes live in `centerlines.json`. This plan consumes those.

**Key design decisions (from the spec, restated so the implementer needn't re-derive):**
- **Volume discriminator = `source ∈ {'box','draw'}`.** Everything else (`preseg`, `manual`, `fit`, legacy `kind:'cuboid'`) is non-volumetric → NN.
- **Two regimes.** `target ≤ len(scan.ply)` → regime A (index, exact). Denser → regime B (replay). The identity case is regime A, so regime B is never asked to reproduce the working array exactly — its guarantee is: volumetric boundaries exact, semantic boundaries NN-approximate, precedence by `seq`.
- **Max-`seq` rule (regime B), per target point `p`:** candidates = every volume `V` with `p ∈ shape(V)` → `(V, seq_V)`; plus a baseline from the nearest `scan.ply` sample's owner `O` — if `O` is non-volumetric → `(O, seq_O)`; if `O` is volumetric and `p ∈ shape(O)` → `(O, seq_O)` (interior defended); if `O` is volumetric and `p ∉ shape(O)` → discard `O`, re-query the nearest **non-volumetric** sample. Winner = max-`seq`; no candidate → background (`-1`). A `-1` owner yields no candidate.

---

## File Structure

- **Create** `backend/labeling/materialize.py` — the core: `raw_sample_spacing`, `collect_volumes`, `replay_labels` (regime B rule), `materialize_downsample` (regime A), `materialize_raw` (regime B streaming), and a top-level `materialize`.
- **Modify** `backend/scenes/scene_registry.py` — broaden raw resolution to fall back to `derivation → sources.json`.
- **Modify** `backend/app/schemas.py` — add `raw_source_available` to `LoadResponse`.
- **Modify** `backend/routes/load.py` — populate `raw_source_available`.
- **Create** `backend/tests/test_materialize.py`; extend an existing test for the raw resolver / load response.

Each task is independently committable. Tasks 2–6 are pure NumPy and need no scene fixtures.

---

### Task 1: Raw resolver + `raw_source_available`

**Files:**
- Modify: `backend/scenes/scene_registry.py:123-149` (the `source_laz_path` resolution block in `_discover_annotated`)
- Modify: `backend/app/schemas.py` (`LoadResponse`)
- Modify: `backend/routes/load.py` (response construction, near the other `LoadResponse` fields ~166-196)
- Test: `backend/tests/test_load_raw_source.py` (new)

Today `source_laz_path` resolves only from `meta.source_laz` (null on regenerated scans). Add a fallback: `meta.derivation.root.source_id` → `scan_schema.Registry` → `lidar_root / entry.path`.

- [ ] **Step 1: Failing test** — an annotated scan whose `meta.source_laz` is null but whose `derivation.root.source_id` is registered in `raw/sources.json` resolves a raw path and reports `raw_source_available: true`; a scan with neither reports `false` and `None`.

Write `backend/tests/test_load_raw_source.py`. Use the existing annotated-scene test fixtures (see `backend/tests/conftest.py`); if the fixture scan has no lineage, synthesize a minimal `sources.json` + `meta.derivation.root.source_id` in a tmp lidar root (mirror `test_rename_scans.py`'s on-disk fixture style). Assert `POST /api/load` returns `raw_source_available` correctly and that `SceneSource.extras["source_laz_path"]` resolves.

- [ ] **Step 2: Run → fails** (`raw_source_available` not a field yet). `.venv/bin/pytest backend/tests/test_load_raw_source.py -v`

- [ ] **Step 3: Implement.**
  - In `scene_registry.py`, after the existing `meta.source_laz` lookup, if `source_laz_path` is still `None`, try the registry:
    ```python
    if source_laz_path is None:
        try:
            from scan_schema import Registry  # VCS dep already in requirements
            deriv = (meta.get("derivation") or {})
            root = (deriv.get("root") or {})
            sid = root.get("source_id")
            if sid and lidar_root:
                reg = Registry.load(lidar_root)          # reads raw/sources.json
                entry = reg.root(sid)
                if entry:
                    cand = lidar_root / entry.path       # "raw/<file>.laz"
                    if cand.exists():
                        source_laz_path = str(cand)
        except Exception:  # noqa: BLE001 — raw lineage is best-effort; never break discovery
            pass
    ```
    (Keep it defensive: a registry error must not break scene discovery.)
  - `schemas.py`: add to `LoadResponse` → `raw_source_available: bool = False`.
  - `load.py`: set `raw_source_available = bool(src.extras.get("source_laz_path"))` and pass it into the `LoadResponse(...)`.

- [ ] **Step 4: Run → passes.** Then full suite: `.venv/bin/pytest backend/tests/ -q -p no:warnings`.
- [ ] **Step 5: Commit.** `git add -A && git commit -m "feat(materialize): resolve raw cloud via derivation→sources.json; add raw_source_available"`

---

### Task 2: Regime A — downsample materialize

**Files:** Create `backend/labeling/materialize.py`; Test `backend/tests/test_materialize.py`

- [ ] **Step 1: Failing tests.**
```python
import numpy as np
from labeling.materialize import materialize_downsample

def _cloud(n):
    rng = np.random.default_rng(0)
    pos = rng.random((n, 3)).astype(np.float32)
    col = (rng.random((n, 3)) * 255).astype(np.uint8)
    cls = rng.integers(-1, 4, n).astype(np.int8)
    inst = rng.integers(-1, 10, n).astype(np.int32)
    return pos, col, cls, inst

def test_downsample_identity():
    pos, col, cls, inst = _cloud(100)
    o = materialize_downsample(pos, col, cls, inst, n=100)
    assert np.array_equal(o[2], cls) and np.array_equal(o[3], inst)
    assert len(o[0]) == 100

def test_downsample_indexes_labels():
    pos, col, cls, inst = _cloud(100)
    p2, c2, cl2, in2 = materialize_downsample(pos, col, cls, inst, n=30)
    assert len(p2) == 30
    # every output label equals the source label at the chosen index (no interpolation)
    # (reconstruct the index set deterministically and compare)
    from app.core import _subsample_indices
    idx = _subsample_indices(100, 30)
    assert np.array_equal(cl2, cls[idx]) and np.array_equal(in2, inst[idx])
```

- [ ] **Step 2: Run → fails** (module/func missing).
- [ ] **Step 3: Implement** `materialize_downsample(positions, colors, class_ids, instance_ids, n)`:
```python
import numpy as np
from app.core import _subsample_indices  # seeded, ascending indices into range(N)

def materialize_downsample(positions, colors, class_ids, instance_ids, n):
    """Regime A: exact down-sample. n >= len -> identity; else index every array
    by the same seeded subsample so labels are transferred, never interpolated."""
    N = len(positions)
    if n >= N:
        return positions, colors, class_ids, instance_ids
    idx = _subsample_indices(N, n)
    return positions[idx], colors[idx], class_ids[idx], instance_ids[idx]
```
- [ ] **Step 4: Run → passes.**
- [ ] **Step 5: Commit.** `feat(materialize): regime A downsample (index-transfer)`

---

### Task 3: Collect volumetric instances (shapes + seq)

**Files:** Modify `backend/labeling/materialize.py`; Test `backend/tests/test_materialize.py`

Build, from a session's instance list + centerline paths, the volumetric-instance list and the seq lookup the replay needs.

- [ ] **Step 1: Failing tests** — a `source:'box'` instance yields an `obb` volume with its `center/size/rotation` + `seq`; a `source:'draw'` instance yields a `tube` volume whose paths are the union of that instance's `centerlines.json` paths + `seq`; `preseg`/legacy `cuboid` instances are NOT volumes. Provide inline dict fixtures (no scene needed).

```python
from labeling.materialize import collect_volumes

def test_collect_volumes_box_and_tube():
    instances = [
      {"id":"a","source":"box","kind":"pointset","segId":5,"seq":2,
       "center":[0,0,0],"size":[1,1,1],"rotation":[0,0,0]},
      {"id":"b","source":"draw","kind":"pointset","segId":9,"seq":3},
      {"id":"c","source":"preseg","kind":"pointset","segId":1,"seq":1},
      {"id":"d","source":"manual","kind":"cuboid","seq":0,
       "center":[9,9,9],"size":[1,1,1],"rotation":[0,0,0]},  # legacy -> NOT a volume
    ]
    centerlines = {"paths":[
      {"instance_id":9,"points":[[0,0,0],[1,0,0]],"radius":0.2,"smooth":False},
      {"instance_id":9,"points":[[1,0,0],[1,1,0]],"radius":0.2,"smooth":True},
    ]}
    vols = collect_volumes(instances, centerlines)
    kinds = {v["instance_id"]: v["kind"] for v in vols}
    assert kinds == {5:"obb", 9:"tube"}          # preseg + legacy excluded
    tube = next(v for v in vols if v["instance_id"]==9)
    assert len(tube["paths"]) == 2 and tube["seq"] == 3   # both paths unioned
    obb = next(v for v in vols if v["instance_id"]==5)
    assert obb["seq"] == 2 and obb["shape"]["size"] == [1,1,1]
```

- [ ] **Step 2: Run → fails.**
- [ ] **Step 3: Implement** `collect_volumes(instances, centerlines)`:
```python
def collect_volumes(instances, centerlines):
    """Volumetric instances only (source in {'box','draw'}). Box -> obb from
    center/size/rotation; draw -> tube from the instance's centerlines paths
    (grouped by instance_id == segId). Each carries its apply-order `seq`."""
    paths_by_inst = {}
    for p in (centerlines or {}).get("paths", []):
        paths_by_inst.setdefault(int(p["instance_id"]), []).append(p)
    out = []
    for inst in instances:
        src = inst.get("source")
        seq = inst.get("seq")
        iid = inst.get("segId")
        if src == "box" and inst.get("center") and inst.get("size"):
            out.append({"kind":"obb","instance_id":iid,"seq":seq,
                        "shape":{"center":inst["center"],"size":inst["size"],
                                 "rotation":inst.get("rotation",[0,0,0])}})
        elif src == "draw" and iid in paths_by_inst:
            out.append({"kind":"tube","instance_id":iid,"seq":seq,
                        "paths":paths_by_inst[iid]})
    return out
```
Note: instances with `seq is None` shouldn't occur post-Phase-1 (stamped on save), but the replay (Task 4) must treat a missing `seq` defensively — assign it `-inf` precedence so a real instance always wins. Document that in Task 4.

- [ ] **Step 4: Run → passes.** **Step 5: Commit.** `feat(materialize): collect volumetric instances (obb/tube + seq)`

---

### Task 4: Regime B — the max-`seq` replay rule (in-memory)

**Files:** Modify `backend/labeling/materialize.py`; Test `backend/tests/test_materialize.py`

The heart. Pure function: given `scan.ply` positions + its working `(class_ids, instance_ids)`, the volumes (Task 3) + an instance→seq map, and a **target cloud** (already in the display frame), return target `(class_ids, instance_ids)`.

- [ ] **Step 1: Failing tests** — encode the spec's cases with tiny synthetic clouds so they're unambiguous:
  - **two overlapping boxes, dense interior** (`seq_B>seq_A`): a target point inside both → instance B.
  - **box interior defended vs higher-`seq` adjacent preseg**: preseg `P` (`seq_P>seq_V`) occupies scan.ply points *next to* box `V`; a target point deep inside `V` (nearest scan.ply sample is V-owned) → `V`, not `P`.
  - **exclusion**: a target point just *outside* box `V` whose nearest sample is a V-edge point → the surrounding preseg wall, never background.
  - **legacy cuboid → NN**: a `kind:'cuboid'` instance is not a volume; its region transfers by NN only.
  - **multi-path tube**: a target point within `radius` of either path of a 2-path tube instance → that instance.
  Build each as ~10–50 point synthetic `scan.ply` + explicit shapes; assert exact target labels.

- [ ] **Step 2: Run → fails.**
- [ ] **Step 3: Implement** `replay_labels(scan_pos, work_cls, work_inst, volumes, seq_by_inst, target_pos)`:
  - Precompute per-volume membership testers reusing existing code: OBB → `shapes.obb_indices(target_pos, shape)` gives indices of target points inside; tube → `centerline.tube_indices(target_pos, paths)`. Convert each to a boolean mask over target points. Record `(mask, instance_id, seq)` per volume.
  - `vol_owned` = boolean over `scan.ply` points whose `work_inst` belongs to a volumetric instance (`instance_id ∈ {v.instance_id}`).
  - `tree_all = cKDTree(scan_pos)`; `nonvol_idx = np.where(~vol_owned)[0]`; `tree_nonvol = cKDTree(scan_pos[nonvol_idx])`.
  - For each target point (vectorize where practical):
    - baseline: nearest via `tree_all` → owner sample `s`; if `s` is vol-owned by volume `O` and target point ∉ `shape(O)` → re-query `tree_nonvol` for the baseline owner; else baseline = `s`. Baseline candidate = `(inst_at(baseline), seq_of(that inst))`; a `-1` class at baseline contributes no candidate.
    - volume candidates: for each volume mask that includes this point → `(v.instance_id, v.seq)`.
    - winner = candidate with max `seq` (treat `seq is None` as `-inf`); class = the working class of the winning instance (look up via a precomputed `inst_id -> class_id` map built from the working arrays), or `-1` if no candidate.
  - Return `(target_cls, target_inst)`.
  - **Vectorization note:** implement correctly first (a Python loop over target points is fine for the unit tests); a follow-up may vectorize the common branches. Keep the loop readable and matching the spec's rule exactly.

- [ ] **Step 4: Run → passes** (all spec cases green).
- [ ] **Step 5: Commit.** `feat(materialize): regime B max-seq replay rule (interior defense + NN baseline)`

---

### Task 5: Regime B — raw LAZ streamed through the rule

**Files:** Modify `backend/labeling/materialize.py`; Test `backend/tests/test_materialize.py`

Wire a raw LAZ (source frame, Z-up/UTM) through frame alignment into `replay_labels`, in chunks so a 156M cloud never loads whole.

- [ ] **Step 1: Failing test** — using a small existing LAZ fixture (see `backend/tests/` LAZ fixtures used by `test_export_ply_endpoint.py` / `load_laz`), materialize onto the raw cloud: assert output length == raw point count, and that points geometrically inside a defined OBB get that instance's label. Keep the fixture tiny.

- [ ] **Step 2: Run → fails.**
- [ ] **Step 3: Implement** `materialize_raw(scan_pos, work_cls, work_inst, volumes, seq_by_inst, raw_path, scene_is_z_up, offset, chunk=1_000_000)` — a generator (or accumulator) that:
  - reads the raw LAZ in chunks (reuse `lidar_io.load_laz`/its chunk iterator),
  - maps each chunk's xyz into the display frame via `core._to_display_frame(xyz, scene_is_z_up, offset)`,
  - runs `replay_labels(scan_pos, work_cls, work_inst, volumes, seq_by_inst, chunk_display_xyz)`,
  - yields `(chunk_xyz_source_or_display, chunk_rgb, chunk_cls, chunk_inst)` per chunk (the writer in Phase B decides the output frame — default: the display frame, matching how the viewer/edit-export operate).
  Build the two KD-trees once (outside the chunk loop) and pass them in, or memoize — do NOT rebuild per chunk.

- [ ] **Step 4: Run → passes.** **Step 5: Commit.** `feat(materialize): regime B raw-LAZ streaming through the replay`

---

### Task 6: Accuracy metric (labeling-density sample spacing)

**Files:** Modify `backend/labeling/materialize.py`; Test `backend/tests/test_materialize.py`

- [ ] **Step 1: Failing tests** — `raw_sample_spacing(scan_pos)` returns `(p50, p90)` with `p90 >= p50 > 0`, matching a direct nearest-neighbor compute on a small fixture within tolerance.
- [ ] **Step 2: Run → fails.**
- [ ] **Step 3: Implement** `raw_sample_spacing(scan_pos, sample=100_000, seed=0)`:
```python
from scipy.spatial import cKDTree
import numpy as np

def raw_sample_spacing(scan_pos, sample=100_000, seed=0):
    """Nearest-neighbor spacing of scan.ply (its true sampling pitch). Returns
    (p50, p90) over a bounded random subsample; p90 is the honest boundary bound
    under non-uniform LiDAR sampling. Build independent of any regime KD-tree."""
    n = len(scan_pos)
    if n < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    q = scan_pos if n <= sample else scan_pos[rng.choice(n, sample, replace=False)]
    d, _ = cKDTree(scan_pos).query(q, k=2)      # k=2: nearest non-self
    nn = d[:, 1]
    return float(np.percentile(nn, 50)), float(np.percentile(nn, 90))
```
- [ ] **Step 4: Run → passes.** **Step 5: Commit.** `feat(materialize): p50/p90 sample-spacing accuracy metric`

---

### Task 7: Top-level `materialize()` dispatcher

**Files:** Modify `backend/labeling/materialize.py`; Test `backend/tests/test_materialize.py`

Tie it together into the single entry point Phase B will call.

- [ ] **Step 1: Failing tests** — `materialize(ctx, {"kind":"scan"})` returns the working arrays at scan density with accuracy present; `{"kind":"subsample","n":k}` (k<len) returns k points; `{"kind":"raw"}` (small LAZ fixture) returns raw-length arrays. `ctx` is a small dataclass/dict carrying `scan_pos, colors, work_cls, work_inst, volumes, seq_by_inst, raw_path, scene_is_z_up, offset`.
- [ ] **Step 2: Run → fails.**
- [ ] **Step 3: Implement** `materialize(ctx, resolution)`:
  - `kind in ("scan","subsample")` → `materialize_downsample(...)` (regime A). `n = len(scan_pos)` for `"scan"`, else `resolution["n"]` (validated ≤ len upstream).
  - `kind == "raw"` → assemble from `materialize_raw(...)` chunks.
  - Always attach `accuracy = raw_sample_spacing(ctx.scan_pos)` (p50/p90) to the returned meta.
  - Return `(positions, colors, class_ids, instance_ids, meta)` where `meta` includes `accuracy` and the target point count. (The `ctx` is assembled by Phase B from `_state` + the session's working arrays + `instances_gt.json` + `centerlines.json`; Phase A only defines the function contract and unit-tests it with synthetic `ctx`.)
- [ ] **Step 4: Run → passes.** Full suite `.venv/bin/pytest backend/tests/ -q -p no:warnings`.
- [ ] **Step 5: Commit.** `feat(materialize): top-level materialize() dispatcher (regime A/B + accuracy)`

---

## Done criteria (Phase A)

- `raw_source_available` on `LoadResponse`, resolved via `derivation → sources.json` for lineage scans; `false` for lineage-less scans.
- `backend/labeling/materialize.py` exposes `materialize(ctx, resolution)` covering regime A (exact index) and regime B (raw NN + volumetric max-`seq` replay with interior defense), plus `raw_sample_spacing` (p50/p90).
- Every spec case in §3 is a green unit test (overlapping boxes, interior defense, exclusion, legacy-cuboid NN, multi-path tube).
- Full backend suite green.

## Not in this phase

- The `POST /api/labels/export` endpoint, filters/remap, PLY+zip writing, the wizard UI, and the download flow — all **Phase B** (`export-wizard` spec). Phase A ships the callable core + `raw_source_available` only.
- Vectorizing the regime-B per-point loop for speed — correctness first; optimize only if a real raw export is too slow (follow-up, `log`/measure before optimizing).

## Docs

Last step before the Phase-A PR: update the resolution-independent-labels spec's Status to note Phase 2/materialize-core is now implemented (link this plan), and add a one-line pointer in `CLAUDE.md` architecture notes that `backend/labeling/materialize.py` is the density-agnostic label materializer (regime A index / regime B replay).
