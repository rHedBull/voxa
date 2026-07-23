# Region Density Gate: Raw-Source Measurement Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the eval-grade region gate and the export accuracy readouts measure point-cloud density against the scan's registered raw source instead of the session's subsampled working cloud, so eval-grade eligibility stops depending on viewer/session point-cap size.

**Architecture:** Two new pure functions in `backend/labeling/materialize.py` (`raw_region_sample_spacing` for a region's own AABB+prism, `raw_reservoir_sample_spacing` for a whole-scan chunk-level sample) both wrap existing raw-LAZ-streaming infrastructure (`scenes.lidar_io.load_laz_region`, `_laz_chunk_iter`) and feed into the existing `raw_sample_spacing()` KD-tree calc. Three call sites switch to them: `labeling/regions.py::flip_status` (hard-gates: refuses eval-grade without a raw source), `GET /api/labels/accuracy` (soft-degrades: falls back to session cloud without raw source), and `POST /api/labels/export`'s manifest stamping (raw-backed only when the export itself is at raw resolution).

**Tech Stack:** Python/FastAPI backend, numpy/scipy (`cKDTree`), laspy for LAZ/LAS I/O, pytest.

**Spec:** `docs/superpowers/specs/2026-07-23-region-density-raw-source-design.md` — read it in full before starting; this plan implements it task-by-task and does not repeat its rationale.

---

## Task 1: `raw_region_sample_spacing` — region-scoped raw density

**Files:**
- Modify: `backend/labeling/materialize.py`
- Test: `backend/tests/test_materialize.py`

This adds a prism-AABB helper and `raw_region_sample_spacing`, which streams a raw LAZ filtered to a region's AABB (via the existing `scenes.lidar_io.load_laz_region`), re-filters through the exact prism (via the already-imported `prism_indices`), and measures spacing via the existing `raw_sample_spacing`.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_materialize.py` (near the existing `raw_sample_spacing` tests at the bottom of the file; reuse the `_write_tiny_las_at` helper already defined earlier in that file):

```python
# ---------------------------------------------------------------------------
# Raw-source region density (region-density-raw-source spec, Task 1)
# ---------------------------------------------------------------------------

def test_prism_aabb_covers_footprint_and_height():
    prism = {"polygon": [[0.0, 0.0], [2.0, 0.0], [2.0, 1.0], [0.0, 1.0]],
             "y0": -0.5, "height": 3.0}
    aabb_min, aabb_max = prism_aabb(prism)
    np.testing.assert_allclose(aabb_min, [0.0, -0.5, 0.0])
    np.testing.assert_allclose(aabb_max, [2.0, 2.5, 1.0])


def test_prism_aabb_handles_negative_and_unordered_polygon_coords():
    prism = {"polygon": [[3.0, -2.0], [-1.0, 5.0], [0.0, 0.0]],
             "y0": 10.0, "height": 1.5}
    aabb_min, aabb_max = prism_aabb(prism)
    np.testing.assert_allclose(aabb_min, [-1.0, 10.0, -2.0])
    np.testing.assert_allclose(aabb_max, [3.0, 11.5, 5.0])


def test_raw_region_sample_spacing_measures_only_in_prism_points(tmp_path):
    # A dense 5mm grid INSIDE the prism, plus a sparse decoy cluster OUTSIDE
    # the prism's AABB, plus a dense cluster inside the AABB but outside the
    # exact (non-rectangular-adjacent) prism footprint — only the first group
    # should drive the measured spacing.
    inside = [[x * 0.005, 0.0, z * 0.005] for x in range(20) for z in range(20)]
    outside_aabb = [[50.0, 0.0, 50.0], [51.0, 0.0, 51.0]]
    # Inside the AABB (x in [-0.5, 1.5]) but outside the triangular prism
    # footprint below, at a much sparser (20mm) spacing that would fail the
    # gate if wrongly included.
    outside_prism_inside_aabb = [[1.4 + i * 0.02, 0.0, 1.4] for i in range(5)]
    points = inside + outside_aabb + outside_prism_inside_aabb
    las_path = tmp_path / "region_raw.las"
    _write_tiny_las_at(las_path, points)

    prism = {"polygon": [[-0.5, -0.5], [1.5, -0.5], [1.5, 1.5], [-0.5, -0.4]],
             "y0": -0.1, "height": 0.2}
    # Triangle-ish quad chosen so (1.4, 1.4) sits outside it while the 5mm
    # grid (bounded [0, 0.095]) sits comfortably inside.

    p50, p90 = raw_region_sample_spacing(
        las_path, prism, scene_is_z_up=False, offset=np.zeros(3))
    assert p50 == pytest.approx(0.005, abs=1e-3)
    assert p90 == pytest.approx(0.005, abs=1e-3)


def test_raw_region_sample_spacing_empty_region_returns_zero(tmp_path):
    points = [[50.0, 0.0, 50.0], [51.0, 0.0, 51.0]]
    las_path = tmp_path / "region_raw_empty.las"
    _write_tiny_las_at(las_path, points)
    prism = {"polygon": [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]],
             "y0": -0.5, "height": 1.0}
    p50, p90 = raw_region_sample_spacing(
        las_path, prism, scene_is_z_up=False, offset=np.zeros(3))
    assert (p50, p90) == (0.0, 0.0)
```

Add the needed imports at the top of `test_materialize.py` if not already present:

```python
from labeling.materialize import (
    ...,  # keep existing imports
    prism_aabb,
    raw_region_sample_spacing,
)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_materialize.py -k "prism_aabb or raw_region_sample_spacing" -v` (or `../.venv/bin/pytest` from repo root, matching whatever the repo's existing `.venv` path is — see `npm run test:backend` / `scripts/test.sh` for the exact invocation this repo uses)

Expected: FAIL with `ImportError` / `AttributeError: module 'labeling.materialize' has no attribute 'prism_aabb'`

- [ ] **Step 3: Implement `prism_aabb` and `raw_region_sample_spacing`**

Add to `backend/labeling/materialize.py`, near `raw_sample_spacing`:

```python
def prism_aabb(prism: dict) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of a vertical prism (footprint polygon in
    XZ + a Y band), as (min, max) float64 (x, y, z) arrays. Used to pre-filter
    a raw point stream to a small region before the exact prism_indices test —
    no equivalent helper exists elsewhere (shapes.py only tests membership,
    never bounds)."""
    poly = np.asarray(prism["polygon"], dtype=np.float64)
    y0 = float(prism["y0"])
    height = float(prism["height"])
    x_min, x_max = float(poly[:, 0].min()), float(poly[:, 0].max())
    z_min, z_max = float(poly[:, 1].min()), float(poly[:, 1].max())
    return (np.array([x_min, y0, z_min]),
            np.array([x_max, y0 + height, z_max]))


def raw_region_sample_spacing(raw_path, prism: dict, scene_is_z_up: bool,
                               offset: np.ndarray) -> tuple[float, float]:
    """p50/p90 nearest-neighbour spacing of a scan's raw source, scoped to one
    eval region. Streams+filters the raw LAZ to the prism's AABB (cheap: only
    touches points local to one region, never the whole file in memory), then
    re-filters through the exact prism so points just outside a non-rectangular
    footprint (but inside its AABB) can't skew the measurement."""
    from scenes.lidar_io import load_laz_region

    aabb_min, aabb_max = prism_aabb(prism)
    positions, _colors = load_laz_region(
        raw_path, aabb_min.astype(np.float32), aabb_max.astype(np.float32),
        is_z_up=scene_is_z_up, offset=np.asarray(offset, dtype=np.float64))
    if len(positions) == 0:
        return 0.0, 0.0
    idx = prism_indices(positions, prism)
    return raw_sample_spacing(positions[idx])
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_materialize.py -k "prism_aabb or raw_region_sample_spacing" -v`

Expected: PASS (all 4 new tests)

- [ ] **Step 5: Run the full materialize test file to check for regressions**

Run: `cd backend && .venv/bin/pytest tests/test_materialize.py -v`

Expected: PASS (all tests, old and new)

- [ ] **Step 6: Commit**

```bash
git add backend/labeling/materialize.py backend/tests/test_materialize.py
git commit -m "feat: add raw_region_sample_spacing for region-scoped raw density"
```

---

## Task 2: `raw_reservoir_sample_spacing` — whole-scan raw density via chunk sampling

**Files:**
- Modify: `backend/labeling/materialize.py`
- Test: `backend/tests/test_materialize.py`

This is the corrected (round-2-review) design: reservoir-sample whole **chunks**, not individual points, so every point that ends up in the KD-tree corpus is at true native density.

- [ ] **Step 1: Write the failing tests**

Add to `backend/tests/test_materialize.py`:

```python
def test_raw_reservoir_sample_spacing_bounded_chunk_count(tmp_path):
    # 12 chunks of 500 pts each (6000 total) at a known 10mm grid spacing;
    # n_chunks=3 must still measure close to the true 10mm spacing, proving
    # no per-point thinning happened within a retained chunk.
    pts = [[x * 0.01, 0.0, z * 0.01] for x in range(60) for z in range(100)]
    assert len(pts) == 6000
    las_path = tmp_path / "reservoir_raw.las"
    _write_tiny_las_at(las_path, pts)

    p50, p90 = raw_reservoir_sample_spacing(
        las_path, scene_is_z_up=False, offset=np.zeros(3),
        n_chunks=3, chunk=500, seed=0)
    assert p50 == pytest.approx(0.01, abs=2e-3)
    assert p90 == pytest.approx(0.01, abs=2e-3)


def test_raw_reservoir_sample_spacing_seeded_deterministic(tmp_path):
    pts = [[x * 0.01, 0.0, z * 0.01] for x in range(30) for z in range(30)]
    las_path = tmp_path / "reservoir_raw_seed.las"
    _write_tiny_las_at(las_path, pts)

    a = raw_reservoir_sample_spacing(
        las_path, scene_is_z_up=False, offset=np.zeros(3),
        n_chunks=2, chunk=100, seed=7)
    b = raw_reservoir_sample_spacing(
        las_path, scene_is_z_up=False, offset=np.zeros(3),
        n_chunks=2, chunk=100, seed=7)
    assert a == b


def test_raw_reservoir_sample_spacing_fewer_chunks_than_n_chunks(tmp_path):
    # Only 2 chunks exist in the file; n_chunks=5 must not error, just use
    # everything available.
    pts = [[x * 0.005, 0.0, 0.0] for x in range(150)]
    las_path = tmp_path / "reservoir_raw_small.las"
    _write_tiny_las_at(las_path, pts)

    p50, p90 = raw_reservoir_sample_spacing(
        las_path, scene_is_z_up=False, offset=np.zeros(3),
        n_chunks=5, chunk=100, seed=0)
    assert p50 == pytest.approx(0.005, abs=1e-3)
```

Update the import block added in Task 1 to also pull in `raw_reservoir_sample_spacing`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_materialize.py -k "reservoir" -v`

Expected: FAIL with `AttributeError: ... has no attribute 'raw_reservoir_sample_spacing'`

- [ ] **Step 3: Implement `raw_reservoir_sample_spacing`**

Add to `backend/labeling/materialize.py`:

```python
def raw_reservoir_sample_spacing(raw_path, scene_is_z_up: bool, offset: np.ndarray,
                                  n_chunks: int = 5, chunk: int = 1_000_000,
                                  seed: int = 0) -> tuple[float, float]:
    """p50/p90 nearest-neighbour spacing of a scan's raw source, sampled
    across the WHOLE file (no AABB to filter by, unlike
    raw_region_sample_spacing). Reservoir-samples whole CHUNKS via Algorithm R
    over the chunk stream — not individual points — so every point in the
    resulting corpus is at true native local density; a flat point-level
    reservoir would collapse density by the reservoir/file-size ratio and
    silently defeat the point of measuring "raw" density at all (see the
    design spec's round-2 correction).

    ASSUMES a `_laz_chunk_iter` chunk (sequential points in on-disk order) is
    spatially coherent, which holds for typical LiDAR/NavVis scan-order
    exports but is not guaranteed for a pre-shuffled or globally-tiled file —
    acceptable for this informational (non-gating) measurement.
    """
    from app.core import _to_display_frame
    from scenes.lidar_io import _laz_chunk_iter

    rng = np.random.default_rng(seed)
    reservoir: list[np.ndarray] = []
    seen = 0
    for _hdr, las_chunk in _laz_chunk_iter(raw_path, chunk_size=chunk):
        xyz_src = np.column_stack([
            np.asarray(las_chunk.x, dtype=np.float64),
            np.asarray(las_chunk.y, dtype=np.float64),
            np.asarray(las_chunk.z, dtype=np.float64),
        ])
        display_xyz = _to_display_frame(xyz_src, scene_is_z_up, np.asarray(offset))
        if len(reservoir) < n_chunks:
            reservoir.append(display_xyz)
        else:
            j = int(rng.integers(0, seen + 1))
            if j < n_chunks:
                reservoir[j] = display_xyz
        seen += 1

    if not reservoir:
        return 0.0, 0.0
    positions = np.concatenate(reservoir, axis=0).astype(np.float32)
    return raw_sample_spacing(positions, seed=seed)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_materialize.py -k "reservoir" -v`

Expected: PASS (all 3 new tests)

- [ ] **Step 5: Run the full materialize test file**

Run: `cd backend && .venv/bin/pytest tests/test_materialize.py -v`

Expected: PASS

- [ ] **Step 6: Commit**

```bash
git add backend/labeling/materialize.py backend/tests/test_materialize.py
git commit -m "feat: add raw_reservoir_sample_spacing with chunk-level reservoir sampling"
```

---

## Task 3: `flip_status` measures raw density; refuses without a raw source

**Files:**
- Modify: `backend/labeling/regions.py`
- Test: `backend/tests/test_regions.py`

- [ ] **Step 1: Write the failing tests**

`test_regions.py` needs a real tiny-LAS writer (mirroring `_write_tiny_las_at` in `test_materialize.py` — each test file keeps its own copy, matching existing convention). Add near the top of `backend/tests/test_regions.py`, after the `grid_positions` helper:

```python
def _write_tiny_las_at(path, points):
    """Minimal LAS 1.2 file at explicit XYZ points, for raw-source gate tests."""
    import laspy

    n = len(points)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.0001, 0.0001, 0.0001])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    pts = np.asarray(points, dtype=np.float64)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    rng = np.random.default_rng(1)
    las.red = rng.integers(0, 65535, n).astype(np.uint16)
    las.green = rng.integers(0, 65535, n).astype(np.uint16)
    las.blue = rng.integers(0, 65535, n).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(path))
```

Then modify the existing gate tests to pass `raw_path`/`scene_is_z_up`, and add the new raw-specific tests. Replace the whole `# ---- gate` section (from `def test_gate_passes_on_5mm_grid_and_records_accuracy` through `def test_gate_unknown_status_rejected`) with:

```python
def test_gate_passes_on_5mm_grid_and_records_accuracy(tmp_path):
    grid = grid_positions(0.005)
    las_path = tmp_path / "gate_raw.las"
    _write_tiny_las_at(las_path, grid.tolist())
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, {"polygon": [[-0.01, -0.01], [0.2, -0.01], [0.2, 0.2], [-0.01, 0.2]],
                                "y0": -0.5, "height": 1.0})
    out = reg.flip_status(doc, r["id"], "eval_grade", grid,
                          raw_path=las_path, scene_is_z_up=False)
    assert out["status"] == "eval_grade"
    acc = out["accuracy"]
    assert acc["p90"] == pytest.approx(0.005, abs=1e-4)
    assert acc["loa"] == "LOA40"
    assert "measured_at" in acc
    # n_points now reflects the RAW region's point count, not len(positions).
    assert acc["n_points"] == 400


def test_gate_refuses_on_empty_raw_region(tmp_path):
    """Zero raw points in the region (e.g. the prism sits somewhere the raw
    file has no coverage) must hit the point-floor refusal, not the
    coincident-points ("p50 ~ 0") refusal — the floor check must run BEFORE
    the spacing measurement, since raw_region_sample_spacing legitimately
    returns (0.0, 0.0) for an empty input and that must not be confused with
    a real "0 spacing" measurement."""
    las_path = tmp_path / "gate_raw_far_away.las"
    _write_tiny_las_at(las_path, [[500.0, 0.0, 500.0], [501.0, 0.0, 501.0]])
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, PRISM)
    with pytest.raises(reg.RegionError, match="100"):
        reg.flip_status(doc, r["id"], "eval_grade", grid_positions(0.005),
                        raw_path=las_path, scene_is_z_up=False)


def test_gate_measures_raw_not_session_cloud(tmp_path):
    """The exact bim/water_treatment scenario: a SPARSE session cloud (would
    fail the 10mm bar) but a DENSE raw source (passes) -> gate passes,
    proving the measurement now comes from raw_path, not `positions`."""
    sparse_session = grid_positions(0.020, n=10)   # 20mm spacing, would fail
    dense_raw = grid_positions(0.003, n=40).tolist()  # 3mm spacing, passes
    las_path = tmp_path / "dense_raw.las"
    _write_tiny_las_at(las_path, dense_raw)

    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, {"polygon": [[-0.01, -0.01], [0.1, -0.01], [0.1, 0.1], [-0.01, 0.1]],
                                "y0": -0.5, "height": 1.0})
    out = reg.flip_status(doc, r["id"], "eval_grade", sparse_session,
                          raw_path=las_path, scene_is_z_up=False)
    assert out["status"] == "eval_grade"
    assert out["accuracy"]["p90"] == pytest.approx(0.003, abs=1e-3)


def test_gate_refuses_without_raw_source():
    """No raw source registered -> refuse outright, regardless of how dense
    the session cloud is. Never silently fall back to `positions`."""
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, PRISM)
    dense_session = grid_positions(0.001, n=40)   # would easily pass on its own
    with pytest.raises(reg.RegionError, match="raw source"):
        reg.flip_status(doc, r["id"], "eval_grade", dense_session,
                        raw_path=None, scene_is_z_up=False)
    assert doc["regions"][0]["status"] == "draft"


def test_gate_refuses_20mm_grid(tmp_path):
    grid = grid_positions(0.020)
    las_path = tmp_path / "gate_raw_sparse.las"
    _write_tiny_las_at(las_path, grid.tolist())
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, {"polygon": [[-0.1, -0.1], [0.5, -0.1], [0.5, 0.5], [-0.1, 0.5]],
                                "y0": -0.5, "height": 1.0})
    with pytest.raises(reg.RegionError, match="p90"):
        reg.flip_status(doc, r["id"], "eval_grade", grid,
                        raw_path=las_path, scene_is_z_up=False)
    assert doc["regions"][0]["status"] == "draft"


def test_gate_refuses_below_point_floor(tmp_path):
    # 100-point floor now applies to the RAW region's point count.
    few = grid_positions(0.005, n=5).tolist()   # 25 pts inside
    las_path = tmp_path / "gate_raw_sparse_floor.las"
    _write_tiny_las_at(las_path, few)
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, PRISM)
    with pytest.raises(reg.RegionError, match="100"):
        reg.flip_status(doc, r["id"], "eval_grade", np.asarray(few, dtype=np.float32),
                        raw_path=las_path, scene_is_z_up=False)


def test_gate_refuses_zero_spacing_sample(tmp_path):
    dup = np.tile(np.array([[0.5, 0.0, 0.5]], dtype=np.float32), (400, 1))
    las_path = tmp_path / "gate_raw_dup.las"
    _write_tiny_las_at(las_path, dup.tolist())
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, PRISM)
    with pytest.raises(reg.RegionError, match="coincident"):
        reg.flip_status(doc, r["id"], "eval_grade", dup,
                        raw_path=las_path, scene_is_z_up=False)
    assert doc["regions"][0]["status"] == "draft"


def test_gate_unknown_status_rejected(tmp_path):
    grid = grid_positions(0.005).tolist()
    las_path = tmp_path / "gate_raw_unknown.las"
    _write_tiny_las_at(las_path, grid)
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, PRISM)
    with pytest.raises(reg.RegionError):
        reg.flip_status(doc, r["id"], "frozen", grid_positions(0.005),
                        raw_path=las_path, scene_is_z_up=False)
```

Also update the two remaining internal `flip_status` call sites earlier in the file — `_eval_grade_doc()` and `test_draft_flip_unlocks_and_clears_accuracy` — to pass a raw source:

```python
def _eval_grade_doc(tmp_path=None):
    import tempfile
    from pathlib import Path
    tmp_path = tmp_path or Path(tempfile.mkdtemp())
    grid = grid_positions(0.005).tolist()
    las_path = tmp_path / "eval_grade_doc_raw.las"
    _write_tiny_las_at(las_path, grid)
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, {"polygon": [[-0.01, -0.01], [0.2, -0.01], [0.2, 0.2], [-0.01, 0.2]],
                                "y0": -0.5, "height": 1.0})
    reg.flip_status(doc, r["id"], "eval_grade", grid_positions(0.005),
                    raw_path=las_path, scene_is_z_up=False)
    return doc, r["id"], las_path


def test_eval_grade_locks_geometry_and_delete(tmp_path):
    doc, rid, _las_path = _eval_grade_doc(tmp_path)
    with pytest.raises(reg.RegionError):
        reg.set_geometry(doc, rid, PRISM)
    with pytest.raises(reg.RegionError):
        reg.delete_region(doc, rid)
    reg.rename_region(doc, rid, "still fine")


def test_draft_flip_unlocks_and_clears_accuracy(tmp_path):
    doc, rid, las_path = _eval_grade_doc(tmp_path)
    reg.flip_status(doc, rid, "draft", grid_positions(0.005),
                    raw_path=las_path, scene_is_z_up=False)
    r = doc["regions"][0]
    assert r["status"] == "draft"
    assert "accuracy" not in r
    reg.set_geometry(doc, rid, PRISM)
    reg.delete_region(doc, rid)
    assert doc["regions"] == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest tests/test_regions.py -v`

Expected: FAIL — `flip_status() got an unexpected keyword argument 'raw_path'` across the gate tests.

- [ ] **Step 3: Add `raw_region_point_count` to `materialize.py`**

The point-floor check must read the RAW region's point count (per the spec: "point-floor... applies to whichever point set was actually measured"), and — critically — it must run **BEFORE** the spacing measurement: `raw_region_sample_spacing` legitimately returns `(0.0, 0.0)` for an empty region, which would otherwise be misread as "coincident points" (the `MIN_GATE_P50_M` check) rather than "too few points" (the `MIN_GATE_POINTS` check). Add this to `backend/labeling/materialize.py`, right after `raw_region_sample_spacing`:

```python
def raw_region_point_count(raw_path, prism: dict, scene_is_z_up: bool,
                            offset: np.ndarray) -> int:
    """How many raw points fall exactly inside a region's prism. Shares
    raw_region_sample_spacing's AABB-prefilter + exact prism_indices logic
    but returns a count instead of a spacing measurement — used by the
    eval-grade gate's point-floor check, which must run BEFORE the spacing
    measurement (an empty region measures spacing (0.0, 0.0), which must not
    be confused with a real "coincident points" reading)."""
    from scenes.lidar_io import load_laz_region

    aabb_min, aabb_max = prism_aabb(prism)
    positions, _colors = load_laz_region(
        raw_path, aabb_min.astype(np.float32), aabb_max.astype(np.float32),
        is_z_up=scene_is_z_up, offset=np.asarray(offset, dtype=np.float64))
    if len(positions) == 0:
        return 0
    return int(len(prism_indices(positions, prism)))
```

Add its test to `backend/tests/test_materialize.py`, alongside this task's other new tests (import `raw_region_point_count` alongside the other new imports):

```python
def test_raw_region_point_count_matches_prism_indices(tmp_path):
    grid = grid_positions(0.005, n=10).tolist()
    las_path = tmp_path / "count_raw.las"
    _write_tiny_las_at(las_path, grid)
    prism = {"polygon": [[-0.01, -0.01], [0.2, -0.01], [0.2, 0.2], [-0.01, 0.2]],
             "y0": -0.5, "height": 1.0}
    n = raw_region_point_count(las_path, prism, scene_is_z_up=False, offset=np.zeros(3))
    assert n == 100   # full 10x10 grid at 5mm sits inside this prism
```

Run: `cd backend && .venv/bin/pytest tests/test_materialize.py -k raw_region_point_count -v` — expect PASS before moving on.

- [ ] **Step 4: Update `flip_status`**

In `backend/labeling/regions.py`, update the import line (drop `raw_sample_spacing` — after this change nothing in this file calls it directly anymore) and the function body. This is the complete, final version — floor check first, spacing measurement second:

```python
from labeling.materialize import (
    loa_band, raw_region_sample_spacing, raw_region_point_count,
)
```

```python
def flip_status(doc: dict, rid: int, status: str, positions,
                offset=(0.0, 0.0, 0.0), categories=None,
                raw_path=None, scene_is_z_up: bool = False) -> dict:
    """draft <-> eval_grade. The eval_grade flip is the gate: measure the
    region's RAW-SOURCE spacing (not the session working cloud — see
    docs/superpowers/specs/2026-07-23-region-density-raw-source-design.md)
    and refuse (RegionError) if no raw source is registered, below the point
    floor, above the 10 mm bar, or over the excluded-review budget
    (`categories`, the phase-2 point-category array; None skips that check
    for callers with no session). The point-floor check runs BEFORE the
    spacing measurement: an empty raw region measures (0.0, 0.0) spacing,
    which must not be misread as "coincident points"."""
    r = _get(doc, rid)
    if status == "draft":
        r["status"] = "draft"
        r.pop("accuracy", None)
        return r
    if status != "eval_grade":
        raise RegionError(f"unknown status {status!r}")
    if r["status"] == "eval_grade":
        return r
    if raw_path is None:
        raise RegionError(
            "no raw source registered for this scan — cannot verify true "
            "density; register a raw source before flipping regions to "
            "eval-grade")
    runtime = shift_prism(r["prism"], [-v for v in offset])
    n_points = raw_region_point_count(raw_path, runtime, scene_is_z_up, offset)
    if n_points < MIN_GATE_POINTS:
        raise RegionError(
            f"region holds {n_points} raw point{'' if n_points == 1 else 's'} "
            f"— at least {MIN_GATE_POINTS} needed to measure spacing")
    p50, p90 = raw_region_sample_spacing(raw_path, runtime, scene_is_z_up, offset)
    if p50 <= MIN_GATE_P50_M:
        raise RegionError(
            "measured p50 spacing is ~0 — the region's points are coincident "
            "or duplicated, so the spacing measurement is meaningless")
    if p90 > EVAL_GRADE_P90_M:
        raise RegionError(
            f"measured p90 spacing {p90 * 1000:.1f} mm exceeds the "
            f"{EVAL_GRADE_P90_M * 1000:.0f} mm eval-grade bar")
    if categories is not None:
        # categories/review-budget stays keyed off the SESSION's own working
        # array — that array only exists at session resolution, it has no
        # raw-scoped equivalent.
        session_positions = np.asarray(positions, dtype=np.float32).reshape(-1, 3)
        session_idx = prism_indices(session_positions, runtime)
        n_review = _n_review(categories, session_idx)
        frac = n_review / max(len(session_idx), 1)
        if frac > REVIEW_BUDGET_FRAC:
            raise RegionError(
                f"{n_review} of {len(session_idx)} points ({frac * 100:.1f}%) are "
                f"excluded-review — over the "
                f"{REVIEW_BUDGET_FRAC * 100:.0f}% budget; resolve or relabel "
                f"them before flipping to eval-grade")
    r["status"] = "eval_grade"
    r["accuracy"] = {"p50": p50, "p90": p90, "loa": loa_band(p90),
                     "n_points": int(n_points), "measured_at": _now()}
    return r
```

Note `n_points` in the recorded `accuracy` dict is now the RAW region's count, not the session's — update the `test_gate_passes_on_5mm_grid_and_records_accuracy` assertion above accordingly (already done: `assert acc["n_points"] == 400`, matching the 400-point raw grid fixture in that test, not a session-array count).

Add one more explicit regression test for the raw-vs-session count distinction, alongside the other gate tests:

```python
def test_gate_point_floor_uses_raw_count_not_session_count(tmp_path):
    """A DENSE session cloud but a SPARSE raw region must still refuse on the
    point floor — proves the floor reads the raw count, not len(positions)."""
    dense_session = grid_positions(0.001, n=40)   # 1600 session points
    sparse_raw = grid_positions(0.005, n=5).tolist()  # 25 raw points
    las_path = tmp_path / "floor_raw_sparse.las"
    _write_tiny_las_at(las_path, sparse_raw)
    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, PRISM)
    with pytest.raises(reg.RegionError, match="100"):
        reg.flip_status(doc, r["id"], "eval_grade", dense_session,
                        raw_path=las_path, scene_is_z_up=False)
```

Add one more test for the edge case the `max(len(session_idx), 1)` zero-division guard in the categories/review-budget block exists to protect against — a raw region dense enough to pass the floor/spacing checks, but whose session cloud happens to have zero points inside the same prism (the two point sets are filtered independently, so this is possible: e.g. a session cloud that hasn't caught up with a since-widened region, or simply doesn't cover that area at session resolution):

```python
def test_gate_passes_with_no_session_points_in_region(tmp_path):
    """Raw region is dense enough to pass on its own; the session cloud has
    ZERO points inside the same prism (categories=None here, so the
    review-budget branch is skipped entirely — this just proves the gate
    doesn't crash on prism_indices returning empty for `positions`)."""
    dense_raw = grid_positions(0.003, n=40).tolist()
    las_path = tmp_path / "raw_dense_no_session_overlap.las"
    _write_tiny_las_at(las_path, dense_raw)
    session_elsewhere = np.array([[500.0, 0.0, 500.0]], dtype=np.float32)

    doc = {"version": 1, "next_region_id": 1, "regions": []}
    r = reg.create_region(doc, {"polygon": [[-0.01, -0.01], [0.1, -0.01], [0.1, 0.1], [-0.01, 0.1]],
                                "y0": -0.5, "height": 1.0})
    out = reg.flip_status(doc, r["id"], "eval_grade", session_elsewhere,
                          raw_path=las_path, scene_is_z_up=False)
    assert out["status"] == "eval_grade"
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest tests/test_materialize.py tests/test_regions.py -v`

Expected: PASS (all tests)

- [ ] **Step 6: Commit**

```bash
git add backend/labeling/materialize.py backend/labeling/regions.py backend/tests/test_materialize.py backend/tests/test_regions.py
git commit -m "feat: flip_status measures raw-source density, refuses without a raw source"
```

---

## Task 4: Wire `raw_path`/`scene_is_z_up` through the `/api/regions` routes

**Files:**
- Modify: `backend/routes/regions.py`

- [ ] **Step 1: Update `_ctx()` and `patch_region`**

In `backend/routes/regions.py`, change `_ctx()` to also resolve the raw source (mirroring the exact pattern already used in `routes/load.py:229` / `routes/export.py:105`):

```python
def _ctx():
    seg = _require_session_seg()
    scan_dir = seg.session_dir.parent.parent      # sessions/<id>/ -> scan root
    offset = [float(v) for v in (_state.get("recenter_offset") or (0.0, 0.0, 0.0))]
    src = _resolve(_state.get("scene"))
    raw_path = src.extras.get("source_laz_path")
    scene_is_z_up = _scene_is_z_up(src)
    return seg, scan_dir, offset, raw_path, scene_is_z_up
```

This changes `_ctx()`'s return arity from 3 to 5 — update every call site in the same file:

```python
@router.get("/api/regions")
def list_regions():
    _seg, scan_dir, off, _raw_path, _z_up = _ctx()
    doc = regstore.load_regions(scan_dir)
    return {"regions": [_to_runtime(r, off) for r in doc["regions"]]}


@router.post("/api/regions")
def create_region(req: CreateRegionRequest):
    _seg, scan_dir, off, _raw_path, _z_up = _ctx()
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
    seg, scan_dir, off, raw_path, scene_is_z_up = _ctx()
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
            region = regstore.flip_status(doc, rid, req.status, seg.positions,
                                          off, categories=seg.categories,
                                          raw_path=raw_path, scene_is_z_up=scene_is_z_up)
    except regstore.RegionNotFound as e:
        raise HTTPException(404, str(e))
    except regstore.RegionError as e:
        raise HTTPException(422, str(e))
    regstore.save_regions(scan_dir, doc)
    return _to_runtime(region, off)


@router.delete("/api/regions/{rid}")
def delete_region(rid: int):
    _seg, scan_dir, off, _raw_path, _z_up = _ctx()
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
    seg, scan_dir, off, _raw_path, _z_up = _ctx()
    doc = regstore.load_regions(scan_dir)
    return {"regions": regstore.region_stats(
        doc, seg.positions, seg.class_ids, seg.instance_ids, off,
        categories=seg.categories)}
```

`_resolve` and `_scene_is_z_up` are already available via the existing `from app.core import *` wildcard import at the top of this file — no new import needed.

- [ ] **Step 2: Run the existing endpoint test file to see what breaks**

Run: `cd backend && .venv/bin/pytest tests/test_regions_endpoints.py -v`

Expected: several FAILs — tests that expect an eval-grade flip to return 200 now get 422 (`"no raw source registered..."`) because the test fixtures don't register a raw source yet. This is expected; Task 5 fixes the fixtures. Confirm the *shape* of the failure matches this (422 with that message), not an unrelated crash (e.g. an `AttributeError` from a stale `_ctx()` call site would be a bug in this task, not a fixture gap).

- [ ] **Step 3: Commit**

```bash
git add backend/routes/regions.py
git commit -m "feat: thread raw_path/scene_is_z_up through /api/regions routes"
```

(Committing here even though `test_regions_endpoints.py` has expected failures — Task 5 is the fixture fix and lands as its own commit, keeping this diff reviewable on its own.)

---

## Task 5: Register a raw source in the endpoint-test fixtures

**Files:**
- Modify: `backend/tests/conftest.py`
- Modify: `backend/tests/test_regions_endpoints.py`

**Why this is needed:** `test_regions_endpoints.py`'s dense-scene fixtures build a synthetic annotated scan via `build_annotated_root(tmp_path, pts=dense_grid_pts())` with no raw LAZ registered — `SceneSource.extras["source_laz_path"]` resolves to `None` for these. Before this plan, that was fine (the gate read `positions` directly). After Task 3/4, `raw_path=None` makes the gate refuse outright, so any endpoint test that expects an eval-grade PATCH to succeed needs a fixture that actually registers a matching raw source.

- [ ] **Step 1: Add a raw-source-registering helper to conftest.py**

Add to `backend/tests/conftest.py`, after `build_annotated_root`:

```python
def register_raw_source(root: Path, scan_name: str, points, source_id: str = "test-raw") -> Path:
    """Register a REAL raw LAS file as `scan_name`'s raw source, via the
    derivation-lineage fallback path (scene_registry.py's `source_laz_path`
    resolution) rather than `meta.json::source_laz` — this deliberately
    avoids touching `source_mesh`/`source_laz` on meta.json, which would flip
    `is_z_up_from_meta` and rotate coordinates unexpectedly. The written LAS
    points are in the SAME (already Y-up, unrotated) frame as the scan's own
    points, since is_z_up stays False for these synthetic fixtures — see
    scan_meta.py::is_z_up_from_meta.

    Returns the path to the written LAS file."""
    import laspy

    raw_rel = f"raw/{scan_name}.laz"
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "raw" / "sources.json").write_text(json.dumps({
        "sources": [{"source_id": source_id, "path": raw_rel, "format": "laz"}],
    }))
    raw_path = root / raw_rel
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    n = len(points)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.0001, 0.0001, 0.0001])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    pts = np.asarray(points, dtype=np.float64)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    rng = np.random.default_rng(1)
    las.red = rng.integers(0, 65535, n).astype(np.uint16)
    las.green = rng.integers(0, 65535, n).astype(np.uint16)
    las.blue = rng.integers(0, 65535, n).astype(np.uint16)
    las.write(str(raw_path))

    meta_path = root / "annotated" / scan_name / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["derivation"] = {"root": {"source_id": source_id}}
    meta_path.write_text(json.dumps(meta))
    return raw_path
```

- [ ] **Step 2: Update `client_with_dense_annotated_scene` to also register a matching raw source**

Modify the fixture in `conftest.py`:

```python
@pytest.fixture
def client_with_dense_annotated_scene(monkeypatch, tmp_path):
    """Loaded annotated scene whose 400 points sit on a 5 mm grid — dense
    enough to pass the eval-grade gate (p90 = 5 mm <= 10 mm) — with a
    matching raw source registered so the gate (which now measures raw
    density, not the session cloud) actually passes."""
    import main
    from fastapi.testclient import TestClient

    pts = dense_grid_pts()
    root, _sid = build_annotated_root(tmp_path, pts=pts)
    register_raw_source(root, "demo", pts)
    monkeypatch.setattr("app.constants.LIDAR_ROOT", root, raising=False)
    client = TestClient(main.app)
    r = client.post("/api/load", json={"name": "annotated/demo", "max_points": 100})
    assert r.status_code == 200
    return client
```

This is a **transparent** change to every existing consumer of `client_with_dense_annotated_scene` — no test-file edits needed for any of them. Run `grep -n "client_with_dense_annotated_scene" backend/tests/test_regions_endpoints.py` before this step to get the exact current list (it includes at least `test_gate_passes_on_dense_scene_and_locks`, `test_patch_cannot_unlock_and_redraw_in_one_request`, `test_stats`, and several review-budget/category tests such as `test_gate_refuses_over_review_budget`, `test_gate_passes_at_the_budget_edge`, `test_stats_report_review_points` — don't assume the list above is exhaustive.

- [ ] **Step 3: Fix `test_gate_refuses_sparse_fixture`'s expected message**

`client_with_loaded_annotated_scene` (used by this test) has no raw source registered at all, and its scan has only 8 points. Under the new priority (raw-source check happens before any point-count check, since without raw nothing can be measured), the refusal reason is now "no raw source", not "100". Update the test in `backend/tests/test_regions_endpoints.py`:

```python
def test_gate_refuses_without_raw_source(client_with_loaded_annotated_scene):
    # The default fixture has no raw source registered.
    client = client_with_loaded_annotated_scene
    client.post("/api/regions", json={"prism": {"polygon": [[-10, -10], [10, -10], [10, 10], [-10, 10]],
                                                "y0": -10.0, "height": 20.0}})
    r = client.patch("/api/regions/1", json={"status": "eval_grade"})
    assert r.status_code == 422
    assert "raw source" in r.json()["detail"]
```

(Renamed from `test_gate_refuses_sparse_fixture` — the fixture is still sparse, but that's no longer the operative reason for refusal, so the old name would be misleading.)

Add a companion endpoint test that a raw-registered-but-sparse region still hits the point floor (covers the same case as Task 3's `test_gate_point_floor_uses_raw_count_not_session_count`, but through the HTTP layer):

```python
def test_gate_refuses_below_point_floor_with_raw_registered(monkeypatch, tmp_path):
    import main
    from fastapi.testclient import TestClient
    from tests.conftest import build_annotated_root, register_raw_source

    few_pts = [[i * 0.005, 0.0, 0.0] for i in range(25)]
    root, _sid = build_annotated_root(tmp_path, pts=np.asarray(few_pts, dtype=np.float32),
                                      n_instance0_points=25)
    register_raw_source(root, "demo", few_pts)
    monkeypatch.setattr("app.constants.LIDAR_ROOT", root, raising=False)
    client = TestClient(main.app)
    r = client.post("/api/load", json={"name": "annotated/demo", "max_points": 100})
    assert r.status_code == 200

    client.post("/api/regions", json={"prism": {"polygon": [[-1, -1], [1, -1], [1, 1], [-1, 1]],
                                                "y0": -1.0, "height": 2.0}})
    r = client.patch("/api/regions/1", json={"status": "eval_grade"})
    assert r.status_code == 422
    assert "100" in r.json()["detail"]
```

- [ ] **Step 4: Fix `test_frame_roundtrip_with_recenter`**

This test builds its own scene directly (not via `client_with_dense_annotated_scene`) and needs a raw source registered with matching offset-shifted points. In `backend/tests/test_regions_endpoints.py`, update:

```python
def test_frame_roundtrip_with_recenter(monkeypatch, tmp_path):
    """A scene whose coords exceed 1e3 triggers _recenter; the stored file
    must hold stored-frame geometry while the API speaks the runtime frame."""
    import main
    from fastapi.testclient import TestClient
    from tests.conftest import build_annotated_root, dense_grid_pts, register_raw_source

    off = (5000.0, 0.0, 3000.0)
    pts = dense_grid_pts(offset=off)
    root, _sid = build_annotated_root(tmp_path, pts=pts)
    register_raw_source(root, "demo", pts)
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
    # The gate works across the frame shift too (grid is 5 mm), now proving
    # the RAW-source measurement also survives the recenter round-trip.
    assert client.patch("/api/regions/1", json={"status": "eval_grade"}).status_code == 200
```

Note the raw LAS is written with points in the **pre-recenter runtime frame** (`dense_grid_pts(offset=off)`, matching what `build_annotated_root` uses for the scan.ply), and `register_raw_source` deliberately does not itself apply any recenter — `load_laz_region`'s own `offset` parameter (threaded from `_state["recenter_offset"]` through `_ctx()`) handles that at query time, exactly as it does for the scan.ply-derived session cloud. This is the explicit frame-consistency test the spec's Error Handling section calls for.

- [ ] **Step 5: Run the full endpoint test file**

Run: `cd backend && .venv/bin/pytest tests/test_regions_endpoints.py -v`

Expected: PASS (all tests)

- [ ] **Step 6: Run the full backend test suite to check for any other regressions**

Run: `cd backend && .venv/bin/pytest -v` (or `npm run test:backend` from repo root)

Expected: PASS. Pay particular attention to any other test file that calls `flip_status` or hits `/api/regions` — grep first: `grep -rln "flip_status\|/api/regions" backend/tests/*.py`

- [ ] **Step 7: Commit**

```bash
git add backend/tests/conftest.py backend/tests/test_regions_endpoints.py
git commit -m "test: register a raw source in eval-grade endpoint fixtures"
```

---

## Task 6: `GET /api/labels/accuracy` measures raw density when available

**Files:**
- Modify: `backend/routes/export.py`
- Test: `backend/tests/test_export_wizard.py` (or wherever existing `/api/labels/accuracy` tests live — run `grep -rln "labels/accuracy" backend/tests/*.py` first to confirm the file)

- [ ] **Step 1: Locate existing accuracy-endpoint tests**

Run: `grep -rn "labels/accuracy" backend/tests/*.py`

Read the matching test file's existing fixtures/imports before writing new tests, so the new tests match its established patterns (likely reuses `_build_materialize_ctx`-adjacent fixtures or a `client_with_*` fixture from conftest).

- [ ] **Step 2: Write the failing tests**

Add two tests to that file (adapt fixture setup to match the file's existing style):

```python
def test_labels_accuracy_is_raw_true_when_raw_source_registered(monkeypatch, tmp_path):
    # Reuse register_raw_source from conftest with a scan whose session cloud
    # is sparse but whose raw source is dense — proves the endpoint reads
    # raw density now, not ctx.scan_pos.
    ...  # build scene via build_annotated_root + register_raw_source,
         # load it, call GET /api/labels/accuracy, assert is_raw is True and
         # p90 reflects the DENSE raw points, not the sparser session cloud.


def test_labels_accuracy_is_raw_false_without_raw_source(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.get("/api/labels/accuracy", params={"scene": "annotated/demo", "session_id": ...})
    assert r.status_code == 200
    assert r.json()["is_raw"] is False
```

(Fill in the exact session_id / scene param plumbing to match this test file's existing accuracy-endpoint tests — copy their setup verbatim and only change the assertions.)

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd backend && .venv/bin/pytest <test_file> -k "labels_accuracy" -v`

Expected: FAIL — `KeyError: 'is_raw'` (endpoint doesn't return that field yet)

- [ ] **Step 4: Implement**

In `backend/routes/export.py`, update the import block and `labels_accuracy`:

```python
from labeling.materialize import (
    MaterializeCtx,
    collect_volumes,
    build_replay_index,
    materialize_downsample,
    materialize_raw,
    raw_sample_spacing,
    raw_reservoir_sample_spacing,
    loa_band,
)
```

```python
@router.get("/api/labels/accuracy")
def labels_accuracy(scene: str, session_id: str) -> dict:
    """p50/p90 nearest-neighbor sample spacing — the labeling-density boundary
    uncertainty shown in the export wizard. Raw-backed (`is_raw: true`) when
    the scan has a registered raw source; otherwise a best-effort fallback
    against the loaded session cloud (`is_raw: false`) — this endpoint is
    informational, not a hard gate, so it degrades gracefully rather than
    refusing (contrast with the eval-grade region gate, which refuses)."""
    if _state.get("scene") != scene:
        raise HTTPException(409, f"scene mismatch — server has '{_state.get('scene')}', request was '{scene}'")
    if _state.get("session_id") != session_id:
        raise HTTPException(409, f"session mismatch — server has '{_state.get('session_id')}', request was '{session_id}'")
    pc = _state.get("pc")
    if pc is None:
        raise HTTPException(409, "no scene loaded")
    src = _state.get("source")
    raw_path = src.extras.get("source_laz_path") if src is not None else None
    if raw_path:
        offset = np.asarray(_state.get("recenter_offset") or [0.0, 0.0, 0.0], dtype=np.float64)
        p50, p90 = raw_reservoir_sample_spacing(raw_path, _scene_is_z_up(src), offset)
        is_raw = True
    else:
        p50, p90 = raw_sample_spacing(pc.points)
        is_raw = False
    return {"p50": p50, "p90": p90, "loa": loa_band(p90), "is_raw": is_raw}
```

Check how `_state["source"]` is actually populated (grep `_state\[.source.\]\s*=` in `app/core.py`/`routes/load.py`) before assuming that key name — use whatever the existing code already stores there (`_build_materialize_ctx` in this same file resolves `src` some way; match that, rather than introducing a second lookup pattern).

- [ ] **Step 5: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest <test_file> -k "labels_accuracy" -v`

Expected: PASS

- [ ] **Step 6: Run the full backend suite**

Run: `cd backend && .venv/bin/pytest -v`

Expected: PASS

- [ ] **Step 7: Commit**

```bash
git add backend/routes/export.py backend/tests/<test_file>
git commit -m "feat: GET /api/labels/accuracy measures raw-source density when available"
```

---

## Task 7: `export_labels` manifest reports raw-backed accuracy for raw exports

**Files:**
- Modify: `backend/routes/export.py`
- Test: same file located in Task 6, Step 1 (likely also holds `export_labels` tests) — confirm with `grep -rln "labels/export\b" backend/tests/*.py`

- [ ] **Step 1: Write the failing test**

```python
def test_export_manifest_raw_resolution_uses_raw_density(...):
    # Build a scene with a sparse session cloud + dense registered raw
    # source. Export at resolution.kind == "raw". Assert the manifest's
    # accuracy.p90 reflects the DENSE raw spacing, not the sparser session
    # cloud's spacing.
    ...


def test_export_manifest_scan_resolution_still_uses_session_cloud(...):
    # Same fixture, but resolution.kind == "scan". Manifest accuracy must
    # match today's raw_sample_spacing(ctx.scan_pos) behavior unchanged.
    ...
```

(Adapt to this test file's existing `export_labels` test setup/fixtures — there should already be at least one passing raw-resolution export test to copy the request-building boilerplate from.)

- [ ] **Step 2: Run to verify it fails**

Run: `cd backend && .venv/bin/pytest <test_file> -k "export_manifest" -v`

Expected: FAIL (manifest accuracy currently always reflects `ctx.scan_pos`, so a raw-dense/session-sparse fixture's `p90` won't match the raw expectation)

- [ ] **Step 3: Implement**

In `backend/routes/export.py::export_labels`, change:

```python
    p50, p90 = raw_sample_spacing(ctx.scan_pos)
```

to:

```python
    if req.resolution.kind == "raw" and ctx.raw_path is not None:
        p50, p90 = raw_reservoir_sample_spacing(ctx.raw_path, ctx.scene_is_z_up, ctx.offset)
    else:
        p50, p90 = raw_sample_spacing(ctx.scan_pos)
```

Keep this above the `if req.resolution.kind in ("scan", "subsample")` / `elif == "raw"` export-writing branch, same as today (it doesn't need to move — it only needs its *source* to depend on which branch is about to run).

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && .venv/bin/pytest <test_file> -k "export_manifest" -v`

Expected: PASS

- [ ] **Step 5: Run the full backend suite**

Run: `cd backend && .venv/bin/pytest -v`

Expected: PASS, full green

- [ ] **Step 6: Commit**

```bash
git add backend/routes/export.py backend/tests/<test_file>
git commit -m "feat: raw-resolution exports stamp their manifest with raw-backed accuracy"
```

---

## Task 8: Docs + final full-suite verification

**Files:**
- Modify: `/home/hendrik/coding/engine/tools/labeling/voxa/CLAUDE.md`

- [ ] **Step 1: Update the Region eval-grade description in CLAUDE.md**

Find the paragraph beginning `**Region** — the 7th rail tool...` in `CLAUDE.md`. Locate the sentence: `the **eval-grade gate** — flipping a region to eval_grade measures p50/p90 nearest-neighbour spacing over the region's FULL-RES points (labeling.materialize.raw_sample_spacing) ...`. Update it to reflect that "full-res" now means the scan's registered **raw source** (not the session's working-array density), and that the gate refuses outright when no raw source is registered:

```
the **eval-grade gate** — flipping a region to eval_grade measures p50/p90 nearest-neighbour
spacing over the region's points in the scan's registered RAW source
(labeling.materialize.raw_region_sample_spacing, not the session's subsampled working cloud —
see docs/superpowers/specs/2026-07-23-region-density-raw-source-design.md) and refuses
(RegionError) if no raw source is registered, below a 100-point floor, above a 10 mm p90 bar,
or over the 3% `excluded_review` budget (phase 2, `REVIEW_BUDGET_FRAC`), recording
`accuracy:{p50,p90,loa,measured_at}` on success —
```

Keep the rest of that sentence (geometry lock, etc.) unchanged.

- [ ] **Step 2: Run the full test suite (frontend + backend) one more time**

Run: `npm test` from the repo root.

Expected: PASS, full green — this is the final verification gate before considering the work done.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: note region eval-grade gate now measures raw-source density"
```

---

## Notes for the implementer

- Task 3's `flip_status` change has one design nuance flagged inline (Step 3): the point-floor check must read the RAW region's point count, not `len(positions)` — don't skip the `raw_region_point_count` helper.
- Tasks 6 and 7 have deliberately loose test-file/fixture references ("locate the existing test file", "adapt to existing style") because the exact test file and fixture names for the export/accuracy endpoints weren't pinned down during planning — the first step of each task is to find and read them before writing new tests, per the repo's existing conventions.
- Run `grep -rln "flip_status\|/api/regions\|labels/accuracy\|labels/export" backend/tests/*.py` once at the start of implementation to get the full list of files this change touches, beyond what's named above — the plan's file list is not guaranteed exhaustive for a codebase this size.
- Non-goals from the spec still apply: do not attempt bim's E57 ingestion, do not add a spatial index to `load_laz_region`, do not touch preseg/SAM accuracy, do not resurrect phase 0b, and leave the frontend `is_raw` copy/treatment for a follow-up (the field just needs to exist on the wire).
