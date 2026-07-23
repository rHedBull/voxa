"""Pure-function tests for the eval-region store, gate, and stats
(backend/labeling/regions.py). Endpoint tests live in
test_regions_endpoints.py."""
from __future__ import annotations


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


# ---- gate -----------------------------------------------------------------

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


# ---- frame shift ------------------------------------------------------------

def test_shift_prism_roundtrip():
    off = [574184.0, 49.0, 6220868.0]        # (dx, dy, dz)
    stored = reg.shift_prism(PRISM, off)
    assert stored["polygon"][1] == [1.0 + 574184.0, 0.0 + 6220868.0]
    assert stored["y0"] == pytest.approx(-0.5 + 49.0)
    assert stored["height"] == PRISM["height"]
    back = reg.shift_prism(stored, [-v for v in off])
    assert np.allclose(back["polygon"], PRISM["polygon"])
    assert back["y0"] == pytest.approx(PRISM["y0"])


# ---- stats ------------------------------------------------------------------

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
