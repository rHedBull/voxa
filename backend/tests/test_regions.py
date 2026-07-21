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
