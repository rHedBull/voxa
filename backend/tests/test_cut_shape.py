"""Tests for the /api/segment/cut-shape endpoint.

Fixture geometry (see conftest.build_annotated_root): 8 points, preseg
instance array [-1, 0, 0, 1, 1, 2, -1, 3] -> cluster 0 = points {1,2}
(class 0/"pipe"), cluster 1 = points {3,4} (class 1/"tank"), cluster 2 =
points {5} (class 2/"equipment"), cluster 3 = points {7} (class 2). A fresh
session is seeded directly from the preseg, so instance_ids/class_ids start
equal to that partition (see session_store.create_session).
"""
from __future__ import annotations

import numpy as np


def _huge_obb(**over):
    """Box big enough to contain every point in the 8-point fixture cloud."""
    body = {"type": "obb", "center": [0.0, 0.0, 0.0],
            "size": [1e7, 1e7, 1e7], "rotation": [0.0, 0.0, 0.0]}
    body.update(over)
    return body


def _fixture_points() -> np.ndarray:
    """Reproduce the exact positions written by conftest.write_scene_ply
    (same seed) so tests can build a shape around a single known point."""
    rng = np.random.default_rng(0)
    return rng.standard_normal((8, 3)).astype(np.float32)


def test_cut_shape_partitions_two_presegments(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/cut-shape", json={
        "shape": _huge_obb(),
        "sources": [{"kind": "preseg", "seg_id": 0}, {"kind": "preseg", "seg_id": 1}],
        "protect_instances": [],
    })
    assert r.status_code == 200
    j = r.json()
    assert len(j["materialized"]) == 2
    assert all(m["source"] == "preseg" for m in j["materialized"])
    assert sorted(m["n_points"] for m in j["materialized"]) == [2, 2]
    assert j["instance"] is None


def test_cut_shape_instance_source_inherits_class(client_with_loaded_annotated_scene):
    import main
    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    src_inst_id = 0
    src_class = int(seg.class_ids[np.flatnonzero(seg.instance_ids == src_inst_id)[0]])

    r = client.post("/api/segment/cut-shape", json={
        "shape": _huge_obb(),
        "sources": [{"kind": "instance", "seg_id": src_inst_id}],
        "protect_instances": [],
    })
    assert r.status_code == 200
    j = r.json()
    assert j["instance"] is not None
    new_id = j["instance"]["instance_id"]
    assert new_id != src_inst_id

    new_mask = seg.instance_ids == new_id
    assert new_mask.any()
    assert (seg.class_ids[new_mask] == src_class).all()


def test_cut_shape_instance_source_protect_instances_blocks_self(
        client_with_loaded_annotated_scene):
    """protect_instances must actually be forwarded to apply_reassign on the
    instance-source path: protecting the very instance being cut from means
    every candidate point is dropped, so nothing gets materialized. This is
    the confirmed-source case in the real UI: the presegment you're cutting
    from was already promoted+confirmed as instance 0, and the cut must not
    silently steal its points."""
    import main
    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    src_inst_id = 0
    n_src_points = int((seg.instance_ids == src_inst_id).sum())
    assert n_src_points > 0

    r = client.post("/api/segment/cut-shape", json={
        "shape": _huge_obb(),
        "sources": [{"kind": "instance", "seg_id": src_inst_id}],
        "protect_instances": [src_inst_id],
    })
    assert r.status_code == 200
    j = r.json()
    # apply_reassign with every candidate protected returns n_affected=0 and
    # never allocates a fresh instance id (see SegmentSession.apply_reassign).
    assert j["instance"] is None
    assert j["materialized"] == []
    assert j["n_protected"] == n_src_points
    # instance_ids/class_ids for the source instance are untouched.
    assert int((seg.instance_ids == src_inst_id).sum()) == n_src_points


def test_cut_shape_preseg_source_protect_instances_drops_protected_points(
        client_with_loaded_annotated_scene):
    """protect_instances must also be forwarded on the preseg/sam
    materialize_sam_segment path: cutting from preseg seg_id 0 (points {1,2})
    while protecting instance 0 (which currently owns exactly those points,
    since the session is seeded 1:1 from the preseg) must drop every
    candidate point — nothing materializes into the SAM candidate layer."""
    import main
    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    src_seg_id = 0
    n_src_points = int((seg.preseg_ids == src_seg_id).sum())
    assert n_src_points > 0
    # Sanity: instance 0 currently owns the exact same points as preseg
    # segment 0 (fresh session seeded 1:1 from the preseg), so protecting
    # instance 0 genuinely overlaps this cut's partition.
    assert (seg.instance_ids[seg.preseg_ids == src_seg_id] == 0).all()

    r = client.post("/api/segment/cut-shape", json={
        "shape": _huge_obb(),
        "sources": [{"kind": "preseg", "seg_id": src_seg_id}],
        "protect_instances": [0],
    })
    assert r.status_code == 200
    j = r.json()
    assert j["materialized"] == []
    assert j["instance"] is None
    assert j["n_protected"] == n_src_points
    # No new SAM candidate was allocated over the protected points.
    assert not (seg.sam_ids[seg.preseg_ids == src_seg_id] >= 0).any()


def test_cut_shape_empty_source_partition_produces_no_entry(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    pts = _fixture_points()
    # A tight box around point 1 only (cluster 0 / preseg seg_id 0); no other
    # fixture point coincides with it, so cluster 1 (preseg seg_id 1) is empty.
    box = _huge_obb(center=pts[1].tolist(), size=[1e-3, 1e-3, 1e-3], rotation=[0.0, 0.0, 0.0])
    r = client.post("/api/segment/cut-shape", json={
        "shape": box,
        "sources": [{"kind": "preseg", "seg_id": 0}, {"kind": "preseg", "seg_id": 1}],
        "protect_instances": [],
    })
    assert r.status_code == 200
    j = r.json()
    assert len(j["materialized"]) == 1
    assert j["materialized"][0]["source"] == "preseg"
    assert j["materialized"][0]["n_points"] == 1


def test_cut_shape_unknown_source_kind_422(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/cut-shape", json={
        "shape": _huge_obb(),
        "sources": [{"kind": "bogus", "seg_id": 1}],
        "protect_instances": [],
    })
    assert r.status_code == 422
