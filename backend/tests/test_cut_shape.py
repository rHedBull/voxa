"""Tests for the /api/segment/cut-shape endpoint.

Fixture geometry (see conftest.build_annotated_root): 8 points, preseg
instance array [-1, 0, 0, 1, 1, 2, -1, 3] -> cluster 0 = points {1,2}
(class 0/"pipe"), cluster 1 = points {3,4} (class 1/"tank"), cluster 2 =
points {5} (class 2/"equipment"), cluster 3 = points {7} (class 2). A fresh
session is seeded directly from the preseg, so instance_ids/class_ids start
equal to that partition (see session_store.create_session).
"""
from __future__ import annotations

import base64

import numpy as np


def _decode_indices(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.int32)


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

    # scan_indices_b64 must decode to the exact per-source point partition,
    # not just the right count — cluster 0 = {1,2}, cluster 1 = {3,4}.
    decoded_sets = [frozenset(_decode_indices(m["scan_indices_b64"]).tolist())
                    for m in j["materialized"]]
    assert sorted(decoded_sets, key=sorted) == [frozenset({1, 2}), frozenset({3, 4})]


def test_cut_shape_instance_source_inherits_class(client_with_loaded_annotated_scene):
    # instance 1 (points {3,4}, class 1/"tank") rather than instance 0 —
    # instance 0 carries frozen legacy class 0/"pipe" and is rejected by the
    # frozen-class write-guard (see test_frozen_guard.py::test_cut_instance_frozen_422).
    import main
    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    src_inst_id = 1
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

    decoded = _decode_indices(j["instance"]["scan_indices_b64"])
    assert sorted(decoded.tolist()) == sorted(np.flatnonzero(new_mask).tolist())


def test_cut_shape_instance_source_protect_instances_blocks_self(
        client_with_loaded_annotated_scene):
    """protect_instances must actually be forwarded to apply_reassign on the
    instance-source path: protecting the very instance being cut from means
    every candidate point is dropped, so nothing gets materialized. This is
    the confirmed-source case in the real UI: the presegment you're cutting
    from was already promoted+confirmed as instance 1, and the cut must not
    silently steal its points. Uses instance 1 (class 1/"tank"), not
    instance 0 — instance 0 carries frozen legacy class 0/"pipe" and is
    rejected outright by the frozen-class write-guard."""
    import main
    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    src_inst_id = 1
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


def test_cut_shape_sam_source_partitions_and_remains_sam(client_with_loaded_annotated_scene):
    """The sam branch of _cut_shape_core's per-source dispatch has no direct
    coverage elsewhere: materialize a source:'sam' candidate over cluster 1
    (points {3,4}), then cut a sub-region containing only point 3 out of it.
    The partition must land back in the SAM layer tagged source:'sam' (not
    'preseg'), and only the cut point must move — point 4 keeps its original
    sam candidacy."""
    import main
    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    pts = _fixture_points()

    sam_out = seg.materialize_sam_segment(np.array([3, 4], dtype=np.int32), source="sam")
    sam_seg_id = sam_out["sam_seg_id"]
    assert sam_seg_id is not None
    assert seg.sam_segments[sam_seg_id]["source"] == "sam"

    box = _huge_obb(center=pts[3].tolist(), size=[1e-3, 1e-3, 1e-3], rotation=[0.0, 0.0, 0.0])
    r = client.post("/api/segment/cut-shape", json={
        "shape": box,
        "sources": [{"kind": "sam", "seg_id": sam_seg_id}],
        "protect_instances": [],
    })
    assert r.status_code == 200
    j = r.json()
    assert j["instance"] is None
    assert len(j["materialized"]) == 1
    m = j["materialized"][0]
    assert m["source"] == "sam"
    assert m["n_points"] == 1
    assert _decode_indices(m["scan_indices_b64"]).tolist() == [3]

    new_sam_seg_id = m["sam_seg_id"]
    assert new_sam_seg_id != sam_seg_id
    assert seg.sam_segments[new_sam_seg_id]["source"] == "sam"
    assert int(seg.sam_ids[3]) == new_sam_seg_id
    # Point 4 was not part of the cut box, so its original sam candidacy
    # survives unshrunk (partitioning must not disturb the un-cut remainder).
    assert int(seg.sam_ids[4]) == sam_seg_id


def test_cut_shape_preseg_source_uses_instance_ids_not_preseg_ids(
        client_with_loaded_annotated_scene):
    """A 'preseg'-kind cut source must partition against instance_ids, not
    the immutable preseg_ids array. PresegmentList shows every id in
    segState.summary (instance_ids), not only genuine preseg-sourced rows —
    e.g. an orphaned already-classified group with no matching instances_gt.json
    row still surfaces there and gets tagged kind:'preseg' on cut
    (segment-tools.jsx unconditionally tags its rows 'preseg'). The frontend's
    own buildCutCloud (cut-mode.jsx) already tests instanceFull for 'preseg'
    sources, not preseg_ids — the backend must agree, or a real preseg_id is
    the only case that ever cuts successfully."""
    import main
    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    # Simulate: this group has no real presegmentation membership (as in a
    # session with no registered preseg), even though instance_ids/class_ids
    # are real — the exact split this bug hides behind.
    seg.preseg_ids[:] = -1
    n_src_points = int((seg.instance_ids == 0).sum())
    assert n_src_points > 0

    r = client.post("/api/segment/cut-shape", json={
        "shape": _huge_obb(),
        "sources": [{"kind": "preseg", "seg_id": 0}],
        "protect_instances": [],
    })
    assert r.status_code == 200
    j = r.json()
    assert len(j["materialized"]) == 1
    m = j["materialized"][0]
    assert m["source"] == "preseg"
    assert m["n_points"] == n_src_points
    assert sorted(_decode_indices(m["scan_indices_b64"]).tolist()) == \
        sorted(np.flatnonzero(seg.instance_ids == 0).tolist())


def test_cut_shape_unknown_source_kind_422(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/cut-shape", json={
        "shape": _huge_obb(),
        "sources": [{"kind": "bogus", "seg_id": 1}],
        "protect_instances": [],
    })
    assert r.status_code == 422
