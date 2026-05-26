"""Endpoint test for POST /api/segment/presegment.

Bypasses /api/load (which needs a real PLY on disk) by stuffing a
SegmentSession directly into ``main._state`` with a synthetic plane +
cylinder cloud. Exercises:

  - 200 + correct response shape
  - the new SegmentSession in _state reflects the result
  - 409 when no segment session is loaded
"""
from __future__ import annotations

import base64

import numpy as np
import pytest

pytest.importorskip("open3d")


def _make_scene(rng: np.random.Generator) -> np.ndarray:
    plane = np.column_stack([
        rng.uniform(-1, 1, 1500),
        rng.uniform(-1, 1, 1500),
        rng.normal(0, 0.002, 1500),
    ])
    theta = rng.uniform(0, 2 * np.pi, 1500)
    along = rng.uniform(-0.6, 0.6, 1500)
    r = 0.08 + rng.normal(0, 0.001, 1500)
    cyl = np.column_stack([
        0.6 + r * np.cos(theta),
        along,
        0.4 + r * np.sin(theta),
    ])
    return np.vstack([plane, cyl]).astype(np.float32)


@pytest.fixture
def client_with_synthetic_session():
    """TestClient + ``main`` with a SegmentSession loaded directly into _state."""
    from fastapi.testclient import TestClient
    import main
    from labeling.segment_state import SegmentSession

    rng = np.random.default_rng(11)
    xyz = _make_scene(rng)
    n = len(xyz)
    main._state["seg"] = SegmentSession(
        class_ids=np.full(n, -1, dtype=np.int8),
        instance_ids=np.full(n, -1, dtype=np.int32),
        positions=xyz,
        is_from_prelabel=False,
    )
    try:
        yield TestClient(main.app)
    finally:
        main._state["seg"] = None


def test_presegment_replaces_session_and_returns_arrays(client_with_synthetic_session):
    client = client_with_synthetic_session
    r = client.post("/api/segment/presegment")
    assert r.status_code == 200, r.text
    j = r.json()

    assert j["n_assigned"] > 0
    assert j["n_segments"] >= 2
    assert j["is_from_prelabel"] is True

    inst = np.frombuffer(base64.b64decode(j["full_instance_ids"]), dtype=np.int32)
    cls = np.frombuffer(base64.b64decode(j["full_class_ids"]), dtype=np.int8)
    assert inst.size == 3000
    assert cls.size == 3000

    # In-state reflects the response.
    import main
    seg = main._state["seg"]
    assert seg is not None
    assert seg.is_from_prelabel is True
    assert seg.dirty is True
    np.testing.assert_array_equal(seg.instance_ids, inst.astype(np.int32))
    np.testing.assert_array_equal(seg.class_ids, cls.astype(np.int8))


def test_presegment_409_when_no_session_and_no_cloud(client):
    """No segment session AND no loaded cloud → 409."""
    import main
    main._state["seg"] = None
    main._state["pc"] = None
    r = client.post("/api/segment/presegment")
    assert r.status_code == 409


def test_presegment_preserves_classified_points(client_with_synthetic_session):
    """preserve_labeled=True keeps already-classified (class_id >= 0)
    points untouched and only re-presegments the rest. New supervoxel
    ids must not collide with the kept instance ids."""
    import main
    client = client_with_synthetic_session
    seg = main._state["seg"]

    # Mark the first 500 points as already labeled (class 7, instance 42).
    seg.class_ids[:500] = 7
    seg.instance_ids[:500] = 42
    kept_inst_before = seg.instance_ids[:500].copy()
    kept_cls_before = seg.class_ids[:500].copy()

    r = client.post(
        "/api/segment/presegment",
        json={"resolution": 0.1, "preserve_labeled": True},
    )
    assert r.status_code == 200, r.text

    seg = main._state["seg"]
    np.testing.assert_array_equal(seg.instance_ids[:500], kept_inst_before)
    np.testing.assert_array_equal(seg.class_ids[:500], kept_cls_before)
    # New supervoxels must live above the kept instance id (42).
    new_ids = seg.instance_ids[500:]
    assigned = new_ids[new_ids >= 0]
    assert assigned.size > 0
    assert assigned.min() > 42


def test_presegment_full_replace_when_preserve_false(client_with_synthetic_session):
    """preserve_labeled=False replaces everything (legacy behavior)."""
    import main
    client = client_with_synthetic_session
    seg = main._state["seg"]
    seg.class_ids[:500] = 7
    seg.instance_ids[:500] = 42

    r = client.post(
        "/api/segment/presegment",
        json={"resolution": 0.1, "preserve_labeled": False},
    )
    assert r.status_code == 200, r.text

    seg = main._state["seg"]
    # Class ids reset (the synthetic scene has no name→id mapping that
    # would re-classify supervoxels), instance ids start from 0.
    assert (seg.class_ids == -1).all()
    assert seg.instance_ids.min() >= 0


def test_presegment_bootstraps_from_cloud_without_session():
    """No segment session but a cloud is loaded → bootstraps a fresh
    SegmentSession from the cloud's points and runs presegmentation."""
    from fastapi.testclient import TestClient
    import main
    from scenes.point_cloud import PointCloud

    rng = np.random.default_rng(13)
    xyz = _make_scene(rng)
    main._state["seg"] = None
    main._state["pc"] = PointCloud(points=xyz)
    try:
        client = TestClient(main.app)
        r = client.post("/api/segment/presegment")
        assert r.status_code == 200, r.text
        j = r.json()
        assert j["n_segments"] >= 2
        # Bootstrap path leaves a real SegmentSession in _state.
        seg = main._state["seg"]
        assert seg is not None
        assert seg.is_from_prelabel is True
    finally:
        main._state["seg"] = None
        main._state["pc"] = None


def test_preseg_freezes_layer_and_seeds_instances(client_with_synthetic_session):
    """After preseg, the immutable preseg layer must be stamped with a
    fingerprint so undo/redo + linkage works (Task 9)."""
    client = client_with_synthetic_session
    r = client.post("/api/segment/presegment", json={"resolution": 0.1})
    assert r.status_code == 200, r.text

    state = client.get("/api/segment/state").json()
    assert state["has_seg"] is True
    assert state["preseg_fingerprint"] is not None
    assert state["preseg_fingerprint"].startswith("sha256:")

    import main
    seg = main._state["seg"]
    # preseg_ids must be set, non-trivial.
    assert seg.preseg_ids.shape == seg.instance_ids.shape
    assert (seg.preseg_ids >= 0).any()


def test_preseg_with_preserve_labeled_keeps_existing_labels(client_with_synthetic_session):
    """preserve_labeled=True must leave class_ids intact on already-labeled
    points even when a re-preseg would have overwritten them. Spec §2."""
    import main
    client = client_with_synthetic_session

    # Brush a class onto a few points via /api/segment/apply.
    indices_b64 = base64.b64encode(np.array([0, 1, 2], dtype=np.int32).tobytes()).decode()
    r = client.post("/api/segment/apply", json={
        "op": "reassign",
        "indices": indices_b64,
        "payload": {"target_inst": -1, "target_class": 2},
    })
    assert r.status_code == 200, r.text

    seg_before = main._state["seg"]
    pre_labels = seg_before.class_ids.copy()
    labeled_idx = (pre_labels >= 0)
    assert labeled_idx.any()

    r = client.post("/api/segment/presegment",
                    json={"resolution": 0.1, "preserve_labeled": True})
    assert r.status_code == 200, r.text

    seg_after = main._state["seg"]
    np.testing.assert_array_equal(
        seg_after.class_ids[labeled_idx], pre_labels[labeled_idx],
    )
