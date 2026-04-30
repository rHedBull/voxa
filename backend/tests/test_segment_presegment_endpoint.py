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
pytest.importorskip("sklearn")


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
    from segment_state import SegmentSession

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


def test_presegment_bootstraps_from_cloud_without_session():
    """No segment session but a cloud is loaded → bootstraps a fresh
    SegmentSession from the cloud's points and runs presegmentation."""
    from fastapi.testclient import TestClient
    import main
    from point_cloud import PointCloud

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
