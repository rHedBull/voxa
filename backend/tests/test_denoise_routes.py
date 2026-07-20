# backend/tests/test_denoise_routes.py
import numpy as np
from fastapi.testclient import TestClient
from labeling.segment_state import SegmentSession


def _session(n=100):
    pos = np.random.default_rng(0).normal(size=(n, 3)).astype(np.float32)
    cls = np.full(n, -1, dtype=np.int8)
    inst = np.full(n, -1, dtype=np.int32)
    return SegmentSession(cls, inst, pos)


def test_remove_sam_points_shrinks_candidate():
    seg = _session()
    out = seg.materialize_sam_segment(np.arange(0, 40, dtype=np.int32), source="sam")
    sid = out["sam_seg_id"]
    assert seg.sam_segments[sid]["n_points"] == 40
    seg.remove_sam_points(np.arange(0, 10, dtype=np.int32))
    assert seg.sam_segments[sid]["n_points"] == 30
    assert int((seg.sam_ids == sid).sum()) == 30
    # removed points are back to no candidacy
    assert bool((seg.sam_ids[np.arange(0, 10)] == -1).all())


def _client_with_cloud():
    """Load a session with a planted-speck cloud into _state and return a
    TestClient. Mirrors how other route tests seed _state."""
    import main
    core = np.random.default_rng(0).normal(0, 0.1, size=(400, 3)).astype(np.float32)
    specks = np.array([[9, 0, 0], [0, 9, 0], [0, 0, 9]], dtype=np.float32)
    pos = np.vstack([core, specks])
    seg = SegmentSession(np.full(pos.shape[0], -1, np.int8),
                         np.full(pos.shape[0], -1, np.int32), pos)
    main._state["seg"] = seg
    return TestClient(main.app), seg


def test_denoise_materializes_exclude_instance():
    client, seg = _client_with_cloud()
    r = client.post("/api/segment/denoise", json={"std_ratio": 2.0})
    assert r.status_code == 200
    body = r.json()
    assert body["n_affected"] == 3            # the three specks
    inst = body["instance_id"]
    # the three speck points now carry the Exclude class (id 6)
    speck_idx = [400, 401, 402]
    assert bool((seg.class_ids[speck_idx] == 6).all())
    assert bool((seg.instance_ids[speck_idx] == inst).all())


def test_denoise_replace_inst_erases_prior():
    client, seg = _client_with_cloud()
    first = client.post("/api/segment/denoise", json={"std_ratio": 2.0}).json()
    inst1 = first["instance_id"]
    # Re-run replacing inst1: a stricter ratio flags fewer/equal points, and
    # the prior instance's points must be erased (no orphan Exclude labels
    # under a dead instance id).
    second = client.post("/api/segment/denoise",
                         json={"std_ratio": 2.0, "replace_inst": inst1}).json()
    inst2 = second["instance_id"]
    assert inst2 != inst1
    # No point still carries the dead inst1 id.
    assert int((seg.instance_ids == inst1).sum()) == 0


def test_denoise_empty_returns_null_instance():
    client, seg = _client_with_cloud()
    # Absurdly high ratio flags nothing.
    body = client.post("/api/segment/denoise", json={"std_ratio": 50.0}).json()
    assert body["instance_id"] is None
    assert body["n_affected"] == 0
