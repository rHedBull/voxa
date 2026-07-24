# backend/tests/test_denoise_routes.py
import numpy as np
from fastapi.testclient import TestClient
from labeling.categories import CATEGORY_ARTIFACT, CATEGORY_EXCLUDED_REVIEW
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


def test_denoise_materializes_artifact_blob_specks():
    client, seg = _client_with_cloud()
    r = client.post("/api/segment/denoise", json={"std_ratio": 2.0})
    assert r.status_code == 200
    body = r.json()
    assert body["n_affected"] == 3            # the three specks
    inst = body["instance_id"]
    speck_idx = [400, 401, 402]
    # Sensor noise is an artifact, not a review blob: the specks carry the
    # `artifact` category, never the banned archive class id 6.
    assert bool((seg.categories[speck_idx] == CATEGORY_ARTIFACT).all())
    assert bool((seg.class_ids[speck_idx] == -1).all())
    assert bool((seg.instance_ids[speck_idx] == inst).all())
    assert int((seg.class_ids == 6).sum()) == 0


def test_denoise_materializes_artifact_blob():
    from labeling.categories import CATEGORY_ARTIFACT
    client, seg = _client_with_cloud()
    r = client.post("/api/segment/denoise", json={"std_ratio": 1.0, "k": 8})
    assert r.status_code == 200
    body = r.json()
    inst = body["instance_id"]
    assert inst is not None
    flagged = np.flatnonzero(seg.instance_ids == inst)
    assert flagged.size == body["n_affected"] > 0
    assert bool((seg.categories[flagged] == CATEGORY_ARTIFACT).all())
    assert bool((seg.class_ids[flagged] == -1).all())


def test_denoise_replace_inst_erases_prior_artifact_blob():
    client, seg = _client_with_cloud()
    first = client.post("/api/segment/denoise", json={"std_ratio": 1.0, "k": 8}).json()
    inst1 = first["instance_id"]
    assert inst1 is not None
    second = client.post("/api/segment/denoise",
                         json={"std_ratio": 1.2, "k": 8, "replace_inst": inst1}).json()
    assert np.flatnonzero(seg.instance_ids == inst1).size == 0
    assert second["instance_id"] != inst1


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


def _client_with_selection(kind):
    """Seed a SAM candidate or an instance whose members = a tight core plus
    a couple of far strays, and return (client, seg, id, stray_full_indices)."""
    import main
    from app.core import _state
    core = np.random.default_rng(2).normal(0, 0.1, size=(120, 3)).astype(np.float32)
    strays = np.array([[7, 0, 0], [0, 7, 0]], dtype=np.float32)
    pos = np.vstack([core, strays])          # strays at 120, 121
    seg = SegmentSession(np.full(pos.shape[0], -1, np.int8),
                         np.full(pos.shape[0], -1, np.int32), pos)
    members = np.arange(0, 122, dtype=np.int32)   # whole cloud is the selection
    if kind == "sam":
        out = seg.materialize_sam_segment(members, source="sam")
        sid = out["sam_seg_id"]
    else:
        out = seg.apply_reassign(members, target_inst=-1, target_class=0)  # class pipe
        sid = out["new_instance_id"]
    _state["seg"] = seg
    return TestClient(main.app), seg, sid, [120, 121]


def test_denoise_selection_sam_retires_strays():
    client, seg, sid, strays = _client_with_selection("sam")
    r = client.post("/api/segment/denoise-selection",
                    json={"source": "sam", "id": sid, "std_ratio": 1.5})
    assert r.status_code == 200
    body = r.json()
    assert body["n_removed"] == 2
    assert bool((seg.sam_ids[strays] == -1).all())
    assert seg.sam_segments[sid]["n_points"] == 120


def test_denoise_selection_instance_erases_strays():
    client, seg, sid, strays = _client_with_selection("instance")
    r = client.post("/api/segment/denoise-selection",
                    json={"source": "instance", "id": sid, "std_ratio": 1.5})
    body = r.json()
    assert body["n_removed"] == 2
    # strays back to unlabeled; core still labelled with the instance
    assert bool((seg.instance_ids[strays] == -1).all())
    assert int((seg.instance_ids == sid).sum()) == 120
