# backend/tests/test_denoise_routes.py
import numpy as np
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
