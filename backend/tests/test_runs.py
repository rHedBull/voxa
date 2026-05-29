import numpy as np

from labeling.run_merge import merge_runs
from labeling.runs_io import (
    list_runs,
    read_run,
    read_runs_index,
    write_run,
    write_runs_index,
)


def test_run_roundtrip_and_index(tmp_path):
    coll = tmp_path / "labels"
    write_run(coll, "alice", {"gt_class_ids": np.array([0, 1, -1]),
                              "gt_segment_ids": np.array([0, 0, -1])},
              {"kind": "human", "annotator": "alice"})
    write_run(coll, "model_v2", {"gt_class_ids": np.array([0, 2, 2]),
                                 "gt_segment_ids": np.array([0, 1, 1])},
              {"kind": "model"})
    arrays, meta = read_run(coll, "alice")
    assert meta["kind"] == "human" and meta["run_id"] == "alice"
    assert np.array_equal(arrays["gt_class_ids"], [0, 1, -1])
    assert {r["run_id"] for r in list_runs(coll)} == {"alice", "model_v2"}
    write_runs_index(coll, default_run="alice")
    assert read_runs_index(coll)["default_run"] == "alice"


def test_merge_priority():
    # run0 labels pts 0,1 ; run1 labels pts 1(conflict),2
    c0 = np.array([5, 5, -1]); s0 = np.array([0, 0, -1])
    c1 = np.array([-1, 9, 7]); s1 = np.array([-1, 3, 3])
    mc, ms, conflicts, prov = merge_runs([c0, c1], [s0, s1], policy="priority")
    assert list(mc) == [5, 5, 7]            # run0 wins the pt-1 conflict (priority)
    assert list(ms) == [0, 0, 3]
    assert list(conflicts) == [False, True, False]
    assert list(prov) == [0, 0, 1]


def test_merge_vote_majority_and_tie():
    # pt0: 2 vs 1 -> class 5 wins; pt1: 1 vs 1 vs 1 distinct -> tie -> -1
    c0 = np.array([5, 5]); c1 = np.array([5, 6]); c2 = np.array([8, 7])
    s0 = np.array([0, 0]); s1 = np.array([0, 1]); s2 = np.array([2, 2])
    mc, ms, conflicts, prov = merge_runs([c0, c1, c2], [s0, s1, s2], policy="vote")
    assert mc[0] == 5                       # majority
    assert mc[1] == -1                      # 3-way tie -> unlabeled
    assert conflicts[0] and conflicts[1]
