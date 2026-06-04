"""Tests for per-point comparison: metrics module + /api/compare-points route."""
from __future__ import annotations

import numpy as np
import pytest

from labeling.compare_points import compare_class_arrays


def test_agreement_excludes_both_unlabeled():
    a = np.array([-1, -1, 0, 0, 1], dtype=np.int8)
    b = np.array([-1,  0, 0, 1, 1], dtype=np.int8)
    m = compare_class_arrays(a, b)
    # 4 points labeled in at least one side; matches among them: idx2 (0==0), idx4 (1==1)
    assert m["n_points"] == 5
    assert m["n_labeled_a"] == 3
    assert m["n_labeled_b"] == 4
    assert m["agreement"] == pytest.approx(2 / 4)
    assert m["agreement_all"] == pytest.approx(3 / 5)  # idx0 (-1==-1) counts here


def test_per_class_iou_precision_recall():
    a = np.array([0, 0, 0, 1, -1, -1], dtype=np.int8)
    b = np.array([0, 0, 1, 1,  1, -1], dtype=np.int8)
    m = compare_class_arrays(a, b)
    per = {c["class_id"]: c for c in m["per_class"]}
    # class 0: tp=2, union=3 → iou 2/3; precision=2/2 (B claims 2, both match);
    # recall=2/3 (A has 3)
    assert per[0]["iou"] == pytest.approx(2 / 3)
    assert per[0]["precision"] == pytest.approx(1.0)
    assert per[0]["recall"] == pytest.approx(2 / 3)
    assert per[0]["n_a"] == 3 and per[0]["n_b"] == 2
    # class 1: in_a={idx3}, in_b={idx2,idx3,idx4} → tp=1, union=3 → iou 1/3;
    # precision=1/3; recall=1/1
    assert per[1]["iou"] == pytest.approx(1 / 3)
    assert per[1]["precision"] == pytest.approx(1 / 3)
    assert per[1]["recall"] == pytest.approx(1.0)


def test_per_class_zero_division_is_null():
    a = np.array([0, 0], dtype=np.int8)
    b = np.array([-1, -1], dtype=np.int8)
    per = {c["class_id"]: c for c in compare_class_arrays(a, b)["per_class"]}
    assert per[0]["precision"] is None   # B claims nothing for class 0
    assert per[0]["recall"] == pytest.approx(0.0)
    assert per[0]["iou"] == pytest.approx(0.0)


def test_confusion_pairs_sorted_and_truncated():
    # 3x (0→1), 2x (1→2), 1x (2→0); unlabeled-on-either-side never appears
    a = np.array([0, 0, 0, 1, 1, 2, -1, 0], dtype=np.int8)
    b = np.array([1, 1, 1, 2, 2, 0, 1, -1], dtype=np.int8)
    m = compare_class_arrays(a, b)
    assert m["confusion"][0] == {"a_class": 0, "b_class": 1, "n": 3}
    assert m["confusion"][1] == {"a_class": 1, "b_class": 2, "n": 2}
    assert m["confusion"][2] == {"a_class": 2, "b_class": 0, "n": 1}
    assert len(m["confusion"]) == 3


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        compare_class_arrays(np.zeros(3, dtype=np.int8), np.zeros(4, dtype=np.int8))
