# backend/tests/test_outliers.py
import numpy as np
from labeling.outliers import statistical_outlier_indices


def _cluster(n, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, 0.1, size=(n, 3)).astype(np.float32)


def test_flags_planted_specks_only():
    core = _cluster(500)
    specks = np.array([[10, 0, 0], [0, 12, 0], [-11, 0, 5]], dtype=np.float32)
    pts = np.vstack([core, specks])
    subset = np.arange(pts.shape[0])
    out = statistical_outlier_indices(pts, subset, k=16, std_ratio=2.0)
    # exactly the three planted specks (indices 500, 501, 502)
    assert set(out.tolist()) == {500, 501, 502}


def test_subset_scoping_returns_full_res_indices():
    # An outlier is judged only against the subset population, and returned
    # indices are into `positions`, not into the subset.
    core = _cluster(300)
    speck = np.array([[8, 8, 8]], dtype=np.float32)
    pts = np.vstack([core, speck])           # speck at index 300
    subset = np.array([250, 260, 270, 280, 290, 300], dtype=np.int64)
    out = statistical_outlier_indices(pts, subset, k=3, std_ratio=1.0)
    assert 300 in out.tolist()
    assert all(i in subset.tolist() for i in out.tolist())


def test_lower_std_ratio_flags_superset():
    core = _cluster(400)
    specks = (np.random.default_rng(1).normal(0, 3, size=(20, 3))
              + np.array([5, 5, 5])).astype(np.float32)
    pts = np.vstack([core, specks])
    subset = np.arange(pts.shape[0])
    greedy = set(statistical_outlier_indices(pts, subset, std_ratio=1.0).tolist())
    strict = set(statistical_outlier_indices(pts, subset, std_ratio=2.5).tolist())
    assert strict.issubset(greedy)


def test_degenerate_inputs_return_empty_not_crash():
    pts = _cluster(10)
    # empty subset
    assert statistical_outlier_indices(pts, np.array([], dtype=np.int64)).size == 0
    # subset smaller than k+1 -> not enough neighbors to judge -> empty
    assert statistical_outlier_indices(pts, np.array([0, 1, 2]), k=16).size == 0
    # all-identical points -> zero variance -> nothing flagged
    same = np.zeros((50, 3), dtype=np.float32)
    assert statistical_outlier_indices(same, np.arange(50)).size == 0
