import threading

import numpy as np
import pytest
from preseg_optimize import SEARCH_SPACE, run_study, score_segmentation


def _plane_xyz(n=2000, rng=None):
    rng = rng or np.random.default_rng(0)
    xy = rng.uniform(-1.0, 1.0, size=(n, 2))
    z = rng.normal(0.0, 0.001, size=(n, 1))
    return np.hstack([xy, z]).astype(np.float64)


def _cylinder_xyz(n=2000, radius=0.3, height=1.5, rng=None):
    rng = rng or np.random.default_rng(1)
    theta = rng.uniform(0, 2 * np.pi, size=n)
    z = rng.uniform(0, height, size=n)
    r = radius + rng.normal(0.0, 0.001, size=n)
    x = 3.0 + r * np.cos(theta)
    y = r * np.sin(theta)
    return np.column_stack([x, y, z]).astype(np.float64)


def _two_primitive_cloud():
    plane = _plane_xyz()
    cyl = _cylinder_xyz()
    xyz = np.vstack([plane, cyl])
    ids = np.concatenate([np.zeros(len(plane), dtype=np.int32),
                          np.ones(len(cyl), dtype=np.int32)])
    return xyz, ids


def _balanced_pure_cloud(n_segs: int = 8, per: int = 200):
    """``n_segs`` primitive-pure segments, each with ``per`` points so no
    single segment dominates."""
    rng = np.random.default_rng(7)
    xs, ids = [], []
    for s in range(n_segs):
        if s % 2 == 0:
            xy = rng.uniform(-1, 1, (per, 2))
            z = rng.normal(0, 1e-3, (per, 1))
            seg = np.hstack([xy, z]) + np.array([5 * s, 0, 0])
        else:
            theta = rng.uniform(0, 2 * np.pi, per)
            seg = np.column_stack([
                5 * s + 0.3 * np.cos(theta),
                0.3 * np.sin(theta),
                rng.uniform(0, 1, per),
            ])
        xs.append(seg)
        ids.append(np.full(per, s, dtype=np.int32))
    return np.vstack(xs).astype(np.float64), np.concatenate(ids)


def test_score_low_for_pure_primitives():
    xyz, ids = _balanced_pure_cloud()
    score = score_segmentation(xyz, ids)
    # mean RMS ≤ 0.005, no oversize penalty (each seg = 12.5%), log bonus ≈ 0.01
    assert score < 0.01, f"pure balanced primitives should score low, got {score}"


def test_score_high_for_mixed_segment():
    xyz, _ = _two_primitive_cloud()
    ids = np.zeros(len(xyz), dtype=np.int32)
    pad_xyz = np.zeros((16, 3))
    pad_ids = np.repeat(np.arange(1, 5, dtype=np.int32), 4)
    xyz_full = np.vstack([xyz, pad_xyz])
    ids_full = np.concatenate([ids, pad_ids])
    score = score_segmentation(xyz_full, ids_full)
    assert score > 0.05, f"mixed segment should score high, got {score}"


def test_score_penalises_oversized_segment():
    """A pure-but-dominant segment should score worse than a balanced split."""
    balanced_xyz, balanced_ids = _balanced_pure_cloud(n_segs=8, per=200)
    balanced_score = score_segmentation(balanced_xyz, balanced_ids)
    # collapse first 4 segments into one giant segment (50% of points)
    dom_ids = balanced_ids.copy()
    dom_ids[dom_ids < 4] = 0
    dom_score = score_segmentation(balanced_xyz, dom_ids)
    assert dom_score > balanced_score + 0.05, \
        f"oversize penalty too weak: balanced={balanced_score}, dominant={dom_score}"


def test_score_penalises_too_few_segments():
    xyz, _ = _two_primitive_cloud()
    ids = np.zeros(len(xyz), dtype=np.int32)
    score = score_segmentation(xyz, ids)
    assert score >= 1e5, f"degenerate (1 segment) should hit penalty, got {score}"


def test_score_penalises_too_many_segments():
    rng = np.random.default_rng(2)
    xyz = rng.uniform(-1, 1, size=(20000, 3)).astype(np.float64)
    ids = np.arange(len(xyz), dtype=np.int32)
    score = score_segmentation(xyz, ids)
    assert score >= 1e5


def test_search_space_mirrors_ransac_defaults():
    from presegment_ransac import RANSAC_DEFAULTS
    assert set(SEARCH_SPACE.keys()) == set(RANSAC_DEFAULTS.keys())


def test_run_study_smoke():
    rng = np.random.default_rng(3)
    plane = rng.uniform(-1, 1, (1500, 2))
    plane = np.hstack([plane, rng.normal(0, 1e-3, (1500, 1))])
    cyl_theta = rng.uniform(0, 2 * np.pi, 1500)
    cyl = np.column_stack([
        3 + 0.3 * np.cos(cyl_theta),
        0.3 * np.sin(cyl_theta),
        rng.uniform(0, 1, 1500),
    ])
    xyz = np.vstack([plane, cyl]).astype(np.float64)

    progress = []
    cancel = threading.Event()

    def cb(info):
        progress.append(info["trial"])

    result = run_study(
        xyz,
        n_trials=3,
        cancel_event=cancel,
        progress_cb=cb,
        class_map={},
    )
    assert result["n_trials_run"] == 3
    assert set(result["best_params"].keys()) == set(SEARCH_SPACE.keys())
    assert isinstance(result["best_score"], float)
    assert progress, "progress_cb should have been called at least once"


def test_run_study_cancel():
    rng = np.random.default_rng(4)
    xyz = rng.uniform(-1, 1, (2000, 3)).astype(np.float64)
    cancel = threading.Event()
    cancel.set()

    def cb(_info):
        pass

    result = run_study(
        xyz,
        n_trials=20,
        cancel_event=cancel,
        progress_cb=cb,
        class_map={},
    )
    assert result["n_trials_run"] < 20
