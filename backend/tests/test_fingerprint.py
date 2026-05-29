import numpy as np
import pytest
from scenes.fingerprint import cloud_fingerprint


def _grid(n=1000, seed=0):
    return np.random.default_rng(seed).uniform(-10, 10, (n, 3))


def test_order_independent():
    xyz = _grid()
    perm = np.random.default_rng(1).permutation(len(xyz))
    assert cloud_fingerprint(xyz) == cloud_fingerprint(xyz[perm])


def test_deterministic():
    # same stored cloud -> same hash (the core guarantee; mm-quantize normalizes
    # most sub-mm noise, but exact mm-boundary coincidences are not guaranteed
    # stable, so we assert determinism on identical input rather than jitter).
    xyz = _grid()
    assert cloud_fingerprint(xyz) == cloud_fingerprint(xyz.copy())


def test_real_move_changes_hash():
    xyz = _grid()
    moved = xyz.copy()
    moved[0, 0] += 0.01  # 10mm
    assert cloud_fingerprint(xyz) != cloud_fingerprint(moved)


def test_recenter_changes_hash():
    # the navvis case: same points, translated frame -> different identity
    xyz = _grid()
    assert cloud_fingerprint(xyz) != cloud_fingerprint(xyz + np.array([20.0, 0, 0]))


def test_prefix_and_shape_guard():
    assert cloud_fingerprint(_grid()).startswith("sha256:")
    with pytest.raises(ValueError):
        cloud_fingerprint(np.zeros((10, 2)))
