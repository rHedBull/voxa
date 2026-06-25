import inspect

import numpy as np

from preseg.sam3_features import _cache_key
from scan_schema.fingerprint import cloud_fingerprint


def test_cache_key_keys_on_fingerprint_not_count():
    # the v1.3 fix: the key derives from cloud content, not point count
    params = list(inspect.signature(_cache_key).parameters)
    assert "source_fingerprint" in params
    assert "n_points" not in params


def test_same_count_different_coords_differ(tmp_path):
    rd = [tmp_path]
    (tmp_path / "manifest.json").write_text("{}")
    a = cloud_fingerprint(np.random.default_rng(0).uniform(0, 1, (1000, 3)))
    b = cloud_fingerprint(np.random.default_rng(0).uniform(0, 1, (1000, 3)) + 20.0)
    ka = _cache_key(rd, a, fpn_level=0, pca_dim=64, orientation="Z+", fov=60.0)
    kb = _cache_key(rd, b, fpn_level=0, pca_dim=64, orientation="Z+", fov=60.0)
    assert ka != kb  # same n_points, different content -> different key


def test_same_content_same_key(tmp_path):
    rd = [tmp_path]
    (tmp_path / "manifest.json").write_text("{}")
    fp = cloud_fingerprint(np.random.default_rng(0).uniform(0, 1, (1000, 3)))
    assert _cache_key(rd, fp, 0, 64, "Z+", 60.0) == _cache_key(rd, fp, 0, 64, "Z+", 60.0)
