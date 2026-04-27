"""Auto-fit with a loaded point cloud — exercises the actual fitting math
(every Label-mode auto-fit cuboid goes through this)."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def loaded_cloud():
    """Stuff a synthetic PointCloud directly into main._state, bypassing the
    PLY/GLB loader. Yields the cluster bounds so tests can assert against them."""
    import main
    from point_cloud import PointCloud

    rng = np.random.default_rng(0)
    # Tight cluster of 200 points around (1, 1, 1) within ±0.05.
    pts = rng.uniform(low=0.95, high=1.05, size=(200, 3)).astype(np.float32)
    pc = PointCloud(points=pts)

    prev = dict(main._state)
    main._state.update(scene="synthetic", pc=pc, mesh=None, subsample_idx=None)
    try:
        yield {"lo": pts.min(axis=0), "hi": pts.max(axis=0)}
    finally:
        main._state.update(prev)


def test_auto_fit_shrinks_to_loaded_points(client, loaded_cloud):
    """A loose AABB should be shrunk to the actual point extent (±5mm pad)."""
    r = client.post(
        "/api/auto-fit",
        json={"bbox_min": [0, 0, 0], "bbox_max": [2, 2, 2], "cls": "pipe", "color": "#22c55e"},
    )
    assert r.status_code == 200
    body = r.json()

    lo, hi = loaded_cloud["lo"], loaded_cloud["hi"]
    expected_center = ((lo + hi) / 2).tolist()
    expected_size = (hi - lo + 0.005).tolist()

    for got, exp in zip(body["center"], expected_center):
        assert abs(got - exp) < 1e-5
    for got, exp in zip(body["size"], expected_size):
        assert abs(got - exp) < 1e-5


def test_auto_fit_falls_back_when_box_misses_cloud(client, loaded_cloud):
    """An AABB containing fewer than 50 points should NOT shrink — falls back
    to echoing the request box. Prevents fitting noise on near-empty regions."""
    r = client.post(
        "/api/auto-fit",
        json={"bbox_min": [10, 10, 10], "bbox_max": [11, 11, 11], "cls": "pipe", "color": "#22c55e"},
    )
    body = r.json()
    assert body["center"] == [10.5, 10.5, 10.5]
    assert body["size"] == [1.0, 1.0, 1.0]
