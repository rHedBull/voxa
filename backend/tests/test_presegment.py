"""Tests for the voxel supervoxel presegmentation pipeline.

Builds a tiny synthetic scene and asserts:
  - returns valid arrays of the right shape and dtype
  - produces multiple distinct supervoxels
  - every point is assigned (no -1 survivors)
  - all summary ids appear in instance_ids
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("open3d")

from preseg.presegment_voxel import presegment  # noqa: E402


def _make_cloud(n: int, *, rng: np.random.Generator) -> np.ndarray:
    """Random cloud spread over a 2×2×1 m box."""
    return rng.uniform([0, 0, 0], [2, 2, 1], (n, 3)).astype(np.float64)


def test_presegment_basic():
    rng = np.random.default_rng(42)
    xyz = _make_cloud(1000, rng=rng)

    instance_ids, summary = presegment(xyz, log=lambda *_: None, resolution=0.1)

    assert instance_ids.shape == (len(xyz),)
    assert instance_ids.dtype == np.int32
    assert isinstance(summary, list)
    assert len(summary) >= 2, "expected multiple supervoxels"


def test_presegment_all_points_assigned():
    rng = np.random.default_rng(7)
    xyz = _make_cloud(500, rng=rng)
    instance_ids, _ = presegment(xyz, log=lambda *_: None, resolution=0.1)
    assert (instance_ids >= 0).all(), (
        f"{int((instance_ids < 0).sum())} points unassigned")


def test_presegment_summary_ids_match():
    rng = np.random.default_rng(99)
    xyz = _make_cloud(800, rng=rng)
    instance_ids, summary = presegment(xyz, log=lambda *_: None, resolution=0.1)
    summary_ids = {s["id"] for s in summary}
    assigned_ids = set(int(x) for x in np.unique(instance_ids[instance_ids >= 0]))
    assert assigned_ids.issubset(summary_ids), (
        f"orphan instance ids: {assigned_ids - summary_ids}")


def test_presegment_finer_resolution_gives_more_segments():
    rng = np.random.default_rng(3)
    xyz = _make_cloud(2000, rng=rng)
    _, s_coarse = presegment(xyz, log=lambda *_: None, resolution=0.2)
    _, s_fine   = presegment(xyz, log=lambda *_: None, resolution=0.05)
    assert len(s_fine) > len(s_coarse), (
        f"expected fine ({len(s_fine)}) > coarse ({len(s_coarse)})")
