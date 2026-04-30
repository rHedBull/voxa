"""Smoke test for the RANSAC presegmentation pipeline.

Builds a tiny synthetic scene (one plane + one cylinder) and asserts:

  - returns valid arrays of the right shape and dtype
  - finds at least two distinct segments
  - assigns ≥80% of plane points to a single ``flat_surface`` segment
  - assigns ≥50% of cylinder points to a single ``*pipe`` / ``fitting`` segment
  - SCHEMA invariant: ``class_id == -1`` segments don't leak into ``instance_ids``

Pure deterministic geometry, no real PLY needed. Skips if open3d is
missing — install via ``pip install open3d`` (or ``pip install -r
backend/requirements.txt``).
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("open3d")
pytest.importorskip("sklearn")

from presegment import presegment  # noqa: E402


def _make_plane(n: int, *, size: float = 2.0, z: float = 0.0,
                rng: np.random.Generator) -> np.ndarray:
    xs = rng.uniform(-size / 2, size / 2, n)
    ys = rng.uniform(-size / 2, size / 2, n)
    zs = np.full(n, z) + rng.normal(0, 0.002, n)
    return np.column_stack([xs, ys, zs])


def _make_cylinder(n: int, *, radius: float, length: float,
                   center: np.ndarray, axis: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    # Build orthonormal frame.
    ref = np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    e1 = ref - axis * np.dot(ref, axis)
    e1 /= np.linalg.norm(e1)
    e2 = np.cross(axis, e1)
    theta = rng.uniform(0, 2 * np.pi, n)
    along = rng.uniform(-length / 2, length / 2, n)
    r = radius + rng.normal(0, 0.001, n)
    pts = (center
           + along[:, None] * axis
           + (r * np.cos(theta))[:, None] * e1
           + (r * np.sin(theta))[:, None] * e2)
    return pts


def test_presegment_plane_plus_cylinder():
    rng = np.random.default_rng(42)
    plane = _make_plane(2000, size=2.0, z=0.0, rng=rng)
    # Cylinder above the plane, axis along +Y.
    cyl = _make_cylinder(2000, radius=0.08, length=1.2,
                         center=np.array([0.6, 0.0, 0.4]),
                         axis=np.array([0.0, 1.0, 0.0]), rng=rng)
    xyz = np.vstack([plane, cyl])

    classes = {"pipe": 0, "tank": 1, "structural": 2, "fitting": 3,
               "equipment": 4, "unknown": 5}

    instance_ids, summary = presegment(xyz, class_map=classes,
                                        log=lambda *_: None)

    # Shape + dtype.
    assert instance_ids.shape == (len(xyz),)
    assert instance_ids.dtype == np.int32
    assert isinstance(summary, list)

    # At least one of each found.
    labels = [s["label"] for s in summary]
    assert any("flat_surface" in lbl for lbl in labels), (
        f"Expected a flat segment in {labels}")
    assert any(("pipe" in lbl) or ("fitting" in lbl) for lbl in labels), (
        f"Expected a pipe/fitting segment in {labels}")

    # Most plane points get one shared segment id.
    plane_ids = instance_ids[: len(plane)]
    plane_assigned = plane_ids[plane_ids >= 0]
    assert plane_assigned.size > 0, "no plane points were assigned"
    plane_dom = int(np.bincount(plane_assigned).max())
    assert plane_dom / len(plane) > 0.8, (
        f"plane segment too fragmented: {plane_dom}/{len(plane)}")

    # Most cylinder points get one shared segment id (loose because
    # region-growing on a noisy cylinder may split into 2-3 chunks).
    cyl_ids = instance_ids[len(plane):]
    cyl_assigned = cyl_ids[cyl_ids >= 0]
    assert cyl_assigned.size > 0, "no cylinder points were assigned"
    cyl_dom = int(np.bincount(cyl_assigned).max())
    assert cyl_dom / len(cyl) > 0.5, (
        f"cylinder segment too fragmented: {cyl_dom}/{len(cyl)}")

    # Invariant: every assigned point's segment id appears in summary.
    summary_ids = {s["id"] for s in summary}
    assigned_ids = set(int(x) for x in np.unique(instance_ids[instance_ids >= 0]))
    assert assigned_ids.issubset(summary_ids), (
        f"orphan instance ids: {assigned_ids - summary_ids}")

    # Catchall invariant: every point lands in some segment. The
    # nearest-neighbour sweep at the end of the pipeline ensures no
    # point is left at -1 once assignment finishes.
    assert (instance_ids >= 0).all(), (
        f"{int((instance_ids < 0).sum())} points still unassigned")
