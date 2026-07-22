"""TDD for build_instance_glbs (instance-mesh export)."""
import numpy as np
import trimesh

from labeling.instance_meshes import MIN_POINTS_FOR_MESH, build_instance_glbs


def _cube_points(n, center=(0.0, 0.0, 0.0), scale=1.0, seed=0):
    """n points scattered on/near a cube's surface — always ≥4 non-coplanar,
    so ConvexHull succeeds regardless of n."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-scale, scale, size=(n, 3)).astype(np.float32)
    return pts + np.asarray(center, dtype=np.float32)


def test_min_points_threshold_is_100():
    assert MIN_POINTS_FOR_MESH == 100


def test_happy_path_writes_glb_for_surviving_instance():
    pts_a = _cube_points(MIN_POINTS_FOR_MESH, center=(0, 0, 0), seed=1)
    pts_b = _cube_points(5, center=(10, 0, 0), seed=2)  # below threshold
    points = np.concatenate([pts_a, pts_b])
    instance_ids = np.concatenate([
        np.zeros(len(pts_a), dtype=np.int32),
        np.ones(len(pts_b), dtype=np.int32),
    ])

    glbs, skipped = build_instance_glbs(points, instance_ids, {0, 1})

    assert set(glbs.keys()) == {0}
    mesh = trimesh.load(trimesh.util.wrap_as_stream(glbs[0]), file_type="glb", force="mesh")
    assert len(mesh.vertices) >= 4
    assert skipped == [(1, f"only {len(pts_b)} points")]


def test_below_threshold_is_skipped():
    pts = _cube_points(MIN_POINTS_FOR_MESH - 1, seed=3)
    instance_ids = np.zeros(len(pts), dtype=np.int32)

    glbs, skipped = build_instance_glbs(pts, instance_ids, {0})

    assert glbs == {}
    assert skipped == [(0, f"only {MIN_POINTS_FOR_MESH - 1} points")]


def test_coplanar_points_skipped_with_qhullerror_reason():
    # All points on the z=0 plane, ≥ MIN_POINTS_FOR_MESH of them — enough
    # points, but degenerate (no volume), so ConvexHull raises QhullError.
    rng = np.random.default_rng(4)
    xy = rng.uniform(-1, 1, size=(MIN_POINTS_FOR_MESH, 2)).astype(np.float32)
    pts = np.concatenate([xy, np.zeros((MIN_POINTS_FOR_MESH, 1), dtype=np.float32)], axis=1)
    instance_ids = np.zeros(len(pts), dtype=np.int32)

    glbs, skipped = build_instance_glbs(pts, instance_ids, {0})

    assert glbs == {}
    assert len(skipped) == 1
    assert skipped[0][0] == 0
    assert "coplanar" in skipped[0][1] or "QhullError" in skipped[0][1]


def test_only_requested_ids_are_considered():
    pts = _cube_points(MIN_POINTS_FOR_MESH, seed=5)
    instance_ids = np.zeros(len(pts), dtype=np.int32)

    glbs, skipped = build_instance_glbs(pts, instance_ids, set())

    assert glbs == {} and skipped == []
