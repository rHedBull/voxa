import numpy as np

from scenes.reproject import (
    ORIENTATION_PRESETS,
    depth_buffer_mask,
    euler_xyz_matrix,
    look_at_view,
    project_points,
)


def test_point_ahead_projects_to_center():
    pos = np.array([0.0, 0, 0])
    tgt = np.array([0.0, 0, -1])  # look down -Z
    view = look_at_view(pos, tgt)
    pt = np.array([[0.0, 0, -5]])  # straight ahead
    u, v, z, infront = project_points(pt, view, 60.0, 100, 100)
    assert infront[0] and abs(u[0] - 50) < 1 and abs(v[0] - 50) < 1 and z[0] > 0


def test_point_behind_not_in_front():
    view = look_at_view(np.array([0.0, 0, 0]), np.array([0.0, 0, -1]))
    _, _, _, infront = project_points(np.array([[0.0, 0, 5]]), view, 60.0, 100, 100)
    assert not infront[0]


def test_occlusion_near_hides_far():
    view = look_at_view(np.array([0.0, 0, 0]), np.array([0.0, 0, -1]))
    pts = np.array([[0.0, 0, -2], [0.0, 0, -8]])  # same pixel, near + far
    u, v, z, inf = project_points(pts, view, 60.0, 100, 100)
    idx, _, _ = depth_buffer_mask(u, v, z, inf, 100, 100)
    assert 0 in idx and 1 not in idx  # only the near point is visible


def test_zplus_preset_maps_zup_to_yup():
    # a Z-up point with height in z should land in +y after the Z+ rotation
    R = euler_xyz_matrix(*ORIENTATION_PRESETS["Z+"])
    p = np.array([0.0, 0.0, 3.0])  # 3m "up" in Z-up
    rotated = p @ R.T
    assert rotated[1] > 2.9 and abs(rotated[2]) < 1e-6
