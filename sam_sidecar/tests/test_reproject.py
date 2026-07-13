import numpy as np
from reproject import look_at_view, project_points, depth_buffer_mask

def test_center_point_projects_to_image_center():
    pos = np.array([0.0, -10.0, 0.0]); target = np.array([0.0, 0.0, 0.0])
    view = look_at_view(pos, target, up=(0.0, 0.0, 1.0))   # Z-up world
    u, v, z, infront = project_points(np.array([[0.0, 0.0, 0.0]]), view, 60.0, 100, 100)
    assert infront[0]
    assert abs(u[0] - 50) < 1e-6 and abs(v[0] - 50) < 1e-6
    assert abs(z[0] - 10.0) < 1e-6

def test_occluded_point_rejected():
    pos = np.array([0.0, -10.0, 0.0]); target = np.array([0.0, 0.0, 0.0])
    view = look_at_view(pos, target, up=(0.0, 0.0, 1.0))
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]])   # near + far-behind-it
    u, v, z, infront = project_points(pts, view, 60.0, 100, 100)
    idx, uu, vv = depth_buffer_mask(u, v, z, infront, 100, 100)
    assert 0 in idx and 1 not in idx
