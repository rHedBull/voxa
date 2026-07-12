import numpy as np
from backproject import select_in_mask
from render import render_view
from reproject import look_at_view

def test_selects_masked_visible_points():
    g = np.linspace(-1, 1, 120); xx, zz = np.meshgrid(g, g)
    wall = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()]).astype(np.float32)
    rgb = np.tile(np.array([180, 180, 180], np.uint8), (wall.shape[0], 1))
    view = look_at_view(np.array([0.0, -5.0, 0.0]), np.zeros(3), up=(0, 0, 1))
    _, depth = render_view(wall, rgb, view, 60.0, 128, 128)
    mask = np.zeros((128, 128), bool); mask[40:88, 40:88] = True   # center square
    scan = wall[::4]
    sel = select_in_mask(scan, view, 60.0, 128, 128, mask, depth)
    assert sel.size > 0
    from reproject import project_points
    u, v, z, infront = project_points(scan[sel], view, 60.0, 128, 128)
    assert np.all(mask[v.astype(int), u.astype(int)])

def test_occluded_scan_points_excluded():
    g = np.linspace(-1, 1, 120); xx, zz = np.meshgrid(g, g)
    near = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()]).astype(np.float32)
    far = near + np.array([0, 3.0, 0], np.float32)
    rgb = np.tile(np.array([180, 180, 180], np.uint8), (near.shape[0], 1))
    view = look_at_view(np.array([0.0, -5.0, 0.0]), np.zeros(3), up=(0, 0, 1))
    _, depth = render_view(near, rgb, view, 60.0, 128, 128)   # depth from NEAR wall
    scan = np.vstack([near[::4], far[::4]])
    mask = np.ones((128, 128), bool)
    sel = select_in_mask(scan, view, 60.0, 128, 128, mask, depth)
    n_near = near[::4].shape[0]
    assert np.all(sel < n_near)   # no far-wall points survive occlusion
