import numpy as np
from render import render_view
from reproject import look_at_view

def test_render_produces_image_and_depth():
    # a 2x2m colored wall at y=0, camera on -Y looking at it (Z-up)
    g = np.linspace(-1, 1, 200)
    xx, zz = np.meshgrid(g, g)
    xyz = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()]).astype(np.float32)
    rgb = np.tile(np.array([200, 50, 50], np.uint8), (xyz.shape[0], 1))
    view = look_at_view(np.array([0.0, -5.0, 0.0]), np.zeros(3), up=(0, 0, 1))
    img, depth = render_view(xyz, rgb, view, fov_y=60.0, W=128, H=128)
    assert img.shape == (128, 128, 3) and depth.shape == (128, 128)
    # center pixels are the wall color, corners are background (inf depth)
    assert tuple(img[64, 64]) == (200, 50, 50)
    assert np.isinf(depth[0, 0])
    assert abs(depth[64, 64] - 5.0) < 0.2
