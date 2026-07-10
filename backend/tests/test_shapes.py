import numpy as np
from labeling.shapes import obb_indices


def test_obb_axis_aligned_selects_interior():
    # 3x3x3 grid of points at integer coords 0..2 on each axis (27 pts).
    xs = np.array([0, 1, 2], dtype=np.float32)
    pts = np.array([[x, y, z] for x in xs for y in xs for z in xs],
                   dtype=np.float32).reshape(-1)
    # Box centered at (1,1,1), size 1.0 -> half-extent 0.5 -> only the center point.
    box = {"center": [1.0, 1.0, 1.0], "size": [1.0, 1.0, 1.0],
           "rotation": [0.0, 0.0, 0.0]}
    idx = obb_indices(pts, box)
    assert idx.tolist() == [13]  # the (1,1,1) point is index 13 in the grid


def test_obb_rotated_matches_local_frame():
    # A point offset +0.9 along world X. A box rotated 45 deg about Z with
    # half-extent 0.5 in local x should NOT contain it (0.9 > 0.5 in any frame),
    # but a box half-extent 0.5 centered ON it should.
    pts = np.array([0.9, 0.0, 0.0], dtype=np.float32)
    inside = {"center": [0.9, 0.0, 0.0], "size": [1.0, 1.0, 1.0],
              "rotation": [0.0, 0.0, np.pi / 4]}
    outside = {"center": [0.0, 0.0, 0.0], "size": [1.0, 1.0, 1.0],
               "rotation": [0.0, 0.0, np.pi / 4]}
    assert obb_indices(pts, inside).tolist() == [0]
    assert obb_indices(pts, outside).tolist() == []
