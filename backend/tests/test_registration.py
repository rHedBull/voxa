import numpy as np

from preseg.registration import check_registration, registration_score


def _wall(n=40):
    g = np.linspace(-2, 2, n)
    xx, yy = np.meshgrid(g, g)
    return np.stack([xx.ravel(), yy.ravel(), -5 * np.ones(xx.size)], -1)  # wall at z=-5


def _frame(pos, tgt):
    return {"position": list(pos), "target": list(tgt)}


def test_coverage_high_when_looking_at_cloud():
    xyz = _wall()
    s = registration_score(xyz, [_frame([0, 0, 0], [0, 0, -1])], fov_y_deg=60, W=200, H=200)
    assert s["coverage"] > 0.5


def test_coverage_zero_when_looking_away():
    xyz = _wall()
    s = registration_score(xyz, [_frame([0, 0, 0], [0, 0, 1])], fov_y_deg=60, W=200, H=200)
    assert s["coverage"] < 0.01


def test_photometric_matches_solid_image():
    xyz = _wall()
    rgb = np.tile(np.array([200, 50, 50], np.uint8), (len(xyz), 1))
    red = np.zeros((200, 200, 3), np.uint8); red[:] = (200, 50, 50)
    blue = np.zeros((200, 200, 3), np.uint8); blue[:] = (50, 50, 200)
    f = [_frame([0, 0, 0], [0, 0, -1])]
    s_ok = registration_score(xyz, f, fov_y_deg=60, W=200, H=200, rgb=rgb, image_loader=lambda _f: red)
    s_bad = registration_score(xyz, f, fov_y_deg=60, W=200, H=200, rgb=rgb, image_loader=lambda _f: blue)
    assert s_ok["photometric"] > 0.9 and s_bad["photometric"] < 0.1


def test_check_fails_below_threshold():
    ok, reasons = check_registration({"coverage": 0.15, "photometric": 0.02},
                                     min_coverage=0.35, min_photometric=0.5)
    assert not ok and reasons


def test_check_passes_when_good():
    ok, reasons = check_registration({"coverage": 0.8, "photometric": 0.9})
    assert ok and not reasons
