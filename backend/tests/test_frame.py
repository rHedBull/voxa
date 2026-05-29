import numpy as np
import pytest

from scenes.frame import Frame, apply_transform, compose_a_to_v, frame_from_dict, is_rigid


def _rot_z(deg, t=(0, 0, 0)):
    a = np.deg2rad(deg)
    c, s = np.cos(a), np.sin(a)
    M = np.eye(4)
    M[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    M[:3, 3] = t
    return M


def test_compose_maps_A_into_V():  # Appendix A worked example
    A = Frame(_rot_z(90, (5, 2, 0)), "scan#local")
    V = Frame(np.eye(4), "scan#local")
    T = compose_a_to_v(A, V)
    p = apply_transform(T, np.array([[1.0, 0, 0]]))[0]
    assert np.allclose(p, [5, 3, 0], atol=1e-9)  # (1,0,0) rot90 -> (0,1,0), +t(5,2,0)


def test_compose_nonidentity_target_inverts():
    A = Frame(_rot_z(30, (1, 0, 0)), "s#local")
    V = Frame(_rot_z(30, (1, 0, 0)), "s#local")
    T = compose_a_to_v(A, V)
    assert np.allclose(T, np.eye(4), atol=1e-9)  # same frame -> identity


def test_is_rigid_rejects_nonorthonormal():
    bad = np.eye(4)
    bad[0, 0] = 2.0
    assert is_rigid(_rot_z(45, (3, 1, 2))) and not is_rigid(bad)


def test_roundtrip_dict():
    f = Frame(_rot_z(15, (2, 3, 4)), "s#local",
              georef={"crs": "EPSG:32632", "offset_m": [1, 2, 3]})
    g = frame_from_dict(f.to_dict())
    assert np.allclose(f.transform_to_canonical, g.transform_to_canonical)
    assert g.canonical_id == "s#local" and g.georef["crs"] == "EPSG:32632"


def test_frame_from_dict_validates_shape():
    with pytest.raises(ValueError):
        frame_from_dict({"transform_to_canonical": [[1, 0], [0, 1]], "canonical_id": "x"})
