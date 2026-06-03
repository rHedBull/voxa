import numpy as np
import pytest

from preseg.preseg_store import list_presegs, load_preseg, register_preseg
from scenes.scan_layout import ScanLayout


@pytest.fixture
def scan(tmp_path):
    d = tmp_path / "annotated" / "demo"
    d.mkdir(parents=True)
    return ScanLayout(d)


def _inst(n=8):
    return np.array([0, 0, 1, 1, 2, 2, -1, -1][:n], dtype=np.int32)


def test_register_then_list_and_load(scan):
    info = register_preseg(scan, "ransac", _inst(),
                           summary={"segments": [{"id": 0, "class_id": -1},
                                                 {"id": 1, "class_id": -1},
                                                 {"id": 2, "class_id": -1}]},
                           generator="ransac", params={"eps": 0.1})
    assert info.preseg_id == "ransac"
    assert info.fingerprint.startswith("sha256:")
    assert info.n_segments == 3
    listed = list_presegs(scan)
    assert [p.preseg_id for p in listed] == ["ransac"]
    assert listed[0].n_segments == 3  # read from meta.json, not the array
    ci, ii = load_preseg(scan, "ransac", n_points=8)
    assert ii.tolist() == _inst().tolist()
    assert ci.dtype == np.int8  # class map applied; all -1 here


def test_register_rejects_bad_id(scan):
    with pytest.raises(ValueError, match="preseg_id"):
        register_preseg(scan, "Bad ID!", _inst(), summary={"segments": []},
                        generator="x", params={})


def test_load_shape_mismatch_raises(scan):
    register_preseg(scan, "ransac", _inst(), summary={"segments": []},
                    generator="x", params={})
    with pytest.raises(ValueError, match="shape"):
        load_preseg(scan, "ransac", n_points=99)


def test_fingerprint_stable_across_reload(scan):
    info = register_preseg(scan, "ransac", _inst(), summary={"segments": []},
                           generator="x", params={})
    assert list_presegs(scan)[0].fingerprint == info.fingerprint
