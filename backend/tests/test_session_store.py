import json
import time

import numpy as np
import pytest

from labeling import session_store as ss
from preseg.preseg_store import register_preseg
from scenes.scan_layout import ScanLayout


@pytest.fixture
def scan(tmp_path):
    d = tmp_path / "annotated" / "demo"
    d.mkdir(parents=True)
    lay = ScanLayout(d)
    register_preseg(lay, "ransac",
                    np.array([0, 0, 1, 1, -1, -1, 2, 2], dtype=np.int32),
                    summary={"segments": [{"id": i, "class_id": -1} for i in range(3)]},
                    generator="ransac", params={})
    return lay


SRC_FP = "sha256:dummysource"


def test_create_seeded_session(scan):
    info = ss.create_session(scan, name="first", preseg_id="ransac",
                             n_points=8, source_fp=SRC_FP)
    assert info.session_id.endswith("_ransac")
    sp = scan.session(info.session_id)
    aux = json.loads(sp.session_json.read_text())
    assert aux["preseg_id"] == "ransac"
    assert aux["preseg_fingerprint"].startswith("sha256:")
    assert aux["source_fingerprint"] == SRC_FP
    assert aux["name"] == "first"
    assert "is_from_prelabel" not in aux
    ii = np.load(sp.working_segment_ids)
    assert ii.tolist() == [0, 0, 1, 1, -1, -1, 2, 2]
    assert np.load(sp.working_class_ids).dtype == np.int8


def test_create_blank_session(scan):
    info = ss.create_session(scan, name="blank", preseg_id=None,
                             n_points=8, source_fp=SRC_FP)
    assert info.preseg_id is None
    sp = scan.session(info.session_id)
    assert (np.load(sp.working_segment_ids) == -1).all()


def test_create_missing_preseg_fails(scan):
    with pytest.raises(FileNotFoundError):
        ss.create_session(scan, name="x", preseg_id="nope",
                          n_points=8, source_fp=SRC_FP)


def test_create_applies_min_segment_points(scan):
    # segment 2 has 2 points; threshold 3 drops it to (-1,-1) at seed time
    info = ss.create_session(scan, name="x", preseg_id="ransac",
                             n_points=8, source_fp=SRC_FP,
                             min_segment_points=3)
    sp = scan.session(info.session_id)
    ii = np.load(sp.working_segment_ids)
    assert (ii == -1).all()  # all segments have 2 points -> all dropped


def test_list_rename_delete(scan):
    a = ss.create_session(scan, name="a", preseg_id=None, n_points=8, source_fp=SRC_FP)
    infos = ss.list_sessions(scan)
    assert [i.session_id for i in infos] == [a.session_id]
    ss.rename_session(scan, a.session_id, "renamed")
    assert ss.list_sessions(scan)[0].name == "renamed"
    ss.delete_session(scan, a.session_id)
    assert ss.list_sessions(scan) == []


def test_last_worked_ordering(scan):
    a = ss.create_session(scan, name="a", preseg_id=None, n_points=8, source_fp=SRC_FP)
    time.sleep(1.1)  # saved_at has seconds resolution
    b = ss.create_session(scan, name="b", preseg_id=None, n_points=8, source_fp=SRC_FP)
    assert ss.last_worked(ss.list_sessions(scan)) == b.session_id
    # an edit (autosave) to a moves it ahead — saved_at means last persisted edit
    time.sleep(1.1)
    from labeling.segment_io import load_session_aux, save_session_aux
    sp = scan.session(a.session_id)
    save_session_aux(sp.dir, load_session_aux(sp.dir))
    assert ss.last_worked(ss.list_sessions(scan)) == a.session_id


def test_verify_pins_ok_and_mismatches(scan):
    info = ss.create_session(scan, name="x", preseg_id="ransac",
                             n_points=8, source_fp=SRC_FP)
    ss.verify_pins(scan, info.session_id, source_fp=SRC_FP)  # no raise
    with pytest.raises(ss.PinMismatch) as e:
        ss.verify_pins(scan, info.session_id, source_fp="sha256:other")
    assert e.value.diverged == "source"
    # re-register the preseg with different content → preseg pin diverges
    # (pin check compares the session pin against the preseg's DECLARED
    # fingerprint in meta.json — re-registering is the supported way a
    # preseg changes; out-of-band npy edits are out of scope by design)
    register_preseg(scan, "ransac", np.array([9] * 8, dtype=np.int32),
                    summary={"segments": [{"id": 9, "class_id": -1}]},
                    generator="ransac", params={})
    with pytest.raises(ss.PinMismatch) as e:
        ss.verify_pins(scan, info.session_id, source_fp=SRC_FP)
    assert e.value.diverged == "preseg"


def test_corrupt_session_listed_not_hidden(scan):
    a = ss.create_session(scan, name="a", preseg_id=None, n_points=8, source_fp=SRC_FP)
    scan.session(a.session_id).session_json.write_text("{broken")
    infos = ss.list_sessions(scan)
    assert infos[0].corrupt is True


def test_rename_delete_reject_traversal_ids(scan):
    """sids arrive from URL segments; the routes 404 traversal, but the
    store guards too (defense in depth — delete is rmtree)."""
    for bad in ("../evil", "..", ".hidden", "a/b"):
        with pytest.raises(ValueError, match="session_id"):
            ss.delete_session(scan, bad)
        with pytest.raises(ValueError, match="session_id"):
            ss.rename_session(scan, bad, "x")


def test_verify_pins_missing_session_raises(scan):
    with pytest.raises(FileNotFoundError):
        ss.verify_pins(scan, "20990101-000000_blank", source_fp=SRC_FP)


def test_last_worked_empty_and_corrupt_only(scan):
    assert ss.last_worked([]) is None
    a = ss.create_session(scan, name="a", preseg_id=None, n_points=8, source_fp=SRC_FP)
    scan.session(a.session_id).session_json.write_text("{broken")
    assert ss.last_worked(ss.list_sessions(scan)) is None
