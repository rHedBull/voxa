"""Tests for per-point comparison: metrics module + /api/compare-points route."""
from __future__ import annotations

import numpy as np
import pytest

from labeling.compare_points import compare_class_arrays


def test_agreement_excludes_both_unlabeled():
    a = np.array([-1, -1, 0, 0, 1], dtype=np.int8)
    b = np.array([-1,  0, 0, 1, 1], dtype=np.int8)
    m = compare_class_arrays(a, b)
    # 4 points labeled in at least one side; matches among them: idx2 (0==0), idx4 (1==1)
    assert m["n_points"] == 5
    assert m["n_labeled_a"] == 3
    assert m["n_labeled_b"] == 4
    assert m["agreement"] == pytest.approx(2 / 4)
    assert m["agreement_all"] == pytest.approx(3 / 5)  # idx0 (-1==-1) counts here


def test_per_class_iou_precision_recall():
    a = np.array([0, 0, 0, 1, -1, -1], dtype=np.int8)
    b = np.array([0, 0, 1, 1,  1, -1], dtype=np.int8)
    m = compare_class_arrays(a, b)
    per = {c["class_id"]: c for c in m["per_class"]}
    # class 0: tp=2, union=3 → iou 2/3; precision=2/2 (B claims 2, both match);
    # recall=2/3 (A has 3)
    assert per[0]["iou"] == pytest.approx(2 / 3)
    assert per[0]["precision"] == pytest.approx(1.0)
    assert per[0]["recall"] == pytest.approx(2 / 3)
    assert per[0]["n_a"] == 3 and per[0]["n_b"] == 2
    # class 1: in_a={idx3}, in_b={idx2,idx3,idx4} → tp=1, union=3 → iou 1/3;
    # precision=1/3; recall=1/1
    assert per[1]["iou"] == pytest.approx(1 / 3)
    assert per[1]["precision"] == pytest.approx(1 / 3)
    assert per[1]["recall"] == pytest.approx(1.0)


def test_per_class_zero_division_is_null():
    a = np.array([0, 0], dtype=np.int8)
    b = np.array([-1, -1], dtype=np.int8)
    per = {c["class_id"]: c for c in compare_class_arrays(a, b)["per_class"]}
    assert per[0]["precision"] is None   # B claims nothing for class 0
    assert per[0]["recall"] == pytest.approx(0.0)
    assert per[0]["iou"] == pytest.approx(0.0)


def test_confusion_pairs_sorted_and_truncated():
    # 3x (0→1), 2x (1→2), 1x (2→0); unlabeled-on-either-side never appears
    a = np.array([0, 0, 0, 1, 1, 2, -1, 0], dtype=np.int8)
    b = np.array([1, 1, 1, 2, 2, 0, 1, -1], dtype=np.int8)
    m = compare_class_arrays(a, b)
    assert m["confusion"][0] == {"a_class": 0, "b_class": 1, "n": 3}
    assert m["confusion"][1] == {"a_class": 1, "b_class": 2, "n": 2}
    assert m["confusion"][2] == {"a_class": 2, "b_class": 0, "n": 1}
    assert len(m["confusion"]) == 3


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        compare_class_arrays(np.zeros(3, dtype=np.int8), np.zeros(4, dtype=np.int8))


# ---- route tests ----

def _save_fixture_session(client):
    """Give the fixture session a saved output via the real save route."""
    r = client.put("/api/segment/save")
    assert r.status_code == 200


def test_compare_session_vs_preseg(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    _save_fixture_session(client)
    # discover the session id from the sessions endpoint
    sid = client.get("/api/scenes/annotated/demo/sessions").json()["sessions"][0]["session_id"]
    r = client.post("/api/compare-points/annotated/demo", json={
        "a": {"kind": "session", "id": sid},
        "b": {"kind": "preseg", "id": "ransac"},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metrics"]["n_points"] == 8
    assert 0.0 <= body["metrics"]["agreement_all"] <= 1.0
    import base64, numpy as np
    a = np.frombuffer(base64.b64decode(body["a_class_ids"]), dtype=np.int8)
    assert a.shape == (8,)
    assert isinstance(body["palette"], list)


def test_compare_session_vs_session(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    _save_fixture_session(client)
    sid = client.get("/api/scenes/annotated/demo/sessions").json()["sessions"][0]["session_id"]
    import time; time.sleep(1.1)  # session ids are second-resolution
    r = client.post("/api/scenes/annotated/demo/sessions",
                    json={"name": "b side", "preseg_id": "ransac"})
    assert r.status_code == 200
    sid_b = r.json()["session_id"]
    # activate + save the new session so it has output
    assert client.post("/api/load", json={"name": "annotated/demo",
                                          "session_id": sid_b}).status_code == 200
    assert client.put("/api/segment/save").status_code == 200
    r = client.post("/api/compare-points/annotated/demo", json={
        "a": {"kind": "session", "id": sid},
        "b": {"kind": "session", "id": sid_b},
    })
    assert r.status_code == 200, r.text
    assert r.json()["metrics"]["agreement"] is not None


def test_compare_no_output_409(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    sid = client.get("/api/scenes/annotated/demo/sessions").json()["sessions"][0]["session_id"]
    r = client.post("/api/compare-points/annotated/demo", json={
        "a": {"kind": "session", "id": sid},
        "b": {"kind": "preseg", "id": "ransac"},
    })
    assert r.status_code == 409
    assert "no saved output" in r.json()["detail"]


def test_compare_unknown_ids_404(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    _save_fixture_session(client)
    sid = client.get("/api/scenes/annotated/demo/sessions").json()["sessions"][0]["session_id"]
    for body in ({"a": {"kind": "session", "id": "nope"}, "b": {"kind": "preseg", "id": "ransac"}},
                 {"a": {"kind": "session", "id": sid}, "b": {"kind": "preseg", "id": "nope"}}):
        assert client.post("/api/compare-points/annotated/demo", json=body).status_code == 404


def test_compare_length_mismatch_409(client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    _save_fixture_session(client)
    sid = client.get("/api/scenes/annotated/demo/sessions").json()["sessions"][0]["session_id"]
    # truncate the saved output to force a length mismatch vs the preseg
    import numpy as np
    p = scan_dir_for_loaded_scene / "sessions" / sid / "output" / "gt_class_ids.npy"
    np.save(p, np.load(p)[:5])
    r = client.post("/api/compare-points/annotated/demo", json={
        "a": {"kind": "session", "id": sid},
        "b": {"kind": "preseg", "id": "ransac"},
    })
    assert r.status_code == 409
    assert "different clouds" in r.json()["detail"]


def test_compare_non_annotated_tier_409(client_with_annotated_scene):
    """compare-points only exists for annotated scans; other tiers 409."""
    client, _scene_id, _sid = client_with_annotated_scene
    # The legacy tier is served from VOXA_DATA_DIR/scenes — create a stub.
    import os
    from pathlib import Path
    legacy = Path(os.environ["VOXA_DATA_DIR"]) / "scenes" / "stub409"
    legacy.mkdir(parents=True, exist_ok=True)
    (legacy / "source.ply").write_bytes(b"")  # discovery only checks existence
    try:
        r = client.post("/api/compare-points/legacy/stub409", json={
            "a": {"kind": "session", "id": "x"},
            "b": {"kind": "preseg", "id": "y"},
        })
        assert r.status_code == 409
        assert "annotated" in r.json()["detail"]
    finally:
        (legacy / "source.ply").unlink()
        legacy.rmdir()


def test_cuboid_compare_endpoint_is_gone(client):
    assert client.post("/api/compare/legacy/foo").status_code in (404, 405)
