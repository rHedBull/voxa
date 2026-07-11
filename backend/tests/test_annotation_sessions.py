"""Session-scoped instance docs: /api/annotations/{kind}/{scene}?session_id=…

For annotated scans the cuboid/pointset instance doc must live inside the
session dir (sessions/<id>/instances_gt.json), not in the scene-global
ANNOT_DIR — otherwise every session of a scan shows the same right-panel
instances (the cross-session leak this file regression-tests).
"""
from __future__ import annotations

import json


def _doc(scene, n=1):
    return {
        "scene": scene, "kind": "gt", "meta": {},
        "instances": [
            {"id": f"i{k}", "cls": "pipe", "kind": "pointset", "segId": 100 + k,
             "confirmed": True}
            for k in range(n)
        ],
    }


def test_put_writes_into_session_dir(client_with_annotated_scene, tmp_path):
    client, scene_id, session_id = client_with_annotated_scene
    r = client.put(f"/api/annotations/gt/{scene_id}?session_id={session_id}",
                   json=_doc(scene_id, 2))
    assert r.status_code == 200
    p = (tmp_path / "lidar" / "annotated" / "demo" / "sessions" / session_id
         / "instances_gt.json")
    assert p.exists(), f"expected session-scoped doc at {p}"
    assert len(json.loads(p.read_text())["instances"]) == 2


def test_sessions_do_not_leak_instances(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    # Save 2 instances into the fixture session…
    client.put(f"/api/annotations/gt/{scene_id}?session_id={session_id}",
               json=_doc(scene_id, 2))
    # …create a second session…
    r = client.post("/api/load", json={"name": scene_id, "max_points": 100})
    assert r.status_code == 200
    r = client.post(f"/api/scenes/annotated/demo/sessions", json={"name": "other"})
    assert r.status_code == 200
    other = r.json()["session_id"]
    # …which must see an EMPTY instance list, not the first session's doc.
    r = client.get(f"/api/annotations/gt/{scene_id}?session_id={other}")
    assert r.status_code == 200
    assert r.json()["instances"] == []
    # And the original session still round-trips its own doc.
    r = client.get(f"/api/annotations/gt/{scene_id}?session_id={session_id}")
    assert [i["segId"] for i in r.json()["instances"]] == [100, 101]


def test_unknown_session_404(client_with_annotated_scene):
    client, scene_id, _sid = client_with_annotated_scene
    r = client.get(f"/api/annotations/gt/{scene_id}?session_id=nope-123")
    assert r.status_code == 404
    r = client.put(f"/api/annotations/gt/{scene_id}?session_id=nope-123",
                   json=_doc(scene_id))
    assert r.status_code == 404


def test_bad_session_id_rejected(client_with_annotated_scene):
    client, scene_id, _sid = client_with_annotated_scene
    # Path-traversal shapes must be rejected, not joined into a path.
    r = client.get(f"/api/annotations/gt/{scene_id}?session_id=../escape")
    assert r.status_code in (400, 404, 422)


def test_no_session_id_keeps_legacy_global_path(client, tmp_path):
    # data-dir (legacy tier) scenes keep the scene-global ANNOT_DIR doc.
    r = client.put("/api/annotations/gt/somescene", json=_doc("somescene"))
    assert r.status_code == 200
    r = client.get("/api/annotations/gt/somescene")
    assert r.status_code == 200
    assert len(r.json()["instances"]) == 1


def test_box_obb_and_seq_round_trip(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    doc = {
        "scene": scene_id, "kind": "gt", "meta": {},
        "instances": [
            {"id": "b1", "cls": "pipe", "kind": "pointset", "segId": 7,
             "source": "box", "confirmed": True,
             "center": [1.0, 2.0, 3.0], "size": [0.5, 0.5, 2.0],
             "rotation": [0.0, 0.1, 0.0], "seq": 5},
        ],
    }
    r = client.put(f"/api/annotations/gt/{scene_id}?session_id={session_id}", json=doc)
    assert r.status_code == 200
    got = client.get(f"/api/annotations/gt/{scene_id}?session_id={session_id}").json()
    inst = got["instances"][0]
    assert inst["center"] == [1.0, 2.0, 3.0]
    assert inst["size"] == [0.5, 0.5, 2.0]
    assert inst["rotation"] == [0.0, 0.1, 0.0]
    assert inst["seq"] == 5


def _instances(*specs):
    # spec = (id, seq_or_None)
    out = []
    for k, (iid, seq) in enumerate(specs):
        c = {"id": iid, "cls": "pipe", "kind": "pointset", "segId": 100 + k}
        if seq is not None:
            c["seq"] = seq
        out.append(c)
    return out


def test_seq_backfilled_by_list_order(client_with_annotated_scene):
    client, scene_id, sid = client_with_annotated_scene
    doc = {"scene": scene_id, "kind": "gt", "meta": {},
           "instances": _instances(("a", None), ("b", None), ("c", None))}
    client.put(f"/api/annotations/gt/{scene_id}?session_id={sid}", json=doc)
    got = client.get(f"/api/annotations/gt/{scene_id}?session_id={sid}").json()
    assert [i["seq"] for i in got["instances"]] == [0, 1, 2]


def test_seq_preserves_existing_fills_after_max(client_with_annotated_scene):
    client, scene_id, sid = client_with_annotated_scene
    doc = {"scene": scene_id, "kind": "gt", "meta": {},
           "instances": _instances(("a", 5), ("b", None), ("c", 2))}
    client.put(f"/api/annotations/gt/{scene_id}?session_id={sid}", json=doc)
    got = client.get(f"/api/annotations/gt/{scene_id}?session_id={sid}").json()
    seqs = {i["id"]: i["seq"] for i in got["instances"]}
    assert seqs["a"] == 5 and seqs["c"] == 2       # preserved
    assert seqs["b"] == 6                            # filled after max(5)


def test_seq_stamping_idempotent(client_with_annotated_scene):
    client, scene_id, sid = client_with_annotated_scene
    doc = {"scene": scene_id, "kind": "gt", "meta": {},
           "instances": _instances(("a", None), ("b", None))}
    client.put(f"/api/annotations/gt/{scene_id}?session_id={sid}", json=doc)
    first = client.get(f"/api/annotations/gt/{scene_id}?session_id={sid}").json()["instances"]
    # Re-save exactly what we got back; seqs must not drift.
    client.put(f"/api/annotations/gt/{scene_id}?session_id={sid}",
               json={"scene": scene_id, "kind": "gt", "meta": {}, "instances": first})
    second = client.get(f"/api/annotations/gt/{scene_id}?session_id={sid}").json()["instances"]
    assert [i["seq"] for i in first] == [i["seq"] for i in second] == [0, 1]
