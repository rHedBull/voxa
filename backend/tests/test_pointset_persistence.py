"""Pointset instances round-trip through annotations save/load.

Pointset instances have null center/size and carry kind="pointset" + segId.
Pre-fix, the strict Cuboid schema rejected them and the fields silently
dropped on reload — so confirmed pointsets effectively vanished.
"""
from __future__ import annotations


def test_pointset_round_trip_preserves_kind_and_segid(client):
    scene = "ps-roundtrip"
    pointset = {
        "id": "i1", "cls": "pipe", "label": "Pipe 1", "color": "#22c55e",
        "kind": "pointset", "segId": 42, "confirmed": True,
        "source": "preseg",
    }
    cuboid = {
        "id": "i2", "cls": "tank", "label": "Tank 1", "color": "#5b8def",
        "center": [1.0, 2.0, 3.0], "size": [0.5, 0.5, 0.5],
        "rotation": [0.0, 0.0, 0.0], "conf": 1.0, "source": "manual",
    }
    r = client.put(
        "/api/annotations/gt/" + scene,
        json={"scene": scene, "kind": "gt", "instances": [pointset, cuboid], "meta": {}},
    )
    assert r.status_code == 200, r.text

    out = client.get("/api/annotations/gt/" + scene).json()
    assert len(out["instances"]) == 2
    by_id = {i["id"]: i for i in out["instances"]}
    ps = by_id["i1"]
    assert ps["kind"] == "pointset"
    assert ps["segId"] == 42
    assert ps["confirmed"] is True
    assert ps["center"] is None
    assert ps["size"] is None

    cb = by_id["i2"]
    assert cb["kind"] == "cuboid"
    assert cb["center"] == [1.0, 2.0, 3.0]


