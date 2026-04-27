"""Smoke tests for the Voxa backend API."""

from __future__ import annotations


def _cuboid(id_: str, cls: str = "pipe", center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0)):
    return {
        "id": id_,
        "cls": cls,
        "label": cls.title(),
        "color": "#22c55e",
        "center": list(center),
        "size": list(size),
        "rotation": [0.0, 0.0, 0.0],
        "conf": 1.0,
        "source": "manual",
    }


def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_config_classes_loaded(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    classes = r.json()["classes"]
    # Either the on-disk classes.yaml or the built-in defaults should populate this.
    assert isinstance(classes, list) and len(classes) > 0
    for c in classes:
        assert {"id", "label", "color", "hotkey"} <= set(c.keys())


def test_scenes_returns_list(client):
    r = client.get("/api/scenes")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_load_unknown_scene_404(client):
    r = client.post("/api/load", json={"name": "does-not-exist", "max_points": 1000})
    assert r.status_code == 404


def test_annotation_roundtrip(client):
    scene = "rt-scene"
    doc = {"scene": scene, "kind": "gt", "instances": [_cuboid("inst-1")], "meta": {}}

    put = client.put(f"/api/annotations/{scene}/gt", json=doc)
    assert put.status_code == 200
    assert put.json()["count"] == 1

    got = client.get(f"/api/annotations/{scene}/gt")
    assert got.status_code == 200
    body = got.json()
    assert body["scene"] == scene
    assert len(body["instances"]) == 1
    assert body["instances"][0]["id"] == "inst-1"


def test_get_annotation_missing_returns_empty(client):
    r = client.get("/api/annotations/never-saved/gt")
    assert r.status_code == 200
    assert r.json()["instances"] == []


def test_compare_perfect_match(client):
    scene = "cmp-perfect"
    inst = _cuboid("inst-gt-1")
    pred = _cuboid("inst-pr-1")  # same center/size/cls → IoU = 1.0

    client.put(
        f"/api/annotations/{scene}/gt",
        json={"scene": scene, "kind": "gt", "instances": [inst], "meta": {}},
    )
    client.put(
        f"/api/annotations/{scene}/pred",
        json={"scene": scene, "kind": "pred", "instances": [pred], "meta": {}},
    )

    r = client.post(f"/api/compare/{scene}", json={"scene": scene, "iou_threshold": 0.3})
    assert r.status_code == 200
    body = r.json()
    assert body["tp"] == 1
    assert body["fp"] == 0
    assert body["fn"] == 0
    assert body["precision"] == 1.0
    assert body["recall"] == 1.0


def test_compare_disjoint_is_fp_fn(client):
    scene = "cmp-disjoint"
    gt = _cuboid("inst-gt-1", center=(0, 0, 0))
    pr = _cuboid("inst-pr-1", center=(10, 10, 10))  # no overlap

    client.put(
        f"/api/annotations/{scene}/gt",
        json={"scene": scene, "kind": "gt", "instances": [gt], "meta": {}},
    )
    client.put(
        f"/api/annotations/{scene}/pred",
        json={"scene": scene, "kind": "pred", "instances": [pr], "meta": {}},
    )

    r = client.post(f"/api/compare/{scene}", json={"scene": scene, "iou_threshold": 0.3})
    body = r.json()
    assert body["tp"] == 0
    assert body["fp"] == 1
    assert body["fn"] == 1


def test_auto_fit_without_loaded_cloud(client):
    """With no PC loaded, auto-fit should echo the request box back as a cuboid."""
    r = client.post(
        "/api/auto-fit",
        json={"bbox_min": [0, 0, 0], "bbox_max": [1, 2, 3], "cls": "pipe", "color": "#22c55e"},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["cls"] == "pipe"
    assert body["center"] == [0.5, 1.0, 1.5]
    assert body["size"] == [1.0, 2.0, 3.0]
