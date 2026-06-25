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
        assert {"id", "label", "color", "hotkey", "class_id"} <= set(c.keys())


def test_class_name_to_id_uses_explicit_ids(tmp_path, monkeypatch):
    """Explicit `id:` in classes.yaml wins over yaml position (the positional
    fallback silently corrupted labels when the yaml was reordered)."""
    import app.core as core
    cfg = tmp_path / "classes.yaml"
    cfg.write_text(
        "classes:\n"
        "  pipe:\n    id: 0\n    label: Pipe\n"
        "  fitting:\n    id: 5\n    label: Fitting\n"
        "  tank:\n    id: 1\n    label: Tank\n"
        "  legacy_positional:\n    label: NoId\n"
    )
    monkeypatch.setattr(core, "CONFIG_PATH", cfg)
    assert core._voxa_class_name_to_id() == {
        "pipe": 0, "fitting": 5, "tank": 1, "legacy_positional": 3,
    }


def test_class_name_to_id_rejects_duplicate_ids(tmp_path, monkeypatch):
    import pytest
    import app.core as core
    cfg = tmp_path / "classes.yaml"
    cfg.write_text("classes:\n  pipe:\n    id: 1\n  tank:\n    id: 1\n")
    monkeypatch.setattr(core, "CONFIG_PATH", cfg)
    with pytest.raises(ValueError, match="duplicate class ids"):
        core._voxa_class_name_to_id()


def test_scenes_returns_list(client):
    r = client.get("/api/scenes")
    assert r.status_code == 200
    assert isinstance(r.json(), list)


def test_load_unknown_scene_404(client):
    r = client.post("/api/load", json={"name": "does-not-exist", "max_points": 1000})
    assert r.status_code == 404


def test_annotation_roundtrip_preserves_all_fields(client):
    """Every Cuboid field plus AnnotationDoc.meta must survive a save+load.
    Silent field drop = lost annotation work, so we assert the full payload."""
    scene = "rt-scene"
    inst = {
        "id": "inst-1",
        "cls": "pipe",
        "label": "Pipe 3",
        "color": "#22c55e",
        "center": [0.12, 0.40, -0.05],
        "size": [0.20, 0.05, 0.05],
        "rotation": [0.1, 0.2, 0.3],
        "conf": 0.87,
        "source": "auto",
    }
    doc = {"scene": scene, "kind": "gt", "instances": [inst], "meta": {"note": "hello"}}

    put = client.put(f"/api/annotations/gt/{scene}", json=doc)
    assert put.status_code == 200
    assert put.json()["count"] == 1

    body = client.get(f"/api/annotations/gt/{scene}").json()
    assert body["scene"] == scene
    assert body["meta"] == {"note": "hello"}
    assert len(body["instances"]) == 1
    got = body["instances"][0]
    for k, v in inst.items():
        assert got[k] == v, f"field {k!r} not preserved: {got[k]!r} != {v!r}"


def test_get_annotation_missing_returns_empty(client):
    r = client.get("/api/annotations/gt/never-saved")
    assert r.status_code == 200
    assert r.json()["instances"] == []


def test_annotations_round_trip_tier_prefixed_scene(client):
    """Tier-prefixed ids contain `/`; the route puts kind first so scene:path
    can match greedily."""
    scene = "annotated/smart_ais"
    doc = {"scene": scene, "kind": "gt", "instances": [], "meta": {}}
    put = client.put(f"/api/annotations/gt/{scene}", json=doc)
    assert put.status_code == 200
    body = client.get(f"/api/annotations/gt/{scene}").json()
    assert body["scene"] == scene


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
