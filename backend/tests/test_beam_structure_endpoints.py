"""Tests for GET/PUT /api/segment/structure (Beam tool persistence)."""
from __future__ import annotations

import json

EMPTY = {"nodes": [], "edges": [], "committed_beams": []}

DOC = {
    "nodes": [{"id": 1, "pos": [0.0, 0.0, 0.0]}, {"id": 2, "pos": [2.0, 0.0, 0.0]}],
    "edges": [{"id": 3, "a": 1, "b": 2, "width": 0.2, "class_id": 10,
               "instance_id": None, "dirty": True}],
    "committed_beams": [{"a": [0.0, 0.0, 0.0], "b": [0.0, 2.0, 0.0], "width": 0.3,
                         "class_id": 11, "instance_id": 42}],
}


def test_structure_empty_then_populated(
        client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    assert client.get("/api/segment/structure").json() == EMPTY
    r = client.put("/api/segment/structure", json=DOC)
    assert r.status_code == 200
    assert client.get("/api/segment/structure").json() == DOC
    # Written to the active session's structure.json on disk.
    sessions = list((scan_dir_for_loaded_scene / "sessions").iterdir())
    docs = [d / "structure.json" for d in sessions if (d / "structure.json").exists()]
    assert len(docs) == 1
    assert json.loads(docs[0].read_text()) == DOC


def test_structure_put_replaces_wholesale(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    client.put("/api/segment/structure", json=DOC)
    client.put("/api/segment/structure", json=EMPTY)
    assert client.get("/api/segment/structure").json() == EMPTY


def test_structure_409_without_session(client):
    assert client.get("/api/segment/structure").status_code == 409
    assert client.put("/api/segment/structure", json=EMPTY).status_code == 409


def test_structure_put_validation(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    bad_width = {**EMPTY, "edges": [{"id": 1, "a": 1, "b": 2, "width": 0.0, "class_id": 10}]}
    assert client.put("/api/segment/structure", json=bad_width).status_code == 422
    bad_pos = {**EMPTY, "nodes": [{"id": 1, "pos": [0.0, 0.0]}]}
    assert client.put("/api/segment/structure", json=bad_pos).status_code == 422


def test_structure_get_session_pin(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    sid = client.get("/api/segment/state").json()["session_id"]
    assert client.get(f"/api/segment/structure?session_id={sid}").status_code == 200
    assert client.get("/api/segment/structure?session_id=nope").status_code == 409


def test_structure_put_session_pin_mismatch(
        client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    r = client.put("/api/segment/structure",
                   json={**DOC, "session_id": "not-the-active-session"})
    assert r.status_code == 409
    # The rejected write must not land on disk.
    sessions = (scan_dir_for_loaded_scene / "sessions").iterdir()
    assert not any((d / "structure.json").exists() for d in sessions)
    # A pin matching the active session is accepted.
    active = client.get("/api/segment/state").json()["session_id"]
    r = client.put("/api/segment/structure", json={**DOC, "session_id": active})
    assert r.status_code == 200
    assert client.get("/api/segment/structure").json() == DOC
