"""Shared pytest fixtures.

VOXA_DATA_DIR must be set before main.py is imported because main.py reads
it at module load time. Setting it here in conftest.py is safe — pytest
imports conftest before any test module.
"""

from __future__ import annotations

import os
import tempfile

import pytest

os.environ["VOXA_DATA_DIR"] = tempfile.mkdtemp(prefix="voxa-test-")


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import main

    return TestClient(main.app)


@pytest.fixture
def client_with_annotated_scene(monkeypatch, tmp_path):
    """TestClient + scene_id for a synthesized annotated scene with labels.

    Reused by Tasks 8–12. Uses monkeypatch.setattr on main.LIDAR_ROOT to
    avoid reloading main (which breaks pydantic model identity).
    """
    from tests.test_lidar_io import _build_annotated_root
    import main
    from fastapi.testclient import TestClient

    root = _build_annotated_root(tmp_path)
    monkeypatch.setattr(main, "LIDAR_ROOT", root, raising=False)
    return TestClient(main.app), "annotated/demo"


@pytest.fixture
def client_with_loaded_annotated_scene(client_with_annotated_scene):
    """TestClient with annotated/demo already loaded (full-resolution labels in _state)."""
    client, scene_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "max_points": 100})
    assert r.status_code == 200
    return client
