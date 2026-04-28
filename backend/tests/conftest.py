"""Shared pytest fixtures.

VOXA_DATA_DIR must be set before main.py is imported because main.py reads
it at module load time. Setting it here in conftest.py is safe — pytest
imports conftest before any test module.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

os.environ["VOXA_DATA_DIR"] = tempfile.mkdtemp(prefix="voxa-test-")


@pytest.fixture(autouse=True)
def _reset_main_state():
    """main._state is a module-level singleton mutated by /api/load and
    /api/segment/*. Reset between tests so order doesn't influence outcomes."""
    yield
    import main
    main._state.update(
        scene=None, pc=None, mesh=None, subsample_idx=None,
        intensity=None, labels=None, recenter_offset=[0.0, 0.0, 0.0], seg=None,
    )


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import main

    return TestClient(main.app)


def _write_annotated_scene_ply(path: Path, n: int = 8) -> None:
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    arr = np.zeros(n, dtype=dtype)
    arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr['red'] = arr['green'] = arr['blue'] = 200
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(str(path))


def build_annotated_root(tmp_path: Path) -> Path:
    """Build a synthesized lidar root with one annotated/demo scene with
    real labels. Lives in conftest (not tests/test_lidar_io) so multiple
    test files can use it without cross-module helper imports."""
    root = tmp_path / "lidar"
    scan_dir = root / "annotated" / "demo"
    _write_annotated_scene_ply(scan_dir / "source" / "scan.ply", n=8)
    (scan_dir / "labels").mkdir(parents=True, exist_ok=True)
    np.save(scan_dir / "labels" / "gt_class_ids.npy",
            np.array([-1, 0, 0, 1, 1, 2, -1, 2], dtype=np.int32))
    np.save(scan_dir / "labels" / "gt_segment_ids.npy",
            np.array([-1, 0, 0, 1, 1, 2, -1, 3], dtype=np.int32))
    (scan_dir / "labels" / "gt_segment_metadata.json").write_text(json.dumps({
        "n_points": 8, "n_gt_segments": 4, "n_labeled_points": 6,
        "class_map": {"pipe": 0, "tank": 1, "equipment": 2},
        "segments": [],
    }))
    (scan_dir / "meta.json").write_text(json.dumps({
        "scan_name": "demo", "n_points": 8, "coords": "world", "units": "meters",
    }))
    (root / "classes.json").write_text(json.dumps({
        "version": 1, "unlabeled_id": -1,
        "classes": [{"id": 0, "name": "pipe"},
                    {"id": 1, "name": "tank"},
                    {"id": 2, "name": "equipment"}],
    }))
    return root


@pytest.fixture
def client_with_annotated_scene(monkeypatch, tmp_path):
    """TestClient + scene_id for a synthesized annotated scene with labels.

    Reused by Tasks 8–12. Uses monkeypatch.setattr on main.LIDAR_ROOT to
    avoid reloading main (which breaks pydantic model identity).
    """
    import main
    from fastapi.testclient import TestClient

    root = build_annotated_root(tmp_path)
    monkeypatch.setattr(main, "LIDAR_ROOT", root, raising=False)
    return TestClient(main.app), "annotated/demo"


@pytest.fixture
def client_with_loaded_annotated_scene(client_with_annotated_scene):
    """TestClient with annotated/demo already loaded (full-resolution labels in _state)."""
    client, scene_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "max_points": 100})
    assert r.status_code == 200
    return client
