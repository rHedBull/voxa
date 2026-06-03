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
# Tests use 8-point fixtures; the production default of 10 would filter
# every segment to (-1,-1) and break label/save round-trips.
os.environ.setdefault("VOXA_MIN_SEGMENT_POINTS", "1")


@pytest.fixture(autouse=True)
def _reset_main_state():
    """main._state is a module-level singleton mutated by /api/load and
    /api/segment/*. Reset between tests so order doesn't influence outcomes."""
    yield
    import main
    main._state.update(
        scene=None, pc=None, mesh=None, subsample_idx=None,
        intensity=None, labels=None, recenter_offset=[0.0, 0.0, 0.0], seg=None,
        session_id=None, source_fp=None,
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


def build_annotated_root(tmp_path: Path) -> tuple[Path, str]:
    """Build a synthesized lidar root with one annotated/demo scene (scan-schema
    v2: a registered preseg result + one session seeded from it). Returns
    (root, session_id). Lives in conftest (not tests/test_lidar_io) so multiple
    test files can use it without cross-module helper imports."""
    root = tmp_path / "lidar"
    scan_dir = root / "annotated" / "demo"
    _write_annotated_scene_ply(scan_dir / "source" / "scan.ply", n=8)
    (scan_dir / "meta.json").write_text(json.dumps({
        "scan_name": "demo", "n_points": 8, "coords": "world", "units": "meters",
        "schema_version": "2.0", "class_map_version": 1, "source_mesh": "mesh.glb",
    }))
    (root / "classes.json").write_text(json.dumps({
        "version": 1, "unlabeled_id": -1,
        "classes": [{"id": 0, "name": "pipe"},
                    {"id": 1, "name": "tank"},
                    {"id": 2, "name": "equipment"}],
    }))

    # v2: preseg result + one session seeded from it
    from preseg.preseg_store import register_preseg
    from labeling.session_store import create_session
    from labeling.segment_io import compute_fingerprint
    from scenes.scan_layout import ScanLayout
    from plyfile import PlyData
    lay = ScanLayout(scan_dir)
    inst = np.array([-1, 0, 0, 1, 1, 2, -1, 3], dtype=np.int32)
    register_preseg(lay, "ransac", inst,
                    summary={"segments": [{"id": 0, "class_id": 0},
                                          {"id": 1, "class_id": 1},
                                          {"id": 2, "class_id": 2},
                                          {"id": 3, "class_id": 2}]},
                    generator="ransac", params={})
    ply = PlyData.read(str(scan_dir / "source" / "scan.ply"))["vertex"]
    pts = np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float32)
    source_fp = compute_fingerprint(pts)
    sess = create_session(lay, name="demo session", preseg_id="ransac",
                          n_points=8, source_fp=source_fp)
    return root, sess.session_id


@pytest.fixture
def client_with_annotated_scene(monkeypatch, tmp_path):
    """TestClient + scene_id + session_id for a synthesized annotated scene.

    Reused by Tasks 8–12. Uses monkeypatch.setattr on main.LIDAR_ROOT to
    avoid reloading main (which breaks pydantic model identity).
    """
    import main
    from fastapi.testclient import TestClient

    root, session_id = build_annotated_root(tmp_path)
    monkeypatch.setattr("app.constants.LIDAR_ROOT", root, raising=False)
    return TestClient(main.app), "annotated/demo", session_id


@pytest.fixture
def client_with_loaded_annotated_scene(client_with_annotated_scene):
    """TestClient with annotated/demo already loaded (full-resolution labels in _state)."""
    client, scene_id, _session_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "max_points": 100})
    assert r.status_code == 200
    return client


@pytest.fixture
def scan_dir_for_loaded_scene(client_with_loaded_annotated_scene, tmp_path):
    """On-disk scan_dir for the annotated/demo scene built by build_annotated_root.

    Depends on client_with_loaded_annotated_scene to enforce that the scene
    exists on disk before this path is read — without that link, the fixture
    silently returns a path under a different tmp_path when used standalone.
    """
    del client_with_loaded_annotated_scene
    return tmp_path / "lidar" / "annotated" / "demo"
