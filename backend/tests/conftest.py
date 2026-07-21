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


def write_scene_ply(path: Path, n: int = 8, pts: np.ndarray | None = None) -> None:
    if pts is None:
        rng = np.random.default_rng(0)
        pts = rng.standard_normal((n, 3)).astype(np.float32)
    n = len(pts)
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    arr = np.zeros(n, dtype=dtype)
    arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr['red'] = arr['green'] = arr['blue'] = 200
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(str(path))


def build_annotated_root(tmp_path: Path, pts: np.ndarray | None = None) -> tuple[Path, str]:
    """Build a synthesized lidar root with one annotated/demo scene (scan-schema
    v2: a registered preseg result + one session seeded from it). Returns
    (root, session_id). Lives in conftest (not tests/test_lidar_io) so multiple
    test files can use it without cross-module helper imports.

    ``pts`` (default None) lets a caller supply a custom point cloud (e.g. a
    dense grid for the eval-grade gate) — everything else about the fixture
    (n=8 random points, the 4-segment preseg) is unchanged when omitted."""
    root = tmp_path / "lidar"
    scan_dir = root / "annotated" / "demo"
    write_scene_ply(scan_dir / "source" / "scan.ply", n=8, pts=pts)
    n_points = len(pts) if pts is not None else 8
    (scan_dir / "meta.json").write_text(json.dumps({
        "scan_name": "demo", "n_points": n_points, "coords": "world", "units": "meters",
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
    from scan_schema.fingerprint import array_fingerprint as compute_fingerprint
    from scan_schema.layout import ScanLayout
    lay = ScanLayout(scan_dir)
    if pts is None:
        inst = np.array([-1, 0, 0, 1, 1, 2, -1, 3], dtype=np.int32)
        summary = {"segments": [{"id": 0, "class_id": 0},
                                 {"id": 1, "class_id": 1},
                                 {"id": 2, "class_id": 2},
                                 {"id": 3, "class_id": 2}]}
    else:
        inst = np.full(n_points, -1, dtype=np.int32)
        inst[:2] = 0
        summary = {"segments": [{"id": 0, "class_id": 0}]}
    register_preseg(lay, "ransac", inst,
                    summary=summary,
                    generator="ransac", params={})
    # /api/load fingerprints the cloud AFTER load_ply + _recenter (app/core.py)
    # — go through the exact same helpers here (rather than re-deriving the
    # points/centroid by hand) since np.vstack(...).T vs np.stack(axis=1)
    # order-of-summation differences shift the float32 mean by enough to
    # change the fingerprint hash even though the geometry is identical.
    from scenes.point_cloud import load_ply
    from app.core import _recenter
    scan_pc, _ = load_ply(scan_dir / "source" / "scan.ply")
    scan_pc, _offset = _recenter(scan_pc)
    source_fp = compute_fingerprint(scan_pc.points.astype(np.float32))
    sess = create_session(lay, name="demo session", preseg_id="ransac",
                          n_points=n_points, source_fp=source_fp)
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


def dense_grid_pts(spacing=0.005, n=20, offset=(0.0, 0.0, 0.0)):
    """n×n XZ grid at the given spacing (nn distance == spacing), optionally
    translated — offsets > 1e3 trigger the load-time auto-recenter."""
    g = np.arange(n, dtype=np.float32) * spacing
    xs, zs = np.meshgrid(g, g)
    pts = np.stack([xs.ravel(), np.zeros(n * n, dtype=np.float32), zs.ravel()], axis=1)
    return (pts + np.asarray(offset, dtype=np.float32)).astype(np.float32)


@pytest.fixture
def client_with_dense_annotated_scene(monkeypatch, tmp_path):
    """Loaded annotated scene whose 400 points sit on a 5 mm grid — dense
    enough to pass the eval-grade gate (p90 = 5 mm <= 10 mm)."""
    import main
    from fastapi.testclient import TestClient

    root, _sid = build_annotated_root(tmp_path, pts=dense_grid_pts())
    monkeypatch.setattr("app.constants.LIDAR_ROOT", root, raising=False)
    client = TestClient(main.app)
    r = client.post("/api/load", json={"name": "annotated/demo", "max_points": 100})
    assert r.status_code == 200
    return client
