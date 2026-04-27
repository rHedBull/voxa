"""End-to-end /api/load with the multi-root registry.

These tests exercise main.py against a synthesized lidar root + voxa data
dir; conftest.py sets VOXA_DATA_DIR before main.py imports, and we set
VOXA_LIDAR_ROOT here before client construction so main.LIDAR_ROOT picks it up.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from plyfile import PlyData, PlyElement


def _write_ply(path: Path, n: int = 8) -> None:
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    arr = np.zeros(n, dtype=dtype)
    arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr['red'] = arr['green'] = arr['blue'] = 200
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(str(path))


@pytest.fixture
def lidar_client(tmp_path, monkeypatch):
    """Spin up a fresh main.app rooted at a synthesized lidar tree."""
    lidar = tmp_path / "lidar"
    scan = lidar / "annotated" / "demo"
    _write_ply(scan / "source" / "scan.ply", n=8)
    (scan / "labels").mkdir(parents=True, exist_ok=True)
    np.save(scan / "labels" / "gt_class_ids.npy",
            np.array([-1, 0, 0, 1, 1, 2, -1, 2], dtype=np.int32))
    np.save(scan / "labels" / "gt_segment_ids.npy",
            np.array([-1, 0, 0, 1, 1, 2, -1, 3], dtype=np.int32))
    (scan / "labels" / "gt_segment_metadata.json").write_text(json.dumps({
        "n_points": 8, "class_map": {"pipe": 0, "tank": 1, "equipment": 2},
        "n_gt_segments": 4, "n_labeled_points": 6, "segments": [],
    }))
    (scan / "meta.json").write_text(json.dumps({"scan_name": "demo", "n_points": 8}))

    # Bare PLY scene as well, for the no-labels case.
    bare = lidar / "annotated" / "stub"
    _write_ply(bare / "source" / "scan.ply", n=4)
    (bare / "labels").mkdir(parents=True, exist_ok=True)
    (bare / "meta.json").write_text(json.dumps({"scan_name": "stub", "n_points": 4}))

    # Patch main.LIDAR_ROOT in place. Reloading main breaks pydantic model
    # identity (ClassDef-from-old-load isn't ClassDef-from-new-load), so we
    # mutate the module attribute directly instead.
    import main
    monkeypatch.setattr(main, "LIDAR_ROOT", lidar, raising=False)
    return TestClient(main.app)


def test_scenes_lists_annotated_with_label_flag(lidar_client):
    body = lidar_client.get("/api/scenes").json()
    by_id = {s["id"]: s for s in body}
    assert by_id["annotated/demo"]["has_labels"] is True
    assert by_id["annotated/demo"]["tier"] == "annotated"
    assert by_id["annotated/stub"]["has_labels"] is False


def test_load_annotated_surfaces_labels_and_palette(lidar_client):
    r = lidar_client.post("/api/load",
                          json={"name": "annotated/demo", "max_points": 100})
    assert r.status_code == 200
    body = r.json()

    assert body["scene"] == "annotated/demo"
    assert body["num_points"] == 8
    assert body["class_ids"] is not None
    assert body["instance_ids"] is not None
    assert body["class_palette"] is not None and len(body["class_palette"]) >= 3
    assert body["n_classes"] == 3
    assert body["n_instances"] == 4

    # Decode and confirm the round-trip preserved values.
    class_bytes = base64.b64decode(body["class_ids"])
    class_arr = np.frombuffer(class_bytes, dtype=np.int8)
    assert class_arr.tolist() == [-1, 0, 0, 1, 1, 2, -1, 2]

    inst_bytes = base64.b64decode(body["instance_ids"])
    inst_arr = np.frombuffer(inst_bytes, dtype=np.int32)
    assert inst_arr.tolist() == [-1, 0, 0, 1, 1, 2, -1, 3]


def test_load_unlabeled_omits_label_fields(lidar_client):
    body = lidar_client.post("/api/load",
                             json={"name": "annotated/stub", "max_points": 100}).json()
    assert body["class_ids"] is None
    assert body["instance_ids"] is None
    # Palette is allowed to be absent or empty.
    assert not body["class_palette"]


def test_recenter_zero_for_already_centered_scene(lidar_client):
    body = lidar_client.post("/api/load",
                             json={"name": "annotated/demo", "max_points": 100}).json()
    # Tiny synthesized cloud is centered around origin; recenter offset
    # is the all-zero "no recenter needed" sentinel.
    assert body["recenter_offset"] == [0.0, 0.0, 0.0]
