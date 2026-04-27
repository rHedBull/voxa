"""Scene discovery + resolution across legacy / annotated / decimated / raw roots."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from scene_registry import discover, resolve


def _write_tiny_ply(path: Path, n: int = 8) -> None:
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    arr = np.zeros(n, dtype=dtype)
    arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr['red'] = arr['green'] = arr['blue'] = 200
    el = PlyElement.describe(arr, 'vertex')
    path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([el], text=False).write(str(path))


def _make_annotated(scan_dir: Path, *, with_labels: bool = True, n: int = 8) -> None:
    """Create a SCHEMA-conformant annotated/<name>/ directory."""
    _write_tiny_ply(scan_dir / "source" / "scan.ply", n=n)
    (scan_dir / "labels").mkdir(parents=True, exist_ok=True)
    if with_labels:
        np.save(scan_dir / "labels" / "gt_class_ids.npy", np.array([-1, 0, 0, 1, 1, 2, -1, 2], dtype=np.int32))
        np.save(scan_dir / "labels" / "gt_segment_ids.npy", np.array([-1, 0, 0, 1, 1, 2, -1, 3], dtype=np.int32))
        (scan_dir / "labels" / "gt_segment_metadata.json").write_text(json.dumps({
            "n_points": n,
            "n_gt_segments": 4,
            "n_labeled_points": 6,
            "class_map": {"pipe": 0, "tank": 1, "equipment": 2},
            "segments": [],
        }))
    (scan_dir / "meta.json").write_text(json.dumps({
        "scan_name": scan_dir.name, "n_points": n, "coords": "world",
        "units": "meters", "class_map_version": 1,
    }))


@pytest.fixture
def lidar_root(tmp_path):
    root = tmp_path / "lidar"
    # annotated scenes
    _make_annotated(root / "annotated" / "munich_water_pump", with_labels=True)
    _make_annotated(root / "annotated" / "factory_large", with_labels=False)
    # decimated PLYs
    _write_tiny_ply(root / "ply_viewer" / "Construction-site-sample-data.ply")
    _write_tiny_ply(root / "ply_viewer" / "Factory-large.ply")
    # raw LAZ — just an empty placeholder file; load_laz isn't called here
    (root / "laz").mkdir(parents=True, exist_ok=True)
    (root / "laz" / "Factory-large.laz").write_bytes(b"placeholder")
    # classes.json
    (root / "classes.json").write_text(json.dumps({
        "version": 1, "unlabeled_id": -1,
        "classes": [
            {"id": 0, "name": "pipe"},
            {"id": 1, "name": "tank"},
            {"id": 2, "name": "equipment"},
        ],
    }))
    return root


@pytest.fixture
def voxa_data(tmp_path):
    """Legacy voxa/data/scenes/* alongside the lidar root."""
    data = tmp_path / "voxa-data"
    _write_tiny_ply(data / "scenes" / "test_scene" / "source.ply")
    return data


def test_discover_finds_all_tiers(voxa_data, lidar_root):
    scenes = discover(voxa_data, lidar_root)
    by_id = {s.scene_id: s for s in scenes}
    assert "legacy/test_scene" in by_id
    assert "annotated/munich_water_pump" in by_id
    assert "annotated/factory_large" in by_id
    assert "decimated/Construction-site-sample-data" in by_id
    assert "decimated/Factory-large" in by_id
    assert "raw/Factory-large" in by_id


def test_discover_sorting_legacy_first_then_tier_name(voxa_data, lidar_root):
    scenes = discover(voxa_data, lidar_root)
    tiers = [s.tier for s in scenes]
    # Legacy comes first, then annotated, decimated, raw.
    assert tiers == sorted(tiers, key=lambda t: ("legacy annotated decimated raw".split()).index(t))


def test_discover_collision_factory_large_in_two_tiers(voxa_data, lidar_root):
    scenes = discover(voxa_data, lidar_root)
    matches = [s for s in scenes if s.name == "Factory-large"]
    assert len(matches) == 2
    assert {m.tier for m in matches} == {"decimated", "raw"}


def test_annotated_has_labels_flag_only_when_arrays_present(voxa_data, lidar_root):
    scenes = {s.scene_id: s for s in discover(voxa_data, lidar_root)}
    assert scenes["annotated/munich_water_pump"].has_labels is True
    assert scenes["annotated/factory_large"].has_labels is False


def test_resolve_tier_prefixed_id(voxa_data, lidar_root):
    s = resolve("annotated/munich_water_pump", voxa_data, lidar_root)
    assert s.tier == "annotated" and s.name == "munich_water_pump"


def test_resolve_bare_legacy_name_back_compat(voxa_data, lidar_root):
    s = resolve("test_scene", voxa_data, lidar_root)
    assert s.tier == "legacy" and s.name == "test_scene"


def test_resolve_unknown_raises_keyerror(voxa_data, lidar_root):
    with pytest.raises(KeyError):
        resolve("annotated/does_not_exist", voxa_data, lidar_root)
    with pytest.raises(KeyError):
        resolve("does_not_exist", voxa_data, lidar_root)


def test_resolve_collision_prefers_tier_prefixed(voxa_data, lidar_root):
    # Both decimated/Factory-large and raw/Factory-large exist; bare 'Factory-large'
    # is allowed to fall through to *any* match (current contract: legacy first,
    # then any tier). Asserting the tier-prefixed forms succeed independently is
    # the contract that actually matters.
    a = resolve("decimated/Factory-large", voxa_data, lidar_root)
    b = resolve("raw/Factory-large", voxa_data, lidar_root)
    assert a.tier == "decimated" and b.tier == "raw"


def test_discover_no_lidar_root_returns_only_legacy(voxa_data):
    scenes = discover(voxa_data, None)
    assert {s.tier for s in scenes} == {"legacy"}
    assert scenes[0].name == "test_scene"
