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


def test_session_dir_for_annotated_tier(voxa_data, lidar_root):
    s = resolve("annotated/munich_water_pump", voxa_data, lidar_root)
    assert s.session_dir == lidar_root / "annotated" / "munich_water_pump" / "session"


def test_session_dir_for_legacy_tier(voxa_data, lidar_root):
    s = resolve("legacy/test_scene", voxa_data, lidar_root)
    assert s.session_dir == voxa_data / "sessions" / "legacy__test_scene"


def test_session_dir_for_decimated_tier(voxa_data, lidar_root):
    s = resolve("decimated/Factory-large", voxa_data, lidar_root)
    assert s.session_dir == voxa_data / "sessions" / "decimated__Factory-large"


def test_session_dir_for_raw_tier(voxa_data, lidar_root):
    s = resolve("raw/Factory-large", voxa_data, lidar_root)
    assert s.session_dir == voxa_data / "sessions" / "raw__Factory-large"


def test_session_dir_raises_when_data_dir_missing_for_legacy():
    """Non-annotated tiers require data_dir; surfacing as ValueError prevents
    writing session files to None."""
    from scene_registry import _session_dir_for
    with pytest.raises(ValueError, match="data_dir"):
        _session_dir_for("legacy", "foo", None, None)
    with pytest.raises(ValueError, match="data_dir"):
        _session_dir_for("decimated", "foo", None, None)
    with pytest.raises(ValueError, match="data_dir"):
        _session_dir_for("raw", "foo", None, None)


# ── Mesh discovery + GLB scene-graph validity ──────────────────────────────
#
# Backstory: SMART-AIS scans ship a `mesh.optimized.glb` whose scene root has
# no children that reference the mesh node — GLTFLoader loads an empty scene
# and the mesh-companion window shows only the floor + axes. Registry should
# (a) prefer the canonical `mesh.glb`, (b) fall back to `mesh.r05*.glb`
# variants which are known good, and (c) NOT pick `mesh.optimized.glb` even
# if it's the only file present.


def _valid_glb_bytes() -> bytes:
    """Hand-built minimal binary glTF: 1 triangle, scene → node → mesh.

    Uses pygltflib so we exercise the same writer the build pipeline uses.
    """
    import struct
    from pygltflib import (
        GLTF2, Scene, Node, Mesh, Primitive, Attributes, Accessor,
        BufferView, Buffer, ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, FLOAT,
        UNSIGNED_INT, SCALAR, VEC3,
    )
    # 3 verts (one triangle) + 3 indices
    verts = struct.pack('<9f', 0, 0, 0,  1, 0, 0,  0, 1, 0)
    idx = struct.pack('<3I', 0, 1, 2)
    blob = verts + idx
    g = GLTF2()
    g.buffers = [Buffer(byteLength=len(blob))]
    g.bufferViews = [
        BufferView(buffer=0, byteOffset=0, byteLength=len(verts), target=ARRAY_BUFFER),
        BufferView(buffer=0, byteOffset=len(verts), byteLength=len(idx), target=ELEMENT_ARRAY_BUFFER),
    ]
    g.accessors = [
        Accessor(bufferView=0, componentType=FLOAT, count=3, type=VEC3,
                 min=[0, 0, 0], max=[1, 1, 0]),
        Accessor(bufferView=1, componentType=UNSIGNED_INT, count=3, type=SCALAR),
    ]
    g.meshes = [Mesh(primitives=[Primitive(attributes=Attributes(POSITION=0), indices=1)])]
    g.nodes = [Node(mesh=0)]
    g.scenes = [Scene(nodes=[0])]
    g.scene = 0
    g.set_binary_blob(blob)
    return b"".join(g.save_to_bytes())


def _broken_glb_bytes() -> bytes:
    """GLB with the mesh node orphaned from the scene root.

    Mirrors the actual `mesh.optimized.glb` bug: scene → 'world' node which
    has no children; the mesh-bearing node[0] is unreachable from any scene
    root, so GLTFLoader's `gltf.scene` is an empty Group.
    """
    import struct
    from pygltflib import (
        GLTF2, Scene, Node, Mesh, Primitive, Attributes, Accessor,
        BufferView, Buffer, ARRAY_BUFFER, ELEMENT_ARRAY_BUFFER, FLOAT,
        UNSIGNED_INT, SCALAR, VEC3,
    )
    verts = struct.pack('<9f', 0, 0, 0,  1, 0, 0,  0, 1, 0)
    idx = struct.pack('<3I', 0, 1, 2)
    blob = verts + idx
    g = GLTF2()
    g.buffers = [Buffer(byteLength=len(blob))]
    g.bufferViews = [
        BufferView(buffer=0, byteOffset=0, byteLength=len(verts), target=ARRAY_BUFFER),
        BufferView(buffer=0, byteOffset=len(verts), byteLength=len(idx), target=ELEMENT_ARRAY_BUFFER),
    ]
    g.accessors = [
        Accessor(bufferView=0, componentType=FLOAT, count=3, type=VEC3,
                 min=[0, 0, 0], max=[1, 1, 0]),
        Accessor(bufferView=1, componentType=UNSIGNED_INT, count=3, type=SCALAR),
    ]
    g.meshes = [Mesh(primitives=[Primitive(attributes=Attributes(POSITION=0), indices=1)])]
    # node[0] has the mesh but is orphaned; scene root points at node[1]
    # ('world') which has no children — same shape as mesh.optimized.glb.
    g.nodes = [Node(mesh=0), Node(name="world")]
    g.scenes = [Scene(nodes=[1])]
    g.scene = 0
    g.set_binary_blob(blob)
    return b"".join(g.save_to_bytes())


def _glb_has_reachable_mesh(path: Path) -> bool:
    """Sanity-check helper: a GLB whose scene root reaches at least one
    mesh-bearing node. The companion test uses this to assert that our
    `_broken_glb_bytes` writer actually produces the bug we're guarding
    against — without this we'd just be testing filename plumbing."""
    from pygltflib import GLTF2
    g = GLTF2().load(str(path))
    if not g.scenes or g.scene is None:
        return False
    seen, stack = set(), list(g.scenes[g.scene].nodes or [])
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        n = g.nodes[i]
        if n.mesh is not None:
            return True
        stack.extend(n.children or [])
    return False


def _scan_with_mesh(scan_dir: Path, *mesh_files: tuple[str, bytes]) -> None:
    """Build a minimal annotated scan and drop the given mesh files into
    `source/`. `mesh_files` is a sequence of (filename, glb_bytes)."""
    _make_annotated(scan_dir)
    src = scan_dir / "source"
    src.mkdir(parents=True, exist_ok=True)
    for fname, blob in mesh_files:
        (src / fname).write_bytes(blob)


def test_glb_helpers_round_trip(tmp_path):
    """Sanity: our valid GLB has a reachable mesh; our broken GLB does not.
    Without this the registry tests below could pass for the wrong reason."""
    valid = tmp_path / "v.glb"; valid.write_bytes(_valid_glb_bytes())
    broken = tmp_path / "b.glb"; broken.write_bytes(_broken_glb_bytes())
    assert _glb_has_reachable_mesh(valid)
    assert not _glb_has_reachable_mesh(broken)


def test_registry_picks_canonical_mesh_glb(tmp_path):
    """SCHEMA v1.2: the registry only honors `<scan>/source/mesh.glb`.
    Build pipelines that emit other names (mesh.optimized.glb,
    mesh.r05*.glb, etc.) must rename or symlink to the canonical name
    to be picked up — keeps the registry trivial and prevents a single
    broken GLB from being silently advertised as the scene mesh."""
    lidar = tmp_path / "lidar"
    _scan_with_mesh(lidar / "annotated" / "scan_a", ("mesh.glb", _valid_glb_bytes()))
    s = resolve("annotated/scan_a", tmp_path / "data", lidar)
    assert s.has_mesh
    assert s.extras["mesh_path"].endswith("/source/mesh.glb")


def test_registry_ignores_non_canonical_glb_variants(tmp_path):
    """Build-pipeline variants under non-canonical names are NOT picked
    up — including the broken mesh.optimized.glb shape from the older
    pipeline. The registry doesn't guess which variant is good."""
    lidar = tmp_path / "lidar"
    _scan_with_mesh(lidar / "annotated" / "scan_b",
                    ("mesh.r05.glb", _valid_glb_bytes()),
                    ("mesh.r05.small.glb", _valid_glb_bytes()),
                    ("mesh.optimized.glb", _broken_glb_bytes()))
    s = resolve("annotated/scan_b", tmp_path / "data", lidar)
    assert not s.has_mesh
    assert s.extras["mesh_path"] is None


# ── SAM3 render discovery via SCHEMA v1.2 per-scene renders/ ────────────────


def test_sam3_renders_discovery_via_scan_dir(tmp_path):
    """SCHEMA v1.2: SAM3 render runs live under `<scan>/renders/<run>/`
    next to a manifest.json. discover_render_runs should find them when
    given the scan dir, with no env-var or external root involved."""
    from sam3_features import discover_render_runs
    lidar = tmp_path / "lidar"
    scan = lidar / "annotated" / "scene_with_renders"
    _make_annotated(scan)
    run = scan / "renders" / "orbit01__ultra__20260101-000000"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text(json.dumps({
        "scene": "scene_with_renders",
        "frames": [
            {"file": "frame_000.png", "position": [0, 0, 0], "target": [1, 0, 0]},
            {"file": "frame_001.png", "position": [0, 0, 1], "target": [1, 0, 0]},
        ],
    }))
    # An empty sibling without manifest.json is ignored (not a render run).
    (scan / "renders" / "incomplete").mkdir()

    runs = discover_render_runs("scene_with_renders", scan_dir=scan)
    assert len(runs) == 1
    assert runs[0].name == "orbit01__ultra__20260101-000000"
    assert runs[0].n_frames == 2
    assert runs[0].has_orbit_target is True


def test_sam3_renders_discovery_empty_scan(tmp_path):
    """No renders/ dir → empty list (not an error)."""
    from sam3_features import discover_render_runs
    lidar = tmp_path / "lidar"
    scan = lidar / "annotated" / "no_renders"
    _make_annotated(scan)
    assert discover_render_runs("no_renders", scan_dir=scan) == []
