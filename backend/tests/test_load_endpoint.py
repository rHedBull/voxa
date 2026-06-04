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
from PIL import Image
from scenes.fingerprint import cloud_fingerprint  # noqa: E402  (backend on path via conftest)
from scenes.frame import Frame
from scenes.render_meta import write_render_meta


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


def _seed_v2_session(scan_dir: Path, inst: np.ndarray, *, summary_segments: list,
                     name: str = "s") -> str:
    """Register a preseg + create a session seeded from it (scan-schema v2).
    Returns the session_id. The scan dir must already have source/scan.ply +
    a v2 meta.json + a sibling classes.json under the lidar root."""
    from preseg.preseg_store import register_preseg
    from labeling.session_store import create_session
    from labeling.segment_io import compute_fingerprint
    from scenes.scan_layout import ScanLayout
    lay = ScanLayout(scan_dir)
    register_preseg(lay, "ransac", inst.astype(np.int32),
                    summary={"segments": summary_segments},
                    generator="ransac", params={})
    ply = PlyData.read(str(scan_dir / "source" / "scan.ply"))["vertex"]
    pts = np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float32)
    sess = create_session(lay, name=name, preseg_id="ransac",
                          n_points=len(inst), source_fp=compute_fingerprint(pts))
    return sess.session_id


def _write_classes_json(lidar: Path) -> None:
    (lidar / "classes.json").write_text(json.dumps({
        "version": 1, "unlabeled_id": -1,
        "classes": [{"id": 0, "name": "pipe"},
                    {"id": 1, "name": "tank"},
                    {"id": 2, "name": "equipment"}],
    }))


@pytest.fixture
def lidar_client(tmp_path, monkeypatch):
    """Spin up a fresh main.app rooted at a synthesized lidar tree (scan-schema v2)."""
    lidar = tmp_path / "lidar"
    scan = lidar / "annotated" / "demo"
    _write_ply(scan / "source" / "scan.ply", n=8)
    (scan / "meta.json").write_text(json.dumps({
        "scan_name": "demo", "n_points": 8, "schema_version": "2.0",
        "class_map_version": 1, "source_mesh": "mesh.glb"}))
    _write_classes_json(lidar)
    _seed_v2_session(scan, np.array([-1, 0, 0, 1, 1, 2, -1, 3], dtype=np.int32),
                     summary_segments=[{"id": 0, "class_id": 0},
                                       {"id": 1, "class_id": 1},
                                       {"id": 2, "class_id": 2},
                                       {"id": 3, "class_id": 2}], name="demo session")

    # Bare PLY scene as well, for the no-sessions case.
    bare = lidar / "annotated" / "stub"
    _write_ply(bare / "source" / "scan.ply", n=4)
    (bare / "meta.json").write_text(json.dumps({
        "scan_name": "stub", "n_points": 4, "schema_version": "2.0",
        "source_mesh": "mesh.glb"}))

    # Patch main.LIDAR_ROOT in place. Reloading main breaks pydantic model
    # identity (ClassDef-from-old-load isn't ClassDef-from-new-load), so we
    # mutate the module attribute directly instead.
    import main
    monkeypatch.setattr("app.constants.LIDAR_ROOT", lidar, raising=False)
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


def test_load_scan_without_sessions_returns_empty(lidar_client):
    """No sessions on disk → no active session → class/instance arrays absent."""
    body = lidar_client.post("/api/load",
                             json={"name": "annotated/stub", "max_points": 100}).json()
    assert body["class_ids"] is None
    assert body["instance_ids"] is None
    assert body["session_id"] is None
    assert body["sessions"] == []


def test_load_resumes_last_worked_session(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "max_points": 100})
    assert r.status_code == 200
    j = r.json()
    assert j["session_id"] == session_id
    assert j["class_ids"] is not None
    assert {s["session_id"] for s in j["sessions"]} == {session_id}


def test_load_explicit_unknown_session_404(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "session_id": "nope"})
    assert r.status_code == 404


def test_load_pin_mismatch_409(client_with_annotated_scene, tmp_path):
    """Re-register the preseg with different content after the session was
    pinned → its declared fingerprint changes → resume raises 409."""
    client, scene_id, session_id = client_with_annotated_scene
    from main import _resolve
    from scenes.scan_layout import ScanLayout
    from preseg.preseg_store import register_preseg
    src = _resolve(scene_id)
    lay = ScanLayout(Path(src.extras["scan_dir"]))
    register_preseg(lay, "ransac",
                    np.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int32),
                    summary={"segments": [{"id": 0, "class_id": 0},
                                          {"id": 1, "class_id": 1}]},
                    generator="ransac", params={})
    r = client.post("/api/load", json={"name": scene_id, "session_id": session_id})
    assert r.status_code == 409
    detail = r.json()["detail"]
    assert detail["error"] == "session_pin_mismatch"
    assert detail["diverged"] == "preseg"
    assert detail["session_id"] == session_id
    assert detail["message"]


def test_load_explicit_corrupt_session_409(client_with_annotated_scene):
    """A corrupt session is listed (corrupt=True) so it passes the 404
    membership check; explicitly resuming it must be a clean 409, not a 500."""
    client, scene_id, session_id = client_with_annotated_scene
    from main import _resolve
    from scenes.scan_layout import ScanLayout
    src = _resolve(scene_id)
    lay = ScanLayout(Path(src.extras["scan_dir"]))
    lay.session(session_id).session_json.write_text("{broken")
    r = client.post("/api/load", json={"name": scene_id, "session_id": session_id})
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "session_unreadable"


def test_recenter_zero_for_already_centered_scene(lidar_client):
    body = lidar_client.post("/api/load",
                             json={"name": "annotated/demo", "max_points": 100}).json()
    # Tiny synthesized cloud is centered around origin; recenter offset
    # is the all-zero "no recenter needed" sentinel.
    assert body["recenter_offset"] == [0.0, 0.0, 0.0]


def test_mesh_endpoint_404_when_no_glb(lidar_client):
    r = lidar_client.get("/api/mesh/annotated/demo")
    assert r.status_code == 404


def test_mesh_endpoint_serves_glb_when_present(lidar_client, tmp_path, monkeypatch):
    """When source/mesh.glb exists, /api/mesh streams it with the GLB
    content-type, and /api/load surfaces the URL in mesh_url."""
    import main
    from plyfile import PlyData, PlyElement

    lidar = tmp_path / "lidar-mesh"
    scan = lidar / "annotated" / "withmesh"
    (scan / "source").mkdir(parents=True, exist_ok=True)
    # Tiny valid PLY.
    arr = np.zeros(4, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(
        str(scan / "source" / "scan.ply"))
    (scan / "meta.json").write_text(
        '{"scan_name": "withmesh", "n_points": 4, "schema_version": "2.0"}')
    # Pretend mesh — content body doesn't matter for these assertions.
    (scan / "source" / "mesh.glb").write_bytes(b"glTF\x02\x00\x00\x00")

    monkeypatch.setattr("app.constants.LIDAR_ROOT", lidar, raising=False)

    body = lidar_client.post("/api/load",
                             json={"name": "annotated/withmesh", "max_points": 10}).json()
    # mesh_url carries an mtime cache-buster (?v=…) that varies per run.
    assert body["mesh_url"].startswith("/api/mesh/annotated/withmesh")

    r = lidar_client.get(body["mesh_url"])
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("model/gltf-binary")
    assert r.content == b"glTF\x02\x00\x00\x00"

    # /api/scenes should also flag has_mesh.
    info = next(s for s in lidar_client.get("/api/scenes").json()
                if s["id"] == "annotated/withmesh")
    assert info["has_mesh"] is True


def _write_tall_ply(scan_dir: Path, n: int = 10):
    """A 2m × 4m × 20m cloud, with the longest axis on Z (source-frame height)."""
    from plyfile import PlyData, PlyElement
    pts = np.zeros((n, 3), dtype=np.float32)
    pts[:, 0] = np.linspace(-1, 1, n)        # X: 2m
    pts[:, 1] = np.linspace(-2, 2, n)        # Y: 4m
    pts[:, 2] = np.linspace(-10, 10, n)      # Z: 20m — tallest
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    arr = np.zeros(n, dtype=dtype)
    arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr['red'] = arr['green'] = arr['blue'] = 200
    (scan_dir / "source").mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(
        str(scan_dir / "source" / "scan.ply"))
    (scan_dir / "labels").mkdir(parents=True, exist_ok=True)


def test_annotated_z_up_swap_when_source_is_laz(lidar_client, tmp_path, monkeypatch):
    """meta.json points at source_laz → loader rotates Z-up → Y-up."""
    import main
    lidar = tmp_path / "lidar-zup"
    scan_dir = lidar / "annotated" / "tall_laz"
    _write_tall_ply(scan_dir)
    (scan_dir / "meta.json").write_text(
        '{"scan_name": "tall_laz", "n_points": 10, "source_laz": "x.laz",'
        ' "schema_version": "2.0"}')

    monkeypatch.setattr("app.constants.LIDAR_ROOT", lidar, raising=False)
    body = lidar_client.post("/api/load",
                             json={"name": "annotated/tall_laz", "max_points": 100}).json()
    extents = [body["bbox_max"][i] - body["bbox_min"][i] for i in range(3)]
    # 20m source-Z → Y after swap.
    assert extents[1] == pytest.approx(20.0, abs=1e-3), f"got extents={extents}"
    assert extents[0] == pytest.approx(2.0, abs=1e-3)
    assert extents[2] == pytest.approx(4.0, abs=1e-3)
    assert body["mesh_is_z_up"] is False  # no mesh on this scene


def test_annotated_no_swap_when_source_is_glb(lidar_client, tmp_path, monkeypatch):
    """meta.json points at source_mesh (glTF Y-up) → loader leaves the cloud as-is."""
    import main
    lidar = tmp_path / "lidar-yup"
    scan_dir = lidar / "annotated" / "tall_glb"
    _write_tall_ply(scan_dir)
    (scan_dir / "meta.json").write_text(
        '{"scan_name": "tall_glb", "n_points": 10, "source_mesh": "source/mesh.glb",'
        ' "schema_version": "2.0"}')

    monkeypatch.setattr("app.constants.LIDAR_ROOT", lidar, raising=False)
    body = lidar_client.post("/api/load",
                             json={"name": "annotated/tall_glb", "max_points": 100}).json()
    extents = [body["bbox_max"][i] - body["bbox_min"][i] for i in range(3)]
    # No swap → Z stays Z (the 20m tall axis), Y stays Y (4m), X stays X (2m).
    assert extents[2] == pytest.approx(20.0, abs=1e-3), f"got extents={extents}"
    assert extents[1] == pytest.approx(4.0, abs=1e-3)
    assert extents[0] == pytest.approx(2.0, abs=1e-3)


def test_load_annotated_creates_segment_session(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "max_points": 100})
    assert r.status_code == 200
    import main
    assert main._state["seg"] is not None
    assert main._state["seg"].class_ids.shape == main._state["seg"].instance_ids.shape


def test_load_with_want_full_labels_returns_full_arrays(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "want_full_labels": True})
    assert r.status_code == 200
    j = r.json()
    assert j["full_class_ids"] is not None
    assert j["full_instance_ids"] is not None
    assert j["full_positions"] is not None
    assert j["full_n"] is not None
    assert isinstance(j["segment_summary"], dict)


def test_load_without_flag_omits_full_arrays(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id})
    j = r.json()
    assert j.get("full_class_ids") is None
    assert j.get("full_positions") is None


def test_subsample_idx_absent_when_no_subsampling(client_with_annotated_scene):
    """Cloud fits in max_points → no subsampling → subsample_idx is null."""
    client, scene_id, session_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "max_points": 100})
    assert r.status_code == 200
    j = r.json()
    assert j.get("subsample_idx") is None


def test_subsample_idx_present_and_correct_when_subsampling(tmp_path, monkeypatch):
    """Cloud exceeds max_points → subsample_idx maps sub rows → full indices."""
    import main
    from fastapi.testclient import TestClient

    lidar = tmp_path / "lidar-sub"
    scan_dir = lidar / "annotated" / "big"
    n = 20
    rng = np.random.default_rng(1)
    pts = rng.standard_normal((n, 3)).astype(np.float32)
    from plyfile import PlyData, PlyElement
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    arr = np.zeros(n, dtype=dtype)
    arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr['red'] = arr['green'] = arr['blue'] = 200
    (scan_dir / "source").mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(
        str(scan_dir / "source" / "scan.ply"))
    (scan_dir / "meta.json").write_text(
        '{"scan_name": "big", "n_points": 20, "schema_version": "2.0",'
        ' "source_mesh": "mesh.glb"}')

    monkeypatch.setattr("app.constants.LIDAR_ROOT", lidar, raising=False)
    client = TestClient(main.app)

    max_pts = 5
    r = client.post("/api/load", json={"name": "annotated/big", "max_points": max_pts})
    assert r.status_code == 200
    j = r.json()
    assert j["num_points"] == n
    assert j["num_subsampled"] == max_pts
    assert j["subsample_idx"] is not None

    idx_bytes = base64.b64decode(j["subsample_idx"])
    idx_arr = np.frombuffer(idx_bytes, dtype=np.int32)
    assert len(idx_arr) == max_pts
    # All indices must be valid full-res indices, sorted, and unique.
    assert idx_arr.min() >= 0
    assert idx_arr.max() < n
    assert len(np.unique(idx_arr)) == max_pts
    # Sorted because _safe_subsample calls idx.sort().
    assert (np.diff(idx_arr) > 0).all()


def test_load_recovers_in_progress_session_after_server_restart(
    client_with_annotated_scene,
):
    """First load -> mutate -> simulate server restart -> second load recovers
    the working_*.npy state via session.json commit-pointer."""
    client, scene_id, session_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "max_points": 100,
                                       "want_full_labels": True})
    assert r.status_code == 200, r.text

    # Brush a class onto a few points via the apply/reassign endpoint.
    import main
    seg_before = main._state["seg"]
    assert seg_before is not None
    idx_b64 = base64.b64encode(np.array([0, 1, 2], dtype=np.int32).tobytes()).decode()
    r = client.post("/api/segment/apply", json={
        "op": "reassign",
        "indices": idx_b64,
        "payload": {"target_inst": -1, "target_class": 2},
    })
    assert r.status_code == 200, r.text

    # Force-drain autosave debounce so working_* + session.json are on disk.
    seg_before.flush_autosave()
    assert seg_before.session_dir is not None
    assert (seg_before.session_dir / "session.json").exists()
    assert (seg_before.session_dir / "working_class_ids.npy").exists()

    # Force-clear in-memory state to simulate a server restart.
    main._state.update(scene=None, pc=None, seg=None)

    # Reload same scene — recovery path should kick in.
    r = client.post("/api/load", json={"name": scene_id, "max_points": 100,
                                       "want_full_labels": True})
    assert r.status_code == 200, r.text

    # The state endpoint reports the recovered dirty session.
    r = client.get("/api/segment/state")
    body = r.json()
    assert body["has_seg"] is True
    assert body["dirty"] is True


def _wall_ply(path, rgb, R):
    # Wall at z=-5 facing a camera at the origin looking down -z. Store it
    # PRE-rotated by R so that the gate's default orientation="Z+" (xyz @ R.T)
    # cancels back to this wall — otherwise Z+ rotates the plane out of view
    # and a correctly-registered scene would score coverage~0 and false-409.
    g = np.linspace(-2, 2, 40)
    xx, yy = np.meshgrid(g, g)
    wall = np.stack([xx.ravel(), yy.ravel(), -5 * np.ones(xx.size)], -1).astype(np.float64)
    stored = (wall @ R).astype(np.float32)            # stored @ R.T == wall
    arr = np.zeros(len(stored), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                       ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    arr['x'], arr['y'], arr['z'] = stored[:, 0], stored[:, 1], stored[:, 2]
    arr['red'], arr['green'], arr['blue'] = rgb
    path.parent.mkdir(parents=True, exist_ok=True)
    from plyfile import PlyData, PlyElement
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(str(path))
    return np.asarray(stored, dtype=np.float64)


def _build_render_scene(lidar, name, *, cloud_rgb, img_rgb):
    import json
    from scenes.reproject import ORIENTATION_PRESETS, euler_xyz_matrix
    R = euler_xyz_matrix(*ORIENTATION_PRESETS["Z+"])   # the gate's production default orientation
    scan = lidar / "annotated" / name
    pts = _wall_ply(scan / "source" / "scan.ply", cloud_rgb, R)
    fp = cloud_fingerprint(pts)
    scan.joinpath("meta.json").write_text(json.dumps({
        "scan_name": name, "n_points": len(pts), "units": "meters", "schema_version": "2.0",
        "frame": {"canonical_id": f"{name}#local", "transform_to_canonical": np.eye(4).tolist(),
                  "units": "meters", "frame_uncertain": False},
        "derivation": {"scan_id": name, "variant_id": "v1", "parent": "original", "op": "asis",
                       "varies": [], "source_fingerprint": fp, "role": "labeling"},
    }))
    run = scan / "renders" / "run0"; run.mkdir(parents=True)
    frames = [{"file": "f0.png", "position": [0, 0, 0], "target": [0, 0, -1]}]
    Image.fromarray(np.full((120, 120, 3), img_rgb, np.uint8)).save(run / "f0.png")
    (run / "manifest.json").write_text(json.dumps({"frames": frames}))
    write_render_meta(run, run_id="run0",
                      generated_from={"scan_id": name, "variant_id": "v1", "source_fingerprint": fp},
                      frame=Frame(np.eye(4), f"{name}#local"),
                      intrinsics={"fov_deg": 60, "fov_axis": "vertical", "width": 120, "height": 120})


def _client_for(lidar, monkeypatch):
    # Mirror the existing lidar_client fixture: patch app.constants.LIDAR_ROOT in
    # place (read live by _resolve via `constants.LIDAR_ROOT`). Do NOT reload main
    # — that breaks pydantic model identity and the env var wouldn't re-read anyway.
    from fastapi.testclient import TestClient
    import main
    monkeypatch.setattr("app.constants.LIDAR_ROOT", lidar, raising=False)
    return TestClient(main.app)


def test_load_blocks_on_registration_failure(tmp_path, monkeypatch):
    import preseg.registration as reg
    reg._VERDICT_CACHE.clear()
    lidar = tmp_path / "lidar"
    _build_render_scene(lidar, "bad", cloud_rgb=(200, 50, 50), img_rgb=(50, 50, 200))
    client = _client_for(lidar, monkeypatch)
    r = client.post("/api/load", json={"name": "annotated/bad"})
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "frame_registration_failed"


def test_load_passes_and_surfaces_frame_check(tmp_path, monkeypatch):
    import preseg.registration as reg
    reg._VERDICT_CACHE.clear()
    lidar = tmp_path / "lidar"
    _build_render_scene(lidar, "good", cloud_rgb=(200, 50, 50), img_rgb=(200, 50, 50))
    client = _client_for(lidar, monkeypatch)
    r = client.post("/api/load", json={"name": "annotated/good"})
    assert r.status_code == 200
    assert r.json()["frame_check"]["ok"] is True
