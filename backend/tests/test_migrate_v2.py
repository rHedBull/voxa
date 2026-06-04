"""Tests for scripts/migrate_scan_v2.py — one-shot v1.3 → v2 migration."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
from plyfile import PlyData, PlyElement

from tests.conftest import write_scene_ply

# backend/ is already on sys.path via conftest.py (VOXA_DATA_DIR import triggers conftest).
# We locate the worktree root (backend/tests/ = parents[0], backend/ = parents[1],
# worktree root = parents[2]) so we can pass cwd= to subprocess.
_WORKTREE_ROOT = Path(__file__).resolve().parents[2]
_SCRIPT = _WORKTREE_ROOT / "scripts" / "migrate_scan_v2.py"



def build_v13_root(tmp_path: Path, *, with_session: bool = True) -> Path:
    """Build a v1.3 lidar root under tmp_path with one scan named 'demo'.

    Adapted from the v1.3 block in git show 525b587:backend/tests/conftest.py.
    Uses source_mesh (no source_laz) so the scan is Y-up — no z-up rotation
    applied to fingerprint, keeping test assertions simple.
    """
    root = tmp_path / "lidar"
    scan_dir = root / "annotated" / "demo"

    # source/scan.ply
    write_scene_ply(scan_dir / "source" / "scan.ply", n=8)

    # labels/
    (scan_dir / "labels").mkdir(parents=True, exist_ok=True)
    gt_cls = np.array([-1, 0, 0, 1, 1, 2, -1, 3], dtype=np.int32)
    gt_seg = np.array([-1, 0, 0, 1, 1, 2, -1, 3], dtype=np.int32)
    np.save(scan_dir / "labels" / "gt_class_ids.npy", gt_cls)
    np.save(scan_dir / "labels" / "gt_segment_ids.npy", gt_seg)
    (scan_dir / "labels" / "gt_segment_metadata.json").write_text(json.dumps({
        "n_points": 8, "n_gt_segments": 4, "n_labeled_points": 6,
        "class_map": {"pipe": 0, "tank": 1, "equipment": 2},
        "segments": [],
    }))

    # prelabel/ (flat v1.3 layout)
    (scan_dir / "prelabel").mkdir(parents=True, exist_ok=True)
    ransac_inst = np.array([-1, 0, 0, 1, 1, 2, -1, 3], dtype=np.int32)
    np.save(scan_dir / "prelabel" / "ransac_instance_ids.npy", ransac_inst)
    (scan_dir / "prelabel" / "ransac_segment_summary.json").write_text(json.dumps({
        "segments": [
            {"id": 0, "class_id": 0, "label": "pipe"},
            {"id": 1, "class_id": 1, "label": "tank"},
            {"id": 2, "class_id": 2, "label": "equipment"},
            {"id": 3, "class_id": 2, "label": "equipment"},
        ]
    }))

    # session/ (optional)
    if with_session:
        (scan_dir / "session").mkdir(parents=True, exist_ok=True)
        working_cls = np.array([-1, 0, 0, 1, 1, 2, -1, 3], dtype=np.int8)
        working_seg = np.array([-1, 0, 0, 1, 1, 2, -1, 3], dtype=np.int32)
        np.save(scan_dir / "session" / "working_class_ids.npy", working_cls)
        np.save(scan_dir / "session" / "working_segment_ids.npy", working_seg)
        (scan_dir / "session" / "current.json").write_text(json.dumps({
            "hidden_inst_ids": [5],
            "dirty": True,
            "preseg_id": "ransac",
        }))

    # annotation_history/
    history_snap = scan_dir / "annotation_history" / "20260101_000000"
    history_snap.mkdir(parents=True, exist_ok=True)
    (history_snap / "snapshot.json").write_text(json.dumps({"note": "test"}))

    # meta.json — source_mesh means Y-up (no z-up rotation)
    (scan_dir / "meta.json").write_text(json.dumps({
        "scan_name": "demo", "n_points": 8, "schema_version": "1.3",
        "source_mesh": "mesh.glb",
    }))

    # classes.json at lidar root
    (root / "classes.json").write_text(json.dumps({
        "version": 1, "unlabeled_id": -1,
        "classes": [{"id": 0, "name": "pipe"},
                    {"id": 1, "name": "tank"},
                    {"id": 2, "name": "equipment"}],
    }))

    return root


def _run_migrate(root: Path, *extra_args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(_SCRIPT), *extra_args, str(root)],
        capture_output=True, text=True,
        cwd=str(_WORKTREE_ROOT),
    )


# ---------------------------------------------------------------------------
# (a) full migration
# ---------------------------------------------------------------------------

def test_full_migrate(tmp_path):
    root = build_v13_root(tmp_path)
    scan_dir = root / "annotated" / "demo"

    result = _run_migrate(root)
    assert result.returncode == 0, result.stderr
    assert "MIGRATED demo" in result.stdout

    # meta schema_version updated
    meta = json.loads((scan_dir / "meta.json").read_text())
    assert meta["schema_version"] == "2.0"

    # GT files moved to sessions/legacy/output/
    output_dir = scan_dir / "sessions" / "legacy" / "output"
    assert (output_dir / "gt_class_ids.npy").exists()
    assert (output_dir / "gt_segment_ids.npy").exists()
    assert (output_dir / "gt_segment_metadata.json").exists()

    # GT values preserved
    gt_cls = np.load(output_dir / "gt_class_ids.npy")
    assert gt_cls.tolist() == [-1, 0, 0, 1, 1, 2, -1, 3]
    gt_seg = np.load(output_dir / "gt_segment_ids.npy")
    assert gt_seg.tolist() == [-1, 0, 0, 1, 1, 2, -1, 3]

    # Old flat prelabel files gone
    assert not (scan_dir / "prelabel" / "ransac_instance_ids.npy").exists()
    assert not (scan_dir / "prelabel" / "ransac_segment_summary.json").exists()

    # prelabel/ransac/ v2 subdir created with meta.json
    assert (scan_dir / "prelabel" / "ransac" / "meta.json").exists()
    preseg_meta = json.loads((scan_dir / "prelabel" / "ransac" / "meta.json").read_text())
    assert "fingerprint" in preseg_meta

    # session.json has correct pins
    from labeling.segment_io import compute_fingerprint
    inst = np.load(scan_dir / "prelabel" / "ransac" / "instance_ids.npy").astype(np.int32)
    expected_preseg_fp = compute_fingerprint(inst)
    assert preseg_meta["fingerprint"] == expected_preseg_fp

    session_json = json.loads((scan_dir / "sessions" / "legacy" / "session.json").read_text())
    assert session_json["preseg_id"] == "ransac"
    assert session_json["preseg_fingerprint"] == expected_preseg_fp
    assert session_json["name"] == "legacy"
    assert session_json["schema_version"] == 2

    # source_fingerprint matches what we'd compute (Y-up scan, no recenter for std_normal)
    ply = PlyData.read(str(scan_dir / "source" / "scan.ply"))["vertex"]
    pts = np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float32)
    expected_source_fp = compute_fingerprint(pts)  # Y-up, no rotation, no recenter
    assert session_json["source_fingerprint"] == expected_source_fp

    # No stale labels/, session/, annotation_history/ dirs
    assert not (scan_dir / "labels").exists()
    assert not (scan_dir / "session").exists()
    assert not (scan_dir / "annotation_history").exists()

    # History moved
    assert (scan_dir / "sessions" / "legacy" / "history" / "20260101_000000" / "snapshot.json").exists()


# ---------------------------------------------------------------------------
# (b) idempotent: second run exits 0, prints "already v2"
# ---------------------------------------------------------------------------

def test_idempotent(tmp_path):
    root = build_v13_root(tmp_path)
    scan_dir = root / "annotated" / "demo"

    r1 = _run_migrate(root)
    assert r1.returncode == 0, r1.stderr

    # Capture mtime of a key file before second run
    session_json_path = scan_dir / "sessions" / "legacy" / "session.json"
    mtime_before = session_json_path.stat().st_mtime

    r2 = _run_migrate(root)
    assert r2.returncode == 0, r2.stderr
    assert "already v2" in r2.stdout

    # File unchanged
    assert session_json_path.stat().st_mtime == mtime_before


# ---------------------------------------------------------------------------
# (c) labels-without-session: working arrays == GT (int8 cast)
# ---------------------------------------------------------------------------

def test_labels_without_session(tmp_path):
    root = build_v13_root(tmp_path, with_session=False)
    scan_dir = root / "annotated" / "demo"

    result = _run_migrate(root)
    assert result.returncode == 0, result.stderr

    sp = scan_dir / "sessions" / "legacy"
    working_cls = np.load(sp / "working_class_ids.npy")
    working_seg = np.load(sp / "working_segment_ids.npy")

    # Working class IDs should be int8 cast of GT
    expected_cls = np.array([-1, 0, 0, 1, 1, 2, -1, 3], dtype=np.int8)
    np.testing.assert_array_equal(working_cls, expected_cls)
    expected_seg = np.array([-1, 0, 0, 1, 1, 2, -1, 3], dtype=np.int32)
    np.testing.assert_array_equal(working_seg, expected_seg)


# ---------------------------------------------------------------------------
# (d) refusal: stray file in prelabel/ → exit 1, disk untouched
# ---------------------------------------------------------------------------

def test_refusal_existing_sessions_dir(tmp_path):
    """A v1.3-meta scan that already has sessions/ (e.g. a crashed prior
    migration) is refused, not guessed at."""
    root = build_v13_root(tmp_path)
    scan_dir = root / "annotated" / "demo"
    (scan_dir / "sessions").mkdir()

    result = _run_migrate(root)
    assert result.returncode == 1
    assert "REFUSED" in result.stdout
    assert "sessions/" in result.stdout
    assert (scan_dir / "labels").exists()  # untouched


def test_refusal_stray_prelabel_file(tmp_path):
    root = build_v13_root(tmp_path)
    scan_dir = root / "annotated" / "demo"

    # Add stray file
    (scan_dir / "prelabel" / "extra.txt").write_text("oops")

    result = _run_migrate(root)
    assert result.returncode == 1
    assert "REFUSED" in result.stdout
    assert "extra.txt" in result.stdout

    # Disk untouched: labels/ still present, no sessions/
    assert (scan_dir / "labels").exists()
    assert not (scan_dir / "sessions").exists()


# ---------------------------------------------------------------------------
# (e) --dry-run: exit 0, disk completely unchanged
# ---------------------------------------------------------------------------

def test_dry_run_no_changes(tmp_path):
    root = build_v13_root(tmp_path)
    scan_dir = root / "annotated" / "demo"

    # Snapshot current state
    def _list_files(base: Path) -> set[str]:
        return {str(p.relative_to(base)) for p in base.rglob("*") if p.is_file()}

    files_before = _list_files(root)

    result = _run_migrate(root, "--dry-run")
    assert result.returncode == 0, result.stderr
    assert "DRY-RUN" in result.stdout

    files_after = _list_files(root)
    assert files_before == files_after, (
        f"--dry-run changed disk:\n"
        f"  added: {files_after - files_before}\n"
        f"  removed: {files_before - files_after}"
    )


# ---------------------------------------------------------------------------
# (f) post-migration load: /api/load returns 200 with session_id == "legacy"
# ---------------------------------------------------------------------------

def test_post_migration_load(tmp_path, monkeypatch):
    """End-to-end: migrate then load via the API — the migrated scan must be
    loadable with session_id == 'legacy' and class_ids present."""
    root = build_v13_root(tmp_path)

    result = _run_migrate(root)
    assert result.returncode == 0, result.stderr

    import main
    from fastapi.testclient import TestClient

    monkeypatch.setattr("app.constants.LIDAR_ROOT", root, raising=False)
    client = TestClient(main.app)

    r = client.post("/api/load", json={"name": "annotated/demo", "max_points": 100})
    assert r.status_code == 200, r.text
    body = r.json()

    assert body["session_id"] == "legacy"
    assert body["class_ids"] is not None, "class_ids should be present after migration"
    assert body["num_points"] == 8


def test_post_migration_load_zup_utm_scan(tmp_path, monkeypatch):
    """Regression: a Z-up scan at UTM-scale coordinates exercises BOTH the
    z-up→y-up rotation AND the recenter rule in the migration's
    source-fingerprint replication. Any divergence from the loader's
    transforms (core._z_up_to_y_up / core._recenter) makes the pin mismatch
    and this load come back 409 instead of 200."""
    root = tmp_path / "lidar"
    scan_dir = root / "annotated" / "utm"

    rng = np.random.default_rng(1)
    pts = rng.standard_normal((8, 3)).astype(np.float32)
    pts[:, 0] += 700_000.0   # UTM easting
    pts[:, 1] += 5_300_000.0  # UTM northing
    pts[:, 2] += 400.0        # elevation
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    arr = np.zeros(8, dtype=dtype)
    arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr['red'] = arr['green'] = arr['blue'] = 200
    (scan_dir / "source").mkdir(parents=True)
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(
        str(scan_dir / "source" / "scan.ply"))

    (scan_dir / "labels").mkdir()
    gt = np.array([-1, 0, 0, 1, 1, 2, -1, 3], dtype=np.int32)
    np.save(scan_dir / "labels" / "gt_class_ids.npy", gt)
    np.save(scan_dir / "labels" / "gt_segment_ids.npy", gt)
    (scan_dir / "labels" / "gt_segment_metadata.json").write_text(
        json.dumps({"n_points": 8, "segments": []}))
    # source_laz (and no source_mesh) → the loader treats the scan as Z-up.
    (scan_dir / "meta.json").write_text(json.dumps({
        "scan_name": "utm", "n_points": 8, "schema_version": "1.3",
        "source_laz": "lidar/laz/utm.laz",
    }))
    (root / "classes.json").write_text(json.dumps({
        "version": 1, "unlabeled_id": -1,
        "classes": [{"id": 0, "name": "pipe"}, {"id": 1, "name": "tank"},
                    {"id": 2, "name": "equipment"}],
    }))

    result = _run_migrate(root)
    assert result.returncode == 0, result.stdout + result.stderr

    import main
    from fastapi.testclient import TestClient
    monkeypatch.setattr("app.constants.LIDAR_ROOT", root, raising=False)
    client = TestClient(main.app)
    r = client.post("/api/load", json={"name": "annotated/utm", "max_points": 100})
    assert r.status_code == 200, r.text
    assert r.json()["session_id"] == "legacy"
