"""ScanLayout pins the documented scan-schema v2 directory contract."""
from pathlib import Path

from scenes.scan_layout import ScanLayout


SCAN = Path("/lidar/annotated/munich_water_pump")


def test_paths_match_documented_schema():
    lay = ScanLayout(SCAN)
    assert lay.source_dir == SCAN / "source"
    assert lay.scan_ply == SCAN / "source" / "scan.ply"
    assert lay.mesh_glb == SCAN / "source" / "mesh.glb"
    assert lay.renders_dir == SCAN / "renders"
    assert lay.sam3_dir == SCAN / "sam3"
    assert lay.meta_json == SCAN / "meta.json"


def test_classes_json_is_archive_level():
    # <lidar_root>/annotated/<scan>/ -> <lidar_root>/classes.json
    assert ScanLayout(SCAN).classes_json == Path("/lidar/classes.json")


def test_v2_preseg_paths(tmp_path):
    lay = ScanLayout(tmp_path / "annotated" / "demo")
    d = lay.preseg_dir("ransac")
    assert d == lay.presegs_root / "ransac"
    assert lay.presegs_root == lay.scan_dir / "prelabel"


def test_v2_session_paths(tmp_path):
    lay = ScanLayout(tmp_path / "annotated" / "demo")
    s = lay.session("20260603-120000_ransac")
    assert lay.sessions_root == lay.scan_dir / "sessions"
    assert s.dir == lay.sessions_root / "20260603-120000_ransac"
    assert s.session_json == s.dir / "session.json"
    assert s.working_class_ids == s.dir / "working_class_ids.npy"
    assert s.working_segment_ids == s.dir / "working_segment_ids.npy"
    assert s.output_dir == s.dir / "output"
    assert s.output_gt_class_ids == s.output_dir / "gt_class_ids.npy"
    assert s.output_gt_segment_ids == s.output_dir / "gt_segment_ids.npy"
    assert s.output_gt_segment_metadata == s.output_dir / "gt_segment_metadata.json"
    assert s.history_dir == s.dir / "history"
