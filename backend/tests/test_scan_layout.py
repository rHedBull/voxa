"""ScanLayout pins the documented scan-schema v1.3 directory contract."""
from pathlib import Path

from scenes.scan_layout import ScanLayout


SCAN = Path("/lidar/annotated/munich_water_pump")


def test_paths_match_documented_schema():
    lay = ScanLayout(SCAN)
    assert lay.source_dir == SCAN / "source"
    assert lay.scan_ply == SCAN / "source" / "scan.ply"
    assert lay.mesh_glb == SCAN / "source" / "mesh.glb"
    assert lay.labels_dir == SCAN / "labels"
    assert lay.gt_class_ids == SCAN / "labels" / "gt_class_ids.npy"
    assert lay.gt_segment_ids == SCAN / "labels" / "gt_segment_ids.npy"
    assert lay.gt_segment_metadata == SCAN / "labels" / "gt_segment_metadata.json"
    assert lay.prelabel_dir == SCAN / "prelabel"
    assert lay.ransac_instance_ids == SCAN / "prelabel" / "ransac_instance_ids.npy"
    assert lay.ransac_segment_summary == SCAN / "prelabel" / "ransac_segment_summary.json"
    assert lay.session_dir == SCAN / "session"
    assert lay.renders_dir == SCAN / "renders"
    assert lay.sam3_dir == SCAN / "sam3"
    assert lay.annotation_history_dir == SCAN / "annotation_history"
    assert lay.meta_json == SCAN / "meta.json"


def test_classes_json_is_archive_level():
    # <lidar_root>/annotated/<scan>/ -> <lidar_root>/classes.json
    assert ScanLayout(SCAN).classes_json == Path("/lidar/classes.json")
