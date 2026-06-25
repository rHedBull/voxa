"""Voxa-side scan_meta tests: frame_summary (UI summary). The canonical reader
read_scan_meta is tested in the scan_schema package (test_read_scan_meta.py)."""
import json

import numpy as np


def _write(scan, meta):
    (scan / "meta.json").write_text(json.dumps(meta))


def test_frame_summary_v13(tmp_path):
    from scenes.scan_meta import frame_summary
    _write(tmp_path, {"schema_version": "1.3",
                      "frame": {"canonical_id": "s#local",
                                "transform_to_canonical": np.eye(4).tolist(),
                                "georef": {"offset_m": [1.0, 2.0, 3.0]}},
                      "derivation": {"scan_id": "s", "variant_id": "v", "varies": ["density"]}})
    s = frame_summary(tmp_path)
    assert s["schema_version"] == "1.3" and s["variant_id"] == "v"
    assert s["frame_canonical_id"] == "s#local" and s["frame_uncertain"] is False
    assert s["georef_offset"] == [1.0, 2.0, 3.0]


def test_frame_summary_legacy_uncertain(tmp_path):
    from scenes.scan_meta import frame_summary
    _write(tmp_path, {"scan_name": "navvis", "coords": "world_minus_offset",
                      "coord_offset_m": [5.0, 6.0, 7.0]})
    s = frame_summary(tmp_path)
    assert s["frame_uncertain"] is True and s["georef_offset"] == [5.0, 6.0, 7.0]


def test_frame_summary_no_meta(tmp_path):
    from scenes.scan_meta import frame_summary
    assert frame_summary(tmp_path) == {}
