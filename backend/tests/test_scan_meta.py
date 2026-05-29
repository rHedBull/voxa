import json

import numpy as np

from scenes.scan_meta import read_scan_meta


def _write(scan, meta):
    (scan / "meta.json").write_text(json.dumps(meta))


def test_v13_frame_used_directly(tmp_path):
    _write(tmp_path, {"schema_version": "1.3",
                      "frame": {"canonical_id": "s#local",
                                "transform_to_canonical": np.eye(4).tolist(),
                                "units": "meters"},
                      "derivation": {"scan_id": "s", "variant_id": "v", "varies": ["density"]}})
    m = read_scan_meta(tmp_path)
    assert m["frame"].canonical_id == "s#local" and not m["frame"].frame_uncertain
    assert m["derivation"]["variant_id"] == "v"


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


def test_legacy_coords_synthesizes_uncertain_frame(tmp_path):
    _write(tmp_path, {"scan_name": "navvis", "coords": "world_minus_offset",
                      "coord_offset_m": [574184.0, 6220868.0, 49.0], "class_map_version": 1})
    m = read_scan_meta(tmp_path)
    f = m["frame"]
    assert f.frame_uncertain                                   # forces the §6 check
    assert np.allclose(f.transform_to_canonical, np.eye(4))    # stored = its own canonical-local
    assert f.georef["offset_m"] == [574184.0, 6220868.0, 49.0]
    assert m["derivation"]["scan_id"] == "navvis"              # derived from scan_name
