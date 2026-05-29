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


def test_legacy_coords_synthesizes_uncertain_frame(tmp_path):
    _write(tmp_path, {"scan_name": "navvis", "coords": "world_minus_offset",
                      "coord_offset_m": [574184.0, 6220868.0, 49.0], "class_map_version": 1})
    m = read_scan_meta(tmp_path)
    f = m["frame"]
    assert f.frame_uncertain                                   # forces the §6 check
    assert np.allclose(f.transform_to_canonical, np.eye(4))    # stored = its own canonical-local
    assert f.georef["offset_m"] == [574184.0, 6220868.0, 49.0]
    assert m["derivation"]["scan_id"] == "navvis"              # derived from scan_name
