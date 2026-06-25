import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "scan"))
from validate_scan import validate_scan_dir  # noqa: E402


def _scan(tmp_path, meta):
    (tmp_path / "source").mkdir(parents=True, exist_ok=True)
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return tmp_path


def _good_meta(version="3.0"):
    return {"schema_version": version, "scan_name": "s",
            "n_points": 8, "units": "meters", "class_map_version": 1,
            "frame": {"canonical_id": "s#local",
                      "transform_to_canonical": np.eye(4).tolist(), "units": "meters"},
            "derivation": {"scan_id": "s", "variant_id": "v", "varies": ["density"], "role": None}}


def test_good_scan_no_violations(tmp_path):
    assert validate_scan_dir(_scan(tmp_path, _good_meta())) == []


def test_bad_transform_flagged_on_3x(tmp_path):
    m = _good_meta("3.0")
    m["frame"]["transform_to_canonical"] = [[1, 0], [0, 1]]
    v = validate_scan_dir(_scan(tmp_path, m))
    assert any("transform" in x for x in v)


def test_bad_varies_flagged_on_3x(tmp_path):
    m = _good_meta("3.0")
    m["derivation"]["varies"] = ["density", "bogus"]
    v = validate_scan_dir(_scan(tmp_path, m))
    assert any("varies" in x for x in v)


def test_2x_grandfathers_missing_frame_and_derivation(tmp_path):
    # 2.x scans: frame/derivation issues are warnings, not errors → no violations.
    m = {"schema_version": "2.0", "scan_name": "s",
         "n_points": 8, "units": "meters", "class_map_version": 1}
    assert validate_scan_dir(_scan(tmp_path, m)) == []


def test_missing_required_field_flagged(tmp_path):
    # required-always fields (here: units) are errors at any version.
    m = _good_meta("2.0")
    del m["units"]
    v = validate_scan_dir(_scan(tmp_path, m))
    assert any("units" in x for x in v)


def test_unsupported_version_flagged(tmp_path):
    v = validate_scan_dir(_scan(tmp_path, _good_meta("1.3")))
    assert any("schema_version" in x for x in v)


def test_render_run_missing_meta_flagged(tmp_path):
    scan = _scan(tmp_path, _good_meta())
    run = scan / "renders" / "lower"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text("{}")  # run dir but no meta.json
    v = validate_scan_dir(scan)
    assert any("lower" in x and "meta.json" in x for x in v)
