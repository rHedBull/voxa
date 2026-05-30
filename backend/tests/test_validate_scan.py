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


def _good_meta():
    return {"schema_version": "1.3", "scan_name": "s",
            "frame": {"canonical_id": "s#local",
                      "transform_to_canonical": np.eye(4).tolist(), "units": "meters"},
            "derivation": {"scan_id": "s", "variant_id": "v", "varies": ["density"], "role": None}}


def test_good_scan_no_violations(tmp_path):
    assert validate_scan_dir(_scan(tmp_path, _good_meta())) == []


def test_bad_transform_flagged(tmp_path):
    m = _good_meta()
    m["frame"]["transform_to_canonical"] = [[1, 0], [0, 1]]
    v = validate_scan_dir(_scan(tmp_path, m))
    assert any("transform" in x for x in v)


def test_bad_varies_flagged(tmp_path):
    m = _good_meta()
    m["derivation"]["varies"] = ["density", "bogus"]
    v = validate_scan_dir(_scan(tmp_path, m))
    assert any("varies" in x for x in v)


def test_render_run_missing_meta_flagged(tmp_path):
    scan = _scan(tmp_path, _good_meta())
    run = scan / "renders" / "lower"
    run.mkdir(parents=True)
    (run / "manifest.json").write_text("{}")  # run dir but no meta.json
    v = validate_scan_dir(scan)
    assert any("lower" in x and "meta.json" in x for x in v)
