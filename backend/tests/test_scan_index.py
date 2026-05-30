import json
import sys
import pathlib

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "scan"))
from scan_index import build_variants_index  # noqa: E402

from scenes.frame import Frame  # noqa: E402
from scenes.render_meta import write_render_meta  # noqa: E402


def _scan(tmp_path):
    (tmp_path / "source").mkdir(parents=True)
    (tmp_path / "meta.json").write_text(json.dumps({
        "schema_version": "1.3", "scan_name": tmp_path.name,
        "frame": {"canonical_id": f"{tmp_path.name}#local",
                  "transform_to_canonical": np.eye(4).tolist()},
        "derivation": {"scan_id": tmp_path.name, "variant_id": "labelcloud",
                       "varies": ["density"], "role": "labeling",
                       "source_fingerprint": "sha256:aaa"},
    }))
    run = tmp_path / "renders" / "r1"; run.mkdir(parents=True)
    (run / "manifest.json").write_text("{}")
    M = np.eye(4); M[:3, 3] = [10, 0, 0]
    write_render_meta(run, run_id="r1",
                      generated_from={"scan_id": tmp_path.name, "variant_id": "aligned15M",
                                      "source_fingerprint": "sha256:bbb", "n_points": 11820483},
                      frame=Frame(M, f"{tmp_path.name}#local"),
                      intrinsics={"fov_deg": 60, "width": 10, "height": 10})
    return tmp_path


def test_index_includes_label_and_render_variants(tmp_path):
    scan = _scan(tmp_path)
    idx = build_variants_index(scan)
    assert idx["scan_id"] == scan.name
    assert idx["labeling_variant"] == "labelcloud"
    by_id = {v["variant_id"]: v for v in idx["variants"]}
    assert set(by_id) == {"labelcloud", "aligned15M"}
    assert by_id["labelcloud"]["source_fingerprint"] == "sha256:aaa"
    assert by_id["aligned15M"]["source_fingerprint"] == "sha256:bbb"
    # render variant carries the recorded transform_to_canonical
    assert np.allclose(np.asarray(by_id["aligned15M"]["transform_to_canonical"])[:3, 3], [10, 0, 0])


def test_index_dedups_repeated_render_variant(tmp_path):
    scan = _scan(tmp_path)
    run2 = scan / "renders" / "r2"; run2.mkdir()
    (run2 / "manifest.json").write_text("{}")
    write_render_meta(run2, run_id="r2",
                      generated_from={"scan_id": scan.name, "variant_id": "aligned15M",
                                      "source_fingerprint": "sha256:bbb"},
                      frame=Frame(np.eye(4), f"{scan.name}#local"),
                      intrinsics={"fov_deg": 60, "width": 10, "height": 10})
    idx = build_variants_index(scan)
    assert sum(v["variant_id"] == "aligned15M" for v in idx["variants"]) == 1
