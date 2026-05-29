import numpy as np

from scenes.frame import Frame
from scenes.render_meta import read_render_meta, write_render_meta


def test_write_then_read_roundtrip(tmp_path):
    run = tmp_path / "lower"
    run.mkdir()
    frame = Frame(np.eye(4), "navvis#local")
    write_render_meta(
        run, run_id="lower",
        generated_from={"scan_id": "navvis", "variant_id": "aligned15M",
                        "source_fingerprint": "sha256:abc", "n_points": 11820483},
        frame=frame,
        intrinsics={"fov_deg": 60, "fov_axis": "vertical", "aspect": 1.169,
                    "width": 926, "height": 792},
    )
    m = read_render_meta(run)
    assert m["run_id"] == "lower"
    assert m["generated_from"]["variant_id"] == "aligned15M"
    assert m["intrinsics"]["fov_deg"] == 60
    assert isinstance(m["frame"], Frame)
    assert np.allclose(m["frame"].transform_to_canonical, np.eye(4))


def test_read_missing_returns_none(tmp_path):
    assert read_render_meta(tmp_path / "nope") is None
