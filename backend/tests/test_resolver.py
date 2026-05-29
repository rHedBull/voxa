import numpy as np
import pytest

from preseg.resolver import dir_cloud_transforms, resolve_render_run
from scenes.frame import Frame
from scenes.render_meta import write_render_meta


def _runmeta(variant_id, fp, T, canonical="navvis#local", scan="navvis"):
    return {"generated_from": {"variant_id": variant_id, "scan_id": scan,
                               "source_fingerprint": fp},
            "frame": Frame(T, canonical)}


def test_direct_when_variant_and_fp_match():
    cf = Frame(np.eye(4), "navvis#local")
    r = resolve_render_run(cf, "v1", "sha256:a", _runmeta("v1", "sha256:a", np.eye(4)))
    assert r.action == "use_direct" and np.allclose(r.transform, np.eye(4))


def test_remap_when_same_scan_diff_frame():
    cf = Frame(np.eye(4), "navvis#local")            # cloud at canonical
    T = np.eye(4); T[:3, 3] = [20, 2, -8]            # render frame offset from canonical
    r = resolve_render_run(cf, "v_unaligned", "sha256:b", _runmeta("aligned15M", "sha256:c", T))
    assert r.action == "remap" and np.allclose(r.transform, T)  # poses -> cloud frame


def test_fail_when_different_scan():
    cf = Frame(np.eye(4), "navvis#local")
    run = _runmeta("v1", "sha256:a", np.eye(4), canonical="OTHER#local", scan="other")
    r = resolve_render_run(cf, "v1", "sha256:a", run)
    assert r.action == "fail" and r.reasons


def test_fail_when_pin_missing():
    cf = Frame(np.eye(4), "navvis#local")
    r = resolve_render_run(cf, "v1", "sha256:a", {"frame": cf, "generated_from": {}})
    assert r.action == "fail"


def _write_run(tmp_path, name, variant_id, fp, T, canonical="navvis#local"):
    run = tmp_path / name
    run.mkdir()
    write_render_meta(run, run_id=name,
                      generated_from={"scan_id": "navvis", "variant_id": variant_id,
                                      "source_fingerprint": fp},
                      frame=Frame(T, canonical),
                      intrinsics={"fov_deg": 60, "width": 10, "height": 10})
    return run


def test_dir_transforms_identity_for_use_direct(tmp_path):
    cf = Frame(np.eye(4), "navvis#local")
    run = _write_run(tmp_path, "r", "v1", "sha256:a", np.eye(4))
    T = dir_cloud_transforms([run], cf, "v1", "sha256:a", np.eye(3))
    assert np.allclose(T[run], np.eye(4))


def test_dir_transforms_remap_translation(tmp_path):
    cf = Frame(np.eye(4), "navvis#local")                # cloud at canonical
    M = np.eye(4); M[:3, 3] = [5, 0, 0]                   # run frame: render->canonical +5x
    run = _write_run(tmp_path, "r", "aligned", "sha256:z", M)
    T = dir_cloud_transforms([run], cf, "v1", "sha256:a", np.eye(3))  # orientation identity
    assert np.allclose(T[run][:3, 3], [-5, 0, 0])        # bring cloud into render-native -> -5x


def test_dir_transforms_none_when_no_meta(tmp_path):
    run = tmp_path / "bare"; run.mkdir()
    cf = Frame(np.eye(4), "navvis#local")
    assert dir_cloud_transforms([run], cf, "v1", "sha256:a", np.eye(3))[run] is None


def test_dir_transforms_raises_on_fail(tmp_path):
    cf = Frame(np.eye(4), "navvis#local")
    run = _write_run(tmp_path, "r", "v1", "sha256:a", np.eye(4), canonical="OTHER#local")
    with pytest.raises(ValueError):
        dir_cloud_transforms([run], cf, "v1", "sha256:a", np.eye(3))
