import numpy as np

from preseg.resolver import resolve_render_run
from scenes.frame import Frame


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
