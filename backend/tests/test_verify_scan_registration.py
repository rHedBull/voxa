import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from plyfile import PlyData, PlyElement

from preseg.registration import verify_scan_registration
from scenes.fingerprint import cloud_fingerprint
from scenes.frame import Frame
from scenes.render_meta import write_render_meta


# A wall at z=-5, in front of a camera at the origin looking down -z.
def _wall(n=40):
    g = np.linspace(-2, 2, n)
    xx, yy = np.meshgrid(g, g)
    return np.stack([xx.ravel(), yy.ravel(), -5 * np.ones(xx.size)], -1).astype(np.float32)


def _write_ply(path: Path, pts: np.ndarray, rgb: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.zeros(len(pts), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr['red'], arr['green'], arr['blue'] = rgb
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(str(path))


def build_scan(tmp_path: Path, *, cloud_rgb=(200, 50, 50), img_rgb=(200, 50, 50),
               n_frames=2, write_images=True, render_canonical="demo#local",
               W=120, H=120) -> Path:
    """A v1.3 scan dir whose render run uses the SAME variant+fingerprint as the
    cloud (so the resolver returns use_direct, transform = identity under the Y+
    identity orientation). Pixels are solid img_rgb; cloud colour is cloud_rgb —
    set them equal for a PASS, different for a photometric FAIL."""
    scan = tmp_path / "scan"
    pts = _wall()
    _write_ply(scan / "source" / "scan.ply", pts, cloud_rgb)
    fp = cloud_fingerprint(np.asarray(pts, dtype=np.float64))

    scan.joinpath("meta.json").write_text(json.dumps({
        "scan_name": "demo", "n_points": len(pts), "units": "meters",
        "schema_version": "1.3",
        "frame": {"canonical_id": "demo#local",
                  "transform_to_canonical": np.eye(4).tolist(),
                  "units": "meters", "frame_uncertain": False},
        "derivation": {"scan_id": "demo", "variant_id": "v1", "parent": "original",
                       "op": "asis", "varies": [],
                       "source_fingerprint": fp, "role": "labeling"},
    }))

    run = scan / "renders" / "run0"
    run.mkdir(parents=True)
    frames = []
    for i in range(n_frames):
        fname = f"frame_{i:03d}.png"
        frames.append({"file": fname, "position": [0, 0, 0], "target": [0, 0, -1]})
        if write_images:
            Image.fromarray(np.full((H, W, 3), img_rgb, np.uint8)).save(run / fname)
    (run / "manifest.json").write_text(json.dumps({"frames": frames}))
    write_render_meta(
        run, run_id="run0",
        generated_from={"scan_id": "demo", "variant_id": "v1", "source_fingerprint": fp},
        frame=Frame(np.eye(4), render_canonical),
        intrinsics={"fov_deg": 60, "fov_axis": "vertical", "width": W, "height": H},
    )
    return scan


def test_pass_when_cloud_matches_renders(tmp_path):
    scan = build_scan(tmp_path, cloud_rgb=(200, 50, 50), img_rgb=(200, 50, 50))
    v = verify_scan_registration(scan, orientation="Y+", use_cache=False)
    assert v["checked"] is True and v["ok"] is True
    assert v["runs"] and v["runs"][0]["run_id"] == "run0"
    assert v["runs"][0]["photometric"] is not None and v["runs"][0]["photometric"] > 0.9


def test_fail_when_photometric_mismatch(tmp_path):
    scan = build_scan(tmp_path, cloud_rgb=(200, 50, 50), img_rgb=(50, 50, 200))
    v = verify_scan_registration(scan, orientation="Y+", use_cache=False)
    assert v["checked"] is True and v["ok"] is False and v["reasons"]


def test_skip_when_no_renders(tmp_path):
    scan = build_scan(tmp_path)
    import shutil
    shutil.rmtree(scan / "renders")
    v = verify_scan_registration(scan, orientation="Y+", use_cache=False)
    assert v["checked"] is False and v["ok"] is True


def test_skip_when_no_images_on_disk(tmp_path):
    scan = build_scan(tmp_path, write_images=False)
    v = verify_scan_registration(scan, orientation="Y+", use_cache=False)
    assert v["checked"] is False and v["ok"] is True   # cannot verify => must not block


def test_hard_fail_when_render_is_cross_scan(tmp_path):
    scan = build_scan(tmp_path, render_canonical="other#local")
    v = verify_scan_registration(scan, orientation="Y+", use_cache=False)
    assert v["checked"] is True and v["ok"] is False and v["reasons"]
