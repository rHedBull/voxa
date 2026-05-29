# §6 Registration Auto-Check at Load — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `/api/load` run the scan-schema v1.3 §6 registration health-check on every annotated scan that has renders, hard-blocking (HTTP 409) when the cloud doesn't register to its render poses.

**Architecture:** Extract a reusable `verify_scan_registration(scan_dir, …)` into `backend/preseg/registration.py` (loads the cloud, applies orientation + recorded v1.3 remap, samples frames, scores with the existing photometric-primary `registration_score`/`check_registration`). The `/api/load` route calls it and translates the verdict to 200+`frame_check` or 409. The CLI `scripts/verify_registration.py` is refactored to a thin wrapper over the same function. A content-keyed in-process cache avoids recompute.

**Tech Stack:** Python, FastAPI, numpy, Pillow, plyfile, pytest. Spec: `docs/superpowers/specs/2026-05-29-frame-registration-load-check-design.md`.

---

## File structure

| File | Responsibility | Change |
|------|----------------|--------|
| `backend/preseg/registration.py` | scoring (exists) + new whole-scan verdict + verdict cache | Modify |
| `scripts/verify_registration.py` | human CLI: call the function, print table, exit code | Rewrite |
| `backend/app/schemas.py` | `LoadResponse.frame_check` field | Modify |
| `backend/routes/load.py` | call the check; 409 on fail; attach `frame_check` | Modify |
| `backend/tests/test_verify_scan_registration.py` | unit tests for the function + cache | Create |
| `backend/tests/test_load_endpoint.py` | integration: 200+frame_check / 409 | Modify |

Run all backend commands from the worktree root with `.venv/bin/python`/`.venv/bin/pytest`. If `.venv` is missing, create it: `./scripts/test.sh` bootstraps it (installs `requirements-dev.txt`).

---

### Task 1: `verify_scan_registration` — happy path + mismatch

**Files:**
- Modify: `backend/preseg/registration.py`
- Test: `backend/tests/test_verify_scan_registration.py`

- [ ] **Step 1: Write the failing test**

Create `backend/tests/test_verify_scan_registration.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest backend/tests/test_verify_scan_registration.py -v`
Expected: FAIL — `ImportError: cannot import name 'verify_scan_registration'`.

- [ ] **Step 3: Write minimal implementation**

Append to `backend/preseg/registration.py` (top of file, add `import json`, `import math`, `from pathlib import Path`):

```python
def _fov_y_from_intrinsics(intr: dict, W: int, H: int) -> float:
    """Vertical FOV (deg) for registration_score, honouring intrinsics.fov_axis.
    Horizontal FOV is converted to vertical via the render aspect; missing/vertical
    is used as-is. Default 60.0 when intrinsics absent."""
    intr = intr or {}
    fov = float(intr.get("fov_deg", 60.0))
    if intr.get("fov_axis") == "horizontal" and W and H:
        fov_x = math.radians(fov)
        return math.degrees(2.0 * math.atan(math.tan(fov_x / 2.0) / (W / H)))
    return fov


def verify_scan_registration(scan_dir, *, max_frames: int = 8, orientation: str = "Z+",
                             min_coverage: float = 0.35, min_photometric: float = 0.5,
                             coverage_floor: float = 0.05, color_tol: int = 40,
                             use_cache: bool = True) -> dict:
    """Whole-scan scan-schema v1.3 §6 health-check against the scan's render runs.

    Returns {checked, ok, runs:[{run_id, ok, coverage, photometric, n_seen,
    n_points, n_frames, reasons}], reasons}. checked=False (ok=True) when there is
    nothing verifiable (no renders / no images / resolves to legacy as-is with no
    images) — callers must NOT block in that case. A resolver cross-scan ValueError
    is a hard fail (checked=True, ok=False)."""
    from PIL import Image

    from preseg.resolver import dir_cloud_transforms
    from scenes.fingerprint import cloud_fingerprint
    from scenes.frame import apply_transform
    from scenes.point_cloud import load_ply
    from scenes.render_meta import read_render_meta
    from scenes.reproject import ORIENTATION_PRESETS, euler_xyz_matrix
    from scenes.scan_meta import read_scan_meta

    scan_dir = Path(scan_dir)
    renders_root = scan_dir / "renders"
    runs = (sorted(d for d in renders_root.iterdir()
                   if d.is_dir() and (d / "manifest.json").exists())
            if renders_root.is_dir() else [])
    skip = {"checked": False, "ok": True, "runs": [], "reasons": []}
    if not runs:
        return skip

    pc, _ = load_ply(scan_dir / "source" / "scan.ply")
    xyz_raw = np.asarray(pc.points, dtype=np.float64)
    rgb = (np.asarray(pc.colors).astype(np.uint8)
           if pc.colors is not None and len(pc.colors) else None)
    fp = cloud_fingerprint(xyz_raw)

    key = None
    if use_cache:
        run_fps = tuple(sorted(
            (r.name, ((read_render_meta(r) or {}).get("generated_from") or {}).get("source_fingerprint"))
            for r in runs))
        key = (fp, run_fps)
        if key in _VERDICT_CACHE:
            return _VERDICT_CACHE[key]

    R = euler_xyz_matrix(*ORIENTATION_PRESETS[orientation])
    xyz = xyz_raw @ R.T

    sm = read_scan_meta(scan_dir)
    try:
        dir_T = dir_cloud_transforms(runs, sm["frame"], sm["derivation"]["variant_id"], fp, R)
    except ValueError as e:
        verdict = {"checked": True, "ok": False, "runs": [], "reasons": [str(e)]}
        if key is not None:
            _VERDICT_CACHE[key] = verdict
        return verdict

    results = []
    for run in runs:
        manifest = json.loads((run / "manifest.json").read_text())
        frames = [f for f in manifest.get("frames", []) if (run / f["file"]).exists()]
        if not frames:
            continue
        step = max(1, len(frames) // max_frames)
        frames = frames[::step][:max_frames]
        T = dir_T.get(run)
        xyz_run = xyz if T is None else apply_transform(T, xyz)
        W, H = Image.open(run / frames[0]["file"]).size
        intr = (read_render_meta(run) or {}).get("intrinsics") or {}
        fov_y = _fov_y_from_intrinsics(intr, W, H)
        loader = lambda f, _run=run: np.array(Image.open(_run / f["file"]).convert("RGB"))
        s = registration_score(xyz_run, frames, fov_y_deg=fov_y, W=W, H=H,
                               rgb=rgb, image_loader=loader, color_tol=color_tol)
        ok, reasons = check_registration(s, min_coverage=min_coverage,
                                         min_photometric=min_photometric,
                                         coverage_floor=coverage_floor)
        results.append({"run_id": run.name, "ok": ok, "coverage": s["coverage"],
                        "photometric": s["photometric"], "n_seen": s["n_seen"],
                        "n_points": s["n_points"], "n_frames": s["n_frames"],
                        "reasons": reasons})

    if not results:
        verdict = skip
    else:
        verdict = {"checked": True, "ok": all(r["ok"] for r in results), "runs": results,
                   "reasons": [f"{r['run_id']}: {x}" for r in results for x in r["reasons"]]}
    if key is not None:
        _VERDICT_CACHE[key] = verdict
    return verdict
```

Also add the cache dict near the top of the module (after imports):

```python
_VERDICT_CACHE: dict = {}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest backend/tests/test_verify_scan_registration.py -v`
Expected: PASS (both tests).

- [ ] **Step 5: Commit**

```bash
git add backend/preseg/registration.py backend/tests/test_verify_scan_registration.py
git commit -m "feat(preseg): verify_scan_registration whole-scan §6 health-check"
```

---

### Task 2: Unverifiable + cross-scan edge cases

**Files:**
- Test: `backend/tests/test_verify_scan_registration.py`
- Modify: `backend/preseg/registration.py` (only if a test surfaces a gap)

- [ ] **Step 1: Write the failing tests**

Append to `test_verify_scan_registration.py`:

```python
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
```

- [ ] **Step 2: Run to verify**

Run: `.venv/bin/pytest backend/tests/test_verify_scan_registration.py -v`
Expected: PASS — the Task 1 implementation already covers these branches (`skip` for no-renders/no-images; `ValueError` path for cross-scan). If any fails, fix `verify_scan_registration` accordingly, not the test.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_verify_scan_registration.py
git commit -m "test(preseg): cover skip-not-block + cross-scan hard-fail"
```

---

### Task 3: Content-keyed verdict cache

**Files:**
- Test: `backend/tests/test_verify_scan_registration.py`
- Modify: `backend/preseg/registration.py` (only if needed)

- [ ] **Step 1: Write the failing test**

Append (monkeypatches the PLY loader to count reads; cache hit must skip the second read):

```python
def test_cache_skips_reload_for_unchanged_content(tmp_path, monkeypatch):
    scan = build_scan(tmp_path)
    import preseg.registration as reg
    import scenes.point_cloud as pcmod
    reg._VERDICT_CACHE.clear()

    calls = {"n": 0}
    real = pcmod.load_ply

    def counting_load_ply(path):
        calls["n"] += 1
        return real(path)

    monkeypatch.setattr(pcmod, "load_ply", counting_load_ply)
    a = verify_scan_registration(scan, orientation="Y+", use_cache=True)
    b = verify_scan_registration(scan, orientation="Y+", use_cache=True)
    assert a == b
    assert calls["n"] == 1   # second call served from cache, no reload
```

- [ ] **Step 2: Run to verify**

Run: `.venv/bin/pytest backend/tests/test_verify_scan_registration.py::test_cache_skips_reload_for_unchanged_content -v`
Expected: PASS. (Cache key is read *before* `load_ply`, so a hit returns first.)

Note: the function imports `load_ply` locally as `from scenes.point_cloud import load_ply`, which resolves the attribute on `scenes.point_cloud` at call time, so monkeypatching `scenes.point_cloud.load_ply` is effective.

- [ ] **Step 3: Commit**

```bash
git add backend/tests/test_verify_scan_registration.py
git commit -m "test(preseg): verdict cache skips reload for unchanged content"
```

---

### Task 4: Refactor `verify_registration.py` CLI to a thin wrapper

**Files:**
- Rewrite: `scripts/verify_registration.py`

- [ ] **Step 1: Rewrite the CLI**

Replace the body of `main()` with a call to the new function. Full file:

```python
"""Verify a scan's source/scan.ply registers to its renders/<run> poses.

Exit 0 if all checkable runs pass; 2 if any fails OR nothing was verifiable;
3 if there are no render runs at all. Run with voxa's .venv (no torch needed):

    .venv/bin/python scripts/verify_registration.py <scan_dir> [--run <name>]

This is the scan-schema v1.3 §6 registration health-check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from preseg.registration import verify_scan_registration  # noqa: E402
from scenes.reproject import ORIENTATION_PRESETS  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", type=Path)
    ap.add_argument("--fov", type=float, default=60.0,
                    help="(ignored unless a run lacks intrinsics) fallback FOV")
    ap.add_argument("--orientation", default="Z+", choices=list(ORIENTATION_PRESETS))
    ap.add_argument("--min-coverage", type=float, default=0.35)
    ap.add_argument("--min-photometric", type=float, default=0.5)
    args = ap.parse_args()

    renders_root = args.scan_dir / "renders"
    runs = [d for d in renders_root.iterdir()
            if d.is_dir() and (d / "manifest.json").exists()] if renders_root.is_dir() else []
    if not runs:
        print(f"ERROR: no render runs under {renders_root}", file=sys.stderr)
        return 3

    v = verify_scan_registration(args.scan_dir, orientation=args.orientation,
                                 min_coverage=args.min_coverage,
                                 min_photometric=args.min_photometric, use_cache=False)
    print(f"[verify] {args.scan_dir.name}: checked={v['checked']} ok={v['ok']}")
    for r in v["runs"]:
        ph = f"{r['photometric']:.1%}" if r["photometric"] is not None else "n/a"
        flag = "OK  " if r["ok"] else "FAIL"
        print(f"  [{flag}] {r['run_id']}: coverage {r['coverage']:.1%}, photometric {ph} "
              f"({r['n_seen']:,}/{r['n_points']:,}, {r['n_frames']} frames)")
        for reason in r["reasons"]:
            print(f"         - {reason}")
    for reason in v["reasons"]:
        if not v["runs"]:
            print(f"  - {reason}")
    return 0 if (v["checked"] and v["ok"]) else 2


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Smoke-test against a real backfilled scan (manual sanity, not in suite)**

Run: `.venv/bin/python scripts/verify_registration.py /home/hendrik/coding/engine/data/lidar/annotated/navvis_vlx3_water_treatment`
Expected: prints `checked=True ok=True` with per-run OK lines (this is the known-good backfilled scan); exit 0.

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_registration.py
git commit -m "refactor(verify): CLI is a thin wrapper over verify_scan_registration"
```

---

### Task 5: `LoadResponse.frame_check` schema field

**Files:**
- Modify: `backend/app/schemas.py`

- [ ] **Step 1: Locate `LoadResponse` and add the field**

Find `class LoadResponse` (it already has `frame_uncertain`, `georef_offset`). Add, next to those:

```python
    frame_check: Optional[dict] = None   # §6 verdict when the scan was verified; None otherwise
```

Confirm `Optional` is imported (the file already uses `Optional[...]` for `georef_offset`, so it is).

- [ ] **Step 2: Verify it imports cleanly**

Run: `.venv/bin/python -c "import sys; sys.path.insert(0,'backend'); from app.schemas import LoadResponse; print('frame_check' in LoadResponse.model_fields)"`
Expected: `True`.

- [ ] **Step 3: Commit**

```bash
git add backend/app/schemas.py
git commit -m "feat(schemas): add LoadResponse.frame_check"
```

---

### Task 6: Wire the gate into `/api/load` + integration tests

**Files:**
- Modify: `backend/routes/load.py`
- Test: `backend/tests/test_load_endpoint.py`

- [ ] **Step 1: Write the failing integration tests**

Add to `backend/tests/test_load_endpoint.py`. Reuse its existing `_write_ply` helper for the cloud, but these scenes need a v1.3 meta + a render run. Add a builder + two tests:

```python
import numpy as np
from PIL import Image
from scenes.fingerprint import cloud_fingerprint  # noqa: E402  (backend on path via conftest)
from scenes.frame import Frame
from scenes.render_meta import write_render_meta


def _wall_ply(path, rgb):
    g = np.linspace(-2, 2, 40)
    xx, yy = np.meshgrid(g, g)
    pts = np.stack([xx.ravel(), yy.ravel(), -5 * np.ones(xx.size)], -1).astype(np.float32)
    arr = np.zeros(len(pts), dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                    ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
    arr['x'], arr['y'], arr['z'] = pts[:, 0], pts[:, 1], pts[:, 2]
    arr['red'], arr['green'], arr['blue'] = rgb
    path.parent.mkdir(parents=True, exist_ok=True)
    from plyfile import PlyData, PlyElement
    PlyData([PlyElement.describe(arr, 'vertex')], text=False).write(str(path))
    return np.asarray(pts, dtype=np.float64)


def _build_render_scene(lidar, name, *, cloud_rgb, img_rgb):
    import json
    scan = lidar / "annotated" / name
    pts = _wall_ply(scan / "source" / "scan.ply", cloud_rgb)
    fp = cloud_fingerprint(pts)
    scan.joinpath("meta.json").write_text(json.dumps({
        "scan_name": name, "n_points": len(pts), "units": "meters", "schema_version": "1.3",
        "frame": {"canonical_id": f"{name}#local", "transform_to_canonical": np.eye(4).tolist(),
                  "units": "meters", "frame_uncertain": False},
        "derivation": {"scan_id": name, "variant_id": "v1", "parent": "original", "op": "asis",
                       "varies": [], "source_fingerprint": fp, "role": "labeling"},
    }))
    run = scan / "renders" / "run0"; run.mkdir(parents=True)
    frames = [{"file": "f0.png", "position": [0, 0, 0], "target": [0, 0, -1]}]
    Image.fromarray(np.full((120, 120, 3), img_rgb, np.uint8)).save(run / "f0.png")
    (run / "manifest.json").write_text(json.dumps({"frames": frames}))
    write_render_meta(run, run_id="run0",
                      generated_from={"scan_id": name, "variant_id": "v1", "source_fingerprint": fp},
                      frame=Frame(np.eye(4), f"{name}#local"),
                      intrinsics={"fov_deg": 60, "fov_axis": "vertical", "width": 120, "height": 120})


def _client_for(lidar, monkeypatch):
    from fastapi.testclient import TestClient
    monkeypatch.setenv("VOXA_LIDAR_ROOT", str(lidar))
    import importlib, main
    importlib.reload(main)
    return TestClient(main.app)


def test_load_blocks_on_registration_failure(tmp_path, monkeypatch):
    import preseg.registration as reg
    reg._VERDICT_CACHE.clear()
    lidar = tmp_path / "lidar"
    _build_render_scene(lidar, "bad", cloud_rgb=(200, 50, 50), img_rgb=(50, 50, 200))
    client = _client_for(lidar, monkeypatch)
    r = client.post("/api/load", json={"name": "annotated/bad"})
    assert r.status_code == 409
    assert r.json()["detail"]["error"] == "frame_registration_failed"


def test_load_passes_and_surfaces_frame_check(tmp_path, monkeypatch):
    import preseg.registration as reg
    reg._VERDICT_CACHE.clear()
    lidar = tmp_path / "lidar"
    _build_render_scene(lidar, "good", cloud_rgb=(200, 50, 50), img_rgb=(200, 50, 50))
    client = _client_for(lidar, monkeypatch)
    r = client.post("/api/load", json={"name": "annotated/good"})
    assert r.status_code == 200
    assert r.json()["frame_check"]["ok"] is True
```

> Note on orientation: the gate uses the production default `orientation="Z+"`, which rotates the cloud. The wall fixture (a flat z=-5 plane) re-projects fine under Z+ because the camera frame is identity (use_direct) — coverage stays high and photometric is decided purely by pixel-vs-cloud colour. If coverage proves too low under Z+ in practice, the fixture's wall should be defined in the post-rotation frame; verify with the Step-3 run before adjusting.

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_load_endpoint.py -k "registration_failure or surfaces_frame_check" -v`
Expected: FAIL — currently `/api/load` neither blocks (no 409) nor returns `frame_check`.

- [ ] **Step 3: Implement the gate in `load.py`**

In `backend/routes/load.py`, right after the `fsum` block (around line 195, after `fsum = frame_summary(...)` / `except`) and before `return LoadResponse(`:

```python
    # scan-schema v1.3 §6: verify the cloud registers to its renders. Block the
    # load (409) on a real mismatch; never block a scan we cannot verify; a check
    # bug must never break loading a good scan.
    frame_check = None
    from pathlib import Path as _Path
    if _scan_dir and (_Path(_scan_dir) / "renders").is_dir():
        from preseg.registration import verify_scan_registration
        try:
            _v = verify_scan_registration(_Path(_scan_dir))
        except Exception:  # noqa: BLE001 — degrade to "unverified", never break a good load
            _v = {"checked": False, "ok": True, "runs": [], "reasons": []}
        if _v["checked"] and not _v["ok"]:
            raise HTTPException(status_code=409, detail={
                "error": "frame_registration_failed",
                "message": ("Scan does not register to its renders (scan-schema v1.3 §6); "
                            "the cloud and render poses appear to be in different frames."),
                "scan": src.scene_id,
                "frame_check": _v,
            })
        if _v["checked"]:
            frame_check = _v
```

Then add `frame_check=frame_check,` to the `LoadResponse(...)` constructor (next to `frame_uncertain=...`).

- [ ] **Step 4: Run the integration tests + full suite**

Run: `.venv/bin/pytest backend/tests/test_load_endpoint.py -k "registration_failure or surfaces_frame_check" -v`
Expected: PASS (409 for bad, 200 + `frame_check.ok` for good).

Run: `.venv/bin/pytest backend/tests/ -q`
Expected: all pass (no regression; renderless/non-annotated loads unaffected — they never enter the gate).

- [ ] **Step 5: Commit**

```bash
git add backend/routes/load.py backend/tests/test_load_endpoint.py
git commit -m "feat(load): block /api/load on §6 registration failure; surface frame_check"
```

---

## Done criteria

- `.venv/bin/pytest backend/tests/ -q` green.
- `/api/load` of a render-having annotated scan that mismatches → 409 `frame_registration_failed`; a matching one → 200 with `frame_check.ok == true`.
- Renderless annotated scans and non-annotated tiers load 200 with `frame_check == null` (no behaviour change).
- `scripts/verify_registration.py` still works as a CLI (manual run on navvis → exit 0).
