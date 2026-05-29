# Scan Schema v1.3 — Phase 1: Foundation (fingerprint, content cache key, registration health-check) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

> **Status: ✅ IMPLEMENTED (2026-05-29)** — all 6 tasks committed on `worktree-scan-schema-v13`; 17 new tests, full backend suite 204 passing; `verify_registration` fails navvis (exit 2), gate blocks SAM3 (exit 6).

**Goal:** Ship the smallest slice of scan-schema v1.3 that *prevents the navvis frame-mismatch bug class* on existing data, with no schema migration: a deterministic cloud fingerprint, a content-based SAM3 cache key, and a registration health-check that fails loud when a cloud and a render set don't share a frame.

**Architecture:** Three small, self-contained, pure-numpy modules under `backend/` plus one CLI, all unit-testable with synthetic data. Nothing here changes on-disk schema or requires torch; it operates on the *existing* layout and is independently shippable. Later phases (scan/render `meta.json` blocks, `variants.json`, multi-run, writers) build on these primitives.

**Tech stack:** Python 3.12, numpy, pytest (`pythonpath=["backend"]` per root `pyproject.toml`), `plyfile`/Open3D for PLY load (existing), PIL for render images. Targets the **restructured** backend layout (`backend/preseg/`, `backend/scenes/`).

> **Base-branch note (read first):** the v1.3 spec was committed on a worktree branched from the *stale, flat* `origin/main`. This plan targets the **restructured** layout on your local `main` (`backend/preseg/…`, `backend/scenes/…`). Implement on a worktree cut from **local `main` (`c9e484e`)**, not `origin/main`. The spec + this plan are layout-agnostic docs and merge cleanly onto local `main`.

---

## File structure

| File | Responsibility |
|------|----------------|
| `backend/scenes/fingerprint.py` (create) | `cloud_fingerprint(xyz) -> "sha256:…"` — deterministic content identity (§3.2). |
| `backend/scenes/reproject.py` (create) | Pure-numpy camera math: `look_at_view`, `project_points`, `depth_buffer_mask`. Canonical home for the projection currently duplicated in `sam3_features.py` and `scripts/dry_sam3/project_masks.py`. |
| `backend/preseg/registration.py` (create) | `registration_score(...)` + `check_registration(...)` — coverage + photometric agreement of a cloud against render poses (§6). |
| `scripts/verify_registration.py` (create) | CLI: load `source/scan.ply` + `renders/<run>/manifest.json`, score, exit non-zero on fail. |
| `backend/preseg/sam3_features.py` (modify) | Switch `_cache_key` from `n_points` to `source_fingerprint` (§4.5). |
| `scripts/presegment_sam3_features.py` (modify) | Call `check_registration` as a pre-compute gate (with `--skip-registration-check`). |
| `backend/tests/test_fingerprint.py`, `test_reproject.py`, `test_registration.py`, `test_sam3_cache_key.py` (create) | Unit tests. |

Run tests with: `.venv/bin/pytest backend/tests/<file>.py -v` (from voxa root; `pyproject.toml` sets `pythonpath=["backend"]`, so import as `from scenes.fingerprint import …`).

---

## Task 1: Deterministic cloud fingerprint

**Files:**
- Create: `backend/scenes/fingerprint.py`
- Test: `backend/tests/test_fingerprint.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_fingerprint.py
import numpy as np
import pytest
from scenes.fingerprint import cloud_fingerprint

def _grid(n=1000, seed=0):
    return np.random.default_rng(seed).uniform(-10, 10, (n, 3))

def test_order_independent():
    xyz = _grid()
    perm = np.random.default_rng(1).permutation(len(xyz))
    assert cloud_fingerprint(xyz) == cloud_fingerprint(xyz[perm])

def test_submm_jitter_stable():
    xyz = _grid()
    jittered = xyz + np.random.default_rng(2).uniform(-2e-4, 2e-4, xyz.shape)  # <0.5mm
    assert cloud_fingerprint(xyz) == cloud_fingerprint(jittered)

def test_real_move_changes_hash():
    xyz = _grid()
    moved = xyz.copy(); moved[0, 0] += 0.01  # 10mm
    assert cloud_fingerprint(xyz) != cloud_fingerprint(moved)

def test_recenter_changes_hash():
    # the navvis case: same points, translated frame -> different identity
    xyz = _grid()
    assert cloud_fingerprint(xyz) != cloud_fingerprint(xyz + np.array([20.0, 0, 0]))

def test_prefix_and_shape_guard():
    assert cloud_fingerprint(_grid()).startswith("sha256:")
    with pytest.raises(ValueError):
        cloud_fingerprint(np.zeros((10, 2)))
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/pytest backend/tests/test_fingerprint.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scenes.fingerprint'`.

- [ ] **Step 3: Implement**

```python
# backend/scenes/fingerprint.py
"""Deterministic content fingerprint for a point cloud (scan-schema v1.3 §3.2)."""
from __future__ import annotations
import hashlib
import numpy as np

def cloud_fingerprint(xyz: np.ndarray) -> str:
    """sha256 over xyz quantized to integer millimetres, lexicographically
    sorted, serialized as little-endian int32.

    - Order-independent (sorted) -> voxel/thinning reordering doesn't change it.
    - Robust to float round-trips (mm quantize).
    - xyz only; identifies a variant's stored cloud *in its own frame*
      (NOT frame-invariant, NOT equal across variants). Content-identity
      heuristic, not a cryptographic guarantee.
    """
    q = np.round(np.asarray(xyz, dtype=np.float64) * 1000.0).astype("<i4")
    if q.ndim != 2 or q.shape[1] != 3:
        raise ValueError(f"xyz must be (N,3), got {q.shape}")
    order = np.lexsort((q[:, 2], q[:, 1], q[:, 0]))
    digest = hashlib.sha256(np.ascontiguousarray(q[order]).tobytes()).hexdigest()
    return "sha256:" + digest
```

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/pytest backend/tests/test_fingerprint.py -v` → Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/scenes/fingerprint.py backend/tests/test_fingerprint.py
git commit -m "feat(schema): deterministic cloud fingerprint (v1.3 §3.2)"
```

---

## Task 2: Content-based SAM3 cache key (kills stale reuse)

**Files:**
- Modify: `backend/preseg/sam3_features.py` (`_cache_key`, and the one call site in `extract_or_load`)
- Test: `backend/tests/test_sam3_cache_key.py`

**Context:** `_cache_key` currently bakes `n={n_points}` into the hash. The navvis stale-cache bug: a recentered-but-same-count cloud reused an old cache. Replace `n_points` with the §3.2 fingerprint. `_cache_key` is pure (hashlib/json) and importable without torch — the test does not need a GPU.

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_sam3_cache_key.py
import numpy as np
from pathlib import Path
from preseg.sam3_features import _cache_key
from scenes.fingerprint import cloud_fingerprint

def test_same_count_different_coords_differ(tmp_path):
    rd = [tmp_path]; (tmp_path / "manifest.json").write_text("{}")
    a = cloud_fingerprint(np.random.default_rng(0).uniform(0, 1, (1000, 3)))
    b = cloud_fingerprint(np.random.default_rng(0).uniform(0, 1, (1000, 3)) + 20.0)
    ka = _cache_key(rd, a, fpn_level=0, pca_dim=64, orientation="Z+", fov=60.0)
    kb = _cache_key(rd, b, fpn_level=0, pca_dim=64, orientation="Z+", fov=60.0)
    assert ka != kb  # same n_points, different content -> different key

def test_same_content_same_key(tmp_path):
    rd = [tmp_path]; (tmp_path / "manifest.json").write_text("{}")
    fp = cloud_fingerprint(np.random.default_rng(0).uniform(0, 1, (1000, 3)))
    assert _cache_key(rd, fp, 0, 64, "Z+", 60.0) == _cache_key(rd, fp, 0, 64, "Z+", 60.0)
```

- [ ] **Step 2: Run test, verify it fails**

Run: `.venv/bin/pytest backend/tests/test_sam3_cache_key.py -v`
Expected: FAIL — `_cache_key` signature still takes `n_points: int` (TypeError or wrong-key behavior).

- [ ] **Step 3: Implement**

In `backend/preseg/sam3_features.py`, change the `_cache_key` signature `n_points: int` → `source_fingerprint: str` and the hashed line:

```python
def _cache_key(render_dirs, source_fingerprint, fpn_level, pca_dim, orientation, fov):
    h = hashlib.sha256()
    for rd in sorted(str(p.resolve()) for p in render_dirs):
        h.update(rd.encode())
        try:
            h.update(str((Path(rd) / "manifest.json").stat().st_mtime).encode())
        except OSError:
            pass
    h.update(f"fp={source_fingerprint}|fpn={fpn_level}|pca={pca_dim}|"
             f"orient={orientation}|fov={fov}".encode())
    return h.hexdigest()[:16]
```

In `extract_or_load`, compute the fingerprint and pass it (replace the `n = int(xyz.shape[0])` / `key = _cache_key(..., n, ...)` site):

```python
from scenes.fingerprint import cloud_fingerprint   # top-of-file import
...
source_fp = cloud_fingerprint(xyz)
key = _cache_key(render_dirs, source_fp, fpn_level, pca_dim, orientation, fov)
```

Also store `"source_fingerprint": source_fp` in the saved `meta` dict (so the cache self-describes its source).

- [ ] **Step 4: Run test, verify it passes**

Run: `.venv/bin/pytest backend/tests/test_sam3_cache_key.py -v` → Expected: 2 passed.
Then guard against import regressions: `.venv/bin/pytest backend/tests/test_smoke.py -v` → Expected: pass.

- [ ] **Step 5: Commit**

```bash
git add backend/preseg/sam3_features.py backend/tests/test_sam3_cache_key.py
git commit -m "fix(sam3): key feature cache on cloud fingerprint, not point count (v1.3 §4.5)"
```

---

## Task 3: Reusable reprojection module (DRY the camera math)

**Files:**
- Create: `backend/scenes/reproject.py`
- Test: `backend/tests/test_reproject.py`

**Context:** The look-at / project / z-buffer math is duplicated in `backend/preseg/sam3_features.py` (private `_*`) and `scripts/dry_sam3/project_masks.py`. The health-check (Task 4) must project *exactly* as the pipeline does. Create one canonical, public, pure-numpy module. (Migrating `sam3_features.py` to import these is a Phase-2 cleanup — out of scope here to avoid touching torch-adjacent code.)

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_reproject.py
import numpy as np
from scenes.reproject import look_at_view, project_points, depth_buffer_mask

def test_point_ahead_projects_to_center():
    pos = np.array([0.0, 0, 0]); tgt = np.array([0.0, 0, -1])  # look down -Z
    view = look_at_view(pos, tgt)
    pt = np.array([[0.0, 0, -5]])  # straight ahead
    u, v, z, infront = project_points(pt, view, 60.0, 100, 100)
    assert infront[0] and abs(u[0] - 50) < 1 and abs(v[0] - 50) < 1 and z[0] > 0

def test_point_behind_not_in_front():
    view = look_at_view(np.array([0.0, 0, 0]), np.array([0.0, 0, -1]))
    _, _, _, infront = project_points(np.array([[0.0, 0, 5]]), view, 60.0, 100, 100)
    assert not infront[0]

def test_occlusion_near_hides_far():
    view = look_at_view(np.array([0.0, 0, 0]), np.array([0.0, 0, -1]))
    pts = np.array([[0.0, 0, -2], [0.0, 0, -8]])  # same pixel, near + far
    u, v, z, inf = project_points(pts, view, 60.0, 100, 100)
    idx, _, _ = depth_buffer_mask(u, v, z, inf, 100, 100)
    assert 0 in idx and 1 not in idx  # only the near point is visible
```

- [ ] **Step 2: Run test, verify it fails** — `ModuleNotFoundError: scenes.reproject`.

- [ ] **Step 3: Implement** — copy **five** symbols verbatim from `scripts/dry_sam3/project_masks.py` into `backend/scenes/reproject.py`: `look_at_view`, `project_points`, `depth_buffer_mask`, **`euler_xyz_matrix`**, and **`ORIENTATION_PRESETS`**. Add a module docstring noting it is the canonical projection used by both the SAM3 pipeline and the registration check.

  **Why the orientation symbols matter (review C3):** `sam3_features.extract_or_load` rotates the cloud by `ORIENTATION_PRESETS[orientation]` (`pts_rot = xyz @ R.T`) **before** projecting. So any caller of the health-check must apply the *same* rotation before calling `registration_score`, or the check projects a differently-oriented cloud than the pipeline. We keep `registration_score` pure ("project the points you're given") and make the **caller** rotate (Tasks 5 and 6) — mirroring how `extract_or_load` rotates then calls `_project_points`. The Task 4 synthetic tests place points already in the final frame, so they need no rotation.

- [ ] **Step 4: Run test, verify it passes** → Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/scenes/reproject.py backend/tests/test_reproject.py
git commit -m "feat(scenes): canonical reprojection module (look_at/project/zbuffer)"
```

---

## Task 4: Registration score + check

**Files:**
- Create: `backend/preseg/registration.py`
- Test: `backend/tests/test_registration.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_registration.py
import numpy as np
from preseg.registration import registration_score, check_registration

def _wall(n=40):
    g = np.linspace(-2, 2, n)
    xx, yy = np.meshgrid(g, g)
    return np.stack([xx.ravel(), yy.ravel(), -5 * np.ones(xx.size)], -1)  # wall at z=-5

def _frame(pos, tgt):
    return {"position": list(pos), "target": list(tgt)}

def test_coverage_high_when_looking_at_cloud():
    xyz = _wall()
    s = registration_score(xyz, [_frame([0, 0, 0], [0, 0, -1])], fov_y_deg=60, W=200, H=200)
    assert s["coverage"] > 0.5

def test_coverage_zero_when_looking_away():
    xyz = _wall()
    s = registration_score(xyz, [_frame([0, 0, 0], [0, 0, 1])], fov_y_deg=60, W=200, H=200)
    assert s["coverage"] < 0.01

def test_photometric_matches_solid_image():
    xyz = _wall()
    rgb = np.tile(np.array([200, 50, 50], np.uint8), (len(xyz), 1))
    red = np.zeros((200, 200, 3), np.uint8); red[:] = (200, 50, 50)
    blue = np.zeros((200, 200, 3), np.uint8); blue[:] = (50, 50, 200)
    f = [_frame([0, 0, 0], [0, 0, -1])]
    s_ok = registration_score(xyz, f, fov_y_deg=60, W=200, H=200, rgb=rgb, image_loader=lambda _f: red)
    s_bad = registration_score(xyz, f, fov_y_deg=60, W=200, H=200, rgb=rgb, image_loader=lambda _f: blue)
    assert s_ok["photometric"] > 0.9 and s_bad["photometric"] < 0.1

def test_check_fails_below_threshold():
    ok, reasons = check_registration({"coverage": 0.15, "photometric": 0.02}, min_coverage=0.35, min_photometric=0.5)
    assert not ok and reasons
```

- [ ] **Step 2: Run test, verify it fails** — `ModuleNotFoundError: preseg.registration`.

- [ ] **Step 3: Implement**

```python
# backend/preseg/registration.py
"""Registration health-check (scan-schema v1.3 §6): does a cloud actually
project into a render set's poses? Catches frame/version mismatch before any
expensive downstream work, independent of metadata correctness."""
from __future__ import annotations
from typing import Callable, Optional
import numpy as np
from scenes.reproject import look_at_view, project_points, depth_buffer_mask

def registration_score(xyz, frames, *, fov_y_deg, W, H,
                       rgb: Optional[np.ndarray] = None,
                       image_loader: Optional[Callable] = None,
                       color_tol: int = 40) -> dict:
    xyz = np.asarray(xyz, dtype=np.float64)
    seen = np.zeros(xyz.shape[0], dtype=bool)
    agree = total = 0
    for fr in frames:
        pos = np.asarray(fr["position"], float)
        tgt = np.asarray(fr["target"], float) if "target" in fr else \
            pos + np.array([np.cos(fr.get("yaw", 0.0)), 0.0, np.sin(fr.get("yaw", 0.0))])
        u, v, z, infront = project_points(xyz, look_at_view(pos, tgt), fov_y_deg, W, H)
        idx, ui, vi = depth_buffer_mask(u, v, z, infront, W, H)
        seen[idx] = True
        if rgb is not None and image_loader is not None and idx.size:
            img = np.asarray(image_loader(fr))
            diff = np.abs(rgb[idx].astype(int) - img[vi, ui].astype(int)).mean(1)
            agree += int((diff < color_tol).sum()); total += idx.size
    return {
        "coverage": float(seen.mean()),
        "photometric": (agree / total) if total else None,
        "n_seen": int(seen.sum()), "n_points": int(xyz.shape[0]),
        "n_frames": len(frames),
    }

def check_registration(score: dict, *, min_coverage: float = 0.35,
                       min_photometric: float = 0.5) -> tuple[bool, list[str]]:
    reasons = []
    if score["coverage"] < min_coverage:
        reasons.append(f"coverage {score['coverage']:.1%} < {min_coverage:.0%} "
                       f"— cloud likely not in the renders' frame")
    p = score.get("photometric")
    if p is not None and p < min_photometric:
        reasons.append(f"photometric agreement {p:.1%} < {min_photometric:.0%} "
                       f"— projected colours don't match the renders")
    return (not reasons), reasons
```

- [ ] **Step 4: Run test, verify it passes** → Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/preseg/registration.py backend/tests/test_registration.py
git commit -m "feat(preseg): registration health-check — coverage + photometric (v1.3 §6)"
```

---

## Task 5: `verify_registration.py` CLI

**Files:**
- Create: `scripts/verify_registration.py`
- (No new unit test — exercised manually against navvis as the known-fail and a known-good scene; the core logic is covered by Task 4.)

- [ ] **Step 1: Implement the CLI**

```python
# scripts/verify_registration.py
"""Verify a scan's source/scan.ply registers to its renders/<run> poses.

Exit 0 if coverage+photometric pass thresholds; non-zero otherwise. Run with
voxa's .venv (no torch needed):
    .venv/bin/python scripts/verify_registration.py <scan_dir> [--run <name>]
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
from scenes.point_cloud import load_ply              # noqa: E402
from scenes.fingerprint import cloud_fingerprint     # noqa: E402
from scenes.reproject import euler_xyz_matrix, ORIENTATION_PRESETS  # noqa: E402
from preseg.registration import registration_score, check_registration  # noqa: E402

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scan_dir", type=Path)
    ap.add_argument("--run", default=None, help="renders/<run> name (default: all runs)")
    ap.add_argument("--fov", type=float, default=60.0)
    ap.add_argument("--orientation", default="Z+")
    ap.add_argument("--min-coverage", type=float, default=0.35)
    ap.add_argument("--min-photometric", type=float, default=0.5)
    args = ap.parse_args()

    pc, _ = load_ply(args.scan_dir / "source" / "scan.ply")   # load_ply takes NO orientation arg
    xyz = np.asarray(pc.points, dtype=np.float64)
    R = euler_xyz_matrix(*ORIENTATION_PRESETS[args.orientation])   # rotate exactly like extract_or_load
    xyz = xyz @ R.T
    _c = getattr(pc, "colors", None)
    rgb = np.asarray(_c).astype(np.uint8) if _c is not None and len(_c) else None  # .colors is ALREADY uint8 0-255 (no *255)
    print(f"[verify] {args.scan_dir.name}: {len(xyz):,} pts  fp={cloud_fingerprint(xyz)[:23]}…")

    runs = ([args.scan_dir / "renders" / args.run] if args.run
            else sorted(d for d in (args.scan_dir / "renders").iterdir() if (d / "manifest.json").exists()))
    from PIL import Image
    ok_all = True
    for run in runs:
        m = json.loads((run / "manifest.json").read_text())
        frames = [f for f in m.get("frames", []) if (run / f["file"]).exists()]
        if not frames:
            print(f"  [skip] {run.name}: no frame images on disk"); continue
        loader = lambda f, _run=run: np.array(Image.open(_run / f["file"]).convert("RGB"))
        W, H = Image.open(run / frames[0]["file"]).size
        s = registration_score(xyz, frames, fov_y_deg=args.fov, W=W, H=H, rgb=rgb, image_loader=loader)
        ok, reasons = check_registration(s, min_coverage=args.min_coverage, min_photometric=args.min_photometric)
        flag = "OK " if ok else "FAIL"
        ph = f"{s['photometric']:.1%}" if s["photometric"] is not None else "n/a"
        print(f"  [{flag}] {run.name}: coverage {s['coverage']:.1%}, photometric {ph} "
              f"({s['n_seen']:,}/{s['n_points']:,}, {s['n_frames']} frames)")
        for r in reasons:
            print(f"         - {r}")
        ok_all &= ok
    return 0 if ok_all else 2

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Manual verification against the known cases**

Run (known-FAIL — the navvis incident):
`.venv/bin/python scripts/verify_registration.py /home/hendrik/coding/engine/data/lidar/annotated/navvis_vlx3_water_treatment`
Expected: non-zero exit, `[FAIL]` lines. Measured aggregate coverage on navvis is **≈30%** (not 15% — that figure was a single run/older cloud), which is below τ_cov=0.35 so the **coverage** gate fires — but the margin is thin, so the **photometric** agreement (which is low here because the projected colours don't match the renders) is the robust discriminator. If you find coverage hovering near 0.35 on a *good* scene, trust photometric and widen τ_cov rather than tightening it.

Run (known-GOOD — pick a scene whose scan.ply matches its renders, e.g. `smart_ais_clean`):
`.venv/bin/python scripts/verify_registration.py /home/hendrik/coding/engine/data/lidar/annotated/smart_ais_clean`
Expected: higher coverage; use it to sanity-check the default thresholds (tune `--min-coverage`/`--min-photometric` if needed and note the chosen values).

- [ ] **Step 3: Commit**

```bash
git add scripts/verify_registration.py
git commit -m "feat(scripts): verify_registration CLI — fail loud on cloud/render frame mismatch"
```

---

## Task 6: Wire the health-check as a pre-SAM3 gate

**Files:**
- Modify: `scripts/presegment_sam3_features.py`

**Context:** Stage 1 currently runs SAM3 (expensive) regardless of whether the cloud registers to the renders. Gate it.

- [ ] **Step 1: Add the gate** — after the render runs are discovered and `xyz`/`pc` are loaded, before `sam3.extract_or_load(...)`, run the check. The snippet below defines **all** its variables (review C5: the earlier draft referenced undefined `rgb`/`loader`/`W`/`H`). Note it (a) captures `pc.colors` as uint8, (b) rotates `xyz` by the orientation preset to match `extract_or_load`, and (c) builds a per-run image loader.

```python
ap.add_argument("--skip-registration-check", action="store_true",
                help="bypass the cloud↔renders registration health-check (not recommended)")
...
if not args.skip_registration_check:
    from PIL import Image
    from scenes.reproject import euler_xyz_matrix, ORIENTATION_PRESETS
    from preseg.registration import registration_score, check_registration

    # orientation here must equal the one passed to extract_or_load below (default "Z+")
    Rcheck = euler_xyz_matrix(*ORIENTATION_PRESETS["Z+"])
    xyz_chk = xyz @ Rcheck.T
    _c = getattr(pc, "colors", None)
    rgb = np.asarray(_c).astype(np.uint8) if _c is not None and len(_c) else None

    sample, loaders = [], {}
    for r in runs:                              # runs: list[RenderRun]; r.path is its dir
        m = json.loads((r.path / "manifest.json").read_text())
        frs = [f for f in m.get("frames", []) if (r.path / f["file"]).exists()][:8]
        if not frs:
            continue
        if "W" not in dir():
            W, H = Image.open(r.path / frs[0]["file"]).size
        for f in frs:
            f["_run"] = str(r.path); sample.append(f)
    def loader(f):
        return np.array(Image.open(Path(f["_run"]) / f["file"]).convert("RGB"))

    score = registration_score(xyz_chk, sample, fov_y_deg=60.0, W=W, H=H,
                               rgb=rgb, image_loader=loader)
    ok, reasons = check_registration(score)
    if not ok:
        print("ERROR: cloud does not register to its renders — refusing to compute "
              "SAM3 features:", file=sys.stderr)
        for r in reasons:
            print(f"  - {r}", file=sys.stderr)
        print("  (re-run with --skip-registration-check to override)", file=sys.stderr)
        return 6
```

(If `pc` isn't in scope at that point in the script, load colours the same way the script already loads `xyz`; the key fixes are: capture `.colors` as uint8, rotate `xyz` by the preset, and define `W`/`H`/`loader`/`sample` explicitly.)

- [ ] **Step 2: Verify the gate fires on navvis**

Run: `/home/hendrik/anaconda3/bin/python scripts/presegment_sam3_features.py <navvis scan_dir>`
Expected: exits 6 with the registration-failure message, *before* loading the SAM3 model.

- [ ] **Step 3: Commit**

```bash
git add scripts/presegment_sam3_features.py
git commit -m "feat(sam3): gate feature extraction on the registration health-check"
```

---

## Done criteria

- `.venv/bin/pytest backend/tests/test_fingerprint.py backend/tests/test_reproject.py backend/tests/test_registration.py backend/tests/test_sam3_cache_key.py -v` all pass.
- `verify_registration.py` exits non-zero on navvis (coverage ≈ 30%, below τ_cov, *and* low photometric agreement) and zero on a known-good scene.
- The SAM3 stage-1 script refuses to run on a mis-registered cloud.
- No on-disk schema change, no torch dependency added to the tested path — Phase 1 is independently shippable.

## Not in this plan (later phases)

- Scan/variant `meta.json` `frame` + `derivation` blocks; render-run `meta.json`; `variants.json` (spec §4.1–4.3).
- Multi-run `labels/runs/` + `prelabel/runs/` and Compare/merge (spec §4.6).
- Cross-variant consumer rules + label propagation (spec §5); writer updates across voxa/SAM3/walker; replacing `lidar/SCHEMA.md` (spec §9).
