# SAM-aided Labeling Tool Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a single-view SAM (Segment Anything v3) labeling tool to voxa — a 5th Label-mode rail tool where the user boxes or text-prompts an object, SAM segments a server-rendered view, and the mask back-projects to a scan-resolution point selection that flows through voxa's existing apply→confirm pipeline.

**Architecture:** A standalone **SAM sidecar** (separate GPU/anaconda process) renders the raw cloud from voxa's camera pose, runs SAM, and back-projects masks to `scan.ply` point indices. A thin **voxa backend proxy** (`backend/routes/sam.py`) forwards to the sidecar and applies returned indices via the existing `apply_reassign(..., protect_instances=)`. The **voxa frontend** adds the rail tool, a box-drag capture gesture, and a mask-review panel. voxa's Three.js viewer is unchanged — it is only a camera-pose + box source. See spec: `docs/superpowers/specs/2026-07-12-sam-labeling-tool-design.md`.

**Tech Stack:** Backend sidecar: FastAPI + uvicorn, numpy, scipy, laspy[lazrs], plyfile, Pillow, torch + sam3 (anaconda env). voxa backend: FastAPI (existing). Frontend: React 18 + Three.js + Vite (existing). Tests: pytest (sidecar + voxa backend), vitest (frontend).

**Reference sources (read, adapt — do not modify):**
- PoC (validated method): `/home/hendrik/coding/engine/tools/labeling/quick-select-visual-test/` — `reproject.py`, `sam_infer.py`, `main.py`, `cloud.py`, `session.py`, `tests/test_reproject.py`.
- Validated perspective-splat render: scratchpad `persp_raw.py` / `persp_splat.py` from the design session, and `/home/hendrik/coding/engine/tools/data_tools/render_scans.py`.
- voxa canonical projection math: `backend/scenes/reproject.py`.
- Apply pipeline: `backend/labeling/segment_state.py::apply_reassign` (lines 98-134), `backend/routes/segment.py::_apply_shape_core` (lines 89-126).
- Existing rail tool for pattern: `frontend/src/beam-mode.jsx`, `frontend/src/draw-mode.jsx`, `frontend/src/tool-options.jsx`, `frontend/src/label-tools.js`.

**Scan location for manual/e2e testing:** `smart_ais_navvis` (annotated tier). Raw cloud: `/home/hendrik/coding/engine/data/lidar/raw/Sample-Data-VLX3-ProcessIndustry-SMART-AIS.laz`.

**Sidecar location:** new top-level dir `sam_sidecar/` in the voxa repo (sibling of `backend/`, `frontend/`).

---

## File Structure

**Sidecar (`sam_sidecar/`, new):**
- `reproject.py` — projection math (copied verbatim from `backend/scenes/reproject.py`).
- `render.py` — perspective point-splat render → RGB image + depth-buffer.
- `backproject.py` — project a point set through a camera + occlusion test against a depth-buffer → visible indices in mask.
- `cloud.py` — raw-LAZ loader (cached `.npy`) + `scan.ply` loader.
- `sam_infer.py` — SAM3 box + concept wrappers (adapted from PoC).
- `scan_store.py` — per-scan lazy load keyed by source fingerprint; the identity guard.
- `main.py` — FastAPI app: `/capture`, `/project`, `/health`.
- `requirements.txt`, `run.sh`.
- `tests/test_reproject.py`, `tests/test_render.py`, `tests/test_backproject.py`, `tests/test_scan_store.py`.

**voxa backend:**
- `backend/routes/sam.py` (new) — proxy router `/api/sam/capture`, `/api/sam/project`.
- `backend/app/schemas.py` (modify) — SAM request/response models.
- `backend/main.py` (modify, line 35) — register the `sam` router.
- `backend/tests/test_sam_proxy.py` (new) — proxy tests with a mocked sidecar.

**voxa frontend:**
- `frontend/src/label-tools.js` (modify) — add the `sam` tool + gating.
- `frontend/src/api.js` (modify) — `samCapture`, `samProject`.
- `frontend/src/sam-util.js` (new) — pure fns: box normalization, camera-pose payload.
- `frontend/src/sam-util.test.js` (new) — pure-fn tests.
- `frontend/src/sam-mode.jsx` (new) — box-drag capture + mask-review panel (sibling of `beam-mode.jsx`).
- `frontend/src/tool-options.jsx` (modify) — render the SAM panel when `activeTool==='sam'`.
- `frontend/src/mode-label.jsx` (modify) — gate the tool, thread `protectedSegIds`, mount `sam-mode`.

**Docs:**
- `CLAUDE.md` (modify) — 5th rail tool + sidecar note.
- `sam_sidecar/README.md` (new) — run instructions.

---

## Task 1: Sidecar scaffold + projection sanity test

**Files:**
- Create: `sam_sidecar/reproject.py`, `sam_sidecar/requirements.txt`, `sam_sidecar/.gitignore`
- Test: `sam_sidecar/tests/test_reproject.py`

- [ ] **Step 1: Copy projection math verbatim**

Copy `backend/scenes/reproject.py` → `sam_sidecar/reproject.py` unchanged. It provides `look_at_view`, `project_points`, `depth_buffer_mask`, `euler_xyz_matrix`.

- [ ] **Step 2: Write the failing projection test** (`sam_sidecar/tests/test_reproject.py`)

```python
import numpy as np
from reproject import look_at_view, project_points, depth_buffer_mask

def test_center_point_projects_to_image_center():
    pos = np.array([0.0, -10.0, 0.0]); target = np.array([0.0, 0.0, 0.0])
    view = look_at_view(pos, target, up=(0.0, 0.0, 1.0))   # Z-up world
    u, v, z, infront = project_points(np.array([[0.0, 0.0, 0.0]]), view, 60.0, 100, 100)
    assert infront[0]
    assert abs(u[0] - 50) < 1e-6 and abs(v[0] - 50) < 1e-6
    assert abs(z[0] - 10.0) < 1e-6

def test_occluded_point_rejected():
    pos = np.array([0.0, -10.0, 0.0]); target = np.array([0.0, 0.0, 0.0])
    view = look_at_view(pos, target, up=(0.0, 0.0, 1.0))
    pts = np.array([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]])   # near + far-behind-it
    u, v, z, infront = project_points(pts, view, 60.0, 100, 100)
    idx, uu, vv = depth_buffer_mask(u, v, z, infront, 100, 100)
    assert 0 in idx and 1 not in idx
```

- [ ] **Step 3: `requirements.txt`**

```
fastapi
uvicorn[standard]
laspy[lazrs]
plyfile
pillow
numpy
scipy
pytest
httpx
```
(`torch`/`sam3` already in the anaconda env — do NOT pin.) `.gitignore`: `__pycache__/`, `*.npy`, `.pytest_cache/`.

- [ ] **Step 4: Run the test**

Run: `cd sam_sidecar && /home/hendrik/anaconda3/bin/python -m pytest tests/test_reproject.py -v`
Expected: 2 PASS. (If `v` fails, the pixel-flip/up convention is wrong — stop and fix; everything downstream depends on it.)

- [ ] **Step 5: Commit**

```bash
git add sam_sidecar/reproject.py sam_sidecar/requirements.txt sam_sidecar/.gitignore sam_sidecar/tests/test_reproject.py
git commit -m "feat(sam-sidecar): scaffold + projection sanity test"
```

---

## Task 2: Perspective splat render

**Files:**
- Create: `sam_sidecar/render.py`
- Test: `sam_sidecar/tests/test_render.py`

- [ ] **Step 1: Write the failing render test**

```python
import numpy as np
from render import render_view
from reproject import look_at_view

def test_render_produces_image_and_depth():
    # a 2x2m colored wall at y=0, camera on -Y looking at it (Z-up)
    g = np.linspace(-1, 1, 200)
    xx, zz = np.meshgrid(g, g)
    xyz = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()]).astype(np.float32)
    rgb = np.tile(np.array([200, 50, 50], np.uint8), (xyz.shape[0], 1))
    view = look_at_view(np.array([0.0, -5.0, 0.0]), np.zeros(3), up=(0, 0, 1))
    img, depth = render_view(xyz, rgb, view, fov_y=60.0, W=128, H=128)
    assert img.shape == (128, 128, 3) and depth.shape == (128, 128)
    # center pixels are the wall color, corners are background (inf depth)
    assert tuple(img[64, 64]) == (200, 50, 50)
    assert np.isinf(depth[0, 0])
    assert abs(depth[64, 64] - 5.0) < 0.2
```

- [ ] **Step 2: Run it to confirm it fails**

Run: `cd sam_sidecar && /home/hendrik/anaconda3/bin/python -m pytest tests/test_render.py -v`
Expected: FAIL (`No module named 'render'`).

- [ ] **Step 3: Implement `render.py`**

Port the validated scratchpad splat (distance-adaptive radius). Camera-space math must match `reproject.project_points` (looks down -Z, square pixels, v flipped).

```python
"""Perspective point-splat render → RGB image + depth-buffer. Pure numpy.
Validated at full 188M density (design session persp_raw.py)."""
from __future__ import annotations
import numpy as np

def render_view(xyz, rgb, view, fov_y, W, H, splat=2, bg=(0, 0, 0)):
    """xyz float[N,3] (native frame), rgb uint8[N,3], view=world->camera 4x4.
    Returns (color uint8[H,W,3], depth float32[H,W] with inf where empty)."""
    R = view[:3, :3].astype(np.float32); t = view[:3, 3].astype(np.float32)
    f = np.float32((H / 2.0) / np.tan(np.deg2rad(fov_y) / 2.0))
    cam = xyz.astype(np.float32) @ R.T + t
    z = -cam[:, 2]
    u = (cam[:, 0] * f) / np.maximum(z, 1e-6) + W / 2.0
    v = (-cam[:, 1] * f) / np.maximum(z, 1e-6) + H / 2.0
    ix = u.astype(np.int64); iy = v.astype(np.int64)
    ok = (z > 0.05) & (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
    ix, iy, z, rgb = ix[ok], iy[ok], z[ok].astype(np.float32), rgb[ok]
    order = np.argsort(-z, kind="stable")     # far first → nearest wins on last write
    ix, iy, z, rgb = ix[order], iy[order], z[order], rgb[order]
    color = np.full((H, W, 3), bg, np.uint8)
    depth = np.full((H, W), np.inf, np.float32)
    for dv in range(-splat, splat + 1):
        for du in range(-splat, splat + 1):
            xx = np.clip(ix + du, 0, W - 1); yy = np.clip(iy + dv, 0, H - 1)
            closer = z < depth[yy, xx]
            yc, xc = yy[closer], xx[closer]
            depth[yc, xc] = z[closer]; color[yc, xc] = rgb[closer]
    return color, depth
```

- [ ] **Step 4: Run the test**

Run: `cd sam_sidecar && /home/hendrik/anaconda3/bin/python -m pytest tests/test_render.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add sam_sidecar/render.py sam_sidecar/tests/test_render.py
git commit -m "feat(sam-sidecar): perspective splat render + depth-buffer"
```

---

## Task 3: Back-projection (mask + depth-buffer → visible scan indices)

**Files:**
- Create: `sam_sidecar/backproject.py`
- Test: `sam_sidecar/tests/test_backproject.py`

Selects the scan-res points that fall inside a boolean mask **and** are visible per the raw depth-buffer. Tolerances reuse the PoC (`tol_rel=0.01`, `tol_abs=0.15`) — a scan point is visible if its depth ≤ the (splatted) raw depth at its pixel + tol.

- [ ] **Step 1: Write the failing test**

```python
import numpy as np
from backproject import select_in_mask
from render import render_view
from reproject import look_at_view

def test_selects_masked_visible_points():
    g = np.linspace(-1, 1, 120); xx, zz = np.meshgrid(g, g)
    wall = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()]).astype(np.float32)
    rgb = np.tile(np.array([180, 180, 180], np.uint8), (wall.shape[0], 1))
    view = look_at_view(np.array([0.0, -5.0, 0.0]), np.zeros(3), up=(0, 0, 1))
    _, depth = render_view(wall, rgb, view, 60.0, 128, 128)
    mask = np.zeros((128, 128), bool); mask[40:88, 40:88] = True   # center square
    # scan-res cloud = the same wall subsampled
    scan = wall[::4]
    sel = select_in_mask(scan, view, 60.0, 128, 128, mask, depth)
    assert sel.size > 0
    # every selected scan point projects inside the mask box
    from reproject import project_points
    u, v, z, infront = project_points(scan[sel], view, 60.0, 128, 128)
    assert np.all(mask[v.astype(int), u.astype(int)])

def test_occluded_scan_points_excluded():
    # a near wall + a far wall directly behind; only the near should be selected
    g = np.linspace(-1, 1, 120); xx, zz = np.meshgrid(g, g)
    near = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()]).astype(np.float32)
    far = near + np.array([0, 3.0, 0], np.float32)
    rgb = np.tile(np.array([180, 180, 180], np.uint8), (near.shape[0], 1))
    view = look_at_view(np.array([0.0, -5.0, 0.0]), np.zeros(3), up=(0, 0, 1))
    _, depth = render_view(near, rgb, view, 60.0, 128, 128)   # depth from NEAR wall
    scan = np.vstack([near[::4], far[::4]])
    mask = np.ones((128, 128), bool)
    sel = select_in_mask(scan, view, 60.0, 128, 128, mask, depth)
    n_near = near[::4].shape[0]
    assert np.all(sel < n_near)   # no far-wall points survive occlusion
```

- [ ] **Step 2: Run to confirm failure.** `pytest tests/test_backproject.py -v` → FAIL.

- [ ] **Step 3: Implement `backproject.py`**

```python
"""Mask + raw depth-buffer → visible scan-res point indices."""
from __future__ import annotations
import numpy as np
from scipy.ndimage import minimum_filter
from reproject import project_points

def select_in_mask(scan_xyz, view, fov_y, W, H, mask, depth,
                   tol_rel=0.01, tol_abs=0.15, splat=2):
    u, v, z, infront = project_points(scan_xyz, view, fov_y, W, H)
    ui = u.astype(np.int64); vi = v.astype(np.int64)
    ok = infront & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
    idx = np.where(ok)[0]
    ui, vi, zz = ui[ok], vi[ok], z[ok].astype(np.float32)
    zb = minimum_filter(depth, size=2 * splat + 1, mode="nearest") if splat else depth
    z_at = zb[vi, ui]
    tol = np.maximum(tol_abs, tol_rel * np.where(np.isinf(z_at), 0.0, z_at))
    visible = zz <= z_at + tol            # inf background → never visible (zz finite)
    in_mask = mask[vi, ui]
    return idx[visible & in_mask].astype(np.int32)
```

- [ ] **Step 4: Run the test.** `pytest tests/test_backproject.py -v` → 2 PASS.
- [ ] **Step 5: Commit**

```bash
git add sam_sidecar/backproject.py sam_sidecar/tests/test_backproject.py
git commit -m "feat(sam-sidecar): mask+depth back-projection to scan indices"
```

---

## Task 4: Cloud loaders + per-scan store (identity guard)

**Files:**
- Create: `sam_sidecar/cloud.py`, `sam_sidecar/scan_store.py`
- Test: `sam_sidecar/tests/test_scan_store.py`

`scan_store` is the correctness-critical identity guard from the spec: one loaded scan at a time, keyed by `source_fingerprint`; a mismatch is a hard error, never a silent wrong-cloud serve.

- [ ] **Step 1: `cloud.py`** (I/O; no unit test — smoke-checked in Task 8)

```python
"""Cloud loaders for the sidecar. Raw LAZ (cached .npy) + scan.ply."""
from __future__ import annotations
from pathlib import Path
import numpy as np

def load_raw(laz_path: str, cache_dir: str = ".") -> tuple[np.ndarray, np.ndarray]:
    """Full-res raw cloud (native frame). Returns (xyz float32[N,3], rgb uint8[N,3])."""
    cache = Path(cache_dir) / (Path(laz_path).stem + ".rawcache.npz")
    if cache.exists():
        d = np.load(cache); return d["xyz"], d["rgb"]
    import laspy
    xyz_parts, rgb_parts = [], []
    with laspy.open(laz_path) as fh:
        for ch in fh.chunk_iterator(6_000_000):
            xyz_parts.append(np.column_stack([ch.x, ch.y, ch.z]).astype(np.float32))
            r = (np.asarray(ch.red, np.uint32) >> 8).astype(np.uint8)
            g = (np.asarray(ch.green, np.uint32) >> 8).astype(np.uint8)
            b = (np.asarray(ch.blue, np.uint32) >> 8).astype(np.uint8)
            rgb_parts.append(np.stack([r, g, b], 1))
    xyz = np.concatenate(xyz_parts); rgb = np.concatenate(rgb_parts)
    np.savez(cache, xyz=xyz, rgb=rgb)
    return xyz, rgb

def load_scan_ply(ply_path: str) -> np.ndarray:
    """scan.ply xyz in native frame, in file order (indices must match voxa's session)."""
    from plyfile import PlyData
    v = PlyData.read(ply_path)["vertex"].data
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
```

- [ ] **Step 2: Write the failing store test** (`tests/test_scan_store.py`)

```python
import numpy as np, pytest
from scan_store import ScanStore, FingerprintMismatch

def _fake_loader(scan_id):
    xyz = np.zeros((10, 3), np.float32); rgb = np.zeros((10, 3), np.uint8)
    return xyz, rgb, xyz.copy()   # raw_xyz, raw_rgb, scan_xyz

def test_loads_and_serves_matching_scan():
    s = ScanStore(loader=_fake_loader)
    s.ensure("scanA", "fp1")
    assert s.scan_id == "scanA" and s.fingerprint == "fp1"
    assert s.scan_xyz.shape == (10, 3)

def test_different_scan_id_reloads():
    calls = []
    def loader(sid): calls.append(sid); return _fake_loader(sid)
    s = ScanStore(loader=loader)
    s.ensure("scanA", "fp1"); s.ensure("scanB", "fp2")
    assert s.scan_id == "scanB" and calls == ["scanA", "scanB"]

def test_same_id_different_fingerprint_raises():
    s = ScanStore(loader=_fake_loader)
    s.ensure("scanA", "fp1")
    with pytest.raises(FingerprintMismatch):
        s.ensure("scanA", "fp-DIFFERENT")
```

- [ ] **Step 3: Run to confirm failure.** `pytest tests/test_scan_store.py -v` → FAIL.

- [ ] **Step 4: Implement `scan_store.py`**

```python
"""One loaded scan at a time, keyed by source fingerprint. The cross-scan
identity guard: same scan_id + different fingerprint is a hard error so
/project can never return indices for the wrong cloud."""
from __future__ import annotations
from typing import Callable

class FingerprintMismatch(Exception):
    pass

class ScanStore:
    def __init__(self, loader: Callable[[str], tuple]):
        self._loader = loader
        self.scan_id = None; self.fingerprint = None
        self.raw_xyz = None; self.raw_rgb = None; self.scan_xyz = None

    def ensure(self, scan_id: str, fingerprint: str) -> None:
        if self.scan_id == scan_id:
            if self.fingerprint != fingerprint:
                raise FingerprintMismatch(
                    f"scan '{scan_id}' loaded at fingerprint {self.fingerprint}, "
                    f"request carried {fingerprint}")
            return
        self.raw_xyz, self.raw_rgb, self.scan_xyz = self._loader(scan_id)
        self.scan_id = scan_id; self.fingerprint = fingerprint
```

The real loader (wired in `main.py`, Task 6) resolves the raw LAZ + `scan.ply` paths for a `scan_id` and calls `cloud.load_raw` / `cloud.load_scan_ply`. Path resolution reads the scan's `derivation→sources.json` (raw) and `source/scan.ply`; the scan root comes from `$VOXA_LIDAR_ROOT/annotated/<name>/` (mirror `backend/scene_registry.py`). Encapsulate as `sam_sidecar/resolve.py` if it grows.

- [ ] **Step 5: Run the test.** `pytest tests/test_scan_store.py -v` → 3 PASS.
- [ ] **Step 6: Commit**

```bash
git add sam_sidecar/cloud.py sam_sidecar/scan_store.py sam_sidecar/tests/test_scan_store.py
git commit -m "feat(sam-sidecar): cloud loaders + fingerprint-guarded scan store"
```

---

## Task 5: SAM inference wrappers (box + concept)

**Files:**
- Create: `sam_sidecar/sam_infer.py`
- Test: none automated (requires CUDA weights) — smoke-checked manually in Step 3.

Adapt from the PoC `sam_infer.py` (already validated): `build_processor`, `segment_box` (best mask), `segment_concept` (all instances ≥ `min_score`). **Drop** `segment_everything` (out of v1 scope). Import `sam3` lazily so the module imports without CUDA.

- [ ] **Step 1: Copy PoC `sam_infer.py` → `sam_sidecar/sam_infer.py`**, then delete `_iou` and `segment_everything`. Keep `BPE_PATH`, `build_processor`, `segment_box`, `segment_concept`, `_masks_scores`. `segment_concept` default `min_score=0.3` (spec tunable).

- [ ] **Step 2: Confirm it imports without CUDA**

Run: `cd sam_sidecar && /home/hendrik/anaconda3/bin/python -c "import sam_infer; print('ok')"`
Expected: `ok` (lazy import means no torch needed to import the module).

- [ ] **Step 3: Manual GPU smoke test** (skip in CI)

Run: `/home/hendrik/anaconda3/bin/python -c "from PIL import Image; import numpy as np; import sam_infer as s; p=s.build_processor(); img=Image.fromarray(np.random.randint(0,255,(400,400,3),np.uint8)); m,sc=s.segment_box(p,img,[0.5,0.5,0.4,0.4]); print(m.shape, m.sum(), sc)"`
Expected: prints a `(400,400)` mask shape + a score. Confirms weights load + box path runs.

- [ ] **Step 4: Commit**

```bash
git add sam_sidecar/sam_infer.py
git commit -m "feat(sam-sidecar): SAM3 box + concept wrappers"
```

---

## Task 6: Sidecar FastAPI app (`/capture`, `/project`, `/health`)

**Files:**
- Create: `sam_sidecar/main.py`, `sam_sidecar/resolve.py`, `sam_sidecar/run.sh`
- Test: `sam_sidecar/tests/test_api.py` (mask logic with a monkeypatched SAM; no CUDA)

Endpoint contract (from spec):
- `POST /capture` `{scan_id, source_fingerprint, camera:{pos,target,fov,W,H}, mode:"box"|"concept", box?, text?}` → `{capture_id, overlay_png_b64, masks:[{mask_id, score}]}`. Renders raw, runs SAM, stashes `{depth, masks, camera}` under a fresh `capture_id` (replaces prior). `pos`/`target` are in **native** coords (voxa added `recenter_offset` before calling).
- `POST /project` `{scan_id, source_fingerprint, capture_id, mask_ids:[...]}` → `{instances:[{mask_id, scan_indices_b64}]}` (b64 int32). Stale `capture_id` → 409; `FingerprintMismatch` → 409 `{diverged:"source"}`.

- [ ] **Step 1: Write the failing API test** (monkeypatch SAM so no CUDA is needed)

```python
import base64, numpy as np
from fastapi.testclient import TestClient

def _make_client(monkeypatch):
    import main
    # fake store: a small wall as both raw and scan cloud
    g = np.linspace(-1, 1, 60); xx, zz = np.meshgrid(g, g)
    wall = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()]).astype(np.float32)
    rgb = np.tile(np.array([180,180,180], np.uint8), (wall.shape[0],1))
    main.STORE._loader = lambda sid: (wall, rgb, wall.copy())
    # fake SAM: a full-frame mask, score 0.9
    monkeypatch.setattr(main, "_sam_box", lambda img, box, text: (np.ones(img.size[::-1], bool), 0.9))
    return TestClient(main.app)

def test_capture_then_project(monkeypatch):
    c = _make_client(monkeypatch)
    cam = {"pos": [0,-5,0], "target": [0,0,0], "fov": 60.0, "W": 128, "H": 128}
    r = c.post("/capture", json={"scan_id":"A","source_fingerprint":"fp","camera":cam,
                                 "mode":"box","box":[0.5,0.5,0.6,0.6]})
    assert r.status_code == 200
    cap = r.json(); assert cap["masks"] and "capture_id" in cap
    r2 = c.post("/project", json={"scan_id":"A","source_fingerprint":"fp",
                                  "capture_id":cap["capture_id"],"mask_ids":[cap["masks"][0]["mask_id"]]})
    assert r2.status_code == 200
    inst = r2.json()["instances"][0]
    sel = np.frombuffer(base64.b64decode(inst["scan_indices_b64"]), np.int32)
    assert sel.size > 0

def test_stale_capture_id_409(monkeypatch):
    c = _make_client(monkeypatch)
    cam = {"pos":[0,-5,0],"target":[0,0,0],"fov":60.0,"W":128,"H":128}
    c.post("/capture", json={"scan_id":"A","source_fingerprint":"fp","camera":cam,"mode":"box","box":[0.5,0.5,0.6,0.6]})
    r = c.post("/project", json={"scan_id":"A","source_fingerprint":"fp","capture_id":"stale","mask_ids":[0]})
    assert r.status_code == 409

def test_fingerprint_mismatch_409(monkeypatch):
    c = _make_client(monkeypatch)
    cam = {"pos":[0,-5,0],"target":[0,0,0],"fov":60.0,"W":128,"H":128}
    c.post("/capture", json={"scan_id":"A","source_fingerprint":"fp","camera":cam,"mode":"box","box":[0.5,0.5,0.6,0.6]})
    r = c.post("/capture", json={"scan_id":"A","source_fingerprint":"DIFF","camera":cam,"mode":"box","box":[0.5,0.5,0.6,0.6]})
    assert r.status_code == 409 and r.json()["detail"].get("diverged") == "source"
```

- [ ] **Step 2: Run to confirm failure.** `pytest tests/test_api.py -v` → FAIL.

- [ ] **Step 3: Implement `main.py`.** Structure (fill from the reference patterns already built):

```python
import base64, io, uuid
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image
from pydantic import BaseModel

import cloud, resolve
from render import render_view
from backproject import select_in_mask
from reproject import look_at_view
from scan_store import ScanStore, FingerprintMismatch

STORE = ScanStore(loader=lambda sid: (
    *cloud.load_raw(resolve.raw_laz(sid)), cloud.load_scan_ply(resolve.scan_ply(sid))))
CAPTURES: dict = {}          # one live capture_id at a time
_PROC = {"proc": None}       # lazy SAM processor

def _proc():
    if _PROC["proc"] is None:
        import sam_infer; _PROC["proc"] = sam_infer.build_processor()
    return _PROC["proc"]

def _sam_box(img, box, text):          # wrappers → monkeypatchable in tests
    import sam_infer; return sam_infer.segment_box(_proc(), img, box, text)
def _sam_concept(img, text):
    import sam_infer; return sam_infer.segment_concept(_proc(), img, text)

app = FastAPI()

class Camera(BaseModel):
    pos: list[float]; target: list[float]; fov: float; W: int; H: int
class CaptureReq(BaseModel):
    scan_id: str; source_fingerprint: str; camera: Camera
    mode: str; box: list[float] | None = None; text: str | None = None
class ProjectReq(BaseModel):
    scan_id: str; source_fingerprint: str; capture_id: str; mask_ids: list[int]

def _ensure(scan_id, fp):
    try:
        STORE.ensure(scan_id, fp)
    except FingerprintMismatch as e:
        raise HTTPException(409, {"diverged": "source", "detail": str(e)})

@app.get("/health")
def health(): return {"ok": True, "scan_id": STORE.scan_id}

@app.post("/capture")
def capture(req: CaptureReq):
    _ensure(req.scan_id, req.source_fingerprint)
    cam = req.camera
    view = look_at_view(cam.pos, cam.target, up=(0.0, 0.0, 1.0))
    color, depth = render_view(STORE.raw_xyz, STORE.raw_rgb, view, cam.fov, cam.W, cam.H)
    frame = Image.fromarray(color, "RGB")
    if req.mode == "concept":
        if not req.text: raise HTTPException(400, "text required for concept mode")
        insts = _sam_concept(frame, req.text)               # [(mask,score), ...]
    else:
        m, sc = _sam_box(frame, req.box, req.text); insts = [(m, sc)] if m.any() else []
    masks = [ _resize(m, cam.H, cam.W) for m, _ in insts ]
    cid = uuid.uuid4().hex
    CAPTURES.clear()
    CAPTURES[cid] = {"depth": depth, "masks": masks, "camera": cam.model_dump()}
    overlay = _overlay_png(color, masks)
    return {"capture_id": cid, "overlay_png_b64": overlay,
            "masks": [{"mask_id": i, "score": float(insts[i][1])} for i in range(len(masks))]}

@app.post("/project")
def project(req: ProjectReq):
    _ensure(req.scan_id, req.source_fingerprint)
    cap = CAPTURES.get(req.capture_id)
    if cap is None: raise HTTPException(409, "stale or unknown capture_id")
    cam = cap["camera"]; view = look_at_view(cam["pos"], cam["target"], up=(0.0, 0.0, 1.0))
    out = []
    for mid in req.mask_ids:
        if mid < 0 or mid >= len(cap["masks"]): continue
        sel = select_in_mask(STORE.scan_xyz, view, cam["fov"], cam["W"], cam["H"],
                             cap["masks"][mid], cap["depth"])
        out.append({"mask_id": mid,
                    "scan_indices_b64": base64.b64encode(sel.tobytes()).decode()})
    return {"instances": out}
```

Add helpers `_resize(mask,H,W)` (NEAREST resize to render size, like PoC), `_overlay_png(color, masks)` (multi-color wash → data-URL b64, palette from PoC `_palette`). Wrap `HTTPException(409, {...})` so the dict reaches `detail` — FastAPI serializes dict details as JSON. `resolve.py`: `raw_laz(scan_id)` and `scan_ply(scan_id)` resolve paths under `$VOXA_LIDAR_ROOT/annotated/<name>/` (strip an `annotated/` tier prefix if present). `run.sh`: `exec /home/hendrik/anaconda3/bin/python -m uvicorn main:app --host 127.0.0.1 --port 8011`.

- [ ] **Step 4: Run the test.** `pytest tests/test_api.py -v` → 3 PASS.
- [ ] **Step 5: Commit**

```bash
git add sam_sidecar/main.py sam_sidecar/resolve.py sam_sidecar/run.sh sam_sidecar/tests/test_api.py
git commit -m "feat(sam-sidecar): /capture + /project endpoints with identity guard"
```

---

## Task 7: voxa backend proxy (`/api/sam/*`)

**Files:**
- Create: `backend/routes/sam.py`
- Modify: `backend/app/schemas.py`, `backend/main.py` (line 35 import list)
- Test: `backend/tests/test_sam_proxy.py`

The proxy forwards to the sidecar (adding `recenter_offset` and the active session's `scan_id`/`source_fp`), then on `/project` applies each instance's indices via `apply_reassign(..., protect_instances=)`. Sidecar URL from `VOXA_SAM_SIDECAR_URL`; absent → 503 "SAM sidecar not configured".

- [ ] **Step 1: Add schemas** to `backend/app/schemas.py`:

```python
class SamCaptureRequest(BaseModel):
    camera: dict                      # {pos,target,fov,W,H} in the recentered frame
    mode: str                         # "box" | "concept"
    box: list[float] | None = None
    text: str | None = None

class SamProjectRequest(BaseModel):
    capture_id: str
    mask_ids: list[int]
    target_class: int | str
    protect_instances: list[int] = []
```

- [ ] **Step 2: Write the failing proxy test** (mock the sidecar HTTP with monkeypatch)

```python
# backend/tests/test_sam_proxy.py — conftest already sets VOXA_DATA_DIR before importing main
import base64, numpy as np, pytest
from fastapi.testclient import TestClient

# Assumes a fixture that loads a small annotated session into _state (reuse the
# existing session fixture used by test_segment / test_apply_shape). Selected
# indices are applied via apply_reassign; assert an unconfirmed pointset results.

def test_project_applies_indices_with_protection(monkeypatch, seg_session):
    import main
    from app import core
    core._state["scene"] = "scanA"; core._state["source_fp"] = "fp1"
    # fake sidecar responses
    idx = np.array([0,1,2], np.int32)
    def fake_post(url, json, **kw):
        class R:
            status_code = 200
            def json(self): 
                if url.endswith("/capture"): return {"capture_id":"c1","overlay_png_b64":"x","masks":[{"mask_id":0,"score":0.9}]}
                return {"instances":[{"mask_id":0,"scan_indices_b64":base64.b64encode(idx.tobytes()).decode()}]}
            def raise_for_status(self): pass
        return R()
    monkeypatch.setenv("VOXA_SAM_SIDECAR_URL", "http://side")
    monkeypatch.setattr("routes.sam.httpx.post", fake_post)
    c = TestClient(main.app)
    r = c.post("/api/sam/capture", json={"camera":{"pos":[0,0,0],"target":[0,0,1],"fov":60,"W":128,"H":128},"mode":"box","box":[0.5,0.5,0.4,0.4]})
    assert r.status_code == 200 and r.json()["capture_id"] == "c1"
    r2 = c.post("/api/sam/project", json={"capture_id":"c1","mask_ids":[0],"target_class":1,"protect_instances":[]})
    assert r2.status_code == 200
    assert r2.json()["instances"][0]["n_affected"] >= 0   # applied through apply_reassign

def test_missing_sidecar_url_503(monkeypatch, seg_session):
    monkeypatch.delenv("VOXA_SAM_SIDECAR_URL", raising=False)
    import main
    c = TestClient(main.app)
    r = c.post("/api/sam/capture", json={"camera":{"pos":[0,0,0],"target":[0,0,1],"fov":60,"W":128,"H":128},"mode":"box","box":[0.5,0.5,0.4,0.4]})
    assert r.status_code == 503
```

- [ ] **Step 3: Run to confirm failure.** `.venv/bin/pytest backend/tests/test_sam_proxy.py -v` → FAIL.

- [ ] **Step 4: Implement `backend/routes/sam.py`**

```python
"""Proxy to the SAM sidecar. /capture forwards the pose (+recenter_offset, scan
identity); /project forwards mask picks, then applies the returned scan indices
through the shared apply_reassign pipeline (protect_instances = confirmed = locked)."""
from __future__ import annotations
import base64, os
import numpy as np
import httpx
from fastapi import APIRouter, HTTPException

from app.core import _state, _require_seg
from app.schemas import SamCaptureRequest, SamProjectRequest
from routes.segment import _serialize_apply, _coerce_class_id  # reuse existing helpers

router = APIRouter()
_TIMEOUT = 120.0

def _sidecar_url() -> str:
    url = os.environ.get("VOXA_SAM_SIDECAR_URL")
    if not url:
        raise HTTPException(503, "SAM sidecar not configured (VOXA_SAM_SIDECAR_URL)")
    return url.rstrip("/")

def _identity() -> dict:
    return {"scan_id": _state.get("scene"), "source_fingerprint": _state.get("source_fp")}

@router.post("/api/sam/capture")
def sam_capture(req: SamCaptureRequest):
    off = _state.get("recenter_offset") or [0.0, 0.0, 0.0]
    cam = dict(req.camera)
    cam["pos"] = [p + o for p, o in zip(cam["pos"], off)]        # recentered → native
    cam["target"] = [p + o for p, o in zip(cam["target"], off)]
    body = {**_identity(), "camera": cam, "mode": req.mode, "box": req.box, "text": req.text}
    try:
        r = httpx.post(f"{_sidecar_url()}/capture", json=body, timeout=_TIMEOUT)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"SAM sidecar unreachable: {e}")
    if r.status_code == 409:
        raise HTTPException(409, r.json().get("detail"))
    r.raise_for_status()
    return r.json()

@router.post("/api/sam/project")
def sam_project(req: SamProjectRequest):
    seg = _require_seg()
    body = {**_identity(), "capture_id": req.capture_id, "mask_ids": req.mask_ids}
    try:
        r = httpx.post(f"{_sidecar_url()}/project", json=body, timeout=_TIMEOUT)
    except httpx.HTTPError as e:
        raise HTTPException(502, f"SAM sidecar unreachable: {e}")
    if r.status_code == 409:
        raise HTTPException(409, r.json().get("detail"))
    r.raise_for_status()
    target_class = _coerce_class_id(req.target_class)
    results = []
    for inst in r.json()["instances"]:
        idx = np.frombuffer(base64.b64decode(inst["scan_indices_b64"]), np.int32)
        out = seg.apply_reassign(idx, target_inst=-1, target_class=target_class,
                                 protect_instances=req.protect_instances)
        results.append({"mask_id": inst["mask_id"], **_serialize_apply(out)})
    return {"instances": results}
```

Confirm `_require_seg`, `_serialize_apply`, `_coerce_class_id` are importable from those modules (they are used in `routes/segment.py`); if any is module-private, lift it or duplicate minimally.

- [ ] **Step 5: Register the router** — `backend/main.py` line 35:

```python
from routes import compare, export, load, meta, sam, segment, sessions
```
(The loop at line 48 auto-includes every imported module's `router`.)

- [ ] **Step 6: Run the tests.** `.venv/bin/pytest backend/tests/test_sam_proxy.py -v` → PASS. Then full backend: `npm run test:backend`.
- [ ] **Step 7: Commit**

```bash
git add backend/routes/sam.py backend/app/schemas.py backend/main.py backend/tests/test_sam_proxy.py
git commit -m "feat(sam): voxa backend proxy → sidecar → apply_reassign"
```

---

## Task 8: Sidecar live smoke test (real cloud + real SAM)

**Files:** none (manual verification, no commit unless a fix is needed).

- [ ] **Step 1:** Install web deps once: `/home/hendrik/anaconda3/bin/python -m pip install -r sam_sidecar/requirements.txt`.
- [ ] **Step 2:** `cd sam_sidecar && VOXA_LIDAR_ROOT=/home/hendrik/coding/engine/data/lidar bash run.sh` — confirm startup, `curl 127.0.0.1:8011/health` → `{"ok":true,...}`.
- [ ] **Step 3:** Drive one `/capture` + `/project` for `smart_ais_navvis` with a hand-picked interior pose (reuse the design-session `persp_raw.py` pose). Confirm: `/capture` returns an overlay PNG that looks like the plant + ≥1 mask; `/project` returns a non-empty index array. First raw load builds the `.rawcache.npz` (~60s); note the warm per-frame latency (target: a few seconds).

---

## Task 9: Frontend — tool registration + api client + pure-fn utils

**Files:**
- Modify: `frontend/src/label-tools.js`, `frontend/src/api.js`
- Create: `frontend/src/sam-util.js`, `frontend/src/sam-util.test.js`

- [ ] **Step 1: Write the failing util test** (`sam-util.test.js`)

```javascript
import { describe, it, expect } from 'vitest';
import { normalizeBox, capturePayload } from './sam-util.js';

describe('normalizeBox', () => {
  it('CSS px rect → normalized [cx,cy,w,h] in canvas buffer space', () => {
    // canvas buffer 1000x800, CSS 500x400 (2x DPR); drag from (100,100)→(300,300) CSS
    const b = normalizeBox({ x0: 100, y0: 100, x1: 300, y1: 300 },
                           { clientWidth: 500, clientHeight: 400, width: 1000, height: 800 });
    expect(b).toEqual([0.4, 0.5, 0.4, 0.5]); // cx=(100+300)/2/500=0.4 ; w=200/500=0.4 ...
  });
});

describe('capturePayload', () => {
  it('assembles camera pose from a viewer view', () => {
    const view = { position: {toArray:()=>[1,2,3]}, getPivot:()=>({toArray:()=>[4,5,6]}) };
    const p = capturePayload({ view, fov: 60, canvas: { width: 1000, height: 800 },
                               mode: 'box', box: [0.4,0.5,0.4,0.5], text: null });
    expect(p.camera).toEqual({ pos:[1,2,3], target:[4,5,6], fov:60, W:1000, H:800 });
    expect(p.mode).toBe('box'); expect(p.box).toEqual([0.4,0.5,0.4,0.5]);
  });
});
```

- [ ] **Step 2: Run to confirm failure.** `cd frontend && npx vitest run src/sam-util.test.js` → FAIL.

- [ ] **Step 3: Implement `sam-util.js`**

```javascript
// Pure helpers for the SAM tool. No DOM/Three imports (unit-testable).
export function normalizeBox({ x0, y0, x1, y1 }, canvas) {
  // drag rect is in CSS px; normalize against CSS size (buffer scaling cancels out).
  const W = canvas.clientWidth, H = canvas.clientHeight;
  const lx = Math.min(x0, x1), hx = Math.max(x0, x1);
  const ly = Math.min(y0, y1), hy = Math.max(y0, y1);
  return [ (lx + hx) / 2 / W, (ly + hy) / 2 / H, (hx - lx) / W, (hy - ly) / H ];
}

export function capturePayload({ view, fov, canvas, mode, box, text }) {
  return {
    camera: { pos: view.position.toArray(), target: view.getPivot().toArray(),
              fov, W: canvas.width, H: canvas.height },
    mode, box: box ?? null, text: text ?? null,
  };
}
```

- [ ] **Step 4: Run the test.** → PASS.

- [ ] **Step 5: Register the tool** — `frontend/src/label-tools.js`, append to `TOOLS`:

```javascript
  { id: 'sam',        icon: '✦', label: 'SAM' },
```
Extend `toolAvailable`: `sam` needs a session **and** `raw_source_available` (thread the flag from the load response — see mode-label wiring, Task 11):
```javascript
  if (id === 'sam') return !!segState && !!isAnnotated && !!ctx.rawSourceAvailable;
```
(Add `rawSourceAvailable` to the ctx object passed to `toolAvailable`; default false.)

- [ ] **Step 6: Add api methods** — `frontend/src/api.js`, near `applyShape`:

```javascript
  async samCapture({ camera, mode, box = null, text = null }) {
    const r = await fetch('/api/sam/capture', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ camera, mode, box, text }),
    });
    if (!r.ok) throw new Error(`samCapture failed: ${r.status} ${await r.text()}`);
    return r.json();
  },
  async samProject({ captureId, maskIds, targetClass, protectInstances = [] }) {
    const r = await fetch('/api/sam/project', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ capture_id: captureId, mask_ids: maskIds,
                             target_class: targetClass, protect_instances: protectInstances }),
    });
    if (!r.ok) throw new Error(`samProject failed: ${r.status} ${await r.text()}`);
    return r.json();
  },
```

- [ ] **Step 7: Commit**

```bash
git add frontend/src/label-tools.js frontend/src/api.js frontend/src/sam-util.js frontend/src/sam-util.test.js
git commit -m "feat(sam): frontend tool registration, api client, pure-fn utils"
```

---

## Task 10: Frontend — SAM mode (box-drag capture + mask-review)

**Files:**
- Create: `frontend/src/sam-mode.jsx`
- Modify: `frontend/src/tool-options.jsx`

`sam-mode.jsx` owns the interaction, modeled on `beam-mode.jsx`/`draw-mode.jsx` (capture-phase key/mouse handling, a left-rail panel). No automated test (interaction/DOM heavy — covered by Task 12 browser verification); keep all pure logic in `sam-util.js`.

- [ ] **Step 1: Implement `sam-mode.jsx`** with:
  - **Panel state:** `mode` ('box'|'concept'), `text`, `capture` (`{captureId, overlayPng, masks}` | null), `chosen` (Set of mask_id), `autoConfirm`, `busy`, `error`.
  - **Box drag:** a canvas overlay `<div>` that, while `mode==='box'`, captures `mousedown`→`mousemove`→`mouseup` **only when Shift is held** (Shift suppresses orbit; plain drag falls through to the viewer's orbit). Draw a rubber-band rectangle. On mouseup: `normalizeBox` → `capturePayload({view: viewer.scene.view, fov: viewer.scene.cameraP.fov, canvas})` → `api.samCapture(...)` → set `capture`, preselect mask 0.
  - **Concept:** text input + **Segment all** button → `api.samCapture({mode:'concept', text})`.
  - **Mask-review panel:** show `overlayPng` (`<img>`, click-to-enlarge like PoC), and a checkbox list of `masks` (`mask_id` + score); box mode preselects the single mask, concept mode multi-select.
  - **Apply:** "Project selected" button → `api.samProject({captureId, maskIds:[...chosen], targetClass, protectInstances})`; on success clear `capture`, refresh the segment state so the new unconfirmed pointset(s) appear (call the same refresh the Beam/Draw apply uses — see how `beam-mode.jsx` triggers `onApplied`/segment reload). If `autoConfirm`, confirm the new instances via the existing confirm path.
  - **Errors fail loud:** any api throw → set `error`, render a blocking banner (match the preseg-divergence banner style); 409 → "scan changed on the sidecar — reload".
  - **Props:** `{ viewer, canvas, targetClass, protectInstances, autoConfirmDefault, onApplied }`.

- [ ] **Step 2: Wire into `tool-options.jsx`** — when `activeTool === 'sam'`, render `<SamPanel ... />` (the panel portion of `sam-mode.jsx`, following how `tool-options.jsx` already switches on `activeTool` for beam/draw). Pass `targetClass`, `protectInstances`, and the apply callbacks through, exactly as the Beam/Draw panels receive them.

- [ ] **Step 3: Build check.** `cd frontend && npx vite build` (or `npm run build`) → no errors. `npx vitest run` → existing + new pure-fn tests pass.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/sam-mode.jsx frontend/src/tool-options.jsx
git commit -m "feat(sam): box-drag capture + mask-review panel"
```

---

## Task 11: Frontend — mode-label wiring

**Files:**
- Modify: `frontend/src/mode-label.jsx`

- [ ] **Step 1:** Thread `rawSourceAvailable` (from the load response — `LoadResponse.raw_source_available`, already in App/cloud state) into the `toolAvailable` ctx so the SAM rail entry is gated/disabled with a tooltip when no raw cloud resolves.
- [ ] **Step 2:** Mount `<SamMode>` (the interaction half of `sam-mode.jsx`) when `activeTool === 'sam'`, passing `viewer`, the canvas element, `targetClass`, and `protectedSegIds` (the same `protectInstances` value already computed and threaded to Draw/Beam — see `mode-label.jsx::protectedSegIds`). SAM applies on the **raw RGB cloud** like Box/Draw/Beam (not preseg), so `isPreseg` stays false and `colorMode`/`segHulls` are unaffected.
- [ ] **Step 3:** After a SAM apply, reuse the exact post-apply refresh Draw/Beam use (reload segment state + reconcile the Instances panel rows) so the new unconfirmed pointset(s) render and become selectable/confirmable.
- [ ] **Step 4: Build + full frontend tests.** `npm run test:frontend` → green. `npm run build` → no errors.
- [ ] **Step 5: Commit**

```bash
git add frontend/src/mode-label.jsx
git commit -m "feat(sam): wire SAM mode into Label mode (gating, protect, refresh)"
```

---

## Task 12: End-to-end browser verification

**Files:** none (verification). Use the `browser-verification` skill (Chrome DevTools MCP). **Use a throwaway session** — Label apply auto-saves to disk (see memory `feedback_browser_verify_mutates_session`). Restart any stale `:8765` backend first.

- [ ] **Step 1:** Start the sidecar (`sam_sidecar/run.sh`, port 8011). Start voxa with `VOXA_SAM_SIDECAR_URL=http://127.0.0.1:8011 npm run dev`.
- [ ] **Step 2:** Open `http://127.0.0.1:5173`, load `smart_ais_navvis`, create a **new throwaway session**, enter Label mode, select the **SAM** tool.
- [ ] **Step 3 (box):** Shift-drag a box around a discrete vessel/pipe. Verify: `/api/sam/capture` 200; the mask-review panel shows a plausible overlay; accept → `/api/sam/project` 200; a new **unconfirmed** pointset appears on that object in 3D and in the Instances panel. Confirm it; verify confirmed-lock (a second overlapping box does not overwrite it — `n_protected>0`). Screenshot.
- [ ] **Step 4 (concept):** Switch to Concept, type "pipe", Segment all. Verify N masks in the review panel; select a few; Project → N unconfirmed pointsets, each its own hue. Delete one; confirm another. Screenshot.
- [ ] **Step 5:** Confirm **zero console errors** and all network calls succeed. Note warm latency per capture.
- [ ] **Step 6:** If issues found, fix on the branch (each fix its own TDD cycle where a test is possible) and re-verify.

---

## Task 13: Docs

**Files:**
- Modify: `CLAUDE.md`
- Create: `sam_sidecar/README.md`

- [ ] **Step 1:** `CLAUDE.md` — update the Label-mode description from a **4-tool rail** to a **5-tool rail**; add a SAM bullet (server-render → SAM → back-project to scan-res via `apply_reassign`; sidecar deployment; `VOXA_SAM_SIDECAR_URL`; raw-cloud requirement). Add `VOXA_SAM_SIDECAR_URL` and `VOXA_LIDAR_ROOT` (sidecar) to the env-vars section. Note the sidecar in the Architecture section.
- [ ] **Step 2:** `sam_sidecar/README.md` — what it is, the anaconda env + `pip install -r requirements.txt`, `run.sh`, the `/capture`+`/project` contract, the fingerprint identity guard, and that it must be reachable at `VOXA_SAM_SIDECAR_URL`.
- [ ] **Step 3:** Run `simplify` on the full diff (per global workflow) before finishing.
- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md sam_sidecar/README.md
git commit -m "docs(sam): 5-tool rail + sidecar run/contract docs"
```

---

## Done criteria

- All sidecar unit tests pass (`pytest sam_sidecar/tests -v`); voxa backend tests green (`npm run test:backend`); frontend tests green (`npm run test:frontend`); `npm run build` clean.
- Box + Concept verified end-to-end in-browser on `smart_ais_navvis` with zero console errors; confirmed-lock holds (`n_protected>0` on overlap).
- Sidecar fingerprint mismatch returns a loud 409 (never a silent wrong-cloud selection).
- Docs updated in the same PR. Then use `superpowers:finishing-a-development-branch`.
