# SAM sidecar frame-offset fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the SAM sidecar frame mismatch for `smart_ais_navvis` (option B from the 2026-07-13
blocker: recover/record the true UTM offset rather than switching to rendering `scan.ply`) so
`/api/sam/capture` + `/api/sam/project` select real points instead of 0.

**Architecture:** `smart_ais_navvis/meta.json` already has the true offset — it's just sitting in
the ad-hoc `sample_param.recenter_applied_m` field instead of the schema's sanctioned
`frame.georef.offset_m` (which `scan_schema.metadata.read_scan_meta`/`Frame` already parse and which
voxa's `scan_meta.py::frame_summary` already surfaces as `georef_offset` — it's just not threaded
through scene discovery → SAM route → sidecar yet). Root cause confirmed in code:
- `backend/app/core.py::_recenter` only fires when `max(|centroid|) >= 1e3`; `scan.ply` for this
  scan is already pre-recentered on disk (~20m magnitude) so it computes offset `[0,0,0]` — it was
  never meant to recover a *baked-in* offset, only to recenter raw-magnitude coords at load time.
- `backend/routes/sam.py::sam_capture` adds only that (zero) offset to the camera pose before
  forwarding to the sidecar, so the pose stays in `scene_local` while the sidecar's raw render uses
  `STORE.raw_xyz` loaded straight from the UTM `.laz` — camera and cloud are ~88070/434901m apart.
- Less obvious second half of the bug: `sam_sidecar/main.py::project` back-projects using
  `STORE.scan_xyz` (loaded from `scan.ply`, also `scene_local`) against the *raw* camera's depth
  buffer — so even once capture's camera pose is fixed, `scan_xyz` itself must also be shifted into
  the raw/UTM frame for the occlusion test in `select_in_mask` to be geometrically valid. Point
  *indices* returned are positional or so unaffected by translation — no downstream index-mapping
  fix needed.

**Scope confirmed:** only `smart_ais_navvis` is affected (1 of 10 annotated scans) — other navvis
scans keep `scan.ply` in `world`/`world_minus_offset` coords where the live `_recenter()` already
computes a correct offset. `scan.raw_full_156M.ply` / the raw `.laz` are both intact on disk, so
nothing is unrecoverable; this is a plumbing fix, not a data-recovery job.

**Tech Stack:** FastAPI (voxa backend + sidecar), `scan_schema` (shared VCS dependency), pytest.

---

### Task 1: Move the offset into the schema's canonical field

**Files:**
- Modify: `/home/hendrik/coding/engine/data/lidar/annotated/smart_ais_navvis/meta.json`

- [ ] **Step 1: Add `frame.georef`**

Edit the `"frame"` block to add a `georef` key (keep `transform_to_canonical` identity — the offset
is a georeference, not a rigid transform of the stored points):

```json
"frame": {
  "canonical_id": "smart_ais_navvis#local",
  "frame_uncertain": false,
  "transform_to_canonical": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
  "units": "meters",
  "georef": {"crs": "utm", "offset_m": [88070.475, 434901.547, 14.298]}
}
```

Leave `sample_param.recenter_applied_m` in place too (harmless, documents provenance) — just stop
treating it as the only copy.

- [ ] **Step 2: Validate with scan_schema**

Run: `cd /home/hendrik/coding/engine/tools/labeling/voxa && .venv/bin/python -c "
from scan_schema.metadata import check_meta
import json
m = json.load(open('/home/hendrik/coding/engine/data/lidar/annotated/smart_ais_navvis/meta.json'))
print(check_meta(m))
"`
Expected: `([], [])` — no errors, no warnings (unchanged from before this edit).

- [ ] **Step 3: Commit**

This is data outside the git repo (lidar archive), so no `git add` here — just verify the file is
valid JSON and move on. Note in the plan-runner's summary that this step touched
`$VOXA_LIDAR_ROOT`, not the voxa repo.

---

### Task 2: Surface the georef offset from scene discovery

**Files:**
- Modify: `backend/scenes/scene_registry.py`
- Test: `backend/tests/test_scene_registry.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_scene_registry.py` already has `_make_annotated(scan_dir, ..., schema_version=)`
(writes a minimal v2.0 `meta.json` with no `frame` block — fine for grandfathered 2.x scans) and a
`lidar_root` fixture that calls it, plus `discover(voxa_data, lidar_root)` to run discovery. Neither
`_make_annotated` nor the fixtures know about `frame`/`georef`, and `_make_annotated` always writes
`schema_version="2.0"` by default (a v3.0 scan requires a full valid `frame` block per
`scan_schema.metadata.check_meta`, so don't reuse `_make_annotated` unmodified for this — write the
meta.json directly in the new test, following its structure):

```python
def test_georef_offset_surfaced_in_extras(tmp_path, voxa_data):
    scan_dir = tmp_path / "lidar" / "annotated" / "smart_ais_test"
    _write_tiny_ply(scan_dir / "source" / "scan.ply", n=8)
    (scan_dir / "sessions" / "s0").mkdir(parents=True, exist_ok=True)
    (scan_dir / "meta.json").write_text(json.dumps({
        "scan_name": "smart_ais_test", "n_points": 8, "units": "meters",
        "class_map_version": 1, "schema_version": "3.0",
        "derivation": {"scan_id": "smart_ais_test", "variant_id": "smart_ais_test",
                       "varies": [], "role": None},
        "frame": {
            "canonical_id": "smart_ais_test#local", "frame_uncertain": False,
            "transform_to_canonical": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
            "units": "meters", "georef": {"crs": "utm", "offset_m": [1.0, 2.0, 3.0]},
        },
    }))
    (tmp_path / "lidar" / "classes.json").write_text(json.dumps({
        "version": 1, "unlabeled_id": -1, "classes": [{"id": 0, "name": "pipe"}],
    }))
    scenes = discover(voxa_data, tmp_path / "lidar")
    src = next(s for s in scenes if s.name == "smart_ais_test")
    assert src.extras["raw_georef_offset_m"] == [1.0, 2.0, 3.0]

def test_georef_offset_absent_defaults_none(voxa_data, lidar_root):
    scenes = discover(voxa_data, lidar_root)   # munich_water_pump has no frame/georef
    src = next(s for s in scenes if s.name == "munich_water_pump")
    assert src.extras.get("raw_georef_offset_m") is None
```

Check `SceneSource` — confirm whether it exposes `.name` or only `.scene_id`/tier-prefixed ids;
adjust the lookup (`s.name` vs parsing `s.scene_id`) to match reality before running.

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest backend/tests/test_scene_registry.py -k georef -v`
Expected: FAIL — `raw_georef_offset_m` not in extras.

- [ ] **Step 3: Implement**

In `backend/scenes/scene_registry.py`, near the existing `source_laz_path` resolution block
(~line 133-149), add after `meta` is loaded and validated:

```python
raw_georef_offset_m = ((meta.get("frame") or {}).get("georef") or {}).get("offset_m")
```

and add `"raw_georef_offset_m": raw_georef_offset_m` to the `extras={...}` dict passed into
`SceneSource(...)` (alongside `"source_laz_path": source_laz_path`).

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest backend/tests/test_scene_registry.py -k georef -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add backend/scenes/scene_registry.py backend/tests/test_scene_registry.py
git commit -m "feat(sam): surface frame.georef.offset_m from scene discovery"
```

---

### Task 3: Thread the offset into `_state` at load time

**Files:**
- Modify: `backend/routes/load.py`
- Modify: `backend/app/core.py` (only if `_state` needs a new default key)
- Test: `backend/tests/test_load.py` (or wherever load-route tests live — check first)

- [ ] **Step 1: Write the failing test**

Find the existing test that asserts `_state["recenter_offset"]` after `/api/load` on an annotated
scene (search `grep -rn "recenter_offset" backend/tests`) and add a sibling assertion for a scene
whose `extras["raw_georef_offset_m"]` is set — assert `_state["raw_georef_offset_m"] == [...]`
after load, and `[0.0, 0.0, 0.0]` (or `None`) for a scene without it — match whatever sentinel the
rest of `_state` uses for "absent" (check `core.py:46` default dict first).

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest backend/tests/test_load.py -k georef -v` (adjust path)
Expected: FAIL — key missing or wrong.

- [ ] **Step 3: Implement**

In `backend/app/core.py`, add a default alongside line 46 (`"recenter_offset": [0.0, 0.0, 0.0]`):
```python
"raw_georef_offset_m": [0.0, 0.0, 0.0],
```

In `backend/routes/load.py`, right after `src = _resolve(req.name)` (top of `load_scene`), set:
```python
_state["raw_georef_offset_m"] = src.extras.get("raw_georef_offset_m") or [0.0, 0.0, 0.0]
```
(Read the surrounding function first — `_state` may be reset/rebuilt elsewhere in `load_scene`;
place this assignment after whatever full-state reset happens, not before, so it isn't clobbered.)

- [ ] **Step 4: Run test to verify it passes**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Run full backend suite**

Run: `.venv/bin/pytest backend/tests -q`
Expected: all green, no regressions (this touches shared `_state` init).

- [ ] **Step 6: Commit**

```bash
git add backend/app/core.py backend/routes/load.py backend/tests/
git commit -m "feat(sam): thread raw_georef_offset_m into _state on load"
```

---

### Task 4: Combine offsets in `sam_capture` and forward to the sidecar

**Files:**
- Modify: `backend/routes/sam.py`
- Test: `backend/tests/test_sam_proxy.py`

**Axis-order warning (found in plan review):** `_state["recenter_offset"]` is computed by
`_recenter()` in `load.py` **after** `_z_up_to_y_up()` has already run — so it's in **Y-up** axis
order (X right, Y up, Z back), matching the viewer's camera (`cam.pos`/`cam.target` are always
Y-up). But `frame.georef.offset_m` (from Task 1) is copied straight from
`sample_param.recenter_applied_m`, which is in the **raw LAZ's native Z-up** order (easting,
northing, altitude — the `14.298` component is an altitude, not a Y-up "up" value). `scan.ply` for
this scan is Z-up per `scan_meta.py::is_z_up_from_meta` (has `source_laz`, no `source_mesh`). Do
**not** element-wise sum `recenter` (Y-up) and `georef` (Z-up) directly — rotate `georef` into Y-up
first using the same mapping as `_z_up_to_y_up`: `(x, y, z)_zup → (x, z, -y)_yup`, gated on whether
this scene is actually Z-up (`src.extras.get("is_z_up")`, already resolved in `sam_capture` via
`src = _resolve(...)`). If a future scan's `scan.ply` were Y-up (sampled from a mesh), `georef`
would already be in the matching frame and should NOT be rotated.

- [ ] **Step 1: Write the failing test**

Add to `backend/tests/test_sam_proxy.py`, using the existing `_fake_post`/`client_with_loaded_annotated_scene`
pattern. You need to (a) monkeypatch `_state["raw_georef_offset_m"]` to a nonzero value, (b) make
`_Src.extras` report `is_z_up=True`, and (c) capture what `sam_capture` actually POSTs, to assert
both the rotated+shifted camera pose and a new `scan_ply_offset_m` field (unrotated, raw order) in
the outgoing body:

```python
def test_capture_applies_georef_offset_zup(client_with_loaded_annotated_scene, monkeypatch):
    client = client_with_loaded_annotated_scene
    monkeypatch.setenv("VOXA_SAM_SIDECAR_URL", "http://side")
    class _SrcZUp(_Src):
        extras = {"source_laz_path": "/x.laz", "is_z_up": True}
    monkeypatch.setattr("routes.sam._resolve", lambda scene: _SrcZUp())
    import routes.sam as sam_route
    monkeypatch.setitem(sam_route._state, "raw_georef_offset_m", [100.0, 200.0, 5.0])  # x,y,z zup
    monkeypatch.setitem(sam_route._state, "recenter_offset", [1.0, 1.0, 1.0])           # x,y,z yup
    sent = {}
    def _capture_post(url, json, **kw):
        sent.update(json)
        return _fake_post(url, json, **kw)
    monkeypatch.setattr("routes.sam.httpx.post", _capture_post)
    cam = {"pos": [0,0,0], "target": [0,0,1], "fov": 60, "W": 128, "H": 128}
    client.post("/api/sam/capture", json={"camera": cam, "mode": "box", "box": [0.5,0.5,0.4,0.4]})
    # georef [100,200,5]_zup -> [100,5,-200]_yup, plus recenter [1,1,1]_yup -> [101, 6, -199]
    assert sent["camera"]["pos"] == [101.0, 6.0, -199.0]
    assert sent["scan_ply_offset_m"] == [100.0, 200.0, 5.0]   # sidecar wants raw/native order
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest backend/tests/test_sam_proxy.py -k georef -v`
Expected: FAIL (`KeyError: 'scan_ply_offset_m'` or wrong pos).

- [ ] **Step 3: Implement**

In `backend/routes/sam.py::sam_capture`, replace:
```python
off = _state.get("recenter_offset") or [0.0, 0.0, 0.0]
```
with:
```python
recenter = _state.get("recenter_offset") or [0.0, 0.0, 0.0]      # already Y-up
georef = _state.get("raw_georef_offset_m") or [0.0, 0.0, 0.0]    # native frame (Z-up if is_z_up)
src = _resolve(_state.get("scene"))
georef_yup = [georef[0], georef[2], -georef[1]] if src.extras.get("is_z_up") else georef
off = [r + g for r, g in zip(recenter, georef_yup)]
```
(Move the existing `src = _resolve(_state.get("scene"))` line, currently a few lines below, up to
here so it isn't computed twice — read the full function before editing.) Keep the existing
`cam["pos"]`/`cam["target"]` lines unchanged (they already use `off`). Add
`"scan_ply_offset_m": georef` (native/raw order, NOT `georef_yup`) to the `body` dict sent to the
sidecar — the sidecar's `scan_xyz` is loaded straight from `scan.ply` with no rotation applied
(same native frame as `raw_xyz`), so it needs the un-rotated offset. `recenter_offset` never
applies to `scan_xyz` because that value only corrects for voxa's own in-memory load-time
recentering of the *display* copy of the cloud — it has no meaning on the sidecar's independently-
loaded `scan_xyz`.

- [ ] **Step 4: Run test to verify it passes**

Run the same pytest command. Expected: PASS.

- [ ] **Step 5: Run full test_sam_proxy.py to check no regressions**

Run: `.venv/bin/pytest backend/tests/test_sam_proxy.py -v`
Expected: all green (existing tests don't set `raw_georef_offset_m`, so it defaults to
`[0,0,0]` and behavior is unchanged for them).

- [ ] **Step 6: Commit**

```bash
git add backend/routes/sam.py backend/tests/test_sam_proxy.py
git commit -m "feat(sam): combine recenter + georef offsets, forward georef to sidecar"
```

---

### Task 5: Sidecar applies the offset to `scan_xyz`

**Files:**
- Modify: `sam_sidecar/main.py`
- Modify: `sam_sidecar/scan_store.py`
- Test: `sam_sidecar/tests/test_api.py`, `sam_sidecar/tests/test_scan_store.py`

- [ ] **Step 1: Write the failing test (scan_store)**

Read `sam_sidecar/tests/test_scan_store.py` first to match its style, then add a case asserting
`ScanStore.ensure(..., scan_ply_offset_m=[10,0,0])` shifts `self.scan_xyz` by that offset relative
to whatever the loader returned (compare against a captured pre-offset copy).

- [ ] **Step 2: Write the failing test (API)**

In `sam_sidecar/tests/test_api.py`, add a case where the fake loader's `scan_xyz` is offset from
`raw_xyz`/the camera by a known translation, `scan_ply_offset_m` is passed in `/capture`, and
`/project` still returns a non-empty selection (i.e. without the fix, `select_in_mask` against the
un-shifted `scan_xyz` would land outside the mask/behind the depth buffer and return an empty or
wrong selection — construct the fixture so the offset actually matters, e.g. place `wall` at a
translated position and only apply the SAM mask/box over the translated location).

- [ ] **Step 3: Run tests to verify they fail**

Run: `.venv/bin/pytest sam_sidecar/tests/ -k offset -v` (adjust env — sidecar has its own venv per
`sam_sidecar/README.md`)
Expected: FAIL.

- [ ] **Step 4: Implement**

`sam_sidecar/scan_store.py::ensure` — add param and apply it once, right after the loader call:
```python
def ensure(self, scan_id: str, fingerprint: str, raw_laz_path: str = None,
           scan_ply_path: str = None, scan_ply_offset_m=None) -> None:
    if self.scan_id == scan_id:
        if self.fingerprint != fingerprint:
            raise FingerprintMismatch(...)
        return
    self.raw_xyz, self.raw_rgb, self.scan_xyz = self._loader(scan_id, raw_laz_path, scan_ply_path)
    if scan_ply_offset_m:
        self.scan_xyz = self.scan_xyz + np.asarray(scan_ply_offset_m, dtype=np.float32)
    self.scan_id = scan_id; self.fingerprint = fingerprint
```
(add `import numpy as np` at top if not already present).

`sam_sidecar/main.py`:
- Add `scan_ply_offset_m: list[float] = [0.0, 0.0, 0.0]` to `CaptureReq`.
- Update `_ensure(scan_id, fp, raw=None, ply=None, offset=None)` to forward the offset to
  `STORE.ensure(scan_id, fp, raw, ply, offset)`.
- In the `capture()` handler, pass `req.scan_ply_offset_m` through to `_ensure(...)`.

- [ ] **Step 5: Run tests to verify they pass**

Run: `.venv/bin/pytest sam_sidecar/tests/ -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add sam_sidecar/main.py sam_sidecar/scan_store.py sam_sidecar/tests/
git commit -m "fix(sam-sidecar): shift scan_xyz into raw frame via scan_ply_offset_m"
```

---

### Task 6: Re-run the real smoke test (manual, needs GPU env)

**Files:** none (verification only)

- [ ] **Step 1**

With the sidecar running on its GPU/anaconda env (`sam_sidecar/README.md`), run:
```bash
python sam_sidecar/smoke.py
```
against `smart_ais_navvis`. Expected: `/capture` still finds a high-confidence mask, and `/project`
now returns a non-zero point selection (previously 0). This is the exact regression the 2026-07-13
blocker (T8) hit — confirm it's fixed before touching the browser.

- [ ] **Step 2**

If it passes, proceed to T12 (browser e2e on `smart_ais_navvis`, per
`docs/superpowers/plans/2026-07-12-sam-labeling-tool-plan.md` or wherever T12 is tracked) — not part
of this plan's scope, just the natural next step.

---

## Notes for the plan runner

- Tasks 2-5 are independent enough to TDD one at a time but have a linear dependency chain (2→3→4→5)
  since each threads a value the next one consumes — execute in order.
- `scan.raw_full_156M.ply` sitting in `scratch/` instead of `source/` and the `crs: "utm"` string
  being non-specific (no zone) are both pre-existing loose ends, not blockers for this fix — leave
  them alone unless the user asks for a broader data-hygiene pass.
