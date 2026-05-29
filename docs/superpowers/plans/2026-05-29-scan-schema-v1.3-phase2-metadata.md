# Scan Schema v1.3 — Phase 2: Explicit Frames & Provenance Metadata Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Record frame identity and derivation provenance as data, and resolve a cloud↔render pairing deterministically (use-direct / remap-by-transform / fail) — so a frame mismatch is *known and corrected or refused*, not just empirically detected.

**Architecture:** Five small, pure modules + one linter CLI under the existing layout. Tested standalone with synthetic metas and tmp fixture scan dirs. Builds on Phase 1 (the resolver hands off to `registration` for the empirical confirmation). No rewiring of `scene_registry`/voxa-load yet — that integration is Phase 3.

**Tech stack:** Python 3.12, numpy, pytest (`pythonpath=["backend"]`). Targets the restructured layout (`backend/scenes/`, `backend/preseg/`).

> **Base-branch note:** continue on `worktree-scan-schema-v13` (already rebased onto local `main`, restructured layout). Phase-1 modules (`scenes/fingerprint.py`, `scenes/reproject.py`, `preseg/registration.py`) are present.

---

## File structure

| File | Responsibility |
|------|----------------|
| `backend/scenes/frame.py` (create) | `Frame` (4×4 `transform_to_canonical` + `georef` + `canonical_id`); `compose_a_to_v`, `apply_transform`, `is_rigid`, dict (de)serialization (§3.1, §4.1). |
| `backend/scenes/render_meta.py` (create) | read/write `renders/<run>/meta.json` (`generated_from` + `frame` + `intrinsics`) (§4.3). |
| `backend/scenes/scan_meta.py` (create) | `read_scan_meta(scan_dir)` → normalized dict with a `Frame`, synthesizing a translation-only `frame_uncertain` frame from v1.2 `coords`/`coord_offset_m` when absent (§4.1 back-compat). |
| `backend/preseg/resolver.py` (create) | `resolve_render_run(cloud_*, run_meta)` → `Resolution(action, transform, reasons)` per §5 (direct / remap / fail). |
| `scripts/validate_scan.py` (create) | Lint a scan dir against the v1.3 §7 invariants; exit non-zero on violation. |
| `backend/tests/test_frame.py`, `test_render_meta.py`, `test_scan_meta.py`, `test_resolver.py`, `test_validate_scan.py` (create) | Unit tests. |

Run: `.venv/bin/pytest backend/tests/<file>.py -v` from voxa root.

---

## Task 1: Frame model + transform composition

**Files:** Create `backend/scenes/frame.py`, `backend/tests/test_frame.py`

- [ ] **Step 1: Failing test**

```python
# backend/tests/test_frame.py
import numpy as np
import pytest
from scenes.frame import Frame, compose_a_to_v, apply_transform, is_rigid, frame_from_dict


def _rot_z(deg, t=(0, 0, 0)):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    M = np.eye(4)
    M[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
    M[:3, 3] = t
    return M


def test_compose_maps_A_into_V():  # Appendix A worked example
    A = Frame(_rot_z(90, (5, 2, 0)), "scan#local")
    V = Frame(np.eye(4), "scan#local")
    T = compose_a_to_v(A, V)
    p = apply_transform(T, np.array([[1.0, 0, 0]]))[0]
    assert np.allclose(p, [5, 3, 0], atol=1e-9)  # (1,0,0) rot90 -> (0,1,0), +t(5,2,0)


def test_compose_nonidentity_target_inverts():
    A = Frame(_rot_z(30, (1, 0, 0)), "s#local")
    V = Frame(_rot_z(30, (1, 0, 0)), "s#local")
    T = compose_a_to_v(A, V)
    assert np.allclose(T, np.eye(4), atol=1e-9)  # same frame -> identity


def test_is_rigid_rejects_nonorthonormal():
    bad = np.eye(4); bad[0, 0] = 2.0
    assert is_rigid(_rot_z(45, (3, 1, 2))) and not is_rigid(bad)


def test_roundtrip_dict():
    f = Frame(_rot_z(15, (2, 3, 4)), "s#local", georef={"crs": "EPSG:32632", "offset_m": [1, 2, 3]})
    g = frame_from_dict(f.to_dict())
    assert np.allclose(f.transform_to_canonical, g.transform_to_canonical) and g.canonical_id == "s#local"


def test_frame_from_dict_validates_shape():
    with pytest.raises(ValueError):
        frame_from_dict({"transform_to_canonical": [[1, 0], [0, 1]], "canonical_id": "x"})
```

- [ ] **Step 2: Run, verify fail** (`ModuleNotFoundError: scenes.frame`).

- [ ] **Step 3: Implement**

```python
# backend/scenes/frame.py
"""Coordinate-frame model for scan-schema v1.3 (§3.1, §4.1)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np


@dataclass
class Frame:
    transform_to_canonical: np.ndarray   # 4x4, maps this variant's coords -> canonical
    canonical_id: str                    # "<scan_id>#local"
    units: str = "meters"
    georef: Optional[dict] = None        # {"crs": ..., "offset_m": [x,y,z]}  canonical-local -> world
    frame_uncertain: bool = False        # synthesized from legacy coords -> force the §6 check

    def to_dict(self) -> dict:
        d = {"canonical_id": self.canonical_id,
             "transform_to_canonical": np.asarray(self.transform_to_canonical).tolist(),
             "units": self.units, "frame_uncertain": self.frame_uncertain}
        if self.georef is not None:
            d["georef"] = self.georef
        return d


def is_rigid(M, atol: float = 1e-6) -> bool:
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (4, 4):
        return False
    R = M[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=atol):
        return False
    if not np.isclose(abs(np.linalg.det(R)), 1.0, atol=atol):
        return False
    return np.allclose(M[3], [0, 0, 0, 1], atol=atol)


def frame_from_dict(d: dict) -> Frame:
    M = np.asarray(d["transform_to_canonical"], dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(f"transform_to_canonical must be 4x4, got {M.shape}")
    return Frame(M, d["canonical_id"], d.get("units", "meters"),
                 d.get("georef"), bool(d.get("frame_uncertain", False)))


def compose_a_to_v(a: Frame, v: Frame) -> np.ndarray:
    """4x4 mapping a point/pose expressed in frame `a` into frame `v`:
    apply a->canonical, then canonical->v  ==  inv(v.T_can) @ a.T_can."""
    return np.linalg.inv(v.transform_to_canonical) @ a.transform_to_canonical


def apply_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    return (T @ homo.T).T[:, :3]
```

- [ ] **Step 4: Run, verify pass** (5 passed).
- [ ] **Step 5: Commit** — `git commit -m "feat(scenes): Frame model + transform composition (v1.3 §3.1)"`

---

## Task 2: Render-run meta read/write

**Files:** Create `backend/scenes/render_meta.py`, `backend/tests/test_render_meta.py`

- [ ] **Step 1: Failing test**

```python
# backend/tests/test_render_meta.py
import numpy as np
from scenes.frame import Frame
from scenes.render_meta import write_render_meta, read_render_meta


def test_write_then_read_roundtrip(tmp_path):
    run = tmp_path / "lower"; run.mkdir()
    frame = Frame(np.eye(4), "navvis#local")
    write_render_meta(run, run_id="lower",
                      generated_from={"scan_id": "navvis", "variant_id": "aligned15M",
                                      "source_fingerprint": "sha256:abc", "n_points": 11820483},
                      frame=frame,
                      intrinsics={"fov_deg": 60, "fov_axis": "vertical", "aspect": 1.169,
                                  "width": 926, "height": 792})
    m = read_render_meta(run)
    assert m["run_id"] == "lower"
    assert m["generated_from"]["variant_id"] == "aligned15M"
    assert m["intrinsics"]["fov_deg"] == 60
    assert isinstance(m["frame"], Frame)
    assert np.allclose(m["frame"].transform_to_canonical, np.eye(4))


def test_read_missing_returns_none(tmp_path):
    assert read_render_meta(tmp_path / "nope") is None
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement**

```python
# backend/scenes/render_meta.py
"""Per-render-run provenance meta (renders/<run>/meta.json), scan-schema v1.3 §4.3."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional
from scenes.frame import Frame, frame_from_dict

META_NAME = "meta.json"   # sibling to manifest.json


def write_render_meta(run_dir: Path, *, run_id: str, generated_from: dict,
                      frame: Frame, intrinsics: dict, generated_at: str = "",
                      n_frames: Optional[int] = None) -> Path:
    doc = {"run_id": run_id, "generated_from": generated_from,
           "frame": frame.to_dict(), "intrinsics": intrinsics,
           "generated_at": generated_at}
    if n_frames is not None:
        doc["n_frames"] = n_frames
    path = Path(run_dir) / META_NAME
    path.write_text(json.dumps(doc, indent=2))
    return path


def read_render_meta(run_dir: Path) -> Optional[dict]:
    path = Path(run_dir) / META_NAME
    if not path.exists():
        return None
    doc = json.loads(path.read_text())
    if "frame" in doc:
        doc["frame"] = frame_from_dict(doc["frame"])
    return doc
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(scenes): render-run meta.json read/write (v1.3 §4.3)"`

---

## Task 3: Scan meta reader with v1.2 back-compat

**Files:** Create `backend/scenes/scan_meta.py`, `backend/tests/test_scan_meta.py`

**Context:** Existing `meta.json` has `coords` (`"world"|"world_minus_offset"|"recenter:x,y,z"`) + optional `coord_offset_m`, and no `frame`. The reader normalizes to a `Frame`: native v1.3 `frame` if present; else a **translation-only, `frame_uncertain=True`** frame (the cloud is treated as its own canonical-local; `georef.offset_m` recovers world). `frame_uncertain` is the signal that the §6 health-check is mandatory before trusting it.

- [ ] **Step 1: Failing test**

```python
# backend/tests/test_scan_meta.py
import json
import numpy as np
from scenes.scan_meta import read_scan_meta


def _write(scan, meta):
    (scan / "meta.json").write_text(json.dumps(meta))


def test_v13_frame_used_directly(tmp_path):
    _write(tmp_path, {"schema_version": "1.3",
                      "frame": {"canonical_id": "s#local",
                                "transform_to_canonical": np.eye(4).tolist(),
                                "units": "meters"},
                      "derivation": {"scan_id": "s", "variant_id": "v", "varies": ["density"]}})
    m = read_scan_meta(tmp_path)
    assert m["frame"].canonical_id == "s#local" and not m["frame"].frame_uncertain
    assert m["derivation"]["variant_id"] == "v"


def test_legacy_coords_synthesizes_uncertain_frame(tmp_path):
    _write(tmp_path, {"scan_name": "navvis", "coords": "world_minus_offset",
                      "coord_offset_m": [574184.0, 6220868.0, 49.0], "class_map_version": 1})
    m = read_scan_meta(tmp_path)
    f = m["frame"]
    assert f.frame_uncertain                                  # forces the §6 check
    assert np.allclose(f.transform_to_canonical, np.eye(4))   # stored = its own canonical-local
    assert f.georef["offset_m"] == [574184.0, 6220868.0, 49.0]
    assert m["derivation"]["scan_id"] == "navvis"             # derived from scan_name
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement**

```python
# backend/scenes/scan_meta.py
"""Scan/variant meta.json reader with v1.2 -> v1.3 back-compat (§4.1)."""
from __future__ import annotations
import json
from pathlib import Path
from scenes.frame import Frame, frame_from_dict


def _legacy_frame(meta: dict) -> Frame:
    scan_id = meta.get("scan_name", "unknown")
    offset = meta.get("coord_offset_m")
    georef = {"offset_m": offset} if offset else None
    # We don't know the aligned canonical for legacy clouds: treat the stored cloud
    # as its own canonical-local (identity) and flag it uncertain so the §6 check runs.
    return Frame(__import__("numpy").eye(4), f"{scan_id}#local",
                 units=meta.get("units", "meters"), georef=georef, frame_uncertain=True)


def read_scan_meta(scan_dir: Path) -> dict:
    meta = json.loads((Path(scan_dir) / "meta.json").read_text())
    if "frame" in meta:
        meta["frame"] = frame_from_dict(meta["frame"])
    else:
        meta["frame"] = _legacy_frame(meta)
    if "derivation" not in meta:
        meta["derivation"] = {
            "scan_id": meta.get("scan_name", "unknown"),
            "variant_id": meta.get("scan_name", "unknown"),
            "parent": "original", "op": "asis", "varies": [],
            "source_fingerprint": None, "role": None,
        }
    return meta
```

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(scenes): scan meta reader + v1.2 coords back-compat (v1.3 §4.1)"`

---

## Task 4: Render-pairing resolver (§5)

**Files:** Create `backend/preseg/resolver.py`, `backend/tests/test_resolver.py`

**Context:** Given a cloud variant and a render run's meta, decide how to consume the poses: use directly, remap by transform, or refuse. This is the navvis decision made explicit. (Per-point cross-variant reuse / propagation §5.3–5.4 is Phase 3.)

- [ ] **Step 1: Failing test**

```python
# backend/tests/test_resolver.py
import numpy as np
from scenes.frame import Frame
from preseg.resolver import resolve_render_run


def _runmeta(variant_id, fp, T):
    return {"generated_from": {"variant_id": variant_id, "scan_id": "navvis",
                               "source_fingerprint": fp},
            "frame": Frame(T, "navvis#local")}


def test_direct_when_variant_and_fp_match():
    cf = Frame(np.eye(4), "navvis#local")
    r = resolve_render_run(cf, "v1", "sha256:a", _runmeta("v1", "sha256:a", np.eye(4)))
    assert r.action == "use_direct" and np.allclose(r.transform, np.eye(4))


def test_remap_when_same_scan_diff_frame():
    cf = Frame(np.eye(4), "navvis#local")                       # cloud at canonical
    T = np.eye(4); T[:3, 3] = [20, 2, -8]                       # render frame offset
    r = resolve_render_run(cf, "v_unaligned", "sha256:b", _runmeta("aligned15M", "sha256:c", T))
    assert r.action == "remap" and np.allclose(r.transform, T)  # maps render poses into cloud frame


def test_fail_when_different_scan():
    cf = Frame(np.eye(4), "navvis#local")
    run = _runmeta("v1", "sha256:a", np.eye(4)); run["frame"] = Frame(np.eye(4), "OTHER#local")
    run["generated_from"]["scan_id"] = "other"
    r = resolve_render_run(cf, "v1", "sha256:a", run)
    assert r.action == "fail" and r.reasons


def test_fail_when_pin_missing():
    cf = Frame(np.eye(4), "navvis#local")
    r = resolve_render_run(cf, "v1", "sha256:a", {"frame": cf, "generated_from": {}})
    assert r.action == "fail"
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement**

```python
# backend/preseg/resolver.py
"""Resolve a cloud variant <-> render-run pairing (scan-schema v1.3 §5)."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scenes.frame import Frame, compose_a_to_v


@dataclass
class Resolution:
    action: str                          # "use_direct" | "remap" | "fail"
    transform: Optional[np.ndarray] = None   # maps render poses INTO the cloud frame
    reasons: list = field(default_factory=list)


def resolve_render_run(cloud_frame: Frame, cloud_variant_id: str,
                       cloud_fingerprint: str, run_meta: dict) -> Resolution:
    gf = run_meta.get("generated_from") or {}
    src_variant = gf.get("variant_id")
    src_fp = gf.get("source_fingerprint")
    run_frame = run_meta.get("frame")
    if not src_variant or run_frame is None:
        return Resolution("fail", reasons=["render run has no source-variant/frame pin"])

    # 1. exact match -> use poses directly
    if src_variant == cloud_variant_id and (src_fp is None or src_fp == cloud_fingerprint):
        return Resolution("use_direct", transform=np.eye(4))

    # 2. same scan (same canonical frame id) -> remap poses by the composed transform
    if run_frame.canonical_id == cloud_frame.canonical_id:
        T = compose_a_to_v(run_frame, cloud_frame)
        return Resolution("remap", transform=T)

    # 3. different scan / unknown -> refuse
    return Resolution("fail", reasons=[
        f"render run variant '{src_variant}' (frame {run_frame.canonical_id}) is not the "
        f"cloud '{cloud_variant_id}' (frame {cloud_frame.canonical_id}) and not the same scan"])
```

- [ ] **Step 4: Run, verify pass.** **Note:** callers MUST still run the Phase-1 `registration` check after a `remap` before trusting it (metadata can be wrong) — document this in the resolver docstring.
- [ ] **Step 5: Commit** — `git commit -m "feat(preseg): render-run pairing resolver — direct/remap/fail (v1.3 §5)"`

---

## Task 5: `validate_scan.py` invariant linter (§7)

**Files:** Create `scripts/validate_scan.py`, `backend/tests/test_validate_scan.py`

**Context:** Lint a scan dir against the v1.3 §7 invariants and exit non-zero on violation. Factor the checks into a pure `validate_scan_dir(scan_dir) -> list[str]` (returns violations) so it's unit-testable; the CLI wraps it.

- [ ] **Step 1: Failing test**

```python
# backend/tests/test_validate_scan.py
import json
import numpy as np
from validate_scan import validate_scan_dir   # scripts/ on sys.path via the test


def _scan(tmp_path, meta, n=100):
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
    m = _good_meta(); m["frame"]["transform_to_canonical"] = [[1, 0], [0, 1]]
    v = validate_scan_dir(_scan(tmp_path, m))
    assert any("transform" in x for x in v)


def test_bad_varies_flagged(tmp_path):
    m = _good_meta(); m["derivation"]["varies"] = ["density", "bogus"]
    v = validate_scan_dir(_scan(tmp_path, m))
    assert any("varies" in x for x in v)
```

Add `scripts` to the test's import path (top of test file):
```python
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts"))
```

- [ ] **Step 2: Run, verify fail.**
- [ ] **Step 3: Implement** `scripts/validate_scan.py` with `validate_scan_dir(scan_dir) -> list[str]` checking: `schema_version` present; if ≥"1.3" `frame`+`derivation` present and `is_rigid(transform_to_canonical)`; `canonical_id == scan_id + "#local"`; `derivation.varies ⊆ {density,frame,points,color,attributes}`; ≤1 labeling role; each `renders/<run>/` has a `meta.json` with non-blank `generated_from.variant_id`. CLI: `main()` prints violations and returns `0`/`1`. Reuse `scenes.frame.is_rigid` and `scenes.scan_meta.read_scan_meta` (insert `backend` on `sys.path` like the other scripts).

- [ ] **Step 4: Run, verify pass.**
- [ ] **Step 5: Commit** — `git commit -m "feat(scripts): validate_scan linter for v1.3 invariants (§7)"`

---

## Done criteria

- `.venv/bin/pytest backend/tests/test_frame.py test_render_meta.py test_scan_meta.py test_resolver.py test_validate_scan.py -v` all pass; full suite still green.
- `read_scan_meta` returns a usable `Frame` for both a v1.3 scan and a legacy v1.2 `coords` scan (the latter `frame_uncertain=True`).
- `resolve_render_run` returns `use_direct`/`remap`(with correct transform)/`fail` for the three cases; the navvis case (cloud unaligned, run aligned, same scan) → `remap`.
- `validate_scan.py` flags a malformed frame/`varies`.

## Not in this plan (Phase 3)

- Wiring the resolver + `read_scan_meta` into `scene_registry`/voxa load and the SAM3 stage (replace the ad-hoc `coords` handling); making `remap` actually transform render poses in the pipeline and run the §6 check.
- `variants.json` generator (`scripts/scan_index.py`) + cross-variant fingerprint resolution.
- Multi-run `labels/runs/` + `prelabel/runs/` and Compare/merge (§4.6); label propagation (§5.4).
- Writer updates across voxa export + walker render export; backfilling navvis's real `aligned15M` transform; replacing `lidar/SCHEMA.md`.
