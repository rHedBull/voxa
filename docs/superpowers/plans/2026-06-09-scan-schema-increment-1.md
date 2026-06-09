# scan-schema Increment 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the `scan_schema` package as the single executable definition of the lidar scan schema (v3.0), replace the three drifting validators with one, have voxa adopt it without behavior change, rename `lidar/laz/`→`lidar/raw/`, and rewrite `lidar/SCHEMA.md` to v3.0.

**Architecture:** A new standalone git repo `engine/tools/scan-schema/` ships a Python package `scan_schema` (pure path layout + frame model + GT invariants + meta.json contract + a `Storage` seam + a whole-archive validator). voxa adopts it via **thin re-export shims** at the old import paths (`scenes/scan_layout.py`, `scenes/frame.py`) so the 21 existing import sites are untouched; its save-gate delegates to `scan_schema.invariants`. Existing 2.0 scans are grandfathered (read with warnings, never rejected).

**Tech Stack:** Python 3.12, numpy, pytest, `pip install -e`. Lifts from voxa `backend/scenes/scan_layout.py`, `backend/scenes/frame.py`, `backend/labeling/segment_io.py`, `scripts/scan/validate_scan.py`.

**Spec:** `docs/superpowers/specs/2026-06-09-scan-schema-enforcement-design.md`

**Cross-tree note:** this plan touches three trees — the new `engine/tools/scan-schema/` repo (git), the voxa repo (git, `engine/tools/labeling/voxa/`), and the non-git data archive (`engine/data/lidar/`). Commit commands target the relevant repo; data-archive changes are not committed (note them in the voxa-side task that depends on them).

**Scope (this increment):** package modules `layout`, `frame`, `invariants`, `metadata`, `storage`, `validate`; unified validator; voxa adoption; `scratch/` allow-list; `laz/`→`raw/` rename + 5 `source_laz` fixes + resolver update; SCHEMA.md v3.0. **Out:** `sources.py`/`raw/sources.json` registry + lineage validation (increment 2), HTTP service, S3, auto-gate, reader migration, forced legacy migration.

---

## Phase A — Bootstrap the package repo

### Task A1: Create the `scan-schema` repo skeleton

**Files:**
- Create: `engine/tools/scan-schema/pyproject.toml`
- Create: `engine/tools/scan-schema/src/scan_schema/__init__.py`
- Create: `engine/tools/scan-schema/README.md`
- Create: `engine/tools/scan-schema/.gitignore`
- Test: `engine/tools/scan-schema/tests/test_import.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_import.py
def test_package_imports():
    import scan_schema
    assert scan_schema.__version__
```

- [ ] **Step 2: Create `pyproject.toml`**

```toml
[project]
name = "scan-schema"
version = "3.0.0"            # major tracks schema major; bump deliberately on schema change
description = "Single executable definition of the lidar annotated-scan schema (v3.0)"
requires-python = ">=3.12"
dependencies = ["numpy>=1.26"]

[project.optional-dependencies]
dev = ["pytest>=8"]

[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[tool.setuptools.packages.find]
where = ["src"]

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
addopts = "-q"
```

- [ ] **Step 3: Create `src/scan_schema/__init__.py`**

```python
"""scan_schema — the single executable definition of the lidar annotated-scan schema.

See lidar/SCHEMA.md (v3.0). Public API is re-exported here so consumers
`from scan_schema import ScanLayout, validate_archive` without reaching into modules.
"""
__version__ = "3.0.0"

# Populated as modules land (Tasks B/C). Kept minimal here to keep test_import green.
```

- [ ] **Step 4: Create `README.md` and `.gitignore`**

`README.md`: one paragraph (what it is, `pip install -e .`, `python -m scan_schema.validate <lidar_root>`). `.gitignore`: `__pycache__/`, `*.egg-info/`, `.pytest_cache/`, `dist/`, `build/`.

- [ ] **Step 5: Init repo, install, run test**

Run:
```bash
cd engine/tools/scan-schema && git init -q && python -m venv .venv && \
  .venv/bin/pip -q install -e ".[dev]" && .venv/bin/pytest -q
```
Expected: 1 passed.

- [ ] **Step 6: Commit**

```bash
cd engine/tools/scan-schema && git add -A && \
  git commit -m "chore: bootstrap scan-schema package (v3.0.0)"
```

---

## Phase B — Core lifts (verbatim from voxa; behavior preserved)

> These three modules are near-verbatim copies of working, tested voxa code. Lift the source unchanged except where noted, then port the existing voxa test to prove parity. Do **not** redesign.

### Task B1: `layout.py` — ScanLayout + SessionPaths

**Files:**
- Create: `src/scan_schema/layout.py`
- Modify: `src/scan_schema/__init__.py`
- Test: `tests/test_layout.py`

- [ ] **Step 1: Lift the module.** Copy `engine/tools/labeling/voxa/backend/scenes/scan_layout.py` verbatim into `src/scan_schema/layout.py` (it's pure path joins, no voxa imports). Then add two archive-level properties to `ScanLayout` (siblings of `classes_json`) for the renamed raw dir (registry itself is increment 2):

```python
    @property
    def raw_dir(self) -> Path:
        # <lidar_root>/raw/  (scan_dir is <lidar_root>/annotated/<scan>/)
        return self.scan_dir.parent.parent / "raw"

    @property
    def sources_json(self) -> Path:        # populated in increment 2
        return self.raw_dir / "sources.json"

    @property
    def mesh_meta_json(self) -> Path:
        return self.source_dir / "mesh.meta.json"

    @property
    def scratch_dir(self) -> Path:
        return self.scan_dir / "scratch"
```

- [ ] **Step 2: Export from `__init__.py`**

```python
from scan_schema.layout import ScanLayout, SessionPaths  # noqa: E402
```

- [ ] **Step 3: Port the test.** Copy assertions from voxa `backend/tests/test_scan_layout.py` into `tests/test_layout.py` (adjust import to `from scan_schema import ScanLayout`). Add cases for the new `raw_dir`, `scratch_dir`, `mesh_meta_json`.

- [ ] **Step 4: Run**

Run: `.venv/bin/pytest tests/test_layout.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "feat: layout.py (ScanLayout/SessionPaths) + raw/scratch paths"
```

### Task B2: `frame.py` — Frame model + is_rigid

**Files:**
- Create: `src/scan_schema/frame.py`
- Modify: `src/scan_schema/__init__.py`
- Test: `tests/test_frame.py`

- [ ] **Step 1: Lift.** Copy `engine/tools/labeling/voxa/backend/scenes/frame.py` verbatim into `src/scan_schema/frame.py` (depends only on numpy). Exposes `Frame`, `is_rigid`, `frame_from_dict`, `compose_a_to_v`, `apply_transform`.

- [ ] **Step 2: Export** the names from `frame` in `__init__.py` (at least `Frame`, `is_rigid`, `frame_from_dict`).

- [ ] **Step 3: Port the test.** Copy voxa `backend/tests/test_frame.py` → `tests/test_frame.py` (import from `scan_schema.frame`).

- [ ] **Step 4: Run** `.venv/bin/pytest tests/test_frame.py -v` → PASS.

- [ ] **Step 5: Commit** `git commit -am "feat: frame.py (Frame, is_rigid, transforms)"`

### Task B3: `invariants.py` — GT invariants 1–6

**Files:**
- Create: `src/scan_schema/invariants.py`
- Modify: `src/scan_schema/__init__.py`
- Test: `tests/test_invariants.py`

- [ ] **Step 1: Write failing tests** for the two NEW invariants (1–2) plus a parity case for 3–4:

```python
# tests/test_invariants.py
import numpy as np, pytest
from scan_schema.invariants import validate_invariants, check_array_shape_dtype, ArraySpec

def test_invariant3_class_instance_agreement():
    cls = np.array([-1, 0, 0], dtype=np.int32)
    inst = np.array([-1, -1, 5], dtype=np.int32)   # mismatch at idx1
    with pytest.raises(ValueError, match="invariant 3"):
        validate_invariants(cls, inst)

def test_invariant4_per_segment_class_consistency():
    cls = np.array([0, 1], dtype=np.int32)
    inst = np.array([7, 7], dtype=np.int32)         # same seg, two classes
    with pytest.raises(ValueError, match="invariant 4"):
        validate_invariants(cls, inst)

def test_invariant2_shape_dtype():
    arr = np.zeros((5,), dtype=np.int8)
    check_array_shape_dtype(arr, ArraySpec(n_points=5, dtype="int8"))            # ok
    with pytest.raises(ValueError, match="dtype"):
        check_array_shape_dtype(arr, ArraySpec(n_points=5, dtype="int32"))
    with pytest.raises(ValueError, match="shape"):
        check_array_shape_dtype(arr, ArraySpec(n_points=6, dtype="int8"))
```

- [ ] **Step 2: Run → FAIL** (`.venv/bin/pytest tests/test_invariants.py -v`).

- [ ] **Step 3: Implement.** Copy the `_validate_invariants` body from voxa `backend/labeling/segment_io.py` (lines ~90–130) into `validate_invariants(...)` (rename, drop the leading underscore, keep the registry/meta_class_map_version params and messages verbatim — invariants 3–6). Add the shape/dtype helper for invariants 1–2:

```python
from dataclasses import dataclass

@dataclass(frozen=True)
class ArraySpec:
    n_points: int
    dtype: str            # "int8" | "int32"

def check_array_shape_dtype(arr, spec: ArraySpec) -> None:
    if arr.shape != (spec.n_points,):
        raise ValueError(f"invariant 2: shape {arr.shape} != ({spec.n_points},)")
    if arr.dtype != np.dtype(spec.dtype):
        raise ValueError(f"invariant 2: dtype {arr.dtype} != {spec.dtype}")
```

(`validate_invariants` covers 3–6; `check_array_shape_dtype` covers 2; invariant 1 — `len(scan.ply)==n_points` — is checked in `validate.py` where the PLY is read.)

- [ ] **Step 4: Export** `validate_invariants` from `__init__.py`. Run tests → PASS.

- [ ] **Step 5: Commit** `git commit -am "feat: invariants.py (GT invariants 1-6, lifted)"`

---

## Phase C — New modules

### Task C1: `storage.py` — Storage protocol + ReadOnly/Writable

**Files:**
- Create: `src/scan_schema/storage.py`
- Modify: `src/scan_schema/__init__.py`
- Test: `tests/test_storage.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_storage.py
import numpy as np, pytest
from scan_schema.storage import ReadOnlyStorage, WritableStorage, ReadOnlyError

def test_readonly_refuses_writes(tmp_path):
    st = ReadOnlyStorage()
    with pytest.raises(ReadOnlyError):
        st.write_array(tmp_path / "x.npy", np.zeros(3, np.int32))

def test_writable_roundtrips(tmp_path):
    st = WritableStorage()
    p = tmp_path / "sub" / "x.npy"
    st.mkdir(p.parent)
    st.write_array(p, np.arange(3, dtype=np.int32))
    assert st.stat(p).exists
    assert list(ReadOnlyStorage().read_array(p)) == [0, 1, 2]
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement**

```python
"""Transport seam. LocalStorage today; S3Storage later behind the same Protocol.
Also the write-protection seam: ReadOnlyStorage refuses writes loudly so accidental
writes through the package fail with context instead of corrupting data."""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol
import numpy as np

class ReadOnlyError(RuntimeError):
    pass

@dataclass(frozen=True)
class Stat:
    exists: bool
    size: int = 0

class Storage(Protocol):
    def list(self, path: Path) -> Iterable[Path]: ...
    def stat(self, path: Path) -> Stat: ...
    def read_array(self, path: Path) -> np.ndarray: ...
    def open(self, path: Path): ...
    # write ops:
    def mkdir(self, path: Path) -> None: ...
    def write_array(self, path: Path, arr: np.ndarray) -> None: ...
    def open_w(self, path: Path): ...

class _LocalRead:
    def list(self, path):
        p = Path(path)
        return sorted(p.iterdir()) if p.is_dir() else []
    def stat(self, path):
        p = Path(path)
        return Stat(p.exists(), p.stat().st_size if p.exists() else 0)
    def read_array(self, path):
        return np.load(Path(path))
    def open(self, path):
        return open(Path(path), "rb")

class ReadOnlyStorage(_LocalRead):
    def mkdir(self, path):       raise ReadOnlyError(f"read-only: mkdir {path}")
    def write_array(self, path, arr): raise ReadOnlyError(f"read-only: write {path}")
    def open_w(self, path):      raise ReadOnlyError(f"read-only: open-w {path}")

class WritableStorage(_LocalRead):
    def mkdir(self, path):       Path(path).mkdir(parents=True, exist_ok=True)
    def write_array(self, path, arr): np.save(Path(path), arr)
    def open_w(self, path):      return open(Path(path), "wb")

# Default alias used by readers/validator:
LocalStorage = ReadOnlyStorage
```

- [ ] **Step 4: Export** `Storage, ReadOnlyStorage, WritableStorage, LocalStorage, ReadOnlyError` from `__init__.py`. Run → PASS.

- [ ] **Step 5: Commit** `git commit -am "feat: storage.py (Storage protocol + ReadOnly/Writable, write-protection seam)"`

### Task C2: `metadata.py` — meta.json contract, version-gated

**Files:**
- Create: `src/scan_schema/metadata.py`
- Modify: `src/scan_schema/__init__.py`
- Test: `tests/test_metadata.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_metadata.py
from scan_schema.metadata import check_meta

def _meta(v="3.0", **kw):
    m = {"schema_version": v, "scan_name": "s", "n_points": 5, "units": "meters",
         "class_map_version": 1}
    m.update(kw); return m

def test_v3_missing_frame_is_error():
    errs, warns = check_meta(_meta())          # no frame/derivation
    assert any("frame" in e for e in errs)

def test_v2_missing_frame_is_warning():
    errs, warns = check_meta(_meta(v="2.0"))
    assert errs == [] and any("frame" in w for w in warns)

def test_v3_nonrigid_transform_is_error():
    frame = {"canonical_id": "s#local", "units": "meters", "frame_uncertain": False,
             "transform_to_canonical": [[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
    deriv = {"scan_id": "s", "variant_id": "s", "varies": ["density"], "role": "labeling"}
    errs, _ = check_meta(_meta(frame=frame, derivation=deriv))
    assert any("rigid" in e for e in errs)

def test_v3_bad_varies_is_error():
    frame = {"canonical_id": "s#local", "units": "meters", "frame_uncertain": False,
             "transform_to_canonical": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}
    deriv = {"scan_id": "s", "variant_id": "s", "varies": ["bogus"], "role": "labeling"}
    errs, _ = check_meta(_meta(frame=frame, derivation=deriv))
    assert any("varies" in e for e in errs)
```

- [ ] **Step 2: Run → FAIL.**

> **Scope note:** `check_meta` ports only the meta.json frame/derivation contract. It does **not** port `validate_scan.py`'s render-run pin check (deferred to increment 2 — see Task D4).

- [ ] **Step 3: Implement.** Port the frame/derivation logic from `scripts/scan/validate_scan.py::validate_scan_dir` (the `frame`/`derivation`/`varies`/`canonical_id`/`is_rigid` checks) into a pure `check_meta(meta: dict) -> tuple[list[str], list[str]]` (errors, warnings). Version gate: if `schema_version` startswith `"3"` → violations are errors; if startswith `"2"` → the same structural findings (missing frame/derivation) are warnings, and superseded legacy fields (`source_laz`, `parent_scan`, `coords`, `coord_offset_m`) are accepted silently; anything else (or missing) → a single hard error with a migration hint. Required-always: `scan_name`, `n_points`, `units`, `class_map_version`. Use `scan_schema.frame.is_rigid`.

- [ ] **Step 4: Export** `check_meta`. Run → PASS.

- [ ] **Step 5: Commit** `git commit -am "feat: metadata.py (meta.json contract, 3.x error / 2.x warn)"`

### Task C3: `validate.py` — whole-archive audit + CLI

**Files:**
- Create: `src/scan_schema/validate.py`
- Create: `src/scan_schema/__main__.py`
- Modify: `src/scan_schema/__init__.py`
- Test: `tests/test_validate.py`

- [ ] **Step 1: Write failing test against real fixtures**

```python
# tests/test_validate.py
from pathlib import Path
from scan_schema.validate import validate_archive

LIDAR = Path("/home/hendrik/coding/engine/data/lidar")   # real archive

def test_all_current_scans_have_zero_errors():
    report = validate_archive(LIDAR)
    for scan, res in report.items():
        assert res["errors"] == [], f"{scan}: {res['errors']}"

def test_unlabeled_session_not_flagged():
    # a session dir with no output/ must not raise an error
    report = validate_archive(LIDAR)
    assert all(res["errors"] == [] for res in report.values())

def test_known_strays_are_warnings():
    report = validate_archive(LIDAR)
    smart = report.get("smart_ais_clean", {})
    assert any("fresh_run" in w or "scratch" in w for w in smart.get("warnings", []))
```

- [ ] **Step 2: Run → FAIL.**

- [ ] **Step 3: Implement `validate.py`.** `validate_archive(root: Path, storage=None) -> dict[str, dict]`:
  - default `storage = ReadOnlyStorage()`.
  - iterate `root/"annotated"` dirs; for each build `ScanLayout(scan_dir)`.
  - read `meta.json` → run `check_meta` (errors/warnings).
  - **invariant 1:** read `scan.ply` vertex count (use `numpy`/`plyfile`-free: reuse voxa's PLY reader if available, else read header) and compare to `meta["n_points"]`; mismatch → error. *(If a PLY reader isn't trivially available in the package, read only the element-count header line — do NOT load points.)*
  - **per `sessions/*/output/`:** if `output/` absent → skip (legal unlabeled). Else `check_array_shape_dtype` on `gt_class_ids`(int32)/`gt_segment_ids`(int32) and `validate_invariants(...)`.
  - **per `sessions/*/`:** `working_class_ids`(int8)/`working_segment_ids`(int32) shape/dtype only.
  - **per `prelabel/*/`:** `instance_ids.npy` int32 shape `(n_points,)`; `segment_summary.json` parses to `{"segments":[{"id","class_id"}...]}`. No GT invariants.
  - **top-level entries** (scoped to each `annotated/<scan>/`, **not** the archive root): anything not in `ALLOWED_TOPLEVEL = {README.md, meta.json, source/, prelabel/, sessions/, renders/, sam3/, variants.json, scratch/}` → warning. Archival clouds in `source/` (not `scan.ply`/`mesh.glb`/`mesh.meta.json`) → warning. Expose the set as `validate.ALLOWED_TOPLEVEL` so Task F1's doc-consistency check can import it.
  - return `{scan_name: {"errors": [...], "warnings": [...]}}`.

- [ ] **Step 4: Implement `__main__.py`**

```python
import sys
from pathlib import Path
from scan_schema.validate import validate_archive

def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.cwd()
    report = validate_archive(root)
    n_err = 0
    for scan, res in sorted(report.items()):
        for e in res["errors"]:   print(f"ERROR  [{scan}] {e}", file=sys.stderr); n_err += 1
        for w in res["warnings"]: print(f"warn   [{scan}] {w}")
    print(f"{len(report)} scans, {n_err} errors")
    return 1 if n_err else 0

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 5: Run** `.venv/bin/pytest tests/test_validate.py -v` and `.venv/bin/python -m scan_schema.validate /home/hendrik/coding/engine/data/lidar`.
Expected: tests PASS; CLI prints `9 scans, 0 errors` (+ warnings for strays). **If any current scan shows errors, that's a real finding — fix the validator's grandfathering, not the data.**

- [ ] **Step 6: Commit** `git commit -am "feat: validate.py + CLI (whole-archive audit, replaces 3 validators)"`

---

## Phase D — voxa adoption (re-export shims, no behavior change)

> **Regression gate for this phase:** voxa's `backend/tests/test_segment_io.py` (and the full suite) must pass **unchanged** after each task — this is the proof the shims + save-gate delegation didn't change behavior. **Exception:** the validator-specific tests that import the now-deleted `validate_scan.py` (`test_real_scans_validate.py`, `test_validate_scan.py`) are *rewritten/removed in Task D4* — they are not "unchanged" gates past D3. Run from `engine/tools/labeling/voxa/`.

### Task D1: voxa depends on scan-schema; shim `scan_layout.py`

**Files:**
- Modify: `engine/tools/labeling/voxa/backend/requirements.txt` (add editable dep)
- Modify: `engine/tools/labeling/voxa/backend/scenes/scan_layout.py` (→ shim)
- Test: existing voxa suite

- [ ] **Step 1:** Add to voxa `backend/requirements.txt`: `-e ../../../scan-schema`. **Path is relative to the requirements file's dir (`backend/`):** `..`→voxa, `../..`→labeling, `../../..`→tools, so `../../../scan-schema`→`engine/tools/scan-schema`. (The equivalent one-off command run *from the voxa root* is `.venv/bin/pip install -e ../../scan-schema` — two levels, different base dir. Don't confuse the two.) After editing, verify: `cd engine/tools/labeling/voxa && .venv/bin/pip install -e ../../scan-schema && .venv/bin/python -c "import scan_schema"`.

- [ ] **Step 2: Replace `scenes/scan_layout.py` body with a shim**

```python
"""Moved to the shared `scan_schema` package (lidar SCHEMA.md v3.0).
Re-exported here so voxa's existing `from scenes.scan_layout import ScanLayout`
imports keep working. New code should import from scan_schema directly."""
from scan_schema import ScanLayout, SessionPaths  # noqa: F401
```

- [ ] **Step 3: Run voxa suite**

Run: `cd engine/tools/labeling/voxa && .venv/bin/pytest -q`
Expected: same pass count as before this task (no new failures). Pay attention to `test_scan_layout.py`.

- [ ] **Step 4: Commit (voxa repo)** `git commit -am "refactor: source ScanLayout from scan_schema (shim)"`

### Task D2: shim `frame.py`

**Files:**
- Modify: `engine/tools/labeling/voxa/backend/scenes/frame.py` (→ shim)

- [ ] **Step 1: Replace body with shim**

```python
"""Moved to scan_schema.frame (lidar SCHEMA.md v3.0). Re-exported for existing imports."""
from scan_schema.frame import (  # noqa: F401
    Frame, is_rigid, frame_from_dict, compose_a_to_v, apply_transform,
)
```

- [ ] **Step 2: Run** voxa suite (esp. `test_frame.py`, `test_resolver.py`, `test_render_meta.py`) → no new failures.

- [ ] **Step 3: Commit (voxa)** `git commit -am "refactor: source frame model from scan_schema (shim)"`

### Task D3: save-gate delegates to `scan_schema.invariants`

**Files:**
- Modify: `engine/tools/labeling/voxa/backend/labeling/segment_io.py`

- [ ] **Step 1:** Replace the body of `_validate_invariants` with a call into the package, keeping the name/signature so callers (`save_labels`) are untouched:

```python
from scan_schema.invariants import validate_invariants as _ss_validate_invariants

def _validate_invariants(class_ids, instance_ids, registry=None, meta_class_map_version=None):
    _ss_validate_invariants(class_ids, instance_ids, registry=registry,
                            meta_class_map_version=meta_class_map_version)
```

(Remove the now-duplicated invariant body. Leave `compute_fingerprint`, `save_labels`, etc. as-is.)

- [ ] **Step 2: Run** `cd engine/tools/labeling/voxa && .venv/bin/pytest backend/tests/test_segment_io.py backend/tests/test_real_scans_validate.py -v`
Expected: PASS (save behavior identical).

- [ ] **Step 3: Commit (voxa)** `git commit -am "refactor: save-gate invariants delegate to scan_schema"`

### Task D4: delete the two superseded validators + migrate their tests

**Files:**
- Delete: `engine/tools/labeling/voxa/scripts/scan/validate_scan.py`
- Delete/rewrite: `engine/tools/labeling/voxa/backend/tests/test_validate_scan.py`, `backend/tests/test_real_scans_validate.py`
- Delete: `engine/data/tools/validate_annotated.py`

> **Scope note (intentional):** `scan_schema.metadata.check_meta` covers the meta.json frame/derivation contract but **does NOT** port `validate_scan.py`'s render-run pin check (`renders/<run>/meta.json::generated_from.variant_id`, lines 58–70). Render-pin validation is **not** in the increment-1 invariant set — it belongs with the lineage/variants work in increment 2. Dropping it here is deliberate scope-shedding, not an oversight.

- [ ] **Step 1: Map references.** `grep -rn "validate_scan\|validate_annotated" engine/ --include=*.py | grep -v scan_schema`. Expect hits in `backend/tests/test_validate_scan.py`, `backend/tests/test_real_scans_validate.py`, possibly `backfill_scan_frame.py`. If `backfill_scan_frame.py` imports `validate_scan`, repoint it at `scan_schema.metadata.check_meta` (frame/derivation parts only).

- [ ] **Step 2: Migrate the two tests.**
  - `test_real_scans_validate.py` (currently `from validate_scan import validate_scan_dir`, asserts the 9 real scans pass §7): rewrite to assert via `scan_schema.metadata.check_meta` on each scan's `meta.json` that **errors == []** (2.0 grandfathered → only warnings). This preserves the "real scans stay valid" intent against the new definition.
  - `test_validate_scan.py`: its meta.json/frame/derivation cases → port to `scan_schema`'s `tests/test_metadata.py` (already created in Task C2; add any missing cases). Its **render-pin** case (`test_render_run_missing_meta_flagged` or similar) → **delete** (feature dropped per the scope note).
  - Then delete `test_validate_scan.py` from voxa.

- [ ] **Step 3: Delete** `scripts/scan/validate_scan.py` and `engine/data/tools/validate_annotated.py`. Run the full voxa suite → green (no references to the deleted module remain).

- [ ] **Step 4: Commit (voxa)** `git commit -am "chore: remove validate_scan.py; migrate its tests to scan_schema (render-pin check deferred to inc.2)"` and note the data-tree deletion of `validate_annotated.py` in the message (that tree isn't git).

---

## Phase E — laz/→raw/ rename + schema_version 3 acceptance

### Task E1: scene_registry accepts 2.x or 3.x, resolves raw/

**Files:**
- Modify: `engine/tools/labeling/voxa/backend/scenes/scene_registry.py:107` and `:124-132`
- Test: `engine/tools/labeling/voxa/backend/tests/test_scene_registry.py`

- [ ] **Step 1: Write/extend a failing test** asserting a scan with `schema_version: "3.0"` is discovered (not skipped), and that `source_laz` resolves when the file lives under `raw/`.

- [ ] **Step 2: Implement.** Line 107: change the gate to accept `"2"` or `"3"`:

```python
sv = str(meta.get("schema_version") or "")
if meta is None or not (sv.startswith("2") or sv.startswith("3")):
    logging.info("skipping %s: scan-schema %r != 2.x/3.x", sd.name, sv or None)
    continue
```

Lines 124–132: resolve `source_laz` under `raw/` first, then `laz/` (back-compat), then the archive-relative fallback:

```python
src_laz_str = meta.get("source_laz")
if src_laz_str:
    name = Path(src_laz_str).name
    for cand in (lidar_root / "raw" / name, lidar_root / "laz" / name,
                 lidar_root.parent / src_laz_str):
        if cand.exists():
            source_laz_path = str(cand); break
```

- [ ] **Step 3: Run** the registry tests + full voxa suite → PASS.

- [ ] **Step 4: Commit (voxa)** `git commit -am "feat: discovery accepts schema_version 2.x/3.x; resolve source_laz under raw/"`

### Task E2: rename the directory + fix the 5 source_laz paths (data archive)

**Files (data tree, not git):**
- Rename: `engine/data/lidar/laz/` → `engine/data/lidar/raw/`
- Modify: `source_laz` in the 5 scans whose value is non-null

- [ ] **Step 1: Verify which scans carry a non-null `source_laz`**

Run:
```bash
cd engine/data/lidar && for f in annotated/*/meta.json; do \
  python3 -c "import json;d=json.load(open('$f'));v=d.get('source_laz');print(v,'<-','$f') if v else None"; done
```
Expected: 5 lines (smart_ais_clean, factory_large, navvis_mlx, navvis_vlx3_water_treatment, construction_site).

- [ ] **Step 2: Rename the directory**

Run: `cd engine/data/lidar && mv laz raw`
(Note: `laz/` contains a `.remember/` subdir — it rides along harmlessly.)

- [ ] **Step 3: Rewrite the 5 paths** `"lidar/laz/…"` → `"lidar/raw/…"` in those `meta.json` files (a small scripted `json` load/dump preserving formatting, or targeted edits).

- [ ] **Step 4: Verify resolution end-to-end**

Run: `cd engine/tools/labeling/voxa && .venv/bin/python -c "from backend.scenes.scene_registry import ...; discover(...)"` (or the existing discovery smoke path) and confirm all 5 `source_laz_path` resolve to `raw/`. Also re-run `.venv/bin/python -m scan_schema.validate /home/hendrik/coding/engine/data/lidar` → `9 scans, 0 errors`.

- [ ] **Step 5:** No git commit (data tree). Note completion in the next voxa commit.

---

## Phase F — SCHEMA.md v3.0 rewrite

### Task F1: rewrite `lidar/SCHEMA.md` to v3.0

**Files:**
- Modify: `engine/data/lidar/SCHEMA.md`

- [ ] **Step 1:** Rewrite the doc to v3.0 per the spec: header `v3.0`; the unified layout (sessions/, prelabel/, source/ contract, scratch/, raw/); the `meta.json` field contract incl. required `frame`+`derivation` (with the documented-but-not-yet-enforced `derivation.root`/`parent` lineage links, marked "registry: increment 2"); the 6 invariants; the grandfathering rule (2.x read-with-warnings); the allowed top-level entries list; a pointer to `scan_schema` as the executable definition and `python -m scan_schema.validate` as the audit. Keep the v2.0/v1.3 entries in the changelog with a v3.0 entry on top.

- [ ] **Step 2: Validate the doc matches reality** by re-running the validator and confirming the allowed-entries list in the doc equals the set in `validate.py` (single source — consider importing the list into the doc-gen, or add a test that the doc's list matches `validate.ALLOWED_TOPLEVEL`).

- [ ] **Step 3:** No git commit (data tree). 

### Task F2: final end-to-end verification

- [ ] **Step 1:** From a clean shell: install scan-schema, run its tests (`pytest` → all green), run voxa's full suite (→ no new failures vs baseline), run `python -m scan_schema.validate <lidar_root>` (→ `9 scans, 0 errors`).
- [ ] **Step 2:** Confirm `ReadOnlyStorage().write_array(...)` raises and that only voxa's writer path constructs `WritableStorage` (grep `WritableStorage` usage).
- [ ] **Step 3: Final commits** in scan-schema and voxa repos with a summary message.

---

## Write-protection runbook (documentation deliverable, Task F1 appendix)

Add to `lidar/SCHEMA.md` (or a sibling `OPERATIONS.md`) a short hardening section:
- `chattr +i` on every `raw/*.laz` (and `*.ply` roots) and each `annotated/*/source/scan.ply` — they must never change; lineage fingerprints depend on them.
- `lidar/` writable only by the voxa/owner account; readers run without write.
- Optional: track `meta.json`/`*.json`/`sources.json` in a small git repo (clouds git-ignored) for metadata recovery.

---

## Notes for the executor
- **Lifts are verbatim.** Tasks B1–B3 copy working code — resist "improving" it; parity with voxa is the goal (the shims make voxa the regression test).
- **The validator must report 0 errors on all 9 current (2.0) scans.** Their legacy gaps (missing frame/derivation, archival clouds) are *warnings*. If you see errors, the grandfathering logic is wrong — fix `metadata.check_meta`/`validate_archive`, never the data.
- **No forced migration.** Do not backfill frame/derivation or move clouds to scratch/ on existing scans in this increment.
