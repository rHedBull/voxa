# Scan & Source Naming Convention — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enforce a consistent `<scene>_<vendor>[_<density>]` naming convention for `scan_name` and `source_id` via a warn-level check in `scan_schema`, then migrate the existing 9 scans + 7 roots and all internal references.

**Architecture:** Two phases across two repos. **Phase A** (`scan-schema` repo): a `naming.py` module with grammar predicates + a warn-level hook in `validate_archive`; lands + pushes to `main`. **Phase B** (`voxa` repo): bump the scan_schema dependency, then a deterministic `rename_scans.py` migration (dry-run default, full backup, field-exact residual scan, 0-errors gates) run against the real archive, plus docs.

**Tech Stack:** Python 3.12, pytest, `scan_schema` (regex + `Registry`), numpy-free for this work. The lidar archive is JSON + dirs under `/home/hendrik/coding/engine/data/lidar`.

**Spec:** `docs/superpowers/specs/2026-06-26-scan-naming-convention-design.md`

**Repos / paths:**
- scan-schema repo: `/home/hendrik/coding/engine/tools/scan-schema` (own git repo; voxa pins it via `scan-schema @ git+https://github.com/rHedBull/scan_schema.git@main`)
- voxa repo: `/home/hendrik/coding/engine/tools/labeling/voxa`
- archive: `/home/hendrik/coding/engine/data/lidar` (**not** under git)

**Worktree note:** Phase A is committed on a branch in the scan-schema repo and pushed to `main`. Phase B work happens in the voxa repo (a worktree is fine for the script + tests; the actual `--apply` migration mutates the shared archive and must run once, reviewed).

---

## File Structure

**Phase A — `scan-schema` repo:**
- Create `src/scan_schema/naming.py` — `KNOWN_VENDORS`, `is_valid_scan_name`, `is_valid_source_id` (grammar only).
- Modify `src/scan_schema/__init__.py` — export the three symbols; bump `__version__` to `3.1.0`.
- Modify `src/scan_schema/validate.py` — scan_name warning in the per-scan loop; a new pass over `Registry.roots` attaching source_id warnings under a reserved `"raw/sources.json"` result key.
- Modify `src/scan_schema/__main__.py` — exclude the reserved key from the "N scans" count.
- Create `tests/test_naming.py` — predicate unit tests.
- Modify `tests/test_validate.py` — assert the new warnings surface.

**Phase B — `voxa` repo:**
- Create `scripts/scan/rename_scans.py` — the migration (RENAME_MAP, backup, rewrite, residual scan, gates).
- Create `backend/tests/test_rename_scans.py` — migration test on a tmp fixture archive.
- Modify `voxa/docs/scan-schema.md` and `data/lidar/SCHEMA.md` — document the convention; fix the README "required vs recommended" note.

---

## Phase A — scan_schema naming module + enforcement

> Run all Phase A commands from `/home/hendrik/coding/engine/tools/scan-schema`. Use that repo's venv: `.venv/bin/python -m pytest` (or `python -m pytest` if its venv is active).

### Task A1: Naming predicates (`naming.py`)

**Files:**
- Create: `src/scan_schema/naming.py`
- Test: `tests/test_naming.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_naming.py
import pytest
from scan_schema.naming import is_valid_scan_name, is_valid_source_id, KNOWN_VENDORS


@pytest.mark.parametrize("name", [
    "water_treatment_navvis",
    "mechanical_room_matterport",
    "construction_site_navvis",
    "factory_navvis",
    "generator_concrete_navvis",
])
def test_source_id_accepts_conforming(name):
    assert is_valid_source_id(name)


@pytest.mark.parametrize("name", [
    "water_pump_navvis_500k",
    "water_pump_navvis_3m",
    "factory_navvis",                 # no density when single-density
    "smart_ais_navvis",
])
def test_scan_name_accepts_conforming(name):
    assert is_valid_scan_name(name)


@pytest.mark.parametrize("name", [
    "navvis_vlx3_water_treatment",    # vendor not in final position
    "factory_large",                  # no vendor token
    "smart_ais_clean",                # 'clean' not a vendor
    "munich_water_pump",              # 'pump' not a vendor
    "Factory_navvis",                 # uppercase
    "factory_navvis_",                # trailing underscore
    "navvis",                         # no scene before vendor
    "water_pump_navvis_foo",          # trailing token not a density
    "water_pump_navvis_3m_extra",     # extra token after density
])
def test_rejects_nonconforming(name):
    assert not is_valid_scan_name(name)
    assert not is_valid_source_id(name)


def test_source_id_rejects_density_suffix():
    # density is scan_name-only; a source_id must NOT carry one
    assert not is_valid_source_id("water_pump_navvis_3m")


def test_known_vendors_is_extensible_tuple():
    assert "navvis" in KNOWN_VENDORS and "matterport" in KNOWN_VENDORS
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_naming.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'scan_schema.naming'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/scan_schema/naming.py
"""Naming convention for scan_name and source_id (scan-schema v3.1).

    source_id = <scene>_<vendor>
    scan_name = <scene>_<vendor>[_<density>]

      scene   = [a-z0-9]+(_[a-z0-9]+)*   (>= 1 token)
      vendor  = a token in KNOWN_VENDORS
      density = \\d+[km] | full           (scan_name only, optional)

These predicates enforce the GRAMMAR only. The rule "density present only when
>1 density of the same <scene>_<vendor> is materialized on disk" is a human
guideline — it cannot be checked from a single name — and is not enforced here.
Known limitation: a scene token that happens to equal a vendor word is accepted
as long as a vendor also occupies the final (pre-density) position.
"""
from __future__ import annotations

import re

# Extend this tuple to admit a new capture vendor. Shared cross-tool.
KNOWN_VENDORS: tuple[str, ...] = ("matterport", "navvis")

_DENSITY = r"\d+[km]|full"


def _vendor_alt() -> str:
    return "|".join(re.escape(v) for v in KNOWN_VENDORS)


def _source_re() -> re.Pattern[str]:
    return re.compile(rf"^[a-z0-9]+(?:_[a-z0-9]+)*_(?:{_vendor_alt()})$")


def _scan_re() -> re.Pattern[str]:
    return re.compile(
        rf"^[a-z0-9]+(?:_[a-z0-9]+)*_(?:{_vendor_alt()})(?:_(?:{_DENSITY}))?$"
    )


def is_valid_source_id(name: str) -> bool:
    """True iff name matches <scene>_<vendor> (no density token)."""
    return bool(_source_re().fullmatch(name))


def is_valid_scan_name(name: str) -> bool:
    """True iff name matches <scene>_<vendor>[_<density>]."""
    return bool(_scan_re().fullmatch(name))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_naming.py -q`
Expected: PASS (all parametrized cases green)

- [ ] **Step 5: Commit**

```bash
git add src/scan_schema/naming.py tests/test_naming.py
git commit -m "feat: scan_name/source_id naming predicates"
```

### Task A2: Export symbols + version bump

**Files:**
- Modify: `src/scan_schema/__init__.py`
- Test: `tests/test_import.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_import.py`)

```python
def test_naming_exports():
    import scan_schema
    assert scan_schema.is_valid_scan_name("factory_navvis")
    assert scan_schema.is_valid_source_id("factory_navvis")
    assert "navvis" in scan_schema.KNOWN_VENDORS
    assert scan_schema.__version__ == "3.1.0"
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_import.py::test_naming_exports -q`
Expected: FAIL — `AttributeError` / version mismatch

- [ ] **Step 3: Implement** — in `src/scan_schema/__init__.py`: set `__version__ = "3.1.0"` and add after the other re-exports:

```python
from scan_schema.naming import KNOWN_VENDORS, is_valid_scan_name, is_valid_source_id  # noqa: E402
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_import.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/scan_schema/__init__.py tests/test_import.py
git commit -m "feat: export naming predicates; bump to 3.1.0"
```

### Task A3: Warn on non-conforming scan_name in `validate_archive`

**Files:**
- Modify: `src/scan_schema/validate.py` (per-scan loop, after meta is loaded — near the top-level-entries block ~line 277)
- Test: `tests/test_validate.py`

- [ ] **Step 1: Write the failing test** (append to `tests/test_validate.py`; reuse its existing tmp-archive helpers — inspect the file first: the real helpers are `_mk_scan` / `_v3_meta` / `_registry`, there is no `_make_archive`. Either call those directly or add a thin `_make_archive(tmp_path, scan_name)` wrapper around them.)

```python
def test_validate_warns_on_nonconforming_scan_name(tmp_path):
    # Build a minimal valid scan under a non-conforming dir name.
    root = _make_archive(tmp_path, scan_name="factory_large")   # helper from this file
    report = validate_archive(root)
    warns = report["factory_large"]["warnings"]
    assert any("does not match" in w and "scan_name" in w for w in warns)
    assert report["factory_large"]["errors"] == []  # warning, never error


def test_validate_no_naming_warning_for_conforming(tmp_path):
    root = _make_archive(tmp_path, scan_name="factory_navvis")
    report = validate_archive(root)
    assert not any("does not match" in w for w in report["factory_navvis"]["warnings"])
```

> The archive builder must write `annotated/<scan_name>/{meta.json, source/scan.ply}` with a 1-point PLY, matching `n_points`, and the required meta fields (`scan_name`, `n_points`, `units`, `class_map_version`) so `check_meta` returns no errors and only the *naming* warning is exercised.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_validate.py::test_validate_warns_on_nonconforming_scan_name -q`
Expected: FAIL — no such warning

- [ ] **Step 3: Implement** — in `validate.py`, add the import at top:

```python
from scan_schema.naming import is_valid_scan_name, is_valid_source_id
```

and inside the per-scan loop, **after `meta` is loaded** (so the declared `scan_name`
is available) and before `result[scan_name] = ...`. Per spec §5 the check targets
`meta.json::scan_name`, falling back to the dir name when meta is absent/omits it:

```python
        # Naming convention (warn-level, never an error).
        declared = (meta or {}).get("scan_name", scan_name)
        if not is_valid_scan_name(declared):
            warnings.append(
                f"scan_name {declared!r} does not match "
                "<scene>_<vendor>[_<density>]"
            )
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_validate.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/scan_schema/validate.py tests/test_validate.py
git commit -m "feat: warn on non-conforming scan_name in validate_archive"
```

### Task A4: Warn on non-conforming source_id (dedicated roots pass)

**Files:**
- Modify: `src/scan_schema/validate.py` (after the per-scan `for` loop, before `return result`)
- Modify: `src/scan_schema/__main__.py` (count fix)
- Test: `tests/test_validate.py`, `tests/test_main.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_validate.py
def test_validate_warns_on_nonconforming_source_id(tmp_path):
    root = _make_archive(tmp_path, scan_name="factory_navvis")
    # write a sources.json with one bad + one good key
    _write_sources(root, {
        "factory_large": "raw/Factory-large.laz",       # bad: no vendor in final pos
        "factory_navvis": "raw/Factory.laz",            # good
    })
    report = validate_archive(root)
    rk = report["raw/sources.json"]
    assert any("source_id" in w and "factory_large" in w for w in rk["warnings"])
    assert not any("factory_navvis" in w for w in rk["warnings"])
    assert rk["errors"] == []
```

```python
# tests/test_main.py — the reserved key must NOT be counted as a scan
def test_cli_excludes_reserved_key_from_scan_count(tmp_path, capsys):
    root = _make_archive(tmp_path, scan_name="factory_navvis")          # 1 real scan
    _write_sources(root, {"factory_large": "raw/Factory-large.laz"})    # bad root
    from scan_schema.__main__ import main as audit_main                 # match real entrypoint
    audit_main([str(root)])
    out = capsys.readouterr().out
    assert "1 scans, 0 errors" in out          # NOT "2 scans"
    assert "warn   [raw/sources.json]" in out
```

> Inspect `tests/test_main.py` for the existing CLI-invocation pattern (how it calls the audit entrypoint + captures output) and mirror it — the import/arg shape above is illustrative.
> Add a `_write_sources(root, mapping)` helper writing `raw/sources.json` in the `{"sources": [{source_id, path, format, fingerprint, n_points, ...}]}` shape (see real `raw/sources.json`). Fingerprint/n_points can be dummy values; the naming pass only reads keys.

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_validate.py::test_validate_warns_on_nonconforming_source_id -q`
Expected: FAIL — `KeyError: 'raw/sources.json'`

- [ ] **Step 3: Implement** — in `validate.py`, after the per-scan loop and before `return result`:

```python
    # Source-id naming (warn-level). validate_archive is keyed by scan_name and
    # lineage is only visited per-scan, so a root referenced by no scan would
    # never be checked — iterate the registry roots directly. Findings attach
    # under a reserved, non-scan result key.
    root_warnings = [
        f"source_id {sid!r} does not match <scene>_<vendor>"
        for sid in registry.roots
        if not is_valid_source_id(sid)
    ]
    if root_warnings:
        result["raw/sources.json"] = {"errors": [], "warnings": root_warnings}
```

and in `__main__.py`, change the summary so the reserved key isn't counted as a scan:

```python
    n_scans = sum(1 for k in report if k != "raw/sources.json")
    print(f"{n_scans} scans, {n_err} errors")
```

- [ ] **Step 4: Run to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_validate.py tests/test_main.py -q`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/scan_schema/validate.py src/scan_schema/__main__.py tests/test_validate.py tests/test_main.py
git commit -m "feat: warn on non-conforming source_id via registry-roots pass"
```

### Task A5: Full suite + push scan-schema to main

- [ ] **Step 1: Run the whole scan-schema suite**

Run: `.venv/bin/python -m pytest -q`
Expected: PASS (all green, including pre-existing tests)

- [ ] **Step 2: Push to main** (voxa installs from `@main`)

```bash
git push origin main      # or open+merge a PR if main is protected
```

> ⚠️ Gate: voxa cannot pick up the new validator until this is on `main`. Confirm the push/merge landed before Phase B.

---

## Phase B — voxa migration

> Run Phase B commands from `/home/hendrik/coding/engine/tools/labeling/voxa`. Archive root for real runs: `/home/hendrik/coding/engine/data/lidar`.

### Task B1: Bump scan_schema into voxa's venv

**Files:** none (environment only)

- [ ] **Step 1: Reinstall scan_schema from main**

Run (`--no-cache-dir` avoids pip serving a stale git ref from cache):
```bash
.venv/bin/pip install --force-reinstall --no-deps --no-cache-dir \
  "scan-schema @ git+https://github.com/rHedBull/scan_schema.git@main"
.venv/bin/python -c "import scan_schema; print(scan_schema.__version__)"
```
Expected: prints `3.1.0` (if it prints `3.0.0`, the new commit isn't on `main` yet — revisit A5)

- [ ] **Step 2: Sanity-check the new warnings appear on the real archive**

Run: `.venv/bin/python -m scan_schema /home/hendrik/coding/engine/data/lidar`
Expected: still `0 errors`, but now several `warn [<scan>] scan_name ... does not match ...` lines and a `warn [raw/sources.json] source_id ...` block. **This is the pre-migration baseline.**

- [ ] **Step 3: Commit the dep note** (if requirements pin a rev/comment, update it; otherwise no commit)

### Task B2: Migration script + test (TDD on a fixture)

**Files:**
- Modify: `pyproject.toml` (pytest `pythonpath`)
- Create: `scripts/scan/rename_scans.py`
- Test: `backend/tests/test_rename_scans.py`

The script exposes a testable function `run_migration(root, rename_map, apply: bool) -> Report` plus a CLI. `rename_map` is `{"scans": {old: new}, "sources": {old: new}}`. Reading the real RENAME_MAP from a module constant keeps the CLI deterministic; the test injects a small map.

- [ ] **Step 1: Make `scripts/` importable from pytest**

`scripts/` lives at the repo root, but `pyproject.toml` has `pythonpath = ["backend"]` only, so `from scripts.scan.rename_scans import ...` would raise `ModuleNotFoundError`. Add the repo root so `scripts.scan.rename_scans` resolves as a PEP-420 namespace package (no `__init__.py` needed):

```toml
# pyproject.toml — [tool.pytest.ini_options]
pythonpath = ["backend", "."]
```

Run: `grep -n pythonpath pyproject.toml` → confirm `["backend", "."]`.

- [ ] **Step 2: Write the failing test**

```python
# backend/tests/test_rename_scans.py
import json
from pathlib import Path

from scripts.scan.rename_scans import run_migration
import scan_schema


def _ply(path: Path, n: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = "\n".join("0 0 0" for _ in range(n))
    path.write_text(
        f"ply\nformat ascii 1.0\nelement vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n" + pts + "\n"
    )


def _build_fixture(root: Path):
    """Two-scan archive: one exercises the full blast radius (factory_large);
    one exercises the prefix-substring trap (construction_site, whose old name is
    a prefix of its new name construction_site_navvis — spec §8 regression guard)."""
    raw = root / "raw"; raw.mkdir(parents=True)
    (raw / "sources.json").write_text(json.dumps({"sources": [
        {"source_id": "factory_large", "path": "raw/Factory.laz", "format": "laz",
         "fingerprint": "sha256:aa", "n_points": 5, "origin_url": None,
         "registered_at": "2026-01-01T00:00:00+00:00"},
        {"source_id": "construction_site_sample_data", "path": "raw/Construction.laz",
         "format": "laz", "fingerprint": "sha256:cc", "n_points": 5, "origin_url": None,
         "registered_at": "2026-01-01T00:00:00+00:00"},
    ]}))

    # --- second scan: prefix-substring trap (minimal — meta + ply only) ---
    c = root / "annotated" / "construction_site"
    _ply(c / "source" / "scan.ply", 5)
    (c / "meta.json").write_text(json.dumps({
        "schema_version": "3.0", "scan_name": "construction_site", "n_points": 5,
        "units": "meters", "class_map_version": 1,
        "frame": {"canonical_id": "construction_site#local",
                  "transform_to_canonical": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
        "derivation": {"scan_id": "construction_site", "variant_id": "construction_site",
                       "varies": ["density"], "role": "labeling",
                       "root": {"source_id": "construction_site_sample_data",
                                "fingerprint": "sha256:cc"},
                       "parent": {"ref": "construction_site_sample_data",
                                  "fingerprint": "sha256:cc"}},
    }))

    s = root / "annotated" / "factory_large"
    _ply(s / "source" / "scan.ply", 5)
    # NOTE: variant_id / labeling_variant are deliberately set to the OLD scan_name
    # ("factory_large") — the real-world case for 6/9 scans. The migration MUST leave
    # them untouched and the residual gate MUST NOT trip on them (deferred namespace).
    (s / "meta.json").write_text(json.dumps({
        "schema_version": "3.0", "scan_name": "factory_large", "n_points": 5,
        "units": "meters", "class_map_version": 1,
        "frame": {"canonical_id": "factory_large#local",
                  "transform_to_canonical": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
        "derivation": {"scan_id": "factory_large", "variant_id": "factory_large",
                       "varies": ["density"], "role": "labeling",
                       "root": {"source_id": "factory_large", "fingerprint": "sha256:aa"},
                       "parent": {"ref": "factory_large", "fingerprint": "sha256:aa"}},
    }))
    (s / "variants.json").write_text(json.dumps({
        "scan_id": "factory_large", "canonical_id": "factory_large#local",
        "labeling_variant": "factory_large",
        "variants": [{"variant_id": "factory_large", "varies": ["density"], "role": "labeling",
                      "path": str(s), "root_source_id": "factory_large",
                      "root_fingerprint": "sha256:aa", "source_fingerprint": None,
                      "source": "potree:factory_large (scan_15M.las)",
                      "transform_to_canonical": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}],
    }))
    r = s / "renders" / "upper"; r.mkdir(parents=True)
    (r / "meta.json").write_text(json.dumps({
        "run_id": "upper",
        "frame": {"canonical_id": "factory_large#local"},
        "generated_from": {"scan_id": "factory_large", "variant_id": "factory_large",
                           "source": "potree:factory_large (scan_15M.las)",
                           "source_fingerprint": "sha256:bb", "n_points": 5},
    }))
    (r / "manifest.json").write_text(json.dumps({"scene": "factory_large", "frames": []}))
    sess = s / "sessions" / "20260101-000000_blank"; sess.mkdir(parents=True)
    (sess / "session.json").write_text(json.dumps({"source_fingerprint": "x",
                                                    "preseg_fingerprint": None}))
    (sess / "instances_gt.json").write_text(json.dumps({
        "scene": "annotated/factory_large", "instances": []}))
    # stray .bak — must be left untouched
    (sess / "instances_gt.json.bak-inst1").write_text(json.dumps({
        "scene": "annotated/factory_large"}))


RENAME = {
    "scans": {"factory_large": "factory_navvis",
              "construction_site": "construction_site_navvis"},
    "sources": {"factory_large": "factory_navvis",
                "construction_site_sample_data": "construction_site_navvis"},
}


def test_dry_run_changes_nothing(tmp_path):
    _build_fixture(tmp_path)
    run_migration(tmp_path, RENAME, apply=False)
    assert (tmp_path / "annotated" / "factory_large").exists()
    assert not (tmp_path / "annotated" / "factory_navvis").exists()


def test_apply_renames_and_rewrites_all_refs(tmp_path):
    _build_fixture(tmp_path)
    run_migration(tmp_path, RENAME, apply=True)
    new = tmp_path / "annotated" / "factory_navvis"
    assert new.exists() and not (tmp_path / "annotated" / "factory_large").exists()

    meta = json.loads((new / "meta.json").read_text())
    assert meta["scan_name"] == "factory_navvis"
    assert meta["frame"]["canonical_id"] == "factory_navvis#local"
    assert meta["derivation"]["scan_id"] == "factory_navvis"
    assert meta["derivation"]["root"]["source_id"] == "factory_navvis"
    assert meta["derivation"]["parent"]["ref"] == "factory_navvis"
    # deferred namespace: retained even though it equals the OLD scan_name
    assert meta["derivation"]["variant_id"] == "factory_large"

    var = json.loads((new / "variants.json").read_text())
    assert var["scan_id"] == "factory_navvis"
    assert var["canonical_id"] == "factory_navvis#local"
    assert var["variants"][0]["path"].endswith("factory_navvis")
    assert var["variants"][0]["root_source_id"] == "factory_navvis"
    assert var["variants"][0]["source"] == "potree:factory_navvis (scan_15M.las)"
    assert var["variants"][0]["variant_id"] == "factory_large"   # untouched
    assert var["labeling_variant"] == "factory_large"            # untouched

    rmeta = json.loads((new / "renders" / "upper" / "meta.json").read_text())
    assert rmeta["frame"]["canonical_id"] == "factory_navvis#local"
    assert rmeta["generated_from"]["scan_id"] == "factory_navvis"
    assert rmeta["generated_from"]["variant_id"] == "factory_large"  # untouched
    assert rmeta["generated_from"]["source"] == "potree:factory_navvis (scan_15M.las)"
    rman = json.loads((new / "renders" / "upper" / "manifest.json").read_text())
    assert rman["scene"] == "factory_navvis"

    ig = json.loads((new / "sessions" / "20260101-000000_blank" /
                     "instances_gt.json").read_text())
    assert ig["scene"] == "annotated/factory_navvis"

    # stray .bak untouched
    bak = json.loads((new / "sessions" / "20260101-000000_blank" /
                      "instances_gt.json.bak-inst1").read_text())
    assert bak["scene"] == "annotated/factory_large"

    sources = json.loads((tmp_path / "raw" / "sources.json").read_text())
    ids = {e["source_id"] for e in sources["sources"]}
    assert ids == {"factory_navvis", "construction_site_navvis"}

    # prefix-substring trap: construction_site renamed cleanly, no self-trip,
    # and its already-correct prefix wasn't double-appended
    cnew = tmp_path / "annotated" / "construction_site_navvis"
    assert cnew.exists() and not (tmp_path / "annotated" / "construction_site").exists()
    cmeta = json.loads((cnew / "meta.json").read_text())
    assert cmeta["scan_name"] == "construction_site_navvis"
    assert cmeta["frame"]["canonical_id"] == "construction_site_navvis#local"
    assert cmeta["derivation"]["root"]["source_id"] == "construction_site_navvis"
    assert cmeta["derivation"]["variant_id"] == "construction_site"   # retained

    # postconditions: validate_archive error-free + no naming warnings
    report = scan_schema.validate_archive(tmp_path)
    assert all(r["errors"] == [] for r in report.values())
    assert not any("does not match" in w
                   for r in report.values() for w in r["warnings"])
```

- [ ] **Step 3: Run to verify it fails**

Run: `.venv/bin/python -m pytest backend/tests/test_rename_scans.py -q`
Expected: FAIL — `run_migration` undefined (script not yet written)

- [ ] **Step 4: Implement `scripts/scan/rename_scans.py`**

Structure (write complete, following `scripts/scan/promote_to_v3.py` conventions — argparse, `--apply` default-off, archive root arg, atomic writes via `scan_schema.atomic_write_json`):

```python
"""Rename scans + source_ids to the <scene>_<vendor>[_<density>] convention.

Dry-run by default; pass --apply to mutate. The lidar archive is NOT under git,
so --apply first writes a full per-scan-dir backup tarball. See
docs/superpowers/specs/2026-06-26-scan-naming-convention-design.md.
"""
from __future__ import annotations

import argparse
import json
import re
import tarfile
from pathlib import Path

from scan_schema import atomic_write_json, is_valid_scan_name, is_valid_source_id

# Deterministic, reviewed map (spec §4). scans: dir/scan_name; sources: source_id.
RENAME_MAP = {
    "scans": {
        "matterport_mechanical_room": "mechanical_room_matterport",
        "matterport_parkhouse": "parkhouse_matterport",
        "navvis_vlx3_water_treatment": "water_treatment_navvis",
        "smart_ais_clean": "smart_ais_navvis",
        "navvis_mlx": "generator_concrete_navvis",
        "construction_site": "construction_site_navvis",
        "factory_large": "factory_navvis",
        "munich_water_pump": "water_pump_navvis_500k",
        "munich_water_pump_3m": "water_pump_navvis_3m",
    },
    "sources": {
        "construction_site_sample_data": "construction_site_navvis",
        "factory_large": "factory_navvis",
        "navvis_mlx_sample_data": "generator_concrete_navvis",
        "navvis_vlx_3_data_water_treatment_facility": "water_treatment_navvis",
        "sample_data_vlx3_processindustry_smart_ais": "smart_ais_navvis",
        "matterport_mechanical_room": "mechanical_room_matterport",
        "matterport_parkhouse": "parkhouse_matterport",
    },
}

# Architecture: PLAN (pure, build {path: rewritten_obj}) -> RESIDUAL (field-scoped,
# whitelisting the deferred variant_id/labeling_variant namespace + free-text notes)
# -> only THEN mutate (backup, write, rename). The residual gate runs BEFORE any
# disk write, so a missed reference aborts cleanly instead of leaving the archive
# half-migrated. README + notes are reported for manual review, never auto-edited.

# Keys that legitimately retain an old-name-equal value after rename:
#   variant_id / labeling_variant — the deferred namespace (spec non-goal), MUST keep old value
#   notes — free text, hand-reviewed like README
_RETAINED_KEYS = {"variant_id", "labeling_variant"}
_FREETEXT_KEYS = {"notes"}


def _load(p: Path) -> dict:
    return json.loads(p.read_text())


def _sub_name(s, old, new):
    """Replace whole-name occurrences of old with new inside a provenance string,
    on token boundaries (so construction_site doesn't match construction_site_navvis)."""
    return re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", new, s)


def _plan_scan(scan_dir: Path, old: str, new: str, src_map: dict) -> dict:
    """Pure: return {path: rewritten_obj} for one scan. No disk writes."""
    planned: dict[Path, dict] = {}

    mp = scan_dir / "meta.json"
    m = _load(mp)
    m["scan_name"] = new
    fr = m.get("frame")
    if isinstance(fr, dict) and fr.get("canonical_id"):
        fr["canonical_id"] = f"{new}#local"
    d = m.get("derivation")
    if isinstance(d, dict):
        if d.get("scan_id") == old:
            d["scan_id"] = new
        _apply_source_map_to_derivation(d, src_map)   # variant_id left untouched
    planned[mp] = m

    vp = scan_dir / "variants.json"
    if vp.exists():
        v = _load(vp)
        if v.get("scan_id") == old:
            v["scan_id"] = new
        if v.get("canonical_id") == f"{old}#local":
            v["canonical_id"] = f"{new}#local"
        for var in v.get("variants", []):            # variant_id / labeling_variant untouched
            if isinstance(var.get("path"), str):
                var["path"] = _sub_name(var["path"], old, new)
            if var.get("root_source_id") in src_map:
                var["root_source_id"] = src_map[var["root_source_id"]]
            if isinstance(var.get("source"), str):
                var["source"] = _sub_name(var["source"], old, new)
        planned[vp] = v

    rroot = scan_dir / "renders"
    if rroot.is_dir():
        for run in sorted(rroot.iterdir()):
            rm = run / "meta.json"
            if rm.exists():
                j = _load(rm)
                fr = j.get("frame")
                if isinstance(fr, dict) and fr.get("canonical_id"):
                    fr["canonical_id"] = f"{new}#local"
                gf = j.get("generated_from")
                if isinstance(gf, dict):
                    if gf.get("scan_id") == old:
                        gf["scan_id"] = new
                    if isinstance(gf.get("source"), str):
                        gf["source"] = _sub_name(gf["source"], old, new)
                planned[rm] = j
            mf = run / "manifest.json"
            if mf.exists():
                j = _load(mf)
                if j.get("scene") == old:
                    j["scene"] = new
                planned[mf] = j

    sroot = scan_dir / "sessions"
    if sroot.is_dir():
        for sess in sorted(sroot.iterdir()):
            ig = sess / "instances_gt.json"
            if ig.exists():
                j = _load(ig)
                if j.get("scene") == f"annotated/{old}":
                    j["scene"] = f"annotated/{new}"
                planned[ig] = j

    mm = scan_dir / "source" / "mesh.meta.json"
    if mm.exists():
        j = _load(mm)
        if j.get("scene") == old:
            j["scene"] = new
        planned[mm] = j

    return planned


def _apply_source_map_to_derivation(d: dict, src_map: dict):
    root = d.get("root")
    if isinstance(root, dict) and root.get("source_id") in src_map:
        root["source_id"] = src_map[root["source_id"]]
    parent = d.get("parent")
    if isinstance(parent, dict) and parent.get("ref") in src_map:
        parent["ref"] = src_map[parent["ref"]]


def _plan_sources(root: Path, src_map: dict) -> dict:
    sp = root / "raw" / "sources.json"
    body = _load(sp)
    for e in body.get("sources", []):
        if e.get("source_id") in src_map:
            e["source_id"] = src_map[e["source_id"]]
    return {sp: body}


def _residual(view: dict, olds: set) -> tuple[list[str], list[str]]:
    """Recursively scan every string VALUE in the post-migration `view`
    ({path: obj}) for a token-boundary occurrence of any old name.
    Returns (fail_hits, review_hits): hits under _RETAINED_KEYS are ignored
    (expected); hits under _FREETEXT_KEYS are 'review' (reported, non-fatal);
    everything else is a 'fail' (a reference the migration missed)."""
    pats = {o: re.compile(rf"(?<![A-Za-z0-9_]){re.escape(o)}(?![A-Za-z0-9_])") for o in olds}
    fail, review = [], []

    def walk(obj, where, key=None):
        if isinstance(obj, dict):
            for k, v in obj.items():
                walk(v, where, k)
        elif isinstance(obj, list):
            for v in obj:
                walk(v, where, key)
        elif isinstance(obj, str):
            if key in _RETAINED_KEYS:
                return
            for o, p in pats.items():
                if p.search(obj):
                    msg = f"{where} [{key}]: residual {o!r} in {obj!r}"
                    (review if key in _FREETEXT_KEYS else fail).append(msg)

    for path, obj in view.items():
        walk(obj, path)
    return fail, review


def _backup(root: Path, scan_olds, tar_path: Path, log: list[str]):
    log.append(f"  backup -> {tar_path}")
    with tarfile.open(tar_path, "w") as t:
        for old in scan_olds:
            t.add(root / "annotated" / old, arcname=f"annotated/{old}")
        t.add(root / "raw" / "sources.json", arcname="raw/sources.json")


def run_migration(root: Path, rename_map: dict, apply: bool,
                  backup_name: str = "rename_backup.tar") -> dict:
    root = Path(root)
    scan_map, src_map = rename_map["scans"], rename_map["sources"]
    olds = set(scan_map) | set(src_map)
    log: list[str] = []

    # Preconditions (by OLD name; nothing mutated yet).
    for old, new in scan_map.items():
        assert (root / "annotated" / old).is_dir(), f"missing scan {old}"
        assert not (root / "annotated" / new).exists(), f"collision {new}"
        assert is_valid_scan_name(new), f"target not valid: {new}"
    for new in src_map.values():
        assert is_valid_source_id(new), f"target not valid: {new}"

    # PLAN — build the full post-migration view (pure).
    planned: dict[Path, dict] = {}
    for old, new in scan_map.items():
        planned.update(_plan_scan(root / "annotated" / old, old, new, src_map))
    planned.update(_plan_sources(root, src_map))

    # RESIDUAL (before mutating): planned objects + every other *.json (read from
    # disk, .bak* skipped) so an un-rewritten reference is caught, not written over.
    view = dict(planned)
    for jf in (root / "annotated").rglob("*.json"):
        if ".bak" in jf.name or jf in planned:
            continue
        view[jf] = _load(jf)
    fail, review = _residual(view, olds)
    for r in review:
        log.append(f"  REVIEW (free-text, not auto-edited): {r}")
    for old in scan_map:
        rd = root / "annotated" / old / "README.md"
        if rd.exists():
            log.append(f"  REVIEW (free text, not auto-edited): {rd}")
    if fail:
        raise SystemExit(
            "RESIDUAL old names in un-rewritten id-fields (fix the rewriter):\n"
            + "\n".join(fail))

    for p in planned:
        log.append(f"  rewrite {p}")

    # MUTATE — only now. Backup first (archive is not under git).
    if apply:
        _backup(root, scan_map.keys(), root / backup_name, log)
        for p, obj in planned.items():
            atomic_write_json(p, obj)
        for old, new in scan_map.items():
            (root / "annotated" / old).rename(root / "annotated" / new)

    print("\n".join(log))
    return {"log": log, "planned": [str(p) for p in planned], "review": review}


def main(argv=None):
    from datetime import datetime, timezone
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("root", type=Path)
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_migration(a.root, RENAME_MAP, apply=a.apply,
                  backup_name=f"rename_backup_{stamp}.tar")
    print(f"\n{'APPLIED' if a.apply else 'DRY-RUN (no changes)'}")


if __name__ == "__main__":
    main()
```

> Note: the residual gate runs in **both** dry-run and apply, against the in-memory
> post-migration view, **before** any disk write — so a missed reference aborts with
> the archive untouched. Hits under `variant_id`/`labeling_variant` (deferred namespace)
> are expected and ignored; `notes`/README hits are reported for manual review, not failed.

- [ ] **Step 5: Run to verify it passes**

Run: `.venv/bin/python -m pytest backend/tests/test_rename_scans.py -q`
Expected: PASS (both `test_dry_run_changes_nothing` and `test_apply_renames_and_rewrites_all_refs`)

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml scripts/scan/rename_scans.py backend/tests/test_rename_scans.py
git commit -m "feat: rename_scans migration to <scene>_<vendor> convention"
```

### Task B3: Dry-run on the real archive + review

**Files:** none (operational)

- [ ] **Step 1: Dry-run**

Run: `.venv/bin/python scripts/scan/rename_scans.py /home/hendrik/coding/engine/data/lidar`
Expected: prints every file + field it would rewrite for all 9 scans + sources.json, then `DRY-RUN (no changes)`. **Read the full output**; confirm it matches the spec §4 map and lists each README for manual review.

- [ ] **Step 2: Pause for human review** — surface the dry-run output and get explicit go-ahead before `--apply`.

### Task B4: Apply + verify gates

**Files:** none (mutates the archive)

> ⚠️ **Labeled-scan safety** (`munich_*`, `water_treatment_navvis`, `smart_ais_navvis` hold real
> labels — see [[feedback_labeled_scans_precious]]): the migration rewrites only JSON id-strings
> and must NOT alter any label array. Step 1b/Step 5 below *prove* this by sha256-comparing every
> `.npy` before and after.

- [ ] **Step 1: Baseline** — `.venv/bin/python -m scan_schema /home/hendrik/coding/engine/data/lidar` → confirm `0 errors` (record the naming warnings present).
- [ ] **Step 1b: Snapshot label-array hashes** — record a sha256 of every `.npy` under `annotated/` keyed by path-relative-to-its-scan (so a rename doesn't look like a change):

```bash
cd /home/hendrik/coding/engine/data/lidar/annotated
find . -name '*.npy' | sed 's#^\./[^/]*/##' | sort -u >/tmp/npy_rel_before.txt   # sanity: relpath set
find . -name '*.npy' -exec sha256sum {} \; | awk '{n=$2; sub(/^\.\/[^/]*\//,"",n); print $1, n}' | sort -k2 >/tmp/npy_before.txt
wc -l /tmp/npy_before.txt
```

- [ ] **Step 2: Apply** — `.venv/bin/python scripts/scan/rename_scans.py /home/hendrik/coding/engine/data/lidar --apply` → expect a backup tarball path printed, the rewrite log, no residual error, `APPLIED`.
- [ ] **Step 3: Postcondition** — `.venv/bin/python -m scan_schema /home/hendrik/coding/engine/data/lidar` → expect `0 errors` AND **no** `scan_name ... does not match` / `source_id ...` warnings.
- [ ] **Step 4: Test sweep** — `VOXA_LIDAR_ROOT=/home/hendrik/coding/engine/data/lidar .venv/bin/python -m pytest backend/tests/test_real_scans_validate.py -q` → all green.
- [ ] **Step 5: Prove labels untouched** — re-hash and diff against the snapshot (keyed by scan-relative path, so the dir rename is ignored); **must be identical**:

```bash
cd /home/hendrik/coding/engine/data/lidar/annotated
find . -name '*.npy' -exec sha256sum {} \; | awk '{n=$2; sub(/^\.\/[^/]*\//,"",n); print $1, n}' | sort -k2 >/tmp/npy_after.txt
diff /tmp/npy_before.txt /tmp/npy_after.txt && echo "LABELS BYTE-IDENTICAL ✓"
```
Expected: no diff, `LABELS BYTE-IDENTICAL ✓`. If anything differs, STOP and restore from the backup tarball.

- [ ] **Step 6: Manually edit each README.md** to the new name (the script reported them; free text, hand-edited). Re-run Step 3 to confirm still clean.

### Task B5: Docs

**Files:**
- Modify: `voxa/docs/scan-schema.md`
- Modify: `data/lidar/SCHEMA.md`

- [ ] **Step 1: Document the convention** — add a "Naming convention" subsection to both docs: the grammar, `KNOWN_VENDORS`, the density guideline, and that enforcement is warn-level in `scan_schema`. In `voxa/docs/scan-schema.md` also reconcile the README "(required)" vs "(recommended)" wording flagged in the spec.
- [ ] **Step 2: Commit**

```bash
git add docs/scan-schema.md
git commit -m "docs: document scan/source naming convention"
# (data/lidar/SCHEMA.md is outside git — edit in place; note it in the commit body)
```

---

## Out of scope (separate plans)

- **munich v3.0 lineage promotion** (register `mesh.glb` root + write derivation). Cross-links here: munich's new root must be named `water_pump_navvis`.
- **Stray-dir cleanup** (`fresh_run/`, `viz_segments.ply`, `.bak*`). The residual scan whitelists `.bak*`; cleanup removes them.
- `variant_id` / `session_id` / `preseg_id` naming.

## Final verification checklist

- [ ] scan-schema: full suite green; `__version__ == 3.1.0`; pushed to `main`.
- [ ] voxa venv on scan_schema 3.1.0.
- [ ] `python -m scan_schema <archive>` → 0 errors, **no naming warnings**.
- [ ] `test_real_scans_validate.py` green.
- [ ] All 9 dirs renamed; backup tarball exists.
- [ ] READMEs updated; docs updated.
