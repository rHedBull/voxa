# Eval Invariants + Manifest (eval-labeling phase 3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the 9 eval-labeling loader invariants and a scan-level manifest generator in the shared `scan_schema` package, wire them into voxa's save-time gate as hard failures, and migrate the three precious pre-phase-2 scans so they pass under the new gate.

**Architecture:** Two repos, two branches. `scan-schema` (`/home/hendrik/coding/engine/tools/scan-schema`) gets a new `eval_invariants.py` (9 pure-function checks) and `manifest.py` (manifest builder), both wired into `validate_archive` for load-time auditing. `voxa` (`/home/hendrik/coding/engine/tools/labeling/voxa`) wires the same functions into `segment_io.save_labels` for save-time gating, adds a migration script for the precious scans, and — once the scan_schema branch is merged upstream — bumps its dependency pin to that commit.

**Tech Stack:** Python, numpy, pytest, FastAPI (voxa backend). No frontend changes in this phase.

**Spec:** `docs/superpowers/specs/2026-07-22-eval-invariants-manifest-design.md`

---

## File Structure

**scan-schema** (new branch `feat/eval-invariants-manifest`, own worktree):
- Create: `src/scan_schema/eval_invariants.py` — 9 invariant-check functions, pure over parsed arrays/dicts.
- Create: `tests/test_eval_invariants.py` — one pass + one fail test per invariant.
- Create: `src/scan_schema/manifest.py` — `build_manifest(...)`.
- Create: `tests/test_manifest.py`.
- Modify: `src/scan_schema/layout.py` — add `output_gt_point_category` / `output_gt_point_component_ids` to `SessionPaths`.
- Modify: `src/scan_schema/validate.py` — call the new checks per-session inside `validate_archive`, alongside the existing invariant-3-6 call.
- Modify: `src/scan_schema/__main__.py` — new `manifest <scan_dir> <session_id>` subcommand (load-time manifest regeneration/audit, per the spec's "same code" goal).

**voxa** (new branch `feat/eval-invariants-manifest`, own worktree; depends on the scan_schema branch above being installed locally during development):
- Modify: `backend/labeling/segment_io.py` — `save_labels` gains `instances_doc`/`review_blobs_prior`/`eval_regions` params, calls `eval_invariants.check_all` before writing, merges `manifest.build_manifest(...)` output into `gt_segment_metadata.json`.
- Create: `backend/labeling/instances_doc.py` — thin loader for `instances_gt.json` (the file `backend/routes/compare.py` already reads/writes at a known path; `segment_io` needs its own read-only loader since it must not depend on FastAPI route code).
- Modify: `backend/routes/segment.py` — `segment_save()` loads `instances_gt.json` + `eval_regions.json` and the prior save's `gt_segment_metadata.json`, passes them to `save_labels`, and translates a new `EvalInvariantError` into a structured 422.
- Create: `scripts/migrate_eval_invariants.py` — one-off migration for the three precious scans.
- Modify: `requirements.txt` — pin bump (final task, after the scan_schema branch merges).
- Modify: `docs/scan-schema.md` — note the pin-bump discipline.

---

## Task 1: scan_schema — invariants 1 & 2 (category exhaustiveness, review budget)

**Files:**
- Create: `src/scan_schema/eval_invariants.py`
- Test: `tests/test_eval_invariants.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_eval_invariants.py
import numpy as np
import pytest
from scan_schema.eval_invariants import check_category_exhaustive, check_review_budget

CAT_NONE, CAT_ARTIFACT, CAT_TRANSIENT, CAT_EXCLUDED_REVIEW = 0, 1, 2, 3


def test_check_category_exhaustive_ok():
    categories = np.array([0, 1, 2, 3], dtype=np.int8)
    in_region = np.array([True, True, True, True])
    check_category_exhaustive(categories, in_region)  # no raise


def test_check_category_exhaustive_bad_value():
    categories = np.array([0, 9], dtype=np.int8)
    in_region = np.array([True, True])
    with pytest.raises(ValueError, match="eval-invariant 1"):
        check_category_exhaustive(categories, in_region)


def test_check_review_budget_within():
    categories = np.array([0] * 97 + [3] * 3, dtype=np.int8)  # 3%
    check_review_budget(categories, budget_frac=0.03)  # no raise


def test_check_review_budget_over():
    categories = np.array([0] * 96 + [3] * 4, dtype=np.int8)  # 4%
    with pytest.raises(ValueError, match="eval-invariant 2"):
        check_review_budget(categories, budget_frac=0.03)
```

- [ ] **Step 2: Run tests, verify they fail**

Run: `cd /home/hendrik/coding/engine/tools/scan-schema && .venv/bin/pytest tests/test_eval_invariants.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'scan_schema.eval_invariants'`

- [ ] **Step 3: Implement**

```python
# src/scan_schema/eval_invariants.py
"""Eval-labeling loader invariants (upstream spec, 'Loader invariants
(checkable)' section) — a distinct, GT-semantics-level set from the
array-shape/class-consistency SCHEMA invariants in invariants.py (numbered
separately here as eval-invariants 1-9 to avoid collision)."""
from __future__ import annotations

from typing import Optional

import numpy as np

CATEGORY_NONE, CATEGORY_ARTIFACT, CATEGORY_TRANSIENT, CATEGORY_EXCLUDED_REVIEW = 0, 1, 2, 3
_VALID_CATEGORIES = {CATEGORY_NONE, CATEGORY_ARTIFACT, CATEGORY_TRANSIENT, CATEGORY_EXCLUDED_REVIEW}


def check_category_exhaustive(categories: np.ndarray, in_region: np.ndarray) -> None:
    """eval-invariant 1: every in-region point has exactly one of the 4 stored
    category values (confirmed-thing/stuff are derived, never stored, so they
    don't appear here)."""
    bad = np.isin(categories[in_region], list(_VALID_CATEGORIES), invert=True)
    if bad.any():
        raise ValueError(
            f"eval-invariant 1: {int(bad.sum())} in-region points have an "
            "invalid category value")


def check_review_budget(categories: np.ndarray, budget_frac: float = 0.03,
                        in_region: Optional[np.ndarray] = None) -> None:
    """eval-invariant 2: excluded_review points <= budget_frac of the
    (region's) points."""
    scope = categories if in_region is None else categories[in_region]
    if scope.size == 0:
        return
    frac = float((scope == CATEGORY_EXCLUDED_REVIEW).sum()) / scope.size
    if frac > budget_frac:
        raise ValueError(
            f"eval-invariant 2: excluded_review fraction {frac:.4f} exceeds "
            f"budget {budget_frac:.4f}")
```

- [ ] **Step 4: Run tests, verify they pass**

Run: `cd /home/hendrik/coding/engine/tools/scan-schema && .venv/bin/pytest tests/test_eval_invariants.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
cd /home/hendrik/coding/engine/tools/scan-schema
git add src/scan_schema/eval_invariants.py tests/test_eval_invariants.py
git commit -m "feat: eval-invariants 1-2 (category exhaustiveness, review budget)"
```

## Task 2: scan_schema — invariant 3 (instance/class/review-blob consistency)

**Files:**
- Modify: `src/scan_schema/eval_invariants.py`
- Test: `tests/test_eval_invariants.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_check_instance_class_consistency_ok():
    segment_ids = np.array([-1, 5, 5, 7], dtype=np.int32)
    class_ids = np.array([-1, 2, 2, -1], dtype=np.int32)
    categories = np.array([0, 0, 0, 3], dtype=np.int8)
    instances = {5: {"class_id": 2}}          # ordinary instance, real class
    review_blobs = [{"instance_id": 7, "n_points": 1}]
    check_instance_class_consistency(segment_ids, class_ids, categories,
                                     instances, review_blobs)  # no raise


def test_check_instance_class_consistency_missing_instance():
    segment_ids = np.array([5], dtype=np.int32)
    class_ids = np.array([2], dtype=np.int32)
    categories = np.array([0], dtype=np.int8)
    with pytest.raises(ValueError, match="eval-invariant 3"):
        check_instance_class_consistency(segment_ids, class_ids, categories,
                                         instances={}, review_blobs=[])


def test_check_instance_class_consistency_review_point_without_blob():
    # excluded_review point but instance 7 isn't recorded as a review blob
    segment_ids = np.array([-1], dtype=np.int32)   # stripped, as save path does
    class_ids = np.array([-1], dtype=np.int32)
    categories = np.array([3], dtype=np.int8)
    with pytest.raises(ValueError, match="eval-invariant 3"):
        check_instance_class_consistency(segment_ids, class_ids, categories,
                                         instances={}, review_blobs=[])
```

- [ ] **Step 2: Run, verify fail** — `NameError`/`ImportError` on undefined function.

- [ ] **Step 3: Implement**

```python
def check_instance_class_consistency(
    segment_ids: np.ndarray, class_ids: np.ndarray, categories: np.ndarray,
    instances: dict, review_blobs: list[dict],
) -> None:
    """eval-invariant 3. `instances` is {segId: {"class_id": int|None, ...}}
    built from instances_gt.json (segId is the join key to segment_ids, per
    Cuboid.segId). `review_blobs` is gt_segment_metadata.json's list —
    review-blob ids are NEVER present in segment_ids (the existing save-path
    strip removes class-less instances before this runs), so their only
    representation here is this list."""
    labeled = segment_ids >= 0
    present_ids = set(np.unique(segment_ids[labeled]).tolist())
    missing = present_ids - set(instances.keys())
    if missing:
        raise ValueError(
            f"eval-invariant 3: instance ids {sorted(missing)} in "
            "gt_segment_ids.npy have no entry in instances_gt.json")
    for seg_id in present_ids:
        entry = instances[seg_id]
        if entry.get("class_id") is None:
            raise ValueError(
                f"eval-invariant 3: instance {seg_id} has null class but is "
                "present in gt_segment_ids.npy — review blobs must not "
                "appear there (save-path strip should have removed it)")

    review_ids = {b["instance_id"] for b in review_blobs}
    n_review_points = int((categories == CATEGORY_EXCLUDED_REVIEW).sum())
    n_accounted = sum(b["n_points"] for b in review_blobs)
    if n_review_points != n_accounted:
        raise ValueError(
            f"eval-invariant 3: {n_review_points} excluded_review points but "
            f"review_blobs accounts for only {n_accounted}")
    for seg_id in review_ids:
        entry = instances.get(seg_id)
        if entry is not None and entry.get("class_id") is not None:
            raise ValueError(
                f"eval-invariant 3: review blob {seg_id} has a non-null "
                "class in instances_gt.json")
```

- [ ] **Step 4: Run tests, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/scan_schema/eval_invariants.py tests/test_eval_invariants.py
git commit -m "feat: eval-invariant 3 (instance/class/review-blob consistency)"
```

## Task 3: scan_schema — invariant 4 (frozen class ids)

**Files:**
- Modify: `src/scan_schema/eval_invariants.py`
- Test: `tests/test_eval_invariants.py`

- [ ] **Step 1: Write failing tests**

```python
def test_check_no_frozen_classes_ok():
    class_ids = np.array([-1, 2, 10], dtype=np.int32)
    check_no_frozen_classes(class_ids, frozen_ids={0, 3, 4, 5, 6, 13})  # no raise


def test_check_no_frozen_classes_violation():
    class_ids = np.array([6, 2], dtype=np.int32)
    with pytest.raises(ValueError, match="eval-invariant 4"):
        check_no_frozen_classes(class_ids, frozen_ids={0, 3, 4, 5, 6, 13})
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
def check_no_frozen_classes(class_ids: np.ndarray, frozen_ids: set[int]) -> None:
    """eval-invariant 4. `frozen_ids` is the caller's frozen-class set (voxa
    passes `backend.app.core.frozen_class_ids()` — read from classes.yaml's
    `frozen: true` flags, currently {0,3,4,5,6,13}; not hardcoded here so a
    future frozen-class addition needs no scan_schema change)."""
    present = set(np.unique(class_ids[class_ids >= 0]).tolist())
    bad = present & set(frozen_ids)
    if bad:
        raise ValueError(
            f"eval-invariant 4: frozen legacy class ids {sorted(bad)} present "
            "in new GT")
```

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/scan_schema/eval_invariants.py tests/test_eval_invariants.py
git commit -m "feat: eval-invariant 4 (no frozen legacy class ids)"
```

## Task 4: scan_schema — invariant 5 (component/instance coverage)

**Files:**
- Modify: `src/scan_schema/eval_invariants.py`
- Test: `tests/test_eval_invariants.py`

- [ ] **Step 1: Write failing tests**

```python
def test_check_component_instance_coverage_ok():
    segment_ids = np.array([-1, 5, 5], dtype=np.int32)
    component_ids = np.array([-1, 0, 1], dtype=np.int16)
    check_component_instance_coverage(segment_ids, component_ids)  # no raise


def test_check_component_instance_coverage_mismatch():
    segment_ids = np.array([5, -1], dtype=np.int32)
    component_ids = np.array([-1, -1], dtype=np.int16)   # point 0 missing a component
    with pytest.raises(ValueError, match="eval-invariant 5"):
        check_component_instance_coverage(segment_ids, component_ids)
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
def check_component_instance_coverage(segment_ids: np.ndarray, component_ids: np.ndarray) -> None:
    """eval-invariant 5: instance_id >= 0 <=> component_id >= 0, pointwise."""
    has_inst = segment_ids >= 0
    has_comp = component_ids >= 0
    if not np.array_equal(has_inst, has_comp):
        n_bad = int(np.sum(has_inst != has_comp))
        raise ValueError(
            f"eval-invariant 5: instance/component coverage mismatch at "
            f"{n_bad} points")
```

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/scan_schema/eval_invariants.py tests/test_eval_invariants.py
git commit -m "feat: eval-invariant 5 (component/instance coverage agreement)"
```

## Task 5: scan_schema — invariant 7 (instance-id lineage across saves)

**Files:**
- Modify: `src/scan_schema/eval_invariants.py`
- Test: `tests/test_eval_invariants.py`

- [ ] **Step 1: Write failing tests**

```python
def test_check_id_lineage_stable_ok():
    prior_ids = {5, 7}      # union of prior segments[].gt_id and review_blobs[].instance_id
    current_ids = {5, 7, 9}  # 9 is new — fine
    check_id_lineage(prior_ids, current_ids)  # no raise


def test_check_id_lineage_lost_id():
    prior_ids = {5, 7}
    current_ids = {5}       # 7 vanished
    with pytest.raises(ValueError, match="eval-invariant 7"):
        check_id_lineage(prior_ids, current_ids)
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
def check_id_lineage(prior_ids: set[int], current_ids: set[int]) -> None:
    """eval-invariant 7: no instance id present in a prior save may be absent
    from the current save (ids are never reused/renumbered — the stable-id
    contract the future relations layer depends on). New ids are always fine.
    Callers build both sets as the union of ordinary-instance ids (from
    gt_segment_metadata.json's `segments[].gt_id`) and review-blob ids (from
    `review_blobs[].instance_id`) — an id moving from one list to the other
    (a blob later given a real class) is not a loss, since the id itself
    persists; this check only compares the union."""
    lost = prior_ids - current_ids
    if lost:
        raise ValueError(
            f"eval-invariant 7: instance ids {sorted(lost)} present in a "
            "prior save are missing from this save — ids must never be "
            "reused or renumbered")
```

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/scan_schema/eval_invariants.py tests/test_eval_invariants.py
git commit -m "feat: eval-invariant 7 (instance-id lineage across saves)"
```

## Task 6: scan_schema — invariant 8 (eval-grade accuracy band consistency)

**Files:**
- Modify: `src/scan_schema/eval_invariants.py`
- Test: `tests/test_eval_invariants.py`

- [ ] **Step 1: Write failing tests**

```python
def test_check_accuracy_band_ok():
    regions = [{"id": "r1", "status": "eval_grade", "accuracy": {"p90": 0.008}}]
    check_accuracy_band(regions, p90_ceiling=0.010)  # no raise


def test_check_accuracy_band_violation():
    regions = [{"id": "r1", "status": "eval_grade", "accuracy": {"p90": 0.015}}]
    with pytest.raises(ValueError, match="eval-invariant 8"):
        check_accuracy_band(regions, p90_ceiling=0.010)


def test_check_accuracy_band_ignores_draft():
    regions = [{"id": "r1", "status": "draft", "accuracy": {"p90": 0.5}}]
    check_accuracy_band(regions, p90_ceiling=0.010)  # no raise, draft unchecked
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
def check_accuracy_band(regions: list[dict], p90_ceiling: float = 0.010) -> None:
    """eval-invariant 8: every eval_grade region's recorded accuracy.p90 must
    be <= p90_ceiling (10mm). Draft regions are unchecked — the bar is an
    eval-grade admission condition, already enforced once at flip time
    (regions.py::flip_status); this re-verifies it hasn't drifted since."""
    for r in regions:
        if r.get("status") != "eval_grade":
            continue
        p90 = (r.get("accuracy") or {}).get("p90")
        if p90 is None or p90 > p90_ceiling:
            raise ValueError(
                f"eval-invariant 8: region {r.get('id')!r} is eval_grade but "
                f"accuracy.p90={p90} exceeds ceiling {p90_ceiling}")
```

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/scan_schema/eval_invariants.py tests/test_eval_invariants.py
git commit -m "feat: eval-invariant 8 (eval-grade accuracy band consistency)"
```

## Task 7: scan_schema — invariant 9 (confirmed/category reconciliation)

**Files:**
- Modify: `src/scan_schema/eval_invariants.py`
- Test: `tests/test_eval_invariants.py`

- [ ] **Step 1: Write failing tests**

```python
def test_check_confirmed_reconciliation_ok():
    segment_ids = np.array([5, 5], dtype=np.int32)
    categories = np.array([0, 0], dtype=np.int8)
    instances = {5: {"class_id": 2, "confirmed": True}}
    check_confirmed_reconciliation(segment_ids, categories, instances)  # no raise


def test_check_confirmed_reconciliation_marked_after_confirm():
    segment_ids = np.array([5], dtype=np.int32)
    categories = np.array([1], dtype=np.int8)   # artifact
    instances = {5: {"class_id": 2, "confirmed": True}}
    with pytest.raises(ValueError, match="eval-invariant 9"):
        check_confirmed_reconciliation(segment_ids, categories, instances)


def test_check_confirmed_reconciliation_confirmed_review_blob():
    instances = {7: {"class_id": None, "confirmed": True}}
    segment_ids = np.array([], dtype=np.int32)
    categories = np.array([], dtype=np.int8)
    with pytest.raises(ValueError, match="eval-invariant 9"):
        check_confirmed_reconciliation(segment_ids, categories, instances)
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement**

```python
def check_confirmed_reconciliation(
    segment_ids: np.ndarray, categories: np.ndarray, instances: dict,
) -> None:
    """eval-invariant 9 (discovered in phase-3 scoping): a `confirmed`
    instance's points must all carry category `none`, and a confirmed
    instance can never be a review blob (class_id is None)."""
    for seg_id, entry in instances.items():
        if not entry.get("confirmed"):
            continue
        if entry.get("class_id") is None:
            raise ValueError(
                f"eval-invariant 9: instance {seg_id} is confirmed but has "
                "no class (review blobs cannot be confirmed)")
        mask = segment_ids == seg_id
        if mask.any() and (categories[mask] != CATEGORY_NONE).any():
            raise ValueError(
                f"eval-invariant 9: confirmed instance {seg_id} has points "
                "with a non-none category")
```

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/scan_schema/eval_invariants.py tests/test_eval_invariants.py
git commit -m "feat: eval-invariant 9 (confirmed/category reconciliation)"
```

## Task 8: scan_schema — layout.py accessors for the phase-2 array files

**Files:**
- Modify: `src/scan_schema/layout.py:49` (add after `output_gt_segment_metadata`)
- Test: `tests/test_layout.py` (create if it doesn't exist, else add to it — check first: `ls tests/test_layout.py`)

- [ ] **Step 1: Write the failing test**

```python
def test_session_paths_phase2_arrays(tmp_path):
    from scan_schema.layout import SessionPaths
    sp = SessionPaths(tmp_path / "sessions" / "s1")
    assert sp.output_gt_point_category == sp.output_dir / "gt_point_category.npy"
    assert sp.output_gt_point_component_ids == sp.output_dir / "gt_point_component_ids.npy"
```

- [ ] **Step 2: Run, verify fail** (`AttributeError`).

- [ ] **Step 3: Implement**

```python
    @property
    def output_gt_point_category(self) -> Path:
        return self.output_dir / "gt_point_category.npy"

    @property
    def output_gt_point_component_ids(self) -> Path:
        return self.output_dir / "gt_point_component_ids.npy"
```

(insert into `SessionPaths` right after `output_gt_segment_metadata`)

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/scan_schema/layout.py tests/test_layout.py
git commit -m "feat: layout accessors for gt_point_category / gt_point_component_ids"
```

## Task 9: scan_schema — manifest.py

**Files:**
- Create: `src/scan_schema/manifest.py`
- Test: `tests/test_manifest.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_manifest.py
import numpy as np
from scan_schema.manifest import build_manifest


def test_build_manifest_basic_fields():
    class_ids = np.array([-1, 2, 2, 5], dtype=np.int32)
    categories = np.array([0, 0, 0, 3], dtype=np.int8)
    m = build_manifest(
        class_ids=class_ids, categories=categories,
        class_map_version=7, regions=[], review_budget_frac=0.03,
        size_floor_m=0.05, canonical_spacing_m=0.005,
    )
    assert m["class_histogram"] == {"-1": 1, "2": 2, "5": 1}
    assert m["category_histogram"]["excluded_review"] == 1
    assert m["class_map_version"] == 7
    assert m["thresholds"]["review_budget_frac"] == 0.03
    assert m["thresholds"]["size_floor_m"] == 0.05


def test_build_manifest_in_region_fraction():
    class_ids = np.array([2, 2, -1, -1], dtype=np.int32)
    categories = np.array([0, 0, 0, 0], dtype=np.int8)
    regions = [{"id": "r1", "mask": [True, True, False, False]}]
    m = build_manifest(class_ids=class_ids, categories=categories,
                       class_map_version=7, regions=regions,
                       review_budget_frac=0.03, size_floor_m=0.05,
                       canonical_spacing_m=0.005)
    assert m["in_region_fraction"] == 1.0  # both labeled points are in-region
```

- [ ] **Step 2: Run, verify fail** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# src/scan_schema/manifest.py
"""Scan-level manifest generator (eval-labeling phase 3). Regenerated on
every save; drift between a stored manifest and a freshly-built one is
eval-invariant 6, checked by the caller (voxa's save gate, or a future
harness) comparing two build_manifest(...) outputs — this module only
builds, it does not persist or diff."""
from __future__ import annotations

from typing import Optional

import numpy as np


def build_manifest(
    *,
    class_ids: np.ndarray,
    categories: np.ndarray,
    class_map_version: int,
    regions: list[dict],
    review_budget_frac: float,
    size_floor_m: float,
    canonical_spacing_m: float,
    edge_band_width_m: Optional[float] = None,
    provenance: Optional[dict] = None,
) -> dict:
    from scan_schema.eval_invariants import (
        CATEGORY_NONE, CATEGORY_ARTIFACT, CATEGORY_TRANSIENT, CATEGORY_EXCLUDED_REVIEW,
    )

    labeled = class_ids >= 0
    ids, counts = np.unique(class_ids[labeled], return_counts=True) if labeled.any() else ([], [])
    class_histogram = {str(int(i)): int(c) for i, c in zip(ids, counts)}
    if int((~labeled).sum()):
        class_histogram["-1"] = int((~labeled).sum())

    names = {CATEGORY_NONE: "none", CATEGORY_ARTIFACT: "artifact",
             CATEGORY_TRANSIENT: "transient", CATEGORY_EXCLUDED_REVIEW: "excluded_review"}
    category_histogram = {name: int((categories == val).sum()) for val, name in names.items()}

    region_summaries = []
    n_in_region = 0
    for r in regions:
        mask = np.asarray(r.get("mask", []), dtype=bool)
        region_summaries.append({
            "id": r.get("id"),
            "category_histogram": {
                name: int((categories[mask] == val).sum()) if mask.size else 0
                for val, name in names.items()
            },
            "accuracy": r.get("accuracy"),
        })
        n_in_region += int(mask.sum())

    n_labeled = int(labeled.sum())
    in_region_fraction = (
        min(1.0, n_in_region / n_labeled) if n_labeled else 0.0
    )

    return {
        "class_histogram": class_histogram,
        "category_histogram": category_histogram,
        "class_map_version": class_map_version,
        "in_region_fraction": in_region_fraction,
        "regions": region_summaries,
        "thresholds": {
            "review_budget_frac": review_budget_frac,
            "size_floor_m": size_floor_m,
            "edge_band_width_m": edge_band_width_m,
        },
        "canonical_spacing_m": canonical_spacing_m,
        "provenance": provenance,
    }
```

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/scan_schema/manifest.py tests/test_manifest.py
git commit -m "feat: scan-level manifest generator"
```

## Task 10: scan_schema — wire eval invariants + manifest into validate_archive

**Files:**
- Modify: `src/scan_schema/validate.py` (inside the `sessions/<id>/` loop, after the existing `validate_invariants` call at line ~198)
- Test: `tests/test_validate.py` (check existing file name first: `ls tests/test_validate*.py`)

- [ ] **Step 1: Write the failing test**

```python
def test_validate_archive_reports_eval_invariant_violation(tmp_path):
    # Build a minimal scan with a frozen class id 6 in gt_class_ids.npy and
    # confirm validate_archive surfaces it as an error (not silently passing).
    # (Full fixture construction: reuse the existing conftest helpers that
    # build a minimal annotated/<scan>/ tree with meta.json + sessions/<id>/
    # output/*.npy + eval_regions.json — see how other test_validate_archive_*
    # tests in this file build their fixture and follow the same helper.)
    ...
    report = validate_archive(tmp_path)
    assert any("eval-invariant 4" in e for e in report["my_scan"]["errors"])
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** — inside `validate_archive`'s per-session block, after the existing `validate_invariants(cls_arr, inst_arr)` call, load `gt_point_category.npy`, `gt_point_component_ids.npy` (via the new layout accessors), `instances_gt.json`, `eval_regions.json`, and the session's `gt_segment_metadata.json`, and run each `eval_invariants.check_*` function, appending failures to `errors` in the same `f"{session_dir.name}/output: {exc}"` style as the existing call. Missing optional inputs (e.g. no `eval_regions.json` — not every scan has eval regions) skip the invariants that need them rather than erroring — a pre-phase-1 scan is not itself invalid for lacking eval regions.

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add src/scan_schema/validate.py tests/test_validate.py
git commit -m "feat: wire eval invariants into validate_archive load-time audit"
```

## Task 11: scan_schema — open PR, get it merged, note the commit SHA

- [ ] **Step 1:** Push the branch, open a PR against `scan-schema`'s main.
- [ ] **Step 2:** Once merged, record the merge commit SHA — voxa's Task 15 needs it.

---

## Task 12: voxa — instances_gt.json loader for segment_io

**Files:**
- Create: `backend/labeling/instances_doc.py`
- Test: `backend/tests/test_instances_doc.py`

- [ ] **Step 1: Write the failing test**

```python
# backend/tests/test_instances_doc.py
import json
from labeling.instances_doc import load_instances_for_invariants


def test_load_instances_for_invariants(tmp_path):
    session_dir = tmp_path / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    doc = {
        "scene": "x", "kind": "gt",
        "instances": [
            {"id": "a", "kind": "pointset", "segId": 5, "cls": "pipe", "confirmed": True},
            {"id": "b", "kind": "pointset", "segId": 7, "cls": None, "confirmed": False},
        ],
    }
    (session_dir / "instances_gt.json").write_text(json.dumps(doc))
    result = load_instances_for_invariants(session_dir)
    assert result == {
        5: {"class_id": "pipe", "confirmed": True},
        7: {"class_id": None, "confirmed": False},
    }


def test_load_instances_for_invariants_missing_file(tmp_path):
    session_dir = tmp_path / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    assert load_instances_for_invariants(session_dir) == {}
```

- [ ] **Step 2: Run, verify fail** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# backend/labeling/instances_doc.py
"""Read-only loader for instances_gt.json, for the save-time eval-invariant
gate (segment_io.save_labels). Deliberately NOT the same code path as the
route (backend/routes/compare.py's GET/PUT /api/annotations) — this must not
depend on FastAPI, and segment_io is pure I/O by convention."""
from __future__ import annotations

import json
from pathlib import Path


def load_instances_for_invariants(session_dir: Path) -> dict[int, dict]:
    """{segId: {"class_id": cls-or-None, "confirmed": bool}} for every
    kind:'pointset' instance with a segId. Returns {} if instances_gt.json is
    absent (a session with nothing confirmed yet, or a pre-instance-doc
    session) — callers must not treat this as an invariant violation on its
    own; an empty doc with non-empty gt_segment_ids.npy IS still a violation,
    caught by eval-invariant 3 itself."""
    p = session_dir / "instances_gt.json"
    if not p.exists():
        return {}
    raw = json.loads(p.read_text())
    result: dict[int, dict] = {}
    for inst in raw.get("instances", []):
        seg_id = inst.get("segId")
        if inst.get("kind") != "pointset" or seg_id is None:
            continue
        result[int(seg_id)] = {
            "class_id": inst.get("cls"),
            "confirmed": bool(inst.get("confirmed", False)),
        }
    return result
```

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add backend/labeling/instances_doc.py backend/tests/test_instances_doc.py
git commit -m "feat: instances_gt.json loader for the save-time invariant gate"
```

## Task 13: voxa — install scan_schema branch locally for development

- [ ] **Step 1:** In voxa's `.venv`, install the scan_schema branch from Task 11 in editable mode for iteration:

```bash
cd /home/hendrik/coding/engine/tools/labeling/voxa
.venv/bin/pip install -e /home/hendrik/coding/engine/tools/scan-schema
```

This is a local dev-only step — do not commit a change to `requirements.txt` yet (Task 19 does the real pin bump, after Task 11's PR is merged upstream).

## Task 14: voxa — eval_regions.json + prior metadata loaders for the gate

**Files:**
- Modify: `backend/labeling/segment_io.py`
- Test: `backend/tests/test_segment_io.py` (add to existing file)

- [ ] **Step 1: Write the failing tests**

```python
def test_load_eval_regions_for_invariants_missing(tmp_path):
    from labeling.segment_io import load_eval_regions_for_invariants
    assert load_eval_regions_for_invariants(tmp_path) == []


def test_load_prior_segment_metadata_missing(tmp_path):
    from labeling.segment_io import load_prior_segment_metadata
    assert load_prior_segment_metadata(tmp_path) is None
```

- [ ] **Step 2: Run, verify fail.**

- [ ] **Step 3: Implement** (append to `segment_io.py`)

```python
def load_eval_regions_for_invariants(scan_dir: Path) -> list[dict]:
    """Read eval_regions.json (scan root) for the save-time gate. Missing
    file -> [] (a scan with no eval regions yet is not itself invalid)."""
    p = ScanLayout(scan_dir).scan_dir / "eval_regions.json"
    if not p.exists():
        return []
    try:
        return json.loads(p.read_text()).get("regions", [])
    except (OSError, json.JSONDecodeError):
        return []


def load_prior_segment_metadata(scan_dir: Path, session_id: str) -> Optional[dict]:
    """Read the PREVIOUS gt_segment_metadata.json (before this save
    overwrites it) — needed by eval-invariant 7's cross-save id-lineage
    check. Returns None if this is the session's first save."""
    p = ScanLayout(scan_dir).session(session_id).output_gt_segment_metadata
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None
```

- [ ] **Step 4: Run, verify pass.**

- [ ] **Step 5: Commit**

```bash
git add backend/labeling/segment_io.py backend/tests/test_segment_io.py
git commit -m "feat: eval_regions.json + prior-metadata loaders for the invariant gate"
```

## Task 15: voxa — wire eval_invariants.check_all + manifest into save_labels

**Files:**
- Modify: `backend/labeling/segment_io.py::save_labels`
- Test: `backend/tests/test_segment_io.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_save_labels_rejects_frozen_class(tmp_path):
    from labeling.segment_io import save_labels
    scan_dir = tmp_path
    (scan_dir / "sessions" / "s1").mkdir(parents=True)
    class_ids = np.array([6], dtype=np.int32)     # frozen 'unknown'
    instance_ids = np.array([0], dtype=np.int32)
    with pytest.raises(ValueError, match="eval-invariant 4"):
        save_labels(scan_dir, "s1", class_ids, instance_ids,
                   instances_doc={0: {"class_id": "unknown", "confirmed": False}})


def test_save_labels_merges_manifest_fields(tmp_path):
    from labeling.segment_io import save_labels
    scan_dir = tmp_path
    (scan_dir / "sessions" / "s1").mkdir(parents=True)
    class_ids = np.array([2], dtype=np.int32)
    instance_ids = np.array([0], dtype=np.int32)
    save_labels(scan_dir, "s1", class_ids, instance_ids,
               instances_doc={0: {"class_id": "pipe_new", "confirmed": False}})
    meta = json.loads((scan_dir / "sessions" / "s1" / "output" / "gt_segment_metadata.json").read_text())
    assert "class_histogram" in meta
    assert "thresholds" in meta
```

- [ ] **Step 2: Run, verify fail** — `TypeError: save_labels() got an unexpected keyword argument 'instances_doc'`.

- [ ] **Step 3: Implement.** Add `instances_doc: Optional[dict] = None`, `eval_regions: Optional[list[dict]] = None`, `prior_segment_metadata: Optional[dict] = None` params to `save_labels`. After the existing `validate_invariants(...)` call and before writing any files, build the invariant inputs and call each `eval_invariants.check_*` function (import `scan_schema.eval_invariants` and `backend.app.core.frozen_class_ids` — note the layering: `segment_io` importing from `backend.app.core` is new; check for a circular-import risk since `app/core.py` may import `labeling.*` — if so, pass `frozen_ids` in as a parameter from the caller instead of importing `core` directly, keeping `segment_io` dependency-direction-clean). Compute `region masks` from `eval_regions` (each region's `prism` geometry needs `positions` — reuse whatever helper `regions.py::region_stats` already uses to build a region mask, imported locally to avoid a module-level cycle). On any `ValueError` from a check, let it propagate (the existing `except ValueError` in `routes/segment.py::segment_save` already turns this into a 400 — Task 17 upgrades that specific path to 422 with structured detail). On success, call `manifest.build_manifest(...)` and merge its dict into `meta` before writing `gt_segment_metadata.json`.

- [ ] **Step 4: Run, verify pass.** Also run the full existing `test_segment_io.py` suite to confirm no regression: `.venv/bin/pytest backend/tests/test_segment_io.py -v`

- [ ] **Step 5: Commit**

```bash
git add backend/labeling/segment_io.py backend/tests/test_segment_io.py
git commit -m "feat: gate save_labels on the 9 eval invariants, merge manifest fields"
```

## Task 16: voxa — save route wiring + structured 422

**Files:**
- Modify: `backend/routes/segment.py::segment_save` (around line 482-500)
- Test: `backend/tests/test_segment_routes.py` (or wherever existing save-route tests live — check first: `grep -rl "segment/save" backend/tests/`)

- [ ] **Step 1: Write the failing test**

```python
def test_segment_save_rejects_frozen_class_with_422(client, loaded_annotated_session):
    # Arrange a session whose working class_ids include a frozen id (6).
    # (Use the existing test fixture pattern for an active annotated session
    # — see other test_segment_save_* tests in this file for the fixture name
    # and how they mutate seg.class_ids/instance_ids before calling save.)
    ...
    resp = client.put("/api/segment/save")
    assert resp.status_code == 422
    assert "eval-invariant 4" in resp.json()["detail"]
```

- [ ] **Step 2: Run, verify fail** (currently returns 400 from the generic `except ValueError`, or 200 if no such data-path exists yet).

- [ ] **Step 3: Implement.** In `segment_save()`, before calling `save_labels`, load `instances_gt.json` via `instances_doc.load_instances_for_invariants(seg.session_dir)`, `eval_regions.json` via `segment_io.load_eval_regions_for_invariants(scan_dir)`, and the prior metadata via `segment_io.load_prior_segment_metadata(scan_dir, session_id)`; pass all three into `save_labels`. Keep the existing `except ValueError as e: raise HTTPException(400, str(e))` for SCHEMA invariants 3-6 (unchanged behavior), but since all eval-invariant messages are prefixed `"eval-invariant N: ..."`, branch on that prefix to raise `HTTPException(422, str(e))` instead — preserving the existing 400 for the older, lower-level SCHEMA violations and reserving 422 for the new GT-semantic ones, matching the existing convention where 422 means "domain rule rejected" (`reject_frozen_class`, the eval-region gate) and 400 means "malformed request/data".

- [ ] **Step 4: Run, verify pass.** Run the full save-route test file to confirm no regressions.

- [ ] **Step 5: Commit**

```bash
git add backend/routes/segment.py backend/tests/test_segment_routes.py
git commit -m "feat: wire eval-invariant gate into the save route, 422 on violation"
```

## Task 17: voxa — migration script

**Files:**
- Create: `scripts/migrate_eval_invariants.py`
- Test: `backend/tests/test_migrate_eval_invariants.py`

- [ ] **Step 1: Write the failing tests**

```python
# backend/tests/test_migrate_eval_invariants.py
import numpy as np
import json
from scripts.migrate_eval_invariants import migrate_session   # sys.path trick mirrors promote_to_v3.py's test if any exists, else add scripts/ to path in conftest


def test_migrate_session_backfills_missing_arrays(tmp_path):
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    n = 5
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([2, 2, -1, -1, -1], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([0, 0, -1, -1, -1], dtype=np.int32))
    migrate_session(session_dir, n_points=n, dry_run=False)
    cats = np.load(session_dir / "output" / "gt_point_category.npy")
    comps = np.load(session_dir / "output" / "gt_point_component_ids.npy")
    assert (cats == 0).all()          # all backfilled to `none`
    assert comps.tolist() == [0, 0, -1, -1, -1]   # one component per instance


def test_migrate_session_converts_legacy_class_6(tmp_path):
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([6, 6, 2], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([3, 3, 9], dtype=np.int32))
    migrate_session(session_dir, n_points=3, dry_run=False)
    new_cls = np.load(session_dir / "output" / "gt_class_ids.npy")
    new_inst = np.load(session_dir / "output" / "gt_segment_ids.npy")
    cats = np.load(session_dir / "output" / "gt_point_category.npy")
    assert new_cls.tolist() == [-1, -1, 2]          # class-6 points erased
    assert new_inst.tolist() == [-1, -1, 9]         # instance stripped (review blob, class-less)
    assert cats.tolist() == [3, 3, 0]               # excluded_review
    meta = json.loads((session_dir / "output" / "gt_segment_metadata.json").read_text())
    assert any(b["instance_id"] == 3 for b in meta["review_blobs"])


def test_migrate_session_dry_run_writes_nothing(tmp_path):
    session_dir = tmp_path / "sessions" / "s1"
    (session_dir / "output").mkdir(parents=True)
    np.save(session_dir / "output" / "gt_class_ids.npy", np.array([2], dtype=np.int32))
    np.save(session_dir / "output" / "gt_segment_ids.npy", np.array([0], dtype=np.int32))
    migrate_session(session_dir, n_points=1, dry_run=True)
    assert not (session_dir / "output" / "gt_point_category.npy").exists()
```

- [ ] **Step 2: Run, verify fail** — `ModuleNotFoundError`.

- [ ] **Step 3: Implement**

```python
# scripts/migrate_eval_invariants.py
"""Migrate a pre-phase-2 session so it passes the phase-3 eval invariants.

Additive-only EXCEPT for legacy class-id-6 ('unknown') points, which are
rewritten to the phase-2 review-blob representation (category=excluded_review,
class=-1, instance stripped) because eval-invariant 4 rejects any frozen class
id — including 6 — with no grandfather path. Every other already-labeled
point (any class id other than 6, including the other frozen legacy ids
0/3/5/13, which stay readable-but-unassignable) is untouched byte-for-byte.

    .venv/bin/python scripts/migrate_eval_invariants.py <scan_dir> <session_id> [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import numpy as np  # noqa: E402

from labeling.categories import CATEGORY_NONE, CATEGORY_EXCLUDED_REVIEW  # noqa: E402
from labeling.components import component_ids  # noqa: E402
from scan_schema.layout import ScanLayout  # noqa: E402

LEGACY_UNKNOWN_CLASS_ID = 6


def migrate_session(session_dir: Path, n_points: int, *, dry_run: bool) -> dict:
    out = session_dir / "output"
    class_ids = np.load(out / "gt_class_ids.npy")
    instance_ids = np.load(out / "gt_segment_ids.npy")

    categories_path = out / "gt_point_category.npy"
    categories = (np.load(categories_path) if categories_path.exists()
                 else np.full(n_points, CATEGORY_NONE, dtype=np.int8))

    legacy = class_ids == LEGACY_UNKNOWN_CLASS_ID
    n_legacy = int(legacy.sum())
    converted_ids = sorted(set(int(i) for i in np.unique(instance_ids[legacy]).tolist())) if n_legacy else []
    if n_legacy:
        categories = categories.copy()
        categories[legacy] = CATEGORY_EXCLUDED_REVIEW
        class_ids = class_ids.copy(); instance_ids = instance_ids.copy()
        class_ids[legacy] = -1
        instance_ids[legacy] = -1   # review blobs are class-less by construction

    component_path = out / "gt_point_component_ids.npy"
    if component_path.exists():
        comp_ids = np.load(component_path)
    else:
        # No positions available offline (source.ply not loaded here) — one
        # component per instance, matching "no fragment splitting is invented"
        comp_ids = np.where(instance_ids >= 0, 0, -1).astype(np.int16)

    meta_path = out / "gt_segment_metadata.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    review_blobs = list(meta.get("review_blobs", []))
    if n_legacy:
        for iid in converted_ids:
            n_pts = int((instance_ids == iid).sum()) if False else int((legacy & (np.load(out / "gt_segment_ids.npy") == iid)).sum())
            review_blobs.append({"instance_id": iid, "n_points": n_pts})
    meta["review_blobs"] = review_blobs
    meta.setdefault("categories", {})

    result = {
        "n_legacy_converted": n_legacy,
        "converted_instance_ids": converted_ids,
    }
    if dry_run:
        return result

    np.save(out / "gt_class_ids.npy", class_ids)
    np.save(out / "gt_segment_ids.npy", instance_ids)
    np.save(categories_path, categories)
    np.save(component_path, comp_ids)
    meta_path.write_text(json.dumps(meta, indent=2))
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scan_dir", type=Path)
    ap.add_argument("session_id")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    meta = json.loads((a.scan_dir / "meta.json").read_text())
    n_points = int(meta["n_points"])
    session_dir = ScanLayout(a.scan_dir).session(a.session_id).dir
    result = migrate_session(session_dir, n_points, dry_run=a.dry_run)
    print(f"{'[dry-run] ' if a.dry_run else ''}{a.scan_dir.name}/{a.session_id}: "
          f"converted {result['n_legacy_converted']} legacy class-6 points "
          f"(instances {result['converted_instance_ids']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run, verify pass.** Fix the sloppy `n_pts` computation flagged inline above during implementation (it double-loads the file — clean up to compute per-instance counts from the already-loaded pre-mutation `instance_ids` array in one pass, before the `instance_ids[legacy] = -1` mutation).

- [ ] **Step 5: Commit**

```bash
git add scripts/migrate_eval_invariants.py backend/tests/test_migrate_eval_invariants.py
git commit -m "feat: migration script for pre-phase-2 sessions (eval invariants)"
```

## Task 18: voxa — dry-run + real migration against the 3 precious scans

- [ ] **Step 1:** For each of `munich`, `water_treatment_navvis`, `smart_ais_navvis`: copy the scan's session directory to a scratch location, run the migration script with `--dry-run` against the copy first, inspect the printed conversion counts.
- [ ] **Step 2:** Run the migration for real against each **copy**, then diff every `output/*.npy` file outside the legacy-class-6 point set against the original to confirm byte-for-byte preservation (write a small one-off diff script or use `np.array_equal` on the non-legacy mask — this is a manual verification step per the spec's risk mitigation, not a permanent test).
- [ ] **Step 3:** Only after the diff confirms correctness, run the migration script against the real scan directories (no `--dry-run`), one scan at a time, confirming `git status`/backup state of the lidar archive beforehand since this touches precious data outside the voxa git repo.
- [ ] **Step 4:** For each migrated scan, run `scan_schema validate <lidar_root>` (or the new manifest subcommand) and confirm zero eval-invariant errors.

## Task 19: voxa — bump the scan_schema pin

**Files:**
- Modify: `requirements.txt:19`
- Modify: `docs/scan-schema.md`

- [ ] **Step 1:** Replace `scan-schema @ git+https://github.com/rHedBull/scan_schema.git@main` with `scan-schema @ git+https://github.com/rHedBull/scan_schema.git@<merge-commit-sha-from-Task-11>`.
- [ ] **Step 2:** Reinstall: `.venv/bin/pip install -r requirements.txt` (or `backend/requirements.txt`, whichever holds this line) and re-run the full backend test suite to confirm nothing changed behavior between the editable-install dev version and the pinned commit.
- [ ] **Step 3:** Add a short note to `docs/scan-schema.md`: "voxa pins scan_schema to a commit SHA (not `@main`); bump it explicitly whenever scan_schema changes, as an ordinary dependency-bump PR."
- [ ] **Step 4: Run full suite**

Run: `npm run test:backend`
Expected: all pass

- [ ] **Step 5: Commit**

```bash
git add requirements.txt docs/scan-schema.md
git commit -m "chore: pin scan_schema to the eval-invariants merge commit"
```

## Task 20: voxa — full-suite verification + docs

- [ ] **Step 1:** Run `npm test` (frontend + backend) in voxa; confirm green.
- [ ] **Step 2:** Update `CLAUDE.md`'s "Architecture" section to note phase 3's status (mirroring how phase 0/1/2 are documented), and cross-reference the new spec file, per the repo's "docs ship with the code" rule.
- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: note eval-invariants phase 3 status in CLAUDE.md"
```

---

## Notes for the executing agent

- Tasks 1-11 happen in the `scan-schema` repo/worktree; Tasks 12-20 happen in the `voxa` repo/worktree. Use `superpowers:using-git-worktrees` to set up both before starting, and keep them as separate branches/PRs (`feat/eval-invariants-manifest` in each repo).
- Task 15 flags a possible circular-import risk (`segment_io` wanting `backend.app.core.frozen_class_ids`) — resolve by checking `backend/app/core.py`'s existing imports before deciding whether to import it directly or thread `frozen_ids` through as a parameter from the caller (`routes/segment.py` already imports both modules, so it can compute `frozen_class_ids()` once and pass it down).
- Task 17's inline TODO (cleaning up the `n_pts` double-load) must be fixed before Step 5's commit — it's called out deliberately in the plan as a first-draft rough edge, not a finished implementation.
- Task 18 is the highest-risk step in the whole plan — it mutates on-disk data for the three precious labeled scans. Do not skip the copy-first diff verification even under time pressure.
