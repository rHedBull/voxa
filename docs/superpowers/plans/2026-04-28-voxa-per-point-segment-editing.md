# Voxa Per-Point Segment Editing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a per-point segment-editing toolkit to Voxa's Label mode (Pick + 3D Brush), so an annotator can take a model-produced prelabel from `<lidar>/annotated/<scene>/prelabel/`, correct it via segment-level operations and a brush, and save SCHEMA-conformant ground truth to `labels/gt_*.npy`.

**Architecture:** Editing state lives on the FastAPI backend (single source of truth, matches existing single-cloud `_state` pattern). The backend owns the full-resolution arrays, a `scipy.spatial.cKDTree` for brush range queries, and the undo/redo stack. The frontend keeps a mirror of the per-point arrays for rendering, sends ops over JSON, and applies optimistic updates locally. Cuboid editing is unchanged; segment tools coexist as a second tool family within Label mode.

**Tech Stack:** Backend — Python 3.11, FastAPI, NumPy, `scipy.spatial.cKDTree`, pytest. Frontend — Vite + React 18 + Three.js, vitest. No new dependencies on either side.

**Spec:** `docs/superpowers/specs/2026-04-28-voxa-per-point-segment-editing-design.md`

---

## File Structure

### Backend (`backend/`)

```
backend/
├── lidar_io.py            (modify)  add prelabel/ fallback in load_annotated
├── segment_io.py          (create)  prelabel reader, save (gt_*.npy + metadata + history)
├── segment_state.py       (create)  in-memory editing state, apply/undo/redo, KD-tree
├── main.py                (modify)  /api/segment/* endpoints, LoadResponse extensions
└── tests/
    ├── test_lidar_io.py           (modify)  prelabel-fallback cases
    ├── test_segment_io.py         (create)  save/round-trip, invariants, history pruning
    ├── test_segment_state.py      (create)  apply/undo/redo, KD-tree query
    └── test_segment_endpoints.py  (create)  /api/segment/* via TestClient
```

Module responsibilities:

- **`segment_io.py`** — pure I/O. Read `prelabel/ransac_*`, write `labels/gt_*.npy` + recompute `gt_segment_metadata.json`, write/prune `annotation_history/<ts>/`. Validates SCHEMA invariants 3 & 4 before write. No state, no FastAPI.
- **`segment_state.py`** — editing-session state for one scene. Holds `class_full`, `instance_full`, `kdtree`, `undo`, `redo`, `dirty`. Pure functions for `apply_set_class / apply_merge / apply_reassign` + `undo / redo / brush_query`. No FastAPI.
- **`main.py`** — HTTP surface only. Constructs / disposes `SegmentSession` on `/api/load`, dispatches to it from the new endpoints, serializes b64.

### Frontend (`frontend/src/`)

```
frontend/src/
├── api.js                 (modify)  decode full_*, segment endpoints
├── api.test.js            (modify)  cover new b64 fields
├── App.jsx                (modify)  segState lift, dirty, save dispatcher
├── mode-label.jsx         (modify)  tool strip, Pick + Brush integration
├── segment-state.js       (create)  pure reducer for segState
├── segment-state.test.js  (create)  reducer unit tests
├── segment-tools.jsx      (create)  Pick + Brush React components, gizmo
└── viewer.jsx             (modify)  raycast hit-test exposed for cursor follow
```

Module responsibilities:

- **`segment-state.js`** — pure reducer over the local mirror of class/instance arrays. `applyOp({op, indices, payload}) → newState`, `mergeBackendDelta(delta) → newState`. No React, no fetch.
- **`segment-tools.jsx`** — React components for the tool strip and the brush gizmo. Owns the in-flight stroke buffer; emits committed strokes to `App.jsx`.
- **`mode-label.jsx`** — adds the `activeTool` selector and routes pointer events to either the cuboid editor (existing) or the new segment tools.

---

## Branch / Workspace

This plan assumes work on a fresh feature branch off `main`. The user typically lets me pick — `feat/per-point-segment-editing` is the default. All commits go on that branch; PR opened at the end.

---

## Task 1: Branch + plan-tracking setup

**Files:**
- (none — git only)

- [ ] **Step 1: Create feature branch**

```bash
git checkout main
git pull
git checkout -b feat/per-point-segment-editing
```

- [ ] **Step 2: Verify clean baseline tests**

```bash
npm run test:backend
npm run test:frontend
```

Expected: both green. If anything fails, stop and report — don't start work on a red baseline.

---

## Task 2: `segment_io.py` — prelabel reader

Pure read function. Returns `(class_ids: int8 (N,), instance_ids: int32 (N,))` derived from the `prelabel/` files, or `None` if absent / malformed.

**Files:**
- Create: `backend/segment_io.py`
- Create: `backend/tests/test_segment_io.py`

- [ ] **Step 1: Write failing test for happy path**

`backend/tests/test_segment_io.py`:

```python
"""Tests for prelabel ingestion + label save."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from segment_io import load_prelabel


def _write_prelabel(scan_dir: Path, instance_ids: np.ndarray, summary: list[dict]):
    pre = scan_dir / "prelabel"
    pre.mkdir(parents=True, exist_ok=True)
    np.save(pre / "ransac_instance_ids.npy", instance_ids.astype(np.int32))
    (pre / "ransac_segment_summary.json").write_text(
        json.dumps({"segments": summary})
    )


def test_load_prelabel_returns_aligned_arrays(tmp_path):
    scan_dir = tmp_path / "annotated" / "demo"
    inst = np.array([-1, 0, 0, 1, 1, 2, -1, 2], dtype=np.int32)
    summary = [
        {"id": 0, "class_id": 0},
        {"id": 1, "class_id": 1},
        {"id": 2, "class_id": 2},
    ]
    _write_prelabel(scan_dir, inst, summary)

    out = load_prelabel(scan_dir, n_points=8)
    assert out is not None
    cls, ii = out
    assert ii.dtype == np.int32 and ii.shape == (8,)
    assert cls.dtype == np.int8 and cls.shape == (8,)
    np.testing.assert_array_equal(ii, inst)
    np.testing.assert_array_equal(cls, np.array([-1, 0, 0, 1, 1, 2, -1, 2], dtype=np.int8))
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest backend/tests/test_segment_io.py::test_load_prelabel_returns_aligned_arrays -v
```

Expected: `ImportError` or `ModuleNotFoundError` for `segment_io`.

- [ ] **Step 3: Minimal implementation**

`backend/segment_io.py`:

```python
"""SCHEMA-aware reader for prelabel/, writer for labels/, history pruning.

Pure I/O. No FastAPI, no in-memory state. Loaders return aligned arrays;
writers validate invariants and recompute gt_segment_metadata.json from the
arrays before flushing to disk.
"""
from __future__ import annotations

import json
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np


def load_prelabel(
    scan_dir: Path, n_points: int
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Read prelabel/ if present. Returns (class_ids int8, instance_ids int32)
    or None when no prelabel exists / arrays are malformed."""
    pre = scan_dir / "prelabel"
    inst_path = pre / "ransac_instance_ids.npy"
    summary_path = pre / "ransac_segment_summary.json"
    if not inst_path.exists() or not summary_path.exists():
        return None
    try:
        instance_ids = np.load(inst_path).astype(np.int32)
        summary = json.loads(summary_path.read_text())
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if instance_ids.shape != (n_points,):
        return None
    seg_to_class = {int(s["id"]): int(s["class_id"]) for s in summary.get("segments", [])}
    class_ids = np.full(n_points, -1, dtype=np.int8)
    for sid, cid in seg_to_class.items():
        class_ids[instance_ids == sid] = cid
    return class_ids, instance_ids
```

- [ ] **Step 4: Run test to verify it passes**

```bash
.venv/bin/pytest backend/tests/test_segment_io.py -v
```

Expected: PASS.

- [ ] **Step 5: Add edge-case tests, ensure they pass**

Append to `test_segment_io.py`:

```python
def test_load_prelabel_returns_none_when_missing(tmp_path):
    scan_dir = tmp_path / "annotated" / "demo"
    scan_dir.mkdir(parents=True)
    assert load_prelabel(scan_dir, n_points=8) is None


def test_load_prelabel_returns_none_on_size_mismatch(tmp_path):
    scan_dir = tmp_path / "annotated" / "demo"
    _write_prelabel(scan_dir, np.zeros(7, dtype=np.int32), [])
    assert load_prelabel(scan_dir, n_points=8) is None
```

Run: `.venv/bin/pytest backend/tests/test_segment_io.py -v` → all PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/segment_io.py backend/tests/test_segment_io.py
git commit -m "feat(backend): segment_io.load_prelabel for SCHEMA prelabel ingestion"
```

---

## Task 3: `segment_io.py` — save with metadata recompute + history snapshot

Writes `labels/gt_class_ids.npy`, `labels/gt_segment_ids.npy`, recomputes `labels/gt_segment_metadata.json`, optionally writes a snapshot under `annotation_history/<ts>/`, prunes old timestamp dirs to a cap.

**Files:**
- Modify: `backend/segment_io.py`
- Modify: `backend/tests/test_segment_io.py`

- [ ] **Step 1: Write failing tests**

Append to `test_segment_io.py`:

```python
import os

from segment_io import save_labels, prune_history


def _read_npy(path: Path) -> np.ndarray:
    return np.load(path)


def test_save_labels_writes_aligned_arrays_and_metadata(tmp_path):
    scan_dir = tmp_path / "annotated" / "demo"
    cls = np.array([-1, 0, 0, 1, 1], dtype=np.int8)
    inst = np.array([-1, 0, 0, 1, 1], dtype=np.int32)
    save_labels(scan_dir, cls, inst, write_history=False)

    np.testing.assert_array_equal(
        _read_npy(scan_dir / "labels" / "gt_class_ids.npy"), cls.astype(np.int32),
    )
    np.testing.assert_array_equal(
        _read_npy(scan_dir / "labels" / "gt_segment_ids.npy"), inst.astype(np.int32),
    )
    meta = json.loads((scan_dir / "labels" / "gt_segment_metadata.json").read_text())
    assert meta["n_points"] == 5
    assert meta["n_labeled_points"] == 4
    assert meta["n_gt_segments"] == 2
    seg_ids = sorted(s["gt_id"] for s in meta["segments"])
    assert seg_ids == [0, 1]


def test_save_labels_rejects_invariant_violation(tmp_path):
    scan_dir = tmp_path / "annotated" / "demo"
    cls = np.array([0, 0], dtype=np.int8)         # class set
    inst = np.array([-1, 0], dtype=np.int32)      # but instance unlabeled at idx 0
    with pytest.raises(ValueError, match="invariant"):
        save_labels(scan_dir, cls, inst, write_history=False)


def test_save_labels_rejects_class_inconsistency(tmp_path):
    scan_dir = tmp_path / "annotated" / "demo"
    # Two points share inst id 0 but disagree on class id
    cls = np.array([0, 1], dtype=np.int8)
    inst = np.array([0, 0], dtype=np.int32)
    with pytest.raises(ValueError, match="invariant"):
        save_labels(scan_dir, cls, inst, write_history=False)


def test_save_labels_writes_history_snapshot(tmp_path):
    scan_dir = tmp_path / "annotated" / "demo"
    save_labels(scan_dir, np.array([0], dtype=np.int8), np.array([0], dtype=np.int32),
                write_history=True)
    hist = scan_dir / "annotation_history"
    assert hist.exists()
    snaps = list(hist.iterdir())
    assert len(snaps) == 1
    assert re.match(r"^\d{8}_\d{4}$", snaps[0].name)


def test_prune_history_keeps_only_timestamped_dirs(tmp_path):
    hist = tmp_path / "annotation_history"
    hist.mkdir()
    # 12 valid timestamps + 1 user-curated dir
    valid = [hist / f"2026010{i % 10}_{1000 + i:04d}" for i in range(12)]
    for v in valid:
        v.mkdir()
        os.utime(v, (1700000000 + i, 1700000000 + i))  # noqa: F821 — we override below
    # Re-set distinct mtimes so ordering is well-defined
    for i, v in enumerate(valid):
        os.utime(v, (1_700_000_000 + i, 1_700_000_000 + i))
    user = hist / "manual-backup"; user.mkdir()

    prune_history(hist, keep=10)

    remaining = sorted(p.name for p in hist.iterdir())
    assert "manual-backup" in remaining
    assert sum(1 for n in remaining if re.match(r"^\d{8}_\d{4}$", n)) == 10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
.venv/bin/pytest backend/tests/test_segment_io.py -v
```

Expected: 5 new tests fail with `ImportError: cannot import name 'save_labels'`.

- [ ] **Step 3: Implement save_labels + prune_history**

Append to `backend/segment_io.py`:

```python
_TS_RE = re.compile(r"^\d{8}_\d{4}$")


def _validate_invariants(class_ids: np.ndarray, instance_ids: np.ndarray) -> None:
    """SCHEMA invariants 3 & 4. Raises ValueError on violation."""
    cls_unl = class_ids == -1
    inst_unl = instance_ids == -1
    if not np.array_equal(cls_unl, inst_unl):
        n_bad = int(np.sum(cls_unl != inst_unl))
        raise ValueError(
            f"invariant 3: class_ids[i]==-1 ⟺ instance_ids[i]==-1 violated at {n_bad} points",
        )
    labeled = ~cls_unl
    if labeled.any():
        # For every instance id, all its points must agree on class id.
        ii = instance_ids[labeled]
        ci = class_ids[labeled]
        # group: take first class id seen for each instance, compare to all
        order = np.argsort(ii, kind="stable")
        ii_s, ci_s = ii[order], ci[order]
        # instance boundaries
        boundaries = np.concatenate(([True], ii_s[1:] != ii_s[:-1]))
        first_cls = ci_s[boundaries]
        # broadcast first_cls back via cumulative count
        group_idx = np.cumsum(boundaries) - 1
        if not np.array_equal(ci_s, first_cls[group_idx]):
            raise ValueError(
                "invariant 4: per-segment class consistency violated",
            )


def _build_segment_metadata(
    class_ids: np.ndarray, instance_ids: np.ndarray,
    positions: Optional[np.ndarray] = None,
) -> dict:
    n_points = int(instance_ids.shape[0])
    labeled = instance_ids >= 0
    n_labeled = int(labeled.sum())
    segments: list[dict] = []
    for sid in np.unique(instance_ids[labeled]):
        sid_i = int(sid)
        m = instance_ids == sid_i
        cid = int(class_ids[m][0])
        entry: dict = {
            "gt_id": sid_i,
            "class_id": cid,
            "n_points": int(m.sum()),
        }
        if positions is not None:
            sub = positions[m]
            mn = sub.min(axis=0); mx = sub.max(axis=0)
            entry["bbox"] = [float(mn[0]), float(mn[1]), float(mn[2]),
                              float(mx[0]), float(mx[1]), float(mx[2])]
        segments.append(entry)
    return {
        "n_points": n_points,
        "n_gt_segments": len(segments),
        "n_labeled_points": n_labeled,
        "class_map_version": 1,
        "segments": segments,
    }


def _utc_timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M")


def save_labels(
    scan_dir: Path,
    class_ids: np.ndarray,
    instance_ids: np.ndarray,
    *,
    positions: Optional[np.ndarray] = None,
    write_history: bool = True,
    history_keep: int = 10,
) -> None:
    """Atomically write gt_*.npy + metadata. Writes a history snapshot first."""
    _validate_invariants(class_ids, instance_ids)

    labels_dir = scan_dir / "labels"
    labels_dir.mkdir(parents=True, exist_ok=True)

    if write_history and (labels_dir / "gt_class_ids.npy").exists():
        snap_dir = scan_dir / "annotation_history" / _utc_timestamp()
        snap_dir.mkdir(parents=True, exist_ok=True)
        for fname in ("gt_class_ids.npy", "gt_segment_ids.npy",
                      "gt_segment_metadata.json"):
            src = labels_dir / fname
            if src.exists():
                shutil.copy2(src, snap_dir / fname)
        prune_history(scan_dir / "annotation_history", keep=history_keep)
    elif write_history:
        # No prior labels — still write an empty history dir so on-disk state is predictable
        (scan_dir / "annotation_history" / _utc_timestamp()).mkdir(parents=True, exist_ok=True)
        prune_history(scan_dir / "annotation_history", keep=history_keep)

    # Per SCHEMA: int32 on disk, even though we keep int8 in memory for class.
    np.save(labels_dir / "gt_class_ids.npy", class_ids.astype(np.int32))
    np.save(labels_dir / "gt_segment_ids.npy", instance_ids.astype(np.int32))
    meta = _build_segment_metadata(class_ids, instance_ids, positions)
    (labels_dir / "gt_segment_metadata.json").write_text(json.dumps(meta, indent=2))


def prune_history(history_dir: Path, *, keep: int = 10) -> None:
    """Keep the `keep` most-recent timestamp-named subdirs; leave others alone."""
    if not history_dir.exists():
        return
    timestamped = [p for p in history_dir.iterdir() if p.is_dir() and _TS_RE.match(p.name)]
    if len(timestamped) <= keep:
        return
    timestamped.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    for p in timestamped[keep:]:
        shutil.rmtree(p)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest backend/tests/test_segment_io.py -v
```

Expected: all 7 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/segment_io.py backend/tests/test_segment_io.py
git commit -m "feat(backend): segment_io.save_labels + history pruning + invariants"
```

---

## Task 4: Extend `load_annotated` to fall through to prelabel

When `labels/gt_*.npy` is absent but `prelabel/` is populated, surface those arrays as the editable starting state with `is_from_prelabel=True`. When neither exists, return all-`-1` arrays so editing can start from scratch.

**Files:**
- Modify: `backend/lidar_io.py`
- Modify: `backend/tests/test_lidar_io.py`

- [ ] **Step 1: Add `is_from_prelabel` to `AnnotatedScene`**

In `backend/lidar_io.py`, change the dataclass:

```python
@dataclass
class AnnotatedScene:
    pc: PointCloud
    intensity: Optional[np.ndarray]
    labels: Optional[LabelArrays]
    meta: dict
    palette: list[ClassPaletteEntry]
    n_classes: int
    n_instances: int
    is_from_prelabel: bool = False   # ← new
```

- [ ] **Step 2: Write failing test for prelabel-fallback**

Append to `backend/tests/test_lidar_io.py`:

```python
def test_load_annotated_falls_through_to_prelabel(tmp_path):
    """When labels/gt_*.npy is absent, prelabel/ becomes the editable seed."""
    root = tmp_path / "lidar"
    scan_dir = root / "annotated" / "demo"
    _write_ply(scan_dir / "source" / "scan.ply", n=8)
    pre = scan_dir / "prelabel"; pre.mkdir(parents=True)
    np.save(pre / "ransac_instance_ids.npy",
            np.array([-1, 0, 0, 1, 1, 2, -1, 2], dtype=np.int32))
    (pre / "ransac_segment_summary.json").write_text(json.dumps({
        "segments": [{"id": 0, "class_id": 0},
                     {"id": 1, "class_id": 1},
                     {"id": 2, "class_id": 2}],
    }))
    (root / "classes.json").write_text(json.dumps({
        "version": 1, "unlabeled_id": -1,
        "classes": [{"id": 0, "name": "pipe"}, {"id": 1, "name": "tank"},
                    {"id": 2, "name": "equipment"}],
    }))

    src = next(s for s in discover(tmp_path / "voxa-data", root) if s.tier == "annotated")
    out = load_annotated(src, root)

    assert out.is_from_prelabel is True
    assert out.labels is not None
    assert out.labels.instance_ids.shape == (8,)
    assert int(out.labels.instance_ids[1]) == 0
    assert int(out.labels.class_ids[1]) == 0


def test_load_annotated_empty_when_no_labels_no_prelabel(tmp_path):
    root = tmp_path / "lidar"
    scan_dir = root / "annotated" / "demo"
    _write_ply(scan_dir / "source" / "scan.ply", n=8)
    src = next(s for s in discover(tmp_path / "voxa-data", root) if s.tier == "annotated")
    out = load_annotated(src, root)
    # No labels, no prelabel → labels arrays present but all -1, is_from_prelabel False.
    assert out.is_from_prelabel is False
    assert out.labels is not None
    assert int(out.labels.class_ids.min()) == -1 and int(out.labels.class_ids.max()) == -1
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
.venv/bin/pytest backend/tests/test_lidar_io.py -v
```

Expected: 2 new tests fail.

- [ ] **Step 4: Wire prelabel fallback into `load_annotated`**

In `backend/lidar_io.py`, replace the `labels` construction block (the `if labels is None: ...` body and the empty fallback) with:

```python
    from segment_io import load_prelabel  # local import: avoids backend/lidar_io→segment_io→backend cycle

    is_from_prelabel = False

    if labels is None:
        pre = load_prelabel(Path(src.source_path).parent.parent, n_points=len(pc))
        if pre is not None:
            ci8, ii = pre
            labels = LabelArrays(class_ids=ci8, instance_ids=ii)
            valid_classes = ci8[ci8 >= 0]
            valid_inst = ii[ii >= 0]
            n_classes = int(valid_classes.max()) + 1 if valid_classes.size else 0
            n_instances = int(valid_inst.max()) + 1 if valid_inst.size else 0
            is_from_prelabel = True

    if labels is None:
        # All-unlabeled, still editable.
        labels = LabelArrays(
            class_ids=np.full(len(pc), -1, dtype=np.int8),
            instance_ids=np.full(len(pc), -1, dtype=np.int32),
        )
```

Then thread `is_from_prelabel` into the returned `AnnotatedScene(...)`.

- [ ] **Step 5: Run tests to verify they pass**

```bash
.venv/bin/pytest backend/tests/test_lidar_io.py -v
```

Expected: all PASS, including the existing tests (regressions caught early).

- [ ] **Step 6: Commit**

```bash
git add backend/lidar_io.py backend/tests/test_lidar_io.py
git commit -m "feat(backend): load_annotated falls through to prelabel/, exposes is_from_prelabel"
```

---

## Task 5: `segment_state.py` — apply / undo / redo (no KD-tree yet)

Pure session state. Holds class/instance arrays + bounded undo stack. Three apply ops: `set_class`, `merge`, `reassign` (covers brush absorb / create / erase).

**Files:**
- Create: `backend/segment_state.py`
- Create: `backend/tests/test_segment_state.py`

- [ ] **Step 1: Write failing test for set_class**

`backend/tests/test_segment_state.py`:

```python
"""Editing-session state: apply / undo / redo, plus brush_query."""
from __future__ import annotations

import numpy as np
import pytest

from segment_state import SegmentSession


def _seed():
    cls = np.array([-1, 0, 0, 1, 1, 2, -1, 2], dtype=np.int8)
    inst = np.array([-1, 0, 0, 1, 1, 2, -1, 2], dtype=np.int32)
    return SegmentSession(class_ids=cls.copy(), instance_ids=inst.copy(),
                          positions=np.zeros((8, 3), dtype=np.float32))


def test_set_class_changes_specified_indices():
    s = _seed()
    delta = s.apply_set_class(indices=np.array([1, 2], dtype=np.int32),
                              class_id=2)
    assert int(s.class_ids[1]) == 2 and int(s.class_ids[2]) == 2
    # instance ids untouched.
    assert int(s.instance_ids[1]) == 0
    assert delta["op"] == "set_class"
    assert s.dirty is True


def test_undo_restores_pre_state_and_redo_reapplies():
    s = _seed()
    s.apply_set_class(np.array([1, 2], dtype=np.int32), class_id=2)
    s.undo()
    assert int(s.class_ids[1]) == 0 and int(s.class_ids[2]) == 0
    s.redo()
    assert int(s.class_ids[1]) == 2 and int(s.class_ids[2]) == 2


def test_undo_stack_is_bounded():
    s = _seed()
    s.history_cap = 3
    for cid in range(10):
        s.apply_set_class(np.array([1], dtype=np.int32), class_id=cid % 3)
    # Only 3 undos should succeed.
    for _ in range(3):
        assert s.undo() is not None
    assert s.undo() is None
```

- [ ] **Step 2: Run test to verify it fails**

```bash
.venv/bin/pytest backend/tests/test_segment_state.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement `SegmentSession` (no KD-tree yet)**

`backend/segment_state.py`:

```python
"""Per-scene editing-session state.

One SegmentSession instance is held in main._state per loaded cloud.
All mutations are recorded as inverse-deltas on a bounded undo stack so
undo/redo can replay without re-deriving anything.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class _Delta:
    op: str
    indices: np.ndarray            # int32
    before_cls: np.ndarray         # int8
    before_inst: np.ndarray        # int32
    after_cls: np.ndarray          # int8
    after_inst: np.ndarray         # int32


class SegmentSession:
    """In-memory editor state for one scene.

    Class arrays are int8 (voxa class count is small); instance arrays are
    int32. Both are full-resolution, length == positions.shape[0].
    """

    def __init__(
        self,
        class_ids: np.ndarray,
        instance_ids: np.ndarray,
        positions: np.ndarray,
    ) -> None:
        if class_ids.dtype != np.int8:
            class_ids = class_ids.astype(np.int8)
        if instance_ids.dtype != np.int32:
            instance_ids = instance_ids.astype(np.int32)
        n = positions.shape[0]
        if class_ids.shape != (n,) or instance_ids.shape != (n,):
            raise ValueError("class/instance/position lengths disagree")
        self.class_ids = class_ids
        self.instance_ids = instance_ids
        self.positions = positions.astype(np.float32, copy=False)
        self._undo: deque[_Delta] = deque()
        self._redo: deque[_Delta] = deque()
        self.history_cap: int = 100
        self.dirty: bool = False

    # ── Public ops ──

    def apply_set_class(self, indices: np.ndarray, class_id: int) -> dict:
        return self._apply("set_class", indices, dict(class_id=int(class_id)))

    def apply_merge(self, source_inst: int, target_inst: int) -> dict:
        # Merge: every point with instance==source becomes target. Class is
        # taken from the target's existing class. Source must exist.
        if source_inst == target_inst:
            return {"op": "merge", "n_affected": 0, "new_instance_id": target_inst}
        idx = np.flatnonzero(self.instance_ids == source_inst).astype(np.int32)
        if idx.size == 0:
            return {"op": "merge", "n_affected": 0, "new_instance_id": target_inst}
        # Resolve target class id (might be -1 if target has no points; rare).
        tgt_mask = self.instance_ids == target_inst
        if tgt_mask.any():
            target_class = int(self.class_ids[tgt_mask][0])
        else:
            target_class = int(self.class_ids[idx][0])
        return self._apply("merge", idx, dict(target_inst=int(target_inst),
                                              target_class=target_class))

    def apply_reassign(
        self, indices: np.ndarray,
        target_inst: Optional[int],
        target_class: Optional[int],
    ) -> dict:
        """Brush op. target_inst=None + target_class=None → erase to (-1, -1).
        target_inst<0 + target_class>=0 → allocate a new instance id."""
        return self._apply("reassign", indices, dict(
            target_inst=None if target_inst is None else int(target_inst),
            target_class=None if target_class is None else int(target_class),
        ))

    def undo(self) -> Optional[dict]:
        if not self._undo:
            return None
        d = self._undo.pop()
        # Apply 'before' to current
        self.class_ids[d.indices] = d.before_cls
        self.instance_ids[d.indices] = d.before_inst
        self._redo.append(d)
        self.dirty = True
        return self._delta_payload(d, direction="undo")

    def redo(self) -> Optional[dict]:
        if not self._redo:
            return None
        d = self._redo.pop()
        self.class_ids[d.indices] = d.after_cls
        self.instance_ids[d.indices] = d.after_inst
        self._undo.append(d)
        self.dirty = True
        return self._delta_payload(d, direction="redo")

    # ── Internal ──

    def _apply(self, op: str, indices: np.ndarray, payload: dict) -> dict:
        indices = indices.astype(np.int32, copy=False)
        before_cls = self.class_ids[indices].copy()
        before_inst = self.instance_ids[indices].copy()
        new_inst_id: Optional[int] = None

        if op == "set_class":
            self.class_ids[indices] = np.int8(payload["class_id"])
            # instance_ids unchanged
        elif op == "merge":
            self.instance_ids[indices] = np.int32(payload["target_inst"])
            self.class_ids[indices] = np.int8(payload["target_class"])
        elif op == "reassign":
            ti, tc = payload["target_inst"], payload["target_class"]
            if ti is None and tc is None:
                # erase
                self.instance_ids[indices] = np.int32(-1)
                self.class_ids[indices] = np.int8(-1)
            else:
                if ti is None or ti < 0:
                    # Allocate a fresh instance id.
                    new_inst_id = int(self.instance_ids.max(initial=-1)) + 1
                    self.instance_ids[indices] = np.int32(new_inst_id)
                else:
                    self.instance_ids[indices] = np.int32(ti)
                if tc is not None:
                    self.class_ids[indices] = np.int8(tc)
        else:
            raise ValueError(f"unknown op: {op}")

        delta = _Delta(
            op=op, indices=indices,
            before_cls=before_cls, before_inst=before_inst,
            after_cls=self.class_ids[indices].copy(),
            after_inst=self.instance_ids[indices].copy(),
        )
        self._undo.append(delta)
        if len(self._undo) > self.history_cap:
            self._undo.popleft()
        self._redo.clear()
        self.dirty = True

        out = {
            "op": op,
            "n_affected": int(indices.size),
            "indices": indices,
            "after_class": delta.after_cls,
            "after_instance": delta.after_inst,
        }
        if new_inst_id is not None:
            out["new_instance_id"] = new_inst_id
        return out

    def _delta_payload(self, d: _Delta, direction: str) -> dict:
        # Frontend reapplies these by index → (class, instance).
        cls = d.before_cls if direction == "undo" else d.after_cls
        inst = d.before_inst if direction == "undo" else d.after_inst
        return {
            "op": d.op, "direction": direction,
            "indices": d.indices,
            "after_class": cls,
            "after_instance": inst,
            "n_affected": int(d.indices.size),
        }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest backend/tests/test_segment_state.py -v
```

Expected: 3 PASS.

- [ ] **Step 5: Add tests for merge + reassign**

Append:

```python
def test_merge_moves_source_points_to_target_instance_and_class():
    s = _seed()
    s.apply_merge(source_inst=2, target_inst=0)
    assert (s.instance_ids[s.instance_ids != -1] == 0).any()
    # All previously-2 points are now 0.
    assert int((s.instance_ids == 2).sum()) == 0
    # And took target's class (which was 0).
    assert int(s.class_ids[5]) == 0


def test_reassign_with_negative_target_inst_allocates_new_id():
    s = _seed()
    out = s.apply_reassign(np.array([0, 6], dtype=np.int32),
                           target_inst=-1, target_class=2)
    new_id = out["new_instance_id"]
    assert int(s.instance_ids[0]) == new_id
    assert int(s.class_ids[6]) == 2


def test_reassign_with_both_none_erases_to_unlabeled():
    s = _seed()
    s.apply_reassign(np.array([1, 2], dtype=np.int32),
                     target_inst=None, target_class=None)
    assert int(s.instance_ids[1]) == -1
    assert int(s.class_ids[1]) == -1
```

Run: `.venv/bin/pytest backend/tests/test_segment_state.py -v` → all PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/segment_state.py backend/tests/test_segment_state.py
git commit -m "feat(backend): SegmentSession with apply/undo/redo + merge + reassign"
```

---

## Task 6: KD-tree-backed brush_query on `SegmentSession`

Add a `cKDTree` lazily on first query; expose `brush_query(center, radius, depth_cull=None, camera_ray=None) → np.ndarray[int32]`.

**Files:**
- Modify: `backend/segment_state.py`
- Modify: `backend/tests/test_segment_state.py`

- [ ] **Step 1: Write failing tests**

Append to `test_segment_state.py`:

```python
def test_brush_query_returns_indices_within_radius():
    rng = np.random.default_rng(0)
    pts = rng.standard_normal((1000, 3)).astype(np.float32)
    s = SegmentSession(
        class_ids=np.full(1000, -1, dtype=np.int8),
        instance_ids=np.full(1000, -1, dtype=np.int32),
        positions=pts,
    )
    center = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    R = 0.5
    got = set(s.brush_query(center, R).tolist())
    expected = set(np.flatnonzero(np.linalg.norm(pts - center, axis=1) <= R).tolist())
    assert got == expected


def test_brush_query_depth_cull_excludes_far_points_along_ray():
    # Place a near cluster at z=0 and a far cluster at z=10, both within sphere R.
    near = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float32)
    far = np.array([[0.0, 0.0, 10.0], [0.01, 0.0, 10.0]], dtype=np.float32)
    pts = np.concatenate([near, far], axis=0)
    s = SegmentSession(
        class_ids=np.full(4, -1, dtype=np.int8),
        instance_ids=np.full(4, -1, dtype=np.int32),
        positions=pts,
    )
    # Sphere big enough to cover both clusters; depth-cull along +z eliminates far.
    got = s.brush_query(
        np.array([0.0, 0.0, 0.0], dtype=np.float32),
        radius=20.0,
        camera_ray=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        depth_cull=2.0,
    )
    assert set(got.tolist()) == {0, 1}
```

- [ ] **Step 2: Run tests, verify failure**

```bash
.venv/bin/pytest backend/tests/test_segment_state.py -v -k brush_query
```

Expected: AttributeError on `brush_query`.

- [ ] **Step 3: Implement brush_query**

Add to `SegmentSession`:

```python
    def _ensure_tree(self):
        if not hasattr(self, "_tree"):
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.positions)
        return self._tree

    def brush_query(
        self,
        center: np.ndarray,
        radius: float,
        *,
        camera_ray: Optional[np.ndarray] = None,
        depth_cull: Optional[float] = None,
    ) -> np.ndarray:
        tree = self._ensure_tree()
        idx = np.array(tree.query_ball_point(center, r=float(radius)), dtype=np.int32)
        if idx.size == 0:
            return idx
        if camera_ray is not None and depth_cull is not None:
            # Distance along the camera ray, signed; cull points more than
            # `depth_cull` farther along the ray than the cursor center.
            ray = camera_ray.astype(np.float32)
            ray = ray / (np.linalg.norm(ray) + 1e-9)
            disp = self.positions[idx] - center.astype(np.float32)
            along = disp @ ray
            keep = along <= float(depth_cull)
            idx = idx[keep]
        return idx
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
.venv/bin/pytest backend/tests/test_segment_state.py -v
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/segment_state.py backend/tests/test_segment_state.py
git commit -m "feat(backend): SegmentSession.brush_query with KD-tree + depth cull"
```

---

## Task 7: Wire `SegmentSession` into `_state` on `/api/load`

Construct a session whenever an annotated scene loads (or any scene with `LabelArrays`). Cache on `_state["seg"]`. Clear on next load.

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Add `seg` slot to `_state`**

Find the `_state` definition and add `"seg": None,` so it reads:

```python
_state: dict[str, Any] = {
    "scene": None,
    "pc": None,
    "mesh": None,
    "subsample_idx": None,
    "intensity": None,
    "labels": None,
    "recenter_offset": [0.0, 0.0, 0.0],
    "seg": None,    # SegmentSession | None
}
```

- [ ] **Step 2: Construct session at the end of `load_scene`**

In `load_scene`, after `_state.update(...)`, add:

```python
    from segment_state import SegmentSession  # local import — avoids startup cost
    is_from_prelabel = bool(getattr(annotated_scene, "is_from_prelabel", False)) if 'annotated_scene' in locals() else False
    _state["seg"] = (
        SegmentSession(
            class_ids=labels.class_ids,
            instance_ids=labels.instance_ids,
            positions=pc.points,
            is_from_prelabel=is_from_prelabel,
        )
        if labels is not None else None
    )
```

The `_load_scene_source` helper currently returns `labels` but not the `AnnotatedScene` it came from. Change `_load_scene_source` to return `is_from_prelabel` as an extra return value (or restructure to return the full `AnnotatedScene` for annotated tier). Pick whichever fits the existing code shape; both are small refactors.

Update `SegmentSession.__init__` to accept and store `self.is_from_prelabel`:

```python
def __init__(self, class_ids, instance_ids, positions, *, is_from_prelabel: bool = False):
    ...
    self.is_from_prelabel = bool(is_from_prelabel)
```

(If `labels` is `None` because no annotated/no prelabel/no fallback was applied — i.e. legacy non-annotated scenes — `seg` stays `None` and segment endpoints will 409.)

- [ ] **Step 3: Build a `client_with_annotated_scene` fixture in `conftest.py`**

This fixture is reused by Tasks 8–12. Move `_write_ply` and `_build_annotated_root` from `test_lidar_io.py` into `conftest.py` (or import them) and add:

```python
@pytest.fixture
def client_with_annotated_scene(monkeypatch, tmp_path):
    from tests.test_lidar_io import _build_annotated_root  # or move helper to conftest
    root = _build_annotated_root(tmp_path)
    monkeypatch.setenv("VOXA_LIDAR_ROOT", str(root))
    # Re-import scene_registry so it re-reads the env, then construct client.
    import importlib, scene_registry, main
    importlib.reload(scene_registry); importlib.reload(main)
    from fastapi.testclient import TestClient
    client = TestClient(main.app)
    return client, "annotated/demo"


@pytest.fixture
def client_with_loaded_annotated_scene(client_with_annotated_scene):
    client, scene_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "want_full_labels": True})
    assert r.status_code == 200
    return client
```

- [ ] **Step 4: Add the integration test**

Append to `backend/tests/test_load_endpoint.py`:

```python
def test_load_annotated_creates_segment_session(client_with_annotated_scene):
    client, scene_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "want_full_labels": True})
    assert r.status_code == 200
    import main
    assert main._state["seg"] is not None
    assert main._state["seg"].class_ids.shape == main._state["seg"].instance_ids.shape
```

- [ ] **Step 5: Run smoke test to verify nothing regresses**

```bash
npm run test:backend
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add backend/main.py backend/tests/conftest.py backend/tests/test_load_endpoint.py
git commit -m "feat(backend): construct SegmentSession on /api/load when labels exist"
```

---

## Task 8: Extend `LoadResponse` with full-res arrays + `is_from_prelabel`

Wire the opt-in `want_full_labels: bool` field, plumb the flag from request → response, b64-encode full arrays.

**Files:**
- Modify: `backend/main.py`

- [ ] **Step 1: Extend `LoadRequest` and `LoadResponse`**

```python
class LoadRequest(BaseModel):
    name: str
    max_points: int = MAX_POINTS_DEFAULT
    want_full_labels: bool = False    # ← new


class LoadResponse(BaseModel):
    # ... existing fields ...
    full_class_ids: Optional[str] = None     # b64 Int8, full-res
    full_instance_ids: Optional[str] = None  # b64 Int32
    full_positions: Optional[str] = None     # b64 Float32 (xyz, recentered)
    full_n: Optional[int] = None
    is_from_prelabel: bool = False
    segment_summary: Optional[dict] = None   # { "<inst>": {class_id, n_points} }
```

- [ ] **Step 2: Plumb in `load_scene`**

Right before constructing `LoadResponse(...)`, add:

```python
    full_payload: dict[str, Any] = {}
    if req.want_full_labels and labels is not None:
        full_payload["full_class_ids"] = _b64(labels.class_ids.astype(np.int8))
        full_payload["full_instance_ids"] = _b64(labels.instance_ids.astype(np.int32))
        full_payload["full_positions"] = _b64(pc.points.astype(np.float32))
        full_payload["full_n"] = int(len(pc))
        # Lightweight summary (per-instance counts/class) for the UI.
        ii = labels.instance_ids
        ci = labels.class_ids
        m = ii >= 0
        if m.any():
            uids, idx0, counts = np.unique(ii[m], return_index=True, return_counts=True)
            summary = {
                str(int(uid)): {"class_id": int(ci[m][idx0[k]]), "n_points": int(counts[k])}
                for k, uid in enumerate(uids)
            }
        else:
            summary = {}
        full_payload["segment_summary"] = summary
    seg_for_meta = _state.get("seg")
    full_payload["is_from_prelabel"] = bool(seg_for_meta.is_from_prelabel) if seg_for_meta is not None else False
```

`SegmentSession.is_from_prelabel` was added in Task 7 — `_state["seg"]` is the single source of truth here.

- [ ] **Step 3: Pass `**full_payload` into `LoadResponse(...)`**

```python
    return LoadResponse(
        scene=src.scene_id,
        ...,
        **full_payload,
    )
```

- [ ] **Step 4: Add a regression test**

In `backend/tests/test_load_endpoint.py`, add:

```python
def test_load_with_want_full_labels_returns_full_arrays(client_with_annotated_scene):
    client, scene_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "want_full_labels": True})
    assert r.status_code == 200
    j = r.json()
    assert j["full_class_ids"] is not None
    assert j["full_instance_ids"] is not None
    assert j["full_positions"] is not None
    assert j["full_n"] is not None
    assert isinstance(j["segment_summary"], dict)


def test_load_without_flag_omits_full_arrays(client_with_annotated_scene):
    client, scene_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id})
    j = r.json()
    assert j.get("full_class_ids") is None
    assert j.get("full_positions") is None
```

The `client_with_annotated_scene` fixture was added in Task 7 — reuse it.

- [ ] **Step 5: Run tests**

```bash
npm run test:backend
```

Expected: green.

- [ ] **Step 6: Commit**

```bash
git add backend/main.py backend/tests/conftest.py backend/tests/test_load_endpoint.py
git commit -m "feat(backend): /api/load want_full_labels opt-in + full-res arrays"
```

---

## Task 9: `/api/segment/brush-query` endpoint

**Files:**
- Modify: `backend/main.py`
- Create: `backend/tests/test_segment_endpoints.py`

- [ ] **Step 1: Write failing test**

`backend/tests/test_segment_endpoints.py`:

```python
"""Tests for /api/segment/* endpoints."""
from __future__ import annotations

import base64
import numpy as np
import pytest


def _b64_to_int32(b64: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(b64), dtype=np.int32)


def test_brush_query_returns_indices(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    body = {"center": [0.0, 0.0, 0.0], "radius": 100.0}
    r = client.post("/api/segment/brush-query", json=body)
    assert r.status_code == 200
    j = r.json()
    assert "indices" in j and "n" in j
    arr = _b64_to_int32(j["indices"])
    assert arr.size == j["n"]


def test_brush_query_409_when_no_session(client_with_legacy_scene_loaded):
    client = client_with_legacy_scene_loaded
    body = {"center": [0.0, 0.0, 0.0], "radius": 1.0}
    r = client.post("/api/segment/brush-query", json=body)
    assert r.status_code == 409
```

(`client_with_loaded_annotated_scene` and `client_with_legacy_scene_loaded` are conftest fixtures; create them by extending what Task 8 introduced.)

- [ ] **Step 2: Run test, verify failure**

```bash
.venv/bin/pytest backend/tests/test_segment_endpoints.py -v
```

Expected: 404 (endpoint not found).

- [ ] **Step 3: Implement endpoint**

In `backend/main.py`, add near the other endpoints:

```python
class BrushQueryRequest(BaseModel):
    center: list[float]
    radius: float
    camera_ray: Optional[list[float]] = None
    depth_cull: Optional[float] = None


class BrushQueryResponse(BaseModel):
    indices: str        # b64 Int32
    n: int


def _require_seg() -> "SegmentSession":
    seg = _state.get("seg")
    if seg is None:
        raise HTTPException(409, "No segment session loaded — load an annotated scene first")
    return seg


@app.post("/api/segment/brush-query", response_model=BrushQueryResponse)
def brush_query(req: BrushQueryRequest):
    seg = _require_seg()
    center = np.array(req.center, dtype=np.float32)
    cam = np.array(req.camera_ray, dtype=np.float32) if req.camera_ray else None
    idx = seg.brush_query(center, req.radius, camera_ray=cam, depth_cull=req.depth_cull)
    return BrushQueryResponse(indices=_b64(idx.astype(np.int32)), n=int(idx.size))
```

- [ ] **Step 4: Run tests, verify pass**

```bash
.venv/bin/pytest backend/tests/test_segment_endpoints.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/main.py backend/tests/test_segment_endpoints.py
git commit -m "feat(backend): /api/segment/brush-query endpoint"
```

---

## Task 10: `/api/segment/apply` endpoint

Routes to `apply_set_class / apply_merge / apply_reassign`. Returns the indices + new arrays for the affected slice so the frontend can patch its mirror.

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/tests/test_segment_endpoints.py`

- [ ] **Step 1: Write failing tests**

```python
def test_apply_set_class_changes_state(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    body = {
        "op": "set_class",
        "indices": _b64_int32([1, 2]),
        "payload": {"class_id": 2},
    }
    r = client.post("/api/segment/apply", json=body)
    assert r.status_code == 200
    j = r.json()
    assert j["op"] == "set_class"
    assert j["n_affected"] == 2


def test_apply_merge_routes_to_session(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    body = {"op": "merge", "payload": {"source_inst": 2, "target_inst": 0}}
    r = client.post("/api/segment/apply", json=body)
    assert r.status_code == 200
```

Add a helper `_b64_int32(values: list[int]) -> str` at the top of the file.

- [ ] **Step 2: Implement**

```python
class ApplyRequest(BaseModel):
    op: str
    indices: Optional[str] = None     # b64 Int32; required for set_class & reassign
    payload: dict


@app.post("/api/segment/apply")
def segment_apply(req: ApplyRequest):
    seg = _require_seg()
    if req.op == "set_class":
        idx = np.frombuffer(base64.b64decode(req.indices), dtype=np.int32)
        out = seg.apply_set_class(idx, class_id=int(req.payload["class_id"]))
    elif req.op == "merge":
        out = seg.apply_merge(
            source_inst=int(req.payload["source_inst"]),
            target_inst=int(req.payload["target_inst"]),
        )
    elif req.op == "reassign":
        idx = np.frombuffer(base64.b64decode(req.indices), dtype=np.int32)
        out = seg.apply_reassign(
            idx,
            target_inst=req.payload.get("target_inst"),
            target_class=req.payload.get("target_class"),
        )
    else:
        raise HTTPException(400, f"unknown op: {req.op}")
    return _serialize_apply(out)


def _serialize_apply(out: dict) -> dict:
    body = {"op": out["op"], "n_affected": out["n_affected"], "dirty": True}
    if "new_instance_id" in out:
        body["new_instance_id"] = int(out["new_instance_id"])
    if "indices" in out:
        body["indices"] = _b64(out["indices"].astype(np.int32))
        body["after_class"] = _b64(out["after_class"].astype(np.int8))
        body["after_instance"] = _b64(out["after_instance"].astype(np.int32))
    return body
```

- [ ] **Step 3: Run tests, verify pass**

```bash
.venv/bin/pytest backend/tests/test_segment_endpoints.py -v
```

- [ ] **Step 4: Commit**

```bash
git add backend/main.py backend/tests/test_segment_endpoints.py
git commit -m "feat(backend): /api/segment/apply with set_class/merge/reassign"
```

---

## Task 11: `/api/segment/undo` and `/api/segment/redo`

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/tests/test_segment_endpoints.py`

- [ ] **Step 1: Write failing tests**

```python
def test_undo_returns_inverse_delta(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    client.post("/api/segment/apply", json={
        "op": "set_class",
        "indices": _b64_int32([1, 2]),
        "payload": {"class_id": 2},
    })
    r = client.post("/api/segment/undo")
    assert r.status_code == 200
    j = r.json()
    assert j["direction"] == "undo"
    assert j["n_affected"] == 2


def test_undo_returns_204_when_stack_empty(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    r = client.post("/api/segment/undo")
    assert r.status_code == 204
```

- [ ] **Step 2: Implement**

```python
@app.post("/api/segment/undo")
def segment_undo():
    seg = _require_seg()
    out = seg.undo()
    if out is None:
        return Response(status_code=204)
    return _serialize_delta(out)


@app.post("/api/segment/redo")
def segment_redo():
    seg = _require_seg()
    out = seg.redo()
    if out is None:
        return Response(status_code=204)
    return _serialize_delta(out)


def _serialize_delta(out: dict) -> dict:
    return {
        "op": out["op"], "direction": out["direction"],
        "n_affected": out["n_affected"],
        "indices": _b64(out["indices"].astype(np.int32)),
        "after_class": _b64(out["after_class"].astype(np.int8)),
        "after_instance": _b64(out["after_instance"].astype(np.int32)),
    }
```

(Add `from fastapi import Response` if not already imported.)

- [ ] **Step 3: Run tests**

```bash
.venv/bin/pytest backend/tests/test_segment_endpoints.py -v
```

- [ ] **Step 4: Commit**

```bash
git add backend/main.py backend/tests/test_segment_endpoints.py
git commit -m "feat(backend): /api/segment/undo + /redo"
```

---

## Task 12: `/api/segment/save` endpoint

Calls `segment_io.save_labels` against the annotated scan dir. Returns `{ok, n_segments, n_labeled_points}`. Honors `VOXA_DISABLE_ANNOTATION_HISTORY`.

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/tests/test_segment_endpoints.py`

- [ ] **Step 1: Write failing test**

```python
def test_save_writes_labels_to_disk(client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    client.post("/api/segment/apply", json={
        "op": "set_class",
        "indices": _b64_int32([1, 2]),
        "payload": {"class_id": 2},
    })
    r = client.put("/api/segment/save")
    assert r.status_code == 200
    j = r.json()
    assert j["ok"] is True
    arr = np.load(scan_dir_for_loaded_scene / "labels" / "gt_class_ids.npy")
    assert int(arr[1]) == 2 and int(arr[2]) == 2
```

(`scan_dir_for_loaded_scene` is a new fixture: returns the on-disk scan dir for the loaded scene. Build alongside the existing fixtures.)

- [ ] **Step 2: Implement**

```python
@app.put("/api/segment/save")
def segment_save():
    seg = _require_seg()
    src = _resolve(_state["scene"])
    if src.tier != "annotated":
        raise HTTPException(409, "Save is only supported on annotated/<scene> tier")
    scan_dir = Path(src.source_path).parent.parent
    write_history = os.environ.get("VOXA_DISABLE_ANNOTATION_HISTORY") not in ("1", "true", "True")
    try:
        from segment_io import save_labels
        save_labels(
            scan_dir,
            class_ids=seg.class_ids,
            instance_ids=seg.instance_ids,
            positions=seg.positions,
            write_history=write_history,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    seg.dirty = False
    return {"ok": True, "n_labeled_points": int((seg.instance_ids >= 0).sum())}
```

- [ ] **Step 3: Run tests, verify pass**

```bash
.venv/bin/pytest backend/tests/test_segment_endpoints.py -v
```

- [ ] **Step 4: Commit**

```bash
git add backend/main.py backend/tests/test_segment_endpoints.py
git commit -m "feat(backend): /api/segment/save with invariant validation"
```

---

## Task 13: Memory cap enforcement

If `len(pc) > VOXA_MAX_LABEL_POINTS` (default 5_000_000), don't construct the session — `_state["seg"] = None`. `LoadResponse` carries `full_*` only when `seg` exists; the frontend uses absence to grey out segment tools.

**Files:**
- Modify: `backend/main.py`
- Modify: `backend/tests/test_load_endpoint.py`

- [ ] **Step 1: Add the env**

```python
MAX_LABEL_POINTS = int(os.environ.get("VOXA_MAX_LABEL_POINTS", "5000000"))
```

- [ ] **Step 2: Gate session construction**

```python
    if labels is not None and len(pc) <= MAX_LABEL_POINTS:
        _state["seg"] = SegmentSession(...)
    else:
        _state["seg"] = None
```

- [ ] **Step 3: Test (using monkeypatch to lower the cap)**

```python
def test_seg_session_skipped_above_label_cap(monkeypatch, client_with_annotated_scene):
    monkeypatch.setenv("VOXA_MAX_LABEL_POINTS", "1")
    # reload main module so the env is re-read
    import importlib, main
    importlib.reload(main)
    # ... run /api/load ... assert _state['seg'] is None
```

(Reload-the-module pattern matches existing voxa pytest style; if it gets too brittle, accept a lower-fidelity test that calls the cap function directly.)

- [ ] **Step 4: Commit**

```bash
git add backend/main.py backend/tests/test_load_endpoint.py
git commit -m "feat(backend): VOXA_MAX_LABEL_POINTS cap on SegmentSession construction"
```

---

## Task 14: Frontend — extend `api.js` with segment endpoints

**Files:**
- Modify: `frontend/src/api.js`
- Modify: `frontend/src/api.test.js`

- [ ] **Step 1: Write failing tests for new decoders**

`frontend/src/api.test.js`:

```js
import { describe, it, expect } from 'vitest';
import { decodeLoadResponse } from './api.js';   // we'll export this helper

describe('decodeLoadResponse', () => {
  it('decodes full_* fields when present', () => {
    const j = makeFakeLoadResponse({ withFull: true });
    const out = decodeLoadResponse(j);
    expect(out.fullClassIds).toBeInstanceOf(Int8Array);
    expect(out.fullInstanceIds).toBeInstanceOf(Int32Array);
    expect(out.fullPositions).toBeInstanceOf(Float32Array);
  });

  it('returns null fullClassIds when absent', () => {
    const j = makeFakeLoadResponse({ withFull: false });
    const out = decodeLoadResponse(j);
    expect(out.fullClassIds).toBeNull();
  });
});
```

(`makeFakeLoadResponse` is a small helper you'll add — it just builds the JSON shape with b64-encoded test arrays.)

- [ ] **Step 2: Implement `decodeLoadResponse` and segment endpoint helpers**

In `api.js`, refactor `load()` so its decoding lives in a shared `decodeLoadResponse` (so tests can hit it without `fetch`). Add to `VoxaAPI`:

```js
async load(name, { maxPoints = null, wantFullLabels = false } = {}) {
  const body = { name, ...(maxPoints != null ? { max_points: maxPoints } : {}),
                 ...(wantFullLabels ? { want_full_labels: true } : {}) };
  const r = await fetch('/api/load', { method: 'POST',
    headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (!r.ok) throw new Error(`load failed: ${r.status} ${await r.text()}`);
  return decodeLoadResponse(await r.json());
},

async segBrushQuery({ center, radius, cameraRay = null, depthCull = null }) { ... },
async segApply(op, { indices = null, payload = {} }) { ... },
async segUndo() { ... },
async segRedo() { ... },
async segSave() { ... },
```

`decodeLoadResponse` mirrors the existing `load()` decoder + adds:

```js
fullClassIds: j.full_class_ids ? b64ToInt8(j.full_class_ids) : null,
fullInstanceIds: j.full_instance_ids ? b64ToInt32(j.full_instance_ids) : null,
fullPositions: j.full_positions ? b64ToFloat32(j.full_positions) : null,
fullN: j.full_n ?? null,
isFromPrelabel: !!j.is_from_prelabel,
segmentSummary: j.segment_summary || null,
```

- [ ] **Step 3: Run frontend tests**

```bash
npm run test:frontend
```

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/api.js frontend/src/api.test.js
git commit -m "feat(frontend): api.js: full_* decoder + /api/segment/* helpers"
```

---

## Task 15: Frontend — `segment-state.js` reducer

Pure functions — no React, no fetch. Tested with vitest.

**Files:**
- Create: `frontend/src/segment-state.js`
- Create: `frontend/src/segment-state.test.js`

- [ ] **Step 1: Write failing tests**

`frontend/src/segment-state.test.js`:

```js
import { describe, it, expect } from 'vitest';
import { initSegState, applyDelta, recomputeSummary } from './segment-state.js';

const seed = () => initSegState({
  classFull: new Int8Array([-1, 0, 0, 1, 1, 2, -1, 2]),
  instanceFull: new Int32Array([-1, 0, 0, 1, 1, 2, -1, 2]),
  isFromPrelabel: true,
});

describe('segment-state', () => {
  it('applyDelta patches arrays at given indices', () => {
    const s = seed();
    const next = applyDelta(s, {
      indices: new Int32Array([1, 2]),
      after_class: new Int8Array([2, 2]),
      after_instance: new Int32Array([0, 0]),
    });
    expect(next.classFull[1]).toBe(2);
    expect(next.classFull[2]).toBe(2);
    expect(next.instanceFull[1]).toBe(0);
    expect(next.dirty).toBe(true);
  });

  it('recomputeSummary derives per-instance counts', () => {
    const s = seed();
    const sum = recomputeSummary(s);
    expect(sum.get(0).nPoints).toBe(2);
    expect(sum.get(1).classId).toBe(1);
  });
});
```

- [ ] **Step 2: Implement**

`frontend/src/segment-state.js`:

```js
// segment-state.js — pure reducer over the local mirror of class/instance arrays.

export function initSegState({ classFull, instanceFull, isFromPrelabel = false }) {
  return {
    classFull,            // Int8Array
    instanceFull,         // Int32Array
    summary: deriveSummary(classFull, instanceFull),
    dirty: false,
    selection: new Set(),
    activeTool: 'cuboid', // 'cuboid' | 'pick' | 'brush'
    brush: { radius: 0.05, mode: 'create', destInstance: null, destClass: 0 },
    isFromPrelabel,
  };
}

export function applyDelta(state, { indices, after_class, after_instance }) {
  // Mutate in place — caller already has a fresh state ref.
  for (let k = 0; k < indices.length; k++) {
    state.classFull[indices[k]] = after_class[k];
    state.instanceFull[indices[k]] = after_instance[k];
  }
  return { ...state, summary: deriveSummary(state.classFull, state.instanceFull), dirty: true };
}

export function recomputeSummary(state) {
  return deriveSummary(state.classFull, state.instanceFull);
}

function deriveSummary(cls, inst) {
  const m = new Map();   // instanceId → {classId, nPoints}
  for (let i = 0; i < inst.length; i++) {
    const id = inst[i];
    if (id < 0) continue;
    const e = m.get(id);
    if (e === undefined) m.set(id, { classId: cls[i], nPoints: 1 });
    else e.nPoints += 1;
  }
  return m;
}
```

- [ ] **Step 3: Run tests**

```bash
npm run test:frontend
```

Expected: green.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/segment-state.js frontend/src/segment-state.test.js
git commit -m "feat(frontend): segment-state reducer with applyDelta + summary"
```

---

## Task 16: Frontend — `App.jsx` lifts `segState`, requests full labels in Label mode

**Files:**
- Modify: `frontend/src/App.jsx`

- [ ] **Step 1: Add `segState` to the App**

In `App.jsx`:

```jsx
import { initSegState, applyDelta } from './segment-state.js';

const [segState, setSegState] = useState(null);
```

When loading a scene, decide whether to request full labels: only for `annotated/*` scenes when entering or being in Label mode. Simplest: always request when mode === 'label'; re-load on mode switch in/out:

```jsx
useEffect(() => {
  if (!activeScene) return;
  setLoading(true);
  VoxaAPI.load(activeScene.id, { wantFullLabels: mode === 'label' && activeScene.tier === 'annotated' })
    .then(c => {
      setCloud(c);
      if (c.fullClassIds && c.fullInstanceIds) {
        setSegState(initSegState({
          classFull: c.fullClassIds, instanceFull: c.fullInstanceIds,
          isFromPrelabel: c.isFromPrelabel,
        }));
      } else {
        setSegState(null);
      }
    })
    .finally(() => setLoading(false));
}, [activeScene, mode]);
```

- [ ] **Step 2: Pass `segState` and a setter into `LabelMode`**

```jsx
<LabelMode ... segState={segState} setSegState={setSegState} />
```

- [ ] **Step 3: Sanity check that nothing renders broken**

Run `npm run dev`, load any scene in Inspect (no full labels requested), then switch to Label — verify the page doesn't crash and the existing cuboid editor still works. (Manual check, not a test.)

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.jsx
git commit -m "feat(frontend): App.jsx lifts segState, requests full labels in Label mode"
```

---

## Task 17: Frontend — Label tool strip + Pick tool wiring

Adds a left-edge tool strip with three buttons (Cuboid, Pick, Brush) and routes pointer events when Pick is active.

**Files:**
- Create: `frontend/src/segment-tools.jsx`
- Modify: `frontend/src/mode-label.jsx`

- [ ] **Step 1: Stub `segment-tools.jsx`**

```jsx
// segment-tools.jsx — Pick + Brush + tool strip for Label mode segment editing.
import { useState, useEffect, useCallback, useMemo } from 'react';

export function SegmentToolStrip({ activeTool, onChange, hasSegState }) {
  return (
    <div className="seg-toolstrip">
      <button onClick={() => onChange('cuboid')} aria-pressed={activeTool === 'cuboid'}>Cuboid</button>
      <button onClick={() => onChange('pick')} aria-pressed={activeTool === 'pick'} disabled={!hasSegState}>Pick</button>
      <button onClick={() => onChange('brush')} aria-pressed={activeTool === 'brush'} disabled={!hasSegState}>Brush</button>
    </div>
  );
}

// PickTool: subscribes to viewer pointer events when activeTool === 'pick'.
export function PickTool({ viewerRef, segState, onApply, classes }) {
  const [selection, setSelection] = useState(new Set());

  useEffect(() => {
    if (!viewerRef.current || !segState) return;
    const off = viewerRef.current.onPointerPick((idx, evt) => {
      if (idx == null) return;
      const inst = segState.instanceFull[idx];
      if (inst < 0) return;
      const next = new Set(evt.shiftKey ? selection : []);
      next.add(inst);
      setSelection(next);
    });
    return off;
  }, [viewerRef, segState, selection]);

  // Hotkeys — driven by the `hotkey` field on each ClassDef (which the
  // backend serializes from `config/classes.yaml`'s `key`). `R` is reserved
  // as the unlabeled sentinel (writes -1 / -1) — matches existing voxa
  // convention in mode-label.jsx where `classes.find(c => c.hotkey === e.key)`
  // already drives cuboid class hotkeys.
  const keyToClassId = useMemo(() => Object.fromEntries(
    classes.filter(c => c.hotkey).map(c => [c.hotkey.toLowerCase(), c.id])
  ), [classes]);

  useEffect(() => {
    if (!segState) return;
    const on = (e) => {
      if (selection.size === 0) return;
      const k = e.key.toLowerCase();
      const idx = collectIndices(segState.instanceFull, selection);
      if (k === 'r') {
        onApply({ op: 'reassign', indices: idx,
                  payload: { target_inst: null, target_class: null } });
      } else if (k in keyToClassId) {
        onApply({ op: 'set_class', indices: idx,
                  payload: { class_id: keyToClassId[k] } });
      } else if (k === 'm' && selection.size >= 2) {
        const sids = [...selection].sort((a, b) => a - b);
        const target = sids[0];
        for (const src of sids.slice(1)) {
          onApply({ op: 'merge', payload: { source_inst: src, target_inst: target } });
        }
        setSelection(new Set([target]));
      }
    };
    window.addEventListener('keydown', on);
    return () => window.removeEventListener('keydown', on);
  }, [segState, selection, keyToClassId, onApply]);

  return null;   // headless
}

function collectIndices(instanceFull, selection) {
  // Could be optimized with a precomputed index per instance; sufficient for now.
  const out = [];
  for (let i = 0; i < instanceFull.length; i++) {
    if (selection.has(instanceFull[i])) out.push(i);
  }
  return new Int32Array(out);
}
```

- [ ] **Step 2: Wire into `mode-label.jsx`**

Add `activeTool` state, render `<SegmentToolStrip>`, render `<PickTool>` when `activeTool === 'pick'` and `segState` exists. Keep existing cuboid hotkeys gated on `activeTool === 'cuboid'`.

- [ ] **Step 3: Add raycast utilities to `viewer.jsx` imperative handle**

This is the non-trivial Three.js work. The imperative handle exposed by `viewer.jsx` (`useImperativeHandle` already in use for `preset`) gains four new methods:

```js
useImperativeHandle(ref, () => ({
  // existing: preset, ...
  domElement: () => glRef.current.domElement,
  cameraForward: () => {
    const v = new THREE.Vector3();
    cameraRef.current.getWorldDirection(v);
    return [v.x, v.y, v.z];
  },
  // Returns { fullIndex, world } or null. fullIndex is the *full-resolution*
  // point index (not the subsample row). The frontend's subsample→full map
  // is held by the Points.userData.subsampleIdx attached at load time.
  firstHitUnderCursor: (evt) => raycastPoint(evt, pointsRef.current, cameraRef.current, glRef.current),
  // Subscriptions: the viewer doesn't manage React state, just dispatches.
  onPointerPick: (cb) => subscribePick(cb),
  onPointerMove: (cb) => subscribePointerMove(cb),
}), [/* deps */]);
```

Implementation notes:

```js
import * as THREE from 'three';

function raycastPoint(evt, points, camera, renderer) {
  if (!points) return null;
  const rect = renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((evt.clientX - rect.left) / rect.width) * 2 - 1,
    -((evt.clientY - rect.top) / rect.height) * 2 + 1,
  );
  const ray = new THREE.Raycaster();
  ray.setFromCamera(ndc, camera);
  // Threshold scales with rendered point size; without it, sparse clouds rarely hit.
  ray.params.Points.threshold = (points.material.size || 0.012) * 2.0;
  const hits = ray.intersectObject(points, false);
  if (hits.length === 0) return null;
  // Return closest. `index` is the subsample row; map to full-res via userData.
  const sub = hits[0].index;
  const subToFull = points.userData.subsampleIdx;
  const fullIndex = subToFull ? subToFull[sub] : sub;
  return { fullIndex, world: hits[0].point.clone() };
}
```

`subscribePick` and `subscribePointerMove` are simple internal `Set<Callback>` registries dispatched on `pointerdown` / `pointermove` events bound on the canvas.

When loading the cloud, set `pointsRef.current.userData.subsampleIdx = subsampleIdx` (the `Int32Array` decoded in api.js — see Task 14 / Task 20).

- [ ] **Step 4: Sanity-test raycast in isolation**

Add a tiny vitest case in `frontend/src/api.test.js` (or a new `viewer-raycast.test.js`) that covers only the NDC math (Three.js Raycaster itself is well-tested; we just need to confirm the screen-to-NDC translation):

```js
import { describe, it, expect } from 'vitest';
import { evtToNdc } from './viewer.jsx';   // exported helper

it('evtToNdc maps top-left (0,0) to (-1, +1) NDC', () => {
  const rect = { left: 0, top: 0, width: 100, height: 100 };
  const ndc = evtToNdc({ clientX: 0, clientY: 0 }, rect);
  expect(ndc.x).toBeCloseTo(-1);
  expect(ndc.y).toBeCloseTo(1);
});
```

Extract the NDC math into a top-level `evtToNdc` so it's testable without a real renderer.

- [ ] **Step 5: Manual smoke test**

```bash
npm run dev
```

Open `annotated/munich_water_pump`, switch to Label, click Pick, click a point in the viewport. Selection state should update. Press P → all points in that segment recolor. Press M after selecting two segments → they merge.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/segment-tools.jsx frontend/src/mode-label.jsx frontend/src/viewer.jsx
git commit -m "feat(frontend): Label tool strip + Pick tool with class-fix and merge"
```

---

## Task 18: Frontend — Brush tool: gizmo + cursor follow

The brush is a `THREE.Mesh` of a `SphereGeometry`, semi-transparent, that follows the cursor at the first-hit point under the mouse. No painting yet.

**Files:**
- Modify: `frontend/src/segment-tools.jsx`
- Modify: `frontend/src/viewer.jsx`

- [ ] **Step 1: Expose a `firstHitUnderCursor()` method on the viewer ref**

Use a `THREE.Raycaster` against the points geometry; return the closest point's world position + index. Reuse logic from Pick.

- [ ] **Step 2: BrushTool React component**

In `segment-tools.jsx`:

```jsx
export function BrushTool({ viewerRef, segState, classes, activeClassId, onApply }) {
  const [radius, setRadius] = useState(segState?.brush?.radius ?? 0.05);
  const [cursorWorld, setCursorWorld] = useState(null);

  // Mount/unmount the gizmo
  useEffect(() => {
    if (!viewerRef.current) return;
    const handle = viewerRef.current.attachBrushGizmo({
      radius,
      color: classes.find(c => c.id === activeClassId)?.color || '#ffffff',
    });
    return () => handle.remove();
  }, [viewerRef, radius, activeClassId, classes]);

  // Cursor follow
  useEffect(() => {
    if (!viewerRef.current) return;
    const off = viewerRef.current.onPointerMove((evt) => {
      const hit = viewerRef.current.firstHitUnderCursor(evt);
      setCursorWorld(hit?.world || null);
      viewerRef.current.setBrushPosition(hit?.world || null);
    });
    return off;
  }, [viewerRef]);

  // Wheel adjusts radius
  useEffect(() => {
    const on = (e) => {
      if (!viewerRef.current) return;
      e.preventDefault();
      const factor = e.deltaY > 0 ? 1 / 1.2 : 1.2;
      setRadius(r => Math.max(0.005, Math.min(5.0, r * factor)));
    };
    const el = viewerRef.current?.domElement;
    el?.addEventListener('wheel', on, { passive: false });
    return () => el?.removeEventListener('wheel', on);
  }, [viewerRef]);

  return null;
}
```

(`attachBrushGizmo`, `setBrushPosition`, `onPointerMove`, `firstHitUnderCursor`, `domElement` are new methods on the viewer's imperative handle. Implement them in `viewer.jsx`.)

- [ ] **Step 3: Manual smoke test**

`npm run dev` → Label → Brush. Sphere gizmo appears under cursor and follows. Wheel resizes.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/segment-tools.jsx frontend/src/viewer.jsx
git commit -m "feat(frontend): brush gizmo + cursor follow + radius wheel"
```

---

## Task 19: Frontend — Brush paint stroke, optimistic apply

On left-button-drag in Brush, hit `/api/segment/brush-query` per move event, accumulate indices, optimistically apply, fire `/api/segment/apply` (`reassign`) per-stroke (mouse-up commit). Erase mode: hold Alt.

**Files:**
- Modify: `frontend/src/segment-tools.jsx`

- [ ] **Step 1: Stroke-buffer state**

Inside `BrushTool`:

```jsx
const [stroking, setStroking] = useState(false);
const strokeIdx = useRef(new Set());

useEffect(() => {
  const onDown = (e) => { if (e.button === 0) { setStroking(true); strokeIdx.current.clear(); } };
  const onUp = async (e) => {
    if (e.button !== 0 || !stroking) return;
    setStroking(false);
    const idx = Int32Array.from(strokeIdx.current);
    if (idx.length === 0) return;
    const erase = e.altKey;
    // Resolve destination
    const target_inst = (segState.selection.size === 1)
      ? [...segState.selection][0]
      : (erase ? null : -1);  // -1 → backend allocates new
    const target_class = erase ? null
      : (segState.selection.size === 1
          ? segState.summary.get([...segState.selection][0]).classId
          : activeClassId);
    onApply({ op: 'reassign', indices: idx,
              payload: { target_inst, target_class } });
  };
  // attach to viewer canvas
}, [stroking, segState, activeClassId, onApply]);
```

(Real version dispatches per-move queries, but commits one apply per stroke. Keep one apply at mouse-up to keep the undo entry per stroke.)

- [ ] **Step 2: Per-move brush_query**

Inside the move handler (already wired in Task 18), when `stroking`:

```js
const idx = await VoxaAPI.segBrushQuery({
  center: [hit.world.x, hit.world.y, hit.world.z],
  radius,
  cameraRay: viewerRef.current.cameraForward(),
  depthCull: 2 * radius,
});
for (const k of idx) strokeIdx.current.add(k);
// Optimistic recolor: locally update segState colors for these indices.
```

- [ ] **Step 3: Manual test**

`npm run dev` → load `annotated/munich_water_pump` (which has prelabel) → Label → Brush → click and drag → points within sphere recolor. Mouse-up → backend save round-trips.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/segment-tools.jsx
git commit -m "feat(frontend): brush paint stroke with optimistic reassign apply"
```

---

## Task 20: Frontend — color updates via subsample → full index map

Each apply changes `segState.classFull` / `instanceFull`. The rendered cloud uses the *subsampled* indices. Keep a mapping `subToFull: Int32Array` (already returned by backend in subsample path — verify, and add if missing). On `applyDelta`, recompute color attributes only for the subsampled rows whose `full[k]` index is in the affected set.

**Files:**
- Modify: `backend/main.py` — return `subsample_idx_b64` in `LoadResponse` so frontend has the mapping.
- Modify: `frontend/src/api.js` — decode `subsampleIdx`.
- Modify: `frontend/src/viewer.jsx` — `recolorByEdit(affected: Set<int>)` method.

- [ ] **Step 1: Backend — expose subsample index**

In `LoadResponse`:

```python
subsample_idx: Optional[str] = None    # b64 Int32, length == num_subsampled
```

Set it in `load_scene`:

```python
subsample_idx_b64 = _b64(idx.astype(np.int32)) if idx is not None else None
# ...
return LoadResponse(..., subsample_idx=subsample_idx_b64, ...)
```

- [ ] **Step 2: Frontend — decode**

In `decodeLoadResponse` add `subsampleIdx: j.subsample_idx ? b64ToInt32(j.subsample_idx) : null`.

- [ ] **Step 3: Frontend — invalidation hook**

In `viewer.jsx`, add `recolorByEdit({ affectedFullIndices, classFull, instanceFull, colorMode, palette })` that updates the geometry's `color` attribute only at subsample rows whose underlying full index is in `affectedFullIndices`. Cache an inverse map (full → sub) at first call.

- [ ] **Step 4: Wire**

In `mode-label.jsx`, after each `onApply` resolves and `segState` updates, call `viewerRef.current.recolorByEdit(...)`.

- [ ] **Step 5: Manual check**

`npm run dev` → edit a segment → confirm only the affected points recolor (no full-cloud retint flicker).

- [ ] **Step 6: Commit**

```bash
git add backend/main.py frontend/src/api.js frontend/src/viewer.jsx frontend/src/mode-label.jsx
git commit -m "feat: subsample-aware recoloring on segment edits"
```

---

## Task 21: Frontend — Cmd/Ctrl+S extension + dirty indicator

Existing save handler hits cuboid PUT. Extend to fire `VoxaAPI.segSave()` first when `segState.dirty`. Title bar shows a `●` when either family is dirty.

**Files:**
- Modify: `frontend/src/App.jsx`

- [ ] **Step 1: Extend save handler**

```jsx
const handleSave = useCallback(async () => {
  if (segState?.dirty) {
    try {
      await VoxaAPI.segSave();
      setSegState(s => ({ ...s, dirty: false }));
      toast('Segments saved');
    } catch (e) { toast(`Segment save failed: ${e.message}`); return; }
  }
  if (cuboidDirty) { /* existing path */ }
}, [segState, cuboidDirty, ...]);
```

- [ ] **Step 2: Dirty indicator**

In the title-bar component, derive `isDirty = segState?.dirty || cuboidDirty`, render `●` when truthy. Tooltip lists which families are dirty.

- [ ] **Step 3: Manual smoke**

Edit something, verify `●` appears, hit Ctrl+S, verify it goes away and the file changes on disk.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.jsx
git commit -m "feat(frontend): Ctrl+S handles segment save first; dirty indicator"
```

---

## Task 22: Frontend — Undo/Redo keyboard wiring

`Cmd/Ctrl+Z` and `Cmd/Ctrl+Shift+Z` route to `/api/segment/undo` / `redo` when `activeTool` is a segment tool. Apply the returned delta to `segState`.

**Files:**
- Modify: `frontend/src/App.jsx` (or wherever the global keymap lives)

- [ ] **Step 1: Add the keybinds**

```jsx
useEffect(() => {
  const on = async (e) => {
    if (!segState) return;
    const isUndo = (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z' && !e.shiftKey;
    const isRedo = (e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'z' && e.shiftKey;
    if (!isUndo && !isRedo) return;
    e.preventDefault();
    const fn = isUndo ? VoxaAPI.segUndo : VoxaAPI.segRedo;
    const delta = await fn();
    if (!delta) return;
    setSegState(s => applyDelta(s, delta));
    viewerRef.current?.recolorByEdit({ affectedFullIndices: new Set(delta.indices), ... });
  };
  window.addEventListener('keydown', on);
  return () => window.removeEventListener('keydown', on);
}, [segState]);
```

- [ ] **Step 2: Manual test**

Edit, undo, redo, undo. Visual + dirty state behave.

- [ ] **Step 3: Commit**

```bash
git add frontend/src/App.jsx
git commit -m "feat(frontend): Cmd/Ctrl+Z/Shift+Z route to segment undo/redo"
```

---

## Task 23: Frontend — Diff vs prelabel toggle (small)

A toggle in Label HUD: when on, points whose `(class, inst)` differs from the prelabel's are tinted red; unchanged points keep their normal coloring.

**Files:**
- Modify: `frontend/src/App.jsx` — keep `prelabelClassFull` / `prelabelInstanceFull` snapshots at load.
- Modify: `frontend/src/mode-label.jsx` — toggle UI.
- Modify: `frontend/src/viewer.jsx` — accept a `diffMask: Uint8Array` and tint red where set.

- [ ] **Step 1: Snapshot at load**

In `App.jsx`, when `segState` is initialized from a load, also stash `prelabelSnapshot = { classFull: classFull.slice(), instanceFull: instanceFull.slice() }`.

- [ ] **Step 2: Compute diff mask on demand**

```js
const diffMask = useMemo(() => {
  if (!segState || !prelabelSnapshot) return null;
  const m = new Uint8Array(segState.classFull.length);
  for (let i = 0; i < m.length; i++) {
    if (segState.classFull[i] !== prelabelSnapshot.classFull[i] ||
        segState.instanceFull[i] !== prelabelSnapshot.instanceFull[i]) m[i] = 1;
  }
  return m;
}, [segState, prelabelSnapshot]);
```

- [ ] **Step 3: HUD toggle + viewer wiring**

Toggle `showDiff`, pass `diffMask` and `showDiff` into Viewer; viewer overlays a red tint on subsampled rows whose `full[k]` index has `diffMask[full[k]] === 1`.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.jsx frontend/src/mode-label.jsx frontend/src/viewer.jsx
git commit -m "feat(frontend): diff-vs-prelabel toggle in Label HUD"
```

---

## Task 24: Manual UI verification gate

Per CLAUDE.md ("if you can't test the UI, say so"): drive the golden path on `annotated/munich_water_pump` and the negative path. No code changes.

- [ ] **Step 1: Start dev server**

```bash
npm run dev
```

- [ ] **Step 2: Golden path checklist**

- Load `annotated/munich_water_pump`. Confirm Inspect renders unchanged.
- Switch to Label. Confirm full labels load; tool strip shows Cuboid / Pick / Brush; Brush+Pick are enabled (segState present). HUD shows "from prelabel" if labels are absent on disk.
- Pick a segment, press T → it changes to tank's color.
- Shift-pick a second segment, press M → both merge into one.
- Switch to Brush, click-drag to peel a region off → new instance allocated.
- Hold Alt + drag to erase a small region → those points become unlabeled (gray).
- Cmd/Ctrl+Z → reverts the last stroke.
- Cmd/Ctrl+S → toast confirms save; file `<scene>/labels/gt_class_ids.npy` updated; `annotation_history/<ts>/` written.
- Reload the scene → state persists.
- Toggle "Show edits" (diff vs prelabel) → only edited points tint red; toggle off → normal coloring restored.

- [ ] **Step 3: Negative path checklist**

- Load a non-annotated scene (`legacy/test_scene`). Pick + Brush should be disabled.
- Force an invariant violation by overriding `segState.classFull` in the dev console (or by patching `_state.seg` on the backend) and Ctrl+S → expect a clear error toast, no file write.

- [ ] **Step 4: Capture results in PR description**

When opening the PR, include the manual-test checklist with each item ticked or annotated with a known issue.

- [ ] **Step 5: Commit any small fixes you noticed**

If the manual run surfaced bugs, fix them as small follow-up commits on this branch.

---

## Task 25: Open the PR

- [ ] **Step 1: Confirm tests green**

```bash
npm run test:backend
npm run test:frontend
```

- [ ] **Step 2: Push the branch and open the PR**

```bash
git push -u origin feat/per-point-segment-editing
gh pr create --title "Per-point segment editing in Voxa Label mode" --body "$(cat <<'EOF'
## Summary
- Adds Pick + Brush tools to Label mode for editing per-point class + instance IDs on annotated scenes.
- Loads `prelabel/` as the editable starting state when no `labels/` exists; saves SCHEMA-conformant GT to `labels/gt_*.npy` + `gt_segment_metadata.json` with rolling `annotation_history/` snapshots.
- New backend endpoints: `/api/segment/{brush-query, apply, undo, redo, save}`. KD-tree-backed brush queries, bounded undo stack (100 entries), invariant validation on save.
- Cuboid editing untouched. Spec at `docs/superpowers/specs/2026-04-28-voxa-per-point-segment-editing-design.md`.

## Test plan
- [x] Backend pytest green: `npm run test:backend`
- [x] Frontend vitest green: `npm run test:frontend`
- [x] Manual UI walkthrough on `annotated/munich_water_pump` (golden path checklist in Task 24)
- [x] Manual negative-path: legacy scene disables segment tools; invariant violation surfaces error toast cleanly

🤖 Generated with [Claude Code](https://claude.com/claude-code)
EOF
)"
```

---

## Notes for the implementing engineer

- **Backend tests**: `npm run test:backend` runs `pytest` via `scripts/test.sh` which auto-creates `.venv` and installs dev deps. For a single test: `.venv/bin/pytest backend/tests/test_X.py::test_Y -v`.
- **Frontend tests**: `npm run test:frontend` runs vitest. For a single file: `npx vitest run --root frontend src/segment-state.test.js`.
- **Backend autoreload is off by default**. After backend changes, kill and restart `npm run dev` (or set `VOXA_RELOAD=1` for that session).
- **Module-load order matters in pytest**: `conftest.py` sets `VOXA_DATA_DIR` *before* `main` is imported because `main.py` reads the env at import time. Don't move that line.
- **EDITMODE markers** in `App.jsx` are rewritten by Agentation tooling — preserve them when editing tweak defaults.
- **Vite uses polling watchers** intentionally (`vite.config.js`) — don't switch to inotify.
- **`scripts/run.sh` auto-creates `.venv`** the first time it's run; tests do the same via `scripts/test.sh`.
- The plan's TDD examples are *minimal viable tests*. If a test feels too thin to catch real bugs, add the obvious next case before moving on.
