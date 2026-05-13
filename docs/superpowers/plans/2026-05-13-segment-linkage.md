# Segment Linkage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make voxa's editor state survive page reloads, server restarts, and preseg re-runs by adopting a two-layer point-indexed model in `SegmentSession` and persistent session-aux files mirroring the dataset SCHEMA.

**Architecture:** Add immutable `preseg_ids[N]` alongside mutable `instance_ids[N]` in `SegmentSession`. Persist editor state to `session/current.json` + `session/working_*.npy` (commit-pointer pattern, debounced single-flight auto-save). Add three endpoints to hydrate FE on mount and route hide through the server.

**Tech Stack:** Python 3.12, FastAPI, NumPy, pytest, React 18 + Vite, vitest.

**Spec:** `docs/superpowers/specs/2026-05-13-segment-linkage-design.md` — read it before starting any task.

---

## File Map

| File | Role | Action |
|---|---|---|
| `backend/segment_state.py` | `SegmentSession` model + undo/redo | Modify — add layers, hide, freeze_preseg, snap_to_preseg, autosave hook |
| `backend/segment_io.py` | Pure file I/O for SCHEMA dirs | Modify — add `save_session_aux`, `load_session_aux`, `load_working_arrays`, `compute_fingerprint` |
| `backend/scene_registry.py` | Scene discovery + tier paths | Modify — expose `session_dir` on `SceneSource` |
| `backend/main.py` | FastAPI endpoints + `_state` | Modify — wire session dir into load/save, three new endpoints, route preseg through freeze |
| `backend/tests/test_segment_state.py` | SegmentSession unit tests | Modify — add layer + hide + autosave tests |
| `backend/tests/test_segment_io.py` | I/O round-trip tests | Modify — add session-aux tests |
| `backend/tests/test_segment_endpoints.py` | API integration tests | Modify — add /api/segment/state, hide, snap-to-preseg tests |
| `backend/tests/test_load_endpoint.py` | Load + reload tests | Modify — add server-restart-recovery test |
| `frontend/src/api.js` | API client | Modify — add wrappers for new endpoints |
| `frontend/src/segment-state.js` | FE segState slice | Modify — hydration helper, drop local hide cache |
| `frontend/src/segment-tools.jsx` | Edit-mode UI | Modify — route hide through API |
| `frontend/src/segment-state.test.js` | FE unit tests | Modify — add hydration tests |
| `../../data/lidar/SCHEMA.md` | Dataset schema | Modify — additive v1.1 patch |

**Out-of-tree edits** (`engine/data/lidar/SCHEMA.md`) happen at the very end and are committed in the engine repo, not in voxa.

---

## Task 1: Fingerprint + atomic-write helpers in `segment_io.py`

Pure utilities first — no FastAPI dependency, easy to test, used by every later task.

**Files:**
- Modify: `backend/segment_io.py`
- Test: `backend/tests/test_segment_io.py`

- [ ] **Step 1: Write failing tests for `compute_fingerprint` and `atomic_write_npy`**

Append to `backend/tests/test_segment_io.py`:

```python
import numpy as np
from pathlib import Path
from segment_io import compute_fingerprint, atomic_write_npy, atomic_write_json


def test_compute_fingerprint_is_content_addressed():
    a = np.array([1, 2, 3], dtype=np.int32)
    b = np.array([1, 2, 3], dtype=np.int32)
    c = np.array([1, 2, 4], dtype=np.int32)
    assert compute_fingerprint(a) == compute_fingerprint(b)
    assert compute_fingerprint(a) != compute_fingerprint(c)
    assert compute_fingerprint(a).startswith("sha256:")


def test_atomic_write_npy_round_trip(tmp_path):
    p = tmp_path / "x.npy"
    arr = np.arange(100, dtype=np.int32)
    atomic_write_npy(p, arr)
    assert p.exists()
    assert not (tmp_path / "x.npy.tmp").exists()
    np.testing.assert_array_equal(np.load(p), arr)


def test_atomic_write_json_round_trip(tmp_path):
    p = tmp_path / "x.json"
    atomic_write_json(p, {"a": 1, "b": [2, 3]})
    assert p.exists()
    assert not (tmp_path / "x.json.tmp").exists()
    import json
    assert json.loads(p.read_text()) == {"a": 1, "b": [2, 3]}
```

- [ ] **Step 2: Run tests, expect ImportError**

Run: `.venv/bin/pytest backend/tests/test_segment_io.py -v -k "fingerprint or atomic_write"`
Expected: ImportError on `compute_fingerprint` / `atomic_write_npy`.

- [ ] **Step 3: Implement the helpers**

Add to `backend/segment_io.py` (after imports):

```python
import hashlib
import os


def compute_fingerprint(arr: np.ndarray) -> str:
    """Content-addressed sha256 of a numpy array's bytes. Stable across
    save/load (numpy preserves byte layout for fixed dtypes)."""
    h = hashlib.sha256()
    h.update(bytes(str(arr.dtype), "ascii"))
    h.update(b":")
    h.update(bytes(str(arr.shape), "ascii"))
    h.update(b":")
    h.update(arr.tobytes(order="C"))
    return f"sha256:{h.hexdigest()}"


def atomic_write_npy(path: Path, arr: np.ndarray) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    with open(tmp, "wb") as f:
        np.save(f, arr)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(payload, indent=2, sort_keys=True).encode("utf-8")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `.venv/bin/pytest backend/tests/test_segment_io.py -v -k "fingerprint or atomic_write"`
Expected: 3 passing.

- [ ] **Step 5: Commit**

```bash
git add backend/segment_io.py backend/tests/test_segment_io.py
git commit -m "feat(segment-io): content fingerprint + atomic write helpers"
```

---

## Task 2: `save_session_aux` / `load_session_aux` / `load_working_arrays`

Build the on-disk session contract. `current.json` is the commit pointer (per spec §3 atomicity).

**Files:**
- Modify: `backend/segment_io.py`
- Test: `backend/tests/test_segment_io.py`

- [ ] **Step 1: Write failing round-trip + commit-pointer test**

Append to `backend/tests/test_segment_io.py`:

```python
from segment_io import save_session_aux, load_session_aux, load_working_arrays


def test_session_aux_round_trip(tmp_path):
    session_dir = tmp_path / "session"
    class_ids = np.full(100, -1, dtype=np.int8)
    class_ids[10:20] = 2
    inst_ids = np.full(100, -1, dtype=np.int32)
    inst_ids[10:20] = 7
    aux = {
        "schema_version": 1,
        "preseg_run_id": "20260513-100000",
        "preseg_fingerprint": "sha256:abc",
        "source_fingerprint": "sha256:def",
        "hidden_inst_ids": [7],
        "is_from_prelabel": False,
        "dirty": True,
    }
    save_session_aux(session_dir, aux, class_ids=class_ids, instance_ids=inst_ids)
    assert (session_dir / "current.json").exists()
    assert (session_dir / "working_class_ids.npy").exists()
    assert (session_dir / "working_segment_ids.npy").exists()

    out = load_session_aux(session_dir)
    assert out is not None
    assert out["preseg_run_id"] == "20260513-100000"
    assert out["hidden_inst_ids"] == [7]

    wc, wi = load_working_arrays(session_dir, n_points=100)
    np.testing.assert_array_equal(wc, class_ids)
    np.testing.assert_array_equal(wi, inst_ids)


def test_load_working_arrays_returns_none_without_current_json(tmp_path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    # Working arrays present but no current.json → ignore (commit-pointer rule).
    atomic_write_npy(session_dir / "working_class_ids.npy",
                     np.zeros(50, dtype=np.int8))
    atomic_write_npy(session_dir / "working_segment_ids.npy",
                     np.zeros(50, dtype=np.int32))
    assert load_working_arrays(session_dir, n_points=50) is None


def test_load_working_arrays_returns_none_on_shape_mismatch(tmp_path):
    session_dir = tmp_path / "session"
    save_session_aux(session_dir, {"schema_version": 1},
                     class_ids=np.zeros(50, dtype=np.int8),
                     instance_ids=np.zeros(50, dtype=np.int32))
    assert load_working_arrays(session_dir, n_points=999) is None
```

- [ ] **Step 2: Run tests, expect failure**

Run: `.venv/bin/pytest backend/tests/test_segment_io.py -v -k "session_aux or working_arrays"`
Expected: 3 failing on missing imports.

- [ ] **Step 3: Implement the three functions**

Add to `backend/segment_io.py`:

```python
SESSION_SCHEMA_VERSION = 1


def save_session_aux(
    session_dir: Path,
    aux: dict,
    *,
    class_ids: Optional[np.ndarray] = None,
    instance_ids: Optional[np.ndarray] = None,
) -> None:
    """Atomically persist editor session state.

    Order: working_*.npy first, then current.json (commit pointer). On a
    crash between np-renames and current.json rename, the next reload sees
    the previous-consistent current.json and ignores any half-updated
    working_* — see spec §3.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    if class_ids is not None:
        atomic_write_npy(session_dir / "working_class_ids.npy",
                         class_ids.astype(np.int8, copy=False))
    if instance_ids is not None:
        atomic_write_npy(session_dir / "working_segment_ids.npy",
                         instance_ids.astype(np.int32, copy=False))
    payload = dict(aux)
    payload.setdefault("schema_version", SESSION_SCHEMA_VERSION)
    payload["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    atomic_write_json(session_dir / "current.json", payload)


def load_session_aux(session_dir: Path) -> Optional[dict]:
    """Read current.json or return None if absent/unreadable."""
    p = session_dir / "current.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def load_working_arrays(
    session_dir: Path, n_points: int
) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Return (class_ids int8, instance_ids int32) iff current.json exists
    AND both working files are present AND shapes match n_points."""
    if load_session_aux(session_dir) is None:
        return None
    cp = session_dir / "working_class_ids.npy"
    ip = session_dir / "working_segment_ids.npy"
    if not (cp.exists() and ip.exists()):
        return None
    try:
        ci = np.load(cp).astype(np.int8, copy=False)
        ii = np.load(ip).astype(np.int32, copy=False)
    except (OSError, ValueError):
        return None
    if ci.shape != (n_points,) or ii.shape != (n_points,):
        return None
    return ci, ii
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `.venv/bin/pytest backend/tests/test_segment_io.py -v -k "session_aux or working_arrays"`
Expected: 3 passing.

- [ ] **Step 5: Commit**

```bash
git add backend/segment_io.py backend/tests/test_segment_io.py
git commit -m "feat(segment-io): persistent session aux (current.json + working_*.npy)"
```

---

## Task 3: Add `preseg_ids` layer + `freeze_preseg` to `SegmentSession`

The immutable layer. No autosave yet — that comes in Task 6.

**Files:**
- Modify: `backend/segment_state.py`
- Test: `backend/tests/test_segment_state.py`

- [ ] **Step 1: Write failing tests**

Append to `backend/tests/test_segment_state.py`:

```python
def test_segment_session_has_preseg_layer():
    import numpy as np
    from segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.full(10, -1, dtype=np.int8),
        instance_ids=np.full(10, -1, dtype=np.int32),
        positions=pts,
    )
    assert s.preseg_ids.shape == (10,)
    assert s.preseg_ids.dtype == np.int32
    assert (s.preseg_ids == -1).all()
    assert s.preseg_run_id is None
    assert s.preseg_fingerprint is None


def test_freeze_preseg_stamps_run_id_and_fingerprint():
    import numpy as np
    from segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.full(10, -1, dtype=np.int8),
        instance_ids=np.full(10, -1, dtype=np.int32),
        positions=pts,
    )
    new_pre = np.arange(10, dtype=np.int32)
    s.freeze_preseg(new_pre, run_id="abc")
    np.testing.assert_array_equal(s.preseg_ids, new_pre)
    assert s.preseg_run_id == "abc"
    assert s.preseg_fingerprint.startswith("sha256:")


def test_freeze_preseg_immutable_through_merge():
    import numpy as np
    from segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(10, dtype=np.int8),
        instance_ids=np.array([0]*5 + [1]*5, dtype=np.int32),
        positions=pts,
    )
    s.freeze_preseg(np.array([0]*5 + [1]*5, dtype=np.int32))
    s.apply_merge(source_inst=0, target_inst=1)
    assert (s.instance_ids == 1).all()
    np.testing.assert_array_equal(s.preseg_ids, np.array([0]*5 + [1]*5))


def test_current_inst_ids_for_preseg_after_merge():
    """Core bug-fix scenario: hide(preseg=0) after a merge resolves to live id 1."""
    import numpy as np
    from segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(10, dtype=np.int8),
        instance_ids=np.array([0]*5 + [1]*5, dtype=np.int32),
        positions=pts,
    )
    s.freeze_preseg(s.instance_ids.copy())
    s.apply_merge(source_inst=0, target_inst=1)
    assert s.current_inst_ids_for_preseg(0) == {1}
    assert s.current_inst_ids_for_preseg(1) == {1}
```

- [ ] **Step 2: Run tests, expect failure**

Run: `.venv/bin/pytest backend/tests/test_segment_state.py -v -k "preseg_layer or freeze_preseg or immutable_through_merge or current_inst_ids"`
Expected: 4 failing — missing attrs / methods.

- [ ] **Step 3: Add fields and methods to `SegmentSession.__init__`**

Edit `backend/segment_state.py`:

```python
# In __init__, after self.positions = ...:
        self.preseg_ids: np.ndarray = np.full(n, -1, dtype=np.int32)
        self.preseg_run_id: Optional[str] = None
        self.preseg_fingerprint: Optional[str] = None
        self.source_fingerprint: Optional[str] = None
        self.hidden_inst_ids: set[int] = set()
```

Add methods to `SegmentSession`:

```python
    def freeze_preseg(
        self,
        preseg_ids: np.ndarray,
        *,
        run_id: Optional[str] = None,
    ) -> None:
        """Replace the immutable preseg layer. Not undoable; this is a
        session-scope event. See spec §2."""
        from segment_io import compute_fingerprint
        if preseg_ids.shape != self.instance_ids.shape:
            raise ValueError(
                f"freeze_preseg: expected {self.instance_ids.shape}, "
                f"got {preseg_ids.shape}",
            )
        self.preseg_ids = preseg_ids.astype(np.int32, copy=False)
        self.preseg_run_id = run_id
        self.preseg_fingerprint = compute_fingerprint(self.preseg_ids)

    def current_inst_ids_for_preseg(self, preseg_id: int) -> set[int]:
        """Which live instance ids does preseg cluster `preseg_id` currently
        cover? Resolves through any merges/reassigns since freeze_preseg."""
        mask = self.preseg_ids == int(preseg_id)
        if not mask.any():
            return set()
        return set(int(v) for v in np.unique(self.instance_ids[mask]) if v >= 0)
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `.venv/bin/pytest backend/tests/test_segment_state.py -v -k "preseg_layer or freeze_preseg or immutable_through_merge or current_inst_ids"`
Expected: 4 passing.

- [ ] **Step 5: Run full segment_state suite to ensure no regression**

Run: `.venv/bin/pytest backend/tests/test_segment_state.py -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add backend/segment_state.py backend/tests/test_segment_state.py
git commit -m "feat(segment-state): immutable preseg_ids layer + freeze_preseg"
```

---

## Task 4: `hide_instance` / `unhide_instance` + `snap_to_preseg`

Visual-affordance ops (hide) and the preseg-revert op.

**Files:**
- Modify: `backend/segment_state.py`
- Test: `backend/tests/test_segment_state.py`

- [ ] **Step 1: Write failing tests**

```python
def test_hide_unhide_inst():
    import numpy as np
    from segment_state import SegmentSession
    pts = np.zeros((4, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(4, dtype=np.int8),
        instance_ids=np.array([0, 0, 1, 1], dtype=np.int32),
        positions=pts,
    )
    s.hide_instance(0)
    s.hide_instance(1)
    assert s.hidden_inst_ids == {0, 1}
    s.unhide_instance(0)
    assert s.hidden_inst_ids == {1}
    s.unhide_instance(999)  # no-op
    assert s.hidden_inst_ids == {1}


def test_hide_survives_merge():
    """preseg=0 hidden → merge into preseg=1's live id → hide still resolves to the merged live id."""
    import numpy as np
    from segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(10, dtype=np.int8),
        instance_ids=np.array([0]*5 + [1]*5, dtype=np.int32),
        positions=pts,
    )
    s.freeze_preseg(s.instance_ids.copy())
    # User clicks "hide preseg cluster 0" → FE asks server for current id → server hides live id 0.
    s.hide_instance(0)
    s.apply_merge(source_inst=0, target_inst=1)
    # After merge, the points formerly tagged preseg=0 are now live id 1.
    # The FE can re-resolve via current_inst_ids_for_preseg(0) → {1} and hide that.
    assert s.current_inst_ids_for_preseg(0) == {1}


def test_snap_to_preseg_reverts_merged_object():
    import numpy as np
    from segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.full(10, 2, dtype=np.int8),
        instance_ids=np.array([0]*5 + [1]*5, dtype=np.int32),
        positions=pts,
    )
    s.freeze_preseg(np.array([0]*5 + [1]*5, dtype=np.int32))
    s.apply_merge(source_inst=0, target_inst=1)
    assert (s.instance_ids == 1).all()
    s.snap_to_preseg([1])
    # Points that were preseg=0 go back to instance=0; preseg=1 stays as 1.
    np.testing.assert_array_equal(s.instance_ids, np.array([0]*5 + [1]*5))


def test_snap_to_preseg_undoable():
    import numpy as np
    from segment_state import SegmentSession
    pts = np.zeros((10, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(10, dtype=np.int8),
        instance_ids=np.array([0]*5 + [1]*5, dtype=np.int32),
        positions=pts,
    )
    s.freeze_preseg(np.array([0]*5 + [1]*5, dtype=np.int32))
    s.apply_merge(0, 1)
    s.snap_to_preseg([1])
    s.undo()
    assert (s.instance_ids == 1).all()
```

- [ ] **Step 2: Run tests, expect failure**

Run: `.venv/bin/pytest backend/tests/test_segment_state.py -v -k "hide_unhide or hide_survives or snap_to_preseg"`
Expected: 4 failing.

- [ ] **Step 3: Implement methods**

Add to `SegmentSession`:

```python
    def hide_instance(self, inst_id: int) -> None:
        self.hidden_inst_ids.add(int(inst_id))

    def unhide_instance(self, inst_id: int) -> None:
        self.hidden_inst_ids.discard(int(inst_id))

    def snap_to_preseg(self, inst_ids: list[int]) -> dict:
        """For every point whose live instance is in inst_ids, reset its
        instance to its preseg id (class is untouched). On the undo stack."""
        live = np.asarray(inst_ids, dtype=np.int32)
        mask = np.isin(self.instance_ids, live) & (self.preseg_ids >= 0)
        indices = np.flatnonzero(mask).astype(np.int32)
        if indices.size == 0:
            return {"op": "snap_to_preseg", "n_affected": 0}
        before_cls = self.class_ids[indices].copy()
        before_inst = self.instance_ids[indices].copy()
        after_inst = self.preseg_ids[indices].copy().astype(np.int32)
        self.instance_ids[indices] = after_inst
        delta = _Delta(
            op="snap_to_preseg", indices=indices,
            before_cls=before_cls, before_inst=before_inst,
            after_cls=before_cls.copy(),  # class unchanged
            after_inst=after_inst,
        )
        self._undo.append(delta)
        if len(self._undo) > self.history_cap:
            self._undo.popleft()
        self._redo.clear()
        self.dirty = True
        return {
            "op": "snap_to_preseg",
            "n_affected": int(indices.size),
            "indices": indices,
            "after_class": before_cls,
            "after_instance": after_inst,
        }
```

- [ ] **Step 4: Run tests, expect PASS**

Run: `.venv/bin/pytest backend/tests/test_segment_state.py -v`
Expected: all green (full suite).

- [ ] **Step 5: Commit**

```bash
git add backend/segment_state.py backend/tests/test_segment_state.py
git commit -m "feat(segment-state): hide/unhide + snap_to_preseg (undoable)"
```

---

## Task 5: `scene_registry` — expose `session_dir` on `SceneSource`

Resolve the per-scene session path: SCHEMA tier → `<scan>/session/`, others → `data/sessions/<scene_id>/`.

**Files:**
- Modify: `backend/scene_registry.py`
- Test: `backend/tests/test_scene_registry.py`

- [ ] **Step 1: Write failing test**

Append to `backend/tests/test_scene_registry.py`:

```python
def test_session_dir_for_annotated_tier(tmp_path):
    from scene_registry import SceneRegistry
    scan = tmp_path / "annotated" / "munich_water_pump"
    (scan / "source").mkdir(parents=True)
    (scan / "source" / "scan.ply").touch()
    (scan / "labels").mkdir()
    (scan / "labels" / "gt_class_ids.npy").touch()
    (scan / "labels" / "gt_segment_ids.npy").touch()
    (scan / "labels" / "gt_segment_metadata.json").write_text("{}")
    (scan / "meta.json").write_text('{"n_points": 1}')

    reg = SceneRegistry(legacy_root=tmp_path / "nonexistent",
                       lidar_root=tmp_path)
    src = reg.resolve("annotated/munich_water_pump")
    assert src.session_dir == scan / "session"


def test_session_dir_for_legacy_tier(tmp_path):
    from scene_registry import SceneRegistry
    legacy = tmp_path / "legacy"
    scenes = legacy / "scenes" / "foo"
    scenes.mkdir(parents=True)
    (scenes / "source.ply").touch()
    data_dir = tmp_path / "data"
    reg = SceneRegistry(legacy_root=legacy, lidar_root=None, data_dir=data_dir)
    src = reg.resolve("foo")
    assert src.session_dir == data_dir / "sessions" / "foo"
```

- [ ] **Step 2: Run tests, expect failure**

Run: `.venv/bin/pytest backend/tests/test_scene_registry.py -v -k "session_dir"`
Expected: failing on missing `session_dir` attr or missing `data_dir` ctor arg.

- [ ] **Step 3: Modify `SceneSource` and `SceneRegistry`**

In `backend/scene_registry.py`:

1. Add `session_dir: Path` to the `SceneSource` dataclass.
2. Add `data_dir: Optional[Path] = None` to `SceneRegistry.__init__`.
3. In the constructors of `SceneSource` for each tier, set:
   - `annotated` tier: `session_dir = scan_root / "session"`
   - other tiers (legacy/decimated/raw): `session_dir = data_dir / "sessions" / scene_id_safe` where `scene_id_safe = scene_id.replace("/", "__")`. Raise if `data_dir is None`.

- [ ] **Step 4: Wire the new ctor arg in `main.py`**

In `main.py`, where `SceneRegistry(...)` is instantiated, pass `data_dir=DATA_DIR`.

- [ ] **Step 5: Run tests, expect PASS**

Run: `.venv/bin/pytest backend/tests/test_scene_registry.py -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add backend/scene_registry.py backend/main.py backend/tests/test_scene_registry.py
git commit -m "feat(scene-registry): expose session_dir per scene tier"
```

---

## Task 6: Auto-save hook in `SegmentSession._apply` (debounced single-flight)

Wire mutations to disk via `save_session_aux`. The session needs a `session_dir` attr to know where to write.

**Files:**
- Modify: `backend/segment_state.py`
- Test: `backend/tests/test_segment_state.py`

- [ ] **Step 1: Write failing test for autosave-on-apply**

```python
def test_autosave_writes_working_files_and_current_json(tmp_path):
    import numpy as np, json
    from segment_state import SegmentSession
    pts = np.zeros((4, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.full(4, -1, dtype=np.int8),
        instance_ids=np.full(4, -1, dtype=np.int32),
        positions=pts,
        session_dir=tmp_path,
        autosave_debounce_s=0.0,  # synchronous flush in tests
    )
    s.apply_set_class(np.array([0, 1], dtype=np.int32), class_id=2)
    s.flush_autosave()
    assert (tmp_path / "current.json").exists()
    assert (tmp_path / "working_class_ids.npy").exists()
    assert (tmp_path / "working_segment_ids.npy").exists()
    payload = json.loads((tmp_path / "current.json").read_text())
    assert payload["dirty"] is True
    assert payload["schema_version"] == 1


def test_autosave_includes_hidden_and_preseg_run(tmp_path):
    import numpy as np, json
    from segment_state import SegmentSession
    pts = np.zeros((4, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.zeros(4, dtype=np.int8),
        instance_ids=np.array([0, 0, 1, 1], dtype=np.int32),
        positions=pts,
        session_dir=tmp_path,
        autosave_debounce_s=0.0,
    )
    s.freeze_preseg(np.array([0, 0, 1, 1], dtype=np.int32), run_id="r1")
    s.hide_instance(0)
    s.flush_autosave()
    payload = json.loads((tmp_path / "current.json").read_text())
    assert payload["preseg_run_id"] == "r1"
    assert payload["hidden_inst_ids"] == [0]
    assert payload["preseg_fingerprint"].startswith("sha256:")


def test_autosave_disabled_when_no_session_dir(tmp_path):
    """Sessions constructed without session_dir (transient tests, raw-tier scenes
    pre-data_dir wiring) must not crash on apply."""
    import numpy as np
    from segment_state import SegmentSession
    pts = np.zeros((4, 3), dtype=np.float32)
    s = SegmentSession(
        class_ids=np.full(4, -1, dtype=np.int8),
        instance_ids=np.full(4, -1, dtype=np.int32),
        positions=pts,
    )
    s.apply_set_class(np.array([0], dtype=np.int32), class_id=1)
    assert not (tmp_path / "current.json").exists()
```

- [ ] **Step 2: Run tests, expect failure**

Run: `.venv/bin/pytest backend/tests/test_segment_state.py -v -k "autosave"`
Expected: 3 failing.

- [ ] **Step 3: Implement single-flight + debounced autosave**

In `SegmentSession.__init__`, add new kwargs:

```python
    def __init__(
        self,
        class_ids, instance_ids, positions,
        *,
        is_from_prelabel: bool = False,
        session_dir: Optional[Path] = None,
        autosave_debounce_s: float = 0.25,
    ) -> None:
        ...
        self.session_dir: Optional[Path] = Path(session_dir) if session_dir else None
        self._autosave_debounce_s = float(autosave_debounce_s)
        self._autosave_timer: Optional[threading.Timer] = None
        self._autosave_lock = threading.Lock()
```

Add helper methods:

```python
    def _aux_payload(self) -> dict:
        return {
            "schema_version": 1,
            "preseg_run_id": self.preseg_run_id,
            "preseg_fingerprint": self.preseg_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "hidden_inst_ids": sorted(int(x) for x in self.hidden_inst_ids),
            "is_from_prelabel": bool(self.is_from_prelabel),
            "dirty": bool(self.dirty),
        }

    def _do_autosave(self, write_arrays: bool) -> None:
        if self.session_dir is None:
            return
        from segment_io import save_session_aux
        with self._autosave_lock:
            save_session_aux(
                self.session_dir,
                self._aux_payload(),
                class_ids=self.class_ids if write_arrays else None,
                instance_ids=self.instance_ids if write_arrays else None,
            )

    def schedule_autosave(self, *, write_arrays: bool = True) -> None:
        if self.session_dir is None:
            return
        if self._autosave_debounce_s <= 0.0:
            self._do_autosave(write_arrays)
            return
        # Single-flight: cancel any pending timer and replace.
        if self._autosave_timer is not None:
            self._autosave_timer.cancel()
        self._autosave_timer = threading.Timer(
            self._autosave_debounce_s,
            self._do_autosave, args=(write_arrays,),
        )
        self._autosave_timer.daemon = True
        self._autosave_timer.start()

    def flush_autosave(self) -> None:
        if self._autosave_timer is not None:
            self._autosave_timer.cancel()
            self._autosave_timer = None
        self._do_autosave(write_arrays=True)
```

In every existing op that mutates state (`_apply`, `undo`, `redo`, `freeze_preseg`, `hide_instance`, `unhide_instance`, `snap_to_preseg`), add at the end:

- `_apply`, `undo`, `redo`, `snap_to_preseg` → `self.schedule_autosave(write_arrays=True)`
- `freeze_preseg` → also writes arrays (preseg layer changed conceptually counts) → `self.schedule_autosave(write_arrays=True)`
- `hide_instance`, `unhide_instance` → `self.schedule_autosave(write_arrays=False)` (only current.json needs updating)

Add `import threading` and `from pathlib import Path` near the top.

- [ ] **Step 4: Run tests, expect PASS**

Run: `.venv/bin/pytest backend/tests/test_segment_state.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add backend/segment_state.py backend/tests/test_segment_state.py
git commit -m "feat(segment-state): debounced single-flight autosave on every mutation"
```

---

## Task 7: Hydrate `SegmentSession` from session aux on `/api/load`

When the user reloads, restore from `session/working_*` if newer than `labels/`, else fall through to existing path. Also stamp `source_fingerprint` on new sessions.

**Files:**
- Modify: `backend/main.py`
- Test: `backend/tests/test_load_endpoint.py`

- [ ] **Step 1: Write failing recovery test**

Append to `backend/tests/test_load_endpoint.py`:

```python
def test_load_recovers_in_progress_session_after_server_restart(
    client, tmp_data_dir, fake_annotated_scene
):
    """First load → mutate → second load (fresh _state) recovers."""
    scene_id = fake_annotated_scene  # fixture creates an annotated scan dir
    r = client.post("/api/load", json={"scene": scene_id, "want_full_labels": True})
    assert r.status_code == 200

    # Brush a class onto a few points.
    r = client.post("/api/segment/reassign", json={
        "indices": [0, 1, 2], "target_inst": -1, "target_class": 2,
    })
    assert r.status_code == 200

    # Force-clear in-memory state to simulate a server restart.
    from main import _state
    _state.update(scene=None, pc=None, seg=None)

    # Reload same scene — expect working arrays to come back.
    r = client.post("/api/load", json={"scene": scene_id, "want_full_labels": True})
    assert r.status_code == 200

    # State endpoint reports the mutated class on those points.
    r = client.get("/api/segment/state")
    body = r.json()
    assert body["has_seg"] is True
    assert body["dirty"] is True
```

(Use the existing `fake_annotated_scene` fixture from `conftest.py` if present; else add one that builds the smallest valid SCHEMA scan dir under `tmp_data_dir`.)

- [ ] **Step 2: Run test, expect failure (no `/api/segment/state` yet OR no recovery)**

Run: `.venv/bin/pytest backend/tests/test_load_endpoint.py -v -k "recovers_in_progress"`
Expected: failure.

- [ ] **Step 3: Modify the load handler in `main.py`**

In `main.py`, in the `/api/load` handler around the `SegmentSession(...)` construction:

```python
from segment_io import (
    load_session_aux, load_working_arrays, compute_fingerprint,
)

source_fp = compute_fingerprint(pc.points.astype(np.float32))
session_dir = src.session_dir

# Try to recover an in-progress working session.
recovered = None
if session_dir is not None:
    aux = load_session_aux(session_dir)
    if aux is not None and aux.get("source_fingerprint") == source_fp:
        wa = load_working_arrays(session_dir, n_points=len(pc))
        if wa is not None:
            recovered = (wa[0], wa[1], aux)

if recovered is not None:
    wc, wi, aux = recovered
    seg = SegmentSession(
        class_ids=wc, instance_ids=wi, positions=pc.points,
        is_from_prelabel=bool(aux.get("is_from_prelabel", False)),
        session_dir=session_dir,
    )
    seg.source_fingerprint = source_fp
    seg.preseg_run_id = aux.get("preseg_run_id")
    seg.preseg_fingerprint = aux.get("preseg_fingerprint")
    seg.hidden_inst_ids = set(aux.get("hidden_inst_ids", []))
    seg.dirty = bool(aux.get("dirty", False))
    _state["seg"] = seg
elif keep_prev_seg:
    _state["seg"] = prev_seg
elif labels is not None and len(pc) <= MAX_LABEL_POINTS:
    seg = SegmentSession(
        class_ids=labels.class_ids,
        instance_ids=labels.instance_ids,
        positions=pc.points,
        is_from_prelabel=is_from_prelabel,
        session_dir=session_dir,
    )
    seg.source_fingerprint = source_fp
    _state["seg"] = seg
else:
    _state["seg"] = None
```

Add a placeholder `/api/segment/state` endpoint (full implementation in Task 8) so the test can call it:

```python
@app.get("/api/segment/state")
def segment_state_get():
    seg = _state.get("seg")
    if seg is None:
        return {"has_seg": False}
    return {
        "has_seg": True,
        "n_points": int(len(seg.instance_ids)),
        "preseg_run_id": seg.preseg_run_id,
        "preseg_fingerprint": seg.preseg_fingerprint,
        "source_fingerprint": seg.source_fingerprint,
        "hidden_inst_ids": sorted(int(x) for x in seg.hidden_inst_ids),
        "is_from_prelabel": bool(seg.is_from_prelabel),
        "dirty": bool(seg.dirty),
    }
```

- [ ] **Step 4: Run test, expect PASS**

Run: `.venv/bin/pytest backend/tests/test_load_endpoint.py -v -k "recovers_in_progress"`
Expected: passing.

- [ ] **Step 5: Run full backend suite — no regressions**

Run: `.venv/bin/pytest backend/tests -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add backend/main.py backend/tests/test_load_endpoint.py
git commit -m "feat(load): recover in-progress session from session/working_* on reload"
```

---

## Task 8: `/api/segment/state` full payload + hide / unhide / snap-to-preseg endpoints

Promote the placeholder to a typed response model + the three mutating endpoints.

**Files:**
- Modify: `backend/main.py`
- Test: `backend/tests/test_segment_endpoints.py`

- [ ] **Step 1: Write failing tests**

Append to `backend/tests/test_segment_endpoints.py`:

```python
def test_segment_state_full_payload(client, loaded_scene):
    r = client.get("/api/segment/state")
    body = r.json()
    assert body["has_seg"] in (True, False)
    if body["has_seg"]:
        assert {"n_points", "preseg_run_id", "preseg_fingerprint",
                "source_fingerprint", "hidden_inst_ids",
                "is_from_prelabel", "dirty"} <= body.keys()


def test_hide_unhide_round_trip(client, loaded_scene_with_seg):
    r = client.post("/api/segment/hide", json={"inst_id": 0})
    assert r.status_code == 200
    body = client.get("/api/segment/state").json()
    assert 0 in body["hidden_inst_ids"]
    r = client.delete("/api/segment/hide/0")
    assert r.status_code == 200
    body = client.get("/api/segment/state").json()
    assert 0 not in body["hidden_inst_ids"]


def test_snap_to_preseg_reverts_merge(client, loaded_scene_with_preseg_and_merge):
    """Scene fixture has freeze_preseg([0,0,1,1]) + merge(0→1).
    snap-to-preseg([1]) splits the merged points back."""
    r = client.post("/api/segment/snap-to-preseg", json={"inst_ids": [1]})
    assert r.status_code == 200
    body = r.json()
    assert body["n_affected"] > 0
```

(Add the missing fixtures `loaded_scene_with_seg` and `loaded_scene_with_preseg_and_merge` to `conftest.py`.)

- [ ] **Step 2: Run tests, expect failure**

Run: `.venv/bin/pytest backend/tests/test_segment_endpoints.py -v -k "segment_state_full or hide_unhide or snap_to_preseg"`
Expected: failing — endpoints missing.

- [ ] **Step 3: Add Pydantic models + endpoints to `main.py`**

```python
class SegmentStateResponse(BaseModel):
    has_seg: bool
    n_points: Optional[int] = None
    preseg_run_id: Optional[str] = None
    preseg_fingerprint: Optional[str] = None
    source_fingerprint: Optional[str] = None
    hidden_inst_ids: list[int] = []
    is_from_prelabel: bool = False
    dirty: bool = False


class HideRequest(BaseModel):
    inst_id: int


class SnapToPresegRequest(BaseModel):
    inst_ids: list[int]


@app.get("/api/segment/state", response_model=SegmentStateResponse)
def segment_state_get():
    seg = _state.get("seg")
    if seg is None:
        return SegmentStateResponse(has_seg=False)
    return SegmentStateResponse(
        has_seg=True,
        n_points=int(len(seg.instance_ids)),
        preseg_run_id=seg.preseg_run_id,
        preseg_fingerprint=seg.preseg_fingerprint,
        source_fingerprint=seg.source_fingerprint,
        hidden_inst_ids=sorted(int(x) for x in seg.hidden_inst_ids),
        is_from_prelabel=bool(seg.is_from_prelabel),
        dirty=bool(seg.dirty),
    )


@app.post("/api/segment/hide", response_model=SegmentStateResponse)
def segment_hide(req: HideRequest):
    seg = _state.get("seg")
    if seg is None:
        raise HTTPException(409, "no active segment session")
    seg.hide_instance(req.inst_id)
    return segment_state_get()


@app.delete("/api/segment/hide/{inst_id}", response_model=SegmentStateResponse)
def segment_unhide(inst_id: int):
    seg = _state.get("seg")
    if seg is None:
        raise HTTPException(409, "no active segment session")
    seg.unhide_instance(inst_id)
    return segment_state_get()


@app.post("/api/segment/snap-to-preseg")
def segment_snap_to_preseg(req: SnapToPresegRequest):
    seg = _state.get("seg")
    if seg is None:
        raise HTTPException(409, "no active segment session")
    out = seg.snap_to_preseg(req.inst_ids)
    return {"n_affected": int(out["n_affected"])}
```

(Delete the placeholder added in Task 7.)

- [ ] **Step 4: Run tests, expect PASS**

Run: `.venv/bin/pytest backend/tests/test_segment_endpoints.py -v`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add backend/main.py backend/tests/test_segment_endpoints.py backend/tests/conftest.py
git commit -m "feat(api): /api/segment/state + hide/unhide + snap-to-preseg endpoints"
```

---

## Task 9: Route preseg apply + run-load through `freeze_preseg`

Stop overwriting `instance_ids` directly; freeze the preseg layer and apply seeding as an undoable `_apply`.

**Files:**
- Modify: `backend/main.py`
- Test: `backend/tests/test_segment_presegment_endpoint.py`

- [ ] **Step 1: Write failing test**

```python
def test_preseg_freezes_layer_and_seeds_instances(client, loaded_scene):
    r = client.post("/api/segment/presegment", json={"resolution": 0.1})
    assert r.status_code == 200
    state = client.get("/api/segment/state").json()
    assert state["has_seg"] is True
    assert state["preseg_fingerprint"] is not None
    assert state["preseg_fingerprint"].startswith("sha256:")


def test_reload_preseg_run_stamps_run_id(client, loaded_scene_with_preseg_run):
    run_id = loaded_scene_with_preseg_run  # fixture-provided run id
    r = client.post(f"/api/segment/presegment/runs/{run_id}/load")
    assert r.status_code == 200
    state = client.get("/api/segment/state").json()
    assert state["preseg_run_id"] == run_id
```

- [ ] **Step 2: Run tests, expect failure**

Run: `.venv/bin/pytest backend/tests/test_segment_presegment_endpoint.py -v -k "freezes_layer or stamps_run_id"`
Expected: failing — `preseg_fingerprint` is None.

- [ ] **Step 3: Modify `segment_presegment` and `segment_presegment_load_run` in `main.py`**

In `segment_presegment`, after computing `sub_inst` (the full-length int32 instance_ids array from the preseg run):

```python
seg = _state["seg"]
# Freeze the preseg layer first (immutable, content-fingerprinted).
seg.freeze_preseg(sub_inst.copy(), run_id=None)

# Then seed instance/class arrays via the normal _apply path (undoable),
# only on points currently unlabeled when preserve_labeled=True.
unlabeled = (seg.class_ids < 0) if req.preserve_labeled else np.ones_like(seg.class_ids, dtype=bool)
# Group by (instance, class) and call apply_reassign per group so each chunk
# is one undo-able delta with consistent target class.
# (Reuse the existing summary list to map seg id → class id.)
```

For `segment_presegment_load_run`, set `run_id=run_id` in the `freeze_preseg` call.

(The existing code path that builds `class_ids` + writes `instance_ids` directly is replaced by `freeze_preseg(...)` + one or more `apply_reassign` calls grouped per class. Keep the response shape unchanged.)

- [ ] **Step 4: Run tests, expect PASS**

Run: `.venv/bin/pytest backend/tests/test_segment_presegment_endpoint.py -v`
Expected: all green.

- [ ] **Step 5: Run full backend suite — no regressions**

Run: `.venv/bin/pytest backend/tests -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add backend/main.py backend/tests/test_segment_presegment_endpoint.py
git commit -m "feat(preseg): freeze immutable preseg layer + undoable seed via _apply"
```

---

## Task 10: Save path — promote `working_*` to `labels/gt_*` with fingerprints

Explicit Save (Ctrl+S) is handled by the existing labels-write endpoint. Patch it to flush autosave first, then add the new fingerprint fields to `gt_segment_metadata.json`.

**Files:**
- Modify: `backend/segment_io.py` (the existing `write_labels` / metadata writer)
- Modify: `backend/main.py` (Save endpoint)
- Test: `backend/tests/test_segment_io.py`

- [ ] **Step 1: Write failing fingerprint-metadata test**

```python
def test_write_labels_adds_fingerprints(tmp_path):
    """gt_segment_metadata.json must carry prelabel_fingerprint + source_fingerprint
    when supplied by the caller."""
    from segment_io import write_labels  # existing function
    scan = tmp_path
    (scan / "labels").mkdir()
    class_ids = np.zeros(10, dtype=np.int32)
    inst_ids = np.zeros(10, dtype=np.int32)
    write_labels(
        scan, class_ids=class_ids, instance_ids=inst_ids,
        prelabel_fingerprint="sha256:abc",
        source_fingerprint="sha256:def",
    )
    import json
    meta = json.loads((scan / "labels" / "gt_segment_metadata.json").read_text())
    assert meta["prelabel_fingerprint"] == "sha256:abc"
    assert meta["source_fingerprint"] == "sha256:def"
```

- [ ] **Step 2: Run test, expect failure**

Run: `.venv/bin/pytest backend/tests/test_segment_io.py -v -k "fingerprints"`
Expected: failing — `write_labels` doesn't accept those kwargs.

- [ ] **Step 3: Extend `write_labels` signature**

In `segment_io.py`, add optional `prelabel_fingerprint` / `source_fingerprint` kwargs to `write_labels` and inject them into the metadata dict.

- [ ] **Step 4: Wire into the Save endpoint**

In `main.py`'s save handler, before writing, call `seg.flush_autosave()` to drain debounce, then pass `seg.preseg_fingerprint` (as `prelabel_fingerprint`) and `seg.source_fingerprint` into `write_labels`.

- [ ] **Step 5: Run tests, expect PASS**

Run: `.venv/bin/pytest backend/tests -v`
Expected: all green.

- [ ] **Step 6: Commit**

```bash
git add backend/segment_io.py backend/main.py backend/tests/test_segment_io.py
git commit -m "feat(save): record prelabel+source fingerprints in gt_segment_metadata"
```

---

## Task 11: Frontend — API wrappers + hydration

**Files:**
- Modify: `frontend/src/api.js`
- Modify: `frontend/src/segment-state.js`
- Modify: `frontend/src/App.jsx` (call hydrate on scene load)
- Test: `frontend/src/segment-state.test.js`

- [ ] **Step 1: Write failing test for hydration**

Append to `frontend/src/segment-state.test.js`:

```js
import { hydrateFromServerState } from './segment-state.js';

test('hydrateFromServerState pulls hidden + preseg fields', () => {
  const state = makeEmptyState(100);
  const out = hydrateFromServerState(state, {
    has_seg: true, n_points: 100,
    preseg_run_id: 'r1', preseg_fingerprint: 'sha256:x',
    source_fingerprint: 'sha256:y',
    hidden_inst_ids: [3, 7],
    is_from_prelabel: false, dirty: true,
  });
  expect(out.hiddenInstIds).toEqual(new Set([3, 7]));
  expect(out.presegRunId).toBe('r1');
  expect(out.dirty).toBe(true);
});
```

- [ ] **Step 2: Run test, expect failure**

Run: `npx vitest run --root frontend src/segment-state.test.js`
Expected: failing — `hydrateFromServerState` missing.

- [ ] **Step 3: Implement `hydrateFromServerState` + API wrappers**

In `frontend/src/api.js`:

```js
export async function getSegmentState() {
  const r = await fetch('/api/segment/state');
  if (!r.ok) throw new Error(`segment/state ${r.status}`);
  return r.json();
}
export async function hideInstance(instId) {
  const r = await fetch('/api/segment/hide', {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({inst_id: instId}),
  });
  if (!r.ok) throw new Error(`hide ${r.status}`);
  return r.json();
}
export async function unhideInstance(instId) {
  const r = await fetch(`/api/segment/hide/${instId}`, {method: 'DELETE'});
  if (!r.ok) throw new Error(`unhide ${r.status}`);
  return r.json();
}
export async function snapToPreseg(instIds) {
  const r = await fetch('/api/segment/snap-to-preseg', {
    method: 'POST',
    headers: {'content-type': 'application/json'},
    body: JSON.stringify({inst_ids: instIds}),
  });
  if (!r.ok) throw new Error(`snap ${r.status}`);
  return r.json();
}
```

In `frontend/src/segment-state.js`:

```js
export function hydrateFromServerState(state, payload) {
  if (!payload || !payload.has_seg) return state;
  return {
    ...state,
    hiddenInstIds: new Set(payload.hidden_inst_ids || []),
    presegRunId: payload.preseg_run_id ?? null,
    presegFingerprint: payload.preseg_fingerprint ?? null,
    sourceFingerprint: payload.source_fingerprint ?? null,
    dirty: !!payload.dirty,
  };
}
```

In `App.jsx`, after every `/api/load`:

```js
const serverState = await getSegmentState();
setSegState((s) => hydrateFromServerState(s, serverState));
```

- [ ] **Step 4: Run test, expect PASS**

Run: `npx vitest run --root frontend src/segment-state.test.js`
Expected: all green.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/api.js frontend/src/segment-state.js frontend/src/App.jsx frontend/src/segment-state.test.js
git commit -m "feat(fe): hydrate segState from /api/segment/state on scene load"
```

---

## Task 12: Frontend — route hide through API + drop local cache

**Files:**
- Modify: `frontend/src/segment-tools.jsx`
- Modify: `frontend/src/segment-state.js`

- [ ] **Step 1: Locate the existing hide UI**

```bash
grep -n "hide\|hidden" frontend/src/segment-tools.jsx
```

- [ ] **Step 2: Replace local hide mutation with API call**

In every handler that toggles a segment's visibility, call `hideInstance(id)` / `unhideInstance(id)`, then `setSegState((s) => hydrateFromServerState(s, response))`. Remove any local hidden-set ownership that previously lived outside `segState`.

- [ ] **Step 3: Manual smoke**

Start dev server:
```bash
npm run dev
```
Load any scene, preseg it, brush a few points, hide a segment, reload the page in the browser. Expected: hide persists.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/segment-tools.jsx frontend/src/segment-state.js
git commit -m "feat(fe): hide/unhide routed through API; drop local hiddenSet cache"
```

---

## Task 13: SCHEMA v1.1 patch (in `engine/data` repo)

The dataset schema patch is the last logical step. It's an additive doc-only change in a separate repository.

**Files:**
- Modify: `engine/data/lidar/SCHEMA.md`

- [ ] **Step 1: Open `~/coding/engine/data/lidar/SCHEMA.md`**

- [ ] **Step 2: In the "Directory layout" code block, add the `session/` subtree**

Add after the `prelabel/` section, before `annotation_history/`:

```
├── session/                         (optional) live editor state, not part of GT
│   ├── current.json                 commit pointer; references the working files
│   ├── working_class_ids.npy        (int8, optional) in-progress class array
│   ├── working_segment_ids.npy      (int32, optional) in-progress segment array
│   ├── preseg_runs/<run_id>.npz     timestamped preseg snapshots
│   └── graph_review.json            (reserved; not yet emitted)
```

- [ ] **Step 3: Add a "Session state (optional)" section after "Annotation status convention"**

Document:
- `session/` is outside the GT contract (invariants do not apply).
- `current.json` is the commit pointer: `working_*.npy` are honored only when `current.json` references them. Schema for `current.json` (paste from spec §3).
- `preseg_runs/<run_id>.npz` carries `instance_ids: int32[N]`, `summary: json`, `meta: json`.

- [ ] **Step 4: Extend `labels/gt_segment_metadata.json` spec**

Add the two optional fields with a one-line description each:
- `prelabel_fingerprint`: sha256 of the prelabel `ransac_instance_ids.npy` these GT labels were seeded from.
- `source_fingerprint`: sha256 of `scan.ply`'s xyz at label time.

- [ ] **Step 5: Bump heading from "Labeled Lidar Scan Schema (v1)" to "(v1.1)"**

Add a brief changelog at the bottom: "v1.1 (2026-05-13) — additive: optional `session/` subdir, `prelabel_fingerprint` + `source_fingerprint` in metadata."

- [ ] **Step 6: Commit in the engine repo**

```bash
cd ~/coding/engine/data
git add lidar/SCHEMA.md
git commit -m "docs(schema): v1.1 — optional session/ subdir + fingerprint fields"
```

---

## Final Verification

- [ ] Backend full suite green: `.venv/bin/pytest backend/tests -v`
- [ ] Frontend full suite green: `npm test`
- [ ] Manual smoke in dev:
  1. `npm run dev`
  2. Load a SCHEMA scene → preseg → brush some points → hide a segment.
  3. Reload the browser tab. Expected: hide persists, brush persists.
  4. Kill `npm run dev`, restart it, reload the scene. Expected: hide persists, brush persists.
  5. Re-run preseg. Expected: labels survive, fingerprint warning logged (FE surfacing UI can be a follow-up).
- [ ] Memory check: open the largest annotated scene (~3M pts); confirm RSS overhead within +20 MB of pre-change baseline.
- [ ] Push the worktree branch when satisfied.

---

## Out of scope (explicit non-goals — do not implement)

- Cross-scene ID stability
- Mesh-face ↔ point linkage
- Connectivity graph mode (path reserved only)
- Versioned `.v1/.v2` label files
- FE "stale prelabel" surfacing UI (log-only is fine for first pass)
- `data/preseg_runs/<scene>/` migration into `session/preseg_runs/` — keep both paths working; SCHEMA scenes can use the new location going forward, legacy paths read-only

## References
- Spec: `docs/superpowers/specs/2026-05-13-segment-linkage-design.md`
- Prior art: `../industrial-point-labeler/docs/design.md`
- Dataset schema: `~/coding/engine/data/lidar/SCHEMA.md`
