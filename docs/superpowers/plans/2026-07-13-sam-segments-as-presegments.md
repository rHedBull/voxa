# SAM segments as a presegment-like candidate layer — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Accepted SAM masks stop becoming classified pointset instances immediately at `/api/sam/project`. Instead they accumulate as selectable/multi-selectable *candidate* segments — a new `sam_ids` layer parallel to (never merged into) `instance_ids`/`class_ids` — shown in their own bottom-left list and viewport recolor while the SAM tool is active, and only become real labeled instances when the user classifies a selection via the existing Ctrl+Enter/hotkey pipeline.

**Architecture:** Backend: `SegmentSession` gets a new `sam_ids`/`sam_segments` layer (mirrors the existing immutable `preseg_ids` layer, but mutable and session-persistent) and a `materialize_sam_segment()` op that never touches `instance_ids`/`class_ids` (required by SCHEMA invariant 3). `/api/sam/project` calls it instead of `apply_reassign`. A new `_retire_sam_ids()` helper, called from both `materialize_sam_segment` and the existing `_apply()`, keeps the candidate layer consistent (last-materialize-wins; any tool labeling a point retires its SAM candidacy). Persistence: `working_sam_ids.npy` (via the existing `save_session_aux`/`load_working_arrays` machinery) + a new `sam_segments.json` metadata file. Frontend: `segState` gets a parallel `samIds`/`samSegments`/`samSelection`, a new `SamSegmentList` panel (sibling of `PresegmentList`, never mixed with it), a `confirmSamSelection` function (sibling of `confirmSegmentSelection`), and three call-site edits in `mode-label.jsx`'s existing Ctrl+Enter/hotkey/class-picker machinery to route a SAM selection through it.

**Tech Stack:** FastAPI + Pydantic (backend), numpy, React 18 + Three.js (frontend), pytest, vitest.

**Spec:** `docs/superpowers/specs/2026-07-13-sam-segments-as-presegments-design.md` (read first — this plan implements it verbatim; file:line references below are to the current `main` branch).

---

## Before you start

Run the full test suite once to confirm a clean baseline:

```bash
npm test
```

Expected: all backend (pytest) and frontend (vitest) tests pass. If anything is already failing, stop and resolve that first — this plan's tests assume a green baseline.

---

## Task 1: `SegmentSession.sam_ids` / `sam_segments` state + `materialize_sam_segment`

**Files:**
- Modify: `backend/labeling/segment_state.py`
- Test: `backend/tests/test_segment_state.py`

This is the core data-layer change: a new full-res `sam_ids` array (default `-1`) and a `sam_segments: dict[int, dict]` summary, both separate from `instance_ids`/`class_ids` so SCHEMA invariant 3 (`class == -1 ⟺ instance == -1`, `backend/labeling/seg_inference.py:219`) is never at risk. Not on the undo stack (mirrors `preseg_ids`, `segment_state.py:157-160`).

- [ ] **Step 1: Write failing tests for `materialize_sam_segment`**

Append to `backend/tests/test_segment_state.py`:

```python
def test_materialize_sam_segment_allocates_fresh_id_and_writes_sam_ids():
    s = _seed()
    out = s.materialize_sam_segment(np.array([0, 6], dtype=np.int32))
    assert out["sam_seg_id"] == 0
    assert out["n_affected"] == 2
    assert out["n_protected"] == 0
    assert int(s.sam_ids[0]) == 0 and int(s.sam_ids[6]) == 0
    # instance_ids/class_ids are untouched — this is the whole point.
    assert int(s.instance_ids[0]) == -1 and int(s.class_ids[0]) == -1
    assert s.sam_segments[0]["n_points"] == 2
    assert s.dirty is False  # not a working-array edit


def test_materialize_sam_segment_ids_increment():
    s = _seed()
    a = s.materialize_sam_segment(np.array([0], dtype=np.int32))
    b = s.materialize_sam_segment(np.array([6], dtype=np.int32))
    assert a["sam_seg_id"] == 0 and b["sam_seg_id"] == 1


def test_materialize_sam_segment_respects_protect_instances():
    s = _seed()
    # Point 3 already belongs to instance 1 (see _seed()); protect it.
    out = s.materialize_sam_segment(np.array([0, 3], dtype=np.int32),
                                     protect_instances=[1])
    assert out["n_affected"] == 1
    assert out["n_protected"] == 1
    assert int(s.sam_ids[3]) == -1   # protected point never got a sam id
    assert int(s.sam_ids[0]) == 0


def test_materialize_sam_segment_all_protected_creates_nothing():
    s = _seed()
    out = s.materialize_sam_segment(np.array([3, 4], dtype=np.int32),
                                     protect_instances=[1])
    assert out["sam_seg_id"] is None
    assert out["n_affected"] == 0
    assert out["n_protected"] == 2
    assert len(s.sam_segments) == 0


def test_materialize_sam_segment_overlap_is_last_write_wins():
    s = _seed()
    a = s.materialize_sam_segment(np.array([0, 6], dtype=np.int32))
    b = s.materialize_sam_segment(np.array([6], dtype=np.int32))  # overlaps a
    assert int(s.sam_ids[6]) == b["sam_seg_id"]
    assert int(s.sam_ids[0]) == a["sam_seg_id"]
    # a's summary shrank from 2 to 1 (point 6 moved to b), not deleted.
    assert s.sam_segments[a["sam_seg_id"]]["n_points"] == 1
    assert s.sam_segments[b["sam_seg_id"]]["n_points"] == 1


def test_materialize_sam_segment_full_overlap_drops_old_summary_entry():
    s = _seed()
    a = s.materialize_sam_segment(np.array([6], dtype=np.int32))
    s.materialize_sam_segment(np.array([6], dtype=np.int32))  # fully re-covers a
    assert a["sam_seg_id"] not in s.sam_segments


def test_apply_reassign_retires_overlapping_sam_ids():
    """Any tool labeling a point (apply_reassign, set_class, merge) must
    retire that point's SAM candidacy — it's no longer up for grabs."""
    s = _seed()
    out = s.materialize_sam_segment(np.array([0, 6], dtype=np.int32))
    sam_id = out["sam_seg_id"]
    s.apply_reassign(np.array([6], dtype=np.int32), target_inst=-1, target_class=0)
    assert int(s.sam_ids[6]) == -1
    assert int(s.sam_ids[0]) == sam_id       # untouched point keeps its candidacy
    assert s.sam_segments[sam_id]["n_points"] == 1


def test_materialize_sam_segment_not_on_undo_stack():
    s = _seed()
    s.materialize_sam_segment(np.array([0], dtype=np.int32))
    assert s.undo() is None  # nothing to undo — matches preseg_ids' non-edit status
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest backend/tests/test_segment_state.py -k materialize_sam -v
.venv/bin/pytest backend/tests/test_segment_state.py -k retires_overlapping -v
```

Expected: `AttributeError: 'SegmentSession' object has no attribute 'materialize_sam_segment'`.

- [ ] **Step 3: Implement `sam_ids`/`sam_segments` state**

In `backend/labeling/segment_state.py`, add to `SegmentSession.__init__` (after the existing `self.preseg_ids` line, `segment_state.py:59`):

```python
        self.sam_ids: np.ndarray = np.full(n, -1, dtype=np.int32)
        # {sam_seg_id: {"n_points": int, "mask_score": float|None, "created_at": str}}
        # — a mutable, session-scoped candidate layer (unlike preseg_ids, which
        # is immutable and pinned at session creation). Never written into
        # instance_ids/class_ids: a point can carry a sam candidate id with no
        # class, which would violate SCHEMA invariant 3 if it were instance_ids.
        self.sam_segments: dict[int, dict] = {}
        self._next_sam_id: int = 0
```

- [ ] **Step 4: Implement `_retire_sam_ids` + `materialize_sam_segment`**

Add these methods to `SegmentSession`, right after `apply_reassign` (`segment_state.py:132`, before `def undo`):

```python
    def _retire_sam_ids(self, indices: np.ndarray) -> None:
        """Clear SAM candidacy for these points — they're about to carry a
        real label (or a fresher SAM candidate id). Shrinks/drops the
        affected sam_segments summary entries so counts never go stale."""
        old = self.sam_ids[indices]
        live = old[old >= 0]
        if live.size == 0:
            return
        ids, counts = np.unique(live, return_counts=True)
        for sid, cnt in zip(ids.tolist(), counts.tolist()):
            entry = self.sam_segments.get(int(sid))
            if entry is None:
                continue
            entry["n_points"] -= int(cnt)
            if entry["n_points"] <= 0:
                del self.sam_segments[int(sid)]
        self.sam_ids[indices] = -1

    def materialize_sam_segment(
        self, indices: np.ndarray,
        protect_instances: Optional[list[int]] = None,
        mask_score: Optional[float] = None,
    ) -> dict:
        """Add an accepted SAM mask as a candidate in the sam_ids layer — a
        selection aid, not an edit: never touches instance_ids/class_ids, not
        on the undo stack (mirrors preseg_ids). protect_instances mirrors
        apply_reassign's confirmed-lock rule (dropped BEFORE allocating an
        id, same as a locked box/tube can't steal already-labeled points).
        Overlapping candidates are last-materialize-wins via _retire_sam_ids.
        """
        indices = np.asarray(indices, dtype=np.int32)
        n_candidate = int(indices.size)
        if protect_instances:
            protect = {int(v) for v in protect_instances}
            protect.discard(-1)
            if protect:
                keep = ~np.isin(self.instance_ids[indices], list(protect))
                indices = indices[keep]
        n_protected = n_candidate - int(indices.size)
        if indices.size == 0:
            return {"op": "materialize_sam_segment", "sam_seg_id": None,
                    "n_affected": 0, "n_protected": n_protected}
        self._retire_sam_ids(indices)
        sam_seg_id = self._next_sam_id
        self._next_sam_id += 1
        self.sam_ids[indices] = np.int32(sam_seg_id)
        from labeling.segment_io import utc_now_iso
        self.sam_segments[sam_seg_id] = {
            "n_points": int(indices.size),
            "mask_score": mask_score,
            "created_at": utc_now_iso(),
        }
        self.schedule_autosave(write_arrays=True)
        return {"op": "materialize_sam_segment", "sam_seg_id": sam_seg_id,
                "indices": indices, "n_affected": int(indices.size),
                "n_protected": n_protected}
```

Note: `materialize_sam_segment` reuses `schedule_autosave(write_arrays=True)` rather than adding a new flag — it will also rewrite `class_ids.npy`/`instance_ids.npy` with unchanged bytes, a small redundant write that's simpler than threading a new autosave flag through the debounce timer (Task 2 makes `sam_ids`/`sam_segments` ride along on every `write_arrays=True` autosave).

- [ ] **Step 5: Wire `_retire_sam_ids` into `_apply()`**

In `_apply()` (`segment_state.py:223`), add one line right after the `indices = indices.astype(...)` line:

```python
    def _apply(self, op: str, indices: np.ndarray, payload: dict) -> dict:
        indices = indices.astype(np.int32, copy=False)
        self._retire_sam_ids(indices)
        before_cls = self.class_ids[indices].copy()
```

- [ ] **Step 6: Run tests to verify pass**

```bash
.venv/bin/pytest backend/tests/test_segment_state.py -v
```

Expected: all pass, including the 8 new tests.

- [ ] **Step 7: Commit**

```bash
git add backend/labeling/segment_state.py backend/tests/test_segment_state.py
git commit -m "feat(sam): add sam_ids candidate layer + materialize_sam_segment to SegmentSession"
```

---

## Task 2: Persistence — `working_sam_ids.npy` + `sam_segments.json`

**Files:**
- Modify: `backend/labeling/segment_io.py`
- Modify: `backend/labeling/segment_state.py` (`_do_autosave`)
- Test: `backend/tests/test_segment_io.py`

- [ ] **Step 1: Write failing tests for the new I/O functions**

Append to `backend/tests/test_segment_io.py` (check the top of the file for existing imports/style first — match them):

```python
def test_save_session_aux_writes_working_sam_ids(tmp_path):
    from labeling.segment_io import save_session_aux
    sam_ids = np.array([-1, 0, 0, -1], dtype=np.int32)
    save_session_aux(tmp_path, {"name": "x"}, sam_ids=sam_ids)
    assert (tmp_path / "working_sam_ids.npy").exists()
    loaded = np.load(tmp_path / "working_sam_ids.npy")
    assert (loaded == sam_ids).all()


def test_save_session_aux_without_sam_ids_does_not_write_file(tmp_path):
    from labeling.segment_io import save_session_aux
    save_session_aux(tmp_path, {"name": "x"})
    assert not (tmp_path / "working_sam_ids.npy").exists()


def test_load_sam_ids_roundtrip(tmp_path):
    from labeling.segment_io import save_session_aux, load_sam_ids
    sam_ids = np.array([-1, 0, 0, -1], dtype=np.int32)
    save_session_aux(tmp_path, {"name": "x"}, sam_ids=sam_ids)
    loaded = load_sam_ids(tmp_path, n_points=4)
    assert loaded is not None
    assert (loaded == sam_ids).all()


def test_load_sam_ids_absent_file_returns_none(tmp_path):
    from labeling.segment_io import load_sam_ids
    assert load_sam_ids(tmp_path, n_points=4) is None


def test_load_sam_ids_shape_mismatch_raises(tmp_path):
    from labeling.segment_io import save_session_aux, load_sam_ids
    save_session_aux(tmp_path, {"name": "x"},
                     sam_ids=np.array([-1, 0], dtype=np.int32))
    with pytest.raises(ValueError):
        load_sam_ids(tmp_path, n_points=99)


def test_save_and_load_sam_segments_roundtrip(tmp_path):
    from labeling.segment_io import save_sam_segments, load_sam_segments
    segs = {0: {"n_points": 5, "mask_score": 0.9, "created_at": "2026-07-13T00:00:00+00:00"},
            2: {"n_points": 3, "mask_score": None, "created_at": "2026-07-13T00:00:01+00:00"}}
    save_sam_segments(tmp_path, segs)
    assert (tmp_path / "sam_segments.json").exists()
    loaded = load_sam_segments(tmp_path)
    assert loaded == segs


def test_load_sam_segments_absent_file_returns_empty_dict(tmp_path):
    from labeling.segment_io import load_sam_segments
    assert load_sam_segments(tmp_path) == {}
```

Check the file's top for whether `numpy as np` / `pytest` are already imported; add if missing.

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest backend/tests/test_segment_io.py -k sam -v
```

Expected: `TypeError` (unexpected `sam_ids` kwarg) / `ImportError` for the new functions.

- [ ] **Step 3: Implement in `segment_io.py`**

Extend `save_session_aux` (`segment_io.py:193-219`) — add a `sam_ids` param:

```python
def save_session_aux(
    session_dir: Path,
    aux: dict,
    *,
    class_ids: Optional[np.ndarray] = None,
    instance_ids: Optional[np.ndarray] = None,
    sam_ids: Optional[np.ndarray] = None,
) -> dict:
    """Atomically persist editor session state. Returns the payload as
    written (callers use its ``saved_at`` stamp).

    Order: working_*.npy first, then session.json (commit pointer). On a
    crash between the npy renames and session.json rename, the next reload
    sees the previous-consistent session.json and ignores any half-updated
    working_*.
    """
    session_dir.mkdir(parents=True, exist_ok=True)
    if class_ids is not None:
        atomic_write_npy(session_dir / "working_class_ids.npy",
                         class_ids.astype(np.int8, copy=False))
    if instance_ids is not None:
        atomic_write_npy(session_dir / "working_segment_ids.npy",
                         instance_ids.astype(np.int32, copy=False))
    if sam_ids is not None:
        atomic_write_npy(session_dir / "working_sam_ids.npy",
                         sam_ids.astype(np.int32, copy=False))
    payload = dict(aux)
    payload.setdefault("schema_version", SESSION_SCHEMA_VERSION)
    payload["saved_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    atomic_write_json(session_dir / "session.json", payload)
    return payload
```

Add `load_sam_ids` + `save_sam_segments`/`load_sam_segments` at the end of the file:

```python
def load_sam_ids(session_dir: Path, n_points: int) -> Optional[np.ndarray]:
    """Return the SAM candidate layer (int32) if working_sam_ids.npy exists,
    or None if absent (a session with no SAM captures yet — caller defaults
    to all -1). A present-but-wrong-shape file fails loudly (stale/foreign
    data), matching load_working_arrays' posture for a corrupt file."""
    p = session_dir / "working_sam_ids.npy"
    if not p.exists():
        return None
    arr = np.load(p).astype(np.int32, copy=False)
    if arr.shape != (n_points,):
        raise ValueError(f"working_sam_ids.npy shape {arr.shape} != ({n_points},)")
    return arr


def save_sam_segments(session_dir: Path, sam_segments: dict[int, dict]) -> None:
    """Atomically persist the SAM candidate-segment summary
    (sessions/<id>/sam_segments.json). Point membership lives in
    working_sam_ids.npy; this file is metadata only, mirroring prelabel's
    segment_summary.json shape."""
    session_dir.mkdir(parents=True, exist_ok=True)
    payload = {"segments": [{"id": sid, **meta}
                            for sid, meta in sorted(sam_segments.items())]}
    atomic_write_json(session_dir / "sam_segments.json", payload)


def load_sam_segments(session_dir: Path) -> dict[int, dict]:
    """Read sam_segments.json -> {sam_seg_id: {n_points, mask_score,
    created_at}}. Missing file -> empty dict (no SAM captures yet)."""
    p = session_dir / "sam_segments.json"
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return {}
    return {int(e["id"]): {k: v for k, v in e.items() if k != "id"}
            for e in raw.get("segments", [])}
```

- [ ] **Step 4: Wire into `SegmentSession._do_autosave`**

In `segment_state.py:318-328`, update `_do_autosave`:

```python
    def _do_autosave(self, write_arrays: bool) -> None:
        if self.session_dir is None:
            return
        from labeling.segment_io import save_session_aux, save_sam_segments
        with self._autosave_lock:
            save_session_aux(
                self.session_dir,
                self._aux_payload(),
                class_ids=self.class_ids if write_arrays else None,
                instance_ids=self.instance_ids if write_arrays else None,
                sam_ids=self.sam_ids if write_arrays else None,
            )
            if write_arrays:
                save_sam_segments(self.session_dir, self.sam_segments)
```

- [ ] **Step 5: Run tests to verify pass**

```bash
.venv/bin/pytest backend/tests/test_segment_io.py -v
.venv/bin/pytest backend/tests/test_segment_state.py -v
```

- [ ] **Step 6: Commit**

```bash
git add backend/labeling/segment_io.py backend/labeling/segment_state.py backend/tests/test_segment_io.py
git commit -m "feat(sam): persist sam_ids/sam_segments alongside session autosave"
```

---

## Task 3: Resume a session with SAM candidates

**Files:**
- Modify: `backend/app/core.py` (`_resume_session`, `core.py:219-243`)
- Test: `backend/tests/test_segment_endpoints.py`

- [ ] **Step 1: Write a failing end-to-end resume test**

Append to `backend/tests/test_segment_endpoints.py`:

```python
def test_resume_session_restores_sam_candidates(
    client_with_loaded_annotated_scene, scan_dir_for_loaded_scene,
):
    import main
    from labeling.segment_io import save_session_aux, save_sam_segments
    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    n = len(seg.instance_ids)
    sam_ids = np.full(n, -1, dtype=np.int32)
    sam_ids[0] = 7
    sam_ids[1] = 7
    save_session_aux(seg.session_dir, seg._aux_payload(), sam_ids=sam_ids)
    save_sam_segments(seg.session_dir, {7: {"n_points": 2, "mask_score": 0.8,
                                            "created_at": "2026-07-13T00:00:00+00:00"}})

    r = client.get("/api/segment/state")
    assert r.status_code == 200
    # Force a resume by reloading the same scene fresh (simulates a page reload).
    main._state.update(seg=None, scene=None, session_id=None, source_fp=None)
    r2 = client.post("/api/load", json={"name": "annotated/demo", "max_points": 100})
    assert r2.status_code == 200
    seg2 = main._state["seg"]
    assert int(seg2.sam_ids[0]) == 7 and int(seg2.sam_ids[1]) == 7
    assert seg2.sam_segments[7]["n_points"] == 2
    assert seg2._next_sam_id == 8
```

(`np` must already be imported at the top of the test file — check before adding.)

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest backend/tests/test_segment_endpoints.py -k resume_session_restores_sam -v
```

Expected: `AssertionError` — `seg2.sam_ids[0]` is `-1` (not hydrated yet).

- [ ] **Step 3: Implement in `_resume_session`**

In `backend/app/core.py`, update `_resume_session` (`core.py:219-243`):

```python
def _resume_session(lay: ScanLayout, session_id: str, pc, source_fp: str):
    """..."""
    from labeling.segment_state import SegmentSession
    from labeling.segment_io import load_working_arrays, load_sam_ids, load_sam_segments
    from labeling.session_store import verify_pins
    from preseg.preseg_store import load_preseg

    aux = verify_pins(lay, session_id, source_fp=source_fp)
    sp = lay.session(session_id)
    wa = load_working_arrays(sp.dir, n_points=len(pc))
    if wa is None:
        raise HTTPException(409, f"session {session_id}: working arrays "
                                 f"missing or wrong shape for this cloud")
    seg = SegmentSession.from_aux(aux, class_ids=wa[0], instance_ids=wa[1],
                                  positions=pc.points, session_dir=sp.dir)
    seg.source_fingerprint = source_fp
    sam_ids = load_sam_ids(sp.dir, n_points=len(pc))
    if sam_ids is not None:
        seg.sam_ids = sam_ids
        seg.sam_segments = load_sam_segments(sp.dir)
        seg._next_sam_id = (max(seg.sam_segments.keys()) + 1) if seg.sam_segments else 0
    if seg.preseg_id is not None:
        _, pre_ii = load_preseg(lay, seg.preseg_id, n_points=len(pc))
        seg.preseg_ids = pre_ii          # immutable preseg layer for snap-to
    return seg
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/pytest backend/tests/test_segment_endpoints.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backend/app/core.py backend/tests/test_segment_endpoints.py
git commit -m "feat(sam): hydrate sam_ids/sam_segments on session resume"
```

---

## Task 4: Schema changes — `SamProjectRequest`, `SegmentStateResponse`

**Files:**
- Modify: `backend/app/schemas.py`
- Test: `backend/tests/test_segment_endpoints.py`

- [ ] **Step 1: Write a failing test for the new `SegmentStateResponse` fields**

Append to `backend/tests/test_segment_endpoints.py`:

```python
def test_segment_state_includes_full_sam_ids_and_sam_segments(
    client_with_loaded_annotated_scene,
):
    client = client_with_loaded_annotated_scene
    r = client.get("/api/segment/state")
    body = r.json()
    assert "full_sam_ids" in body
    assert "sam_segments" in body
    assert body["sam_segments"] == []  # nothing materialized yet
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest backend/tests/test_segment_endpoints.py -k full_sam_ids -v
```

Expected: `KeyError`/`AssertionError` — field absent.

- [ ] **Step 3: Update `schemas.py`**

Remove `target_class` from `SamProjectRequest` (`schemas.py:179-183`):

```python
class SamProjectRequest(BaseModel):
    capture_id: str
    mask_ids: list[int]
    protect_instances: list[int] = []
```

Add two fields to `SegmentStateResponse` (`schemas.py:185-208`), after `hull_face_seg`:

```python
    hull_face_seg: str = ""
    full_sam_ids: str = ""             # b64 Int32, full-res — SAM candidate layer
    sam_segments: list[dict] = []      # [{id, n_points, mask_score, created_at}]
    session_id: Optional[str] = None
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/pytest backend/tests/test_segment_endpoints.py -v
```

Expected: passes already — the new fields default to `""`/`[]` via the Pydantic model itself, so `"full_sam_ids" in body`/`"sam_segments" in body` are true with no route change yet. Task 5 adds a stronger test that actually forces the route to populate them from live session state.

- [ ] **Step 5: Commit**

```bash
git add backend/app/schemas.py backend/tests/test_segment_endpoints.py
git commit -m "feat(sam): remove target_class from SamProjectRequest, add sam fields to SegmentStateResponse"
```

---

## Task 5: Populate the new fields in `GET /api/segment/state`

**Files:**
- Modify: `backend/routes/segment.py` (`segment_state()`, `segment.py:54-87`)

- [ ] **Step 1: Run the Task 4 test to confirm it now fails for the right reason**

```bash
.venv/bin/pytest backend/tests/test_segment_endpoints.py -k full_sam_ids -v
```

Expected: passes already for `"full_sam_ids" in body` (Pydantic defaults it to `""`), but add a materialized-candidate test to force real wiring:

```python
def test_segment_state_reflects_materialized_sam_segments(
    client_with_loaded_annotated_scene,
):
    import main
    client = client_with_loaded_annotated_scene
    seg = main._state["seg"]
    seg.materialize_sam_segment(np.array([0, 1], dtype=np.int32), mask_score=0.7)
    r = client.get("/api/segment/state")
    body = r.json()
    sam_ids = _b64_to_int32(body["full_sam_ids"])
    assert int(sam_ids[0]) == 0 and int(sam_ids[1]) == 0
    assert body["sam_segments"] == [
        {"id": 0, "n_points": 2, "mask_score": 0.7, "created_at": body["sam_segments"][0]["created_at"]},
    ]
```

(`_b64_to_int32` and `np` should already exist in this test file — check the top; reuse the existing helper used by `test_segment_state_surfaces_full_session_aux` and similar tests.)

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest backend/tests/test_segment_endpoints.py -k reflects_materialized -v
```

Expected: `sam_ids` decodes to all `-1` (field not populated by the route yet).

- [ ] **Step 3: Implement in `segment.py`**

Update `segment_state()` (`segment.py:54-87`):

```python
@router.get("/api/segment/state", response_model=SegmentStateResponse)
def segment_state():
    """Return the active segment session if there is one (no-op otherwise).
    Used by the frontend to hydrate ``segState`` after a page reload."""
    seg = _state.get("seg")
    if seg is None:
        return SegmentStateResponse(has_state=False, has_seg=False, dirty=False)
    instance_ids = seg.instance_ids.astype(np.int32, copy=False)
    class_ids = seg.class_ids.astype(np.int8, copy=False)
    labeled = instance_ids >= 0
    box_ids, box_centers, box_sizes = _compute_segment_boxes(np.asarray(seg.positions), instance_ids)
    from labeling.segment_hulls import compute_hulls as _compute_hulls
    hull_v, hull_f, hull_seg = _compute_hulls(np.asarray(seg.positions), instance_ids)
    sam_segments = [{"id": sid, **meta} for sid, meta in sorted(seg.sam_segments.items())]
    return SegmentStateResponse(
        has_state=True,
        has_seg=True,
        dirty=bool(seg.dirty),
        n_assigned=int(labeled.sum()),
        n_segments=int(np.unique(instance_ids[labeled]).size) if labeled.any() else 0,
        n_points=int(len(seg.instance_ids)),
        preseg_id=seg.preseg_id,
        preseg_fingerprint=seg.preseg_fingerprint,
        source_fingerprint=seg.source_fingerprint,
        is_from_prelabel=bool(seg.is_from_prelabel),
        full_class_ids=_b64(class_ids),
        full_instance_ids=_b64(instance_ids),
        seg_ids=_b64(box_ids),
        seg_centers=_b64(box_centers),
        seg_sizes=_b64(box_sizes),
        hull_vertices=_b64(hull_v),
        hull_faces=_b64(hull_f),
        hull_face_seg=_b64(hull_seg),
        full_sam_ids=_b64(seg.sam_ids.astype(np.int32, copy=False)),
        sam_segments=sam_segments,
        session_id=_state.get("session_id"),
    )
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/pytest backend/tests/test_segment_endpoints.py -v
```

- [ ] **Step 5: Commit**

```bash
git add backend/routes/segment.py backend/tests/test_segment_endpoints.py
git commit -m "feat(sam): populate full_sam_ids/sam_segments in GET /api/segment/state"
```

---

## Task 6: `/api/sam/project` materializes instead of classifying

**Files:**
- Modify: `backend/routes/sam.py`
- Modify: `backend/tests/test_sam_proxy.py`

- [ ] **Step 1: Update the existing tests for the new contract**

In `backend/tests/test_sam_proxy.py`, `_fake_post`'s `/project` branch already returns `{"instances": [...]}` from the *sidecar* (unchanged — the sidecar's own contract doesn't change, only voxa's outward `/api/sam/project` response does). Update the two tests that assert on the old voxa-side response shape:

Replace `test_capture_then_project`:

```python
def test_capture_then_project(client_with_loaded_annotated_scene, monkeypatch):
    client = client_with_loaded_annotated_scene
    monkeypatch.setenv("VOXA_SAM_SIDECAR_URL", "http://side")
    monkeypatch.setattr("routes.sam.httpx.post", _fake_post)
    monkeypatch.setattr("routes.sam._resolve", lambda scene: _Src())
    cam = {"pos": [0,0,0], "target": [0,0,1], "fov": 60, "W": 128, "H": 128}
    r = client.post("/api/sam/capture", json={"camera": cam, "mode": "box", "box": [0.5,0.5,0.4,0.4]})
    assert r.status_code == 200 and r.json()["capture_id"] == "c1"
    r2 = client.post("/api/sam/project",
                     json={"capture_id": "c1", "mask_ids": [0], "protect_instances": []})
    assert r2.status_code == 200
    body = r2.json()
    assert "segments" in body and "instances" not in body
    seg = body["segments"][0]
    assert seg["mask_id"] == 0
    assert seg["sam_seg_id"] == 0
    assert seg["n_affected"] == 3
    assert "scan_indices_b64" in seg
```

Delete `test_bad_class_id_400` entirely (there is no `target_class` to be bad anymore — the request has no such field, so an unknown-class-name 400 can no longer occur at this endpoint).

Add a new test confirming no instance is created:

```python
def test_project_does_not_call_apply_reassign(client_with_loaded_annotated_scene, monkeypatch):
    import main
    client = client_with_loaded_annotated_scene
    monkeypatch.setenv("VOXA_SAM_SIDECAR_URL", "http://side")
    monkeypatch.setattr("routes.sam.httpx.post", _fake_post)
    monkeypatch.setattr("routes.sam._resolve", lambda scene: _Src())
    cam = {"pos": [0,0,0], "target": [0,0,1], "fov": 60, "W": 128, "H": 128}
    client.post("/api/sam/capture", json={"camera": cam, "mode": "box", "box": [0.5,0.5,0.4,0.4]})
    seg = main._state["seg"]
    before = seg.instance_ids.copy()
    client.post("/api/sam/project", json={"capture_id": "c1", "mask_ids": [0], "protect_instances": []})
    assert (seg.instance_ids == before).all()   # untouched — only sam_ids changed
    assert int(seg.sam_ids[0]) >= 0
```

- [ ] **Step 2: Run to verify failure**

```bash
.venv/bin/pytest backend/tests/test_sam_proxy.py -v
```

Expected: `test_capture_then_project` fails (`instances` key present, `segments` absent); `test_project_does_not_call_apply_reassign` fails (`seg.instance_ids` changed).

- [ ] **Step 3: Implement in `routes/sam.py`**

Update the import line (`sam.py:11`) — drop `_serialize_apply`/`_coerce_class_id`, add `_b64`:

```python
from app.core import _state, _require_seg, _resolve, _y_up_to_z_up_xyz, _b64
```

Replace `sam_project` (`sam.py:68-92`):

```python
@router.post("/api/sam/project")
def sam_project(req: SamProjectRequest):
    """Materialize each accepted mask as a SAM candidate segment — NOT a
    classified instance. Classification happens later, from the SAM segment
    list/viewport, through the same apply_reassign path every other tool
    uses (see mode-label.jsx::confirmSamSelection)."""
    base = _sidecar_url()
    seg = _require_seg()
    body = {**_identity(), "capture_id": req.capture_id, "mask_ids": req.mask_ids}
    try:
        r = httpx.post(f"{base}/project", json=body, timeout=_TIMEOUT)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 409:
            raise HTTPException(409, e.response.json().get("detail"))
        raise HTTPException(502, f"SAM sidecar error {e.response.status_code}: {e.response.text}")
    except httpx.HTTPError as e:
        raise HTTPException(502, f"SAM sidecar unreachable: {e}")
    results = []
    for inst in r.json()["instances"]:
        idx = np.frombuffer(base64.b64decode(inst["scan_indices_b64"]), np.int32)
        out = seg.materialize_sam_segment(idx, protect_instances=req.protect_instances)
        entry = {
            "mask_id": inst["mask_id"],
            "sam_seg_id": out["sam_seg_id"],
            "n_affected": out["n_affected"],
            "n_protected": out["n_protected"],
        }
        if out.get("indices") is not None:
            entry["scan_indices_b64"] = _b64(out["indices"].astype(np.int32))
        results.append(entry)
    return {"segments": results}
```

Also update the module docstring at the top of `sam.py:1-4` — it currently says "then applies the returned scan indices through the shared apply_reassign pipeline"; change to:

```python
"""Proxy to the SAM sidecar. /capture forwards the pose (+recenter_offset, scan
identity + resolved cloud paths); /project forwards mask picks, then materializes
the returned scan indices as SAM candidate segments (sam_ids layer) — NOT
classified instances. Classification happens later via the shared
apply_reassign pipeline, from a selection over the materialized candidates."""
```

- [ ] **Step 4: Run tests to verify pass**

```bash
.venv/bin/pytest backend/tests/test_sam_proxy.py -v
.venv/bin/pytest backend/tests -v
```

Expected: full backend suite green.

- [ ] **Step 5: Commit**

```bash
git add backend/routes/sam.py backend/tests/test_sam_proxy.py
git commit -m "feat(sam): /api/sam/project materializes SAM candidates instead of classifying them"
```

---

## Backend checkpoint

Run the full backend suite before moving to the frontend:

```bash
npm run test:backend
```

Expected: all green. This completes the backend half of the feature — `/api/sam/project` no longer creates instances, `GET /api/segment/state` exposes the candidate layer, and it survives session save/resume. The frontend tasks below make this visible/usable in the UI.

---

## Task 7: `segment-state.js` — `samIds`/`samSegments`/`samSelection` + pure functions

**Files:**
- Modify: `frontend/src/segment-state.js`
- Test: `frontend/src/segment-state.test.js`

- [ ] **Step 1: Write failing tests**

Read `frontend/src/segment-state.test.js` first to match its existing style (imports, helper builders), then append:

```js
describe('SAM candidate layer', () => {
  it('initSegState defaults samIds to all -1 and samSegments/samSelection empty', () => {
    const classFull = new Int8Array([-1, -1, -1]);
    const instanceFull = new Int32Array([-1, -1, -1]);
    const s = initSegState({ classFull, instanceFull });
    expect(Array.from(s.samIds)).toEqual([-1, -1, -1]);
    expect(s.samSegments.size).toBe(0);
    expect(s.samSelection.size).toBe(0);
  });

  it('initSegState hydrates samIds/samSegments when provided', () => {
    const classFull = new Int8Array([-1, -1]);
    const instanceFull = new Int32Array([-1, -1]);
    const samIds = new Int32Array([3, 3]);
    const s = initSegState({
      classFull, instanceFull, samIds,
      samSegments: [{ id: 3, n_points: 2, mask_score: 0.5 }],
    });
    expect(Array.from(s.samIds)).toEqual([3, 3]);
    expect(s.samSegments.get(3)).toEqual({ nPoints: 2, maskScore: 0.5 });
  });

  it('applySamDelta writes the new sam id at the given indices', () => {
    const classFull = new Int8Array([-1, -1, -1]);
    const instanceFull = new Int32Array([-1, -1, -1]);
    const s = initSegState({ classFull, instanceFull });
    const next = applySamDelta(s, { indices: [0, 2], samSegId: 5 });
    expect(Array.from(next.samIds)).toEqual([5, -1, 5]);
    expect(next.samSegments.get(5)).toEqual({ nPoints: 2, maskScore: null });
  });

  it('applySamDelta shrinks an overlapping older candidate', () => {
    const classFull = new Int8Array([-1, -1]);
    const instanceFull = new Int32Array([-1, -1]);
    let s = initSegState({ classFull, instanceFull });
    s = applySamDelta(s, { indices: [0, 1], samSegId: 1 });
    s = applySamDelta(s, { indices: [1], samSegId: 2 });   // overlaps id 1 at index 1
    expect(Array.from(s.samIds)).toEqual([1, 2]);
    expect(s.samSegments.get(1)).toEqual({ nPoints: 1, maskScore: null });
    expect(s.samSegments.get(2)).toEqual({ nPoints: 1, maskScore: null });
  });

  it('applySamDelta drops a fully-overlapped older candidate', () => {
    const classFull = new Int8Array([-1]);
    const instanceFull = new Int32Array([-1]);
    let s = initSegState({ classFull, instanceFull });
    s = applySamDelta(s, { indices: [0], samSegId: 1 });
    s = applySamDelta(s, { indices: [0], samSegId: 2 });
    expect(s.samSegments.has(1)).toBe(false);
    expect(s.samSegments.get(2)).toEqual({ nPoints: 1, maskScore: null });
  });

  it('reconcileSamAfterApply clears samIds and removes samSegments/samSelection entries', () => {
    const classFull = new Int8Array([-1, -1]);
    const instanceFull = new Int32Array([-1, -1]);
    let s = initSegState({ classFull, instanceFull });
    s = applySamDelta(s, { indices: [0, 1], samSegId: 4 });
    s = { ...s, samSelection: new Set([4]) };
    const next = reconcileSamAfterApply(s, new Set([4]));
    expect(Array.from(next.samIds)).toEqual([-1, -1]);
    expect(next.samSegments.has(4)).toBe(false);
    expect(next.samSelection.has(4)).toBe(false);
  });

  it('reconcileSamAfterApply is a no-op for an empty id set', () => {
    const classFull = new Int8Array([-1]);
    const instanceFull = new Int32Array([-1]);
    const s = initSegState({ classFull, instanceFull });
    expect(reconcileSamAfterApply(s, new Set())).toBe(s);
  });
});
```

Add `applySamDelta, reconcileSamAfterApply` to the top-of-file import from `./segment-state.js` in the test file.

- [ ] **Step 2: Run to verify failure**

```bash
cd frontend && npx vitest run src/segment-state.test.js
```

Expected: fails — `applySamDelta`/`reconcileSamAfterApply` not exported, `samIds` undefined on `initSegState`'s result.

- [ ] **Step 3: Implement in `segment-state.js`**

Update `initSegState` (`segment-state.js:1-22`):

```js
export function initSegState({
  classFull, instanceFull, isFromPrelabel = false,
  segBoxes = null, segHulls = null,
  samIds = null, samSegments = [],
}) {
  return {
    classFull,
    instanceFull,
    summary: deriveSummary(classFull, instanceFull),
    dirty: false,
    selection: new Set(),
    activeTool: 'cuboid',
    brush: { radius: 0.05, mode: 'create', destInstance: null, destClass: 0 },
    isFromPrelabel,
    segBoxes,  // { segIds, segCenters, segSizes } — kept for fallback / metrics
    segHulls,  // { vertices: Float32Array, faces: Int32Array, faceSeg: Int32Array }
    // Pointset rows removed by an undo, keyed by segId, so a redo revives the
    // row (with its OBB/centerline volume + seq) instead of orphaning points.
    // Living on the state keeps it session-scoped structurally: a scene or
    // session switch rebuilds the state and drops these with it.
    dormant: new Map(),
    // SAM candidate layer — parallel to instanceFull, never merged into it.
    // A point can carry a sam id with no class (unlike instanceFull, which
    // always pairs with a real classFull entry once >= 0).
    samIds: samIds || new Int32Array(classFull.length).fill(-1),
    samSegments: new Map(samSegments.map((s) =>
      [s.id, { nPoints: s.n_points, maskScore: s.mask_score ?? null }])),
    samSelection: new Set(),
  };
}
```

Add `applySamDelta` and `reconcileSamAfterApply` after `applyDelta` (`segment-state.js:24-30`):

```js
// The SAM-layer analogue of applyDelta: writes samSegId at each index and
// shrinks/drops any older candidate those indices used to belong to
// (last-materialize-wins, mirrors SegmentSession._retire_sam_ids).
export function applySamDelta(state, { indices, samSegId }) {
  const samIds = state.samIds.slice();
  const shrink = new Map();
  for (let k = 0; k < indices.length; k++) {
    const old = samIds[indices[k]];
    if (old >= 0 && old !== samSegId) shrink.set(old, (shrink.get(old) || 0) + 1);
    samIds[indices[k]] = samSegId;
  }
  const samSegments = new Map(state.samSegments);
  for (const [oldId, removed] of shrink) {
    const entry = samSegments.get(oldId);
    if (!entry) continue;
    const nPoints = entry.nPoints - removed;
    if (nPoints <= 0) samSegments.delete(oldId);
    else samSegments.set(oldId, { ...entry, nPoints });
  }
  samSegments.set(samSegId, { nPoints: indices.length, maskScore: null });
  return { ...state, samIds, samSegments };
}

// After classifying a SAM selection, retire the absorbed candidates: clear
// their samIds entries and drop them from samSegments/samSelection so they
// vanish from the SAM segment list (mirrors presegments disappearing via
// promotedSegIds once absorbed into an instance).
export function reconcileSamAfterApply(state, appliedSamSegIds) {
  if (!appliedSamSegIds || appliedSamSegIds.size === 0) return state;
  const samIds = state.samIds.slice();
  for (let p = 0; p < samIds.length; p++) {
    if (appliedSamSegIds.has(samIds[p])) samIds[p] = -1;
  }
  const samSegments = new Map(state.samSegments);
  const samSelection = new Set(state.samSelection);
  for (const id of appliedSamSegIds) {
    samSegments.delete(id);
    samSelection.delete(id);
  }
  return { ...state, samIds, samSegments, samSelection };
}
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd frontend && npx vitest run src/segment-state.test.js
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/segment-state.js frontend/src/segment-state.test.js
git commit -m "feat(sam): add samIds/samSegments/samSelection + pure helpers to segState"
```

---

## Task 8: `api.js` — decode `full_sam_ids`/`sam_segments`, update `samProject`

**Files:**
- Modify: `frontend/src/api.js`

No dedicated test file exists for `api.js` (it's a thin fetch wrapper covered indirectly by component/backend tests) — verify this task via the frontend build + the manual browser check in Task 12.

- [ ] **Step 1: Update `segState()`** (`api.js:192-209`)

```js
  async segState() {
    const r = await fetch('/api/segment/state');
    if (!r.ok) throw new Error(`segState failed: ${r.status} ${await r.text()}`);
    const j = await r.json();
    if (!j.has_state) return null;
    return {
      nAssigned: j.n_assigned,
      nSegments: j.n_segments,
      fullClassIds: b64ToInt8(j.full_class_ids),
      fullInstanceIds: b64ToInt32(j.full_instance_ids),
      segIds: j.seg_ids ? b64ToInt32(j.seg_ids) : null,
      segCenters: j.seg_centers ? b64ToFloat32(j.seg_centers) : null,
      segSizes: j.seg_sizes ? b64ToFloat32(j.seg_sizes) : null,
      hullVertices: j.hull_vertices ? b64ToFloat32(j.hull_vertices) : null,
      hullFaces: j.hull_faces ? b64ToInt32(j.hull_faces) : null,
      hullFaceSeg: j.hull_face_seg ? b64ToInt32(j.hull_face_seg) : null,
      fullSamIds: j.full_sam_ids ? b64ToInt32(j.full_sam_ids) : null,
      samSegments: j.sam_segments || [],
    };
  },
```

- [ ] **Step 2: Update `samProject()`** (`api.js:271-286`)

```js
  async samProject({ captureId, maskIds, protectInstances = [] }) {
    const r = await fetch('/api/sam/project', {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ capture_id: captureId, mask_ids: maskIds,
                             protect_instances: protectInstances }),
    });
    if (!r.ok) throw new Error(`samProject failed: ${r.status} ${await r.text()}`);
    const j = await r.json();
    return {
      segments: (j.segments || []).map((s) => ({
        maskId: s.mask_id,
        samSegId: s.sam_seg_id,
        nAffected: s.n_affected,
        nProtected: s.n_protected,
        indices: s.scan_indices_b64 ? b64ToInt32(s.scan_indices_b64) : null,
      })),
    };
  },
```

- [ ] **Step 3: Sanity-check with a quick lint/build**

```bash
cd frontend && npx vitest run
```

Expected: no new failures (this file has no dedicated tests, but other suites that import `api.js` — e.g. `sam-util.test.js` — must still pass, confirming no syntax errors).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/api.js
git commit -m "feat(sam): decode full_sam_ids/sam_segments, drop target_class from samProject"
```

---

## Task 9: `App.jsx` — thread `samIds`/`samSegments` through both hydration call sites

**Files:**
- Modify: `frontend/src/App.jsx` (lines ~262-272 and ~294-301)

No new automated test — covered by the manual browser-verification pass in Task 12 (reload with pending SAM candidates). This task only threads already-decoded fields through existing `initSegState` calls.

- [ ] **Step 1: Update the `segLive` branch** (`App.jsx:262-272`)

```js
        setSegState(initSegState({
          classFull: segLive.fullClassIds,
          instanceFull: segLive.fullInstanceIds,
          isFromPrelabel: !!c.isFromPrelabel,
          segBoxes: (segLive.segIds && segLive.segCenters && segLive.segSizes)
            ? { segIds: segLive.segIds, segCenters: segLive.segCenters, segSizes: segLive.segSizes }
            : null,
          segHulls: (segLive.hullVertices && segLive.hullFaces && segLive.hullFaceSeg)
            ? { vertices: segLive.hullVertices, faces: segLive.hullFaces, faceSeg: segLive.hullFaceSeg }
            : null,
          samIds: segLive.fullSamIds || null,
          samSegments: segLive.samSegments || [],
        }));
```

- [ ] **Step 2: Leave the second branch (`App.jsx:294-301`) unchanged**

That branch fires only when there is no live seg session (`segLive` is null) — it already omits `segHulls` too, for the same reason (no active `SegmentSession` to have produced SAM candidates from). Adding `samIds`/`samSegments` there would require plumbing `full_sam_ids` through `LoadResponse` as well, which is out of scope (see spec's Frontend section — this asymmetry already exists for hulls and is intentional, not a gap).

- [ ] **Step 3: Manual sanity check**

```bash
npm run dev
```

Load an annotated scene in Label mode, confirm no console errors on load (this is a data-threading change with no new UI yet — full verification happens in Task 12 once the SAM list/viewport wiring exists).

- [ ] **Step 4: Commit**

```bash
git add frontend/src/App.jsx
git commit -m "feat(sam): thread samIds/samSegments through segLive hydration"
```

---

## Task 10: `mode-label.jsx` — `confirmSamSelection` + viewport pick/recolor

**Files:**
- Modify: `frontend/src/mode-label.jsx`

This is the largest single-file task. Do it as one commit at the end but implement in the sub-steps below so each piece can be checked independently.

- [ ] **Step 1: Import the new pure functions**

Update the import at `mode-label.jsx:15`:

```js
import { applyDelta, computeDiffMask, reconcileSamAfterApply } from './segment-state.js';
```

- [ ] **Step 2: Add the SAM recolor color constant**

Near the top of the file, next to `LABEL_SEL_BOX_ID`/`LABEL_SEL_BOX_COLOR` (`mode-label.jsx:27-28`):

```js
const LABEL_SEL_BOX_ID = '__label_sel_box__';
const LABEL_SEL_BOX_COLOR = '#ffd24a';
const SAM_CANDIDATE_COLOR = 0x22d3ee; // tailwind cyan-400 — distinct from the
                                       // yellow (0xfacc15) selection overlay
```

- [ ] **Step 3: Extend the viewport pick handler for SAM** (`mode-label.jsx:158-174`)

```js
  useEffectLabel(() => {
    if (!segState) return;
    if (subModeOwnsInput) return;
    const viewer = viewerRef?.current;
    if (!viewer?.onPointerPick) return;
    return viewer.onPointerPick((fullIndex, evt) => {
      if (activeTool === 'sam') {
        if (!evt.ctrlKey && !evt.metaKey && !evt.shiftKey) return;
        const samId = segState.samIds[fullIndex];
        if (samId < 0) return;
        setSegState((s) => {
          if (!s) return s;
          const next = new Set(s.samSelection);
          next.has(samId) ? next.delete(samId) : next.add(samId);
          return { ...s, samSelection: next };
        });
        return;
      }
      if (!evt.ctrlKey && !evt.metaKey) return;
      const instId = segState.instanceFull[fullIndex];
      if (instId < 0) return;
      setSegState((s) => {
        if (!s) return s;
        const next = new Set(s.selection);
        next.has(instId) ? next.delete(instId) : next.add(instId);
        return { ...s, selection: next };
      });
    });
  }, [segState, viewerRef, setSegState, subModeOwnsInput, activeTool]);
```

- [ ] **Step 4: Extend the viewport recolor effect for SAM** (`mode-label.jsx:132-153`)

```js
  useEffectLabel(() => {
    const v = viewerRef?.current;
    if (!v?.setSelectedSegmentMask) return;
    if (!segState || !cloud?.positions) {
      v.setSelectedSegmentMask(null);
      return;
    }
    const subIdx = cloud.subsampleIdx;
    const subN = cloud.positions.length / 3;
    if (activeTool === 'sam') {
      const samIds = segState.samIds;
      const mask = new Uint8Array(subN);
      let any = false;
      for (let p = 0; p < subN; p++) {
        const f = subIdx ? subIdx[p] : p;
        if (samIds[f] >= 0) { mask[p] = 1; any = true; }
      }
      v.setSelectedSegmentMask(any ? mask : null, SAM_CANDIDATE_COLOR);
      return;
    }
    const sel = segState.selection;
    if (sel.size === 0) {
      v.setSelectedSegmentMask(null);
      return;
    }
    const inst = segState.instanceFull;
    const mask = new Uint8Array(subN);
    for (let p = 0; p < subN; p++) {
      const f = subIdx ? subIdx[p] : p;
      if (sel.has(inst[f])) mask[p] = 1;
    }
    v.setSelectedSegmentMask(mask, fastMode ? FAST_HIGHLIGHT_COLOR : undefined);
  }, [segState?.selection, segState?.samIds, segState?.instanceFull, cloud, viewerRef, fastMode, activeTool]);
```

Note: this deliberately shows **every** live SAM candidate tinted, not just the selected ones — matches the design's "recolor points only" call; per-candidate selection is visible via the new SAM segment list's `.selected` row styling (Task 11), not a second viewport color tier.

- [ ] **Step 5: Add `confirmSamSelection`** — sibling of `confirmSegmentSelection`, placed immediately after it (`mode-label.jsx:641-691`)

```js
  const confirmSamSelection = useCallbackLabel(async (clsDef) => {
    const targetCls = clsDef || activeClassDef;
    if (!segState || segState.samSelection.size === 0) return;
    if (!targetCls) return;
    const samIds = segState.samIds;
    const sel = segState.samSelection;
    const idx = [];
    for (let p = 0; p < samIds.length; p++) {
      if (sel.has(samIds[p])) idx.push(p);
    }
    if (idx.length === 0) return;
    const indices = new Int32Array(idx);

    let r;
    try {
      r = await VoxaAPI.segApply('reassign', {
        indices,
        payload: { target_inst: -1, target_class: targetCls.id },
      });
    } catch (err) {
      console.error('confirm sam reassign failed:', err);
      return;
    }

    const newSegId = r.afterInstance && r.afterInstance.length > 0
      ? r.afterInstance[0] : -1;
    const appliedSamSegIds = new Set(sel);
    if (newSegId >= 0) {
      const newInst = {
        id: newId(),
        segId: newSegId,
        kind: 'pointset',
        cls: targetCls.id,
        label: `${targetCls.label} ${(counts[targetCls.id] || 0) + 1}`,
        color: targetCls.color,
        source: 'sam',
        confirmed: !!autoConfirmFor('sam'),
      };
      onChange([...instances, newInst]);
    }

    setSegState((s) => {
      if (!s) return s;
      const next = applyDelta(s, {
        indices: r.indices,
        after_class: r.afterClass,
        after_instance: r.afterInstance,
      });
      return reconcileSamAfterApply(next, appliedSamSegIds);
    });
  }, [segState, activeClassDef, instances, counts, onChange, setSegState, autoConfirm]);
```

- [ ] **Step 6: Wire the Ctrl+Enter gate** (`mode-label.jsx:892-903`)

```js
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if ((segState && segState.selection.size > 0)
          || (activeTool === 'box' && selBox)
          || (activeTool === 'sam' && segState && segState.samSelection.size > 0)) {
          setClassPickerOpen(true);
        } else if (activeTool === 'box') {
          toggleConfirmSelected();
        }
        return;
      }
```

- [ ] **Step 7: Wire the direct class-hotkey path** (`mode-label.jsx:911-923`)

```js
      const cls = classes.find((c) => c.hotkey === e.key);
      if (cls) {
        if (activeTool === 'sam' && segState && segState.samSelection.size > 0) {
          e.preventDefault();
          confirmSamSelection(cls);
        } else if (segState && segState.selection.size > 0) {
          e.preventDefault();
          confirmSegmentSelection(cls);
        } else if (activeTool === 'box' && selBox) {
          e.preventDefault();
          applyBox(cls);
        } else {
          setActiveClass(cls.id);
        }
        return;
      }
```

- [ ] **Step 8: Add `confirmSamSelection` to the keydown effect's dependency array** (`mode-label.jsx:947`)

```js
  }, [classes, selected, isLocked, instances, activeTool, navMode, segState, selBox, confirmSegmentSelection, confirmSamSelection, applyBox, fastMode, subModeOwnsInput]);
```

- [ ] **Step 9: Wire `ClassPickerModal`'s `onPick`** (`mode-label.jsx:969-980`)

```js
      {classPickerOpen && (
        <ClassPickerModal
          classes={classes}
          counts={counts}
          onPick={(cls) => {
            setClassPickerOpen(false);
            if (activeTool === 'sam' && segState && segState.samSelection.size > 0) confirmSamSelection(cls);
            else if (activeTool === 'box' && selBox) applyBox(cls);
            else confirmSegmentSelection(cls);
          }}
          onClose={() => setClassPickerOpen(false)}
        />
      )}
```

- [ ] **Step 10: Manual smoke check (automated coverage lands in Task 13)**

```bash
cd frontend && npx vitest run
```

Expected: existing suites still green (no test targets this file's internals directly yet — `segment-state.test.js` covers the pure functions it now calls).

- [ ] **Step 11: Commit**

```bash
git add frontend/src/mode-label.jsx
git commit -m "feat(sam): wire confirmSamSelection into pick/recolor/Ctrl+Enter/hotkey/picker paths"
```

---

## Task 11: `SamSegmentList` component

**Files:**
- Create: `frontend/src/sam-segment-list.jsx`
- Test: Create `frontend/src/sam-segment-list.test.js`

The project has no `jsdom`/`@testing-library/react` (`CLAUDE.md`: "the current setup is pure-function only") and this plan doesn't add them — matching `PresegmentList`, which has no dedicated component test either. So `SamSegmentList`'s row-selection logic is factored into an exported pure function, `toggleSamSelection`, which gets real vitest coverage; the JSX shell itself is covered by the Task 13 browser-verification pass instead.

- [ ] **Step 1: Write a failing test for `toggleSamSelection`**

```js
import { describe, it, expect } from 'vitest';
import { toggleSamSelection } from './sam-segment-list.jsx';

describe('toggleSamSelection', () => {
  it('adds an id not yet in the selection', () => {
    const next = toggleSamSelection(new Set([1]), 2);
    expect(Array.from(next)).toEqual([1, 2]);
  });

  it('removes an id already in the selection', () => {
    const next = toggleSamSelection(new Set([1, 2]), 2);
    expect(Array.from(next)).toEqual([1]);
  });

  it('does not mutate the input set', () => {
    const input = new Set([1]);
    toggleSamSelection(input, 2);
    expect(Array.from(input)).toEqual([1]);
  });
});
```

- [ ] **Step 2: Run to verify failure**

```bash
cd frontend && npx vitest run src/sam-segment-list.test.js
```

Expected: fails — `frontend/src/sam-segment-list.jsx` doesn't exist yet.

- [ ] **Step 3: Write the component** (mirrors `PresegmentList`, `frontend/src/segment-tools.jsx:62-130`)

```jsx
// sam-segment-list.jsx — bottom-of-panel list of materialized SAM candidate
// segments (accepted masks not yet classified). Sibling of PresegmentList —
// deliberately a SEPARATE component/list; SAM candidates and presegments are
// never mixed in one panel or one selection set.
import { useMemo } from 'react';

export function toggleSamSelection(samSelection, samSegId) {
  const next = new Set(samSelection);
  next.has(samSegId) ? next.delete(samSegId) : next.add(samSegId);
  return next;
}

export function SamSegmentList({ segState, setSegState }) {
  const segments = useMemo(() => {
    if (!segState) return [];
    return Array.from(segState.samSegments.entries())
      .map(([id, meta]) => ({ id, ...meta }))
      .sort((a, b) => b.nPoints - a.nPoints);
  }, [segState]);

  if (!segState) return null;

  const onRowClick = (samSegId, evt) => {
    if (!(evt.ctrlKey || evt.metaKey || evt.shiftKey)) return;
    setSegState((s) => (s ? { ...s, samSelection: toggleSamSelection(s.samSelection, samSegId) } : s));
  };

  return (
    <div className="preseg-panel">
      <div className="side-hd" style={{ marginTop: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
        <span>SAM segments</span>
        <span className="badge-soft">{segments.length}</span>
      </div>
      <div className="inst-list">
        {segments.length === 0 && (
          <div className="sugg-empty" style={{ fontSize: '11px', padding: '6px 4px' }}>
            No SAM segments yet. Shift-drag a box or run a concept capture.
          </div>
        )}
        {segments.map((seg) => {
          const isSel = segState.samSelection.has(seg.id);
          return (
            <div key={seg.id}
              className={'inst-row' + (isSel ? ' selected' : '')}
              onClick={(e) => onRowClick(seg.id, e)}
              title={isSel ? 'Ctrl/Shift-click to deselect' : 'Ctrl/Shift-click to select'}>
              <span className="inst-dot" style={{ background: '#22d3ee' }} />
              <div className="inst-text">
                <b>SAM #{seg.id}</b>
                <em>{seg.nPoints.toLocaleString()} pts</em>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
```

- [ ] **Step 4: Run tests to verify pass**

```bash
cd frontend && npx vitest run src/sam-segment-list.test.js
```

Expected: all 3 pass; the component isn't imported anywhere yet (that's Task 12), so no additional check needed here.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/sam-segment-list.jsx frontend/src/sam-segment-list.test.js
git commit -m "feat(sam): add SamSegmentList panel component"
```

---

## Task 12: Wire `SamSegmentList` + simplify `SamMode`'s accept flow

**Files:**
- Modify: `frontend/src/tool-options.jsx` (`SamOptions`, `tool-options.jsx:107-126`)
- Modify: `frontend/src/sam-mode.jsx` (`doProject`, `SamMode` props, `SamReviewModal` button label)

- [ ] **Step 1: Update `SamOptions` to render the list, drop obsolete props**

In `tool-options.jsx`, add the import:

```js
import { SamSegmentList } from './sam-segment-list.jsx';
```

Replace `SamOptions` (`tool-options.jsx:107-126`):

```js
function SamOptions({
  viewerRef, classes, protectInstances, setSegState, segState,
  autoConfirm, setAutoConfirm, activeSessionId,
}) {
  return (
    <div className="tool-options tool-options-sam">
      <SamMode
        key={activeSessionId}
        viewerRef={viewerRef}
        setSegState={setSegState}
        protectInstances={protectInstances}
      />
      <AutoConfirmToggle tool="sam" autoConfirm={autoConfirm} setAutoConfirm={setAutoConfirm} />
      {segState && <SamSegmentList segState={segState} setSegState={setSegState} />}
    </div>
  );
}
```

Note: `classes`, `activeClass`, `onToolApplied` are dropped from the props `SamMode` needs — classification no longer happens inside the SAM tool panel at all (it happens via the shared Ctrl+Enter/hotkey path once a SAM segment list row is selected, exactly like presegments). Since `ToolOptions` (`tool-options.jsx:165-171`) spreads `{...props}` into every `*Options` component, no caller-side change is needed at the `ToolOptions` call site — `SamOptions` simply now ignores the props it no longer needs.

- [ ] **Step 2: Simplify `SamMode`'s accept flow in `sam-mode.jsx`**

Update the `SamMode` function signature (`sam-mode.jsx:183-186`):

```js
export default function SamMode({
  viewerRef, setSegState, protectInstances = [],
}) {
```

Replace `doProject` (`sam-mode.jsx:241-275`):

```js
  const doProject = useCallback(async () => {
    if (busyRef.current) return;
    if (!capture || chosen.size === 0) return;
    busyRef.current = true;
    setBusy(true);
    setError(null);
    try {
      const res = await VoxaAPI.samProject({
        captureId: capture.captureId,
        maskIds: [...chosen],
        protectInstances: protectInstancesRef.current,
      });
      for (const seg of res.segments || []) {
        if (seg.indices) {
          setSegState?.((s) => (s ? applySamDelta(s, {
            indices: seg.indices,
            samSegId: seg.samSegId,
          }) : s));
        }
      }
      setCapture(null);
      setChosen(new Set());
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [capture, chosen, setSegState]);
```

Update the import line (`sam-mode.jsx:11`):

```js
import { applySamDelta } from './segment-state.js';
```

Update the "Project selected" button label in `SamReviewModal` (`sam-mode.jsx:175-176`) to reflect that accepting no longer classifies:

```jsx
          <button className="ghost-btn" disabled={chosen.size === 0 || busy}
            onClick={onProject}>Add to SAM segments</button>
```

- [ ] **Step 3: Update `test_sam3_cache_key.py`/frontend `sam-util.test.js` if they reference removed props**

```bash
cd frontend && npx vitest run src/sam-util.test.js
.venv/bin/pytest backend/tests/test_sam3_cache_key.py -v
```

Expected: unaffected (these test capture-payload/cache-key pure functions, not `SamMode`'s React props) — run to confirm, no changes expected.

- [ ] **Step 4: Full frontend suite**

```bash
cd frontend && npx vitest run
```

- [ ] **Step 5: Commit**

```bash
git add frontend/src/tool-options.jsx frontend/src/sam-mode.jsx
git commit -m "feat(sam): SAM accept flow materializes candidates instead of classifying; wire SamSegmentList"
```

---

## Task 13: Browser verification

**Files:** none (manual verification pass)

@superpowers:verification-before-completion — do not report this feature complete without running this pass.

- [ ] **Step 1: Start the dev server**

```bash
npm run dev
```

- [ ] **Step 2: Open a throwaway session on an annotated scan with SAM enabled**

Per `feedback_browser_verify_mutates_session` (memory) — SAM apply still autosaves (now via `materialize_sam_segment`'s `schedule_autosave`), so create a throwaway session before testing, not one with real labels. Confirm `VOXA_SAM_SIDECAR_URL` is set (or mock/stub the sidecar per the existing SAM e2e notes) — if no sidecar is available in this environment, do the deepest check possible short of a live capture (Steps 3-4 below) and note explicitly in your report that live-capture verification was skipped and why.

- [ ] **Step 3: Verify the split UI (no sidecar needed)**

Select the SAM tool. Confirm:
- The left-rail panel shows the SAM controls (Box/Concept toggle, capture inputs) AND, below them, an empty "SAM segments" list with the "No SAM segments yet…" hint (Task 11's component).
- No presegment rows are visible in this panel, even if the scan has presegments (separate lists, per your explicit requirement).
- Zero console errors from mounting `SamOptions`/`SamSegmentList`.

- [ ] **Step 4: Verify the classify pipeline reuses cleanly (if a sidecar is available)**

- Shift-drag a box, accept the returned mask in the review modal ("Add to SAM segments" button) → confirm exactly one new row appears in the SAM segment list, and the corresponding points recolor cyan in the viewport (not yellow — that's reserved for the selection overlay).
- Ctrl-click the row (or Ctrl-click a cyan point in the viewport) → row highlights as `.selected`.
- Press a class hotkey → confirm: (a) the row disappears from the SAM segment list, (b) the points are no longer cyan-tinted, (c) a new row appears in the right Instances panel with `source: 'sam'`, confirmed per the auto-confirm toggle's current state.
- Repeat with a second overlapping capture and confirm the first candidate's `n_points` shrinks or disappears if the second capture re-covers those points (last-materialize-wins).
- Reload the page mid-review (materialize a candidate, don't classify it, then refresh) → confirm the still-unclassified candidate reappears in the SAM segment list after reload.

- [ ] **Step 5: Report results**

Summarize what was verified vs. skipped (e.g., "no sidecar available in this environment — verified UI split and empty-state rendering; live capture/classify/reload cycle not exercised") rather than claiming full end-to-end success if Step 4 couldn't run.

---

## Task 14: Docs pass

**Files:**
- Modify: `CLAUDE.md` (SAM tool section)

Per the user's global CLAUDE.md: docs ship with the code, in the same PR, as the last step before opening a PR.

- [ ] **Step 1: Update the SAM bullet in the root `CLAUDE.md`**

The current SAM tool description ends with: "each kept mask becomes an unconfirmed pointset." Update this (and the surrounding sentences about `/api/sam/project` calling `apply_reassign`) to describe the new two-step flow: accept materializes a candidate into the `sam_ids` layer (parallel to `instance_ids`/`class_ids`, mirrors `preseg_ids`), shown in its own SAM segment list + viewport recolor; classifying a selection over those candidates (same Ctrl+Enter/hotkey path as presegments) is what produces the actual pointset instance. Reference `docs/superpowers/specs/2026-07-13-sam-segments-as-presegments-design.md` alongside the existing `2026-07-12-sam-labeling-tool-design.md` reference.

- [ ] **Step 2: Run the `simplify` skill on the full diff**

Per the user's global workflow instructions, run `simplify` on the complete implementation diff before considering this done.

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs(sam): describe the SAM candidate-segment layer in CLAUDE.md"
```

---

## Final checkpoint

```bash
npm test
```

Expected: full backend + frontend suite green. Then run `/code-review` (or the `code-review` skill at `medium`/`high` effort) on the full branch diff before opening a PR, per standard project practice for a multi-file feature.
