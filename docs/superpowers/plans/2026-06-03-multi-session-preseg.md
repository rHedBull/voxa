# Multi-Session Labeling (scan-schema v2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** N labeling sessions per scan, each immutably pinned at creation to one of N presegment results via fingerprint hashes, selectable in the UI; per-session saved outputs replace the single `labels/` GT (scan-schema v1.3 → v2).

**Architecture:** Two new pure-I/O stores (`session_store`, `preseg_store`) own the new `sessions/<id>/` and `prelabel/<preseg_id>/` dirs; `ScanLayout` gains parameterized v2 accessors; the load route resolves a session (explicit or last-worked), hard-fails (409) on pin mismatch, and the save route writes into the active session's `output/`. A one-shot migration script converts v1.3 scans in place; voxa v2 reads only v2.

**Tech Stack:** FastAPI + numpy (backend), pytest, React 18 + Vite (frontend), vitest.

**Spec:** `docs/superpowers/specs/2026-06-03-multi-session-preseg-design.md` — read it first; it is the contract.

**Scope note (planning decision):** multi-session applies to the **annotated tier only**. Non-annotated tiers (legacy/decimated/raw) keep their existing single recovery-session under `<data_dir>/sessions/<tier>__<name>/` exactly as today. The session picker is hidden for them.

**Worktree:** all work happens in `/home/hendrik/coding/engine/tools/labeling/voxa/.claude/worktrees/multi-session-spec` on branch `feat/multi-session-spec`. Run backend tests with `.venv/bin/pytest` from the **original repo root's venv** (`/home/hendrik/coding/engine/tools/labeling/voxa/.venv/bin/pytest`) with `--rootdir` defaulting fine; if imports fail, `npm run test:backend` inside the worktree auto-creates a venv.

---

## File structure (end state)

```
backend/
├── scenes/scan_layout.py        MODIFIED  v2 accessors; v1.3 single-slot props deleted (Task 12)
├── scenes/scene_registry.py     MODIFIED  annotated discovery requires schema_version 2.x
├── scenes/lidar_io.py           MODIFIED  load_annotated stops reading labels//prelabel/
├── preseg/preseg_store.py       NEW       list/load/register preseg results
├── labeling/session_store.py    NEW       session CRUD + pins + last_worked + verify
├── labeling/segment_io.py       MODIFIED  session.json (was current.json), v2 save_labels
├── labeling/segment_state.py    MODIFIED  preseg_run_id → preseg_id; aux gains name/created_at
├── app/core.py                  MODIFIED  _seed_or_recover_session → _resume_session; stale check deleted
├── app/schemas.py               MODIFIED  LoadRequest.session_id; session/preseg models
├── routes/load.py               MODIFIED  session resolution + 409 pins; keep_prev_seg deleted
├── routes/segment.py            MODIFIED  save → active session output/
├── routes/sessions.py           NEW       sessions CRUD + presegs list
└── tests/
    ├── conftest.py              MODIFIED  build_annotated_root emits v2
    ├── test_preseg_store.py     NEW
    ├── test_session_store.py    NEW
    ├── test_session_routes.py   NEW
    └── test_migrate_v2.py       NEW
scripts/
├── migrate_scan_v2.py           NEW       one-shot v1.3 → v2, --dry-run
└── preseg/presegment.py         MODIFIED  writes via register_preseg
frontend/src/
├── api.js                       MODIFIED  session/preseg endpoints; load(sessionId)
├── api.test.js (or new file)    MODIFIED  mapper tests
├── session-picker.jsx           NEW       list/create/rename/delete panel
├── App.jsx                      MODIFIED  sessions state, auto-resume, 409 banner
└── mode-label.jsx               MODIFIED  mounts SessionPicker
docs/scan-schema.md              MODIFIED  rewritten to v2
CLAUDE.md, README.md             MODIFIED  v2 notes
(SIBLING REPO engine/data: tools/scaffold_annotation.py + lidar/SCHEMA.md — Task 11, separate commit there)
```

Key conventions to honor everywhere (from the spec):

- `meta.json::schema_version` is a **string**; existing scans carry `"1.3"`. v2 writes `"2.0"`; the discovery gate is `str(v).startswith("2")`.
- Working class ids `int8`, output `gt_class_ids` `int32` — intentional asymmetry, do not unify.
- Fingerprints via existing `segment_io.compute_fingerprint()`.
- ids match `^[a-z0-9_-]+$` (preseg_id, and the generated session_id is `<YYYYMMDD-HHMMSS>_<preseg_id|blank>`).
- Fail loudly: no silent fallbacks; errors carry the reason.

---

### Task 1: ScanLayout v2 accessors

**Files:**
- Modify: `backend/scenes/scan_layout.py`
- Test: `backend/tests/test_scan_layout.py`

Add v2 accessors **alongside** the v1.3 properties (old ones are deleted in Task 12 after all consumers move).

- [ ] **Step 1: Write failing tests** — append to `backend/tests/test_scan_layout.py`:

```python
def test_v2_preseg_paths(tmp_path):
    lay = ScanLayout(tmp_path / "annotated" / "demo")
    d = lay.preseg_dir("ransac")
    assert d == lay.presegs_root / "ransac"
    assert lay.presegs_root == lay.scan_dir / "prelabel"
    assert (d / "instance_ids.npy").name == "instance_ids.npy"


def test_v2_session_paths(tmp_path):
    lay = ScanLayout(tmp_path / "annotated" / "demo")
    s = lay.session("20260603-120000_ransac")
    assert lay.sessions_root == lay.scan_dir / "sessions"
    assert s.dir == lay.sessions_root / "20260603-120000_ransac"
    assert s.session_json == s.dir / "session.json"
    assert s.working_class_ids == s.dir / "working_class_ids.npy"
    assert s.working_segment_ids == s.dir / "working_segment_ids.npy"
    assert s.output_dir == s.dir / "output"
    assert s.output_gt_class_ids == s.output_dir / "gt_class_ids.npy"
    assert s.output_gt_segment_ids == s.output_dir / "gt_segment_ids.npy"
    assert s.output_gt_segment_metadata == s.output_dir / "gt_segment_metadata.json"
    assert s.history_dir == s.dir / "history"
```

- [ ] **Step 2: Run, verify fail** — `pytest backend/tests/test_scan_layout.py -v` → FAIL (`AttributeError: preseg_dir`).

- [ ] **Step 3: Implement** — in `scan_layout.py` add (keep all existing properties for now; module docstring gains a "v2" line):

```python
@dataclass(frozen=True)
class SessionPaths:
    """Paths inside sessions/<session_id>/ (scan-schema v2)."""
    dir: Path

    @property
    def session_json(self) -> Path: return self.dir / "session.json"
    @property
    def working_class_ids(self) -> Path: return self.dir / "working_class_ids.npy"
    @property
    def working_segment_ids(self) -> Path: return self.dir / "working_segment_ids.npy"
    @property
    def output_dir(self) -> Path: return self.dir / "output"
    @property
    def output_gt_class_ids(self) -> Path: return self.output_dir / "gt_class_ids.npy"
    @property
    def output_gt_segment_ids(self) -> Path: return self.output_dir / "gt_segment_ids.npy"
    @property
    def output_gt_segment_metadata(self) -> Path: return self.output_dir / "gt_segment_metadata.json"
    @property
    def history_dir(self) -> Path: return self.dir / "history"
```

and on `ScanLayout`:

```python
    # v2: prelabel/<preseg_id>/ + sessions/<session_id>/
    @property
    def presegs_root(self) -> Path:
        return self.scan_dir / "prelabel"

    def preseg_dir(self, preseg_id: str) -> Path:
        return self.presegs_root / preseg_id

    @property
    def sessions_root(self) -> Path:
        return self.scan_dir / "sessions"

    def session(self, session_id: str) -> SessionPaths:
        return SessionPaths(self.sessions_root / session_id)
```

- [ ] **Step 4: Run, verify pass** — `pytest backend/tests/test_scan_layout.py -v` → PASS.
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: ScanLayout v2 preseg/session accessors"`

---

### Task 2: preseg_store

**Files:**
- Create: `backend/preseg/preseg_store.py`
- Test: `backend/tests/test_preseg_store.py`

- [ ] **Step 1: Write failing tests** — `backend/tests/test_preseg_store.py`:

```python
import numpy as np
import pytest

from preseg.preseg_store import list_presegs, load_preseg, register_preseg
from scenes.scan_layout import ScanLayout


@pytest.fixture
def scan(tmp_path):
    d = tmp_path / "annotated" / "demo"
    d.mkdir(parents=True)
    return ScanLayout(d)


def _inst(n=8):
    return np.array([0, 0, 1, 1, 2, 2, -1, -1][:n], dtype=np.int32)


def test_register_then_list_and_load(scan):
    info = register_preseg(scan, "ransac", _inst(),
                           summary={"segments": [{"id": 0, "class_id": -1},
                                                 {"id": 1, "class_id": -1},
                                                 {"id": 2, "class_id": -1}]},
                           generator="ransac", params={"eps": 0.1})
    assert info.preseg_id == "ransac"
    assert info.fingerprint.startswith("sha256:")
    listed = list_presegs(scan)
    assert [p.preseg_id for p in listed] == ["ransac"]
    ci, ii = load_preseg(scan, "ransac", n_points=8)
    assert ii.tolist() == _inst().tolist()
    assert ci.dtype == np.int8  # class map applied; all -1 here


def test_register_rejects_bad_id(scan):
    with pytest.raises(ValueError, match="preseg_id"):
        register_preseg(scan, "Bad ID!", _inst(), summary={"segments": []},
                        generator="x", params={})


def test_load_shape_mismatch_raises(scan):
    register_preseg(scan, "ransac", _inst(), summary={"segments": []},
                    generator="x", params={})
    with pytest.raises(ValueError, match="shape"):
        load_preseg(scan, "ransac", n_points=99)


def test_fingerprint_stable_across_reload(scan):
    info = register_preseg(scan, "ransac", _inst(), summary={"segments": []},
                           generator="x", params={})
    assert list_presegs(scan)[0].fingerprint == info.fingerprint
```

- [ ] **Step 2: Run, verify fail** — `pytest backend/tests/test_preseg_store.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement** — `backend/preseg/preseg_store.py`:

```python
"""prelabel/<preseg_id>/ store (scan-schema v2).

Each preseg result is {instance_ids.npy, segment_summary.json, meta.json}.
The only code that reads or writes this layout. Pure I/O — no FastAPI,
no in-memory state. Errors raise; callers decide how to surface them.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from labeling.segment_io import atomic_write_json, atomic_write_npy, compute_fingerprint
from scenes.scan_layout import ScanLayout

_ID_RE = re.compile(r"^[a-z0-9_-]+$")


@dataclass(frozen=True)
class PresegInfo:
    preseg_id: str
    generator: str
    params: dict
    fingerprint: str
    source_fingerprint: Optional[str]
    created_at: str
    n_segments: int


def register_preseg(layout: ScanLayout, preseg_id: str, instance_ids: np.ndarray,
                    *, summary: dict, generator: str, params: dict,
                    source_fingerprint: Optional[str] = None) -> PresegInfo:
    """Publish a preseg result. Computes the fingerprint at write time so
    meta.json is authoritative for pin checks."""
    if not _ID_RE.match(preseg_id):
        raise ValueError(f"preseg_id {preseg_id!r} must match {_ID_RE.pattern}")
    instance_ids = instance_ids.astype(np.int32, copy=False)
    d = layout.preseg_dir(preseg_id)
    atomic_write_npy(d / "instance_ids.npy", instance_ids)
    atomic_write_json(d / "segment_summary.json", summary)
    meta = {
        "preseg_id": preseg_id,
        "generator": generator,
        "params": params,
        "fingerprint": compute_fingerprint(instance_ids),
        "source_fingerprint": source_fingerprint,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    atomic_write_json(d / "meta.json", meta)
    n_segments = int(np.unique(instance_ids[instance_ids >= 0]).size)
    return PresegInfo(n_segments=n_segments, **{k: meta[k] for k in (
        "preseg_id", "generator", "params", "fingerprint",
        "source_fingerprint", "created_at")})


def list_presegs(layout: ScanLayout) -> list[PresegInfo]:
    """Enumerate prelabel/*/meta.json. A dir without a readable meta.json is
    skipped with a warning-by-omission is NOT acceptable here — it raises,
    because a malformed preseg silently vanishing from the picker hides bugs."""
    root = layout.presegs_root
    if not root.is_dir():
        return []
    out: list[PresegInfo] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        meta_path = d / "meta.json"
        if not meta_path.exists():
            raise ValueError(f"preseg dir {d} has no meta.json (not a v2 preseg?)")
        meta = json.loads(meta_path.read_text())
        inst = np.load(d / "instance_ids.npy", mmap_mode="r")
        n_segments = int(np.unique(np.asarray(inst[inst >= 0])).size)
        out.append(PresegInfo(
            preseg_id=meta["preseg_id"], generator=meta.get("generator", "?"),
            params=meta.get("params", {}), fingerprint=meta["fingerprint"],
            source_fingerprint=meta.get("source_fingerprint"),
            created_at=meta.get("created_at", ""), n_segments=n_segments,
        ))
    return out


def load_preseg(layout: ScanLayout, preseg_id: str, n_points: int,
                ) -> tuple[np.ndarray, np.ndarray]:
    """Return (class_ids int8, instance_ids int32) for one preseg result.
    Raises on missing files / shape mismatch — preseg selection is explicit
    in v2, so a broken preseg must surface, not degrade to None."""
    d = layout.preseg_dir(preseg_id)
    inst_path = d / "instance_ids.npy"
    summary_path = d / "segment_summary.json"
    if not inst_path.exists() or not summary_path.exists():
        raise FileNotFoundError(f"preseg '{preseg_id}' incomplete under {d}")
    instance_ids = np.load(inst_path).astype(np.int32)
    if instance_ids.shape != (n_points,):
        raise ValueError(
            f"preseg '{preseg_id}': shape {instance_ids.shape} != ({n_points},)")
    summary = json.loads(summary_path.read_text())
    seg_to_class = {int(s["id"]): int(s["class_id"])
                    for s in summary.get("segments", [])}
    class_ids = np.full(n_points, -1, dtype=np.int8)
    for sid, cid in seg_to_class.items():
        class_ids[instance_ids == sid] = cid
    return class_ids, instance_ids
```

- [ ] **Step 4: Run, verify pass** — `pytest backend/tests/test_preseg_store.py -v` → PASS.
- [ ] **Step 5: Commit** — `git commit -m "feat: preseg_store for prelabel/<preseg_id>/ (v2)"`

---

### Task 3: session_store + segment_io session-file rename

**Files:**
- Create: `backend/labeling/session_store.py`
- Modify: `backend/labeling/segment_io.py` (SESSION_SCHEMA_VERSION → 2; `current.json` → `session.json`)
- Modify: `backend/labeling/segment_state.py` (`preseg_run_id` → `preseg_id`; aux gains `name`, `created_at`)
- Test: `backend/tests/test_session_store.py`; update `backend/tests/test_segment_io.py`, `backend/tests/test_segment_state.py` references to `current.json`/`preseg_run_id`

- [ ] **Step 1: segment_io rename (small, mechanical).** In `segment_io.py`: set `SESSION_SCHEMA_VERSION = 2`; in `save_session_aux`/`load_session_aux` replace `"current.json"` with `"session.json"` (3 occurrences incl. `load_working_arrays` gate via `load_session_aux`). In `segment_state.py`: rename attribute `preseg_run_id` → `preseg_id` (constructor line 56, `freeze_preseg` kwarg stays `run_id` external? **No** — rename the kwarg to `preseg_id` too and fix its callers: `grep -rn "preseg_run_id\|run_id=" backend/ | grep -v renders`), `_aux_payload` becomes:

```python
    def _aux_payload(self) -> dict:
        return {
            "preseg_id": self.preseg_id,
            "preseg_fingerprint": self.preseg_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "hidden_inst_ids": sorted(int(x) for x in self.hidden_inst_ids),
            "is_from_prelabel": bool(self.is_from_prelabel),
            "dirty": bool(self.dirty),
            "name": self.name,
            "created_at": self.created_at,
        }
```

with `self.name: str = ""` and `self.created_at: Optional[str] = None` initialized in `__init__` (schema_version is stamped by `save_session_aux` via `setdefault`, `saved_at` stamped there too — unchanged). Fix `routes/segment.py:74` (`preseg_run_id=seg.preseg_run_id` → keep the **response field name** `preseg_run_id` for now; it is removed/renamed in Task 8) by reading `seg.preseg_id`.

- [ ] **Step 2: Run existing suites** — `pytest backend/tests/test_segment_io.py backend/tests/test_segment_state.py backend/tests/test_segment_endpoints.py -v`; fix any `current.json` / `preseg_run_id` references in those tests. Expected: PASS.

- [ ] **Step 3: Write failing session_store tests** — `backend/tests/test_session_store.py`:

```python
import json
import time

import numpy as np
import pytest

from labeling import session_store as ss
from preseg.preseg_store import register_preseg
from scenes.scan_layout import ScanLayout


@pytest.fixture
def scan(tmp_path):
    d = tmp_path / "annotated" / "demo"
    d.mkdir(parents=True)
    lay = ScanLayout(d)
    register_preseg(lay, "ransac",
                    np.array([0, 0, 1, 1, -1, -1, 2, 2], dtype=np.int32),
                    summary={"segments": [{"id": i, "class_id": -1} for i in range(3)]},
                    generator="ransac", params={})
    return lay


SRC_FP = "sha256:dummysource"


def test_create_seeded_session(scan):
    info = ss.create_session(scan, name="first", preseg_id="ransac",
                             n_points=8, source_fp=SRC_FP)
    assert info.session_id.endswith("_ransac")
    sp = scan.session(info.session_id)
    aux = json.loads(sp.session_json.read_text())
    assert aux["preseg_id"] == "ransac"
    assert aux["preseg_fingerprint"].startswith("sha256:")
    assert aux["source_fingerprint"] == SRC_FP
    assert aux["name"] == "first"
    ii = np.load(sp.working_segment_ids)
    assert ii.tolist() == [0, 0, 1, 1, -1, -1, 2, 2]
    assert np.load(sp.working_class_ids).dtype == np.int8


def test_create_blank_session(scan):
    info = ss.create_session(scan, name="blank", preseg_id=None,
                             n_points=8, source_fp=SRC_FP)
    assert info.preseg_id is None
    sp = scan.session(info.session_id)
    assert (np.load(sp.working_segment_ids) == -1).all()


def test_create_missing_preseg_fails(scan):
    with pytest.raises(FileNotFoundError):
        ss.create_session(scan, name="x", preseg_id="nope",
                          n_points=8, source_fp=SRC_FP)


def test_list_rename_delete(scan):
    a = ss.create_session(scan, name="a", preseg_id=None, n_points=8, source_fp=SRC_FP)
    infos = ss.list_sessions(scan)
    assert [i.session_id for i in infos] == [a.session_id]
    ss.rename_session(scan, a.session_id, "renamed")
    assert ss.list_sessions(scan)[0].name == "renamed"
    ss.delete_session(scan, a.session_id)
    assert ss.list_sessions(scan) == []


def test_last_worked_ordering(scan, monkeypatch):
    a = ss.create_session(scan, name="a", preseg_id=None, n_points=8, source_fp=SRC_FP)
    time.sleep(1.1)  # saved_at has seconds resolution
    b = ss.create_session(scan, name="b", preseg_id=None, n_points=8, source_fp=SRC_FP)
    assert ss.last_worked(scan) == b.session_id
    # touching a's saved_at moves it ahead
    time.sleep(1.1)
    ss.touch_saved_at(scan, a.session_id)
    assert ss.last_worked(scan) == a.session_id


def test_verify_pins_ok_and_mismatches(scan):
    info = ss.create_session(scan, name="x", preseg_id="ransac",
                             n_points=8, source_fp=SRC_FP)
    ss.verify_pins(scan, info.session_id, source_fp=SRC_FP)  # no raise
    with pytest.raises(ss.PinMismatch) as e:
        ss.verify_pins(scan, info.session_id, source_fp="sha256:other")
    assert e.value.diverged == "source"
    # tamper with the preseg array → preseg pin diverges
    p = scan.preseg_dir("ransac") / "instance_ids.npy"
    np.save(p, np.array([9] * 8, dtype=np.int32))
    with pytest.raises(ss.PinMismatch) as e:
        ss.verify_pins(scan, info.session_id, source_fp=SRC_FP)
    assert e.value.diverged == "preseg"


def test_corrupt_session_listed_not_hidden(scan):
    a = ss.create_session(scan, name="a", preseg_id=None, n_points=8, source_fp=SRC_FP)
    scan.session(a.session_id).session_json.write_text("{broken")
    infos = ss.list_sessions(scan)
    assert infos[0].corrupt is True
```

- [ ] **Step 4: Run, verify fail** — `pytest backend/tests/test_session_store.py -v` → FAIL (module missing).

- [ ] **Step 5: Implement** — `backend/labeling/session_store.py`:

```python
"""sessions/<session_id>/ store (scan-schema v2).

Owns session CRUD, the immutable preseg/source pins frozen at creation, and
default-session resolution (last worked). Pure I/O over ScanLayout paths;
the in-memory SegmentSession is constructed by app.core, not here.
"""
from __future__ import annotations

import json
import re
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from labeling.segment_io import (atomic_write_json, compute_fingerprint,
                                 load_session_aux, save_session_aux)
from preseg.preseg_store import load_preseg
from scenes.scan_layout import ScanLayout

_ID_RE = re.compile(r"^[a-z0-9_-]+$")


class PinMismatch(Exception):
    """A session's frozen pin no longer matches what is on disk."""

    def __init__(self, diverged: str, detail: str) -> None:
        super().__init__(detail)
        self.diverged = diverged  # "preseg" | "source"


@dataclass(frozen=True)
class SessionInfo:
    session_id: str
    name: str
    preseg_id: Optional[str]
    created_at: Optional[str]
    saved_at: Optional[str]
    dirty: bool
    has_output: bool
    corrupt: bool = False


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def create_session(layout: ScanLayout, *, name: str, preseg_id: Optional[str],
                   n_points: int, source_fp: str) -> SessionInfo:
    """Seed working arrays from prelabel/<preseg_id>/ (or all -1 for blank)
    and freeze both fingerprint pins. The only moment a preseg is chosen."""
    if preseg_id is not None and not _ID_RE.match(preseg_id):
        raise ValueError(f"preseg_id {preseg_id!r} must match {_ID_RE.pattern}")
    if preseg_id is not None:
        class_ids, instance_ids = load_preseg(layout, preseg_id, n_points)
        preseg_fp = compute_fingerprint(instance_ids)
    else:
        class_ids = np.full(n_points, -1, dtype=np.int8)
        instance_ids = np.full(n_points, -1, dtype=np.int32)
        preseg_fp = None
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    session_id = f"{ts}_{preseg_id or 'blank'}"
    sp = layout.session(session_id)
    if sp.dir.exists():
        raise FileExistsError(f"session {session_id} already exists")
    created_at = _now()
    aux = {
        "preseg_id": preseg_id,
        "preseg_fingerprint": preseg_fp,
        "source_fingerprint": source_fp,
        "hidden_inst_ids": [],
        "is_from_prelabel": preseg_id is not None,
        "dirty": False,
        "name": name,
        "created_at": created_at,
    }
    save_session_aux(sp.dir, aux, class_ids=class_ids, instance_ids=instance_ids)
    return SessionInfo(session_id=session_id, name=name, preseg_id=preseg_id,
                       created_at=created_at, saved_at=created_at,
                       dirty=False, has_output=False)


def list_sessions(layout: ScanLayout) -> list[SessionInfo]:
    root = layout.sessions_root
    if not root.is_dir():
        return []
    out: list[SessionInfo] = []
    for d in sorted(p for p in root.iterdir() if p.is_dir()):
        sp = layout.session(d.name)
        aux = load_session_aux(sp.dir)
        if aux is None:
            out.append(SessionInfo(session_id=d.name, name=d.name,
                                   preseg_id=None, created_at=None,
                                   saved_at=None, dirty=False,
                                   has_output=sp.output_gt_class_ids.exists(),
                                   corrupt=True))
            continue
        out.append(SessionInfo(
            session_id=d.name,
            name=aux.get("name") or d.name,
            preseg_id=aux.get("preseg_id"),
            created_at=aux.get("created_at"),
            saved_at=aux.get("saved_at"),
            dirty=bool(aux.get("dirty", False)),
            has_output=sp.output_gt_class_ids.exists(),
        ))
    return out


def rename_session(layout: ScanLayout, session_id: str, name: str) -> None:
    sp = layout.session(session_id)
    aux = load_session_aux(sp.dir)
    if aux is None:
        raise FileNotFoundError(f"session {session_id}: no readable session.json")
    aux["name"] = name
    atomic_write_json(sp.session_json, aux)


def delete_session(layout: ScanLayout, session_id: str) -> None:
    sp = layout.session(session_id)
    if not sp.dir.is_dir():
        raise FileNotFoundError(f"session {session_id} not found")
    shutil.rmtree(sp.dir)


def touch_saved_at(layout: ScanLayout, session_id: str) -> None:
    """Bump saved_at (used by /api/load resume so last_worked tracks usage)."""
    sp = layout.session(session_id)
    aux = load_session_aux(sp.dir)
    if aux is None:
        raise FileNotFoundError(f"session {session_id}: no readable session.json")
    aux["saved_at"] = _now()
    atomic_write_json(sp.session_json, aux)


def last_worked(layout: ScanLayout) -> Optional[str]:
    """session_id with max saved_at (created_at fallback); None if no
    non-corrupt sessions exist."""
    infos = [i for i in list_sessions(layout) if not i.corrupt]
    if not infos:
        return None
    return max(infos, key=lambda i: (i.saved_at or i.created_at or "")).session_id


def verify_pins(layout: ScanLayout, session_id: str, *, source_fp: str) -> dict:
    """Enforce the immutable pins before resume. Returns the aux dict on
    success so the caller doesn't re-read it."""
    sp = layout.session(session_id)
    aux = load_session_aux(sp.dir)
    if aux is None:
        raise FileNotFoundError(f"session {session_id}: no readable session.json")
    pinned_src = aux.get("source_fingerprint")
    if pinned_src and pinned_src != source_fp:
        raise PinMismatch("source", (
            f"session {session_id} is pinned to source {pinned_src} "
            f"but the loaded cloud is {source_fp}"))
    preseg_id = aux.get("preseg_id")
    if preseg_id is not None:
        p = layout.preseg_dir(preseg_id) / "instance_ids.npy"
        if not p.exists():
            raise PinMismatch("preseg", (
                f"session {session_id} is pinned to preseg '{preseg_id}' "
                f"which no longer exists at {p}"))
        current_fp = compute_fingerprint(np.load(p).astype(np.int32))
        if aux.get("preseg_fingerprint") != current_fp:
            raise PinMismatch("preseg", (
                f"session {session_id}: preseg '{preseg_id}' content changed "
                f"(pinned {aux.get('preseg_fingerprint')}, now {current_fp})"))
    return aux
```

- [ ] **Step 6: Run, verify pass** — `pytest backend/tests/test_session_store.py backend/tests/test_segment_io.py backend/tests/test_segment_state.py -v` → PASS.
- [ ] **Step 7: Commit** — `git commit -m "feat: session_store with fingerprint pins; session.json schema v2"`

---

### Task 4: v2 read path — registry gating, lidar_io, load route, core resume

This is the switchover task: the loader stops reading `labels/`/`prelabel/` and starts resolving sessions. The conftest fixture moves to v2 **in the same task** because they cannot land separately and stay green.

**Files:**
- Modify: `backend/scenes/scene_registry.py` (`_discover_annotated`: require `str(meta.get("schema_version","")).startswith("2")`, else `logging.info("skipping %s: scan-schema %s != 2.x — run scripts/migrate_scan_v2.py", ...)` and `continue`; `has_labels` → `bool(list_sessions(...))`-free cheap check: `ScanLayout(sd).sessions_root.is_dir()`; drop `gt_*`/`labels_dir` extras keys; `session_dir=_session_dir_for(...)` for annotated becomes `lay.sessions_root`)
- Modify: `backend/scenes/lidar_io.py` (`load_annotated`: delete the `labels/gt_*` block (lines ~123–172) and the `load_prelabel` fallback; always return `labels=None, is_from_prelabel=False`; keep palette/meta handling)
- Modify: `backend/app/core.py` (replace `_seed_or_recover_session` with `_resume_session`; delete `_stale_prelabel_check`)
- Modify: `backend/routes/load.py` (session resolution; delete `keep_prev_seg` block lines 43–56 and `prefer_prelabel` plumbing; 409 on `PinMismatch`)
- Modify: `backend/app/schemas.py` (`LoadRequest`: drop `prefer_prelabel`, add `session_id: Optional[str] = None`; `LoadResponse`: add `session_id: Optional[str] = None` and `sessions: list[dict] = []`; `SegmentStateResponse`: rename `preseg_run_id` → `preseg_id`, drop `stale_prelabel`)
- Modify: `backend/routes/segment.py` (drop `stale_prelabel=` kwarg, use `preseg_id=seg.preseg_id`)
- Modify: `backend/tests/conftest.py` (v2 fixture below)
- Modify: failing tests in `test_load_endpoint.py`, `test_lidar_io.py`, `test_scene_registry.py`, `test_segment_endpoints.py` (mechanical updates to v2 expectations)

- [ ] **Step 1: conftest → v2.** Replace the `labels/` block of `build_annotated_root` with:

```python
    # v2: preseg result + one session seeded from it (worked = has saved_at)
    from preseg.preseg_store import register_preseg
    from labeling.session_store import create_session
    from labeling.segment_io import compute_fingerprint
    from scenes.scan_layout import ScanLayout
    from plyfile import PlyData
    lay = ScanLayout(scan_dir)
    inst = np.array([-1, 0, 0, 1, 1, 2, -1, 3], dtype=np.int32)
    register_preseg(lay, "ransac", inst,
                    summary={"segments": [{"id": i, "class_id": c} for i, c in
                             [(0, 0), (1, 1), (2, 2), (3, 2)]]},
                    generator="ransac", params={})
    ply = PlyData.read(str(scan_dir / "source" / "scan.ply"))["vertex"]
    pts = np.stack([ply["x"], ply["y"], ply["z"]], axis=1).astype(np.float32)
    source_fp = compute_fingerprint(pts)   # NOTE: must match the route's
    # fingerprint of the *recentered* cloud; the 8-pt fixture is near origin
    # so _recenter is a no-op (bbox << 1e3) and raw == recentered.
    sess = create_session(lay, name="demo session", preseg_id="ransac",
                          n_points=8, source_fp=source_fp)
```

and update `meta.json` to include `"schema_version": "2.0", "class_map_version": 1`. Make `build_annotated_root` return `(root, sess.session_id)` and fix the two fixtures that call it. **Gotcha:** the route computes `source_fp` from the cloud after `_z_up_to_y_up`; the fixture meta must therefore set `"source_mesh": true` (no `source_laz`) so `is_z_up=False` and no rotation is applied, keeping fingerprints equal.

- [ ] **Step 2: lidar_io + registry edits** as listed above. Run `pytest backend/tests/test_lidar_io.py backend/tests/test_scene_registry.py -v`, update assertions (e.g. labels now always None from `load_annotated`; v1.3 fixture scans must be skipped by discovery — add an explicit test:

```python
def test_v13_scan_not_discovered(tmp_path, caplog):
    # build a dir with meta schema_version "1.3" → absent from discovery
```

- [ ] **Step 3: core `_resume_session`.** Replace `_seed_or_recover_session` (core.py:251-300) with:

```python
def _resume_session(scan_dir: Path, session_id: str, pc, source_fp: str):
    """Resume one on-disk session: verify pins (PinMismatch propagates to the
    route → 409), load working arrays, build the in-memory SegmentSession.
    Fails loudly on unreadable/misshapen arrays — never silently blank."""
    from labeling.segment_state import SegmentSession
    from labeling.segment_io import load_working_arrays
    from labeling.session_store import verify_pins
    from preseg.preseg_store import load_preseg
    from scenes.scan_layout import ScanLayout

    lay = ScanLayout(scan_dir)
    aux = verify_pins(lay, session_id, source_fp=source_fp)
    sp = lay.session(session_id)
    wa = load_working_arrays(sp.dir, n_points=len(pc))
    if wa is None:
        raise HTTPException(409, f"session {session_id}: working arrays "
                                 f"missing or wrong shape for this cloud")
    seg = SegmentSession(class_ids=wa[0], instance_ids=wa[1],
                         positions=pc.points,
                         is_from_prelabel=bool(aux.get("is_from_prelabel", False)),
                         session_dir=sp.dir)
    seg.source_fingerprint = source_fp
    seg.preseg_id = aux.get("preseg_id")
    seg.preseg_fingerprint = aux.get("preseg_fingerprint")
    seg.name = aux.get("name") or session_id
    seg.created_at = aux.get("created_at")
    seg.hidden_inst_ids = set(int(x) for x in aux.get("hidden_inst_ids", []))
    seg.dirty = bool(aux.get("dirty", False))
    if seg.preseg_id is not None:
        _, pre_ii = load_preseg(lay, seg.preseg_id, n_points=len(pc))
        seg.preseg_ids = pre_ii          # immutable preseg layer for snap-to
    return seg
```

Delete `_stale_prelabel_check` entirely (grep `stale_prelabel` across backend+frontend and remove usages — frontend has a banner keyed on it; remove that too).

- [ ] **Step 4: load route rework.** In `routes/load.py`, after `_recenter` + `source_fp` computation and **before** building the response:

```python
    sessions_meta: list[dict] = []
    session_id = None
    seg = None
    if src.tier == "annotated":
        from dataclasses import asdict
        from labeling.session_store import PinMismatch, last_worked, list_sessions, touch_saved_at
        scan_dir = Path(src.extras["scan_dir"])
        lay_sessions = list_sessions(ScanLayout(scan_dir))
        sessions_meta = [asdict(s) for s in lay_sessions]
        session_id = req.session_id or last_worked(ScanLayout(scan_dir))
        if req.session_id and req.session_id not in {s.session_id for s in lay_sessions}:
            raise HTTPException(404, f"session {req.session_id!r} not found")
        if session_id is not None:
            try:
                seg = _resume_session(scan_dir, session_id, pc, source_fp)
            except PinMismatch as e:
                raise HTTPException(status_code=409, detail={
                    "error": "session_pin_mismatch",
                    "diverged": e.diverged,
                    "session_id": session_id,
                    "message": str(e)})
            touch_saved_at(ScanLayout(scan_dir), session_id)
    _state["seg"] = seg
    _state["session_id"] = session_id
    if seg is not None:
        labels = LabelArrays(class_ids=seg.class_ids, instance_ids=seg.instance_ids)
```

Notes for the implementer: `labels` then feeds the existing `_filter_tiny_segments` → **move that filter to `create_session` seeding instead** (tiny-segment cleanup is a seed-time concern; a resumed session must round-trip byte-identical) — i.e. in `session_store.create_session`, after `load_preseg`, apply the same `MIN_SEGMENT_POINTS` filter (import threshold via parameter `min_segment_points: int = 0` passed by the route layer; keep session_store env-free). `n_classes/n_instances/n_labeled` recompute from `seg` arrays. Delete the `prev_scene/prev_seg/keep_prev_seg` block and `prefer_prelabel`. `LoadResponse` gains `session_id=session_id, sessions=sessions_meta`. The subsampled `class_ids`/`instance_ids` payload comes from `labels` exactly as today.

- [ ] **Step 5: Run the endpoint suites** — `pytest backend/tests/test_load_endpoint.py backend/tests/test_segment_endpoints.py backend/tests/test_smoke.py -v`. Update tests: loads of `annotated/demo` now resume the fixture session (assert `r.json()["session_id"]` endswith `_ransac`); add new tests:

```python
def test_load_explicit_unknown_session_404(client_with_annotated_scene): ...
def test_load_pin_mismatch_409(client_with_annotated_scene):
    # tamper prelabel/ransac/instance_ids.npy after fixture build → 409,
    # body detail["diverged"] == "preseg"
def test_load_scan_without_sessions_returns_empty(client_with_annotated_scene):
    # delete sessions/ dir → 200, sessions == [], session_id is None, class_ids None
```

- [ ] **Step 6: Full backend run** — `pytest backend/tests -x -q`; fix remaining fallout (`test_seg_inference.py`, `test_real_scans_validate.py` may reference v1.3 paths — update or mark the real-scan test to skip when no v2 scans exist). Expected: PASS.
- [ ] **Step 7: Commit** — `git commit -m "feat: v2 read path — session resolution on load, 409 pin enforcement"`

---

### Task 5: save route → active session output/

**Files:**
- Modify: `backend/labeling/segment_io.py::save_labels` — signature becomes `save_labels(scan_dir, session_id, ...)`; writes via `ScanLayout(scan_dir).session(session_id)` paths (`output/gt_*.npy`, `output/gt_segment_metadata.json`), history snapshots under `sp.history_dir` (reuse `prune_history`); registry/meta-version reads unchanged (still scan-level). Rename the metadata key `prelabel_fingerprint` → `preseg_fingerprint`.
- Modify: `backend/routes/segment.py::segment_save` — `session_id = _state.get("session_id")`; 409 if None ("no active session"); pass it through; after success update the response of `/api/segment/state` to include `session_id`.
- Test: `backend/tests/test_segment_io.py`, `backend/tests/test_segment_endpoints.py`

- [ ] **Step 1: Write failing tests** — in `test_segment_endpoints.py`:

```python
def test_save_writes_into_session_output(client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    r = client.put("/api/segment/save")
    assert r.status_code == 200
    sessions = list((scan_dir_for_loaded_scene / "sessions").iterdir())
    out = [s / "output" / "gt_class_ids.npy" for s in sessions if (s / "output").is_dir()]
    assert len(out) == 1 and out[0].exists()
    assert not (scan_dir_for_loaded_scene / "labels").exists()  # v2: no top-level labels/
```

- [ ] **Step 2: Run, verify fail**, **Step 3: implement**, **Step 4: run all segment/save/io tests → PASS**.
- [ ] **Step 5: Commit** — `git commit -m "feat: save writes per-session output/ + history (v2)"`

---

### Task 6: sessions + presegs API routes

**Files:**
- Create: `backend/routes/sessions.py`
- Modify: `backend/main.py` (include router — follow the existing `routes/load.py` include pattern, check `grep -n "include_router" backend/main.py`)
- Modify: `backend/app/schemas.py` (request/response models)
- Test: `backend/tests/test_session_routes.py`

Routes (all resolve the scene via the existing `_resolve`, require tier `annotated`, 409 otherwise):

```python
"""Voxa API routes: labeling sessions + preseg results (scan-schema v2)."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from fastapi import APIRouter, HTTPException

from app.constants import *  # noqa: F401,F403
from app.schemas import *  # noqa: F401,F403
from app.core import *  # noqa: F401,F403

router = APIRouter()


def _annotated_layout(scene_id: str):
    from scenes.scan_layout import ScanLayout
    src = _resolve(scene_id)
    if src.tier != "annotated":
        raise HTTPException(409, "sessions exist only on annotated/<scene>")
    return src, ScanLayout(Path(src.extras["scan_dir"]))


@router.get("/api/scenes/{tier}/{name}/sessions")
def sessions_list(tier: str, name: str):
    from labeling.session_store import list_sessions
    _, lay = _annotated_layout(f"{tier}/{name}")
    return {"sessions": [asdict(s) for s in list_sessions(lay)]}


@router.post("/api/scenes/{tier}/{name}/sessions")
def sessions_create(tier: str, name: str, req: CreateSessionRequest):
    from labeling.session_store import create_session
    src, lay = _annotated_layout(f"{tier}/{name}")
    # n_points + source_fp must describe the cloud as the loader sees it →
    # require the scene to be loaded (same single-cloud model as /segment/*).
    if _state.get("scene") != src.scene_id or _state.get("pc") is None:
        raise HTTPException(409, "load the scene before creating a session")
    from labeling.segment_io import compute_fingerprint
    source_fp = compute_fingerprint(_state["pc"].points.astype(np.float32))
    try:
        info = create_session(lay, name=req.name, preseg_id=req.preseg_id,
                              n_points=len(_state["pc"]), source_fp=source_fp,
                              min_segment_points=MIN_SEGMENT_POINTS)
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(400, str(e))
    return asdict(info)


@router.patch("/api/scenes/{tier}/{name}/sessions/{sid}")
def sessions_rename(tier: str, name: str, sid: str, req: RenameSessionRequest):
    from labeling.session_store import rename_session
    _, lay = _annotated_layout(f"{tier}/{name}")
    try:
        rename_session(lay, sid, req.name)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    return {"ok": True}


@router.delete("/api/scenes/{tier}/{name}/sessions/{sid}")
def sessions_delete(tier: str, name: str, sid: str, confirm: bool = False):
    from labeling.session_store import delete_session
    _, lay = _annotated_layout(f"{tier}/{name}")
    if not confirm:
        raise HTTPException(400, "pass ?confirm=true to delete a session")
    try:
        delete_session(lay, sid)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    if _state.get("session_id") == sid:
        _state["seg"] = None
        _state["session_id"] = None
    return {"ok": True}


@router.get("/api/scenes/{tier}/{name}/presegs")
def presegs_list(tier: str, name: str):
    from preseg.preseg_store import list_presegs
    _, lay = _annotated_layout(f"{tier}/{name}")
    try:
        return {"presegs": [asdict(p) for p in list_presegs(lay)]}
    except ValueError as e:
        raise HTTPException(500, str(e))
```

Schemas: `class CreateSessionRequest(BaseModel): name: str; preseg_id: Optional[str] = None` and `class RenameSessionRequest(BaseModel): name: str`.

- [ ] **Step 1: failing tests** — `test_session_routes.py` covering: list (fixture session present), create-blank, create-with-preseg, create-unknown-preseg → 400, create-without-load → 409, rename, rename-unknown → 404, delete-without-confirm → 400, delete (incl. active → segment/state has_state False), presegs list (1 entry, fingerprint present), all on `client_with_loaded_annotated_scene` / `client_with_annotated_scene`.
- [ ] **Step 2: fail run**, **Step 3: implement + register router**, **Step 4: `pytest backend/tests/test_session_routes.py -v` → PASS**.
- [ ] **Step 5: Commit** — `git commit -m "feat: session CRUD + preseg list endpoints"`

---

### Task 7: migration script

**Files:**
- Create: `scripts/migrate_scan_v2.py`
- Test: `backend/tests/test_migrate_v2.py`

CLI: `python scripts/migrate_scan_v2.py [--dry-run] [--scan NAME ...] LIDAR_ROOT` — iterates `LIDAR_ROOT/annotated/*`. Per scan, **in this order**, refusing loudly (print + nonzero exit, skip remaining steps for that scan) on anything unexpected:

1. Skip if `meta.json` `schema_version` starts with `"2"` (idempotency; print "already v2").
2. Refuse if `sessions/` already exists, or `prelabel/` contains anything other than the two `ransac_*` files, or label/working array shapes mismatch `n_points` read from `source/scan.ply` header.
3. `prelabel/ransac_*` → `prelabel/ransac/{instance_ids.npy, segment_summary.json}` + write `meta.json` via `preseg_store.register_preseg`-equivalent fields (generator `"ransac"`, params `{}`, fingerprint computed; use `register_preseg` on the loaded array, then delete the old flat files).
4. `labels/gt_*` → `sessions/legacy/output/` (move); `session/{current.json,working_*.npy}` → `sessions/legacy/` with `session.json` synthesized: `name="legacy"`, `preseg_id="ransac"` iff the preseg dir exists else `None`, **pins recomputed** from the migrated `prelabel/ransac/instance_ids.npy` and the cloud (load scan.ply positions float32 → `compute_fingerprint`), `created_at`/`saved_at` from file mtimes (`datetime.fromtimestamp(p.stat().st_mtime, tz=timezone.utc)`); if no `session/` existed but `labels/` did, synthesize working arrays from the GT (class → int8).
5. `annotation_history/*` → `sessions/legacy/history/` (move).
6. Remove now-empty `labels/`, `session/`, `annotation_history/` dirs; set `meta.json` `schema_version: "2.0"` (preserve all other keys).
7. `--dry-run` prints the per-scan plan and changes nothing.

Implementation note: import backend modules with `sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))` — same pattern as `scripts/preseg/_common.py` (check and mirror it).

- [ ] **Step 1: failing tests** — `test_migrate_v2.py` with a `build_v13_root(tmp_path)` helper (copy the **old** v1.3 fixture body that Task 4 removed from conftest — labels/, session/current.json, prelabel/ransac_*, meta `"1.3"`). Tests: (a) full migrate → v2 layout asserts (sessions/legacy/output/gt_*.npy exist, session.json pins == recomputed fingerprints, prelabel/ransac/meta.json exists, no labels/ dir, meta schema "2.0"); (b) idempotent second run no-op; (c) labels-without-session synthesizes working arrays equal to GT; (d) refusal: stray file in prelabel/ → scan skipped, exit code != 0, disk untouched; (e) `--dry-run` changes nothing.
- [ ] **Step 2: fail run**, **Step 3: implement**, **Step 4: `pytest backend/tests/test_migrate_v2.py -v` → PASS**.
- [ ] **Step 5: Commit** — `git commit -m "feat: migrate_scan_v2 one-shot v1.3→v2 migration"`

---

### Task 8: offline preseg pipeline writes via register_preseg

**Files:**
- Modify: `scripts/preseg/presegment.py` (and `presegment_sam3.py` if it writes `prelabel/ransac_*` — `grep -n "ransac_\|prelabel" scripts/preseg/*.py` first)

- [ ] **Step 1:** Read the scripts' output sections; replace direct `prelabel/ransac_*` writes with `preseg_store.register_preseg(layout, preseg_id, ...)` where `preseg_id` comes from a new `--preseg-id` CLI arg (default `ransac` for the RANSAC script, `sam3_<runid>` for the SAM3 one), `generator`/`params` filled from the script's own config.
- [ ] **Step 2:** `pytest backend/tests/test_presegment.py -v` (update if it asserts old paths) → PASS.
- [ ] **Step 3: Commit** — `git commit -m "feat: preseg pipelines publish via register_preseg"`

---

### Task 9: frontend api.js

**Files:**
- Modify: `frontend/src/api.js`
- Test: `frontend/src/api.test.js` (extend)

- [ ] **Step 1: failing vitest** — mapper tests for the new payloads (pure-function only, matching the existing test style):

```js
import { decodeLoadResponse } from './api.js';
// assert sessions/session_id pass through; sessionId option lands in body (test the body builder if present, else skip fetch-dependent parts)
```

- [ ] **Step 2: implement** — `load(name, { sessionId = null, ... })` adds `...(sessionId != null ? { session_id: sessionId } : {})`; `decodeLoadResponse` gains `sessionId: j.session_id ?? null, sessions: j.sessions || []`. New methods (mirror existing fetch style, `throw` on `!r.ok`):

```js
  async listSessions(scene) { const r = await fetch(`/api/scenes/${scene}/sessions`); ... },
  async createSession(scene, { name, presegId = null }) { /* POST {name, preseg_id} */ },
  async renameSession(scene, sid, name) { /* PATCH */ },
  async deleteSession(scene, sid) { /* DELETE ?confirm=true */ },
  async listPresegs(scene) { /* GET */ },
```

**409 surfacing:** in `load()`, on `r.status === 409` parse the JSON body and throw an `Error` with `.detail` attached so App.jsx can render the pin-mismatch banner with `diverged`/`message`.

- [ ] **Step 3:** `npx vitest run frontend/src/api.test.js --root frontend` → PASS.
- [ ] **Step 4: Commit** — `git commit -m "feat: api.js session/preseg client"`

---

### Task 10: frontend — session picker + App wiring

**Files:**
- Create: `frontend/src/session-picker.jsx`
- Modify: `frontend/src/App.jsx`, `frontend/src/mode-label.jsx`

No component-test infra exists (vitest env `node`) — this task is verified in the browser (Task 12). Keep components small and dumb; all data flows through props from App.

- [ ] **Step 1: App.jsx state + load flow.** Add `const [sessions, setSessions] = useStateApp([]); const [activeSessionId, setActiveSessionId] = useStateApp(null); const [pinError, setPinError] = useStateApp(null);`. In the scene-load effect (App.jsx:169), pass `{ sessionId: activeSessionId }` to `VoxaAPI.load` only when the user explicitly picked one (a ref flag); on success `setSessions(c.sessions); setActiveSessionId(c.sessionId);` — the backend already defaults to last-worked. Catch the 409 pin error → `setPinError(err.detail)` and render a blocking banner (reuse the existing `loadError` banner styling — find it via `grep -n "loadError" frontend/src/App.jsx`) showing `detail.message` + which pin diverged; scene stays unloaded. Reset `activeSessionId` to null on `activeScene` change (each scan resumes its own last-worked).
- [ ] **Step 2: session-picker.jsx.** Props: `{ sessions, activeSessionId, presegs, dirty, onSelect(sid), onCreate({name, presegId}), onRename(sid, name), onDelete(sid), onRefreshPresegs }`. Render: row per session (display name, preseg badge `preseg_id ?? 'blank'`, `saved_at` short date, dirty dot, corrupt rows greyed with "corrupt" tag and no select), inline rename (pencil → text input), delete (window.confirm), `+ New session` expands name input + preseg `<select>` (options from presegs + "blank") + Create button. Style: match the existing panels in `mode-label.jsx` (copy class names from the segments panel — `grep -n "panel" frontend/src/mode-label.jsx | head`).
- [ ] **Step 3: mode-label.jsx.** Mount `<SessionPicker>` above the class panel **only when** the active scene tier is `annotated`. Handlers call `VoxaAPI.*` then re-fetch `listSessions`; `onSelect` with `dirty` → `window.confirm("Unsaved changes are autosaved; switch session?")` then sets the explicit-pick ref + `setActiveSessionId(sid)` (triggers reload via the effect); `onCreate` → `createSession` then select it. When a scan has zero sessions, the picker renders in create mode with a hint ("No sessions yet — create one to start labeling") — segState is null so the canvas is read-only, which matches backend behavior.
- [ ] **Step 4:** `npm run test:frontend` → PASS (no regressions); `npm run build` → builds clean.
- [ ] **Step 5: Commit** — `git commit -m "feat: session picker UI + App session wiring"`

---

### Task 11: docs + sibling-repo companion change

**Files:**
- Modify: `docs/scan-schema.md` — rewrite layout tree, REQUIRES-vs-nice-to-have table, invariants, and file-contract sections to v2 (use the spec's layout block as the source of truth; bump title to v2.0; document `session.json`, preseg `meta.json`, migration pointer).
- Modify: `CLAUDE.md` — update the "Per-point labels", "Scan directory schema", data-layout, and gotcha bullets that reference `labels//session//prelabel/ransac_*`; mention session picker + 409 pin semantics.
- Modify: `README.md` if it shows the scan layout (`grep -n "labels/\|prelabel" README.md`).
- **Sibling repo** `/home/hendrik/coding/engine/data`: update `tools/scaffold_annotation.py` to emit the v2 skeleton (`prelabel/` empty, `sessions/` absent, meta `"schema_version": "2.0"`) and `lidar/SCHEMA.md` to v2. **This is a separate git repo** — commit there on a branch (`git -C /home/hendrik/coding/engine/data checkout -b feat/scan-schema-v2`), do NOT mix into the voxa branch. If that repo has uncommitted changes, stop and surface to the human instead of touching it.

- [ ] Steps: write docs → `pytest backend/tests -q` still green (docs only) → commit voxa docs (`docs: scan-schema v2`); separately commit the data-repo change.

---

### Task 12: cleanup, full verification

- [ ] **Step 1: delete v1.3 accessors.** Remove `labels_dir`, `gt_class_ids`, `gt_segment_ids`, `gt_segment_metadata`, `prelabel_dir`, `ransac_instance_ids`, `ransac_segment_summary`, `session_dir`, `annotation_history_dir` from `ScanLayout`; remove `segment_io.load_prelabel` (superseded by `preseg_store.load_preseg`). Then `grep -rn "ransac_instance_ids\|ransac_segment_summary\|annotation_history_dir\|load_prelabel\|prelabel_fingerprint\|stale_prelabel\|preseg_run_id\|prefer_prelabel\|keep_prev_seg\|current\.json" backend/ frontend/src/ scripts/ --include="*.py" --include="*.js" --include="*.jsx"` → only `migrate_scan_v2.py` (which legitimately reads legacy names) may match.
- [ ] **Step 2: full test run** — `npm test` (vitest + pytest). Expected: all green; paste the summary into the commit/PR notes.
- [ ] **Step 3: live migration + browser verification** (@browser-verification skill). Run `python scripts/migrate_scan_v2.py --dry-run /home/hendrik/coding/engine/data/lidar`, review, then real run (the archive is shared — announce before running). Restart `npm run dev`. In the browser: open an annotated scene → last-worked session auto-resumes; create a second session from a different preseg (or blank); switch between them; rename; delete; tamper-test the 409 path is NOT needed live (covered by tests). Screenshot picker + loaded session; zero console errors; network calls 200.
- [ ] **Step 4: Commit** — `git commit -m "chore: drop scan-schema v1.3 accessors + verification"`

---

## Execution notes

- Tasks 1–3 are independent of each other only in part (3 depends on 2); execute in order.
- Task 4 is the big switchover — budget the most review attention there.
- The shared lidar archive migration (Task 12 step 3) mutates real data: `--dry-run` first, and the archive has scans listed in `data/lidar/annotated/` — `smart_ais_clean` has a fresh preseg the user cares about; verify it survives as `prelabel/ransac/`.
- After implementation completes, per the user's global workflow rule, run the `simplify` skill on the diff before finishing.
