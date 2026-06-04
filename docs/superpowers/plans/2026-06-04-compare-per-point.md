# Per-Point Compare View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the legacy cuboid Compare view with per-point comparison of two finished labelings (session outputs / presegs): agreement %, per-class IoU/P/R, confusion pairs, and a class-colored synced split view.

**Architecture:** A pure-numpy metrics module (`labeling/compare_points.py`) + one disk-reading route (`POST /api/compare-points/{tier}/{name}`) that returns metrics and both full-res class arrays; the frontend reworks `mode-compare.jsx` to two source dropdowns + the existing synced split view colored via the viewer's existing `colorMode='class'` path. The cuboid compare endpoint, its schemas, and `_iou_aabb` are deleted.

**Tech Stack:** FastAPI + numpy, pytest; React 18 + Three.js, vitest.

**Spec:** `docs/superpowers/specs/2026-06-04-compare-per-point-design.md` — the contract; read it first.

**Worktree:** all work in `/home/hendrik/coding/engine/tools/labeling/voxa/.claude/worktrees/compare-points`, branch `feat/compare-per-point`. Backend tests: `/home/hendrik/coding/engine/tools/labeling/voxa/.venv/bin/pytest backend/tests --tb=short -q --junitxml=/tmp/jc.xml` then grep the xml (a plugin swallows the terminal summary; trust exit code + xml). Currently 234 tests, 0 failures. Frontend: `npm run test:frontend` (40 green) + `npm run build`.

---

## File structure (end state)

```
backend/
├── labeling/compare_points.py    NEW       pure-numpy metrics (no FastAPI, no _state)
├── routes/compare.py             MODIFIED  cuboid /api/compare deleted; /api/compare-points added; annotations GET/PUT untouched
├── app/schemas.py                MODIFIED  CompareRequest/DiffRow/CompareResponse deleted; SourceRef/ComparePointsRequest added
├── app/core.py                   MODIFIED  _iou_aabb deleted (compare route was its only caller)
├── scenes/lidar_io.py            MODIFIED  public build_class_palette() wrapper (compare route reuses the load palette)
└── tests/
    ├── test_compare_points.py    NEW       unit + route tests
    └── test_compare.py           DELETED   (tested the removed cuboid endpoint)
frontend/src/
├── api.js                        MODIFIED  compare() → comparePoints(); decoder
├── api.test.js                   MODIFIED  decoder tests
├── mode-compare.jsx              REWRITTEN source dropdowns + per-class table + class-colored split view
├── App.jsx                       MODIFIED  CompareMode props (sessions/presegs/activeSessionId instead of gt/pred instances)
└── app.css                       MODIFIED  cmp-bar source-select styles; per-class table tweaks
CLAUDE.md, docs/scan-schema.md    MODIFIED  Compare description; predictions-as-presegs note
```

Conventions: `-1` = unlabeled; class arrays int8 on the wire (b64, same as `LoadResponse.class_ids`); fail loudly; scene ids embedded raw in URLs (slash intact).

---

### Task 1: compare_points metrics module

**Files:**
- Create: `backend/labeling/compare_points.py`
- Test: `backend/tests/test_compare_points.py` (unit section)

- [ ] **Step 1: Write failing unit tests** — `backend/tests/test_compare_points.py`:

```python
"""Tests for per-point comparison: metrics module + /api/compare-points route."""
from __future__ import annotations

import numpy as np
import pytest

from labeling.compare_points import compare_class_arrays


def test_agreement_excludes_both_unlabeled():
    a = np.array([-1, -1, 0, 0, 1], dtype=np.int8)
    b = np.array([-1,  0, 0, 1, 1], dtype=np.int8)
    m = compare_class_arrays(a, b)
    # 4 points labeled in at least one side; matches among them: idx2 (0==0), idx4 (1==1)
    assert m["n_points"] == 5
    assert m["n_labeled_a"] == 3
    assert m["n_labeled_b"] == 4
    assert m["agreement"] == pytest.approx(2 / 4)
    assert m["agreement_all"] == pytest.approx(3 / 5)  # idx0 (-1==-1) counts here


def test_per_class_iou_precision_recall():
    a = np.array([0, 0, 0, 1, -1, -1], dtype=np.int8)
    b = np.array([0, 0, 1, 1,  1, -1], dtype=np.int8)
    m = compare_class_arrays(a, b)
    per = {c["class_id"]: c for c in m["per_class"]}
    # class 0: tp=2, union=3 → iou 2/3; precision=2/2 (B claims 2, both match);
    # recall=2/3 (A has 3)
    assert per[0]["iou"] == pytest.approx(2 / 3)
    assert per[0]["precision"] == pytest.approx(1.0)
    assert per[0]["recall"] == pytest.approx(2 / 3)
    assert per[0]["n_a"] == 3 and per[0]["n_b"] == 2
    # class 1: tp=1, union=4 → 0.25; precision=1/3; recall=1/1
    assert per[1]["iou"] == pytest.approx(0.25)
    assert per[1]["precision"] == pytest.approx(1 / 3)
    assert per[1]["recall"] == pytest.approx(1.0)


def test_per_class_zero_division_is_null():
    a = np.array([0, 0], dtype=np.int8)
    b = np.array([-1, -1], dtype=np.int8)
    per = {c["class_id"]: c for c in compare_class_arrays(a, b)["per_class"]}
    assert per[0]["precision"] is None   # B claims nothing for class 0
    assert per[0]["recall"] == pytest.approx(0.0)
    assert per[0]["iou"] == pytest.approx(0.0)


def test_confusion_pairs_sorted_and_truncated():
    # 3x (0→1), 2x (1→2), 1x (2→0); unlabeled-on-either-side never appears
    a = np.array([0, 0, 0, 1, 1, 2, -1, 0], dtype=np.int8)
    b = np.array([1, 1, 1, 2, 2, 0, 1, -1], dtype=np.int8)
    m = compare_class_arrays(a, b)
    assert m["confusion"][0] == {"a_class": 0, "b_class": 1, "n": 3}
    assert m["confusion"][1] == {"a_class": 1, "b_class": 2, "n": 2}
    assert m["confusion"][2] == {"a_class": 2, "b_class": 0, "n": 1}
    assert len(m["confusion"]) == 3


def test_length_mismatch_raises():
    with pytest.raises(ValueError, match="length"):
        compare_class_arrays(np.zeros(3, dtype=np.int8), np.zeros(4, dtype=np.int8))
```

- [ ] **Step 2: Run, verify fail** — `pytest backend/tests/test_compare_points.py -v` → FAIL (module missing).

- [ ] **Step 3: Implement** — `backend/labeling/compare_points.py`:

```python
"""Per-point comparison of two class labelings (scan-schema v2 Compare).

Pure numpy. No FastAPI, no in-memory state. Both inputs are int class
arrays of equal length; -1 means unlabeled.
"""
from __future__ import annotations

import numpy as np

CONFUSION_TOP_N = 20


def compare_class_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    """Agreement, per-class IoU/precision/recall (A as reference), and the
    top disagreeing class pairs. ``agreement`` is computed over points
    labeled in AT LEAST ONE side — both-unlabeled points carry no signal
    and would inflate agreement on sparse labelings; ``agreement_all``
    keeps them for reference."""
    if a.shape != b.shape:
        raise ValueError(f"length mismatch: a has {a.shape[0]}, b has {b.shape[0]}")
    a = a.astype(np.int32, copy=False)
    b = b.astype(np.int32, copy=False)
    n = int(a.shape[0])
    labeled_a = a >= 0
    labeled_b = b >= 0
    either = labeled_a | labeled_b
    eq = a == b

    out: dict = {
        "n_points": n,
        "n_labeled_a": int(labeled_a.sum()),
        "n_labeled_b": int(labeled_b.sum()),
        "agreement": float(eq[either].mean()) if either.any() else None,
        "agreement_all": float(eq.mean()) if n else None,
    }

    classes = np.union1d(np.unique(a[labeled_a]), np.unique(b[labeled_b]))
    per_class = []
    for c in classes.tolist():
        in_a = a == c
        in_b = b == c
        tp = int((in_a & in_b).sum())
        union = int((in_a | in_b).sum())
        n_a = int(in_a.sum())
        n_b = int(in_b.sum())
        per_class.append({
            "class_id": int(c),
            "iou": tp / union if union else None,
            "precision": tp / n_b if n_b else None,
            "recall": tp / n_a if n_a else None,
            "n_a": n_a,
            "n_b": n_b,
        })
    out["per_class"] = per_class

    # Confusion: both labeled, classes differ. Vectorized pair counting via a
    # single unique() over packed (a, b) pairs — no Python loop over points.
    both = labeled_a & labeled_b & ~eq
    if both.any():
        pairs = a[both].astype(np.int64) * 100_000 + b[both].astype(np.int64)
        uniq, counts = np.unique(pairs, return_counts=True)
        order = np.argsort(counts)[::-1][:CONFUSION_TOP_N]
        out["confusion"] = [
            {"a_class": int(uniq[i] // 100_000), "b_class": int(uniq[i] % 100_000),
             "n": int(counts[i])}
            for i in order
        ]
    else:
        out["confusion"] = []
    return out
```

- [ ] **Step 4: Run, verify pass** — `pytest backend/tests/test_compare_points.py -v` → PASS (5 tests).
- [ ] **Step 5: Commit** — `git add -A && git commit -m "feat: per-point comparison metrics module"`

---

### Task 2: /api/compare-points route + cuboid compare removal

**Files:**
- Modify: `backend/routes/compare.py` (delete the `POST /api/compare/{scene:path}` handler; keep annotations GET/PUT and `/api/auto-fit`; add the new route)
- Modify: `backend/app/schemas.py:117-145` (delete `CompareRequest`, `DiffRow`, `CompareResponse`; add `SourceRef`, `ComparePointsRequest`)
- Modify: `backend/app/core.py` (delete `_iou_aabb` — `grep -rn "_iou_aabb" backend/` first; the compare route must be its only caller)
- Modify: `backend/scenes/lidar_io.py` (add public wrapper near `_build_palette`):

```python
def build_class_palette(lidar_root, segment_meta: dict | None = None) -> list[ClassPaletteEntry]:
    """Public palette entry point for routes that don't go through
    load_annotated (e.g. /api/compare-points)."""
    return _build_palette(_read_classes_json(lidar_root), segment_meta or {})
```

- Delete: `backend/tests/test_compare.py`
- Test: `backend/tests/test_compare_points.py` (route section appended)

New schemas (match the file's style):

```python
class SourceRef(BaseModel):
    kind: str                 # "session" | "preseg"
    id: str

class ComparePointsRequest(BaseModel):
    a: SourceRef
    b: SourceRef
```

New route in `backend/routes/compare.py`:

```python
@router.post("/api/compare-points/{tier}/{name}")
def compare_points(tier: str, name: str, req: ComparePointsRequest):
    """Compare two finished per-point labelings of one scan. Reads both
    sources from disk — no dependency on the in-memory loaded scene."""
    from labeling.compare_points import compare_class_arrays
    from scenes.scan_layout import ScanLayout

    src = _resolve(f"{tier}/{name}")
    if src.tier != "annotated":
        raise HTTPException(409, "compare-points needs an annotated/<scene> scan")
    lay = ScanLayout(Path(src.extras["scan_dir"]))

    # expected_n for load_preseg's shape check: meta n_points when present,
    # else the first-loaded source's length (the cross-source check below is
    # the invariant that matters). The cloud is never loaded.
    meta_n = src.n_points

    def load_source(ref: SourceRef, expected_n):
        if ref.kind == "session":
            sp = lay.session(ref.id)
            if not sp.dir.is_dir():
                raise HTTPException(404, f"session {ref.id!r} not found")
            if not sp.output_gt_class_ids.exists():
                raise HTTPException(409, (
                    f"session {ref.id!r} has no saved output — save it "
                    f"(Ctrl+S) before comparing"))
            return np.load(sp.output_gt_class_ids).astype(np.int32)
        if ref.kind == "preseg":
            from preseg.preseg_store import load_preseg
            try:
                n = expected_n if expected_n else None
                if n is None:
                    raise HTTPException(409, (
                        "cannot size-check a preseg source without meta "
                        "n_points or a session source — compare a session "
                        "first or fix meta.json"))
                class_ids, _ = load_preseg(lay, ref.id, n_points=int(n))
            except FileNotFoundError as e:
                raise HTTPException(404, str(e))
            except ValueError as e:
                raise HTTPException(400, str(e))
            return class_ids.astype(np.int32)
        raise HTTPException(400, f"unknown source kind {ref.kind!r}")

    a = load_source(req.a, meta_n)
    b = load_source(req.b, meta_n or len(a))
    if a.shape != b.shape:
        raise HTTPException(409, (
            f"sources cover different clouds: a has {a.shape[0]} points, "
            f"b has {b.shape[0]}"))

    from scenes.lidar_io import build_class_palette
    metrics = compare_class_arrays(a, b)
    palette = build_class_palette(constants.LIDAR_ROOT)
    return {
        "metrics": metrics,
        "a_class_ids": _b64(a.astype(np.int8)),
        "b_class_ids": _b64(b.astype(np.int8)),
        "palette": [p.__dict__ for p in palette],
    }
```

Implementation notes: `_resolve`, `_b64`, `Path`, `np`, `constants` come via the file's existing star imports — check what's already in scope (`constants` is imported in routes/load.py as `from app import constants`; mirror it). `src.n_points` is `Optional[int]` from discovery. Ordering subtlety: when BOTH sources are presegs and meta has no n_points, the route 409s with the fix-meta message (acceptable per spec — `expected_n` rule).

Route tests to append to `test_compare_points.py` (use existing conftest fixtures — `client_with_annotated_scene` → `(client, "annotated/demo", session_id)`; the fixture session has NO output yet; `client_with_loaded_annotated_scene` loads it; `scan_dir_for_loaded_scene` is the scan dir):

```python
# ---- route tests ----

def _save_fixture_session(client):
    """Give the fixture session a saved output via the real save route."""
    r = client.put("/api/segment/save")
    assert r.status_code == 200


def test_compare_session_vs_preseg(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    _save_fixture_session(client)
    # discover the session id from the sessions endpoint
    sid = client.get("/api/scenes/annotated/demo/sessions").json()["sessions"][0]["session_id"]
    r = client.post("/api/compare-points/annotated/demo", json={
        "a": {"kind": "session", "id": sid},
        "b": {"kind": "preseg", "id": "ransac"},
    })
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["metrics"]["n_points"] == 8
    assert 0.0 <= body["metrics"]["agreement_all"] <= 1.0
    import base64, numpy as np
    a = np.frombuffer(base64.b64decode(body["a_class_ids"]), dtype=np.int8)
    assert a.shape == (8,)
    assert isinstance(body["palette"], list)


def test_compare_session_vs_session(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    _save_fixture_session(client)
    sid = client.get("/api/scenes/annotated/demo/sessions").json()["sessions"][0]["session_id"]
    import time; time.sleep(1.1)  # session ids are second-resolution
    r = client.post("/api/scenes/annotated/demo/sessions",
                    json={"name": "b side", "preseg_id": "ransac"})
    assert r.status_code == 200
    sid_b = r.json()["session_id"]
    # activate + save the new session so it has output
    assert client.post("/api/load", json={"name": "annotated/demo",
                                          "session_id": sid_b}).status_code == 200
    assert client.put("/api/segment/save").status_code == 200
    r = client.post("/api/compare-points/annotated/demo", json={
        "a": {"kind": "session", "id": sid},
        "b": {"kind": "session", "id": sid_b},
    })
    assert r.status_code == 200, r.text
    assert r.json()["metrics"]["agreement"] is not None


def test_compare_no_output_409(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    sid = client.get("/api/scenes/annotated/demo/sessions").json()["sessions"][0]["session_id"]
    r = client.post("/api/compare-points/annotated/demo", json={
        "a": {"kind": "session", "id": sid},
        "b": {"kind": "preseg", "id": "ransac"},
    })
    assert r.status_code == 409
    assert "no saved output" in r.json()["detail"]


def test_compare_unknown_ids_404(client_with_loaded_annotated_scene):
    client = client_with_loaded_annotated_scene
    _save_fixture_session(client)
    sid = client.get("/api/scenes/annotated/demo/sessions").json()["sessions"][0]["session_id"]
    for body in ({"a": {"kind": "session", "id": "nope"}, "b": {"kind": "preseg", "id": "ransac"}},
                 {"a": {"kind": "session", "id": sid}, "b": {"kind": "preseg", "id": "nope"}}):
        assert client.post("/api/compare-points/annotated/demo", json=body).status_code == 404


def test_compare_length_mismatch_409(client_with_loaded_annotated_scene, scan_dir_for_loaded_scene):
    client = client_with_loaded_annotated_scene
    _save_fixture_session(client)
    sid = client.get("/api/scenes/annotated/demo/sessions").json()["sessions"][0]["session_id"]
    # truncate the saved output to force a length mismatch vs the preseg
    import numpy as np
    p = scan_dir_for_loaded_scene / "sessions" / sid / "output" / "gt_class_ids.npy"
    np.save(p, np.load(p)[:5])
    r = client.post("/api/compare-points/annotated/demo", json={
        "a": {"kind": "session", "id": sid},
        "b": {"kind": "preseg", "id": "ransac"},
    })
    assert r.status_code == 409
    assert "different clouds" in r.json()["detail"]


def test_cuboid_compare_endpoint_is_gone(client):
    assert client.post("/api/compare/legacy/foo").status_code in (404, 405)
```

- [ ] **Step 1:** Append route tests; run → FAIL (404 route missing).
- [ ] **Step 2:** Implement route + schemas + `build_class_palette`; DELETE the cuboid handler, the three schemas, `_iou_aabb` (after the grep), and `backend/tests/test_compare.py`.
- [ ] **Step 3:** `pytest backend/tests --tb=short -q` → full suite green (~234 + 11 new − 5 deleted; report exact). Sweep: `grep -rn "CompareResponse\|DiffRow\|CompareRequest\|_iou_aabb\|/api/compare/" backend/ frontend/src/ --include="*.py" --include="*.js*"` → only `frontend/src/api.js` (fixed in Task 3) and CLAUDE.md (Task 5) may remain; list anything else and fix.
- [ ] **Step 4: Commit** — `git commit -m "feat: /api/compare-points route; drop cuboid compare endpoint"`

---

### Task 3: frontend api client

**Files:**
- Modify: `frontend/src/api.js` (delete `compare()`; add `comparePoints` + `decodeCompareResponse`)
- Test: `frontend/src/api.test.js`

- [ ] **Step 1: failing vitest** (decoder is a pure function — same style as `decodeLoadResponse` tests):

```js
import { decodeCompareResponse } from './api.js';

describe('decodeCompareResponse', () => {
  it('decodes metrics, arrays and palette', () => {
    const enc = (arr) => Buffer.from(Int8Array.from(arr).buffer).toString('base64');
    const out = decodeCompareResponse({
      metrics: { agreement: 0.5, per_class: [], confusion: [] },
      a_class_ids: enc([-1, 0, 1]),
      b_class_ids: enc([0, 0, 1]),
      palette: [{ id: 0, label: 'Pipe', color: '#5b8def' }],
    });
    expect(out.metrics.agreement).toBe(0.5);
    expect(Array.from(out.aClassIds)).toEqual([-1, 0, 1]);
    expect(Array.from(out.bClassIds)).toEqual([0, 0, 1]);
    expect(out.palette[0].label).toBe('Pipe');
  });
});
```

- [ ] **Step 2: implement** in `api.js`:

```js
export function decodeCompareResponse(j) {
  return {
    metrics: j.metrics,
    aClassIds: b64ToInt8(j.a_class_ids),
    bClassIds: b64ToInt8(j.b_class_ids),
    palette: j.palette || [],
  };
}
```

and on `VoxaAPI` (replacing `compare`):

```js
  async comparePoints(scene, a, b) {
    const r = await fetch(`/api/compare-points/${scene}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ a, b }),
    });
    if (!r.ok) {
      let detail = null;
      try { detail = (await r.json()).detail; } catch { /* non-JSON body */ }
      const err = new Error(typeof detail === 'string' ? detail : `compare failed: ${r.status}`);
      err.status = r.status;
      throw err;
    }
    return decodeCompareResponse(await r.json());
  },
```

(409 bodies here carry a plain-string `detail` — surface it as the error message for inline display.)

- [ ] **Step 3:** `npm run test:frontend` → green; `npm run build` → green.
- [ ] **Step 4: Commit** — `git commit -m "feat: comparePoints api client"`

---

### Task 4: Compare mode rework + App wiring

**Files:**
- Rewrite: `frontend/src/mode-compare.jsx`
- Modify: `frontend/src/App.jsx` (CompareMode invocation ~line 693: replace `gtInstances/predInstances` props with `sessions`, `presegs`, `activeSessionId`; the scene-load effect already fetches those for annotated scenes; keep passing `cloud/theme/sceneName/navMode/onNavModeChange`; add `isAnnotated` like LabelMode gets)
- Modify: `frontend/src/app.css` (source-select styles; reuse `.cmp-bar`/`.cmp-table` blocks — adjust the table header to the per-class columns)

Behavior contract for the rewritten `mode-compare.jsx` (keep the existing split-view/camera-sync code — it is good; replace data plumbing, bar content, and table):

1. **Sources list** built from props: sessions → `{kind:'session', id, label: name, disabled: !has_output, hint: 'no output'}` (sorted by saved_at desc); presegs → `{kind:'preseg', id: preseg_id, label: `${preseg_id} · ${generator}`}`.
2. **Defaults** (an effect on scene/sessions change): A = active session if it has output, else first enabled session; B = next distinct enabled source (session then preseg). Fewer than 2 enabled sources → empty state "Need two finished labelings — save a second session or register a model preseg." and no request.
3. **Fetch** on (sceneName, a, b) change: `VoxaAPI.comparePoints(sceneName, a, b)` → state `{cmp, error}`; error → inline message in the bar area (`err.message`), no crash.
4. **Split view**: per side, project the full-res array onto the subsampled cloud (`cloud.subsampleIdx`, same loop as App.jsx ~264) and pass `{...cloud, classIds: subArr, classPalette: cmp.palette}` with `colorMode: 'class'`, `showCuboids: false` (drop `instances`/`visibleInstanceIds`/`cuboidStyle`). Panel titles = the source labels (A green `#10b981`, B blue `#5b8def`).
5. **Bar**: two `<select>`s (A/B) + agreement metric `agr` (tooltip: "over points labeled in at least one source; all-points: X") + `A: n · B: n` labeled counts. Keep NavModeToggle / Sync / CameraPresets / Help (update help text: per-point metrics).
6. **Table**: per-class rows from `cmp.metrics.per_class` joined with palette for name+dot, columns `Class | IoU | P | R | pts A | pts B`, sorted IoU ascending nulls-last; below, confusion list lines `${nameA} → ${nameB} · ${n.toLocaleString()} pts` from `cmp.metrics.confusion` (resolve names via palette, fall back to `cls <id>`).
7. **Tier guard**: `!isAnnotated` → the empty state "Compare needs an annotated scan".
8. No new deps; match file's hook-alias style (`useStateCmp` etc.); keep the component under ~300 lines.

- [ ] **Step 1:** App.jsx prop changes (grep `gtInstances={gtInstances} predInstances` in the CompareMode invocation; sessions/presegs/activeSessionId are in scope). Note: `gtInstances` stays for LabelMode — only CompareMode's props change.
- [ ] **Step 2:** Rewrite mode-compare.jsx per the contract.
- [ ] **Step 3:** CSS: `.cmp-source-select` (compact select, ~160px, colored left border per side) + adjust `.cmp-table-hd`/`.cmp-table-row` grid columns to 6 (check current grid-template in app.css and update).
- [ ] **Step 4:** `npm run test:frontend` + `npm run build` → green. (Browser verification happens in Task 5 — backend + frontend must both be in place.)
- [ ] **Step 5: Commit** — `git commit -m "feat: per-point Compare view (source dropdowns, class-colored split, per-class table)"`

---

### Task 5: docs + full verification

**Files:**
- Modify: `CLAUDE.md` (Project paragraph: Compare description → "Compare (two finished labelings — session outputs or presegs — per-point: agreement, per-class IoU/P/R, confusion)"; check the gotchas section for the stale "IoU is axis-aligned in `_iou_aabb`" bullet — REWORD it to cover only the cuboid `rotation` storage note or delete if obsolete)
- Modify: `docs/scan-schema.md` (one line under prelabel/: model predictions are registered as presegs — `register_preseg(generator="model-…")` — making them selectable both as session seeds and Compare sources)

- [ ] **Step 1:** Docs edits.
- [ ] **Step 2:** Full suites: backend junitxml (expect ~240, 0 failures — report exact), `npm run test:frontend`, `npm run build`.
- [ ] **Step 3:** Browser verification (@browser-verification): serve the worktree frontend (`VOXA_FRONTEND_PORT=5174 VOXA_BACKEND=http://127.0.0.1:8765 npx vite` from the worktree — the main backend on :8765 lacks the new route, so ALSO start a worktree backend: `VOXA_PORT=8767 VOXA_LIDAR_ROOT=/home/hendrik/coding/engine/data/lidar .venv-relative scripts/run.sh` — simpler: run `VOXA_PORT=8767 ... npm run dev` from the worktree with `VOXA_FRONTEND_PORT=5174 VOXA_BACKEND=http://127.0.0.1:8767`, leaving the user's :5173/:8765 untouched). On `annotated/munich_water_pump` (has legacy session + you may save a second session): open Compare → dropdowns populated, disabled no-output sessions visible, split view colored by class per side, per-class table + confusion list render, empty-state on a scan with <2 sources, zero console errors. Screenshot.
- [ ] **Step 4: Commit** — `git commit -m "docs: per-point compare"`

---

## Execution notes

- Tasks 1→2→3→4→5 strictly in order (each depends on the previous).
- The real archive is read-only for this feature except test sessions you create/delete via the UI/API during verification — clean up any you create.
- After implementation, per the user's global workflow rule, run the `simplify` skill on the diff before finishing.
