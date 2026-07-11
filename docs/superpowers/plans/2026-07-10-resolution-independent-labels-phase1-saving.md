# Resolution-Independent Labels — Phase 1 (Saving) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Capture the resolution-independent primitives *at labeling time* — persist the Box OBB and stamp a stable monotonic apply-order `seq` on every instance — so a future export (Phase 2) can materialize labels at any density. No consumer is built in this phase.

**Architecture:** Instances live in `sessions/<id>/instances_gt.json`, created by the frontend and round-tripped through `GET/PUT /api/annotations/{kind}/{scene}?session_id=` (`backend/routes/compare.py`). Two changes: (1) the frontend Box-apply stops discarding its OBB and writes `center/size/rotation` onto the instance; (2) the backend annotations route stamps a monotonic `seq` on any instance missing one, preserving existing values (backfill for old sessions). The `Cuboid` schema gains a `seq` field so both survive the round-trip.

**Tech Stack:** FastAPI + Pydantic (backend), pytest (backend tests), React + Three.js (frontend, no unit-test infra — browser-verified).

**Spec:** `docs/superpowers/specs/2026-07-10-resolution-independent-labels-design.md` (§1, §2; Phase-1 testing section).

**Key design decisions (surfaced assumptions):**
- **`seq` is assigned backend-side at save time by list order**, not by a persisted per-session counter (a deliberate simplification of the spec's "session counter" wording). The frontend appends new instances in apply order (`[...instances, new]` — mode-label.jsx:686, 723), so list order == apply order; deriving `seq` as "preserve existing, fill missing after current max, in list order" yields the same stable monotonic apply-order the spec requires, with no extra state.
- **The Box instance stays `kind:'pointset'`.** Every cuboid-edge/gizmo/auto-fit path is gated on `kind !== 'pointset'` (mode-label.jsx:347, 378, 397, 417, 1207) and `F` routes pointsets through the segId-bbox branch of `focusInstance` (mode-label.jsx:555), so the persisted `center/size` is inert display-wise — exactly the spec's "selection volume, not a display cuboid."

---

## File Structure

- **Modify** `backend/app/schemas.py` — add `seq` to `Cuboid`.
- **Modify** `backend/routes/compare.py` — add `_ensure_seq` helper; call it in `put_annotation` (persist) and `get_annotation` (surface).
- **Modify** `frontend/src/mode-label.jsx` — Box-apply instance gains `center/size/rotation` (one object literal, ~line 686-695).
- **Modify** `backend/tests/test_annotation_sessions.py` — add Phase-1 tests (seq stamping/backfill/idempotency; OBB round-trip).

---

### Task 1: Add `seq` to the `Cuboid` schema

**Files:**
- Modify: `backend/app/schemas.py:82-97` (the `Cuboid` model)
- Test: `backend/tests/test_annotation_sessions.py`

- [ ] **Step 1: Write the failing test** — a pointset instance carrying `center/size/rotation` and `seq` round-trips through PUT→GET unchanged.

Add to `backend/tests/test_annotation_sessions.py`:

```python
def test_box_obb_and_seq_round_trip(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    doc = {
        "scene": scene_id, "kind": "gt", "meta": {},
        "instances": [
            {"id": "b1", "cls": "pipe", "kind": "pointset", "segId": 7,
             "source": "box", "confirmed": True,
             "center": [1.0, 2.0, 3.0], "size": [0.5, 0.5, 2.0],
             "rotation": [0.0, 0.1, 0.0], "seq": 5},
        ],
    }
    r = client.put(f"/api/annotations/gt/{scene_id}?session_id={session_id}", json=doc)
    assert r.status_code == 200
    got = client.get(f"/api/annotations/gt/{scene_id}?session_id={session_id}").json()
    inst = got["instances"][0]
    assert inst["center"] == [1.0, 2.0, 3.0]
    assert inst["size"] == [0.5, 0.5, 2.0]
    assert inst["rotation"] == [0.0, 0.1, 0.0]
    assert inst["seq"] == 5
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/pytest backend/tests/test_annotation_sessions.py::test_box_obb_and_seq_round_trip -v`
Expected: FAIL — `seq` is dropped (not a schema field), so `inst["seq"]` KeyError / assertion error.

(If `.venv` is missing — fresh clone — create it once with `npm run test:backend`, which runs `scripts/test.sh` to build `.venv` and install dev deps, then use `.venv/bin/pytest` for single tests.)

- [ ] **Step 3: Add the field**

In `backend/app/schemas.py`, inside `class Cuboid`, add after the `segId` line (97):

```python
    seq: Optional[int] = None  # monotonic apply-order rank; stamped on save. See
    # docs/superpowers/specs/2026-07-10-resolution-independent-labels-design.md §2
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/pytest backend/tests/test_annotation_sessions.py::test_box_obb_and_seq_round_trip -v`
Expected: PASS.

- [ ] **Step 5: Run the full annotations test file (no regressions)**

Run: `.venv/bin/pytest backend/tests/test_annotation_sessions.py -v`
Expected: all PASS.

- [ ] **Step 6: Commit**

```bash
git add backend/app/schemas.py backend/tests/test_annotation_sessions.py
git commit -m "feat(schema): add seq (apply-order) to Cuboid; round-trips OBB+seq"
```

---

### Task 2: Stamp + backfill `seq` in the annotations route

**Files:**
- Modify: `backend/routes/compare.py` (add `_ensure_seq`; call in `put_annotation` and `get_annotation`)
- Test: `backend/tests/test_annotation_sessions.py`

- [ ] **Step 1: Write the failing tests** — (a) instances saved with no `seq` are backfilled 0,1,2… by list order; (b) a mix of set/unset preserves the set ones and fills the rest after the current max; (c) re-saving is idempotent (seqs don't drift).

Add to `backend/tests/test_annotation_sessions.py`:

```python
def _instances(*specs):
    # spec = (id, seq_or_None)
    out = []
    for k, (iid, seq) in enumerate(specs):
        c = {"id": iid, "cls": "pipe", "kind": "pointset", "segId": 100 + k}
        if seq is not None:
            c["seq"] = seq
        out.append(c)
    return out

def test_seq_backfilled_by_list_order(client_with_annotated_scene):
    client, scene_id, sid = client_with_annotated_scene
    doc = {"scene": scene_id, "kind": "gt", "meta": {},
           "instances": _instances(("a", None), ("b", None), ("c", None))}
    client.put(f"/api/annotations/gt/{scene_id}?session_id={sid}", json=doc)
    got = client.get(f"/api/annotations/gt/{scene_id}?session_id={sid}").json()
    assert [i["seq"] for i in got["instances"]] == [0, 1, 2]

def test_seq_preserves_existing_fills_after_max(client_with_annotated_scene):
    client, scene_id, sid = client_with_annotated_scene
    doc = {"scene": scene_id, "kind": "gt", "meta": {},
           "instances": _instances(("a", 5), ("b", None), ("c", 2))}
    client.put(f"/api/annotations/gt/{scene_id}?session_id={sid}", json=doc)
    got = client.get(f"/api/annotations/gt/{scene_id}?session_id={sid}").json()
    seqs = {i["id"]: i["seq"] for i in got["instances"]}
    assert seqs["a"] == 5 and seqs["c"] == 2       # preserved
    assert seqs["b"] == 6                            # filled after max(5)

def test_seq_stamping_idempotent(client_with_annotated_scene):
    client, scene_id, sid = client_with_annotated_scene
    doc = {"scene": scene_id, "kind": "gt", "meta": {},
           "instances": _instances(("a", None), ("b", None))}
    client.put(f"/api/annotations/gt/{scene_id}?session_id={sid}", json=doc)
    first = client.get(f"/api/annotations/gt/{scene_id}?session_id={sid}").json()["instances"]
    # Re-save exactly what we got back; seqs must not drift.
    client.put(f"/api/annotations/gt/{scene_id}?session_id={sid}",
               json={"scene": scene_id, "kind": "gt", "meta": {}, "instances": first})
    second = client.get(f"/api/annotations/gt/{scene_id}?session_id={sid}").json()["instances"]
    assert [i["seq"] for i in first] == [i["seq"] for i in second] == [0, 1]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_annotation_sessions.py -k seq -v`
Expected: FAIL — `seq` comes back `None` (no stamping yet).

- [ ] **Step 3: Add the helper and call it**

In `backend/routes/compare.py`, add above `get_annotation` (after `router = APIRouter()`):

```python
def _ensure_seq(instances):
    """Stamp a stable monotonic apply-order `seq` on any instance missing one.

    Existing seqs are preserved; missing ones are filled after the current max,
    in list order (which is apply order — the frontend appends new instances).
    Idempotent: re-running never changes an already-stamped instance. Mutates
    the Cuboid objects in place. See the resolution-independent-labels spec §2.
    """
    present = [c.seq for c in instances if c.seq is not None]
    nxt = (max(present) + 1) if present else 0
    for c in instances:
        if c.seq is None:
            c.seq = nxt
            nxt += 1
```

In `get_annotation`, after building the `instances` list, stamp before returning:

```python
    insts = [Cuboid(**c) for c in data.get("instances", [])]
    _ensure_seq(insts)
    return AnnotationDoc(scene=scene, kind=kind, instances=insts, meta=data.get("meta", {}))
```

In `put_annotation`, stamp before dumping:

```python
    _ensure_seq(doc.instances)
    body = {
        "scene": scene,
        "kind": kind,
        "instances": [c.model_dump() for c in doc.instances],
        "meta": doc.meta,
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest backend/tests/test_annotation_sessions.py -k seq -v`
Expected: all PASS.

- [ ] **Step 5: Run the whole backend suite (no regressions)**

Run: `.venv/bin/pytest backend/tests/ -q`
Expected: all PASS (the added `seq` field is optional; existing tests assert on `segId`/counts, unaffected).

- [ ] **Step 6: Commit**

```bash
git add backend/routes/compare.py backend/tests/test_annotation_sessions.py
git commit -m "feat(annotations): stamp + backfill monotonic seq on save/load"
```

---

### Task 3: Box apply persists its OBB (frontend)

**Files:**
- Modify: `frontend/src/mode-label.jsx:686-695` (the `applyBox` instance literal)

No frontend unit-test infra exists (vitest is pure-function only per CLAUDE.md), and the round-trip persistence is already guarded by Task 1's backend test. This task is a one-object change plus a browser verification.

- [ ] **Step 1: Add the OBB fields to the created instance**

In `frontend/src/mode-label.jsx`, in `applyBox`, change the pushed instance (currently lines 686-695) to include the box geometry from `selBox` (already in the recentered display frame — the frame the spec stores shapes in):

```jsx
      onChange([...instances, {
        id: newId(),
        segId,
        kind: 'pointset',
        cls: targetCls.id,
        label: `${targetCls.label} ${(counts[targetCls.id] || 0) + 1}`,
        color: targetCls.color,
        source: 'box',
        confirmed: !!autoConfirmFor('box'),
        // Persist the selection OBB (display frame) so a future export can
        // rasterize this box at any density. Stays kind:'pointset' → no gizmo,
        // no cuboid edges (gated on kind !== 'pointset'). Spec §1.
        center: [...selBox.center],
        size: [...selBox.size],
        rotation: [...selBox.rotation],
      }]);
```

- [ ] **Step 2: Build the frontend (typecheck/bundle sanity)**

Run: `npm run build`
Expected: builds with no errors.

- [ ] **Step 3: Browser-verify (REQUIRED SUB-SKILL: browser-verification)**

Follow `browser-verification`. IMPORTANT (memory `feedback_browser_verify_mutates_session`): Label apply auto-saves to disk — use a **throwaway session** on an annotated scan, and restart any stale `:8765` backend first.

- Start dev (`npm run dev`), open `http://127.0.0.1:5173`, load an annotated scan, create/select a scratch session.
- Select the **Box** tool, draw a box around some points, apply with a class hotkey.
- Confirm in the browser: the box outline vanishes, a `box` pointset row appears, **no cuboid edges or gizmo render** for it, and there are **zero console errors**.
- Confirm on disk the OBB persisted:

```bash
python3 -c "import json,glob; f=sorted(glob.glob('/home/hendrik/coding/engine/data/lidar/annotated/*/sessions/*/instances_gt.json'))[-1]; d=json.load(open(f)); b=[i for i in d['instances'] if i.get('source')=='box']; print(f); print(b[-1] if b else 'NO BOX INSTANCE')"
```

Expected: the last box instance shows non-null `center`/`size`/`rotation` and a `seq`.

- [ ] **Step 4: Take a screenshot of the applied-box state** (per global rule for UI changes) and confirm it matches the intended "no visible cuboid" behavior.

Note: one benign behavior change — pressing `F` on a selected box instance now frames the camera on it (previously a no-op, since `center/size` were null). For a box pointset with live `segState`, `focusInstance` frames the actual labeled points (segId branch, mode-label.jsx:555); only in a degraded state (no `segState`) does it fall back to the OBB. Camera-only, no gizmo/edges — acceptable, not a regression against the "no visible cuboid" goal.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/mode-label.jsx
git commit -m "feat(label): Box apply persists its OBB (center/size/rotation) on the instance"
```

---

## Done criteria (Phase 1)

- `Cuboid` carries `seq`; OBB + `seq` round-trip through `PUT/GET /api/annotations`.
- Every instance ends up with a stable monotonic `seq` (new + backfilled), idempotent across re-saves.
- Box apply writes `center/size/rotation`; box instances remain visually non-cuboid.
- Full backend suite green; frontend builds; browser shows correct behavior with zero console errors.

## Not in this phase (Phase 2 — see spec §3-§5)

The materialize algorithm (two regimes, max-`seq` replay, interior defense), raw-cloud resolution via `derivation → sources.json`, frame alignment, and the export endpoint. None of it is built here; Phase 1 only ensures the data it will consume is captured.

## Docs

As the last step before the Phase-1 PR, note in `CLAUDE.md` (the "Cuboids are retired" gotcha) that `source=='box'` pointsets now persist their OBB as a resolution-independent selection volume (still no gizmo), and that instances carry a stamped `seq`. Keep it to the narrow reversal described in the spec.
