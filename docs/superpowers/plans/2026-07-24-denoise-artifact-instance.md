# Detect-outliers → unconfirmed artifact instance — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ✧ Detect-outliers tool materialize its catches as an unconfirmed `artifact` instance (magenta, confirmable, hideable) instead of an `excluded_review` review blob, per the labeler decision guide.

**Architecture:** Backend gains a generic `allocate_instance` flag on `apply_category` so denoise can mint an artifact *blob*; the existing class-less instance-id strip (`segment.py:474-478`) already keeps GT at `instance −1`; a one-line normalization in `load_instances_for_invariants` treats `confirmed` as session-only for class-less blobs (fixes eval-invariant 9, which also closes the latent confirmed-review-blob bug). Frontend swaps the denoise row to an artifact blob row and fixes the category-overlay so a confirmed blob's dots hide.

**Tech Stack:** Python/FastAPI + pytest (backend), React 18 + Three.js + vitest (frontend).

**Spec:** `docs/superpowers/specs/2026-07-24-denoise-artifact-instance-design.md`

---

## File Structure

- `backend/labeling/segment_state.py` — `apply_category` gains `allocate_instance` param. No `_apply` change (its `set_category` branch already allocates when `payload["blob"]`).
- `backend/routes/segment.py` — `_denoise_core` marks `CATEGORY_ARTIFACT` with `allocate_instance=True`.
- `backend/labeling/instances_doc.py` — `load_instances_for_invariants` normalizes `confirmed=False` on class-less rows.
- `frontend/src/point-categories.js` — add `ARTIFACT_COLOR`/`ARTIFACT_LABEL` exports; `buildCategoryOverlay` gains an optional `hiddenMask` param (pure, testable) to drop confirmed-hidden points.
- `frontend/src/mode-label.jsx` — `runDenoise` writes `CATEGORY_ARTIFACT` + appends an `artifactBlobRow`; hoist `confirmedPointsetHideMask` memo above the overlay effect and pass it into `buildCategoryOverlay`.
- Tests: `backend/tests/test_segment_categories.py`, `test_denoise_routes.py`, `test_instances_doc.py`; `frontend/src/point-categories.test.js`.
- Docs: `CLAUDE.md`, `docs/superpowers/specs/2026-07-20-outlier-detection-filtering-design.md`.

---

## Task 1: `apply_category` gains `allocate_instance`

**Files:**
- Modify: `backend/labeling/segment_state.py` (`apply_category`, ~line 148-179)
- Test: `backend/tests/test_segment_categories.py`

- [ ] **Step 1: Write the failing tests**

```python
def test_artifact_with_allocate_instance_mints_a_blob():
    seg = _session()
    idx = np.arange(0, 8, dtype=np.int32)
    out = seg.apply_category(idx, CATEGORY_ARTIFACT, allocate_instance=True)
    assert out["n_affected"] == 8
    assert out.get("new_instance_id") is not None
    blob = out["new_instance_id"]
    assert bool((seg.categories[idx] == CATEGORY_ARTIFACT).all())
    assert bool((seg.class_ids[idx] == -1).all())
    assert bool((seg.instance_ids[idx] == blob).all())

def test_artifact_default_still_erases_instance():
    # allocate_instance defaults to None → today's behavior unchanged
    seg = _session()
    idx = np.arange(0, 8, dtype=np.int32)
    out = seg.apply_category(idx, CATEGORY_ARTIFACT)
    assert out.get("new_instance_id") is None
    assert bool((seg.instance_ids[idx] == -1).all())

def test_review_default_still_allocates_blob():
    seg = _session()
    idx = np.arange(0, 8, dtype=np.int32)
    out = seg.apply_category(idx, CATEGORY_EXCLUDED_REVIEW)
    assert out.get("new_instance_id") is not None

def test_allocate_instance_false_forces_no_blob_on_review():
    seg = _session()
    idx = np.arange(0, 8, dtype=np.int32)
    out = seg.apply_category(idx, CATEGORY_EXCLUDED_REVIEW, allocate_instance=False)
    assert out.get("new_instance_id") is None
    assert bool((seg.instance_ids[idx] == -1).all())
```

- [ ] **Step 2: Run to verify they fail**

Run: `.venv/bin/pytest backend/tests/test_segment_categories.py -k "allocate or blob or erases_instance or still_allocates" -v`
Expected: FAIL (`allocate_instance` is an unexpected keyword arg).

- [ ] **Step 3: Implement**

In `apply_category`, change the signature and the `blob` computation:

```python
def apply_category(
    self, indices: np.ndarray, category,
    protect_instances: Optional[list[int]] = None,
    allocate_instance: Optional[bool] = None,
) -> dict:
    ...
    from labeling.categories import parse_category, CATEGORY_EXCLUDED_REVIEW
    cat = parse_category(category)
    ...
    blob = (cat == CATEGORY_EXCLUDED_REVIEW) if allocate_instance is None else bool(allocate_instance)
    out = self._apply("set_category", indices, dict(category=cat, blob=blob))
    out["n_protected"] = n_protected
    return out
```

Update the docstring to note `allocate_instance` (default `None` = category-driven; `True`/`False` force a blob or not — used by denoise to mint an artifact blob).

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest backend/tests/test_segment_categories.py -v`
Expected: PASS (all, including pre-existing).

- [ ] **Step 5: Commit**

```bash
git add backend/labeling/segment_state.py backend/tests/test_segment_categories.py
git commit -m "feat(backend): apply_category allocate_instance flag for artifact blobs"
```

---

## Task 2: `_denoise_core` marks an artifact blob

**Files:**
- Modify: `backend/routes/segment.py` (`_denoise_core`, ~line 171-209)
- Test: `backend/tests/test_denoise_routes.py`

- [ ] **Step 1: Write the failing test**

Mirror `test_denoise_materializes_review_blob`, asserting artifact category now:

```python
def test_denoise_materializes_artifact_blob():
    client, seg = _client_with_cloud()
    from labeling.categories import CATEGORY_ARTIFACT
    r = client.post("/api/segment/denoise", json={"std_ratio": 1.0, "k": 8})
    assert r.status_code == 200
    body = r.json()
    inst = body["instance_id"]
    assert inst is not None
    # the flagged points carry category=artifact, class=-1, instance=blob id
    flagged = np.flatnonzero(seg.instance_ids == inst)
    assert flagged.size == body["n_affected"] > 0
    assert bool((seg.categories[flagged] == CATEGORY_ARTIFACT).all())
    assert bool((seg.class_ids[flagged] == -1).all())

def test_denoise_replace_inst_erases_prior_artifact_blob():
    client, seg = _client_with_cloud()
    first = client.post("/api/segment/denoise", json={"std_ratio": 1.0, "k": 8}).json()
    inst1 = first["instance_id"]
    second = client.post("/api/segment/denoise",
                         json={"std_ratio": 1.2, "k": 8, "replace_inst": inst1}).json()
    # prior blob fully erased before recompute
    assert np.flatnonzero(seg.instance_ids == inst1).size == 0
    assert second["instance_id"] != inst1
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest backend/tests/test_denoise_routes.py -k artifact -v`
Expected: FAIL (points come back `CATEGORY_EXCLUDED_REVIEW`).

- [ ] **Step 3: Implement**

In `_denoise_core`, swap the import + materialize call:

```python
from labeling.categories import CATEGORY_ARTIFACT
...
out = seg.apply_category(
    outliers.astype(np.int32), CATEGORY_ARTIFACT,
    allocate_instance=True, protect_instances=req.protect_instances)
```

Update the function docstring (it currently says "review blob" / "excluded_review category") to describe an artifact blob. Leave the `replace_inst` erase block and the response contract unchanged.

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest backend/tests/test_denoise_routes.py -v`
Expected: PASS. (If an old test asserted `CATEGORY_EXCLUDED_REVIEW` for global denoise, update it to `CATEGORY_ARTIFACT` — do NOT touch `/denoise-selection` tests.)

- [ ] **Step 5: Commit**

```bash
git add backend/routes/segment.py backend/tests/test_denoise_routes.py
git commit -m "feat(backend): detect-outliers marks an artifact blob, not a review blob"
```

---

## Task 3: `confirmed` is session-only for class-less blobs (eval-invariant 9)

**Files:**
- Modify: `backend/labeling/instances_doc.py` (`load_instances_for_invariants`)
- Test: `backend/tests/test_instances_doc.py`

- [ ] **Step 1: Write the failing test**

```python
def test_classless_row_confirmed_is_normalized_to_false(tmp_path):
    import json
    from labeling.instances_doc import load_instances_for_invariants
    doc = {"instances": [
        {"segId": 5, "kind": "pointset", "cls": None, "confirmed": True},   # blob
        {"segId": 6, "kind": "pointset", "cls": 2, "confirmed": True},      # real
    ]}
    (tmp_path / "instances_gt.json").write_text(json.dumps(doc))
    got = load_instances_for_invariants(tmp_path)
    assert got[5]["confirmed"] is False   # class-less → normalized
    assert got[6]["confirmed"] is True    # classed → untouched
    assert got[5]["class_id"] is None
```

- [ ] **Step 2: Run to verify it fails**

Run: `.venv/bin/pytest backend/tests/test_instances_doc.py -k normalized -v`
Expected: FAIL (`got[5]["confirmed"]` is `True`).

- [ ] **Step 3: Implement**

In `load_instances_for_invariants`, gate `confirmed` on class presence:

```python
cls = inst.get("cls")
result[int(seg_id)] = {
    "class_id": cls,
    # A class-less blob (artifact/review) is a session-only review handle,
    # never a confirmed GT instance (labeler guide; eval-invariant 9). Its
    # in-session `confirmed` drives hide/protect only; strip it here.
    "confirmed": bool(inst.get("confirmed", False)) and cls is not None,
}
```

- [ ] **Step 4: Run to verify pass**

Run: `.venv/bin/pytest backend/tests/test_instances_doc.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add backend/labeling/instances_doc.py backend/tests/test_instances_doc.py
git commit -m "fix(backend): class-less blobs are never confirmed at the invariant layer"
```

---

## Task 4: Confirmed-blob save round-trip passes all 9 invariants

**Files:**
- Test only: `backend/tests/test_segment_io.py`

This is the integration guard proving the whole backend chain saves — both an artifact blob and a confirmed one, plus the review-blob regression.

- [ ] **Step 1: Write the failing/guard test**

Follow the existing save-round-trip pattern in `test_segment_io.py` (a `save_labels(...)` call with `categories=` and `instances_doc=`). Assert:

```python
# after a denoise-style artifact blob + a CONFIRMED instances_gt.json row for it:
#  - save_labels does NOT raise EvalInvariantError
#  - gt_segment_ids has -1 at the blob's points (stripped)
#  - gt_point_category.npy has CATEGORY_ARTIFACT there
#  - meta["review_blobs"] does not list the artifact blob id
# and the sibling regression:
#  - a CONFIRMED excluded_review review blob likewise saves without raising
```

Construct `instances_doc` as the *loaded* dict (call `load_instances_for_invariants` on a tmp session dir holding the confirmed doc) so the normalization from Task 3 is exercised end-to-end. Strip the class-less instance ids the same way `segment_save` does (`out_inst[(inst>=0)&(cls<0)] = -1`) before passing to `save_labels`, matching the real call site.

- [ ] **Step 2: Run to verify it fails** (before Task 3 it would 422; run after Tasks 1-3, it should pass — if the harness ran this first it fails on `EvalInvariantError`).

Run: `.venv/bin/pytest backend/tests/test_segment_io.py -k "artifact or confirmed_blob" -v`

- [ ] **Step 3: (no new impl — Tasks 1-3 provide it).** If the test reveals a gap, fix at root, do not weaken the assertion.

- [ ] **Step 4: Run full backend suite**

Run: `.venv/bin/pytest backend/ -q`
Expected: PASS (all).

- [ ] **Step 5: Commit**

```bash
git add backend/tests/test_segment_io.py
git commit -m "test(backend): confirmed artifact/review blob saves pass all eval invariants"
```

---

## Task 5: Frontend — `ARTIFACT_COLOR`/`ARTIFACT_LABEL` + `buildCategoryOverlay` hidden-mask

**Files:**
- Modify: `frontend/src/point-categories.js`
- Test: `frontend/src/point-categories.test.js`

- [ ] **Step 1: Write the failing tests**

```js
import { ARTIFACT_COLOR, ARTIFACT_LABEL, buildCategoryOverlay,
         CATEGORY_ARTIFACT } from './point-categories.js';

test('artifact color/label exports match the vocabulary', () => {
  expect(ARTIFACT_COLOR).toBe('#ff4dd2');
  expect(ARTIFACT_LABEL).toBe('Artifact');
});

test('buildCategoryOverlay drops points flagged in hiddenMask', () => {
  const cats = new Int8Array([CATEGORY_ARTIFACT, CATEGORY_ARTIFACT, CATEGORY_ARTIFACT]);
  const hidden = new Uint8Array([0, 1, 0]);           // hide the middle one
  const ov = buildCategoryOverlay(cats, null, 3, hidden);
  expect(ov.mask[0]).toBe(1);
  expect(ov.mask[1]).toBe(0);                          // dropped
  expect(ov.mask[2]).toBe(1);
});

test('buildCategoryOverlay without hiddenMask is unchanged', () => {
  const cats = new Int8Array([CATEGORY_ARTIFACT, CATEGORY_ARTIFACT]);
  const ov = buildCategoryOverlay(cats, null, 2);
  expect(ov.mask[0]).toBe(1);
  expect(ov.mask[1]).toBe(1);
});
```

- [ ] **Step 2: Run to verify fail**

Run (from `frontend/`): `npx vitest run src/point-categories.test.js`
Expected: FAIL (exports undefined; 4th param ignored).

- [ ] **Step 3: Implement**

Add exports near `REVIEW_COLOR`/`REVIEW_LABEL`:

```js
export const ARTIFACT_COLOR = '#ff4dd2';
export const ARTIFACT_LABEL = 'Artifact';
```

Extend `buildCategoryOverlay` with an optional `hiddenMask`:

```js
export function buildCategoryOverlay(categories, subIdx, subN, hiddenMask = null) {
  if (!categories) return null;
  const mask = new Uint8Array(subN);
  const colors = new Float32Array(subN * 3);
  let any = false;
  for (let p = 0; p < subN; p++) {
    if (hiddenMask && hiddenMask[p]) continue;         // confirmed-hidden → not painted
    const rgb = CATEGORY_RGB[categories[subIdx ? subIdx[p] : p]];
    if (!rgb) continue;
    mask[p] = 1;
    any = true;
    const o = p * 3;
    colors[o] = rgb[0]; colors[o + 1] = rgb[1]; colors[o + 2] = rgb[2];
  }
  return any ? { mask, colors } : null;
}
```

- [ ] **Step 4: Run to verify pass**

Run: `npx vitest run src/point-categories.test.js`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add frontend/src/point-categories.js frontend/src/point-categories.test.js
git commit -m "feat(frontend): artifact color/label exports + buildCategoryOverlay hiddenMask"
```

---

## Task 6: Frontend — denoise row is an artifact blob; overlay hides confirmed blobs

**Files:**
- Modify: `frontend/src/mode-label.jsx`

No new unit test (this is component-internal wiring); it is browser-verified in Task 8. Keep each change minimal.

- [ ] **Step 1: Add `artifactBlobRow` + swap `runDenoise`**

- Import `ARTIFACT_COLOR, ARTIFACT_LABEL, CATEGORY_ARTIFACT` from `point-categories.js` (line ~32 import block).
- Add `artifactBlobRow(segId, source)` beside `reviewBlobRow` (line ~43): identical shape, but `label: ARTIFACT_LABEL`, `color: ARTIFACT_COLOR`.
- In `runDenoise` (line ~753-760): fill `CATEGORY_ARTIFACT` instead of `CATEGORY_EXCLUDED_REVIEW`, and append `artifactBlobRow(resp.instance_id, 'denoise')` instead of `reviewBlobRow(...)`. Update the nearby comment ("REVIEW BLOB" → "ARTIFACT BLOB").

- [ ] **Step 2: Hoist the hide memo + wire it into the overlay effect**

- Move the `confirmedPointsetHideMask` `useMemo` (line ~819) to ABOVE the category-overlay effect (line ~262). Its deps (`instances, selectedId, cloud, segState?.instanceFull`) are all in scope there. (Prevents a temporal-dead-zone `ReferenceError`.)
- In the overlay effect, change the `buildCategoryOverlay` call (line 275) to pass the hide mask:
  ```js
  const catOverlay = buildCategoryOverlay(
    segState.categoryFull, subIdx, subN,
    hideConfirmed ? confirmedPointsetHideMask : null);
  ```
- Add `hideConfirmed` and `confirmedPointsetHideMask` to the effect's dependency array (currently `[segState, cloud, viewerRef, fastMode, activeTool]`, line ~340).

- [ ] **Step 3: Lint / typecheck-ish sanity**

Run (from `frontend/`): `npx vitest run` (full FE suite — nothing should break).
Expected: PASS.

- [ ] **Step 4: Commit**

```bash
git add frontend/src/mode-label.jsx
git commit -m "feat(frontend): denoise yields an artifact blob; confirmed blobs hide in the overlay"
```

---

## Task 7: Restart backend & browser-verify

**REQUIRED SUB-SKILL:** Use browser-verification. Backend has no autoreload — restart `npm run dev` first (per CLAUDE.md). Use a **throwaway session** (Label apply auto-saves to disk — see the "browser-verify mutates session" note); do not test on a precious session.

- [ ] **Step 1: Restart dev server + sidecar** (kill stale :8765/:5173/:8011 first).
- [ ] **Step 2:** Open Label mode on a scan with a registered raw source, create a fresh session. Click **✧ Detect outliers**.
  - Verify: the caught points render **magenta**, the new Instances row reads **"Artifact"** (not grey "Review"), unconfirmed.
- [ ] **Step 3:** Confirm that Artifact row.
  - Verify: its magenta points **disappear** (base cloud AND overlay dots gone). Re-select the row → dots return.
- [ ] **Step 4:** With it confirmed, run detect-outliers again at a looser σ.
  - Verify: the confirmed points are not re-caught; a new artifact row appears for new strays.
- [ ] **Step 5:** `Ctrl+S`.
  - Verify: **save succeeds (200, no 422)**, no console errors, no failed network requests.
- [ ] **Step 6:** Regression — confirm a pre-existing grey **Review** blob (or make one via the class picker's Review mark) and confirm it hides + saves too.
- [ ] Screenshot each key state; report paths.

---

## Task 8: Docs

**Files:**
- Modify: `CLAUDE.md` (Outlier-filtering bullet's Feature C description; Point-categories bullet's denoise note)
- Modify: `docs/superpowers/specs/2026-07-20-outlier-detection-filtering-design.md` (Feature C: review blob → artifact blob)

- [ ] **Step 1:** Update the Feature C prose in both: global denoise now materializes an **unconfirmed artifact instance** (category `artifact`, magenta, confirmable), not an `excluded_review` review blob; note that `confirmed` on a class-less blob is session-only (normalized off at the invariant layer) and that GT strips the blob id to `instance −1`.
- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md docs/superpowers/specs/2026-07-20-outlier-detection-filtering-design.md
git commit -m "docs: detect-outliers now yields an artifact instance"
```

---

## Done criteria

- `.venv/bin/pytest backend/ -q` green; `npx vitest run` (from `frontend/`) green.
- Browser: detect-outliers → magenta Artifact row; confirm hides it; re-run excludes confirmed; `Ctrl+S` saves without 422; review-blob regression hides + saves.
- Docs updated in the same PR.
