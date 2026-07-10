# Export Wizard (Phase B) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship the export wizard — a `POST /api/labels/export` endpoint that turns the active session's labels into a downloadable zip (labeled `.ply` + `manifest.json`) at a chosen resolution with confirmed-only / include-exclude / class-remap filters, plus the 3-step wizard modal that drives it.

**Architecture:** The endpoint builds a `MaterializeCtx` from `_state` + session files, calls the Phase-A materialize core (regime A in-memory; regime B streamed chunk-by-chunk to a temp file), applies a pure filter/remap pipeline to the per-point labels, writes a binary PLY + `manifest.json`, zips them to a temp file, and returns a `FileResponse` with try/finally cleanup. The frontend adds an Export button + a 3-step modal that POSTs the config and downloads the zip blob.

**Tech Stack:** FastAPI, NumPy, Python `zipfile`/`tempfile`, Starlette `FileResponse`/`BackgroundTask` (backend); React + `fetch`→blob→`<a download>` (frontend, vitest for pure fns).

**Spec:** `docs/superpowers/specs/2026-07-10-export-wizard-design.md` (rev 2, approved).

**Prerequisite SHIPPED (Phase A):** `backend/labeling/materialize.py` — `materialize(ctx, resolution)`, `materialize_raw(index, ...)` generator, `MaterializeCtx`, `build_replay_index`, `collect_volumes`, `raw_sample_spacing`; `raw_source_available` on `LoadResponse`. This plan consumes them.

**Key facts (so the implementer needn't rediscover):**
- Active session state: `_state["seg"]` is a `SegmentSession` with `.class_ids` (int8, N), `.instance_ids` (int32, N), `.session_dir`. `_state["pc"]` is the scan.ply cloud (`.points` (N,3), `.colors` (N,3) — **display/recentered frame**). `_state["scene"]`, `_state["session_id"]`, `_state["recenter_offset"]`, `_state["source"]` (`SceneSource`; `.extras["source_laz_path"]` = raw path or None).
- Instances doc: read via `_annotation_path(scene, "gt", session_id)` → `instances_gt.json` (list of `Cuboid` dicts with `cls` string, `segId`, `confirmed`, `source`, `center/size/rotation`, `seq`). Centerlines: `centerline.load_centerlines(seg.session_dir)`.
- Class palette: `routes/meta.py::get_config()` returns `ClassDef`s each with a string `id` and numeric `class_id`; `app/core.py` has a `{class-name-lower: int-id}` builder (near line 304) + `_coerce_class_id`. Use these to map an instance's `cls` string → numeric `class_id` for `inst_class_id`.
- `_scene_is_z_up(src)`, `_to_display_frame`, `_stream_laz_keep`, `_ply_response_bytes(xyz, rgb)` (xyz+rgb only — Task 4 extends it) all live in `app/core.py` / imported into `routes/export.py` via `from app.core import *`.
- **Output frame:** materialize returns display frame, so the exported PLY is in the **display/recentered** frame (consistent with the `scan` and `raw` resolutions alike). A source/UTM-frame option is a follow-up, not v1.

---

## File Structure

- **Modify** `backend/app/schemas.py` — `ExportLabelsRequest` (+ nested resolution/remap models).
- **Create** `backend/labeling/export_pipeline.py` — pure functions: `validate_export_request`, `apply_filters_remap`, `build_manifest`. (Keep the endpoint thin; put the correctness-critical logic here, unit-tested without HTTP.)
- **Modify** `backend/app/core.py` — extend `_ply_response_bytes` → add a `_ply_labeled_bytes(xyz, rgb, class_ids, instance_ids)` (or a labels kwarg) writing `class_id`/`instance_id` vertex props.
- **Modify** `backend/routes/export.py` — the `POST /api/labels/export` handler + a `_build_materialize_ctx()` helper.
- **Create** `frontend/src/export-wizard.jsx` — the 3-step modal.
- **Modify** `frontend/src/api.js` — `exportLabels(cfg)`; `frontend/src/mode-label.jsx` — the Export button; wire the wizard.
- **Create** `backend/tests/test_export_labels.py`, `frontend/src/export-wizard.test.js`.

---

### Task 1: `ExportLabelsRequest` schema + validation

**Files:** Modify `backend/app/schemas.py`; Create `backend/labeling/export_pipeline.py`; Test `backend/tests/test_export_labels.py`.

- [ ] **Step 1: Failing tests** for `validate_export_request(req, n_scan, palette_ids, raw_available)` covering each 422 case (spec §2): unknown `from` id; overlapping `from` sets; duplicate `to.id` with differing label/color; `to.id` colliding with a kept-through class; `subsample` `n > n_scan` or `n < 1`; `raw` when `raw_available` is False. Each returns a list of error strings (empty = valid).
- [ ] **Step 2: Run → fails.**
- [ ] **Step 3: Implement** the schema + `validate_export_request` (returns error list; the endpoint raises 422 if non-empty). Schema:
```python
class RemapRule(BaseModel):
    from_: list[int] = Field(alias="from")
    to: dict  # {id:int, label:str, color:str}
class ExportResolution(BaseModel):
    kind: str  # "scan" | "subsample" | "raw"
    n: Optional[int] = None
class ExportLabelsRequest(BaseModel):
    scene: str
    session_id: str
    resolution: ExportResolution
    confirmed_only: bool = False
    include_classes: Optional[list[int]] = None
    remap: list[RemapRule] = []
    drop_unlabeled: bool = False
```
- [ ] **Step 4/5:** Run → pass; commit `feat(export): ExportLabelsRequest + validation`.

---

### Task 2: filter/remap pipeline (pure, correctness-critical)

**Files:** Modify `backend/labeling/export_pipeline.py`; Test `backend/tests/test_export_labels.py`.

`apply_filters_remap(class_ids, instance_ids, confirmed_by_inst, req, src_to_tgt) -> (out_cls, out_inst)`. **Chunk-safe:** operates on ONE array (or raw chunk) and returns only the transformed arrays — no aggregate counts (those would double-count across raw chunks). Order = spec §2 (confirmed-only → include/exclude → remap; `drop_unlabeled` applied by the caller). Semantics:
- **confirmed-only:** points whose `instance_id`'s `confirmed` flag is False → `class_id = -1`. Instances absent from `confirmed_by_inst` are treated as confirmed (not zeroed).
- **include/exclude:** points whose (pre-remap) `class_id ∉ include_classes` → `-1` (skip if `include_classes` is None).
- **remap:** map surviving source `class_id`s → target ids using the precomputed `src_to_tgt` (from `build_taxonomy`, built once — see below), so the mapping is identical across all chunks. (`taxonomy` itself, `{target_id:{label,color}}`, can't encode which sources collapse into a target — that's why `src_to_tgt` is the argument, not `taxonomy`.)

Two separate pure helpers (computed ONCE, globally — never per-chunk):
- `build_taxonomy(palette, req) -> (taxonomy, src_to_tgt)`: `taxonomy = {target_id: {label, color}}` built **from the class palette + `req.remap` targets** (palette-driven, NOT `np.unique(class_ids)`-driven, so pass-through classes never present in a given array/chunk still appear). `src_to_tgt` maps every source class_id → its target id.
- `count_absent_instances(work_inst, confirmed_by_inst) -> int`: the number of DISTINCT instance ids present in the full `work_inst` but missing from `confirmed_by_inst`. Computed once from the session arrays, independent of resolution/chunking.

- [ ] **Step 1: Failing tests** (each a small synthetic array): confirmed-only zeros an unconfirmed instance's points; exclude zeros a class; **precedence** — a class both excluded and in a remap `from` is `-1` (exclude wins); remap-merge maps two source classes to one `to.id` in BOTH the array and `taxonomy` while `instance_ids` are UNCHANGED; **pass-through taxonomy** — palette {0,1,2}, array contains only {0,1}, assert `taxonomy` STILL has an entry for class 2 (palette-driven, not data-driven); `count_absent_instances` returns the DISTINCT count (an instance spanning many points counts once); empty-after-filters yields all `-1` without error.
- [ ] **Steps 2-5:** Run→fail; implement; run→pass; commit `feat(export): confirmed/include/remap filter pipeline`.

---

### Task 3: `build_manifest`

**Files:** Modify `backend/labeling/export_pipeline.py`; Test `backend/tests/test_export_labels.py`.

`build_manifest(taxonomy, accuracy_p50, accuracy_p90, scan, session, resolution, points, filters, absent_count, exported_at) -> dict` producing the spec §3 manifest (`classes`, `accuracy` {labeling_points?, sample_spacing_p50_m, sample_spacing_p90_m, semantic_boundary_uncertainty_m=p90, note}, `source`, `resolution`, `filters` incl. `absent_instances`).

- [ ] **Steps 1-5:** Failing test asserts the structure + that `semantic_boundary_uncertainty_m == p90` and the note text is present; implement; commit `feat(export): manifest.json builder`. (Pass `exported_at` in — do NOT call `datetime.now()` inside; the endpoint stamps it.)

---

### Task 4: labeled PLY writer

**Files:** Modify `backend/app/core.py`; Test `backend/tests/test_export_labels.py` (or `test_smoke`/existing ply test).

Add THREE decomposed helpers (so both the regime-A one-shot writer and Task 6's streaming path — which must write the body before it knows the final vertex count — reuse them):
- `_ply_labeled_header(n, has_color) -> bytes` — the ASCII header for `n` vertices with props `x y z` (f32), optional `red green blue` (uchar), `class_id` (int32), `instance_id` (int32).
- `_ply_labeled_chunk_bytes(xyz, rgb, class_ids, instance_ids) -> bytes` — the binary BODY for one array/chunk (structured-dtype approach, mirroring `_ply_response_bytes`, core.py:383), NO header.
- `_ply_labeled_bytes(xyz, rgb, class_ids, instance_ids) -> bytes` = `_ply_labeled_header(len(xyz), rgb is not None) + _ply_labeled_chunk_bytes(...)` — the one-shot writer for regime A.

Task 6's raw path: write `_ply_labeled_chunk_bytes` per chunk to a body temp file while counting kept points, then assemble the final PLY = `_ply_labeled_header(total, has_color)` + the body temp's contents.

- [ ] **Step 1: Failing test** — round-trip `_ply_labeled_bytes`: write N points with known class/instance, parse the PLY header (property list) + read back the class_id/instance_id columns, assert equality; AND assert `_ply_labeled_bytes == _ply_labeled_header(...) + _ply_labeled_chunk_bytes(...)` (so the streaming assembly is byte-identical to the one-shot). (Use `plyfile` if available, else parse the known binary layout.)
- [ ] **Steps 2-5:** implement; commit `feat(export): binary PLY writer with class_id/instance_id`.

---

### Task 5: `_build_materialize_ctx()`

**Files:** Modify `backend/routes/export.py`; Test `backend/tests/test_export_labels.py`.

`_build_materialize_ctx(scene, session_id) -> (MaterializeCtx, instances, confirmed_by_inst, class_id_by_inst)` from `_state` + session files:
- guard `_state["scene"] == scene` and `_state["session_id"] == session_id` (else 409, mirroring `edit_export_ply`'s `req.scene != scene` check at `backend/routes/export.py:27`); **snapshot** the needed `_state` fields into locals now (survive a concurrent `/api/load`).
- `seg = _state["seg"]`; `pc = _state["pc"]`; `src = _state["source"]`.
- **Read instances with an existence guard** (a blank session has no `instances_gt.json` yet — mirror `compare.py:get_annotation` at `routes/compare.py:29-33`, which returns `[]` when the file is absent). `_annotation_path(...)` returns a `Path`, so:
  ```python
  p = _annotation_path(scene, "gt", session_id)
  instances = json.loads(p.read_text()).get("instances", []) if p.exists() else []
  ```
  (Do NOT `json.load(_annotation_path(...))` — that passes a `Path`, not a file object, and crashes on a blank session.)
- `centerlines = load_centerlines(seg.session_dir)`; `volumes = collect_volumes(instances, centerlines)`.
- `class_id_by_inst = {int(i["segId"]): _cls_string_to_num(i["cls"]) for i in instances if i.get("segId") is not None}`; for instances not in the doc but present in `seg.instance_ids`, fill `inst_class_id` from `seg.class_ids` at a point they own (so `replay_labels` never KeyErrors — Phase A's contract). `seq_by_inst = {int(i["segId"]): i.get("seq") for ...}`. `confirmed_by_inst = {int(i["segId"]): bool(i.get("confirmed")) for ...}`.
- `_cls_string_to_num` uses the classes.yaml name→id map (core.py ~304) / `_coerce_class_id`.
- `ctx = MaterializeCtx(scan_pos=pc.points, colors=pc.colors (uint8), work_cls=seg.class_ids, work_inst=seg.instance_ids, volumes=volumes, seq_by_inst=seq_by_inst, inst_class_id=inst_class_id, raw_path=src.extras.get("source_laz_path"), scene_is_z_up=_scene_is_z_up(src), offset=np.asarray(_state["recenter_offset"]))`.

- [ ] **Step 1: Failing test** — load a fixture annotated scene + session, call the builder, assert the ctx arrays have matching lengths and `inst_class_id` covers every id in `work_inst` (the Phase-A invariant). Reuse the `client_with_annotated_scene` fixture + a `/api/load`.
- [ ] **Steps 2-5:** implement; commit `feat(export): build MaterializeCtx from session state`.

---

### Task 6: `POST /api/labels/export` endpoint (streaming + zip)

**Files:** Modify `backend/routes/export.py`; Test `backend/tests/test_export_labels.py`.

Handler:
1. `validate_export_request(...)` → 422 on errors. Guard scene/session (Task 5).
2. Build ctx (Task 5). Compute `p50, p90 = raw_sample_spacing(ctx.scan_pos)`. Build the **global** `(taxonomy, src_to_tgt) = build_taxonomy(palette, req)` and `absent_count = count_absent_instances(ctx.work_inst, confirmed_by_inst)` ONCE here (never per-chunk — see Task 2).
3. Produce the labeled PLY:
   - `scan`/`subsample` → `materialize(ctx, resolution)` (in-memory; regime A) → `apply_filters_remap(...)` once → optional drop_unlabeled → `_ply_labeled_bytes(...)`.
   - `raw` → **stream**: `index = build_replay_index(...)`; **decide `has_color` once up front** (a scan either has RGB or not — probe the raw source, or always write RGB and substitute a zeros block when a chunk's `rgb` is None, matching how `materialize` handles `any_rgb_none`) so the final header agrees with every body record. For each `(xyz, rgb, cls, inst)` from `materialize_raw(index, ...)`, run `apply_filters_remap(cls, inst, confirmed_by_inst, req, src_to_tgt)` (+ drop_unlabeled → mask out `-1` points and their xyz/rgb), append `_ply_labeled_chunk_bytes(...)` to a **body temp file**, and accumulate the kept-point count. After the loop, the final PLY = `_ply_labeled_header(total_count, has_color)` + the body temp's bytes. (The binary header needs the vertex count, which filtering/streaming doesn't know upfront — hence body-temp-then-prepend-header.)
4. `build_manifest(taxonomy, p50, p90, ..., filters={...}, absent_count=absent_count, exported_at=<stamped here>)`.
5. Write PLY + manifest.json into a **zip temp file** (`ZIP_STORED` or `ZIP_DEFLATED` level 1). Return `FileResponse(zip_path, filename="scan_labeled_<kind>.zip", background=BackgroundTask(unlink))`.
6. **`try/finally`** unlinks the temp PLY/body/zip on ANY exception (BackgroundTask only fires on success). Write temps under a dedicated dir.

- [ ] **Step 1: Failing tests** — an annotated fixture at `resolution=scan`: response is a valid zip containing `scan_labeled.ply` + `manifest.json`; PLY vertex count == scan.ply count; manifest classes/accuracy present. A `confirmed_only`/exclude/remap case: labels reflect the filter in the PLY. A `raw` case with a small linked-raw fixture: streams and produces raw-count vertices. Scene/session-mismatch → 409. An empty-after-filters export → valid (0-labeled) zip, no 500.
- [ ] **Steps 2-5:** implement; commit `feat(export): POST /api/labels/export (materialize+filter+remap → zip)`.
- Register the route if `routes/export.py`'s router isn't already mounted for this path (it is — `/api/edit/export-ply` lives here).

---

### Task 6b: `GET /api/labels/accuracy` (Review-step p90)

**Files:** Modify `backend/routes/export.py`; Test `backend/tests/test_export_labels.py`.

A small read-only endpoint the wizard's Review step calls to show the real p50/p90 for the loaded scan.

- [ ] **Step 1: Failing test** — `GET /api/labels/accuracy?scene=<id>&session_id=<id>` on a loaded fixture returns `{"p50":.., "p90":..}` with `p90 >= p50 >= 0` matching `raw_sample_spacing(ctx.scan_pos)` on the fixture; scene/session mismatch or no scene loaded → 409.
- [ ] **Step 2: Run → fails.**
- [ ] **Step 3: Implement** — same scene/session guard as the export endpoint; `p50, p90 = raw_sample_spacing(_state["pc"].points)`; return `{"p50": p50, "p90": p90}`.
- [ ] **Steps 4-5:** Run → pass; commit `feat(export): GET /api/labels/accuracy for the wizard Review step`.

---

### Task 7: frontend pure-function helpers (vitest)

**Files:** Create `frontend/src/export-wizard-util.js` + `frontend/src/export-wizard.test.js`.

Pure fns (testable without DOM): `remapToTaxonomy(classes, remapRows, includeSet)` → the resulting `{id:{label,color}}` preview; `estimatePoints(resolution, scanCount, rawCount)` → the point-count estimate; `pointsAfterFilters(perClassCounts, confirmedOnly, includeSet, ...)` → the Review-step estimate (for the ~0 disable-Export guard).

- [ ] **Steps 1-5:** Failing vitest for each (merge maps two classes → one taxonomy entry; estimate returns scanCount/n/rawCount by kind; ~0 estimate when all classes excluded); implement; commit `feat(export): wizard pure-function helpers + tests`. (Run `npx vitest run src/export-wizard.test.js` from `frontend/`.)

---

### Task 8: Export wizard modal

**Files:** Create `frontend/src/export-wizard.jsx`; Modify `frontend/src/api.js` (`exportLabels(cfg)` → POST, receive blob).

3-step modal (Resolution → Classes → Review) per spec §1: resolution radios with point estimate + raw-disabled-when-`!rawSourceAvailable` + multi-GB warning; confirmed-only toggle + include/exclude list + merge/rename rows (a class in a merge row is disabled in the exclude list and vice versa); Review shows the target taxonomy + the p90 accuracy line, fetched from `GET /api/labels/accuracy` (built in Task 6b) so the real computed p90 is shown (never a hardcoded literal — spec §4). Export → `api.exportLabels(cfg)` → `blob` → `URL.createObjectURL` → `<a download>` → revoke. Busy state; surface 422/409 inline (not `alert`).

- [ ] **Step 1:** Implement the modal + `exportLabels`. (No FE component-test infra; the reducer/estimate logic is already unit-tested in Task 7.)
- [ ] **Step 2:** `npm run build` clean.
- [ ] **Step 3: Browser-verify** (REQUIRED SUB-SKILL: browser-verification) — memory `feedback_browser_verify_mutates_session`: export is READ-ONLY so it won't mutate, but still use a scratch session and restart a stale `:8765`. Load an annotated scan, open Export, run a `scan`-resolution export with a class merge, confirm the zip downloads and contains a valid PLY + manifest with the merged taxonomy; zero console errors; raw radio disabled when no raw source. Screenshot the wizard.
- [ ] **Step 4: Commit** `feat(export): export wizard modal + download`.

---

### Task 9: Export button + accuracy GET (wire-up)

**Files:** Modify `frontend/src/mode-label.jsx` (Export button in the header near Save; opens the wizard, passes `cloud.rawSourceAvailable` + scene/session); `frontend/src/api.js` map `rawSourceAvailable` + add `getAccuracy(scene, session)` calling the Task-6b endpoint.

- [ ] **Steps 1-5:** Button gated on an active session; opens wizard; `api.js` maps `raw_source_available → rawSourceAvailable` in `decodeLoadResponse`. Build + browser-verify the button appears and opens the wizard. Commit `feat(export): Export button in Label header`.

---

## Done criteria (Phase B)

- `POST /api/labels/export` returns a zip (`scan_labeled_<kind>.ply` + `manifest.json`) for `scan`/`subsample`/`raw`, with confirmed-only / include-exclude / class-remap / drop-unlabeled honored in the PLY and the manifest taxonomy/accuracy/provenance correct.
- Raw export streams (no whole-cloud in RAM); temp files cleaned on success AND failure.
- Wizard: 3 steps, raw gated on `rawSourceAvailable`, contradiction-safe class UI, real p90 in Review, zip download; zero console errors.
- Full backend suite + frontend vitest green; browser-verified.

## Not in this phase / follow-ups

- Multi-scene batch, train/val split, `.npy`/other formats, server-side output, reusable remap presets, per-instance manifest rows, source/UTM output frame — all deferred (spec §Out of scope).
- Vectorizing `replay_labels`' per-point loop — deferred to Phase A follow-up; if a real raw export is too slow, measure then optimize.

## Docs

Last step before the Phase-B PR: update `CLAUDE.md` (add the export wizard + `/api/labels/export` to the mode/endpoint notes) and flip the export-wizard spec Status to "implemented".
