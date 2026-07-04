# Voxa Refactor Notes

Consolidated notes for the refactor pass. We go **file by file**; the user dictates notes
and I record them here. In-code `todo!` markers will be swept in later and merged below.

- Started: 2026-05-30
- Convention: in-code markers use `TODO:` (grep target for the later sweep).

---

## Open notes (by file)

<!-- Append per-file notes here as we go. Newest file sections added at the bottom. -->

### `backend/main.py`
Reviewed in this pass. No `TODO:` markers found in the file (56 lines — already slim).
_Add verbal notes here if any._

### `backend/routes/meta.py`
- **L34** (`get_config` defaults): `# TODO: defaults make no sense` — the hardcoded fallback class
  list (boss/fastener/gasket/...) used when `classes.yaml` is absent is arbitrary/wrong.
- **L57** (`list_scenes`): `# TODO: is this still up to date?` — the `predictions.json` existence
  check (`has_predictions`); unsure the path/logic still matches current schema.
- **L72** (end of file): `# TODO: fix where to load the scenes from?` — scene discovery needs to
  adapt to the **new scene data saving schema v1.3**, where a *decimated* point cloud is NOT a
  separate scene (currently it may be treated as one).

### `backend/routes/load.py`
- **L13** (`load_scene`): `# TODO: any loading refactorable, repeated implementation, ...`
  ✅ **MOSTLY DONE (2026-05-30):** load.py + lidar_io shared "repeated implementation" addressed —
  - **Dedup (625cf70):** `load_laz`/`load_laz_region` shared `_laz_rgb_to_uint8` (LAS 8/16-bit color
    decode); z-up rotation single-sourced via new `lidar_io.z_up_to_y_up_xyz` (was duplicated as
    `core._z_up_to_y_up` *and* inline in `load_laz_region` — a drift/correctness risk).
  - **Extraction (d3dbbbe):** the 232-line `load_scene` shed ~66 lines — session-seeding →
    `core._seed_or_recover_session`, stale-prelabel → `core._stale_prelabel_check`.
  - Still open: the "file paths?" sub-question + the LoadResponse field audit below.
- **L218** (`LoadResponse` construction): `# TODO: check if all still needed!?` — ⏳ STILL OPEN:
  verify every field returned in `LoadResponse` is still consumed by the frontend (dead-field hunt).

### `backend/routes/preseg.py`
- **L272–273** (after `optimize/abort`): `# TODO: ... presegmentation now happens offline ...`
  — the new intended usage is that **presegmentation runs offline (not UI-triggered)**; voxa just
  reads presegmentation results from the scene data directory. So the `POST /api/segment/presegment`
  (and the whole optimize/status/abort cluster?) may be **dead or repurposable** — e.g. collapse to
  a single "reload preseg results from disk" endpoint. Ties directly into the "endpoint count high"
  theme: this is where several of the 14 segment/preseg endpoints could disappear.

### `backend/routes/segment.py`
- **L122** (`PUT /api/segment/save`): `# TODO: this is very important — what is saved where, what
  format?` Traced it:
  - **Where:** only allowed on `annotated/<scene>` tier. Writes to `scan_dir = <source>/../..`
    via `labeling.segment_io.save_labels`, into the scan's `labels/` dir (plus optional history,
    gated by `VOXA_DISABLE_ANNOTATION_HISTORY`).
  - **What/format:** `class_ids` + `instance_ids` int arrays (per-point), aligned to `positions`,
    stamped with `prelabel_fingerprint` + `source_fingerprint`. Preseg-only points
    (`inst≥0 & class==-1`) are **stripped to -1 on export** (SCHEMA invariant 3: class==-1 ⟺
    inst==-1) because preseg is a suggestion, not authoritative GT.
  - **Two-write subtlety:** `flush_autosave()` runs FIRST (writes `session/working_*.npy` from the
    *unmutated* working canvas) so reload can recover preseg; THEN the sanitized `labels/` export.
    The in-memory session keeps full `instance_ids`. Returns
    `{ok, n_labeled_points, n_segments, n_dropped_preseg}`.
  - _Refactor angle:_ the save vs autosave split + the strip-on-export rule is subtle and
    history-prone (see the comment about a prior bug that collapsed prelabel segments to -1). Worth
    documenting/centralizing rather than re-deriving.

### `backend/routes/export.py`
Reviewed. **Seems reasonable; leave as-is.** No markers. User note: not used that often
(`POST /api/edit/export-ply`, single endpoint) — low priority for the refactor.

### `backend/preseg/` (whole dir reviewed, 2026-05-30)
Recurring theme across the dir: **presegmentation now runs offline**, so UI-triggered selection
machinery is obsolete. Markers:
- **presegment.py:56** — `# TODO: believe this is also obsolete, since preseg mode selection from
  UI no longer happens, all presegmentation happens offline`. This is the `presegment(mode=...)`
  **dispatch** (voxel / ransac / model→NotImplemented). If the UI no longer picks a mode, this
  dispatcher + the `POST /api/segment/presegment` endpoint can likely go (or collapse to "load
  results from disk"). Note: `presegment_voxel.py` / `presegment_ransac.py` are still the offline
  engines — they stay; it's the *runtime dispatch from the request* that's in question.
- **supervoxels.py:134** — `# TODO: believe supervoxels then also not needed anymore`.
  ✅ **DONE (2026-05-30): deleted `backend/preseg/supervoxels.py`.** Verified zero real importers
  (only a provenance docstring in segment_hulls.py:3 referencing the *old* 3d-labeler tool, left
  intact). 229 backend tests pass before & after. clustering.py + fitting.py still pending.
- **resolver.py:82** — `# TODO: what is resolved here? still needed if all data fits schema v1.3?`
  ✅ **ANSWERED (2026-05-30): KEEP — not obsolete.** It's live SERVER code on the load path:
  `POST /api/load` (load.py:204) → `verify_scan_registration` (registration.py, §6 frame-check gate)
  → `dir_cloud_transforms` (resolver.py). 8 tests in test_resolver.py. Resolves cloud↔render-pose
  alignment when they're different *variants* of the same scan: `use_direct` (identity, exact match),
  `remap` (cross-variant render reuse), `fail` (different scan / bad pin → `frame_registration_failed`
  surfaced to UI). The v1.3 premise is the tell: perfect data → always `use_direct` → resolver is a
  quiet no-op, i.e. **validation passing, not dead code**. It's the layer that *detects* imperfect
  data (the `fail` safety net) + enables cross-variant reuse (the `remap` feature, per HANDOFF). Only
  possible simplification = drop `remap` IF cross-variant render reuse is provably never used — a
  behavior decision on real data, not a cleanup. No change made.
- **sam3_features.py:414** — `# TODO: this is what's needed for the current preseg_sam3d pipeline
  based on sam3d feature vectors, right?` Refers to `extract_or_load` (SAM3 feature extract+cache).
  User is *confirming* this IS the live/keep path (the current SAM3D feature-vector pipeline) — i.e.
  the opposite of dead. Mark as **authoritative / keep**, document its role so it's not mistaken for
  legacy alongside the dead RANSAC trio.

**Net for preseg/:** keep = presegment_voxel, presegment_ransac, preseg_optimize, sam3_features,
resolver (pending v1.3 check), registration. Cut/collapse = supervoxels, clustering, fitting (dead
trio) + the runtime `presegment` dispatch if UI mode-selection is truly gone.

---

## DECIDED: offline preseg model — keep/remove map (2026-05-30)
Goal: presegmentation runs **offline**; voxa loads results from the data dir and lets the user
**list + select** between available presegment variants via a light reload button.

### KEEP — offline pipeline
- **Loader (single path):** `labeling/segment_io.py::load_prelabel(scan_dir)` reads
  `prelabel/ransac_instance_ids.npy` + `ransac_segment_summary.json`; called by
  `lidar_io.py::load_annotated` on `POST /api/load`.
- **Generator scripts:** `scripts/presegment.py` (canonical RANSAC→prelabel),
  `scripts/presegment_sam3_features.py` (Stage 1 feats), `scripts/presegment_sam3.py` (Stage 2),
  `scripts/dry_sam3/*` (R&D).
- **Engines (used by scripts — NOT dead):** `presegment.py` dispatch, `presegment_voxel.py`,
  `presegment_ransac.py`, `sam3_features.py`, `resolver.py`, `registration.py`.
- ⚠️ Corrects earlier TODO at `presegment.py:56`: the `presegment()` dispatch is still used by the
  offline scripts; only the *route-driven* mode selection is obsolete.

### REMOVE — interactive-from-UI surface  ✅ DONE (commit 128ae70, 2026-05-30)
- **`routes/preseg.py` (all 5 endpoints)** — deleted whole file.
- **`preseg/preseg_optimize.py`** + `app/core.py::{_apply_ransac_result_to_session, _new_job_state,
  _preseg_optimize_worker}` — deleted. Removed now-unused `_threading` import (kept `uuid`: it's
  re-exported via `from app.core import *` for compare.py auto-fit — restored after a test caught it).
- **schemas.py** — removed Presegment*/PresegOptimize*/Sam3Render*/Sam3Params/RansacParams.
- **main.py** — dropped preseg router include (+ obsolete "legacy" docstring line).
- **Frontend:** removed `PresegmentButton` from `segment-tools.jsx` (kept `PresegmentList`); removed
  `segPresegment`, optimize trio, `sam3ListRenders` from `api.js`.
- **Tests:** removed `test_segment_presegment_endpoint.py`, `test_preseg_optimize.py`,
  `test_preseg_optimize_endpoint.py`. Kept `test_presegment.py` (engine).
- **Verified:** backend 210 passed, frontend 34 passed, prod build clean. 1728 lines removed.

### ARCHITECTURAL: preseg engines are now offline-only (2026-05-30)
After removing the interactive route, **no HTTP endpoint imports the preseg engines** anymore
(verified: grep of routes/ + app/ for presegment_voxel/ransac/dispatcher = empty). The engines are
called ONLY by `scripts/`, which reach into `backend/preseg/` via a `sys.path.insert(0, ROOT/
"backend")` hack (`scripts/presegment.py:41`, `presegment_sam3.py:39`).

Split of `backend/preseg/` after the removal:
- **Still server code (load path):** `registration` (`load.py` → `verify_scan_registration`),
  `resolver` (frame transforms, used by registration + sam3_features).
- **Now offline-only (scripts only):** `presegment` (dispatcher), `presegment_voxel`,
  `presegment_ransac`, `sam3_features`.

Implication: the engines are an **offline-pipeline library living under `backend/`**, shared via
the sys.path hack — arguably misplaced now that the server doesn't run them. Bigger move (extract a
shared lib / package) — parked, not in scope yet.

**Dispatcher (`presegment.py`) — ✅ REMOVED + inlined (commit d9bee7f, 2026-05-30).** Was pure
offline indirection once the UI route died (every caller passed a fixed mode). Inlined
`scripts/presegment.py`→`presegment_voxel`, `scripts/presegment_sam3.py`→`presegment_ransac` (dropped
`mode="ransac"`), `test_presegment.py`→`presegment_voxel`; deleted `presegment.py` (55 lines + dead
`model` branch). Lazy Open3D import preserved (each script imports only its engine). Verified:
scripts compile, engine imports resolve, test_presegment 4 passed, backend 210 passed.

### NEW — list + select presegments (feature, design pending)
Replaces the interactive trigger with: discover available presegment variants in the data dir →
list them in UI → user selects one → load into live session (light, no full scene reload).
Requires a storage convention for **multiple** variants (today scripts overwrite one
`prelabel/ransac_*`). Design grounded in scan-schema v1.3 — see next section.

**On-disk reality (munich_water_pump, 2026-05-30):** exactly ONE preseg —
`prelabel/ransac_instance_ids.npy` (int32 N) + `ransac_segment_summary.json`
(`{"segments":[{id,class_id}]}`). `sam3/` empty. Schema hints at `sam3/<run_id>/instance_ids.npy`
("per-render-set outputs") but nothing writes it. Loader = `load_prelabel`, hard-coded to the
single `ransac_*` pair. → To support list+select we need a multi-variant convention + migration +
generator-script + loader + schema-doc updates. **Storage convention = open decision.**

---

## Collected `todo!` markers (in-code sweep)

<!-- Populated later by grepping the codebase for `todo!`. Each entry: file:line — text. -->
_None yet — sweep not run._

---

## presegment.py was secretly voxel — fixed to RANSAC (62853e1, 2026-05-30)
`scripts/preseg/presegment.py` docstring + `ransac_*` output said RANSAC, but it imported
`presegment_voxel` and ran voxel (latent bug from old dispatcher default `mode="voxel"` — the call
omitted `mode=`). "We don't use voxel anymore" → switched to `presegment_ransac`. **`presegment_voxel.py`
is now R&D-only:** only `dry_sam3/preseg_with_features.py` (voxel+feature-merge experiment) + its
engine test `test_presegment.py` import it. Kept because dry_sam3 depends on it.

## dry_sam3 ↔ production integration (2026-05-30) — KEEP dry_sam3
Checked whether `scripts/dry_sam3/` R&D was absorbed into the production SAM3 pipeline
(`scripts/preseg/presegment_sam3*.py` + `backend/preseg/sam3_features.py` + `presegment_ransac.py`).
**Only the image-encoder FEATURE path was integrated; the entire text-prompt MASK path was not.**
- ✅ `extract_features.py` → `sam3_features.py::extract_or_load` (fully; +caching, +v1.3 remap).
- ✅ `ransac_with_features.py` → `presegment_ransac.py::_feature_split_instances` — *except* its
  optional feature-cosine **merge** pass (not ported; likely intentional per over-seg preference).
- ❌ `preseg_with_features.py` — voxel preseg + feature-cosine union-find **merge**; only here
  (production voxel preseg is plain spatial, no features).
- ❌ `project_masks.py` — text-prompt → fixed-taxonomy class voting + shared SAM3/camera lib
  (imported by auto_segment + prompt_segment).
- ❌ `prompt_segment.py` — prompt-driven mask-membership seg (most evolved, v1.3-aware).
- ❌ `auto_segment.py` — broad-prompt auto instance maps.
- Grep confirms NO `set_text_prompt`/`union_mask`/vote-argmax in backend/ or scripts/preseg/.
**Verdict: dry_sam3 is the sole home of 4 real capabilities — not obsolete, keep the folder.**

## Cross-cutting / themes

<!-- Patterns that span multiple files (state ownership, naming, boundaries, dead code). -->

### Dead code / orphan analysis (3-agent sweep, 2026-05-30)
Cross-referenced endpoints↔frontend callers, backend call graph, and frontend exports.

**✅ DONE — dead segment endpoints removed (commit 7a826d2, 2026-05-30):**
brush-query, hide+unhide, snap-to-preseg (HTTP surface + schemas + FE wrappers + the unused
`hidden_inst_ids` field + their endpoint tests). KEPT the SegmentSession methods + test_segment_state.py
(re-wireable later). sam3/renders went with the preseg route removal (128ae70). backend 204, FE 32, build clean.

**Dead endpoints — defined but NO frontend caller (5 of 25):**
| Endpoint | Handler | api.js wrapper | Status |
|---|---|---|---|
| GET `/api/sam3/renders` | preseg.py:168 | `sam3ListRenders` | no caller |
| POST `/api/segment/brush-query` | segment.py:13 | `segBrushQuery` | no caller |
| POST `/api/segment/snap-to-preseg` | segment.py:114 | `snapToPreseg` | no caller, no test |
| POST `/api/segment/hide` | segment.py:98 | `hideInstance` | test-only (segment-state.test) |
| DELETE `/api/segment/hide/{id}` | segment.py:106 | `unhideInstance` | test-only |
- Note: `/api/mesh/{tier}/{name}` has no api.js call but IS live (returned as `mesh_url`,
  fetched by GLTFLoader at viewer.jsx:1342) — **not** dead.
- These 5 + the offline-preseg direction (presegment + optimize cluster) are the bulk of the
  "too many endpoints" theme. Decision needed: wire into UI vs. delete.

**Backend modules with ZERO importers anywhere (not even tests) — the legacy RANSAC trio:**
✅ **ALL THREE DELETED (supervoxels 46b3da4; clustering + fitting f65a00d).**
- `preseg/fitting.py` (entire) — `fit_cylinder_ransac`, `fit_box_ransac`, `fit_boxes_in_region`,
  `refine_cylinder_fit`, … RANSAC cylinder/box fitting. ✅ **DELETED (f65a00d).**
- `preseg/clustering.py` (entire) — `region_grow`, `estimate_normals`. ✅ **DELETED (f65a00d).**
- `preseg/supervoxels.py` (entire) — `compute_supervoxels`. ✅ **DELETED 2026-05-30.**
- CLAUDE.md calls these "present for future use"; confirmed **still fully unwired, incl. tests.**

**Other orphaned backend functions (zero refs):**
- `extract_faces_from_mesh` — scenes/point_cloud.py:128
- `save_ply` — scenes/point_cloud.py:167
- `list_runs` — labeling/runs_io.py:38 (not referenced even by its own test module)

**Backend test-only / dormant (parked, verify product intent before deleting):**
- `labeling/seg_inference.py` (whole module) — sole production call site is **commented out** in
  scenes/lidar_io.py:174 ("restore the predict_for_scene call here"). Reachable only from its test.
- `labeling/run_merge.py` + `labeling/runs_io.py` — imported only by tests/test_runs.py.

**Frontend orphaned exports (no importer):**
- `snapToPreseg` — api.js:300 — fully dead (no test either). Highest-confidence delete.
- `TweakSelect` / `TweakText` / `TweakNumber` / `TweakButton` — tweaks-panel.jsx:348/362/371/415 —
  exported UI-kit atoms, never used. (`TweakSlider`/`TweakToggle`/`TweakColor` only appear in
  commented-out examples — functionally dead but grep-matched.) May be deliberately kept as a kit.
- Test-only: `hideInstance`/`unhideInstance` (api.js), `recomputeSummary` (segment-state.js:27).

**Live preseg map (reachable from routes/core):** presegment.py → presegment_voxel.py /
presegment_ransac.py; preseg_optimize.py; sam3_features.py → resolver.py; registration.py →
resolver.py. **Dead:** fitting.py, clustering.py, supervoxels.py.

### Endpoint count feels high (user observation, 2026-05-30)
**25 endpoints** total. Breakdown by router:
- `segment.py` — 9  (undo, redo, save, apply, hide, hide/{id}, brush-query, snap-to-preseg, state)
- `preseg.py` — 5  (presegment, optimize, optimize/status, optimize/abort, snap?)
- `compare.py` — 4
- `meta.py` — 3  (config, scenes, health)
- `load.py` — 3  (load, load-region, annotations get/put)
- `export.py` — 1

**Observation:** segment + preseg = **14 of 25** endpoints. Many are tiny stateful mutations on
the single in-memory `_state` (undo/redo/hide/apply/save/snap). Smells like an over-fragmented
RPC surface where a smaller, more cohesive API (or a single command/session endpoint) might do.
**To revisit:** which of these are actually called by the frontend, and whether the
undo/redo/apply/save cluster can collapse into fewer, fatter endpoints.
