# Voxa per-point segment editing — Label-mode aided-labeling design

**Status**: design draft 2026-04-28
**Scope**: Voxa Label mode gains a per-point segment-editing toolkit so an annotator can correct a model-produced prelabel (`<lidar>/annotated/<scene>/prelabel/`) into authoritative ground truth at `<scene>/labels/gt_*.npy`. The segmentation-model CLI that produces `prelabel/` is a sibling spec, not part of this work.

## 1. Goal

Take a model-produced prelabel (per-point `class_ids` + `instance_ids`), let an annotator correct it through fast segment-level operations and a 3D sphere brush, and save authoritative ground truth back to the SCHEMA-conformant paths:

- `<scene>/labels/gt_class_ids.npy`
- `<scene>/labels/gt_segment_ids.npy`
- `<scene>/labels/gt_segment_metadata.json`

This closes the labeling side of the active-learning loop: prelabel → human correction → GT → retrain. Only the human-correction half is in scope here.

## 2. In scope

- Label mode gets a second tool family — **segment tools** — alongside the existing cuboid tools. Cuboid editing is unaffected.
- Five operations on per-point class + instance IDs: change class, merge segments, split (brush-peel), boundary touch-up (brush-absorb), create instance from unlabeled points (brush + class).
- Loading `prelabel/` arrays as the editing starting point when `labels/` is empty; loading `labels/` directly when GT already exists.
- Saving authored labels back to the SCHEMA paths; recomputing `gt_segment_metadata.json`; rolling backups under `annotation_history/`.
- Undo/redo for per-point operations, bounded to 100 entries per scene.
- Visual feedback: live "color by instance" / "color by class" recolor during edits, brush gizmo, dirty indicator.

## 3. Non-goals

- Building the segmentation-model CLI that produces `prelabel/` — a sibling spec on the segmentation-research side. This design assumes those files exist on disk.
- Multi-scene workflows. Voxa's `_state` stays single-cloud.
- Per-point editing in Inspect or Compare modes.
- Connectivity-graph editing (`connectivity_graph.json`).
- Lasso, screen-space selection, cut-plane — only the 3D sphere brush ships.
- Retraining loops, dataset bookkeeping, label-quality scoring.
- Don't entangle with cuboid state. Cuboid tools and segment tools share Label mode but not state. Hotkeys are scoped to the active sub-tool. Undo histories are separate. Save fires both whatever cuboid edits are dirty and whatever per-point edits are dirty, but each writes its own files.

## 4. Label-mode tool structure

Label mode gains a sub-tool selector. Today it's implicitly "cuboid" — that becomes one tool family among several.

| Family | Tools | When active |
|---|---|---|
| **Cuboid** (existing) | Place, edit, auto-fit | Existing behaviors unchanged |
| **Segment** (new) | Pick, Brush | New per-point editing |

### 4.1 Pick tool (key `V`, default segment tool)

- Click a point → select the segment it belongs to.
- Shift-click → add to multi-selection.
- Hotkeys on the active selection: `P / T / E / S / D` apply class (`pipe / tank / equipment / structural / double`) per `config/classes.yaml`. `R` is a special sentinel for unlabeled (`class_id = -1`, `instance_id = -1`) — not a config lookup; wired directly to the `-1` state.
- With ≥2 segments selected, `M` merges into the lowest segment ID; the merged ID becomes the new selection.

### 4.2 Brush tool (key `B`)

- 3D sphere brush, world-units radius, mouse-wheel adjusts radius (geometric ×1.2 / ÷1.2). Default 0.05 m. Radius persists across tool switches and scene loads.
- Cursor follows the **first-hit point** under the mouse via raycast against the cloud's BVH. Empty space → gizmo dims, clicks no-op.
- Brush has a **destination** state shown in the HUD:
  - With one segment selected when entering Brush: `{mode: 'absorb', destInstance: that segment, destClass: that segment's class}` → painting reassigns points within the sphere to the selected segment (**boundary touch-up**).
  - With nothing selected: `{mode: 'create', destInstance: null, destClass: current class hotkey}` → first stroke allocates a new instance ID = `max(instance_ids) + 1`; subsequent strokes append. This handles **split** (brush inside an existing segment) and **create-from-unlabeled** (brush over `-1` points).
  - Alt held during stroke → temporarily `{mode: 'erase', destInstance: -1, destClass: -1}`. Releases back on stroke end.
- Esc returns to Pick.

### 4.3 Tool switching

A vertical tool strip on the left edge of Label's viewport shows: Cuboid, Pick, Brush. Active tool is highlighted. Switching tools clears any in-flight selection in the previous tool. The keybinds `V` and `B` jump directly. The existing cuboid hotkey routes to Cuboid.

### 4.4 Cursor / interaction conflicts

Orbit/Walk camera controls and tool clicks share left-click. Existing voxa convention: orbit drag uses the empty viewport, click-on-something uses left-click. We follow that. Right-drag remains pan. Brush painting is left-click-and-drag; there's no orbit during brush — orbit by releasing, repositioning, then painting.

### 4.5 Mode-switch behavior

Switching out of Label (to Inspect/Compare) with unsaved per-point edits prompts the same "save first?" dialog the cuboid editor uses — they share one dirty-state hook on `App.jsx`.

## 5. Brush mechanics

Per stroke (each pointer move while left-button is held):

1. Re-raycast → new sphere center.
2. Spatial-index range query for points within `R` of center, on the **full-resolution** point set (not the subsampled render set).
3. Optional depth gate: cull points whose distance **along the camera-ray axis** is more than `2 * R` farther than the cursor hit (with `R` = brush radius in world units, same `R` as the sphere). Avoids painting through occluders. Operates along the camera ray, not world-Z.
4. Apply destination assignment (existing instance, new instance, or `-1`) to the resulting indices.

Continuous strokes accumulate indices into one edit; mouse-up commits one undo entry per stroke.

### 5.1 Spatial index location

Backend computes a `scipy.spatial.cKDTree` once at scene load and serves range queries via `/api/segment/brush-query`. We don't ship the tree to the frontend — a JS kd-tree build per scene is 1–2 s of jank. Backend KD-tree query at R=5 cm on a 2 M-point cloud returns in ~1–2 ms; per-stroke RTT on localhost is sub-5 ms.

If RTT ever feels bad, the fallback is a frontend kd-tree (e.g., `three-mesh-bvh` adapted, or a simple kd-tree). Not a day-one need.

## 6. Backend additions

All new endpoints under `/api/segment/`, all operating on the in-memory `_state` cloud.

| Method | Path | Body | Response |
|---|---|---|---|
| `POST` | `/api/segment/brush-query` | `{center: [x,y,z], radius: float, depth_cull: float?}` | `{indices: b64 Int32, n: int}` (full-res indices) |
| `POST` | `/api/segment/apply` | `{op: "set_class" \| "merge" \| "reassign", indices: b64 Int32, payload: {...}}` | `{dirty: true, n_affected, new_instance_id?}` |
| `POST` | `/api/segment/undo` | `{}` | inverse delta to broadcast to frontend |
| `POST` | `/api/segment/redo` | `{}` | symmetric |
| `PUT`  | `/api/segment/save` | `{}` | flushes labels to disk; writes `annotation_history/<ts>/` unless disabled |

### 6.1 Why backend-side state for undo

The single source of truth for full-res labels is the backend (matches the existing single-cloud `_state` pattern). Frontend keeps a downsampled mirror for rendering. Round-tripping deltas through the backend keeps both consistent. Undo/redo stacks live with the data they invert.

### 6.2 Undo entry shape

```python
{
  "op": str,
  "indices": np.ndarray[int32],
  "before": {"class": np.ndarray[int8], "instance": np.ndarray[int32]},
  "after": {"class": np.ndarray[int8], "instance": np.ndarray[int32]},
}
```

Bounded to last 100 entries per scene; older entries drop. Stack lives keyed by `(scene_id, "segment")` and clears on `/api/load`.

### 6.3 Loading

Existing `load_annotated` already returns `LabelArrays`. Extend `lidar_io.py` decision tree:

1. `labels/gt_*.npy` exist → use them, ignore `prelabel/`. `is_from_prelabel = false`.
2. Else `prelabel/ransac_instance_ids.npy` exists → seed editable state from it (instance ids verbatim; class ids derived from `prelabel/ransac_segment_summary.json` lookup). `is_from_prelabel = true`.
3. Else → all-`-1`, still editable. `is_from_prelabel = false`.

Prelabel arrays themselves are never written. Only `labels/gt_*.npy` is written by save. Re-running the segmentation pipeline overwrites `prelabel/` without touching authored ground truth.

### 6.4 Save format

Per SCHEMA: `int32` arrays of shape `(N_pts,)`, `-1` = unlabeled. After-save validation:

- Invariant 3: `class_ids[i] == -1 ⟺ instance_ids[i] == -1`.
- Invariant 4: per-segment class consistency (every point sharing a segment id has the same class id).

Save fails (HTTP 400) with diagnostic on violation rather than silently corrupting.

`gt_segment_metadata.json` is fully recomputed from the arrays on save: list segments by ID, count points, take majority class (which after invariant 4 is unique), compute AABB. No drift between arrays and metadata.

### 6.5 Annotation history

`annotation_history/<UTC YYYYMMDD_HHMMSS>/` is written before every save by default (per SCHEMA's optional convention; SCHEMA shows `YYYYMMDD_HHMM` as an example, voxa uses seconds granularity to avoid same-minute collisions during rapid save-and-correct cycles). 16 MB per save for a 2 M-point scene; capped at 10 most-recent snapshots per scene; older directories pruned on save. Disabled via env `VOXA_DISABLE_ANNOTATION_HISTORY=1`.

**Pruning identification.** Voxa only prunes directories whose name matches the strict regex `^\d{8}_\d{6}$` (UTC `YYYYMMDD_HHMMSS`). User-curated subdirectories with any other name (e.g. `pre-merge-2026-04-28`, `manual-backup`) are left alone. This protects external tools or annotators that drop snapshots under the same parent.

## 7. Frontend state & wire format

### 7.1 Cloud payload extensions

Existing `LoadResponse` already carries `class_ids` / `instance_ids` as b64 arrays in subsampled form. For Label-mode segment editing we need full-res arrays; piggy-back via opt-in:

```
POST /api/load  body: { scene, want_full_labels: true }
→ adds: full_class_ids   (b64 Int8),
        full_instance_ids (b64 Int32),
        full_positions   (b64 Float32),
        full_n: int,
        is_from_prelabel: bool,
        segment_summary: { [instance_id]: {class_id, n_points, bbox} } | null
```

Subsampled arrays remain in their existing fields and drive the rendered point cloud. Full arrays drive editing math.

`want_full_labels` is opt-in. Inspect and Compare modes continue to send the default (`false`) and pay no extra bandwidth. Only Label mode sets it to `true`. The `LoadResponse` Pydantic model in `backend/main.py` gets the new fields (`full_class_ids`, `full_instance_ids`, `full_positions`, `full_n`, `is_from_prelabel`, `segment_summary`) declared explicitly as `Optional[...]`, defaulting `None` when the flag isn't set.

### 7.2 Render path during edit

Three.js `BufferGeometry` color attribute is recomputed when class/instance IDs of the **subsampled** points change. After each edit the frontend updates only the affected subsampled indices' colors, via the subsample → full index map already returned by `/api/load`. Avoids full re-tint per stroke.

### 7.3 Frontend state shape

Lifted into `App.jsx` alongside `cloud`, `gtInstances`, etc.:

```js
segState = {
  classFull: Int8Array,
  instanceFull: Int32Array,
  segmentSummary: Map<id, {classId, nPoints}>,  // recomputed locally on each apply
  dirty: boolean,
  selection: Set<int>,            // selected instance IDs (Pick tool)
  brush: { radius: float, mode: 'absorb'|'create'|'erase', destInstance: int|null, destClass: int },
  activeTool: 'cuboid'|'pick'|'brush',
  isFromPrelabel: boolean,
}
```

### 7.4 Optimistic apply

Frontend mutates `classFull` / `instanceFull` immediately on stroke commit and only awaits backend confirmation. On mismatch (e.g. validation error) backend wins — frontend re-syncs from server response.

### 7.5 Undo

Triggered via `Cmd/Ctrl+Z` and `Cmd/Ctrl+Shift+Z`. Calls backend `/api/segment/undo`, applies returned delta. Frontend doesn't keep its own stack — single source of truth on backend.

## 8. Save semantics & dirty state

Single Cmd/Ctrl+S entry point (existing). The save handler now fires *both*:

- Cuboid save (existing PUT to `/api/annotations/<scene>/gt`) — only if cuboid-dirty.
- Per-point save (new PUT to `/api/segment/save`) — only if seg-dirty.

Each path errors independently and surfaces a per-family toast: "Cuboids saved" / "Segments saved" / "Save failed: <invariant>".

**Order**: segments first, then cuboids. On segment failure, cuboids are not saved either — atomic-ish from the user's perspective. If cuboids fail (rare), segments don't roll back; surfaced in toast.

Explicitly **not** a 2-phase commit. No transactional rollback machinery. The save handler is a sequential try/catch where the second step is gated on the first step succeeding. The annotation-history snapshot written before save (§6.5) is the recovery affordance, not transactional logic.

**Dirty indicator.** Title bar shows `●` when either family is dirty. Tooltip on hover shows which.

**No autosave.** Ctrl+S only. `beforeunload` warns on dirty state (cuboid editor already does this; reused). Mode-switch shows confirm modal with Save / Discard / Cancel.

## 9. Visual feedback

### 9.1 Color modes during edit

Inspect-mode's `colorMode` pills are reused inside Label/Segment. New default in Label: `instance` (so segment boundaries are visible). Toggle to `class` for class-fix sessions, `rgb` to sanity-check that what the model called a "tank" actually is.

### 9.2 Selected-segment styling

Active segment(s) get a brighter outline color (additive cyan tint) overlaid on whatever the base color mode is. Multi-select tints all selected segments.

### 9.3 Brush gizmo

Three.js `Mesh` of a `SphereGeometry` at radius R, semi-transparent, color-coded by destination state:

- Absorb: tint of destination segment's class color.
- Create: tint of active class hotkey color.
- Erase (Alt held): muted gray.

Follows the cursor in real time via raycast against the cloud's BVH.

### 9.4 Diff vs. prelabel

Optional toggle "Show edits" overlays a 2-color highlight: changed-since-prelabel vs unchanged. Cost: keep `prelabelClassFull` / `prelabelInstanceFull` arrays in frontend at load, never mutated. Useful for sanity-checking how much you've moved from the model's proposal.

## 10. Testing plan

### 10.1 Backend tests (pytest, `backend/tests/`)

- `test_segment_load.py` — `load_annotated` returns full-res labels when `labels/` exists; falls back to prelabel; falls back to all-`-1`. Uses a tiny synthetic SCHEMA scene fixture under `tmp_path`.
- `test_segment_apply.py` — each op mutates arrays as expected; SCHEMA invariants 3 & 4 hold post-op; `gt_segment_metadata.json` stays consistent.
- `test_segment_undo.py` — apply N edits, undo N times → identical to start. Redo restores. Stack bounded at 100; older entries drop without error.
- `test_segment_save.py` — round-trips arrays to disk, recomputes metadata, writes `annotation_history/<ts>/` when not disabled. Save fails loud on invariant violation.
- `test_brush_query.py` — KD-tree query matches a brute-force NumPy mask for a 10k-point synthetic cloud at several radii and centers; depth-cull behaves correctly.

### 10.2 Frontend tests (vitest, pure-function only — matches existing setup)

- `segState` reducers — apply / undo / redo on small fixture arrays.
- Subsample → full index mapping correct after edits (color-update path).
- Wire (de)serialization for the new b64 fields.

### 10.3 Manual validation gate

Per CLAUDE.md ("if you can't test the UI, say so"). After implementation: fire `npm run dev`, load `annotated/munich_water_pump`, verify the golden path: prelabel loads → pick a segment → change class → merge two segments → brush-split a segment → save → reload → labels persisted, metadata consistent. One negative path: introduce an invariant-violating state via dev tools and confirm save fails cleanly.

## 11. Risks & mitigations

1. **Per-stroke RTT chop.** 30–60 Hz brush queries to backend may stutter on slow loops. Mitigation: backend handler is a single cKDTree query (~1 ms); FastAPI sync handler is fine. Fallback: build a JS kd-tree client-side. Decision deferred — measure first.
2. **Memory ceiling on full-res arrays.** A 10 M-point scene means `Float32Array(3e7) + Int32Array + Int8Array ≈ 160 MB` in browser. Voxa's existing scenes top out at ~2 M sampled; raw LAZ can be 50 M+. Mitigation: editing requires a labelable subsample density anyway (1.5 cm voxel for `munich_water_pump` ≈ 2 M). Hard cap at `VOXA_MAX_LABEL_POINTS = 5_000_000`; refuse to enter Segment-tool mode on larger clouds with a clear error.
3. **Single-cloud `_state` clash with concurrent users.** Two annotators on the same backend → last save wins. Existing voxa already has this property for cuboids; we don't make it worse. Documented limitation.
4. **`annotation_history/` disk growth.** 16 MB per save × frequent saves. Mitigation: cap at 10 most-recent snapshots per scene; older directories pruned on save.
5. **Undo across mode switches.** Backend keeps the stack keyed by `(scene_id, "segment")` until scene unloads. Cleared on `/api/load`. Mode-switching within the same scene preserves history.
6. **Cuboid + segment dirty interleaving.** Atomic-ish via segment-first save order (§8).

## 12. Open questions / future

- Do we want a "review mode" with a sortable segment list (id, class, n_points, bbox) to find tiny segments / mislabeled outliers fast? Probably yes — but as a follow-up spec, not in this MVP.
- Should the segment summary include stats (e.g., max-IoU vs prelabel) so the annotator can sort by "most uncertain"? Requires the segmentation model to emit per-segment confidences in `prelabel/`. Coupled to the sibling segmentation-CLI spec.
- Multi-scene batch labeling. Out of scope; the in-memory `_state` would need lifting first.
