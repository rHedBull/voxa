# Segment Linkage: Point Cloud ↔ Mesh ↔ Presegments ↔ Labels

**Status:** approved (2026-05-13), pending implementation plan
**Topic:** make voxa's editor state consistent across page reloads, server restarts, and preseg re-runs by adopting a two-layer point-indexed model that mirrors the on-disk SCHEMA.

## Problem

Voxa already aligns the four data surfaces (point cloud, optional mesh, presegments, labels) by **point index** as the universal join key. What it lacks is identity stability when the live session mutates: after a preseg run, the frontend caches segment IDs derived from the preseg snapshot; subsequent `apply_merge` / `apply_reassign` operations rewrite `instance_ids`, leaving the FE's cached IDs dangling.

The diagnosed symptom (2026-05-12 22:05) is the "hide" linkage breaking after reload — but it's a representative case of any FE feature that holds a segment ID across a mutation: hide, multi-select, favourites, references-from-annotations.

Three reload contexts to handle:

1. **Page reload** — server keeps `SegmentSession`; FE loses local caches (preseg run id, hide list, segId mappings).
2. **Server restart** — both server and FE lose all in-memory state.
3. **Preseg re-run** — overwriting `instance_ids` from a new preseg snapshot silently clobbers any labels the user already applied on top.

## Goals

- Identity for "where did this point come from in the preseg run?" is stable for the lifetime of the scene, regardless of how many merges/reassigns happen.
- All editor state needed to resume work survives page reloads and server restarts.
- Re-running preseg never silently destroys in-progress labels.
- Schema stays clean: the dataset SCHEMA carries authoritative outputs; per-tool session state lives in a clearly-separated subdir.
- Adding a connectivity-graph mode later is drop-in (path reserved, segment-id keyed).

## Non-goals

- Cross-scene ID stability (instance_ids stay scene-scoped).
- Mesh-face ↔ point linkage (no current consumer; mesh stays a view-only overlay with shared coordinate frame).
- A separate stage 2 / stage 3 split (voxa intentionally collapses these into one Edit mode).
- Versioned label files (`.v1/.v2/...`) — `annotation_history/<timestamp>/` and `session/preseg_runs/<run_id>.npz` already cover provenance.

## Prior art

The sibling tool at `../industrial-point-labeler` (see `docs/design.md`) ships a five-stage pipeline whose key idea — parallel point-indexed int32 arrays per stage, immutable upstream + mutable downstream, with a server-persisted session aux file — is what this design adapts. Voxa collapses the stages but keeps the two-layer + session-aux pattern.

The canonical dataset SCHEMA at `engine/data/lidar/SCHEMA.md` already provides the on-disk shape for the two layers: `prelabel/ransac_instance_ids.npy` (immutable auto-seg) vs `labels/gt_segment_ids.npy` + `gt_class_ids.npy` (editable GT). This spec brings voxa's in-memory model and per-tool state files into alignment with that schema.

## Design

### 1. Universal join: point index `i` ∈ `[0, N)`

The canonical `PointCloud` is the spine. All per-point arrays (`class_ids`, `instance_ids`, `preseg_ids`, optional intensity, optional SAM3 features) are length-N parallel arrays indexed by point order. **Invariant: the point order of `_state["pc"]` never changes during a session.** Subsampling produces a view (`subsample_idx`); it never re-spines.

Mesh ↔ point bridge stays implicit through shared 3D coordinates; no per-face ID linkage.

### 2. Two-layer point-indexed identity in `SegmentSession`

```python
class SegmentSession:
    # Point-indexed arrays, parallel to positions[N,3]
    preseg_ids:        np.ndarray   # int32[N]; -1 = no preseg; FROZEN after freeze_preseg()
    instance_ids:      np.ndarray   # int32[N]; mutable; what the labeler edits
    class_ids:         np.ndarray   # int8[N];  mutable in-memory (cast to int32 on Save → labels/gt_class_ids.npy)
    positions:         np.ndarray   # float32[N,3]; reference to pc.points

    # Provenance + auxiliary editor state (persisted to session/current.json)
    preseg_run_id:        Optional[str]   # which saved run was loaded (or None for ad-hoc preseg)
    preseg_fingerprint:   Optional[str]   # sha256 of preseg_ids.tobytes(); content-addressed, survives renames
    source_fingerprint:   Optional[str]   # sha256 of positions.tobytes()
    hidden_inst_ids:      set[int]
    is_from_prelabel:     bool

    # In-memory only (not persisted)
    _undo: deque[_Delta]
    _redo: deque[_Delta]
    dirty: bool
```

Mutations:
- `apply_set_class`, `apply_merge`, `apply_reassign` — operate on `class_ids` and `instance_ids` only. `preseg_ids` is immutable through every op, including undo/redo. The existing `_Delta` payload stays unchanged.
- `freeze_preseg(ids, run_id, fingerprint, *, seed_instances: bool)` — replaces `preseg_ids` wholesale and stamps `preseg_run_id` + `preseg_fingerprint`. The freeze itself is **not undoable** (session-scope, not a point-edit). When `seed_instances=True` (preseg-load behavior, applied to all unlabeled points), the instance/class seeding is implemented as a normal `_apply("seed_preseg", ...)` that *is* on the undo stack — so an accidental "load preseg run" can be reverted point-by-point even though the frozen preseg layer remains updated. When `dirty=True` the caller (HTTP endpoint or FE) must surface a confirmation before passing `seed_instances=True`.
- `hide(inst_id)` / `unhide(inst_id)` — mutate `hidden_inst_ids`. Triggers an auto-save of `session/current.json`. Not on the undo stack (visual affordance, not data).

Resolution helpers:
- `points_for_preseg(preseg_id) → indices`: `np.flatnonzero(self.preseg_ids == preseg_id)`.
- `current_inst_ids_for_preseg(preseg_id) → set[int]`: `set(np.unique(self.instance_ids[self.preseg_ids == preseg_id]))`. This is the "what did preseg cluster 42 become?" query that hide-after-merge needs.
- `snap_to_preseg(inst_ids)` — set `instance_ids[mask] = preseg_ids[mask]` for points whose live instance is in `inst_ids`. Useful for "revert this object back to its preseg clusters". On the undo stack.

Memory cost at 3M points: +12 MB (int32 × N). Acceptable.

### 3. Persistent session aux on disk

Mirror the schema's `prelabel/` vs `labels/` split with a new optional `session/` subdir per scene:

```
<scan_name>/
├── source/scan.ply               (existing)
├── prelabel/                     (existing)
│   ├── ransac_instance_ids.npy
│   └── ransac_segment_summary.json
├── labels/                       (existing, canonical GT — only written on explicit Save)
│   ├── gt_class_ids.npy
│   ├── gt_segment_ids.npy
│   └── gt_segment_metadata.json  (+ new fields, see §4)
├── session/                      (NEW; optional; not part of GT contract)
│   ├── current.json
│   ├── working_class_ids.npy     (optional in-progress class array)
│   ├── working_segment_ids.npy   (optional in-progress segment array)
│   ├── preseg_runs/<run_id>.npz  (relocated from voxa's data/preseg_runs/<scene>/)
│   └── graph_review.json         (reserved for future graph mode; unused today)
└── annotation_history/           (existing)
```

`session/current.json` schema:
```json
{
  "schema_version": 1,
  "preseg_run_id": "20260513-143012",
  "preseg_fingerprint": "sha256:...",
  "source_fingerprint": "sha256:...",
  "hidden_inst_ids": [12, 47],
  "is_from_prelabel": false,
  "dirty": true,
  "saved_at": "2026-05-13T14:55:01"
}
```

Auto-save policy:
- Every successful `_apply` (class/inst mutation, **including undo/redo**) schedules a save. Saves are coalesced with a single-flight + 250 ms trailing debounce; a pending save is replaced rather than queued. On session shutdown, the debounce is flushed.
- Per save: write `working_class_ids.npy.tmp`, `working_segment_ids.npy.tmp` → fsync → rename each to its final name → then write `current.json.tmp` → fsync → rename. **`current.json` is the commit pointer.** On reload, a `working_*` set is honored only if `current.json` exists and references it (via `working_mtime_ns` recorded in current.json). A crash between the two npy renames leaves the old `current.json` pointing at the previous-but-consistent state — acceptable, the half-updated working set is ignored.
- `hide` / `unhide` / `freeze_preseg` also use the debounced save path; they only need `current.json` to change but still go through the single-flight to avoid interleaving with an in-flight working_* write.
- Explicit Save (Ctrl+S) flushes any pending debounce, then atomically promotes `working_*` to `labels/gt_*` (`gt_class_ids.npy` cast `int8 → int32`), updates `gt_segment_metadata.json` (with new fingerprint fields), and clears `dirty`. `working_*` files are left in place so reload still works; the next `_apply` will overwrite them.

Reload policy (`/api/load`):
1. Read `labels/gt_*` (existing path).
2. If `session/current.json` exists AND `source_fingerprint` matches the current `scan.ply` AND `session/working_*` exists with newer mtime than `labels/gt_*` → load working arrays into `SegmentSession`, hydrate aux from `current.json`, surface "resuming in-progress session" to the FE.
3. Else load `labels/gt_*` (or seed from `prelabel/` if requested) and start a fresh session.
4. If `source_fingerprint` mismatches → discard session as stale (with FE-visible warning), fall through to step 3.

For non-SCHEMA scenes (raw LAZ, legacy `scenes/<name>/source.{ply,glb}` tier), voxa keeps its existing `data/sessions/<scene_id>/` registry mirroring the same layout, since there is no canonical scan dir to live inside.

### 4. Fingerprint fields in `labels/gt_segment_metadata.json`

Additive, optional:
```json
{
  "n_points": 500000,
  ...
  "prelabel_fingerprint": "sha256:abc...",   // the prelabel run these GT labels were seeded from
  "source_fingerprint":   "sha256:def..."    // scan.ply at label time
}
```

On reload, when seeding from `prelabel/` while `labels/` already exists, a fingerprint mismatch surfaces a non-fatal warning ("auto-seg has been re-run since these labels were saved — continue, reseed, or back up?") instead of silent overwrite. Same mechanism `industrial-point-labeler` uses.

### 5. New / changed backend endpoints

- `GET /api/segment/state` — hydration endpoint. Returns the full session aux:
  ```json
  {
    "has_seg": true,
    "n_points": 500000,
    "n_segments": 126,
    "preseg_run_id": "20260513-143012",
    "preseg_fingerprint": "sha256:...",
    "source_fingerprint": "sha256:...",
    "hidden_inst_ids": [12, 47],
    "is_from_prelabel": false,
    "dirty": true
  }
  ```
  Frontend calls this after `/api/load` to populate UI state from a single server-owned source.

- `POST /api/segment/hide` with `{inst_id: int}` and `DELETE /api/segment/hide/{inst_id}` — mutate `hidden_inst_ids` and trigger `current.json` auto-save.

- `POST /api/segment/snap-to-preseg` with `{inst_ids: [int]}` — invoke `SegmentSession.snap_to_preseg`. On the undo stack.

- `POST /api/segment/presegment` and `POST /api/segment/presegment/runs/{run_id}/load` — change to call `SegmentSession.freeze_preseg(...)` instead of directly assigning `instance_ids`. Existing response shape unchanged.

- Existing endpoints unchanged otherwise.

### 6. Frontend changes

- On mount (and after `/api/load`): call `/api/segment/state` and hydrate into a single `segState` slice. Drop any local `presegRunMeta` cache that previously lived outside `segState`.
- Hide / unhide: route through API; remove local `hiddenSet` ownership across reloads.
- Preseg-aware UI (cluster outlines, "select all from same preseg cluster") becomes possible via `preseg_ids` returned with the full payload — no schema change needed there; preseg_ids piggybacks on the existing `full_*` blob in `LoadResponse`.

### 7. Backward compatibility

- The SCHEMA additions in §3 and §4 are optional fields and an optional subdir. v1 scenes still load.
- A separate `lidar/SCHEMA.md` patch (v1.1) documents `session/` and the two fingerprint fields. The SCHEMA invariants section stays unchanged — `session/` is explicitly outside the GT contract.
- Voxa scenes from the legacy `data/scenes/` tier (no `prelabel/` or `labels/` on disk) keep working through the in-tool `data/sessions/<scene_id>/` shim.

## Architecture diagram

```
                 ┌─────────────────────────────────────────────┐
                 │  on-disk per scene (SCHEMA-conformant)      │
                 │                                             │
                 │  source/scan.ply           ← spine          │
                 │  prelabel/ransac_*.npy     ← immutable      │
                 │  labels/gt_*               ← canonical GT   │
                 │  session/current.json      ← editor state   │
                 │  session/working_*.npy     ← in-progress    │
                 │  session/preseg_runs/      ← preseg history │
                 └────────────────┬────────────────────────────┘
                                  │ load / save
                                  ▼
                 ┌─────────────────────────────────────────────┐
                 │  backend in-memory (per active scene)       │
                 │                                             │
                 │  _state["pc"]: PointCloud[N,3]   ← spine    │
                 │  _state["mesh"]: trimesh         ← view     │
                 │  _state["seg"]: SegmentSession              │
                 │    preseg_ids[N]    (immutable layer)       │
                 │    instance_ids[N]  (mutable layer)         │
                 │    class_ids[N]                             │
                 │    hidden_inst_ids                          │
                 │    preseg_run_id + fingerprints             │
                 │    undo/redo                                │
                 └────────────────┬────────────────────────────┘
                                  │ /api/load + /api/segment/state
                                  ▼
                 ┌─────────────────────────────────────────────┐
                 │  frontend                                   │
                 │  - hydrates segState from server on mount   │
                 │  - never owns IDs across reload             │
                 │  - mesh companion mirrors via               │
                 │    BroadcastChannel (unchanged)             │
                 └─────────────────────────────────────────────┘
```

## Behavior matrix (before vs after)

| Scenario | Today | After |
|---|---|---|
| Page reload after preseg + merge + hide | Hide breaks (FE-cached preseg id no longer present in `instance_ids`) | Hide persists; `preseg_ids` lets FE re-render preseg outlines |
| Server restart mid-session | Lose hide, lose preseg-run binding, lose in-progress edits | All restored from `session/current.json` + `session/working_*.npy` |
| Re-run preseg with active labels | `instance_ids` silently overwritten | `preseg_ids` updates, `instance_ids` + `class_ids` retained; fingerprint mismatch surfaces warning |
| "Snap object back to preseg clusters" | Not possible | `snap_to_preseg(inst_ids)` |
| Adding graph mode later | Requires session-aux scaffolding | Drop graph_review.json into `session/`, key edges by `instance_ids` |

## Files touched (estimate)

- `backend/segment_state.py` — add `preseg_ids`, `preseg_run_id`, fingerprints, `hidden_inst_ids`, `freeze_preseg`, `hide/unhide`, `snap_to_preseg`, helpers. Auto-save hook in `_apply`. ~120 LOC.
- `backend/segment_io.py` — `save_session_aux(path, session)`, `load_session_aux(path) → dict`, atomic write helpers. ~80 LOC.
- `backend/main.py` — wire `session/` paths into load/save flow; new endpoints `GET /api/segment/state`, `POST/DELETE /api/segment/hide`, `POST /api/segment/snap-to-preseg`; route preseg apply/load through `freeze_preseg`. ~80 LOC.
- `backend/scene_registry.py` — surface `session_dir` on `SceneSource` for SCHEMA-conformant scenes; fall back to `data/sessions/<scene_id>/` for non-SCHEMA tiers. ~30 LOC.
- `frontend/src/segment-state.js` — hydration from `/api/segment/state`, drop local hide cache. ~40 LOC.
- `frontend/src/segment-tools.jsx` — route hide through API; add "snap to preseg" affordance behind existing menu. ~40 LOC.
- `frontend/src/api.js` — three new endpoint wrappers. ~20 LOC.
- Tests:
  - backend: freeze_preseg + merge + hide round-trip; reload after server restart; preseg re-run preserves labels; fingerprint mismatch warning. ~6 tests.
  - frontend: hydration on mount; hide survives merge. ~2 tests.
- SCHEMA patch: `engine/data/lidar/SCHEMA.md` → v1.1 (additive — `session/` + fingerprint fields).

## Open questions

None remaining. Approach approved 2026-05-13.

## References

- `../industrial-point-labeler/docs/design.md` — multi-stage prior art
- `../../data/lidar/SCHEMA.md` — canonical dataset schema (v1)
- `backend/segment_state.py` — current SegmentSession
- `backend/main.py:500-620` — current load + seg-preservation logic
- `.remember/now.md` 2026-05-12 22:05 — diagnosed hide bug
