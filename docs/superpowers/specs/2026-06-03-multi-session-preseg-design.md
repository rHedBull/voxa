# Multiple labeling sessions per scan, pinned to presegment results (scan-schema v2)

**Date:** 2026-06-03
**Status:** Approved design, pre-implementation

## Problem

Scan-schema v1.3 allows exactly one labeling session per scan (`session/`), one
presegment result (`prelabel/ransac_*`), and one ground truth (`labels/gt_*`).
SAM3 and RANSAC pipelines can each produce candidate presegmentations, but only
one can be "active" at a time (by overwriting `prelabel/`), and there is no way
to run two labeling attempts side by side — e.g. one seeded from RANSAC and one
from a SAM3 render run — and compare the finished results.

## Goals

1. A scan can hold **N presegment results**, each individually addressable and
   fingerprinted.
2. A scan can hold **N labeling sessions**. Each session is pinned at creation
   to one presegment result (or none, for a blank start); the pin is immutable
   and enforced via fingerprint hashes on every resume.
3. The UI (Label mode) lets the user list, create, rename, delete, and switch
   sessions, and choose the seeding preseg at creation time. Opening a scan
   resumes the **last-worked** session by default.
4. There is **no single canonical ground truth** anymore: each session has its
   own saved output, and downstream consumers enumerate `sessions/*/output/`
   and pick.

## Non-goals (YAGNI)

- Session duplicate/branch. Not requested; revisit if needed.
- Session-vs-session Compare mode. Compare stays GT-vs-prediction, with the
  active session's output as the GT side.
- Concurrent multi-session editing. The backend keeps ONE active in-memory
  `SegmentSession` at a time, matching the existing single-cloud `_state`
  model. Switching sessions = flush autosave, load the other.
- Automatic promotion of `sam3/<run_id>/` outputs into `prelabel/`. Pipelines
  call `register_preseg()` explicitly.

## Decisions made (with alternatives considered)

| Decision | Chosen | Rejected alternatives |
|---|---|---|
| GT model | Per-session outputs only; no canonical `labels/` | (a) single GT with sessions as drafts, (b) per-session outputs + explicit promote step |
| Preseg store | Formalized `prelabel/<preseg_id>/` dirs; `sam3/` stays a pipeline workspace | Discover-in-place registry over `prelabel/` legacy files + `sam3/<run_id>/` |
| Session layout | Self-contained `sessions/<session_id>/` (pin + working state + output + history in one dir) | Split `session/<id>` (working) + `labels/<id>` (output) |
| Migration | One-shot in-place script; voxa v2 reads ONLY v2 | Read-compat shim (dual code paths forever); auto-migrate-on-load (read mutates disk) |
| UI ops | Create + select, rename, delete, auto-resume last-worked | Duplicate/branch |

## On-disk layout (scan-schema v2)

```
<scan_name>/
├── meta.json                          (required)  provenance — schema_version: 2
├── source/                            (unchanged) scan.ply (required), mesh.glb (optional)
├── prelabel/                          (optional)  N presegment results
│   └── <preseg_id>/                   e.g. ransac, sam3_orbit48
│       ├── instance_ids.npy           int32, shape (N_pts,)
│       ├── segment_summary.json       { "segments": [{"id": int, "class_id": int}, ...] }
│       └── meta.json                  { "preseg_id", "generator", "params",
│                                        "fingerprint", "source_fingerprint", "created_at" }
├── sessions/                          (optional)  N labeling sessions
│   └── <session_id>/                  e.g. 20260603-143000_ransac
│       ├── session.json               pin + state (schema below)
│       ├── working_class_ids.npy      int8,  shape (N_pts,)  — autosave
│       ├── working_segment_ids.npy    int32, shape (N_pts,)  — autosave
│       ├── output/                    written by Save (Ctrl+S); absent until first save
│       │   ├── gt_class_ids.npy       int32, shape (N_pts,)
│       │   ├── gt_segment_ids.npy     int32, shape (N_pts,)
│       │   └── gt_segment_metadata.json
│       └── history/<YYYYMMDD_HHMMSS>/ per-session save backups (keep 10 most recent)
├── renders/<run_id>/                  (unchanged)
└── sam3/                              (unchanged)  feature cache + pipeline workspace
```

Removed relative to v1.3: top-level `labels/`, `session/`, `annotation_history/`
(absorbed into `sessions/<id>/`).

- `preseg_id` and `session_id` are directory names: stable, filesystem-safe
  (`[a-z0-9_-]+`), never renamed. A session's display `name` lives in
  `session.json`; rename is a metadata edit only.
- `session_id` is generated at creation as `<YYYYMMDD-HHMMSS>_<preseg_id|blank>`.
- All per-point arrays in `prelabel/*/` and `sessions/*/` MUST have shape
  `(N_pts,)` matching `source/scan.ply`.
- The dtype asymmetry is intentional and carried over from v1.3: working
  class ids are `int8` (autosave compactness), output `gt_class_ids` are
  `int32` — do not unify.
- v1.3 invariants carry over per session output: invariant 3
  (`class_id == -1 ⟺ instance_id == -1`), invariant 4 (per-segment class
  consistency), invariant 6 (`meta.json::class_map_version` ==
  `classes.json::version`).

## Session pinning

`sessions/<id>/session.json`:

```json
{
  "schema_version": 2,
  "name": "first pass with sam3",
  "preseg_id": "sam3_orbit48",
  "preseg_fingerprint": "sha256:…",
  "source_fingerprint": "sha256:…",
  "created_at": "2026-06-03T14:30:00Z",
  "saved_at": "2026-06-03T15:10:42Z",
  "dirty": true,
  "is_from_prelabel": true,
  "hidden_inst_ids": []
}
```

`preseg_id`/`preseg_fingerprint` are `null` for blank-start sessions.

- **Fingerprints** use the existing `segment_io.compute_fingerprint()` (sha256
  over dtype + shape + bytes). Preseg fingerprint covers `instance_ids.npy`;
  source fingerprint covers the recentered cloud positions (as today).
- **At creation**: working arrays are seeded from `prelabel/<preseg_id>/`
  (or all `-1` for blank); both fingerprints are computed and frozen into
  `session.json`. This is the only moment a preseg is chosen.
- **On resume** (`/api/load`): the backend recomputes both fingerprints and
  compares against the pins. **Any mismatch → HTTP 409** stating which pin
  diverged. No silent fallback, no auto-repair. A deleted/regenerated preseg
  dir makes its pinned sessions refuse to resume (session data stays intact;
  restoring the preseg dir restores resumability).
- This generalizes v1.3's `_stale_prelabel_check` + recovery gating
  (`app/core.py:269-336`): same mechanism, now per-session and symmetric.

## Backend

### `ScanLayout` (`backend/scenes/scan_layout.py`)

Single-slot properties (`labels_dir`, `gt_*`, `prelabel_dir`, `ransac_*`,
`session_dir`, `annotation_history_dir`) are replaced by:

```python
layout.presegs_root                 # prelabel/
layout.preseg_dir(preseg_id)        # prelabel/<id>/
layout.sessions_root                # sessions/
layout.session(session_id)          # SessionPaths view:
#   .session_json .working_class_ids .working_segment_ids
#   .output_dir .output_gt_class_ids .output_gt_segment_ids
#   .output_gt_segment_metadata .history_dir
```

All call sites (`scene_registry`, `lidar_io`, `segment_io`, save route,
stale-prelabel check) update in the same change; no v1.3 accessors remain.

### New module `backend/labeling/session_store.py`

The only code that touches `sessions/*` structure:

- `list_sessions(layout) -> list[SessionInfo]` — id, name, preseg_id,
  saved_at, dirty, has_output.
- `create_session(layout, name, preseg_id: str | None) -> SessionInfo` —
  seeds working arrays, freezes pins, writes `session.json`. Fails if
  `preseg_id` given but `prelabel/<id>/` missing or malformed.
- `rename_session(layout, session_id, name)`
- `delete_session(layout, session_id)` — removes the whole dir.
- `last_worked(layout) -> session_id | None` — max `saved_at` (falls back to
  `created_at` for never-saved sessions).
- `verify_pins(layout, session_id, source_fp)` — raises a typed error naming
  the diverged pin (consumed by the load route → 409).

### New module `backend/preseg/preseg_store.py`

- `list_presegs(layout) -> list[PresegInfo]` — from `prelabel/*/meta.json`.
- `register_preseg(layout, preseg_id, instance_ids, summary, generator,
  params)` — writes the three files, computing fingerprints at write time.
  Offline pipelines (`scripts/preseg/presegment*.py`, SAM3 post-process) call
  this instead of writing `prelabel/ransac_*` directly.

### Adapted, not duplicated

- `SegmentSession` (`labeling/segment_state.py`) remains the single in-memory
  working state with its existing autosave debounce; it is constructed against
  `sessions/<id>/` paths. `_aux_payload()` becomes the `session.json` body
  (gains `name`, `created_at`; `preseg_run_id` renamed `preseg_id`).
  `segment_io.SESSION_SCHEMA_VERSION` bumps 1 → 2; `saved_at` continues to be
  stamped in `save_session_aux()` at write time, not in `_aux_payload()`.
- `save_labels()` (`labeling/segment_io.py`) writes to
  `sessions/<id>/output/` and `sessions/<id>/history/`.
- `load_prelabel()` (`segment_io.py`) takes a `preseg_id`.

### API

| Route | Change |
|---|---|
| `GET /api/scenes/{id}/sessions` | new — list sessions |
| `POST /api/scenes/{id}/sessions` | new — body `{name, preseg_id?}` → create |
| `PATCH /api/scenes/{id}/sessions/{sid}` | new — body `{name}` → rename |
| `DELETE /api/scenes/{id}/sessions/{sid}` | new — delete; requires `?confirm=true` |
| `GET /api/scenes/{id}/presegs` | new — list preseg results |
| `POST /api/load` | gains optional `session_id`; default = last-worked; if the scan has no sessions, loads cloud only and reports `sessions: []` (UI then shows the picker). Pin verification here → 409 `{detail, diverged: "preseg"|"source"}`. The existing `keep_prev_seg` carry-over heuristic (`routes/load.py:49-56`) is removed — "switch session = flush autosave + load the other" supersedes it |
| `PUT /api/segment/save` | unchanged signature; writes into the active session's `output/` |

Session CRUD routes operate on disk via `session_store` and do not require the
scan to be the loaded scene, except `DELETE` of the currently-active session,
which also clears the in-memory state.

## Frontend

- **`api.js`**: `listSessions`, `createSession`, `renameSession`,
  `deleteSession`, `listPresegs`; `load()` takes `sessionId`.
- **`App.jsx`**: owns `sessions` / `activeSessionId` beside `activeScene`. On
  scene change: fetch sessions → auto-load last-worked, or surface the picker
  in create mode when none exist. A 409 pin error renders as a blocking error
  banner (scene stays unloaded), never an empty canvas.
- **Session picker** (Label mode panel, beside the existing class/segments
  panels): rows show name, preseg badge, saved_at, dirty dot; actions: select,
  inline rename, delete (confirm dialog), `+ New session` (name field + preseg
  dropdown from `/presegs`, including "blank"). Switching away from a dirty
  session prompts for confirmation (autosave makes the switch safe; the prompt
  is for explicitness).
- **Inspect mode** shows the active session's per-point labels (same
  `LoadResponse.class_ids/instance_ids` path, sourced from the session).
- **Compare mode** uses the active session's `output/` as the GT side;
  predictions side unchanged.

## Migration

`scripts/migrate_scan_v2.py` — one-shot, in-place, idempotent (skips scans
whose `meta.json` already says `schema_version: 2`):

```
labels/gt_*                  → sessions/legacy/output/
session/{current.json,*.npy} → sessions/legacy/  (session.json synthesized
                               from current.json; name "legacy";
                               created_at/saved_at from file mtimes)
annotation_history/*         → sessions/legacy/history/
prelabel/ransac_*            → prelabel/ransac/{instance_ids.npy,
                               segment_summary.json, meta.json (generator
                               "ransac", fingerprint computed)}
meta.json                    → schema_version: 2
```

- A scan with `labels/` but no `session/` still gets `sessions/legacy/` with
  working arrays copied from the GT (so resuming continues from the save).
- The script refuses loudly per scan on anything unexpected (extra files in
  `prelabel/`, shape mismatches) rather than guessing; `--dry-run` prints the
  plan.
- Voxa v2 reads ONLY v2. A v1.3 scan fails scene discovery with a
  "run scripts/migrate_scan_v2.py" hint.
- The annotated-scan scaffolder is `engine/data/tools/scaffold_annotation.py`
  (sibling data repo, **outside the voxa tree**) — it must be updated to emit
  the v2 skeleton, as a companion change in that repo. Voxa's own
  `scripts/import_scene.sh` only feeds the legacy `data/scenes/` workflow and
  is untouched.
- `docs/scan-schema.md` and the shared `lidar/SCHEMA.md` are updated in the
  same PR; sibling tools adopt v2 when they next touch the archive.

## Error handling summary

| Failure | Behavior |
|---|---|
| Resume with diverged preseg/source fingerprint | 409 with diverged pin named; UI blocking banner |
| Create session with missing/malformed `prelabel/<id>/` | 4xx with reason |
| `session.json` unreadable / arrays wrong shape | Session listed as corrupt (not hidden), resume refused with reason |
| Delete active session | Allowed; clears in-memory state; UI returns to picker |
| v1.3 scan encountered | Not discovered; log hint to run migration |

## Testing

- **`session_store` unit tests**: create (seeded + blank), list, rename,
  delete, last-worked ordering, `verify_pins` happy path + preseg mismatch +
  source mismatch.
- **`preseg_store` unit tests**: register + list + fingerprint stability.
- **Route tests** (existing pytest pattern; `conftest.py` sets `VOXA_DATA_DIR`
  before importing `main` — preserve that ordering): session CRUD, load with
  explicit/default session, 409 on tampered `instance_ids.npy`, save lands in
  the active session's `output/`, invariants 3/4/6 still enforced per session.
- **Migration tests**: synthetic v1.3 fixture → migrate → assert v2 layout;
  idempotency (second run is a no-op); refusal case; labels-without-session
  case.
- **Frontend**: vitest for new `api.js` mappers (current setup is
  pure-function only); session picker verified manually in the browser per the
  browser-verification rule.
