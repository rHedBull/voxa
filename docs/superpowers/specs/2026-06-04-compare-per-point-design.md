# Compare view v2: per-point source comparison

**Date:** 2026-06-04
**Status:** Approved design, pre-implementation

## Problem

The Compare view predates scan-schema v2. It compares **cuboid** annotations
from the legacy `data/annotations/<scene>/{ground_truth,predictions}.json`
files — which nothing writes for annotated v2 scans — so for real scans the
view is empty and unused. Meanwhile the original motivation for multi-session
labeling was exactly "run two labeling attempts side by side and compare the
finished results", and that comparison does not exist.

## Goals

1. Compare two **per-point labelings** of the same scan side by side:
   session vs session, or session vs model prediction.
2. Per-point class metrics: overall agreement, per-class point-IoU /
   precision / recall, top confusion pairs.
3. Visual: synced split view, each side colored by its source's classes.

## Decisions (made during brainstorming)

| Decision | Chosen | Rejected |
|---|---|---|
| What to compare | Session vs session; session vs model prediction | Session-vs-preseg as an advertised feature; keeping the cuboid GT-vs-pred view |
| Prediction ingestion | Predictions are registered as presegs (`register_preseg(generator="model-…")`) — a comparison source is any session output or preseg | A new `predictions/` schema slot |
| Metric level | Per-point class metrics only | Instance-level matching (TP/FP/FN rows); disagreement color-mode visualization |
| Cloud identity | Both sources MUST have arrays of identical length (same cloud assumed); mismatch → 409 | Alignment/resampling across cloud variants |
| Session data | **Saved output** (`sessions/<id>/output/gt_class_ids.npy`) — compare *finished* exports | Working arrays (live unsaved state); working-with-output-fallback |
| Compute location | Server (numpy), one round-trip returning metrics + both class arrays | Client-side JS diff |

Note: a preseg is selectable as a source, so session-vs-preseg works
mechanically; it is simply not a design target (no special UI for it).

## Comparison sources

A source is addressed as `{"kind": "session" | "preseg", "id": "<id>"}`.

- `session` → `sessions/<id>/output/gt_class_ids.npy` (int32 on disk).
  Sessions without an `output/` are NOT comparable: the UI lists them
  disabled with "no output — save first"; the API returns 409.
- `preseg` → `prelabel/<id>/instance_ids.npy` + `segment_summary.json`,
  class ids derived exactly as `preseg_store.load_preseg` does (reuse it).
  Model predictions enter the system this way: an inference pipeline calls
  `register_preseg(layout, "model_x", instance_ids, summary=…,
  generator="model-x", …)` and immediately becomes comparable (and
  selectable as a session seed — that duality is intentional).

`-1` means unlabeled on both sides.

## Backend

### New module `backend/labeling/compare_points.py`

Pure numpy, no FastAPI, no `_state`:

- `compare_class_arrays(a: np.ndarray, b: np.ndarray) -> dict` — both int
  arrays of equal length (caller validates). Returns:
  - `n_points`
  - `n_labeled_a`, `n_labeled_b` (class >= 0)
  - `agreement`: fraction of points with `a == b` over points labeled in
    **at least one** side (points unlabeled in both are excluded — they
    carry no signal and would inflate agreement on sparse labelings)
  - `agreement_all`: fraction over ALL points (for reference)
  - `per_class`: for each class id appearing in either side:
    `{class_id, iou, precision, recall, n_a, n_b}` where, treating A as
    reference: `tp = (a==c)&(b==c)`, `iou = tp / ((a==c)|(b==c))`,
    `precision = tp / (b==c)` (of B's claims, how many match A),
    `recall = tp / (a==c)`. Division by zero → `null`.
  - `confusion`: top 20 `{a_class, b_class, n}` pairs where both sides are
    labeled and disagree, sorted by count descending.

### Route (in `backend/routes/compare.py`)

`POST /api/compare-points/{tier}/{name}` with body
`{"a": {"kind", "id"}, "b": {"kind", "id"}}`:

1. Resolve the scene (`_resolve`); 409 unless tier == `annotated`.
2. Load each source's class array from disk (helper `_load_source(layout,
   kind, id)`): session output (404 unknown session, 409 no output) or
   preseg via `preseg_store.load_preseg` (404/400 per its errors). No
   dependency on the in-memory loaded scene.
3. Length mismatch between a and b → 409 with both lengths in the message.
4. Respond: `{metrics: <compare_class_arrays result>, a_class_ids: <b64
   int8>, b_class_ids: <b64 int8>, palette: [...]}` — class arrays cast to
   int8 for the wire (same convention as `LoadResponse.class_ids`; class
   ids are validated < 127 at save/registration already), palette built the
   same way the load route builds it (classes.json + fallback colors).

### Deletions

- `POST /api/compare/{scene}` (cuboid diff) and its schemas
  (`CompareRequest`, `CompareResponse`, `DiffRow`) are removed.
- `_iou_aabb` and the greedy cuboid matcher in the route are removed
  (verify no other caller first; auto-fit does not use them).
- The annotations GET/PUT endpoints **stay** — Label mode cuboids use them.
- `VoxaAPI.compare` is replaced by `VoxaAPI.comparePoints(scene, a, b)`.

## Frontend (`mode-compare.jsx` rework)

- **Top bar**: two source dropdowns (A tagged green "Source A", B tagged
  blue "Source B") listing sessions (disabled when `!has_output`, suffix
  "no output") and presegs (suffix "preseg · <generator>"); then
  `agreement %` (the at-least-one-labeled variant, with `agreement_all` in
  a tooltip) and labeled counts `A: n · B: n`. Camera Sync / View / nav
  controls stay as today.
- **Defaults**: A = the active session if it has output, else the most
  recently saved session; B = the next distinct source (most recent other
  session output, else first preseg). If fewer than two comparable sources
  exist, show an empty-state: "Need two finished labelings — save a second
  session or register a model preseg."
- **Split view**: both sides render the same cloud; each side's points are
  colored by its source's classes using the palette from the response —
  full-res arrays projected onto the subsampled cloud via
  `cloud.subsampleIdx` (the existing App.jsx pattern). Cuboid props are
  dropped from the panels. Synced cameras unchanged.
- **Table**: per-class rows — class (name + color dot), IoU, P, R,
  `pts A`, `pts B` — sorted by IoU ascending (worst first, that's where
  the action is). Below it a compact confusion list: "Pipe → Tank ·
  12,304 pts" (top pairs).
- **Tier guard**: non-annotated scenes render the empty state "Compare
  needs an annotated scan" (no cuboid fallback).
- Sessions/presegs lists come from App's existing state (already fetched
  for the Label-mode picker); App passes them down. The compare request
  itself is issued by CompareMode on source change.

## Error handling

| Failure | Behavior |
|---|---|
| Source session has no output | API 409; UI disables such sessions in the dropdowns up front |
| Unknown session/preseg id | 404 |
| Array length mismatch (different cloud variants) | 409 naming both lengths; UI shows the message inline |
| Non-annotated tier | 409 from the route; UI never issues the call (guard + empty state) |
| < 2 comparable sources | UI empty state, no request |

## Testing

- `compare_points` unit tests with hand-computed fixtures: agreement (incl.
  both-unlabeled exclusion), `agreement_all`, per-class IoU/P/R incl.
  zero-division nulls, confusion ordering/truncation.
- Route tests: session-vs-session happy path (metrics + b64 arrays round-
  trip), session-vs-preseg, no-output 409, unknown id 404, length-mismatch
  409, non-annotated 409. Cuboid-compare tests deleted with the endpoint.
- Frontend: vitest for the response decoder (b64 → arrays + metrics
  mapping); split view + dropdowns verified in the browser.

## Docs

`CLAUDE.md`'s Compare description updates from "GT vs prediction with
server-computed precision/recall/F1/IoU + per-instance TP/FP/FN" to the
per-point model; `docs/scan-schema.md` gains one line noting model
predictions are registered as presegs.
