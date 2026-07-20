# Outlier Detection Filtering — Design

**Date:** 2026-07-20
**Status:** Approved (design), pending implementation plan

## Problem

Voxa point clouds carry two kinds of stray points that pollute labeling:

1. **General stray points** — random speckle scattered through the cloud (sensor
   noise, reflections, moving objects). Floating specks that belong to no object.
2. **SAM projection edge-strays** — the SAM 2D→3D back-projection drags in points
   around the mask border that are not actually part of the segmented object.

Both are the same shape of problem: *a set of points that is mostly one coherent
object plus some spatially-isolated junk*. That is exactly what statistical
outlier removal (SOR) is good at, so **one core algorithm serves both**.

## Principle

Outlier detection is **just another way to select points** — it fits voxa's
existing `select → apply+label → unconfirmed pointset → confirm` pipeline verbatim.
It introduces no new labeling pipeline: it is a new *selection source* (feature C)
and a *selection refinement* (feature B) over infrastructure that already exists.

## Non-goals

- **No destructive cloud edits.** Nothing removes points from `scan.ply` or the
  working arrays. Outliers become a *label* (Exclude) or leave a *selection*; both
  are fully reversible (delete the instance / undo).
- **No global reversible view-filter.** An earlier option (denoise as a hide/show
  toggle) was dropped in favor of materializing outliers as an Exclude instance —
  it reuses the pipeline and is visible/auditable in the Instances panel.
- **No preseg shrinking** (see Feature B scope below).
- **No auto-clean of SAM masks at `/project` time.** Cleanup is user-triggered via
  feature B, not automatic during back-projection.

## Shared algorithm — `backend/labeling/outliers.py`

One pure, unit-testable function:

```python
statistical_outlier_indices(
    positions: np.ndarray,      # (N, 3) full-res points
    subset_idx: np.ndarray,     # int indices into positions defining the population
    k: int = 16,
    std_ratio: float = 2.0,
) -> np.ndarray                 # int indices (into positions) flagged as outliers
```

Build a `cKDTree` over `positions[subset_idx]`, compute each subset point's **mean
distance to its k nearest neighbors** (excluding self), and flag points whose mean
exceeds `global_mean + std_ratio · global_std` of that distance distribution.
Returns the flagged **outlier** indices, expressed in the full-res index space.

- **Density-adaptive:** the threshold is derived from the subset's own distance
  distribution, so it self-scales across the wildly varying local densities in
  plant scans. (Radius Outlier Removal was rejected — a fixed radius breaks across
  density regimes.)
- **One knob:** `std_ratio` drives aggressiveness (lower = greedier). `k` is fixed
  at 16 (surfaced as advanced-only if it ever proves necessary).

**The population (`subset_idx`) is what differs between the two features:**

| Feature | `subset_idx` | Why a stray gets flagged |
|---------|--------------|--------------------------|
| C (global) | *all* points | a floating speck is far from *any* neighbor cloud-wide |
| B (selection) | the selection's members | an edge-stray is far from the selection's *core* even though it sits near real geometry cloud-wide |

Grounding: `SegmentSession` already holds full-res `self.positions` and a
**lazily-built** `cKDTree` (`self._tree` via `_ensure_tree()`, `segment_state.py:261-265`
— built on first query, not at construction). Global SOR calls `_ensure_tree()` and
reuses `self._tree`; selection SOR builds a small tree over the selected subset.

### Cost

Synchronous request with a spinner. SOR runs at **scan.ply resolution** (the
labeling resolution — the working arrays; single-digit-million points for typical
scans, e.g. smart_ais ≈ 4.99M), which is a few seconds for a `cKDTree` build plus a
`k=16` query. Not the raw ~156M cloud.

## Feature C — global denoise → Exclude

Detect outliers cloud-wide and materialize them as **one unconfirmed `pointset`
instance with class `unknown`** (id 6, "Exclude / Review", hotkey `0` in
`config/classes.yaml`).

### Flow (review-first)

1. User clicks **"Detect outliers"** in Label mode's left rail (near the Export
   button; disabled without an active session).
2. Backend runs SOR over all positions at the current **aggressiveness** slider
   value → outlier indices.
3. Outliers arrive as an **unconfirmed** Exclude pointset, highlighted like any
   selection. User eyeballs what was caught.
4. Too greedy / too timid → adjust the slider and **re-run**. Each run **replaces**
   the prior denoise instance so tweaking the strength never piles up junk
   instances (see re-run identity below).
5. **Confirm** when happy. Confirming locks the points (confirmed = locked), so no
   later volume apply overwrites them, and the export wizard can filter the Exclude
   class out.

### Endpoint

`POST /api/segment/denoise` `{k?, std_ratio, replace_inst?, protect_instances}`
→ (if `replace_inst` set) erase that instance's points to unlabeled first
→ `statistical_outlier_indices(positions, all_idx, k, std_ratio)`
→ `apply_reassign(outliers, target_inst=-1, target_class=6, protect_instances=…)`
  (materializes a fresh `pointset` instance, class `unknown`, unconfirmed; confirmed
  instances are protected so denoise never eats already-confirmed geometry)
→ returns `{instance_id, n_affected, scan_indices_b64}` for the frontend to add the
  row and highlight.

**Re-run identity — backend-owned replacement.** The frontend's `deleteInstance`
(`mode-label.jsx:729`) only drops the metadata row; it does **not** erase working-array
points. So replacing the denoise instance must happen server-side: `/denoise` accepts
an optional `replace_inst` id and, when present, erases that instance's points to
`(-1,-1)` via `apply_reassign(target_inst=None, target_class=None)` **before**
computing/applying the new outlier set. Without this, points that were outliers at the
old strength but not the new one would stay orphaned as Exclude under a dead instance
id. `mode-label.jsx` tracks the current denoise instance id and passes it as
`replace_inst` on each re-run, swapping its row for the returned one. If the user has
meanwhile confirmed the denoise instance, the frontend drops its tracked id and omits
`replace_inst` — the confirmed Exclude instance is protected and kept, and the next run
creates a new one.

**Undo.** Each re-run is ordinary `apply_reassign` calls (an erase of the prior
instance, then a reassign of the new outliers), so both land on the undo stack like
any other apply. Re-runs are therefore undoable, **not** special-cased to be
undo-transparent (YAGNI — the review loop is short and consistency with every other
apply is worth more than a cleaner undo history).

### UI

- Button + **aggressiveness slider** (`std_ratio`) in the left rail
  (`tool-options.jsx` / the Label-mode left rail, sibling of the Export button).
- Spinner during the synchronous call.
- The caught points highlight via the existing unconfirmed-instance rendering — no
  new viewport code.

## Feature B — per-selection "Remove outliers" (pure shrink)

Clean an *existing* selection by stripping its spatially-isolated strays. B's job is
**"clean this selection," not "classify the junk"** — the stripped points go **back
to unlabeled**, never to Exclude. (Global denoise C is the dedicated path for
"these are noise → Exclude." Keeping the two separate means each action does one
thing and B can't mislabel a merely-bad selection boundary.)

### Scope (decided)

B is wired to the two surfaces whose membership is **directly mutable**:

- **SAM candidate** (`sam_ids` / `sam_segments`) — the SAM edge-stray fix (pain #2).
- **Unconfirmed instance** (`instance_ids`) — confirmed instances are locked and
  excluded, same rule as everywhere.

**Presegs are out of scope.** `preseg_ids` is the immutable precomputed layer;
shrinking a preseg would require re-partitioning it into the mutable cut/SAM layer
(the `cut-shape` mechanic) — real added complexity for a source that is usually
already clean. A noisy preseg is still served two ways: global denoise (C) sweeps
its strays into Exclude, or the existing Cut tool trims it manually.

### Flow

1. Right-click a SAM candidate row (`SamSegmentList`) or an unconfirmed instance row
   (Instances panel) → **"Remove outliers"** in the same context menu that hosts
   *"Edit selection…"* (cut).
2. Backend resolves that source's full-res membership → SOR over *that subset* at the
   aggressiveness slider value → outlier indices.
3. Strip the outliers; the selection shrinks to its inlier core:
   - **SAM candidate:** retire the stray points' `sam_ids` claim (the existing
     `_retire_sam_ids` path); candidate shrinks.
   - **Unconfirmed instance:** reassign the stray points to `-1` (unlabeled) via the
     normal `apply_reassign` path; instance shrinks.

### Endpoint

`POST /api/segment/denoise-selection` `{source: 'sam'|'instance', id, k?, std_ratio}`
→ resolve subset membership for `(source, id)`
→ `statistical_outlier_indices(positions, subset_idx, k, std_ratio)`
→ strip outliers per source (retire `sam_ids` / reassign to `-1`)
→ returns the shrunk membership / new point count.

Kept distinct from `/denoise` because it does a different thing (shrink, not
materialize-to-Exclude).

### Eligibility

A helper mirroring `cut-eligibility.js`:

- SAM candidate row: `selectionSize > 0`.
- Instances panel: a single selected **unconfirmed** instance.

## Parameters (shared C + B)

- **Aggressiveness slider** → `std_ratio`, default **2.0**, range ~**1.0–3.0**
  (lower = greedier). One control, shared by both features.
- **`k`** fixed at **16** (neighbors). Advanced-only if ever exposed.

## Testing

- **Pure-fn unit tests** (`backend/tests/`) on `statistical_outlier_indices`:
  - a tight cluster + planted floating specks → *exactly* the specks flagged;
  - `std_ratio` monotonicity (lower flags a superset of higher);
  - degenerate inputs (empty subset, subset smaller than `k`, all-identical points)
    fail loudly or return empty, never crash.
- **Backend endpoint tests:**
  - `/denoise` — Exclude instance materializes with class `unknown`, unconfirmed;
    re-run replaces (does not accumulate).
  - `/denoise-selection` — SAM source retires `sam_ids`; instance source reassigns
    strays to `-1`; confirmed instances rejected.
- **Frontend:** eligibility-helper unit tests + a jsdom context-menu test, matching
  the cut-selection test pattern (`context-menu.test.jsx`,
  `sam-segment-list.jsdom.test.jsx`).

## Files (anticipated)

**Backend**
- `backend/labeling/outliers.py` (new) — `statistical_outlier_indices`.
- `backend/routes/segment.py` — `/denoise`, `/denoise-selection` handlers.
- `backend/labeling/segment_state.py` — reuse `apply_reassign`,
  `materialize_*`/`_retire_sam_ids` as-is (no new state; `/denoise` is stateless,
  replacement is frontend-owned).
- `backend/app/schemas.py` — request/response schemas.

**Frontend**
- Left-rail "Detect outliers" button + aggressiveness slider (`tool-options.jsx` /
  Label-mode left rail).
- Context-menu "Remove outliers" on `SamSegmentList` + Instances panel.
- `outlier-eligibility.js` (new, mirrors `cut-eligibility.js`).
- `api.js` — `denoise` / `denoiseSelection` clients.
- `mode-label.jsx` — wire the handlers, track the re-run instance id.

**Docs**
- `CLAUDE.md` — document the outlier tool in the same PR.
