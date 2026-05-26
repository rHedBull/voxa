# Point-cloud sizing and caps

A cloud passes through several independent size caps on its way from disk to the
viewer. Each guards a different bottleneck, so they are tuned separately. All are
env-overridable; defaults live in `backend/app/constants.py`.

## The caps

| Stage | Cap (default) | Env var | Where | Bottleneck it protects |
|-------|---------------|---------|-------|------------------------|
| Load (source sampling) | stride/voxel to ≥ `max(max_points, 50_000)` | — | `backend/scenes/lidar_io.py` (`load_laz`), `point_cloud.py` (`load_glb`) | RAM while reading LAZ/GLB |
| **Label** | 5,000,000 | `VOXA_MAX_LABEL_POINTS` | `backend/routes/load.py`, `routes/preseg.py` | per-point label arrays + session memory |
| **Viewer** | 3,000,000 | `VOXA_MAX_POINTS` | `backend/app/core.py` (`_safe_subsample`) | Three.js / GPU render + wire payload |
| Preseg | 500,000 | `--preseg-points` | `scripts/presegment_sam3.py` | Open3D `segment_plane` stability ([presegmentation](presegmentation.md)) |
| Recommendation | 200,000 | `subsample_n` (request field) | `backend/routes/preseg.py` (`/optimize`) | parameter-search speed |

The chain in size order: **load (RAM) → label (5M) → viewer (3M) → preseg (500k) → recommend (200k)**.

## Label resolution vs. viewer resolution

Labels always operate on the **full** loaded cloud; the viewer may see a subset.

- At load, `_safe_subsample` (`app/core.py`) draws a **uniform random** subset of up
  to `VOXA_MAX_POINTS` points for display. The draw is seeded (`default_rng(7)`),
  so the displayed subset is stable across reloads.
- The label session (`_state["seg"]`) is built on the full `pc.points`, but only
  if `len(pc) <= VOXA_MAX_LABEL_POINTS`. Above the label cap there is **no session**
  — the scene loads for viewing but cannot be labeled.
- `/api/load` therefore returns two sets of arrays:
  - `positions` / `class_ids` / `instance_ids` — the subsampled view (≤ viewer cap)
  - `full_*` + `subsample_idx` (opt-in via `want_full_labels`) — full-resolution
    arrays plus an index mapping each view row back to its full-cloud index.

Because the viewer cap (3M) now matches typical scan sizes, `_safe_subsample`
**no-ops for clouds ≤ 3M**: `subsample_idx` is `null` and the view arrays already
are the full arrays. Subsampling only kicks in for clouds between the viewer cap
and the label cap.

### Wire-payload note

A 3M-point load is ~72 MB (positions + colors, base64). Fine on localhost; heavy
over a network. Lower `VOXA_MAX_POINTS` if serving remotely.

## Density vs. point count

The caps are **point-count** based, not spacing based. Clouds arrive pre-decimated
at different densities (e.g. `water_treatment` is a 1.7 cm voxel downsample → 3M
points), so "1M points" means a different physical resolution per scene. Subsampling
is uniform-random, which preserves relative density statistically but does **not**
normalize spacing — there is no voxel/Poisson-disk downsample in the load or viewer
path today.

## Other size-related cleanups (at load)

- `_recenter` (`app/core.py`) subtracts the bbox centroid when any coordinate
  exceeds 1e3, keeping float32 precise for UTM-scale LAS scenes. The offset is
  returned in `recenter_offset`.
- `_filter_tiny_segments` drops any prelabel instance smaller than
  `VOXA_MIN_SEGMENT_POINTS` (default 10) to `-1`, defending against degenerate
  prelabels (e.g. one-point-per-instance) that would flood the UI.

## Related

- [Presegmentation pipeline](presegmentation.md) — why preseg is capped at 500k.
- [Scan directory schema](scan-schema.md) — on-disk layout and the `labels/` /
  `prelabel/` contracts.
