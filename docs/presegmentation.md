# Presegmentation pipeline (SAM3-aided)

Presegmentation seeds a scan with auto-suggested instances so labeling starts from
structure instead of a blank canvas. Voxa surfaces a prelabel from
`<scan>/prelabel/ransac_*` whenever `labels/` is empty (see
[scan-schema](scan-schema.md)). This doc covers the offline pipeline that produces
that prelabel using RANSAC geometry + SAM3 image features.

## Why it is two stages, two environments

The work needs two runtimes that don't coexist in one interpreter here:

- **SAM3 feature extraction** needs `torch` + CUDA + the `sam3` package — present
  only in the **anaconda base** python (`/home/hendrik/anaconda3/bin/python`), not
  in voxa's `.venv`.
- **RANSAC presegmentation** uses **Open3D**, whose anaconda build **SIGSEGVs** in
  plane extraction. Voxa's `.venv` Open3D (same version) is stable.

So the pipeline is split: extract features under anaconda, presegment under `.venv`.

## Running it

```bash
SCAN=/home/hendrik/coding/engine/data/lidar/annotated/<scan>

# Stage 1 — SAM3 per-point features (anaconda python; needs renders/ + GPU).
# Caches to <scan>/sam3/<scan>/sam3_features.npz
/home/hendrik/anaconda3/bin/python scripts/presegment_sam3_features.py "$SCAN"

# Stage 2 — RANSAC + feature-aware split → <scan>/prelabel/ransac_* (.venv python)
.venv/bin/python scripts/presegment_sam3.py "$SCAN"
```

Prerequisites on disk (per [scan-schema](scan-schema.md)):
`source/scan.ply`, and at least one `renders/<run>/manifest.json` for stage 1.

Verify it loaded:

```bash
curl -s -X POST http://127.0.0.1:8765/api/load \
  -H 'Content-Type: application/json' \
  -d '{"name":"annotated/<scan>","want_full_labels":true}' | python -m json.tool | grep is_from_prelabel
# → "is_from_prelabel": true
```

## What each stage does

**Stage 1 — `scripts/presegment_sam3_features.py` → `backend/preseg/sam3_features.py`**

Multi-view feature fusion. For each render frame: build a look-at camera from the
manifest pose (Three.js perspective, vertical FOV 60°), project every point, reject
occluded points with a splatted z-buffer, run the SAM3 image encoder, and
bilinear-sample the FPN feature map at each visible point. Features are summed
across frames, then mean-pooled per point, L2-normalized, and PCA-reduced (→ 64).
Output is cached to `<scan>/sam3/<scan>/sam3_features.npz` (`features` (N,64) f16 +
`seen` (N,) int32). Only points visible in ≥1 frame get a real feature — **render
coverage gates the SAM3 contribution** (a partial walkthrough may cover only ~15%
of the cloud).

**Stage 2 — `scripts/presegment_sam3.py` → `backend/preseg/presegment.py` (mode="ransac")**

Loads the cached features (numpy only, no torch), runs RANSAC presegmentation
(planes → cylinders → spheres → leftover clustering), then splits large geometric
instances by k-means in SAM3 feature space so a single plane can break into
semantically distinct sub-regions. Writes `prelabel/ransac_instance_ids.npy` (int32,
shape `(N,)`) + `prelabel/ransac_segment_summary.json`.

## Resolution: full cloud by default

Stage 2 presegments the **full cloud** by default. Voxa's `.venv` Open3D build
handles 3M points fine — measured ~6 min and ~20 GB peak RSS on a 3M cloud,
yielding ~6000 segments (full-resolution boundaries).

The SIGSEGV seen earlier was specific to the **anaconda** Open3D build (used while
prototyping feature extraction), not a scale limit: `.venv` (same Open3D version,
different build) runs full 3M without crashing, and stage 2 runs under `.venv`
anyway. So there is no inherent ~500k ceiling.

### Safety guards

Both stages read the PLY vertex count from the header first (no multi-GB load):

- **> 5M (`VOXA_MAX_LABEL_POINTS`) → refused.** A prelabel for a cloud voxa can't
  load for labeling is useless, and the load/extraction would OOM. Downsample
  `source/scan.ply` first. (This is the `smart_ais_clean` clobbered-156M case.)
- **Within the labelable range, full-res is bounded by RAM.** Stage 2 estimates a
  safe full-res ceiling (~85% of total RAM at ~6.5 GB per million points) and, if
  the cloud exceeds it, auto-subsamples to the ceiling + NN-propagates with a
  warning. On a 32 GB box the ceiling is ~4.3M, so a 3M cloud runs full-res.

To force a specific size, pass `--preseg-points N`. Stage 2 then:

1. presegments a random N-point subsample (with the matching subset of features), then
2. propagates instance ids to the full cloud via nearest-neighbour (`scipy.spatial.cKDTree`).

The prelabel is only a **seed** (refined at full resolution in Label mode), so the
coarser propagated boundaries are acceptable when you take the cheaper path. See
[point-cloud-sizing](point-cloud-sizing.md) for how this fits the other caps.

## In-app alternative (and why we don't use it here)

`POST /api/segment/presegment` (`mode="ransac"` + `sam3` params) runs the same
logic inside the backend and caches features the same way. It is not usable from
the dev server because that backend runs in `.venv` (no torch). It would work only
if the backend itself ran under a torch-enabled interpreter.

## Related

- [Scan directory schema](scan-schema.md) — `prelabel/`, `sam3/`, `renders/` contracts.
- [Point-cloud sizing](point-cloud-sizing.md) — the full cap model.
- [Metrics for aided labeling](metrics-for-aided-labeling.md) — how prelabel quality is scored.
