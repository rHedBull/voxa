# LiDAR Mesh Generation for Voxa Inspect

**Status:** Design
**Date:** 2026-04-27
**Owner:** hendrik

## Problem

Voxa's Inspect mode supports a GLB mesh overlay (commit `e5b347f`) that toggles between a point-cloud view and a textured surface view of the same scene. Today only **one** scene benefits: `annotated/munich_water_pump`, because NavVis shipped that dataset with an authored `source/mesh.glb` (139 MB). The other five annotated scenes (`construction_site`, `factory_large`, `navvis_mlx`, `navvis_vlx3_water_treatment`, `smart_ais`) were sampled from `.laz` files and have no mesh — Inspect can show them as points only.

We want the same mesh-overlay UX on every annotated scene by **reconstructing a surface from each LAZ**, dropping the result at `annotated/<scene>/source/mesh.glb`. Voxa's discovery and viewer already handle that path; the work is upstream of Voxa.

## Goals

1. Produce `source/mesh.glb` for all five LAZ-sourced annotated scenes.
2. Quality-first reconstruction. The munich aesthetic (colored, "lacy" surface that preserves thin features like pipes and railings) is the bar.
3. Voxa requires zero code changes — meshes appear in the Inspect picker as soon as the files exist.
4. Re-runnable, deterministic, batch-tolerant: one failed scene doesn't abort the others.

## Non-goals

- Replacing or regenerating munich's authored NavVis mesh. The pipeline gates on `meta.json::source_laz`; munich (which has `source_mesh` instead) is skipped.
- Real-time / on-demand meshing inside the Voxa backend. This is offline tooling.
- Mesh editing or annotation. The output is read-only viewing geometry.
- Voxa runtime dependency changes. New deps live in the lidar archive's own tooling.

## High-level architecture

```
engine/data/lidar/                           ← lidar archive (canonical)
├── annotated/<scene>/
│   ├── meta.json                            ← source_laz: "..."
│   ├── source/scan.ply                      ← (already exists)
│   └── source/mesh.glb                      ← NEW, written by the script
├── laz/<file>.laz                           ← input
└── scripts/                                 ← NEW directory
    ├── build_meshes.py                      ← driver
    ├── requirements.txt                     ← script-local deps
    └── README.md
```

The script is a sibling of the existing per-scan tooling (`annotated/munich_water_pump/sim/make_scans.py`) but lives at archive scope because it operates across scans. Voxa's `backend/scene_registry.py::_discover_annotated` already picks up `source/mesh.glb` and exposes `mesh_url`; no Voxa change is required.

## Pipeline (per scene)

```
LAZ (full res, on disk)
  └─► chunked stream-read via laspy[lazrs]
  └─► voxel-downsample to ~5–10M points       (Open3D tensor API; CUDA when available, CPU fallback)
  └─► RGB fallback chain: LAZ RGB → intensity (recorded in sidecar)
  └─► normal estimation, kNN-based             (Open3D tensor API; CUDA when available)
  └─► Ball-Pivoting reconstruction on CPU      (multi-radius: 1.5x, 2.5x, 4.0x mean spacing)
  └─► transfer per-vertex color from nearest input point (KDTree)
  └─► optional quadric-edge-collapse decimation if GLB > 400 MB
  └─► export source/mesh.glb                   (trimesh GLB writer with vertex colors)
  └─► write source/mesh.meta.json sidecar      (radii, voxel size, point count, color source, wall-clock)
```

### Algorithm choice: Ball Pivoting

BPA is CPU-only in Open3D and the bottleneck of the pipeline. We picked it deliberately:

- It preserves edges and thin features (pipes, railings, equipment) instead of inflating concavities the way Poisson does. This matches the existing munich GLB aesthetic.
- It produces an "honest" surface — holes where sampling is sparse, no fake fill — which is what we want for industrial inspection.
- The smoother GPU-native alternative (NKSR) would change the look. Quality-first means matching munich, not just rendering fast.

GPU acceleration is opportunistic and applied only to upstream stages (voxel downsample, normal estimation). BPA itself runs on CPU. Wall-clock per scene is expected to be a few minutes for typical scenes, longer for SMART-AIS (~1.7 GB LAZ).

### Color: RGB-first with intensity fallback

LAZ point formats 2/3/5/7/8/10 carry RGB; others carry only intensity. Per scene, we probe the point format on read:

- **RGB present:** transfer R/G/B (uint16 → float [0,1]) per vertex from the nearest input point via KDTree lookup.
- **RGB absent:** map intensity (uint16) → grayscale via percentile-clipped linear stretch (5th–95th), then per-vertex.

The sidecar records which path was taken so the result is reproducible and inspectable.

### Voxel size and BPA radii

We don't hardcode a voxel size. The script:

1. Estimates mean nearest-neighbor spacing on a random 10K-point sample of the LAZ.
2. Picks `voxel_size = 1.5 * mean_spacing` clamped to `[0.02m, 0.10m]`.
3. After downsampling, recomputes mean spacing on the result.
4. Derives BPA radii as `[1.5, 2.5, 4.0] * downsampled_mean_spacing`.

This avoids per-scene tuning while still adapting to wildly different scan densities. `--voxel` overrides the auto value when a scene needs help.

## CLI

```
python build_meshes.py                          # all eligible scenes
python build_meshes.py factory_large            # one scene by name
python build_meshes.py --force factory_large    # overwrite existing mesh.glb
python build_meshes.py --voxel 0.04             # override voxel size
python build_meshes.py --dry-run                # list what would run
python build_meshes.py --no-gpu                 # force CPU prep
```

Eligibility rule: `meta.json::source_laz` is set AND `source/mesh.glb` does not exist (unless `--force`).

Output goes to a structured log:

```
[1/5] construction_site
  laz: lidar/laz/Construction-site-sample-data.laz (201 MB)
  read: 12.3M points, RGB present
  voxel: 0.025m → 8.4M points
  spacing: 0.022m, radii: [0.033, 0.055, 0.088]
  BPA: 4m22s, mesh: 1.2M verts / 2.3M tris
  color: RGB transferred from 8.4M source points (KDTree)
  decimate: skipped (245 MB < 400 MB)
  wrote: annotated/construction_site/source/mesh.glb (245 MB) in 5m11s
```

Final summary lists ✅ / ⏭ skipped / ❌ failed counts.

## Voxa integration

Zero code changes required. The viewer pipeline is already complete:

- `backend/scene_registry.py::_discover_annotated` picks up `source/mesh.glb`, sets `extras["mesh_path"]`, and `has_mesh = True`.
- `backend/main.py::load_scene` returns `mesh_url = /api/mesh/<scene_id>` when `has_mesh`.
- `backend/main.py::get_mesh` streams the file as `model/gltf-binary`.
- `frontend/src/viewer.jsx` loads via `GLTFLoader` on the "show mesh" toggle, applies the Z-up→Y-up rotation and recenter offset, and replaces the point cloud while the mesh is visible.

The Z-up handling already does the right thing for our generated meshes: `_scene_is_z_up` defaults to Z-up for annotated scenes that don't have `source_mesh` in their meta. Our generated meshes inherit the Z-up frame of the source LAZ, so `mesh_is_z_up = True` is correct. **No `meta.json` mutation is required** — leaving `source_mesh: null` (its current value) is the right state, because the meta key signals "Y-up authored mesh", which is munich's case but not ours.

## Dependencies

Script-local, in `engine/data/lidar/scripts/requirements.txt`:

- `open3d>=0.18` — voxel downsample, normal estimation, BPA, decimation, tensor API
- `laspy[lazrs]>=2.5` — chunked LAZ reader (already in Voxa backend)
- `trimesh>=4` — GLB export with per-vertex colors (already in Voxa backend)
- `numpy`
- `tqdm` — progress bars

Voxa's `backend/requirements.txt` is unchanged.

## Testing

This is offline tooling around mature upstream algorithms; tests are deliberately small.

- **Synthetic unit test** (`scripts/tests/test_pipeline_synthetic.py`): a 10K-point sphere with random RGB. Pipeline produces a non-empty GLB; trimesh round-trips it; vertex count > 1K; per-vertex colors are present and within tolerance of nearest input colors.
- **Integration smoke** on `construction_site` (smallest LAZ): verifies the script writes `annotated/construction_site/source/mesh.glb` with > 1K verts and < 400 MB. Skipped in CI by default; run with `pytest -m integration`.
- **Voxa pickup** (existing `backend/tests/test_load_endpoint.py` covers this once the GLB exists): `LoadResponse.mesh_url` becomes non-null, `mesh_is_z_up` is `True`.

No GPU test — the GPU path is a perf optimization with a CPU fallback.

## Failure modes

| Failure | Behavior |
|---|---|
| LAZ has no RGB | Log "color source: intensity"; mesh still produced. |
| BPA leaves catastrophic holes | Log hole stats; mesh still produced. User inspects in Voxa, can re-run with `--voxel`. |
| OOM on SMART-AIS (1.7 GB LAZ) | Voxel downsample is streaming and BPA gets ≤10M points; should fit in 32 GB. If not, script bails with the recommendation to bump `--voxel`. |
| Existing `mesh.glb` present | Skip unless `--force`. Never silently overwrites. |
| One scene fails in batch | Logged, marked ❌, batch continues with the next scene. |

## Open questions

None blocking. Voxel size and BPA radii defaults will be validated empirically on the first real run; the `--voxel` override exists for the inevitable scene that needs hand-tuning.

## References

- Existing mesh integration: commits `e5b347f`, `4f55ef5`
- Multi-root scene discovery design: `docs/superpowers/specs/2026-04-27-lidar-multi-root-loading-design.md`
- Lidar archive schema: `engine/data/lidar/SCHEMA.md`
- Voxa CLAUDE.md (architecture overview)
