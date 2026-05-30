# scripts/

Offline CLIs and dev tooling. The Python tools add `backend/` to `sys.path`
(`ROOT = Path(__file__).resolve().parents[2]`) so they can import voxa's
`preseg` / `scenes` / `labeling` packages; run them with the project venv:
`./.venv/bin/python scripts/<dir>/<tool>.py …`.

## `preseg/` — offline presegmentation generators
Write `prelabel/ransac_*` into a scan dir; voxa surfaces these as suggestions
when `labels/` is empty (`load_prelabel`). See `docs/presegmentation.md`.

| Tool | What it does |
|---|---|
| `presegment.py` | Canonical RANSAC preseg from a PLY → `prelabel/ransac_*`. |
| `presegment_sam3_features.py` | SAM3 pipeline **Stage 1**: extract + cache per-point image features (needs torch/CUDA — anaconda python, not `.venv`). |
| `presegment_sam3.py` | SAM3 pipeline **Stage 2**: feature-aware RANSAC preseg from the cached features → `prelabel/ransac_*`. |

## `scan/` — scan-schema v1.3 tooling
Operate on a SCHEMA scan directory (see `docs/scan-schema.md`).

| Tool | What it does |
|---|---|
| `validate_scan.py` | Lint a scan dir against the v1.3 invariants (§7). |
| `scan_index.py` | Generate the scan's `variants.json` index (§4.2). |
| `verify_registration.py` | Verify `source/scan.ply` registers to its `renders/<run>` poses. |
| `backfill_scan_frame.py` | Recover a render-frame transform and backfill v1.3 frame metadata. |

## `data/` — data / mesh utilities
| Tool | What it does |
|---|---|
| `derive_cuboids.py` | Derive cuboid GT + predictions from per-point label arrays for one scene. |
| `downsample_to_target.py` | Voxel-downsample a PLY to a target point count. |

## `dry_sam3/` — SAM3 R&D (dry runs)
Experimental SAM3 feature/segmentation explorations that write to scratch
output dirs (not `prelabel/`). Not part of the production pipeline.

## Shell infra (repo root, referenced by `package.json` / docs)
| Script | What it does |
|---|---|
| `run.sh` | Bootstrap `.venv` + launch the backend (`npm run dev:backend`). |
| `test.sh` | Bootstrap dev deps + run the backend test suite (`npm run test:backend`). |
| `import_scene.sh` | Import a PLY/GLB into `data/scenes/<name>/`. |
