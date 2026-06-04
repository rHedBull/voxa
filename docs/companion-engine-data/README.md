# Companion files for engine/data — scan-schema v2

These two files must be applied into the `engine/data` tree **at deploy time**, together with running the migration script on any existing scans.

## Apply procedure

```bash
# 1. Copy updated scaffolder
cp voxa/docs/companion-engine-data/scaffold_annotation.py \
   engine/data/tools/scaffold_annotation.py

# 2. Copy updated lidar SCHEMA
cp voxa/docs/companion-engine-data/SCHEMA.md \
   engine/data/lidar/SCHEMA.md

# 3. Dry-run migration (check what will happen; nothing is written)
cd voxa
python scripts/migrate_scan_v2.py --dry-run \
   /home/hendrik/coding/engine/data/lidar

# 4. Run migration for real
python scripts/migrate_scan_v2.py \
   /home/hendrik/coding/engine/data/lidar
```

Steps 1 and 2 can be applied independently (scaffolder and SCHEMA are documentation/tooling only). Step 4 is irreversible per scan — it moves `labels/`, `session/`, and `annotation_history/` into `sessions/legacy/` in place. Keep a backup or run in a scratch copy first if unsure.

## What changes

**`scaffold_annotation.py`** — updated to emit a v2 skeleton:
- `meta.json` now includes `"schema_version": "2.0"`.
- No `labels/` directory is created (v2 has no GT slot until a session saves).
- An empty `prelabel/` directory is created as an optional convenience so preseg pipelines have their target dir ready; discovery needs only `source/scan.ply` and a v2 `meta.json` and does not require `prelabel/` to exist (the preseg pipeline fills it via `register_preseg()`).
- Everything else (source/scan.ply, README.md, meta.json provenance fields, mesh handling, recentering) is unchanged.

**`SCHEMA.md`** — rewritten to v2. Describes `prelabel/<preseg_id>/`, `sessions/<session_id>/`, removes `labels/` / `session/` / `annotation_history/` from the canonical layout.

## Why NOT applied directly to engine/data

`engine/data` is not a git repo and the deployed voxa is still on v1.3 at the time this branch was cut. Applying these files before deploying voxa v2 would produce scans that the running voxa cannot discover. The companion approach keeps the change staged and reviewable until the voxa deployment is ready.
