# Scan & Source Naming Convention — Design

- **Date:** 2026-06-26
- **Status:** Draft (awaiting review)
- **Author:** Hendrik + Claude
- **Related:** `2026-06-09-scan-schema-enforcement-design.md`, `scan_schema` package (`tools/scan-schema`)

## 1. Problem

Names in the lidar archive drifted across every identifier namespace. Today:

- **scan dirs** (`annotated/<name>/`, surfaced as `meta.json::scan_name`): `factory_large`,
  `munich_water_pump_3m`, `navvis_vlx3_water_treatment`, `smart_ais_clean` — size/quality
  descriptors (`large`, `clean`), vendor-or-scene-first inconsistently, scanner abbreviations.
- **source_ids** (`raw/sources.json` keys): worst drift — from `factory_large` to
  `navvis_vlx_3_data_water_treatment_facility` and `sample_data_vlx3_processindustry_smart_ais`
  (slugified LAZ filenames).

The archive validates with **0 errors** today (see the enforcement-design doc) — this is a
*stylistic* problem, not a structural one. But nothing prevents the drift from continuing, and
the names are not predictable from each other.

## 2. Goal

A single, predictable naming convention for **scan_name** and **source_id**, enforced by a
**warning-level** check in `scan_schema`, plus a **one-time migration** that renames the existing
9 scans + 7 roots and rewrites every internal reference. Discovery must never be blocked by a
naming violation (warn, never error), consistent with the schema's existing philosophy.

### Non-goals (explicitly deferred)

- `variant_id` (`voxel3M_unaligned`, `aligned9M`, …), `session_id`, `preseg_id` naming — left
  as-is this increment. The migration must **not** touch `variant_id` values.
- munich lineage promotion (registering its mesh as a root, writing v3.0 derivation) is a
  **separate** workstream. It interacts only at the naming boundary (see §7).

## 3. The convention

```
source_id  = <scene>_<vendor>
scan_name  = <scene>_<vendor>[_<density>]

  scene    = one or more lowercase tokens   [a-z0-9]+(_[a-z0-9]+)*     e.g. water_treatment
  vendor   = a known capture vendor          (allow-list: navvis, matterport, …)
  density  = approximate point count         \d+[km] | full            e.g. 500k, 3m, 9m
             present ONLY when >1 density of the same <scene>_<vendor> is materialized on disk
```

- **Scene-first for both** namespaces, so a single-density scan's `scan_name` equals its
  `source_id` base (e.g. root `water_treatment_navvis` ←→ scan `water_treatment_navvis`). They
  live in different maps (registry roots vs scan catalog); the equality makes lineage
  self-evident and is exactly the existing self-root `parent.ref` link.
- **Vendor-only** (no scanner model — `vlx3`/`mlx` dropped).
- The **vendor allow-list** is the only configuration. It lives as one extensible constant in
  `scan_schema` (shared cross-tool), e.g. `KNOWN_VENDORS = {"navvis", "matterport"}`.

### Regex (reference)

```
VENDOR     = (?:navvis|matterport)                  # from KNOWN_VENDORS, joined
DENSITY    = (?:\d+[km]|full)
SCENE      = [a-z0-9]+(?:_[a-z0-9]+)*
source_id  ^  SCENE _ VENDOR  $
scan_name  ^  SCENE _ VENDOR (?: _ DENSITY )?  $
```

The check splits on `_`, requires exactly one token ∈ `KNOWN_VENDORS`, everything before it is
the scene (≥1 token), an optional trailing density token after the vendor. A name with no
vendor token, or a vendor token not in the allow-list, is a violation (warning).

## 4. Rename map (migration target)

### Scans (9) — `meta.json::scan_name` + dir

| Old | New |
|---|---|
| matterport_mechanical_room | `mechanical_room_matterport` |
| matterport_parkhouse | `parkhouse_matterport` |
| navvis_vlx3_water_treatment | `water_treatment_navvis` |
| smart_ais_clean | `smart_ais_navvis` |
| navvis_mlx | `generator_concrete_navvis` |
| construction_site | `construction_site_navvis` |
| factory_large | `factory_navvis` |
| munich_water_pump (500k, labeled) | `water_pump_navvis_500k` |
| munich_water_pump_3m | `water_pump_navvis_3m` |

### Roots (7) — `raw/sources.json` keys

| Old source_id | New |
|---|---|
| construction_site_sample_data | `construction_site_navvis` |
| factory_large | `factory_navvis` |
| navvis_mlx_sample_data | `generator_concrete_navvis` |
| navvis_vlx_3_data_water_treatment_facility | `water_treatment_navvis` |
| sample_data_vlx3_processindustry_smart_ais | `smart_ais_navvis` |
| matterport_mechanical_room | `mechanical_room_matterport` |
| matterport_parkhouse | `parkhouse_matterport` |

(The two matterport roots only change because their scene/vendor order flips; mechanical/parkhouse
were already vendor-first.)

## 5. Enforcement (warn-level check in scan_schema)

- Add `KNOWN_VENDORS` + two predicates (`is_valid_scan_name`, `is_valid_source_id`) to
  `scan_schema` (new small module `naming.py`, exported from `__init__`).
- `validate_archive` / `check_meta`: when `meta.json::scan_name` fails `is_valid_scan_name`,
  append `warnings: "scan_name '<x>' does not match <scene>_<vendor>[_<density>]"`. When a
  `sources.json` key fails `is_valid_source_id`, append a warning in the registry/lineage pass.
- **Never an error.** A misnamed scan stays discoverable; voxa's discovery gate (errors-only)
  is unaffected.

## 6. Migration (`scripts/scan/rename_scans.py`)

Dry-run by default; `--apply` to commit. Reads a hard-coded RENAME_MAP (the tables in §4) so the
rename is reviewable and deterministic, not inferred.

### Blast radius — exact references to rewrite

**scan_name refs** (per scan):
- directory: `annotated/<old>/` → `annotated/<new>/`
- `meta.json::scan_name`
- `meta.json::frame.canonical_id` (`<scan_name>#local`) — v3.0 scans only
- `meta.json::derivation.scan_id` — v3.0 scans only
- `variants.json::scan_id`, `::canonical_id`, and each `variants[].path` (absolute path to the
  scan dir)
- `source/mesh.meta.json::scene` (provenance label) — where present
- `README.md` — **report only**, do not blind-replace free text (manual edit)

**source_id refs** (archive-wide):
- `raw/sources.json` key rename (7)
- every scan's `meta.json::derivation.root.source_id` + `derivation.parent.ref` (when ref is a
  source_id)
- `variants.json::variants[].root_source_id`

**Untouched (verified):**
- `derivation.variant_id` / `variants.json::labeling_variant` / `variants[].variant_id` —
  deferred namespace, left as-is.
- Session pins: `session.json::source_fingerprint` / `preseg_fingerprint` are **content**
  fingerprints (`cloud_fingerprint` / `array_fingerprint`), independent of names → no 409s after
  rename. Confirmed.

### Algorithm

1. Validate preconditions: `python -m scan_schema <root>` reports **0 errors**; every old name in
   RENAME_MAP exists; no new name collides with an existing dir/key.
2. **Backup**: `data/lidar` is **not** a git repo — tar the affected `annotated/<scan>/{meta.json,
   variants.json,source/mesh.meta.json}` + `raw/sources.json` to a timestamped tarball before any
   write. Print its path.
3. Rewrite JSON id-fields per the blast-radius list (load → set → atomic write via
   `scan_schema.storage`).
4. `git mv`-equivalent dir renames (plain `os.rename`, same filesystem).
5. Scan all `*.json` under each renamed scan for **residual** occurrences of the old
   scan_name/source_id strings and **report** them (catches refs this spec missed) — fail the run
   if any residual is found in a non-free-text field.
6. **Postcondition gate**: re-run `python -m scan_schema <root>` → must still be **0 errors**, and
   the naming warnings must be **gone**.

### Safety / reversibility

- Dry-run prints the full diff (every file + field + old→new) and the residual-scan report.
- The tarball is the undo for the archive.
- The `test_real_scans_validate.py` sweep is the automated gate (run before + after).

## 7. Interaction with munich lineage (cross-link, not in scope here)

munich's two scans are v2.0 (no `frame`/`derivation`), so their rename only touches
`meta.json::scan_name` + README + dir. Their mesh root is **not yet registered**, so it is absent
from the source_id table. When the separate lineage workstream registers munich's `mesh.glb` as a
root, that root **must** be named `water_pump_navvis` per this convention. Order: either workstream
first; if lineage runs first, add `water_pump_navvis` to the source_id allow-set so it validates.

## 8. Testing

- **Unit** (`scan_schema/tests/test_naming.py`): `is_valid_scan_name` / `is_valid_source_id` over
  conforming names and each violation shape (no vendor, unknown vendor, bad density, uppercase,
  trailing/leading underscore).
- **Migration** (`backend/tests/test_rename_scans.py` or a `scan_schema` test): build a tmp
  fixture archive (2 scans, 1 root, full v3.0 meta + variants), run the migration `--apply`,
  assert: dirs renamed, all id-fields rewritten, `variant_id` untouched, `validate_archive`
  error-free, naming warnings gone, residual-scan clean.

## 9. Rollout order

1. Land the `scan_schema` naming module + warn-level check + unit tests (no data change).
2. Land `rename_scans.py` + its test.
3. Dry-run on the real archive, review the diff + residual report.
4. Backup tarball → `--apply` → verify gates (0 errors, warnings gone, pytest sweep green).
5. Update docs: `lidar/SCHEMA.md` + `voxa/docs/scan-schema.md` with the convention; note the
   README "required vs recommended" discrepancy fix in passing.

## 10. Open questions

- **Density token granularity:** `500k` / `3m` (1 sig fig) — confirm rounding rule for, e.g.,
  4.99M (→ `5m`? current smart_ais is single-density so moot until a second density appears).
- **Vendor allow-list governance:** who adds a new vendor — just edit the constant + ship? (Yes,
  for now.)
