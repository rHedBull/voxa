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
the scene (≥1 token), and **at most one** density token immediately after the vendor (nothing
else may follow). Violations (each a warning): no vendor token; a vendor token not in the
allow-list; uppercase / leading / trailing underscore; **any token after the vendor that is not a
single valid density** (e.g. `water_pump_navvis_foo`, or two trailing tokens).

**Enforceable vs guideline:**
- `is_valid_scan_name` / `is_valid_source_id` enforce the *grammar* above (statically, from the
  name alone).
- The rule "density present **only** when >1 density of the same `<scene>_<vendor>` is
  materialized" is a **human guideline**, not machine-checkable from a single name (it needs the
  set of sibling dirs). It is documented, not enforced by the predicate.

**Known limitation:** "exactly one vendor token" means a future *scene* that legitimately contains
a vendor word (e.g. a scene literally named `navvis_lab`) would be flagged. Not triggered by any
current name; documented so a maintainer recognizes the false positive if it ever appears.

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
- **scan_name:** in `validate_archive`'s per-scan pass, when `meta.json::scan_name` fails
  `is_valid_scan_name`, append `warnings: "scan_name '<x>' does not match
  <scene>_<vendor>[_<density>]"` to that scan's result.
- **source_id:** `validate_archive` returns a dict keyed by **scan_name**, and lineage is only
  visited per-scan via `_check_lineage`, so a root referenced by no scan would never be checked.
  Add a **dedicated pass over `Registry.roots`**: for each root key failing `is_valid_source_id`,
  attach a warning under a synthetic result key `"raw/sources.json"` (a reserved, non-scan key in
  the returned dict). This guarantees every root is checked regardless of references.
- **Never an error.** A misnamed scan/root stays discoverable; voxa's discovery gate (errors-only)
  is unaffected.

## 6. Migration (`scripts/scan/rename_scans.py`)

Dry-run by default; `--apply` to commit. Reads a hard-coded RENAME_MAP (the tables in §4) so the
rename is reviewable and deterministic, not inferred.

### Blast radius — exact references to rewrite

This list was verified field-by-field against real scans (`navvis_vlx3_water_treatment`, munich)
during spec review. A `scan_name` is embedded in **id fields** (rewrite) and in **free-text
provenance strings** (`...source`, `notes`, README — rewrite the structured provenance strings,
leave prose). The two kinds are called out below.

**scan_name refs** (per scan):
- directory: `annotated/<old>/` → `annotated/<new>/`
- `meta.json::scan_name`
- `meta.json::frame.canonical_id` (`<scan_name>#local`) — v3.0 scans only
- `meta.json::derivation.scan_id` — v3.0 scans only
- `variants.json::scan_id`, `::canonical_id`, each `variants[].path` (absolute path to the scan
  dir), and each `variants[].source` provenance string (e.g.
  `"potree:<scan_name> (scan_15M.las)"`)
- `renders/<run>/meta.json`: `generated_from.scan_id`, `generated_from.canonical_id`
  (`<scan_name>#local`), `generated_from.source` provenance string (~9 render dirs across v3.0
  scans + munich)
- `renders/<run>/manifest.json::scene`
- `sessions/<id>/instances_gt.json::scene` — value is the **tier-prefixed** id
  `"annotated/<scan_name>"` → rewrite to `"annotated/<new>"` (1 `legacy` + per-session dirs)
- `source/mesh.meta.json::scene` (provenance label) — where present
- `README.md` — **report only**, do not blind-replace free prose (manual edit)

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
- **Stray non-`.json` files** that embed a name (e.g.
  `sessions/.../instances_gt.json.bak-inst178`, `.bak*`): the migration **does not rewrite**
  them and the residual scan **skips** them (whitelisted by suffix). They are pre-existing strays
  slated for removal by the separate stray-cleanup workstream; this is logged, not silently
  ignored.

**Note on schema impact:** none of the render/instance-doc fields above are validated against
scan_name by `validate.py`, so omitting them would **not** surface as a schema error — only the
residual scan (step 5) catches them. That is exactly why the blast radius must be exhaustive
rather than relying on the 0-errors gate.

### Algorithm

1. Validate preconditions: `python -m scan_schema <root>` reports **0 errors**; every old name in
   RENAME_MAP exists; no new name collides with an existing dir/key.
2. **Backup**: `data/lidar` is **not** a git repo, and step 4 does whole-directory renames, so a
   partial-field tarball is not a sufficient undo. Tar each affected `annotated/<old>/` tree **in
   full** plus `raw/sources.json` to a timestamped tarball before any write. Print its path.
3. Rewrite JSON id-fields per the blast-radius list (load → set → atomic write via
   `scan_schema.storage`).
4. Dir renames (plain `os.rename`, same filesystem).
5. **Residual scan**: walk every `*.json` under each renamed scan (skipping whitelisted `.bak*`
   strays) and check each **field value** for an exact / token-boundary match against any old
   name — **not** a naive substring/`grep`. Substring matching would self-fail, since the old
   `construction_site` is a substring of its new `construction_site_navvis` (and similar for
   munich). Fail the run on any residual id-field match.
6. **Postcondition gate**: re-run `python -m scan_schema <root>` → must still be **0 errors**, the
   naming warnings must be **gone**, and the residual scan must be clean.

### Safety / reversibility

- Dry-run prints the full diff (every file + field + old→new) and the residual-scan report.
- The full per-scan-dir tarball is the undo for the archive.
- The `test_real_scans_validate.py` sweep is the automated gate (run before + after).

## 7. Interaction with munich lineage (cross-link, not in scope here)

munich's two scans are v2.0 (no `frame`/`derivation`), so their scan_name rename touches a
**smaller** field set than v3.0 scans — but **not** just meta+README+dir: at least
`munich_water_pump/renders/interior_grid__o3d__20260607/manifest.json` embeds the old name, so the
renders/instance-doc rewrites in §6 apply to munich too. Their mesh root is **not yet registered**,
so it is absent from the source_id table. When the separate lineage workstream registers munich's
`mesh.glb` as a root, that root **must** be named `water_pump_navvis` per this convention — which
already satisfies `is_valid_source_id`, so no enforcement change is needed (there is no separate
"allow-set"; enforcement is `KNOWN_VENDORS` + the name predicates). Either workstream may run
first.

## 8. Testing

- **Unit** (`scan_schema/tests/test_naming.py`): `is_valid_scan_name` / `is_valid_source_id` over
  conforming names and each violation shape (no vendor, unknown vendor, bad density, uppercase,
  trailing/leading underscore).
- **Migration** (`backend/tests/test_rename_scans.py` or a `scan_schema` test): build a tmp
  fixture archive exercising the **full** blast radius — 2 scans + 1 root, with v3.0 meta +
  variants (incl. a `variants[].source` provenance string), a `renders/<run>/{meta.json,
  manifest.json}`, a `sessions/<id>/instances_gt.json` (tier-prefixed `scene`), and a stray
  `.bak-*` file. Run `--apply` and assert: dirs renamed; **every** id-field in §6 rewritten
  (meta, variants, renders, instances_gt, mesh.meta); `variant_id` and the `.bak-*` stray
  untouched; the substring-trap name pair (`construction_site` → `construction_site_navvis`)
  doesn't self-fail the residual scan; `validate_archive` error-free; naming warnings gone;
  residual-scan clean.

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
