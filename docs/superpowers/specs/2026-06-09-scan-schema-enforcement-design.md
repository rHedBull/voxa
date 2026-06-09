# scan-schema: v3.0 unified schema + single-source-of-truth enforcement

**Date:** 2026-06-09
**Status:** Design — approved for spec review
**Related:** `lidar/SCHEMA.md` (v2.0), `voxa/docs/superpowers/specs/2026-06-03-multi-session-preseg-design.md`,
`2026-05-29-scan-schema-v1.3-design.md`, `2026-05-29-frame-registration-load-check-design.md`,
`2026-04-27-lidar-multi-root-loading-design.md`

## Problem

`lidar/annotated/` is the shared datastore of labeled lidar scans. Its schema is re-encoded,
independently, in several places that have drifted apart — and the drift is now two-dimensional:

**1. The schema definition is duplicated across three validators + one doc, and they disagree.**

| Consumer | Role | Schema knowledge | State |
|---|---|---|---|
| `lidar/SCHEMA.md` | human doc | prose spec | layout v2.0, meta.json example **stale** (see below) |
| voxa `scenes/scan_layout.py` + `labeling/segment_io._validate_invariants` | the **only writer** (validates GT at save) | executable | working |
| voxa `scripts/scan/validate_scan.py` | standalone metadata lint | re-encoded by hand | checks "v1.3" frame/derivation |
| `data/tools/validate_annotated.py` | standalone layout audit | re-encoded by hand | **v1.3 — rotted**, rejects all 9 scans |

Because every consumer re-derives the schema, they fall out of sync. The rotted
`validate_annotated.py` is the proof: it still requires `labels/gt_class_ids.npy`, a path
v2.0 removed, so it fails **every one of the 9 current scans**. A new hand-coded validator
would just become the fifth thing to rot.

**2. There are two conflated version numbers, and the doc never reconciled them.**

The directory *layout* is versioned "v2.0" (multi-session `sessions/`, `prelabel/`). The
`meta.json` *metadata model* (the `frame` + `derivation` blocks) was versioned "v1.3" with
its **own** validator and specs. `SCHEMA.md`'s v2.0 `meta.json` example never absorbed the
v1.3 metadata model — its example regressed below its own v1.3 changelog entry. So the
documented schema is internally inconsistent, and **the on-disk reality has fields the doc
doesn't mention at all**:

| `meta.json` field / file | On disk | In `SCHEMA.md`? | What it is |
|---|---|---|---|
| `frame` | 2/9 | ❌ | canonical-frame + `transform_to_canonical`; load-bearing for cross-variant ops |
| `derivation` | 2/9 | ❌ | variant identity (`variant_id`, `varies`, `role`, `source_fingerprint`) |
| `coords` / `coord_offset_m` | 4/9 | partial | legacy v1.2 recentering, superseded by `frame.georef` |
| `variants.json` (file) | 2/9 | ❌ | per-scan variant index; `scan_index.py` generates it |
| `source_laz` | 9/9 | ✅ | raw full-density link (load-region + edit-raw export) |
| `source_full_res`, `bbox_*`, `parent_scan` | 1–2/9 | ❌ | advisory caches / fallback |
| extra `source/*.ply`, `mesh.optimized.glb` | several | ❌ | archival / ignored artifacts |

Enforcing the *documented* v2.0 would reject reality. The fix has to (a) define one schema
that matches what the code actually requires, and (b) collapse the three validators into one
executable definition every consumer imports.

## Goal

Define **v3.0** — one coherent schema number that unifies the v2.0 layout and the v1.3
metadata model and matches what the code requires — and ship it as a standalone, versioned
`scan-schema` package that becomes the single executable definition. A schema change then
becomes one deliberate version bump instead of silent per-consumer divergence.

v3.0 also **defines** a **source families** contract: raw scans function as family roots, and
every derived cloud (subsampled / cleaned / cutout / modified) links to both its immediate
`parent` and its `root` raw source, so any cloud is traceable up its family tree to the raw
scan it came from. v3.0 documents the `derivation.root`/`parent` link shape; building the root
**registry** and the cross-scan lineage **validation** is increment 2 (see Roadmap) — it is an
independent subsystem and is deliberately not bundled with the validator-unification work.

**Non-goals (deferred):** the HTTP service, the S3 backend, auto-gating (pre-commit/CI),
migrating meshbuilder/training readers, and **forced migration of existing scans** (see
"Legacy handling" — existing 2.0 scans are grandfathered, not rewritten).

## v3.0 schema

v3.0 is **v2.0's directory layout** (`sessions/<id>/`, `prelabel/<id>/`, `renders/`, `sam3/`
— unchanged) **plus the v1.3 metadata model promoted into the documented contract**, plus a
`scratch/` allow-list, plus a documented `source/` cloud contract.

### `meta.json` field contract

| Field | v3.0 | Note |
|---|---|---|
| `schema_version` | **required** | `"3.0"` |
| `scan_name`, `n_points`, `units`, `class_map_version` | **required** | core identity |
| `frame` | **required** | `{canonical_id, transform_to_canonical (4×4 rigid), units, frame_uncertain, georef?}` |
| `derivation` | **required** | `{scan_id, variant_id, op, varies, role, source_fingerprint, params?, parent, root}`. The `frame`+identity fields are enforced in increment 1; the `parent`/`root` lineage links are *defined* now but enforced/registered in increment 2 (see "Source families") |
| `capture_date`, `scanner`, `notes`, `source_mesh`, `sample_method`, `sample_param` | optional | provenance, preserved as-is |
| `source_full_res` | optional | fallback raw source (PLY) for scans without a LAZ (matterport) |
| `bbox_*` | optional | advisory caches; absence is never an error |
| `source_laz`, `parent_scan` | **superseded in 3.0** | folded into `derivation.root` / `derivation.parent` (format-agnostic) |
| `coords`, `coord_offset_m` | **removed in 3.0** | folded into `frame.georef.offset_m` |

On a **legacy 2.x** scan, the superseded/removed fields (`source_laz`, `parent_scan`, `coords`,
`coord_offset_m`) are **accepted silently** — they are expected on legacy data and warning on
every one would be noise. Warnings are reserved for *drift*: missing `frame`/`derivation`,
unknown top-level entries, archival clouds in `source/`. (See "Legacy handling".)

`frame` semantics: a scan may exist as multiple point-cloud *variants* (the labeled cloud, a
denser aligned cloud for renders, the raw capture). `frame.transform_to_canonical` is the
rigid map from this cloud into the scan's shared canonical frame (identity ⟺ this cloud is
canonical); `frame_uncertain: true` forces the registration health-check
(`verify_registration.py`) before cross-variant artifacts (render poses, propagated labels)
are trusted. `derivation.role: "labeling"` marks the variant where GT physically lives
(≤1 per scan); `derivation.varies` (subset of `{density, frame, points, color, attributes}`)
drives whether an artifact pinned to another variant can be remapped or must hard-fail.

### `source/` cloud contract

- `source/scan.ply` — **required, fixed name.** THE cloud the viewer/labeler loads
  (`scene_registry` resolves `ScanLayout.scan_ply`; never a different variant).
- `source/mesh.glb` — optional, fixed name; canonical geometry if sampled from a mesh.
- `source/mesh.meta.json` — optional provenance.
- Archival clouds (`scan.cleaned_*.ply`, `scan.raw_full_*.ply`, `scan.from_voxa_*.ply`) and
  `mesh.optimized.glb` are **not** schema members → belong under `scratch/`. The validator
  **warns** (does not error) if they linger in `source/`.

### `variants.json` (conditional)

Per-scan variant index (`scan_id`, `canonical_id`, `labeling_variant`, `variants[]`). Required
**iff** any artifact (render run, preseg, label) pins a `variant_id` other than the one it sits
beside (a cross-variant pin); absent otherwise. Generated by `scripts/scan/scan_index.py`.
The validator requires it only when cross-variant pins are present.

### Source families (lineage) — contract defined in v3.0, built in increment 2

**Raw scans function as family roots**, and every derived cloud links back to one. Roots come
in several formats and locations, so the registry is **format- and location-agnostic** — it is
not "the files in `lidar/laz/`". Among the 9 current scans: 5 root in shared LAZ scans under
`lidar/laz/`; 2 (matterport ×2) root in a full-res **PLY** (`source_full_res`); 2 (munich ×2)
root in a **mesh** (`source/mesh.glb`). All of these are roots and all get registered.

This subsection defines the on-disk **contract** (the `derivation.root`/`parent` shape and the
registry format). Implementing the registry and the cross-scan lineage validation is
**increment 2** — `meta.json::derivation.root`/`parent` are documented in v3.0 SCHEMA.md now,
but the increment-1 validator does not yet build the registry or resolve cross-scan links.

**Raw source registry — `lidar/raw/sources.json`** (archive-level, alongside `classes.json`):

```json
{
  "sources": [
    {
      "source_id": "smart_ais",
      "path": "raw/Sample-Data-VLX3-ProcessIndustry-SMART-AIS.laz",
      "format": "laz",
      "fingerprint": "sha256:…",
      "n_points": 156519044,
      "capture_date": null, "scanner": "NavVis VLX-3", "url": null
    }
  ]
}
```

`lidar/raw/` is the canonical home for *shared* global sources (reorganized from `lidar/laz/`),
but a registry entry's `path` may also point at a root that lives inside a scan's own `source/`
(the matterport full-res PLY, the munich mesh) — `format` is one of `laz`/`ply`/`glb`.
`source_id` is a stable short slug; `fingerprint` is computed once at registration.

**Lineage on every derived cloud — `meta.json::derivation`:**

- `derivation.root`: `{source_id, fingerprint}` — the global full-res ancestor; `source_id`
  must resolve in `raw/sources.json` and `fingerprint` must match it. Always present on a
  derived cloud. Format-agnostic (LAZ or PLY), so it **supersedes the LAZ-specific `source_laz`**.
- `derivation.parent`: `{ref, fingerprint}` — the immediate parent cloud. `ref` is another
  scan's `scan_id`/`variant_id`, or the `root.source_id` when derived directly from the raw
  source; `fingerprint` is the parent cloud's content hash. Walking `parent` reconstructs the
  full chain; `root` is the always-present anchor so traversal never depends on every
  intermediate existing. Resolving a `parent.ref` that names another scan requires an
  archive-wide `scan_id`→scan index (a cross-scan lookup) — this is part of the increment-2
  validator, not the per-scan increment-1 checks.

This makes families traceable both ways: down from a root (enumerate clouds whose
`root.source_id` matches) and up from any leaf (follow `parent` to the root). Fingerprints —
not paths — are the identity, so links survive moves/renames; a stale fingerprint is a
detectable break, not a silent mis-link.

**Legacy & grandfathering (increment 2 behavior).** The 9 existing scans are 2.0: 5 carry a
non-null `source_laz` (→ a LAZ root), the matterport/munich four root in a local PLY/mesh, and
one (`smart_ais_clean`) carries `parent_scan: "smart_ais_3m"` plus a bare
`derivation.parent: "original"` string. The increment-2 validator *reads* these as best-effort
lineage hints and **warns** (never errors) when a legacy link can't be resolved; the bare
legacy `parent` strings are **not** coerced into the new `{ref, fingerprint}` shape. Legacy
scans are not rewritten. New 3.0 clouds carry explicit `derivation.root`/`parent`.

**`laz/` → `raw/` rename happens in increment 1** (the registry that populates `raw/` is
increment 2, but the directory is renamed now): move `lidar/laz/*` → `lidar/raw/`, rewrite the
5 non-null `source_laz` paths (`"lidar/laz/…"` → `"lidar/raw/…"`), and point voxa's
`scene_registry` source_laz resolution at `raw/` (accepting `laz/` too, so a stale path still
resolves rather than failing silently). This is the one piece of existing-data mutation in an
otherwise-grandfathered increment; it is a path rename, not a metadata-model change.

**Scope split:** increment 2 builds `raw/sources.json` (registering every family root,
any format), the `RawSourceRegistry`, and the cross-scan lineage validation. Stamping lineage
onto **cutouts / export outputs** (wiring the edit-raw full-density export) is increment 2's
tail or a later step — the v3.0 contract is defined now so cutouts *can* carry it.

### `scratch/` allow-list (resolves stray-dir drift)

`scratch/` is an allow-listed top-level location for non-schema artifacts: experiment outputs
(`fresh_run/`, `stale_preseg_500k/`), archival clouds, one-off `scripts/`/`sim/` dirs. Its
contents are unchecked; its existence is legal.

Validator policy on **top-level** entries:
- Known schema entry (`README.md`, `meta.json`, `source/`, `prelabel/`, `sessions/`,
  `renders/`, `sam3/`, `variants.json`, `scratch/`) → OK.
- Anything else (`variants` legacy files, `scripts/`, `sim/`, `README_legacy_v2.md`) → **warning**.

### Legacy handling (grandfathering — no forced migration)

The 9 scans on disk are `schema_version: "2.0"` and mostly lack `frame`/`derivation`. v3.0
does **not** rewrite them. The package recognizes both versions:

- **`schema_version` starts with `"3"`** → must fully conform; violations are **errors**.
- **`schema_version` starts with `"2"`** → legacy. Missing `frame`/`derivation`, legacy
  `coords`/`coord_offset_m`, and archival clouds in `source/` are **warnings**, never errors.
  voxa continues to read these (it already synthesizes a default identity `frame` in memory).

voxa's `scene_registry` (today `schema_version.startswith("2")`) is widened to accept `"2"`
**or** `"3"`. New scans voxa writes going forward stamp `"3.0"` with explicit `frame`/`derivation`.
Migrating the 9 legacy scans to 3.0 later is a one-command opt-in (`backfill_scan_frame.py` +
`scan_index.py` + a `scratch/` tidy), explicitly out of scope for this increment.

## Package design

Create `engine/tools/scan-schema/` as **its own git repository**, shipping a Python package
`scan_schema` installed editable (`pip install -e`) and version-pinned by each consumer.
Versioning is the enforcement mechanism: consumers pin `scan-schema==3.x`, so a schema bump is
a deliberate version everyone moves to.

```
engine/tools/scan-schema/
├── pyproject.toml                 # name = "scan-schema", version tracks schema major (3.x)
├── README.md
├── src/scan_schema/
│   ├── __init__.py                # public API
│   ├── layout.py                  # ScanLayout + SessionPaths (lift of voxa scenes/scan_layout.py)
│   ├── frame.py                   # Frame/derivation models + is_rigid (lift of voxa scenes/frame.py)
│   ├── invariants.py              # GT invariants 1–6 (lift of segment_io._validate_invariants + shape/dtype)
│   ├── metadata.py                # meta.json field contract + frame/derivation checks
│   │                              #   (lift of scripts/scan/validate_scan.py logic)
│   ├── sources.py                 # [increment 2] RawSourceRegistry: load raw/sources.json,
│   │                              #   resolve source_id, verify fingerprint; lineage checks
│   ├── storage.py                 # Storage protocol + LocalStorage (the S3 seam)
│   └── validate.py                # whole-archive audit composing all of the above
└── tests/
    ├── test_layout.py
    ├── test_invariants.py
    ├── test_metadata.py
    ├── test_validate.py           # runs against real lidar/annotated fixtures
    ├── test_storage.py            # ReadOnlyStorage write ops raise; WritableStorage round-trips
    └── test_sources.py            # [increment 2] registry + lineage checks
```

### Components

- **`layout.py`** — direct lift of voxa's `scenes/scan_layout.py`: frozen `ScanLayout` /
  `SessionPaths` whose properties *are* the logical resource names. Pure path joins, no disk
  access. This is the seam the later HTTP/S3 increments wrap.
- **`frame.py`** — lift of voxa's `scenes/frame.py` (`Frame` model, `is_rigid`,
  legacy-frame synthesis), so frame/derivation parsing has one home.
- **`invariants.py`** — lift of `segment_io._validate_invariants` (SCHEMA invariants 3–6),
  plus invariants 1–2 (`len(scan.ply) == meta.n_points`; per-array shape `(N_pts,)` and
  **per-array dtype** per the SCHEMA.md table — `working_class_ids` is `int8`,
  `output/gt_class_ids` is `int32`).
- **`metadata.py`** — the `meta.json` field contract above and the frame/derivation checks
  currently in `scripts/scan/validate_scan.py` (`canonical_id == "<scan_id>#local"`,
  `transform_to_canonical` is rigid, `varies` ⊆ allowed set, ≤1 `role: "labeling"`).
  Version-aware: errors for 3.x, warnings for 2.x.
- **`sources.py`** *(increment 2)* — `RawSourceRegistry` loads `raw/sources.json`, resolves a
  `source_id` to its registered entry, and verifies a `derivation.root`/`parent` link (source
  exists, fingerprint matches, cross-scan `ref` resolves via an archive-wide index). Also the
  registration helper that computes a raw file's `{source_id, fingerprint, n_points}` once.
  `ScanLayout` gains archive-level `raw_dir` and `sources_json` properties (siblings of
  `classes_json`). Not built in increment 1.
- **`storage.py`** — small `Storage` protocol (`list`, `stat`, `read_array`, `open`, and the
  write ops `write_array`/`open_w`/`mkdir`). Two impls ship in increment 1: `ReadOnlyStorage`
  (default; write ops raise `ReadOnlyError` with the offending path) and `WritableStorage`
  (filesystem read+write). Readers get `ReadOnlyStorage`; **only voxa** constructs
  `WritableStorage`. The only place that touches a filesystem; the S3 increment adds
  `S3Storage`/`S3ReadOnlyStorage` behind the same protocol, so the S3 move is a new backend,
  not a re-plumbing. `S3Storage` itself is **not** built in this increment.
- **`validate.py`** — walks the archive through `Storage`, resolves paths via `ScanLayout`,
  runs metadata + invariant + top-level-entry checks, returns a structured per-scan report
  (errors + warnings) and a CLI (`python -m scan_schema.validate <root>`) that exits non-zero
  on any error. Replaces both `data/tools/validate_annotated.py` and
  `scripts/scan/validate_scan.py`.

### What gets validated where

- **`sessions/<id>/output/`** — GT pair. Full `validate_invariants` (3–6) + shape/dtype (1–2).
  `output/` is **optional** (absent until first save); a session with no `output/` is a legal
  *unlabeled* session, **not** an error.
- **`prelabel/<id>/`** — preseg result, **no per-point class array**, so GT invariants 3–6 do
  **not** apply. Validate only: `instance_ids.npy` is `int32` shape `(N_pts,)`, and
  `segment_summary.json` parses to `{"segments":[{"id","class_id"},…]}` (`class_id == -1`
  legal — class-agnostic preseg). Do not assert preseg `class_id`s against `classes.json`.
- **`sessions/<id>/working_*.npy`** — shape `(N_pts,)`, dtype per table (`working_class_ids`
  `int8`, `working_segment_ids` `int32`). No 3–6 check (mid-edit, not validated GT).
- **`meta.json`** — field contract + frame/derivation checks, version-gated (errors for 3.x,
  warnings for 2.x).
- **Source lineage** *(increment 2)* — for a 3.x derived cloud, `derivation.root.source_id`
  must resolve in `raw/sources.json` and `root.fingerprint` must match the registry;
  `derivation.parent.ref` must resolve (to the root or another known scan via the archive-wide
  index). Legacy 2.x links (`source_laz`/`parent_scan`/bare `parent`) are best-effort and
  downgrade to warnings. `raw/sources.json` itself is validated for unique `source_id`s and
  well-formed entries. **Increment 1 does not perform lineage checks** — it validates the
  shape of `derivation` (frame/identity fields) only.

### Public API

```python
from scan_schema import ScanLayout, SessionPaths, validate_invariants, validate_archive
from scan_schema.frame import Frame, is_rigid
from scan_schema.storage import Storage, LocalStorage
# increment 2: from scan_schema.sources import RawSourceRegistry
```

### How voxa adopts it

voxa deletes `scenes/scan_layout.py`, the invariant body of
`segment_io._validate_invariants`, and `scripts/scan/validate_scan.py`, depending on
`scan_schema` instead (re-exporting from old import paths if churn is a concern — decided in
the plan). voxa stays the **sole writer** and keeps validating at save against the shared
definition. Adoption must not change voxa's save behavior or break a reader — verified by
voxa's existing `test_segment_io.py` and `test_real_scans_validate.py` passing unchanged.
`data/tools/validate_annotated.py` is deleted; its CLI contract (exit non-zero on error) is
preserved by `python -m scan_schema.validate`.

## Interfaces to the data & write protection

Consumers stop constructing `Path(...)` joins and go through four interface layers (three
in-process now, one over HTTP later):

1. **Path / resource — `ScanLayout`** (increment 1). The "where is X" interface: every logical
   resource is a property (`scan_ply`, `meta_json`, `session(id).output_gt_class_ids`, …). Pure
   path joins. The one place layout is defined.
2. **Transport — `Storage`** (increment 1). The "give me / put bytes" interface
   (`list/stat/read_array/open` + write ops). `LocalStorage` today, `S3Storage` later — same
   protocol, so local→S3 is a swap. This is also the write-protection seam (below).
3. **Validation — the value contract** (increment 1). `validate_invariants(class_ids,
   instance_ids, …)` at point level (voxa's save-gate) and `validate_archive(root, storage)` /
   `python -m scan_schema.validate <root>` for whole-archive audit.
4. **Network — resource API** (increment 4, later). HTTP routes that *are* the `ScanLayout`
   resource names; JSON for small payloads, a location (path now, presigned URL later) for big
   binaries (never proxied).

### Access model

| Consumer | Interface | Access |
|---|---|---|
| **voxa** (sole writer) | `ScanLayout` + `WritableStorage` + `validate_invariants` (save-gate) | read + **write** |
| validator / CI | `validate_archive` / CLI | read-only audit |
| meshbuilder, training (readers) | `ScanLayout` + `ReadOnlyStorage` | read-only |
| lineage tools (increment 2) | `RawSourceRegistry` | read-only |

One writer, many readers. Enforcement is voxa's save-gate (write-time) + the audit validator
(after-the-fact), both calling the *same* `scan_schema` definition, so they cannot disagree.

### Write protection (layered)

Nothing in Python truly *prevents* a write — `ReadOnlyStorage` only stops writes that go
*through the package*. Real prevention is OS perms (local) / IAM (S3). So protection is layered:

- **Software guard (increment 1):** `ReadOnlyStorage` is the default; only voxa constructs
  `WritableStorage`. Accidental writes through the package fail loud with the path, instead of
  silently corrupting data.
- **OS hardening (documented runbook, increment 1):** `lidar/` writable only by the voxa/owner
  account; readers run without write perms. `chattr +i` on `raw/` roots and every
  `source/scan.ply` — these must **never** change, because `derivation` fingerprints depend on
  them; a silently-edited root invalidates a whole family's lineage. This is documented in
  SCHEMA.md / a setup note, not enforced by code.
- **Recovery:** the data is too big for git, but the *metadata* (`meta.json`, `*.json`,
  `sources.json`) is tiny — tracking just those in git (clouds git-ignored) makes accidental
  metadata corruption recoverable.
- **S3 (later increment):** read-only IAM creds for everyone, `PutObject` only for the writer,
  bucket versioning + object-lock. A validating **write-gateway** (sole creds-holder) is the
  maximal version — **deferred** (YAGNI with one trusted writer that already validates at save).

## Data flow

```
                 scan_schema (one v3.0 definition)
                 ┌──────────────────────────────────────────┐
   voxa save ───▶│ layout · frame · invariants · metadata     │
   validator ───▶│              · storage                      │──▶ LocalStorage ──▶ lidar/annotated/
   (readers,     └──────────────────────────────────────────┘
    later)
```

One writer (voxa), many readers, one schema definition under all of them. `schema_version` is
the coordination point: 3.x is enforced, 2.x is read-with-warnings, anything else fails loud
with a migration hint.

## Error handling

Fail loud, never mask. Errors (invariant violations, missing required files, malformed
`meta.json`, non-rigid `transform_to_canonical`, shape/dtype mismatch, missing
`frame`/`derivation` on a 3.x scan) exit non-zero. Warnings (legacy 2.x drift, unknown
top-level entries, archival clouds in `source/`) are reported but non-fatal. A scan that can't
be read is an error with its path, never a silent skip.

## Testing

- `test_layout.py` — property path joins resolve to documented locations.
- `test_invariants.py` — each invariant 1–6 has a passing and a violating case (ports voxa's
  existing invariant tests).
- `test_metadata.py` — frame/derivation checks; a 3.x scan missing `frame` errors, a 2.x scan
  missing it warns; non-rigid `transform_to_canonical` errors; `varies` subset enforced.
- `test_sources.py` *(increment 2)* — `raw/sources.json` loads; a `derivation.root` with an
  unknown `source_id` or mismatched `fingerprint` errors on 3.x, warns on 2.x; duplicate
  `source_id` in the registry errors; legacy `source_laz` resolves to a registry source by
  fingerprint.
- `test_storage.py` — `ReadOnlyStorage` write ops raise `ReadOnlyError`; `WritableStorage`
  round-trips an array; both read identically.
- `test_validate.py` — runs `validate_archive` against the real `lidar/annotated/` fixtures:
  all 9 current (2.0) scans pass with **errors == 0** (their legacy gaps are warnings),
  unlabeled sessions are not flagged, and known strays (`fresh_run/`, archival clouds) surface
  as warnings.
- voxa's `test_segment_io.py` + `test_real_scans_validate.py` pass unchanged after adoption —
  the regression gate.

## Roadmap (out of scope here, recorded so increment 1 doesn't paint us into a corner)

1. **Increment 1 (this spec):** the `scan-schema` package (layout + frame + invariants +
   metadata + storage seam) encoding v3.0; the unified validator replacing all three current
   ones; voxa adoption; the `scratch/` allow-list; the `lidar/laz/`→`lidar/raw/` rename (+ the
   5 `source_laz` path fixes + voxa resolver update); and `lidar/SCHEMA.md` rewritten to v3.0
   (which *documents* the `derivation.root`/`parent` contract). Existing scans otherwise
   grandfathered (warned, not migrated). **No registry, no lineage validation.** Zero new infra.
2. **Increment 2 — source families:** `sources.py` + `raw/sources.json` registering every
   family root (any format/location), the `derivation.root`/`parent` lineage validation
   (incl. the cross-scan index), and the legacy-link fallback. Tail: wire the edit-raw
   full-density export (and any cutout producer) to stamp lineage on its outputs.
3. **Increment 3 — opt-in legacy migration:** one command (`backfill_scan_frame.py` +
   `scan_index.py` + `scratch/` tidy + `source_laz`→`derivation.root`) to bring the 9 scans to
   fully-conformant 3.0.
4. **Increment 4 — HTTP service:** thin FastAPI app whose routes are the `ScanLayout` resource
   names; JSON for small payloads, a location (path now, presigned URL later) for big binaries.
5. **Increment 5 — S3 backend:** add `S3Storage` behind the `Storage` protocol; module,
   validator, service, voxa, readers unchanged.
6. **Later:** migrate meshbuilder/training readers onto `scan_schema`; auto-gate the validator.

## Decisions (pinned 2026-06-09)

- **v3.0 = v2.0 layout + v1.3 metadata, one version number.** `schema_version: "3.0"`.
- **`frame` + `derivation` required for 3.x; warned (grandfathered) for legacy 2.x.** No
  forced backfill of existing scans this increment.
- **Archival clouds + `mesh.optimized.glb` → `scratch/`** (opt-in tidy; validator warns if
  left in `source/`). `coords`/`coord_offset_m` removed in 3.0, folded into `frame.georef`.
- **Source families:** raw scans function as roots, registered in `lidar/raw/sources.json`
  **regardless of format/location** (shared LAZ, matterport full-res PLY, munich mesh). Every
  derived cloud carries `derivation.root` (`{source_id, fingerprint}`, supersedes `source_laz`)
  + `derivation.parent` (`{ref, fingerprint}`), fingerprint-based. v3.0 **defines** this
  contract; the registry + cross-scan validation are **increment 2** (split out per spec
  review — independent subsystem). `laz/`→`raw/` rename + the 5 `source_laz` path fixes happen
  in increment 1.
- **Unify all three validators into the package**: delete `data/tools/validate_annotated.py`
  and voxa `scripts/scan/validate_scan.py`; voxa save-gate imports `scan_schema`.
- **Write protection (layered):** `ReadOnlyStorage` is the default; only voxa constructs
  `WritableStorage` (increment 1). OS hardening (perms + `chattr +i` on `raw/` & `source/scan.ply`)
  documented as a runbook. IAM read-only-by-default is the S3-increment answer; the validating
  write-gateway is deferred (YAGNI with one trusted writer).
- **Location/packaging:** standalone git repo at `engine/tools/scan-schema/`, package
  `scan_schema`, installed editable and pinned by consumers.
- **Increment 1 scope:** package (layout + frame + invariants + metadata + storage incl.
  ReadOnly/Writable) + unified validator + voxa adoption + `scratch/` allow-list + `laz/`→`raw/`
  rename (+ 5 `source_laz` fixes + resolver update) + write-protection runbook + SCHEMA.md v3.0
  rewrite. **Out:** the source-families registry +
  lineage validation (increment 2), auto-gate, service, S3, reader migration, forced legacy
  migration.
