# scan-schema: single-source-of-truth schema enforcement for `lidar/annotated/`

**Date:** 2026-06-09
**Status:** Design — approved for spec review
**Related:** `lidar/SCHEMA.md` (v2.0), `voxa/docs/superpowers/specs/2026-06-03-multi-session-preseg-design.md`

## Problem

`lidar/annotated/` is the shared datastore of labeled lidar scans. Its layout is documented
in `lidar/SCHEMA.md` (currently **v2.0**) and re-encoded, independently, in **five** places:

| Consumer | Role | Schema knowledge | State |
|---|---|---|---|
| `lidar/SCHEMA.md` | human doc | prose spec | v2.0 |
| voxa `scenes/scan_layout.py` + `labeling/segment_io._validate_invariants` | the **only writer** (validates at save) | executable encoding | v2.0, tested, working |
| `data/tools/validate_annotated.py` | standalone audit | re-encoded by hand | **v1.3 — rotted** |
| meshbuilder `discovery.py` | reader | own path resolution | unaudited |
| training pipelines (`train_xscene_real.py`) | reader | own loader | unaudited |

Because every consumer re-derives the schema, they drift apart. The proof is already on
disk: voxa moved to v2.0, the standalone validator silently stayed on v1.3, and it now
fails **all 10 scans with 27 errors** — every error is a missing `labels/gt_class_ids.npy`,
a path v2.0 deliberately removed. **Today nothing is being enforced**: the one validator we
have rejects 100% of valid data.

There is also real on-disk drift — top-level entries that appear nowhere in `SCHEMA.md`:

| Stray entry | Appears in | What it is |
|---|---|---|
| `fresh_run/` | 5 scans | RANSAC / curvature experiments |
| `stale_preseg_500k/` | 4 scans | superseded presegs |
| `scripts/`, `sim/`, `README_legacy_v2.md` | munich_water_pump | one-off legacy cruft |
| `variants.json` | 2 scans | undocumented |
| no `sessions/` at all | matterport ×2 | unlabeled / half-imported |

**The data drift is downstream of the schema-definition drift.** A new hand-coded validator
would just become the sixth thing to rot. The fix has to remove the duplication, not add to it.

## Goal

One executable definition of the v2.0 schema that every consumer imports, so a schema change
is a single deliberate version bump rather than silent per-app divergence — and a validator
built on that same definition that *cannot disagree* with what voxa enforces at write time.

Non-goals (explicitly deferred): the HTTP service, the S3 backend, auto-gating (pre-commit/CI),
and migrating meshbuilder/training readers. Those are later increments (see Roadmap); this spec
covers **increment 1** only.

## Design

### A standalone, versioned package: `scan-schema`

Create `engine/tools/scan-schema/` as **its own git repository** (not a subdir of voxa, not
inside the `lidar/` data archive). It ships a Python package `scan_schema`, installed
editable (`pip install -e`) by each consumer. Versioning is the enforcement mechanism:
consumers pin `scan-schema==2.x`, so a schema bump becomes a version everyone deliberately
moves to — the exact failure mode (silent v1.3↔v2.0 divergence) that rotted the validator
becomes impossible.

```
engine/tools/scan-schema/
├── pyproject.toml                 # name = "scan-schema", version pinned to schema major (2.x)
├── README.md
├── src/scan_schema/
│   ├── __init__.py                # public API surface (see below)
│   ├── layout.py                  # ScanLayout + SessionPaths (lifted from voxa scan_layout.py)
│   ├── invariants.py              # validate_invariants() (lifted from segment_io._validate_invariants)
│   ├── storage.py                 # Storage protocol + LocalStorage (the S3 seam)
│   └── validate.py                # whole-archive audit, built on layout + invariants + storage
└── tests/
    ├── test_layout.py
    ├── test_invariants.py
    └── test_validate.py           # runs against real lidar/annotated fixtures
```

### Components

**`layout.py` — the path contract.** A direct lift of voxa's `scenes/scan_layout.py`:
the frozen `ScanLayout` and `SessionPaths` dataclasses whose properties *are* the logical
resource names (`.scan_ply`, `.session(id).output_gt_class_ids`, `.preseg_dir(id)`, …).
Pure path joins, no disk access. This is the seam the later HTTP/S3 increments wrap; keeping
it pure keeps those increments cheap.

**`invariants.py` — the value contract.** A lift of `segment_io._validate_invariants`, which
already encodes SCHEMA.md invariants 3–6 (class/instance `-1` agreement, per-segment class
consistency, class IDs ∈ `classes.json`, `class_map_version` match). Invariants 1–2
(`len(scan.ply) == meta.n_points`; per-array shape `(N_pts,)` and per-array dtype per the
SCHEMA.md table — note `working_class_ids` is `int8` while `output/gt_class_ids` is `int32`)
are added here so the package covers all six in one place.

**`storage.py` — the transport seam.** A small `Storage` protocol (`list`, `stat`, `open`,
`read_array`) with a single implementation today, `LocalStorage` (thin wrapper over
`pathlib`/`numpy`). This is the only place that touches a filesystem. The S3 increment adds
an `S3Storage` behind the same protocol; layout, invariants, and validate come along for free.
We introduce the seam now (it costs ~30 lines) precisely so the S3 move is "a new backend,"
not a re-plumbing — but we do **not** build `S3Storage` in this increment.

**`validate.py` — the audit.** Walks the archive through `Storage`, resolves every path via
`ScanLayout`, checks required files exist, asserts `meta.json::schema_version == "2.0"`
(any other value → loud failure with a migration hint, never a silent skip), runs
`validate_invariants` over each `sessions/*/output/` and `prelabel/*/`, and reports
**unknown top-level entries** as warnings. This replaces `data/tools/validate_annotated.py`.

### The `scratch/` allow-list (resolves the stray-dir drift)

`SCHEMA.md` gains an explicit **allowed top-level entries** list. To it we add one new legal
entry: `scratch/` — an allow-listed location for experiment outputs (`fresh_run/`,
`stale_preseg_500k/`, etc.). Its *contents* are unchecked; its *existence* is legal. Stray
dirs get moved under `scratch/` (mechanical, done during implementation, tracked in the plan).

Validator policy on top-level entries:
- **Known schema entry** (`README.md`, `meta.json`, `source/`, `prelabel/`, `sessions/`,
  `renders/`, `sam3/`, `scratch/`) → OK.
- **Anything else** (`variants.json`, `scripts/`, `sim/`, `README_legacy_v2.md`) → **warning**
  (reported, non-fatal). This surfaces drift without blocking; promoting specific warnings to
  errors is a later policy decision, not part of this increment.

### Public API

```python
from scan_schema import ScanLayout, SessionPaths, validate_invariants, validate_archive
from scan_schema.storage import Storage, LocalStorage
```

`validate_archive(root, storage=LocalStorage())` returns a structured report
(per-scan errors + warnings) and the CLI entry point (`python -m scan_schema.validate <root>`)
exits non-zero on any error — the same contract the old validator had, so existing muscle
memory / any wrapper keeps working.

### How voxa adopts it

voxa deletes its own `scenes/scan_layout.py` and the invariant body of
`segment_io._validate_invariants`, depending on `scan-schema` instead (re-exporting from the
old import paths if churn is a concern — decided in the plan). voxa stays the **sole writer**
and keeps validating at save; it just validates against the shared definition. Adoption must
not change voxa's save behavior or break any reader — verified by voxa's existing
`test_segment_io.py` and `test_real_scans_validate.py` passing unchanged.

## Data flow

```
                 scan_schema (one definition)
                 ┌───────────────────────────────┐
   voxa save ───▶│ layout · invariants · storage  │
   validator ───▶│                                │──▶ LocalStorage ──▶ lidar/annotated/
   (readers,     └───────────────────────────────┘
    later)
```

One writer (voxa), many readers, one schema definition under all of them. `schema_version`
in `meta.json` is the coordination point: a reader hitting non-"2.0" data fails loud with a
migration hint instead of mis-parsing.

## Error handling

Fail loud, never mask. The validator distinguishes **errors** (invariant violations, missing
required files, wrong `schema_version`, shape/dtype mismatch) from **warnings** (unknown
top-level entries). Errors exit non-zero; warnings are reported but non-fatal. `validate_invariants`
raises `ValueError` with the offending invariant number and point count (preserving voxa's
current messages). No empty excepts, no silent skips — a scan that can't be read is an error
with the path, not an omission.

## Testing

- `test_layout.py` — property path joins resolve to the documented locations.
- `test_invariants.py` — each invariant 1–6 has a passing case and a violating case (ports
  voxa's existing invariant tests so we know behavior is preserved on lift).
- `test_validate.py` — runs `validate_archive` against the real `lidar/annotated/` fixtures:
  all 10 current scans pass (errors == 0), and the known strays surface as warnings.
- voxa's own suite (`test_segment_io.py`, `test_real_scans_validate.py`) passes unchanged
  after voxa switches to the package — the regression gate for adoption.

## Roadmap (out of scope here, recorded so increment 1 doesn't paint us into a corner)

1. **Increment 1 (this spec):** the `scan-schema` package + rebuilt validator + voxa adoption
   + `scratch/` allow-list. Enforcement and the drift-fix land with zero new infra to operate.
2. **Increment 2 — HTTP service:** a thin FastAPI app whose routes are the `ScanLayout`
   resource names (`GET /scenes/{name}/source`, `.../sessions/{id}/output`). Returns validated
   JSON for small/structured payloads; returns a **location** (file path now, presigned URL
   later) for big binaries (PLY/`.npy`) — never proxies hundreds of MB. Justified when
   consumers run on different machines or S3 arrives.
3. **Increment 3 — S3 backend:** add `S3Storage` behind the `Storage` protocol; big-binary
   responses become presigned URLs. Module, validator, service, voxa, readers unchanged.
4. **Later, as needed:** migrate meshbuilder/training readers off hand-rolled path resolution
   onto `scan_schema`; auto-gate the validator (pre-commit in the scan-schema repo / CI).

## Decisions (pinned 2026-06-09)

- **Location/packaging:** standalone git repo at `engine/tools/scan-schema/`, package
  `scan_schema`, installed editable and pinned by consumers.
- **Increment 1 scope:** module + rebuilt validator + voxa adoption + `scratch/` allow-list.
  No auto-gate, no service, no S3, no reader migration yet.
- **Stray dirs:** allow-list a `scratch/` location in `SCHEMA.md` and move experiment outputs
  there; the validator warns (does not error) on any other unknown top-level entry.
