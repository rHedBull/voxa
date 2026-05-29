# Scan Schema v1.3 — Phase 3: Recover Real Frames & Wire Into the Live Pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development or superpowers:executing-plans.

**Goal:** Turn the v1.3 metadata + resolver (Phases 1–2) into an active part of the pipeline — first by recovering and recording the real frame transforms for existing scans (3a), then by wiring the resolver/remap into the live consumers so mismatches are auto-corrected or refused in production (3b).

---

## Phase 3a — recover & record real frames ✅ DONE (2026-05-29)

Committed: `scripts/backfill_scan_frame.py` — ICP-registers `scan.ply` ↔ the aligned
cloud decoded from its Potree octree, writes the scan `meta.json`
`frame`(identity=canonical)+`derivation`(role=labeling) and `renders/<run>/meta.json`
(pins the aligned variant + recovered `transform_to_canonical`), then proves
resolver→remap→health-check.

Applied to **`navvis_vlx3_water_treatment`** (writes in the lidar data archive, not git):
pure-translation transform, ICP **fitness 1.000**; registration **FAIL→PASS**
(coverage 15%→**51.9%**, photometric 43.7%→**92.0%**); `validate_scan` → **OK**.

**3a remaining:** backfill the other render-having scans (notably `smart_ais_clean`,
which also fails the check today) to confirm the mismatch is systemic and to calibrate
`τ_cov`/`τ_photo` against a now-good scene.

---

## Phase 3b — wire resolver/remap into the live pipeline

This modifies live consumer code, so do it TDD with the full suite as the guard.

### 3b.1 SAM3 stage consumes renders via the resolver ✅ DONE (2026-05-29)
`resolver.dir_cloud_transforms` (per-dir remap, raises on cross-scan fail; 4 tests);
`extract_or_load` gained optional `cloud_frame`/`cloud_variant_id`/`cloud_fingerprint_str`
+ per-dir remap of the oriented cloud + the remap folded into the cache key (back-compat:
unchanged when not passed). `presegment_sam3_features.py` reads `scan_meta` and passes the
frame; its gate applies the recorded remap before the coverage/photometric check.
`check_registration` refined: **photometric is primary**, coverage is a low floor (the
8-frame-sample coverage was tripping the old 0.35 threshold despite 92.9% photometric).
Verified on navvis (no-torch, to the GPU boundary): gate PASS post-remap, "remapping 3/3",
cache key changes. Full suite 211 passing.

### 3b.1-original SAM3 stage consumes renders via the resolver
- **`backend/preseg/sam3_features.py::extract_or_load`**: it currently rotates one
  cloud by `orientation` and projects against all render poses. Change it to, per
  render run: read `renders/<run>/meta.json`, `resolve_render_run(cloud_frame,
  cloud_variant, fp, run_meta)`, and on `remap` apply the transform to the cloud
  (`apply_transform(R4 @ inv(M), xyz)`) before projecting that run's frames; on `fail`
  raise. Keep `use_direct` as today. Requires threading the cloud's `Frame` +
  `variant_id` + fingerprint in (from `read_scan_meta`).
- Tests: a synthetic scan dir (tmp) with a cloud + a render run whose `meta.json`
  encodes a known offset → assert projected coverage matches the remapped expectation;
  a `fail` case raises.

### 3b.2 voxa load surfaces the frame ✅ DONE (2026-05-29)
`scan_meta.frame_summary(scan_dir)` (schema_version / variant_id / frame_canonical_id /
frame_uncertain / georef_offset; `{}` for non-annotated tiers). `LoadResponse` gained
those optional fields; `/api/load` populates them (additive, try/except — never breaks a
load). Verified: navvis → schema 1.3 + frame + georef; legacy munich → `frame_uncertain:
true`. Full suite 214. (Remaining 3b.2 nicety: have the loader/UI actively *run* the §6
check when `frame_uncertain` — currently it only surfaces the flag.)

### 3b.2-original voxa load / scene_registry frame-aware
- **`backend/scenes/scene_registry.py`** (and `routes/load.py`): replace the ad-hoc
  `coords`/`source_laz`→Z-up decision with `read_scan_meta` → `Frame`; surface the
  frame (and `frame_uncertain`) so Inspect/Compare know the cloud's frame.
- When `frame_uncertain` is true, the loader should run / surface the §6 check.

### 3b.3 honor `frame_uncertain`
- Anywhere a legacy (v1.2) scan is consumed with renders, force the registration
  health-check before trusting poses.

## Phase 4 — completeness (separate plans)

- **`variants.json` generator** `scripts/scan_index.py` (§4.2) + cross-variant
  fingerprint resolution (resolver M5) so cross-variant pins resolve on disk.
- **Multi-run** `labels/runs/` + `prelabel/runs/` collections + Compare/merge (§4.6)
  and label propagation §5.4 (the §5.4 propagation algorithm: NN within
  `propagate_radius_m`, unmatched→-1, persist + re-fingerprint).
- **Writers**: voxa export + walker render export emit v1.3 metas natively (born
  conformant); backfill remaining scans.
- **Replace `lidar/SCHEMA.md`** with the v1.3 design doc; bump header + changelog.
- Wire `verify_registration` + `validate_scan` into CI as gates.

## Done criteria (3b)
- Running the SAM3 stage on a mis-registered-but-pinned scan auto-remaps and produces
  correct coverage (no manual transform); an unpinned/cross-scan render run is refused.
- voxa load reads the frame from `read_scan_meta`; `frame_uncertain` scans trigger the
  health-check. Full backend suite green.
