# Scan Schema v1.3 — Versioned Clouds, Explicit Frames, Pinned Artifacts

**Status:** Draft for review · **Date:** 2026-05-29 · **Supersedes:** `lidar/SCHEMA.md` v1.2
**Owner of record:** `engine/data/lidar/SCHEMA.md` (this becomes its content on adoption)

> Version note: the live `SCHEMA.md` header says "v1.1" but its changelog already
> records a **v1.2** (2026-05-16, the `renders/`+`sam3/` additions). To avoid a
> colliding version string this spec is **v1.3**, not v1.2.

## 1. Motivation

A scan has **one original capture** and **many derived versions** (downsampled,
aligned, cleaned, re-origined, Potree-converted). Artifacts — renders + camera
poses, SAM3 feature caches, prelabels, GT labels — are only valid in the coordinate
frame **and** point set of the version they were derived from. v1.2 cannot record
*which version* an artifact came from, and its `coords` string
(`"world_minus_offset"` + a translation offset) cannot represent a rigid
**alignment** (rotation+translation) at all.

### The incident this prevents (navvis_vlx3_water_treatment, 2026-05)

- voxa's `source/scan.ply` was the **unaligned** voxel downsample of the raw LAZ.
- The `renders/` (hence the SAM3 pipeline's camera poses) were generated from a
  **different, re-aligned** cloud (`scan_parent_aligned.las` / `scan_15M.las`,
  origin-anchored bbox `[0, 32.78]`). Verified: render camera positions span
  x∈[11,30], z∈[−14,−7]; `scan.ply` spans x∈[−20,12], z∈[−2,4] — **the frames do
  not overlap.**
- Projecting `scan.ply` through those poses hit the **wrong points** (~15 % coverage,
  labels on geometry invisible from each view). **No error was raised.**
- The SAM3 cache key hashed `n_points`, so a recentered-but-same-count cloud silently
  reused a stale cache.

The fix is structural: **make frame identity explicit, make every artifact name the
version it came from, make a content change detectable, and — above all — make a
mismatch *loud* via an empirical check that does not depend on the metadata being
correct.**

## 2. Decisions (resolved from review)

1. **GT labels live on one designated *labeling variant*** (`role: "labeling"`, ≤1
   per scan). Per-point arrays are physically bound to a point set; one labeling
   variant is the GT home. Other variants receive labels by explicit propagation
   (§5.4), never implicitly.
2. **The canonical frame is a per-scan *local metric* frame** (recentered, small
   coords — float32-safe), **not** raw UTM. World georeference is stored separately
   (`frame.georef`). This resolves the float32-vs-UTM precision conflict; the
   labeling variant defines the canonical frame (`transform_to_canonical = identity`).
3. **v1.3 is in-place and backward-compatible** — new blocks/files added to the
   existing layout; no physical migration required. A future v2 MAY adopt the
   `scans/<scan_id>/…` tree (§8.3).
4. **The empirical health-check (§6) is the primary safety mechanism.** The metadata
   pins (§4–§5) are the fast path and the audit trail; they are *not* trusted as the
   sole guard, because metadata can be wrong. The check caught navvis at 15 %
   coverage independent of any metadata.
5. **Annotations and presegs are parallel, coexisting *equal runs*** (review option
   B), not single slots. No run is privileged "of-record"; tools enumerate runs and
   may compare or merge them (§4.6). The v1.2 single `labels/`/`prelabel/` files
   survive only as optional back-compat aliases to a nominated `default_run`.

## 3. Core model

```
scan_id  ─ the ONE source of truth (a physical capture)
   │
   ├── canonical frame  ─ a per-scan LOCAL metric frame (recentered); georef stored separately
   │
   └── variants[]       ─ derived clouds, each a node in a derivation DAG
         variant_id                unique within the scan
         parent                    variant_id | "original"      (informational provenance)
         op                        voxel_downsample|align|clean|crop|recolor|asis (informational)
         varies[]                  WHAT changed vs parent ⊂ {density, frame, points, color, attributes}
         params                    op-specific                   (informational)
         transform_to_canonical    4×4 rigid; maps THIS variant's coords → canonical (identity if canonical)
         source_fingerprint        content identity of this variant's stored cloud (§3.2)
         role?                     "labeling" for the GT home (≤1 per scan)
```

`op`/`params`/`parent` are **informational provenance** — no enforcement rule
consumes them; implementers need not build DAG-walking logic. Enforcement uses only
`transform_to_canonical`, `source_fingerprint`, `varies`, and `variant_id`.

### 3.1 Frame compatibility & transform composition (normative)

`transform_to_canonical` (call it `T_can`) maps a point expressed in the variant's
coords into the scan's canonical frame, as a 4×4 acting on homogeneous column
vectors: `p_canonical = T_can · p_variant`.

To move data (points **or** camera poses) **from** a source artifact's frame `A`
**into** a target variant's frame `V`:

```
T_{A→V} = inverse(V.T_can) · A.T_can          # apply A→canonical, then canonical→V
p_V = T_{A→V} · p_A
```

(For a camera pose, transform both `position` and `target` by `T_{A→V}`; the rotation
part also rotates any stored basis. `T_{A→V}` is identity iff the two `T_can` are
equal — that is the frame-compatibility test.) Worked example with a non-identity
rotation is in Appendix A.

### 3.2 `source_fingerprint` (normative — was under-specified)

```
source_fingerprint = "sha256:" + sha256(
    bytes( sort_lexicographic( round(xyz_meters * 1000) as int32 ),
           little-endian, xyz only, no color/normals ) )
```

- **Quantize to integer millimetres** → robust to float round-trips on re-save.
- **Sort lexicographically (x,y,z)** → order-independent (voxel/thinning reordering
  does not change it).
- **xyz only**, fixed dtype/endianness → recolor/attribute edits don't false-trip it
  (those are `varies:["color"|"attributes"]`, tracked separately).
- It is the identity of **this variant's stored cloud in its own frame** — it is
  *not* frame-invariant and is *not* expected to be equal across variants. Its job is
  to detect "this exact file changed under the artifacts that pinned it."
- It is a content-identity heuristic, not a cryptographic guarantee: clouds differing
  only by duplicate/coincident points at mm resolution can collide, and a
  dedup-on-resave may not trip it. Acceptable for staleness detection; do not use it
  as a security boundary.

## 4. Schema changes

### 4.1 Scan/variant `meta.json` — add `frame` + `derivation`, deprecate `coords`

```json
{
  "scan_name": "navvis_vlx3_water_treatment",
  "schema_version": "1.3",

  "frame": {
    "canonical_id": "navvis_vlx3_water_treatment#local",
    "label": "unaligned",                       // human hint; MUST be consistent with the transform (validated)
    "transform_to_canonical": [[ … 4×4 rigid … ]],
    "georef": { "crs": "EPSG:32632", "offset_m": [574184.0, 6220868.0, 49.0] },  // canonical-local → world
    "units": "meters",
    "frame_uncertain": false                    // true ⇒ synthesized from legacy coords; forces §6 check
  },

  "derivation": {
    "scan_id": "navvis_vlx3_water_treatment",
    "variant_id": "voxel3M_unaligned",
    "parent": "original",
    "op": "voxel_downsample",
    "varies": ["density"],
    "params": {"voxel_size_m": 0.01715, "target_points": 3000000},
    "source_fingerprint": "sha256:…",
    "role": null
  },

  "source_laz": "lidar/laz/NavVis-VLX-3-data-water-treatment-facility.laz",
  "n_points": 3000000, "units": "meters", "class_map_version": 1,
  "capture_date": null, "scanner": null, "notes": ""
}
```

- `frame.transform_to_canonical` is a full 4×4 (a pure offset is just translation; an
  alignment carries rotation — the v1.2 gap).
- `frame.canonical` is **local** (float32-safe); `frame.georef` recovers world coords.
- `frame.label` and `frame.canonical_id` are not authoritative — the **transform** is.
  A validator (§7) checks `label`/`canonical_id` don't contradict it (resolves the
  "frame.id can silently disagree" problem; there is no standalone `frame.id`).
  `canonical_id` is **derived**, not free-text: it MUST equal `<scan_id> + "#local"`,
  so every variant of a scan carries the identical string (a validator enforces it) —
  it is a readability alias for the canonical frame, never a second identity.
- **Back-compat:** if `frame`/`derivation` absent, read v1.2 `coords`/`coord_offset_m`
  as `op:"recenter", varies:["frame"]`, **translation-only**, and set
  `frame_uncertain:true` → §6 health-check is then mandatory (a legacy cloud that was
  actually *aligned* would otherwise be mis-described as translation-only).

### 4.2 `variants.json` (generated index, required iff cross-variant pins exist)

A scan-level index of the sibling variants and their transforms, so a consumer can
resolve a pinned `variant_id` without scanning roots. **Authoritative data is each
variant's own `meta.json`; `variants.json` is a generated cache** (regenerated by a
tool, never hand-edited) to avoid a new drift class. It is **required whenever any
artifact pins a different variant than the one it sits beside** (otherwise a
cross-variant pin is unresolvable). Paths are resolved **relative to the engine repo
root** (so walker-hosted Potree variants resolve too).

```json
{
  "scan_id": "navvis_vlx3_water_treatment",
  "canonical_id": "navvis_vlx3_water_treatment#local",
  "labeling_variant": "voxel3M_aligned",
  "generated_by": "tools/scan_index.py", "generated_at": "…",
  "variants": [
    {"variant_id": "voxel3M_unaligned", "varies": ["density"], "source_fingerprint": "sha256:…",
     "transform_to_canonical": [[…]], "path": "engine/data/lidar/annotated/navvis_vlx3_water_treatment"},
    {"variant_id": "aligned15M", "varies": ["frame","density"], "source_fingerprint": "sha256:…",
     "transform_to_canonical": [[…]], "path": "engine/tools/walker/robot-patrol-sim/static/pointclouds/navvis_vlx3_water_treatment"}
  ]
}
```

### 4.3 Render run `meta.json` (new, required per render run)

`renders/<run_id>/` gains a `meta.json` beside `manifest.json`. **This is the file
that closes the navvis hole** — it names the cloud the poses were rendered from and
the frame they live in. `run_id` **is the `renders/<dir>` directory name**
(authoritative; not the manifest's internal `name`).

```json
{
  "run_id": "lower_behind_stairs",
  "generated_from": {
    "scan_id": "navvis_vlx3_water_treatment",
    "variant_id": "aligned15M",
    "source": "potree:navvis_vlx3_water_treatment (scan_15M.las)",
    "source_fingerprint": "sha256:…",
    "n_points": 11820483
  },
  "frame": { "canonical_id": "navvis_vlx3_water_treatment#local", "transform_to_canonical": [[…]], "units": "meters" },
  "intrinsics": { "fov_deg": 60, "fov_axis": "vertical", "aspect": 1.169, "width": 926, "height": 792 },
  "renderer": { "engine": "potree/three.js", "up_axis": "Y",
                "pose_convention": "look_at(position,target,up=+Y) then roll by roll_deg about view axis" },
  "generated_at": "2026-05-21T13:16:39Z", "n_frames": 13
}
```

- `intrinsics` is authoritative (no consumer-side FOV guess); `fov_axis` removes the
  vertical/horizontal ambiguity (I had to assume "vertical" by hand on navvis).
- `pose_convention` states how `manifest.json`'s per-frame `roll_deg` composes (roll
  is **not** expressible by look_at+up alone — was an unrecorded DoF).
- `manifest.json` (per-frame poses) is unchanged; per-frame `intrinsics` become
  redundant once the run-level block exists.

### 4.4 Artifact pinning — `sam3`, `prelabel`, `labels`

Every derived artifact records the variant(s) + content it was computed from:

- `sam3/sam3_features.npz` meta: `source_variant_id`, `source_fingerprint`,
  **`render_run_ids` (a list** — a cache legitimately fuses multiple runs; navvis used
  `[upper, lower, lower_behind_stairs]`), and `frame`.
- `prelabel/ransac_segment_summary.json`: `source_variant_id`, `source_fingerprint`.
- `labels/gt_segment_metadata.json`: already has `source_fingerprint`/`prelabel_fingerprint`
  (v1.1) — add `source_variant_id` and `frame`. These become **required** for v1.3
  artifacts (were optional in v1.1).

### 4.5 Cache keys hash content, not counts

Any cache keyed on a cloud (notably `sam3_features`) MUST include `source_fingerprint`
(§3.2), **not** `n_points`. Converts the v1.2 silent stale-reuse into a clean miss.

### 4.6 Multiple runs — parallel annotations & presegs (first-class)

Both annotations and presegs are **collections of coexisting, equal runs**. A *run*
is a standalone, self-describing labeling of one variant. No run is privileged; tools
enumerate runs and may compare or merge them. The v1.2 single-slot files become
optional aliases (§4.6.4). Runs live under the variant they label:

```
labels/                          # the GT collection on the role:"labeling" variant
├── runs.json                    # generated registry (kinds, annotators, lineage)
├── runs/<run_id>/
│   ├── gt_class_ids.npy         # int32 (N,)  -1 = unlabeled
│   ├── gt_segment_ids.npy       # int32 (N,)  -1 = unlabeled
│   ├── gt_segment_metadata.json # per-segment metadata
│   └── run.json                 # run meta (below)
├── connectivity_graph.json      # optional, for the default run
└── gt_*.npy / gt_segment_metadata.json   # OPTIONAL v1.2 alias → default_run

prelabel/
├── runs.json
└── runs/<run_id>/{instance_ids.npy, summary.json, run.json}
└── ransac_instance_ids.npy + ransac_segment_summary.json   # OPTIONAL v1.2 alias → default preseg run
```

#### 4.6.1 `run.json`

```json
{
  "run_id": "alice_20260529",
  "kind": "human",                       // human | model | preseg | merge
  "annotator": "alice",                  // person, model name, or tool
  "status": "in-progress | complete",
  "class_map_version": 1,
  "source_variant_id": "voxel3M_aligned",
  "source_fingerprint": "sha256:…",      // the cloud this run is bound to (its own variant)
  "frame": { "canonical_id": "…#local", "transform_to_canonical": [[…]] },
  "base_prelabel_run": "ransac_sam3_20260528",   // seed run id, optional
  "derived_from": [],                    // kind:merge → [run_ids] it was merged from
  "merge_policy": null,                  // kind:merge → see 4.6.3
  "created_at": "…"
}
```

#### 4.6.2 Run kinds & compare

- **`human`** — manual annotation · **`model`** — a prediction (subsumes the v1.2
  `predictions.json`) · **`preseg`** — auto-segmentation (the `prelabel/` runs) ·
  **`merge`** — derived from other runs.
- **Compare** (generalizes voxa Compare mode) — pick **any two runs of the same
  variant**; the server computes precision/recall/F1/IoU. GT-vs-prediction is just
  `human` vs `model`. Runs on *different* variants must first be made compatible
  (same variant, or §5.4 propagation) — **never compared across frames silently** (§5).

#### 4.6.3 Merge & conflicts

A `kind:"merge"` run names `derived_from: [run_ids]` (all the **same variant**) and a
`merge_policy`:
- `priority: [run_ids]` — first listed run wins per point;
- `vote` — majority class/segment per point; ties → `-1`;
- `manual` — resolved in the editor.

A point where sources disagree is a **conflict**; the policy resolves it. The merge
run MAY persist a `conflicts` boolean mask `(N,)` and a per-point `provenance` array
(which source run won) for audit. **Source runs are never mutated.**

#### 4.6.4 Back-compat

The legacy single-slot `labels/gt_*.npy` / `prelabel/ransac_instance_ids.npy` are
**optional aliases** (copy or pointer) to the `default_run` named in `runs.json`. v1.2
readers keep working against the alias; v1.3 readers enumerate `runs/`. If no
`default_run` is set the alias is absent and v1.2-only tools see an unlabeled scan
(acceptable — they predate multi-run).

## 5. Consumer / enforcement rules

When a consumer pairs a cloud variant `V` with an artifact `A` (render run, feature
cache, prelabel):

1. **Direct** — `A.source_variant_id == V.variant_id` **and** fingerprints match → use
   directly. (Still subject to §6 before expensive SAM3 work.)
2. **Frame-only remap** — same `scan_id`, and the A↔V delta is **frame-only** (no
   `points`/`density`/`color` in the symmetric `varies` difference): compute
   `T_{A→V}` (§3.1), **run the §6 health-check, and proceed only if it passes.**
   Otherwise **hard-fail**. (This is the navvis case done *safely* — never silent.)
3. **Point-set differs** — the delta includes `points`/`density`: labels/features are
   **not** directly transferable. **Hard-fail by default.** Cross-point-set transfer
   is allowed only via the explicit, logged propagation of §5.4 — never as an
   implicit step inside a "use these poses" call.
4. **Unresolvable** — different `scan_id`, or a v1.3 artifact with a missing/blank pin
   → **hard-fail**, naming both `variant_id`s and frames. (Fail-closed; v1.2 artifacts
   without pins take the legacy path and force §6.)

> Earlier draft made step 2 *unconditional and silent* — that re-created the incident
> with a green light. It is now gated on the empirical check and refuses point-set
> mismatches.

### 5.4 Label propagation across variants (normative)

To carry GT from the labeling variant `L` to another variant `W` of the same scan:

1. Bring `L`'s labeled points into `W`'s frame via `T_{L→W}`.
2. For each point of `W`, assign the label of its nearest `L` point **within
   `propagate_radius_m`** (default = 1.5 × `W`'s mean point spacing; k=1).
3. Points with **no** `L` neighbour within the radius → `class_id = −1`,
   `segment_id = −1` (preserving invariant §7); count them in the output meta.
4. Propagated labels are **persisted** as `W`'s `labels/`. Because they are bound to
   `W`'s point set, their `source_variant_id = W` and `source_fingerprint = W`'s own
   fingerprint (so the §7 "fingerprint matches the variant it sits on" check holds).
   The cross-variant lineage goes in a separate
   `propagated_from = {variant_id: L, fingerprint: <L@propagation>}` plus
   `propagate_radius_m`. They are regular GT thereafter.

## 6. Enforcement layer — registration health-check (primary defense)

Metadata declares intent; this verifies reality, cheaply, with no SAM3/GPU, and
**independently of metadata correctness**:

- Project `V` into a sample of a run's poses; require **coverage ≥ τ_cov** and
  **photometric agreement ≥ τ_photo** (projected point colours vs render pixels).
- Run as a **required gate before any expensive SAM3 stage**, before §5.2 remaps, and
  as a CI check on scan/artifact writes. On navvis it fires hard (15 % coverage / ~0
  photometric agreement) regardless of what the metadata claims.
- Implementation `scripts/verify_registration.py`; `τ_cov`, `τ_photo` calibrated
  against known-good scans and stored in pipeline config (initial guess τ_cov≈0.35,
  τ_photo≈0.5 — to be tuned). Full design in the companion health-check spec.

A renderless variant (e.g. a density tier used only for Inspect) is **valid** and
skips §6; only *pairing it with renders* invokes §6.

## 7. Invariants (checked by `validate_scan.py`)

Carried from v1.2: every per-point array in `labels/`/`prelabel/` is shape
`(n_points,)`; `class_id == −1 ⟺ segment_id == −1`; `session/` is outside the GT
contract. Added in v1.3:

- `schema_version` present; if ≥ "1.3", `frame` + `derivation` present;
  `transform_to_canonical` is a valid 4×4 rigid (orthonormal R, bottom row `[0,0,0,1]`).
- `frame.label`/`canonical_id` consistent with the transform (no silent disagreement).
- `derivation.varies ⊆ {density, frame, points, color, attributes}`.
- ≤1 variant per `scan_id` has `role:"labeling"`; GT lives there.
- Each render run has `meta.json` with non-blank `generated_from.variant_id` and a `frame`.
- Every v1.3 artifact's `source_fingerprint` matches the current fingerprint of the
  variant it is **stored under** (its own cloud) — mismatch ⇒ stale, flagged loudly.
  Separately, **cross-variant lineage pins** (render `generated_from.variant_id`, sam3
  `source_variant_id` when it names a different variant, label `propagated_from`) are
  resolved via `variants.json` and validated against the *named* variant's fingerprint.
  `canonical_id` equals `<scan_id> + "#local"` on every variant.
- `variants.json` present whenever any cross-variant pin exists, and every referenced
  `path` resolves.
- **Per-run (§4.6):** the per-point shape and `class_id==-1 ⟺ segment_id==-1`
  invariants apply to **each run independently**. Every `run.json` pins
  `source_variant_id` + `source_fingerprint` + `frame`; `run_id`s are unique within
  their collection (`labels/runs`, `prelabel/runs`). `runs.json` is a generated index.
- A `kind:"merge"` run's `derived_from` runs all exist and share its variant; its
  `conflicts`/`provenance` arrays (if present) are shape `(n_points,)`. Compare/merge
  is permitted only across same-variant (or §5-compatible) runs.
- `default_run` (if set) references an existing run; the legacy alias files match it.

## 8. Directory layout

### 8.1 Where everything lives (in-place v1.3 — one directory **is** one variant)

```
engine/data/lidar/
├── classes.json                              # class id↔name map (scan-independent)
└── annotated/<scan_dir>/                      # ONE VARIANT of a scan (v1.2-compatible dir)
    ├── meta.json                  (required)  frame + derivation + fingerprint (§4.1)
    ├── README.md                  (required)  free-form description
    ├── variants.json              (cond.)     generated sibling-variant index (§4.2)
    ├── source/
    │   ├── scan.ply               (required)  THIS variant's points (xyz [+ rgb])
    │   └── mesh.glb               (optional)  canonical geometry, if sampled from a mesh
    ├── labels/                    (GT — parallel runs; only on the role:"labeling" variant; §4.6)
    │   ├── runs.json                          generated registry of runs
    │   ├── runs/<run_id>/
    │   │   ├── gt_class_ids.npy               per-point class id    (-1 = unlabeled)
    │   │   ├── gt_segment_ids.npy             per-point instance id (-1 = unlabeled)
    │   │   ├── gt_segment_metadata.json       per-segment metadata
    │   │   └── run.json                       kind, annotator, source_variant_id, frame, derived_from…
    │   ├── connectivity_graph.json (optional) functional graph (default run)
    │   └── gt_*.npy / gt_segment_metadata.json   (legacy) optional v1.2 alias → default_run
    ├── prelabel/                  (optional)  auto-preseg — parallel runs; §4.6
    │   ├── runs.json
    │   ├── runs/<run_id>/{instance_ids.npy, summary.json, run.json}
    │   └── ransac_instance_ids.npy + ransac_segment_summary.json   (legacy) alias → default preseg run
    ├── session/                   (optional)  live editor state — OUTSIDE the GT contract
    │   ├── current.json                       commit pointer
    │   ├── working_*.npy                      in-progress arrays
    │   └── preseg_runs/<run_id>.npz
    ├── renders/                   (optional)  per-view captures
    │   └── <run_id>/                          run_id = this dir name (§4.3)
    │       ├── meta.json          (required)  generated_from + frame + intrinsics  ← NEW, closes the hole
    │       ├── manifest.json                  per-frame poses (position/target/yaw/roll_deg/file)
    │       └── frame_*.png
    └── sam3/                      (optional)  SAM3 outputs
        ├── sam3_features.npz                  + source_variant_id, source_fingerprint, render_run_ids[], frame
        └── <run_id>/                          optional per-run outputs (instance_ids.npy, info.json)
```

### 8.2 Multiple variants of one scan

Each variant is its own directory (or an externally-registered cloud, e.g. the
walker's Potree dirs). They are tied together by a shared
`derivation.scan_id` and listed in `variants.json`. Variants may live under different
roots; `variants.json[].path` (engine-repo-root-relative) is how a consumer finds a
pinned `variant_id` regardless of where it sits.

**Annotation homes (clarification):** `labels/` under the labeling variant is the
**GT of record**. Tool-local stores — voxa's `data/annotations/<name>/` and any
`session/` dir — are working/scratch state **outside the GT contract**; loaders that
want ground truth read `labels/`, not those.

### 8.3 Target layout (future v2, optional — not required by v1.3)

```
scans/<scan_id>/
├── canonical/                                # original capture; transform_to_canonical = identity
├── variants/<variant_id>/{source/, meta.json}
├── variants.json
└── labels/                                   # on the labeling variant, propagated by §5.4
```

## 9. Migration & adoption

1. Add `schema_version`, `frame`, `derivation` to existing `meta.json` (translate v1.2
   `coords`/`coord_offset_m` → translation-only `frame` with `frame_uncertain:true`).
2. Compute `source_fingerprint` (§3.2) for every `scan.ply`.
3. Emit `renders/<run>/meta.json` for existing runs (for navvis, record `aligned15M` +
   its transform — recovered by registering the decoded Potree cloud to `scan.ply`).
4. Switch the SAM3 cache key to `source_fingerprint`.
5. Land `verify_registration.py` (§6) and wire it as the pre-SAM3 gate.
6. Generate `variants.json` (`tools/scan_index.py`).
7. Update writers (voxa export, SAM3 pipeline, walker render export) to emit v1.3 metas.
8. Replace `lidar/SCHEMA.md` with v1.3 content; bump header + changelog.

## 10. Out of scope (YAGNI)

Non-rigid/per-point transforms (rigid only); global cross-scan instance-ID uniqueness
(still per-scan); the physical v2 re-layout (§8.3, documented not required).

## Appendix A — worked transform example

Variant `A` (renders) `T_can^A = [[0,-1,0,5],[1,0,0,2],[0,0,1,0],[0,0,0,1]]` (90° about
+Z, translate (5,2,0)); target `V` (voxa) `T_can^V = I`. A pose at `position=(1,0,0)`
in A maps into V by `T_{A→V}=inverse(I)·T_can^A = T_can^A`, giving `(0,3,0)` — i.e. the
pose is rotated+translated into canonical(=V). If instead `T_can^V` had its own
rotation, `inverse(V.T_can)` undoes it after A→canonical. Implementations MUST unit-test
this with a non-identity `T_can^V`.

## Open items for reviewer
- §6 thresholds `τ_cov`/`τ_photo` — calibrate against known-good scans.
- §2.2 canonical-as-local vs storing UTM — confirm float32-local + `georef` is acceptable.
- §5.4 `propagate_radius_m` default (1.5× spacing) — validate on a real density pair.

## Implementation status (2026-05-29)

Implemented on branch `worktree-scan-schema-v13` (TDD; full backend suite 204 passing).
Plans: `docs/superpowers/plans/2026-05-29-scan-schema-v1.3-phase{1,2,3}-*.md`.

**Done**
- **Phase 1 (protection):** `cloud_fingerprint` (§3.2); content-based SAM3 cache key
  (§4.5); canonical `reproject`; `registration` health-check (§6); `verify_registration`
  CLI; pre-SAM3 gate. Catches the bug class empirically.
- **Phase 2 (metadata + resolver):** `Frame` + compose/apply/is_rigid (§3.1);
  `render_meta` (§4.3) + `scan_meta` reader with v1.2 `coords` back-compat (§4.1);
  `resolve_render_run` (§5, direct/remap/fail); `validate_scan` linter (§7).
- **Phase 3a (recover + record real frames):** `backfill_scan_frame.py` (ICP recover
  → write `frame`/`derivation` + render-run metas → prove). Applied to
  `navvis_vlx3_water_treatment` (pure translation, ICP fitness 1.000; FAIL→PASS,
  photometric 43.7%→92.0%) and `smart_ais_clean` (ICP fitness 1.000; FAIL→PASS,
  photometric ~40%→85%). Both `validate_scan` → OK. The mismatch was **systemic**.
- **Phase 3b (live wiring):** `extract_or_load` resolves+remaps render runs per-dir
  (3b.1); `/api/load` surfaces the frame (3b.2); `verify_registration` applies the
  recorded remap (`--no-remap` to override).
- **Threshold calibration:** across both backfilled scans, photometric agreement is
  **84.6–93.8% when correctly registered** vs **~36–44% when mismatched** — so the
  default `min_photometric=0.5` separates cleanly with wide margin. Coverage is too
  noisy per-run (13.8–65.7%) to gate on, confirming the photometric-primary check;
  `coverage_floor=0.05` only catches "nothing projects". **Defaults validated.**

**Remaining (TODO)**
- **Phase 3b — live wiring (invasive):** make `extract_or_load` (and voxa load /
  `scene_registry`) resolve each render run via `resolve_render_run` and actually
  apply the `remap` transform before projecting; replace the ad-hoc `coords`/orientation
  handling; honor `frame_uncertain` by forcing the §6 check.
- **`variants.json` generator** (§4.2) + cross-variant fingerprint resolution (M5).
- **Multi-run** `labels/runs/` + `prelabel/runs/` + Compare/merge (§4.6); label
  propagation §5.4.
- **Writers** in voxa export + the walker render export to emit v1.3 metas natively
  (so new scans/renders are born conformant, not backfilled).
- **Replace `lidar/SCHEMA.md`** with this doc; bump header + changelog to v1.3.
- Backfill the other scans' frames; wire `verify_registration`/`validate_scan` into CI.
