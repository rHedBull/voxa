# Handoff — SAM3 floor/wall segmentation + scan-schema v1.3

_Date: 2026-05-29. Read this first, then the spec/plans under `docs/superpowers/`._

## TL;DR

The goal was a **SAM3 prompt-driven floor/wall segmentation** dry-run on
`navvis_vlx3_water_treatment` (`scripts/dry_sam3/prompt_floor_wall.py`). It produced
garbage. Debugging found the root cause was **not** the prompts: voxa's `scan.ply`
was in a **different coordinate frame than the renders** the SAM3 pipeline projects
against (unaligned downsample vs the aligned Potree cloud the renders came from), so
points were projected onto the wrong pixels. We built **scan-schema v1.3** to prevent
that whole bug class, recovered + recorded the real frames, made the pipeline
frame-aware, and are now **about to re-run the floor/wall experiment on the fixed,
correctly-registered cloud.**

## What we did (all merged to `main` + pushed to origin `14b6b9b`)

scan-schema **v1.3** — explicit frames, content fingerprints, pinned artifacts:
- **Phase 1 (protection):** `scenes/fingerprint.py` (deterministic cloud fp);
  content-based SAM3 cache key; `scenes/reproject.py`; `preseg/registration.py`
  (coverage + **photometric** health-check); `scripts/verify_registration.py`;
  pre-SAM3 gate.
- **Phase 2 (metadata + resolver):** `scenes/frame.py` (4×4 `transform_to_canonical`,
  compose); `scenes/render_meta.py`, `scenes/scan_meta.py` (+ v1.2 back-compat);
  `preseg/resolver.py` (`resolve_render_run`: direct/remap/fail + `dir_cloud_transforms`);
  `scripts/validate_scan.py`.
- **Phase 3a (recover real frames):** `scripts/backfill_scan_frame.py` — ICP-registers
  `scan.ply` ↔ the aligned Potree cloud, writes `meta.json` frame/derivation +
  `renders/<run>/meta.json`. Applied to **navvis** and **smart_ais_clean** (both ICP
  fitness 1.000, FAIL→PASS).
- **Phase 3b (live wiring):** `extract_or_load` resolves+remaps render runs per-dir;
  `/api/load` surfaces the frame; `verify_registration` applies the recorded remap.
- **Phase 4:** `scripts/scan_index.py` (`variants.json`); multi-run libs
  `backend/labeling/runs_io.py` + `run_merge.py`; `lidar/SCHEMA.md` bumped to v1.3;
  validate-gate test. (221 backend tests pass.)

**Calibration:** photometric agreement is ~85–94% when correctly registered vs ~40%
when mismatched → `min_photometric=0.5` is the robust separator (coverage is too noisy
per-run). `check_registration` is photometric-primary.

## Current state

- `main` == origin/main == branch `worktree-scan-schema-v13` at `14b6b9b` (pushed).
- **navvis + smart_ais are v1.3-conformant** (`meta.json` schema 1.3, frame recorded,
  `renders/*/meta.json` pinned, `variants.json` generated). `validate_scan` → OK; both
  pass `verify_registration` (with remap).
- The lidar archive files (`engine/data/lidar/...` scan metas, `SCHEMA.md`,
  `variants.json`) are **outside the voxa git repo** (separate data location) — they
  persist on disk, were NOT part of the push.
- User has **uncommitted WIP** in the main checkout (4 files: `backend/app/constants.py`,
  `backend/preseg/sam3_features.py` [CPU-normalize patch], `config/classes.yaml`,
  `frontend/src/mode-label.jsx`) — preserved through every merge via stash/pop. Don't
  clobber it; it has been kept disjoint from the v1.3 changes.

## ⏭️ Immediate next step — resume the floor/wall experiment

`scripts/dry_sam3/prompt_floor_wall.py` was just rewritten (IN THE WORKTREE — the main
checkout still has the OLD untracked copy) to be **frame-aware**: it reads the recorded
frame + render metas and **remaps the cloud into each render run's pose frame** before
projecting, and defaults to **per-pixel `argmax`** floor-vs-wall (mutually exclusive,
fixing the "wall mask subsumes floor" 2D overlap we found).

**Run it (needs anaconda base for torch+SAM3; from the worktree dir):**
```bash
/home/hendrik/anaconda3/bin/python scripts/dry_sam3/prompt_floor_wall.py \
  --scan /home/hendrik/coding/engine/data/lidar/annotated/navvis_vlx3_water_treatment \
  --out /home/hendrik/coding/engine/tools/labeling/voxa/data/fw_experiment/remap
# compare flags: --no-remap (old behaviour), --assign union, --floor-prompts / --wall-prompts sweeps
```
Then view `floor_wall.ply` (Open3D from `.venv`: anaconda's Open3D segfaults) and read
`summary.json` (`n_conflict_both` should now be low with argmax; coverage should be far
higher than the original 15% because the remap is applied).

**Why it wasn't run yet:** a transient Bash safety-classifier outage blocked execution
at handoff time. Just re-run the command above.

**Then:** if floor/wall now separates cleanly, commit the updated
`prompt_floor_wall.py` (currently only in the worktree). If `wall` still over-grabs,
sweep prompts (`--floor-prompts "floor,ground"`, `--wall-prompts "wall,concrete wall"`)
or raise `--mask-thresh`.

## Gotchas for the next agent

- **Two interpreters:** SAM3/torch only in **anaconda base** (`/home/hendrik/anaconda3/bin/python`);
  everything else (tests, Open3D, the .venv pipeline) in voxa's **`.venv`**. Anaconda's
  Open3D SIGSEGVs; voxa's `.venv` has no torch.
- **Worktree base:** this worktree was rebased onto local `main` (restructured
  `backend/preseg`, `backend/scenes` layout). `origin/main` was stale/flat before the
  push; it's now current.
- **Decode the aligned cloud** from Potree at
  `tools/walker/robot-patrol-sim/static/pointclouds/<name>/octree.bin` — DEFAULT encoding,
  bytes/point = sum of metadata `attributes` sizes; position int32×3 at bytes 0:12
  (`*scale + offset`). `backfill_scan_frame.py` already does this.
- **Verify alignment by photometric agreement / cloud-from-pose-vs-photo, NOT coverage
  count** — a coverage jump alone misled an earlier debugging pass.

## Remaining v1.3 work (not started; lower priority than the floor/wall re-run)

- Frontend Compare/merge UI wiring for multi-run (`labeling/runs_io` + `run_merge`).
- Native v1.3 writers in voxa export + the **walker render-export (separate repo)** so
  new scans/renders are born conformant (today they're backfilled).
- `frame_uncertain` auto-§6 check at load; cross-variant reuse (resolver M5) + §5.4
  label propagation. Backfill remaining render-having scans; wire verify/validate into CI.
