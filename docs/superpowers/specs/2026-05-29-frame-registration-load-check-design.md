# §6 Registration Auto-Check at Load — Design

_Date: 2026-05-29. Implements scan-schema v1.3 §6 enforcement at the `/api/load` boundary._

## 1. Motivation

Scan-schema v1.3 (`docs/superpowers/specs/2026-05-29-scan-schema-v1.3-design.md`)
introduced the **registration health-check** (§6): project a cloud into a render
run's poses and require coverage + photometric agreement, to catch the case where
the labeling cloud and the renders' camera poses live in different frames — the
`navvis_vlx3_water_treatment` incident where a mis-registered cloud produced garbage
SAM3 output.

That check exists today as a **pre-SAM3 gate** and a CLI (`scripts/verify_registration.py`),
but `/api/load` does **not** run it. It only *surfaces* `frame_uncertain` in the
response (`backend/routes/load.py`); nothing verifies that a scan being loaded actually
registers to its renders. Per spec §4.1, `frame_uncertain: true ⇒ the §6 check is
mandatory` — and more generally, a render-having scan whose frame silently regresses
should be caught before a user labels against a broken projection.

This design wires the §6 check into the load path so that **every annotated scan with
renders is verified on load, and a failing check blocks the load**.

## 2. Decisions (resolved in brainstorming)

1. **On failure → hard block.** `/api/load` raises an HTTP error and withholds the
   cloud when the check fails. The reasons are returned so the UI can explain why.
2. **Scope → every annotated scan with renders.** The check runs on load for any
   annotated scan that has a `renders/` directory, not only `frame_uncertain` ones —
   this also catches silent regression of a previously-good (recorded) frame.
3. **Unverifiable ⇒ do not block.** If a scan has no renders, or no render run has
   frame images on disk, the check returns `checked=false` and the load proceeds
   normally. Blocking an unverifiable scan would be a false positive.
4. **Photometric is primary** (reuse the calibrated thresholds already in
   `check_registration`). A correctly-registered but partially-viewed cloud must not
   be blocked on coverage alone.
5. **Cache the verdict by content.** Because the check runs on every render-having
   load, the verdict is memoized in-process keyed by cloud + render fingerprints, so
   repeat loads of an unchanged scene skip the recompute.

## 3. Core model

### 3.1 New reusable function

`backend/preseg/registration.py`:

```python
def verify_scan_registration(
    scan_dir: Path, *,
    max_frames: int = 8,
    orientation: str = "Z+",
    min_coverage: float = 0.35,
    min_photometric: float = 0.5,
    coverage_floor: float = 0.05,
    color_tol: int = 40,
) -> dict:
    """Run the scan-schema v1.3 §6 health-check for a scan against its render runs.

    Loads source/scan.ply (raw coords + colours), applies the orientation preset
    and the recorded v1.3 per-run remap (preseg.resolver.dir_cloud_transforms),
    samples up to `max_frames` poses per render run that has images on disk, and
    scores each run with registration_score + check_registration.

    Returns:
      {
        "checked": bool,          # False when nothing was verifiable (skip => caller must not block)
        "ok": bool,               # True if checked and all checkable runs pass (True when not checked)
        "runs": [
          {"run_id": str, "ok": bool, "coverage": float,
           "photometric": float | None, "n_seen": int, "n_points": int,
           "n_frames": int, "reasons": [str]},
          ...
        ],
        "reasons": [str],         # flattened failure reasons across runs
      }
    """
```

Behaviour, mirroring `scripts/verify_registration.py` exactly so the load gate and
the CLI verify the same thing:

- Load `source/scan.ply` once; compute `xyz_raw` (float64) and `rgb` (uint8 or None).
- `R = euler_xyz_matrix(*ORIENTATION_PRESETS[orientation])`; `xyz = xyz_raw @ R.T`.
- Resolve per-run transforms via `dir_cloud_transforms(runs, frame, variant_id,
  cloud_fingerprint(xyz_raw), R)`. A resolver `ValueError` (cross-scan / incompatible
  variant) is a **hard fail**: return `{checked: True, ok: False, reasons: [str(e)]}`.
- For each run directory under `renders/` with a `manifest.json`:
  - Keep only frames whose image file exists on disk. If none, **skip the run**
    (cannot verify).
  - Sample evenly down to `max_frames` (e.g. `frames[:: max(1, len//max_frames)][:max_frames]`).
  - `xyz_run = xyz if T is None else apply_transform(T, xyz)`.
  - `score = registration_score(xyz_run, frames, fov_y_deg=fov, W, H, rgb, image_loader)`.
  - `ok, reasons = check_registration(score, ...)`.
- `checked = (at least one run was scored)`. `ok = checked implies all scored runs ok`
  (vacuously `True` when `checked=false`).

`fov` comes from the render run's `meta.json` `intrinsics.fov_deg` when present,
else default 60.0 (matches the CLI default and the renderer).

### 3.2 CLI refactor

`scripts/verify_registration.py` is rewritten as a thin wrapper: parse args, call
`verify_scan_registration`, print the per-run table it already prints, and
`return 0 if result["ok"] and result["checked"] else 2` (a scan with nothing to
verify is reported and exits non-zero in the CLI, preserving today's "no runs" → exit 3
semantics via an explicit empty-runs check). The duplicated projection/remap loop is
deleted.

### 3.3 In-process verdict cache

Module-level dict in `registration.py`:

```python
_VERDICT_CACHE: dict[tuple, dict] = {}
```

Key = `(source_fingerprint, tuple(sorted(run_id -> generated_from.source_fingerprint)))`
read from each render run's `meta.json`. On a hit, return the cached verdict without
re-reading the PLY or projecting. The key is **content-derived**, so any change to the
cloud or the render pins invalidates it automatically. Cache lives for the process
lifetime (single-user tool; acceptable). A `verify_scan_registration(..., use_cache=True)`
flag (default `True` for the load path, `False` for the CLI so it always recomputes).

## 4. Load path integration

`backend/routes/load.py`, after `fsum` is built and before constructing `LoadResponse`:

```python
frame_check = None
renders_dir = Path(_scan_dir) / "renders" if _scan_dir else None
if renders_dir and renders_dir.is_dir():
    from preseg.registration import verify_scan_registration
    try:
        v = verify_scan_registration(Path(_scan_dir))
    except Exception:        # never let a check bug break loading a good scan
        v = {"checked": False, "ok": True, "runs": [], "reasons": []}
    if v["checked"] and not v["ok"]:
        raise HTTPException(
            status_code=409,
            detail={
                "error": "frame_registration_failed",
                "message": "Scan does not register to its renders (scan-schema v1.3 §6); "
                           "the cloud and render poses appear to be in different frames.",
                "scan": src.scene_id,
                "frame_check": v,
            },
        )
    if v["checked"]:
        frame_check = v
```

`frame_check` (the passing verdict, or `None` when not checked) is added to
`LoadResponse`. On block, the 409 carries the full verdict in `detail.frame_check`.

### 4.1 Response schema

`backend/app/schemas.py` — add to `LoadResponse`:

```python
frame_check: Optional[dict] = None   # §6 verdict when the scan was verified; None otherwise
```

(Kept as an open dict to avoid over-modelling a diagnostic payload; the UI reads
`ok`, `runs[].coverage`, `runs[].photometric`, `reasons`.)

### 4.2 Defensive guard

A bug inside the check must never break loading a *good* scan: the call is wrapped so
any unexpected exception degrades to `checked=False` (load proceeds). Only an explicit
`checked=True, ok=False` verdict blocks. (The resolver `ValueError` hard-fail is
returned *as* such a verdict by the function, not raised past it.)

## 5. Components & boundaries

| Unit | Responsibility | Depends on |
|------|----------------|------------|
| `verify_scan_registration` | Whole-scan §6 verdict (load PLY, remap, sample, score) | `registration_score`, `check_registration`, `resolver.dir_cloud_transforms`, `scenes.fingerprint`, `scenes.frame`, `scenes.point_cloud.load_ply` |
| `_VERDICT_CACHE` | Skip recompute for unchanged content | fingerprints only |
| `verify_registration.py` (CLI) | Human-facing wrapper + exit codes | `verify_scan_registration` |
| `load_scene` gate | Translate verdict → 200+`frame_check` / 409 | `verify_scan_registration` |

The check function is pure w.r.t. inputs (scan dir on disk) and has no FastAPI/HTTP
knowledge; the HTTP translation lives only in `load.py`.

## 6. Error handling

| Situation | Behaviour |
|-----------|-----------|
| No `renders/` dir | Skip; `checked=false`; load 200 |
| Renders exist, no images on disk | Skip those runs; if none checkable, `checked=false`; load 200 |
| Resolver `ValueError` (cross-scan/variant) | `checked=true, ok=false`; load **409** |
| Coverage/photometric below threshold | run `ok=false` ⇒ scan `ok=false`; load **409** |
| All checkable runs pass | `checked=true, ok=true`; load **200** + `frame_check` |
| Unexpected exception in check | Degrade to `checked=false`; load 200 (never block a good scan on a check bug) |

## 7. Testing

Backend (`backend/tests/`), pytest:

1. **Unit — `test_verify_scan_registration.py`** (synthetic fixture written to a tmp scan dir):
   - A cloud + render run whose poses match → `ok=true, checked=true`.
   - The same cloud with poses offset / a remap that misaligns → `ok=false`.
   - Render run with no image files → `checked=false` (and load must not block).
   - Cache: second call with unchanged content does not re-read the PLY (assert via a
     read counter / monkeypatched loader).
2. **Integration — extend the load tests**:
   - `/api/load` of a matching annotated scan with renders → **200**, `frame_check.ok is true`.
   - `/api/load` of a mismatched one → **409**, `detail.error == "frame_registration_failed"`.
3. **Regression**: a non-annotated tier and a renderless annotated scan still load 200
   with `frame_check is None` (no behaviour change).

Fixtures use tiny clouds (≤ a few hundred points) and 1–2 small generated PNG frames so
the suite stays fast and GPU-free (the check is pure numpy projection).

## 8. Out of scope (YAGNI)

- Persisting the verdict to disk / across restarts (in-memory cache suffices).
- A frontend banner for the 409 / `frame_check` — backend contract only here; the UI
  wiring is follow-up work alongside the multi-run Compare UI (v1.3 item A).
- Re-registering (ICP) automatically on failure — failure blocks and points the user at
  `backfill_scan_frame.py`; auto-repair is a separate concern.
- Threshold re-tuning — reuse the calibrated `check_registration` defaults.

## 9. Files touched

- `backend/preseg/registration.py` — add `verify_scan_registration` + `_VERDICT_CACHE`.
- `scripts/verify_registration.py` — rewrite as a thin wrapper.
- `backend/routes/load.py` — call the check; 409 on fail; attach `frame_check`.
- `backend/app/schemas.py` — `LoadResponse.frame_check: Optional[dict]`.
- `backend/tests/test_verify_scan_registration.py` (new) + extend load tests.
