# SAM sidecar

A standalone GPU service that powers voxa's **SAM** labeling tool. It renders the
raw point cloud from a camera pose, runs SAM (Segment Anything v3) on that render,
and back-projects the resulting mask(s) onto the scan-resolution cloud as point
indices. voxa's backend proxies to it; it holds the heavy state (the 188M-class raw
cloud + the SAM model) so voxa's own `.venv` stays lean.

It runs in the **anaconda environment** that has `torch` + `sam3` (CUDA), *not*
voxa's `.venv`. SAM's model load is slow, so it's a long-lived process (load once,
serve many requests).

## Install & run

```bash
# one-time: add the web deps into the anaconda env
/home/hendrik/anaconda3/bin/python -m pip install -r sam_sidecar/requirements.txt

# run (port 8011)
cd sam_sidecar && bash run.sh
curl 127.0.0.1:8011/health         # {"ok": true, "scan_id": null}
```

Then point voxa at it: `VOXA_SAM_SIDECAR_URL=http://127.0.0.1:8011 npm run dev`.
Without that env var, voxa disables the SAM tool (`/api/sam/*` → 503).

## Endpoints

The sidecar is driven entirely by voxa's proxy (`backend/routes/sam.py`); you
normally don't call it directly. Contract:

- `POST /capture` — `{scan_id, source_fingerprint, raw_laz_path, scan_ply_path,
  camera:{pos,target,fov,W,H}, mode:"box"|"concept", box?, text?}` → renders the
  raw cloud from the (native-frame) camera, runs SAM (box → best mask; concept →
  all instances of the text prompt), stashes the render + masks under a fresh
  `capture_id`, and returns `{capture_id, overlay_png_b64, masks:[{mask_id, score}]}`.
  **No projection, no state mutation.** Only one live `capture_id` at a time.
- `POST /project` — `{scan_id, source_fingerprint, capture_id, mask_ids:[...]}` →
  for each chosen mask, projects `scan.ply` through the stored camera and keeps the
  points inside the mask that pass the raw depth-buffer occlusion test; returns
  `{instances:[{mask_id, scan_indices_b64}]}` (b64 int32 indices into scan.ply order).
- `GET /health`.

**voxa passes the cloud paths** (`raw_laz_path`, `scan_ply_path`) — the sidecar does
not resolve them itself (voxa already resolves them for export). The scan is loaded
lazily and cached in memory keyed by `source_fingerprint`.

## Scan-identity guard (correctness-critical)

The sidecar is a separate long-lived process; voxa's active scan can change under it.
Every request carries `{scan_id, source_fingerprint}`. Rules (all fail-loud, see
`scan_store.py`):

- Different `scan_id` → reload that scan (drop the previous one).
- Same `scan_id` but a different `source_fingerprint` → **409** `{diverged:"source"}`,
  surfaced by voxa as a blocking banner. Never serve a mismatched cloud.
- A stale `capture_id` at `/project` → **409** (`/capture` replaced it).

This is what stops `/project` from ever returning indices for the wrong cloud (which
`apply_reassign` would silently write onto the wrong points).

## Layout

| file | role |
|------|------|
| `main.py` | FastAPI app: `/capture`, `/project`, `/health` |
| `render.py` | perspective point-splat render → RGB image + depth-buffer (pure numpy) |
| `backproject.py` | mask + depth-buffer → visible scan-res point indices |
| `reproject.py` | projection math (copied verbatim from `backend/scenes/reproject.py`) |
| `cloud.py` | raw-LAZ loader (cached `.rawcache.npz`) + `scan.ply` loader |
| `scan_store.py` | one loaded scan at a time, fingerprint-guarded |
| `sam_infer.py` | SAM3 box + concept wrappers (lazy CUDA import) |
| `tests/` | pytest — projection, render, back-projection, store, endpoints (all CUDA-free via monkeypatched SAM) |

## Tests

```bash
cd sam_sidecar && /home/hendrik/anaconda3/bin/python -m pytest tests/ -v
```
The suite is CUDA-free (SAM is monkeypatched); the real-weights path is exercised by
the end-to-end browser test against a live sidecar.

## Design

Single-view v1 (box + concept modes; no multi-view instance merging yet). See the
design + plan under `docs/superpowers/`:
`specs/2026-07-12-sam-labeling-tool-design.md`, `plans/2026-07-12-sam-labeling-tool.md`.
