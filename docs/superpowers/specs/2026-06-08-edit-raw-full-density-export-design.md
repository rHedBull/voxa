# Edit-mode raw full-density export

**Date:** 2026-06-08
**Status:** Approved — ready for implementation plan

## Problem

Edit mode already offers a "Full density (server)" export checkbox that replays
the active slice's box-op chain against the scene source at native resolution
(`POST /api/edit/export-ply`). For **annotated** scans this does nothing useful:
`load_annotated` loads *all* of `source/scan.ply` into the viewer (no
subsampling), and the exporter re-reads that same `scan.ply`. So "full density"
returns the exact cloud the viewer already shows (~3M points).

The denser source already exists and is already linked: each annotated scan's
`meta.json::source_laz` points at the raw LAZ that `scan.ply` was subsampled
from, and the discovery layer resolves it into
`SceneSource.extras["source_laz_path"]`. The viewer keeps working with the ~3M
proxy; the full-density export should pull the cutout region from that raw LAZ.

## Goal

When "Full density (server)" export runs on an annotated scan with a linked raw
LAZ, write the cutout from the **raw LAZ** (largest available cloud) instead of
`scan.ply`. Disable the option when no denser source is linked, so it never
silently exports the same density as the viewer.

Scope (decided during brainstorming):
- **Annotated scans only** — use the explicit `source_laz` link. Decimated→raw
  name-matching is out of scope.
- **Disable the checkbox** when no raw LAZ is linked (rather than silently
  falling back).
- **Count-estimate fix deferred** — the sidebar estimate still scales against
  `scan.ply`'s total; tracked as a follow-up, not in this change.

## Design

### 1. Backend — exporter pulls from raw LAZ

`backend/routes/export.py::edit_export_ply`, the `source_format == "ply"`
branch. Today it re-reads `scan.ply`. Resolve a raw LAZ first, mirroring the
resolution already used by `/api/load-region` (`load.py:219-232`):

```python
elif src.source_format == "ply":
    raw_laz = src.extras.get("source_laz_path") if src.tier == "annotated" else None
    if raw_laz:
        kept_xyz, kept_rgb = _stream_laz_keep(Path(raw_laz), ops, scene_is_z_up, offset)
    else:
        from scenes.point_cloud import load_ply  # type: ignore
        full_pc, _ = load_ply(src.source_path)
        ...  # unchanged scan.ply path
```

**Why the transforms already line up:** the op-chain is authored in the display
frame (Y-up + recentered). The exporter maps any source-frame points into that
frame with `_to_display_frame(xyz, scene_is_z_up, offset)` before masking, and
`_stream_laz_keep` emits kept points back in the source LAZ frame. `scan.ply` is
a documented subsample of the raw LAZ — same Z-up surveying frame, and the
recenter `offset` is a constant shift applied identically to both — so the box
ops mask the LAZ exactly as they mask what the user sees. `scene_is_z_up`
(`_scene_is_z_up(src)`, the annotated scan's flag) is reused unchanged, since
that is the flag the viewer's display frame was built with. No transform code
changes.

The raw tier's existing `source_format == "laz"` branch is untouched (already
full-density). Legacy / decimated / annotated-without-LAZ keep the current
`scan.ply` re-read as a defensive fallback — the frontend disables the option
for these, but the server stays correct if called directly.

### 2. Backend — surface availability to the frontend

Add to `LoadResponse` (`backend/app/schemas.py`):

```python
raw_source_available: bool = False
```

Set in `backend/routes/load.py` where the response is built:

```python
raw_source_available = (src.tier == "raw") or (
    src.tier == "annotated" and bool(src.extras.get("source_laz_path")))
```

True only when full-density export pulls something denser than the viewer:
- **raw** tier — viewer is stride-sampled, export streams the full LAZ;
- **annotated** with a linked LAZ — viewer is `scan.ply`, export streams raw LAZ.

False for legacy, decimated, and annotated scans without a resolvable
`source_laz` (the export would re-read the same cloud the viewer shows).

### 3. Frontend — disable checkbox when no raw source

- `frontend/src/api.js` — map `rawSourceAvailable: !!j.raw_source_available` in
  `decodeLoadResponse`.
- `frontend/src/mode-edit.jsx`:
  - Pass `cloud.rawSourceAvailable` through `EditMode` → `EditSidePanel`.
  - Disable the "Full density (server)" checkbox when it is false, with a
    tooltip explaining no raw source is linked.
  - Force `exportFull` back to `false` when the active scene has no raw source,
    so a stale check carried over from a previous scene can't trigger a no-op
    server export.

## Testing

- **Backend** — extend `backend/tests/test_export_ply_endpoint.py`:
  - An annotated scene whose `SceneSource` carries a `source_laz_path` exports
    points streamed from the LAZ (more points than `scan.ply`; coordinates in
    the LAZ frame). Reuse existing LAZ fixtures.
  - An annotated scene without `source_laz_path` falls back to `scan.ply`.
  - (If cheap) assert `raw_source_available` is set correctly per tier in the
    load response.
- **Frontend** — existing edit/export pure-function tests stay green. The
  checkbox-disable is thin UI glue; verify in the browser (load an annotated
  scene with a linked LAZ → checkbox enabled and export is denser; load one
  without → checkbox disabled).

## Out of scope / follow-ups

- Sidebar full-density **count estimate** still scales against `scan.ply`'s
  total, so it under-reports for annotated→LAZ exports. Plumbing the raw LAZ
  header point-count through is a separate follow-up.
- Decimated→raw resolution by filename.
