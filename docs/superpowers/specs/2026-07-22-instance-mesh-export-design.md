# Instance mesh export — design

## Context

`scripts/build_instance_meshes.py` is a standalone script that loopback-HTTPs
into a running voxa backend (`POST /api/load`), fetches full-resolution
labels, and computes a per-instance convex-hull `.glb` for each confirmed
asset — used to give `product/demo`'s collision detection real per-instance
collision geometry. It duplicates logic (confirmed-instance filtering, hull
computation) that already exists inside the backend's export pipeline, and
requires a separately-running script instead of being a normal export option.

This moves that capability into the backend itself, as an option on the
existing `/api/labels/export` wizard, and retires the script.

## Scope

- Add `include_meshes` to the export-labels flow: when set, the export zip
  gains a `meshes/<instance_id>.glb` per surviving instance, plus a `meshes`
  summary block in `manifest.json`.
- Mesh generation respects the same `confirmed_only` / `include_classes`
  filters as the point export, so meshes and points in one export are always
  for the same instance set.
- Instances with `<100` points, or a `QhullError` from coplanar points, are
  skipped, not bbox-faked. The old script's `MIN_POINTS_FOR_HULL = 4` was
  only the geometric floor (a hull needs ≥4 non-coplanar points to exist at
  all); `100` is a quality bar on top of that — a technically-valid hull from
  a handful of points is too noisy/unrepresentative to hand to collision
  detection, so `MIN_POINTS_FOR_MESH = 100` replaces it as the skip
  threshold. This is a distinct convention from `labeling/segment_hulls.py`'s
  bbox fallback, which serves the frontend's hull-preview overlay (a
  different consumer that needs every segment to render something,
  regardless of quality).
- Delete `scripts/build_instance_meshes.py` (currently untracked — never
  committed) and its mention in `scripts/README.md`.
- Out of scope: no changes to `product/demo`. Its existing pre-baked
  `inventory.json` / `meshes/*.glb` stay as-is; nothing currently
  re-generates them from the new export path.

## Design

### `backend/labeling/instance_meshes.py` (new)

```python
def build_instance_glbs(
    points: np.ndarray,        # (N, 3) float, scan-resolution
    instance_ids: np.ndarray,  # (N,) int, scan-resolution
    surviving_ids: set[int],
) -> tuple[dict[int, bytes], list[tuple[int, str]]]:
    """Convex-hull .glb per id in `surviving_ids`.

    Returns (glbs, skipped) where glbs maps instance_id -> glb bytes, and
    skipped lists (instance_id, reason) for ids with < MIN_POINTS_FOR_MESH
    points or a degenerate/coplanar hull (QhullError).
    """
```

`MIN_POINTS_FOR_MESH = 100` (module constant, up from the old script's
`MIN_POINTS_FOR_HULL = 4`) — a quality bar, not just the geometric minimum:
instances below it are skipped even though a hull may be geometrically
constructible, because too few points make too unreliable a shape for
collision detection to trust.

Ported from the old script's per-id loop: mask `points` by `instance_ids ==
id`, `scipy.spatial.ConvexHull`, `trimesh.Trimesh(vertices=pts,
faces=hull.simplices, process=True).export(file_type="glb")`. Pure function,
no `_state`, no I/O — same shape as the rest of `export_pipeline.py`'s
helpers.

### `export.py` wiring

`ExportLabelsRequest` (schemas.py) gains:

```python
include_meshes: bool = False
```

In `export_labels`, after the existing `ctx`/`confirmed_by_inst`/`src_to_tgt`
setup (already computed for the point export), compute the surviving
instance set once, independent of `resolution.kind`:

```python
mesh_cls, mesh_inst = apply_filters_remap(
    ctx.work_cls, ctx.work_inst, confirmed_by_inst, req, src_to_tgt)
surviving_ids = set(int(i) for i in np.unique(mesh_inst[mesh_cls >= 0]) if i >= 0)
```

This runs on the full scan-resolution arrays already in memory (`ctx.work_cls`
/ `ctx.work_inst`), the same arrays the "scan" resolution point export uses —
no extra fetch, no re-running `/api/load`. `surviving_ids` can be computed
right after `src_to_tgt`/`confirmed_by_inst` exist, but `build_instance_glbs`
itself must be *called* later — see placement below.

**Placement matters.** `zf` (the `zipfile.ZipFile`) only exists inside the
`with zipfile.ZipFile(...) as zf:` block, which opens after `manifest =
build_manifest(...)` (export.py:264) and is immediately followed by `zf.write
(ply_path, "scan_labeled.ply")` then `zf.writestr("manifest.json",
json.dumps(manifest, ...))` (export.py:274-275). For the `meshes` block to
land inside the serialized `manifest.json`, mesh generation must run
*between those two `zf` calls*, mutating `manifest` before it's dumped —
not right after the ctx/src_to_tgt setup as a standalone step:

```python
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
    zf.write(ply_path, "scan_labeled.ply")
    if req.include_meshes:
        glbs, skipped = build_instance_glbs(ctx.scan_pos, ctx.work_inst, surviving_ids)
        for iid, data in glbs.items():
            zf.writestr(f"meshes/{iid}.glb", data)
        manifest["meshes"] = {
            "written": len(glbs),
            "skipped": [{"id": iid, "reason": reason} for iid, reason in skipped],
        }
    zf.writestr("manifest.json", json.dumps(manifest, indent=2))
```

`manifest["meshes"]` is only present when `include_meshes` was requested —
existing exports without it are unaffected (`build_manifest`'s signature
doesn't change; the block is added to the dict after the call, before
serialization).

### Frontend — `export-wizard.jsx`

Step 2 (Classes), next to the existing "Confirmed instances only" checkbox:

```jsx
<label className="ew-check">
  <input type="checkbox" checked={includeMeshes}
    onChange={(e) => setIncludeMeshes(e.target.checked)} />
  <span>Include instance meshes <em>(.glb per instance, for collision detection)</em></span>
</label>
```

New `includeMeshes` state (default `false`), included in `doExport`'s `cfg` as
`include_meshes: includeMeshes`. No changes to the Review step's summary —
mesh counts land in `manifest.json` after unzip, same as the rest of the
export's stats aren't pre-computed live either (e.g. exact remap output isn't
shown before download).

### Cleanup

- Delete `scripts/build_instance_meshes.py` (untracked, so a plain file
  removal — no git history to preserve). It's not listed in
  `scripts/README.md`'s tool tables, so no doc edit is needed there.

## Testing

- `tests/test_instance_meshes.py` (new): `build_instance_glbs` happy path
  (≥100 non-coplanar points → non-empty glb bytes, correct id set) and skip
  cases (`<100` points, and a coplanar/collinear point set at ≥100 points
  triggering `QhullError` → skipped with a reason, absent from `glbs`).
- Extend `tests/test_export_labels.py` at the wiring level only — the happy
  path (a real hull → real glb bytes) is already covered at the unit level
  above, and `conftest.py`'s `build_annotated_root` can't easily produce a
  ≥100-point instance: its `pts` override (used by the unrelated dense-grid
  fixture at line ~171) still hardcodes `inst[:2] = 0` regardless of point
  count, so only 2 points ever carry an instance id no matter how large
  `pts` is. Forcing a 100+-point instance through it would mean reaching
  past its documented purpose. Instead, using the default 8-point/4-instance
  fixture (every instance `<100` points, so all skipped):
  - `include_meshes=False` (default): zip has no `meshes/` entries, no
    `meshes` key in `manifest.json` — exact parity with pre-change exports.
  - `include_meshes=True`: zip still has no `meshes/*.glb` (all instances
    are below the point-count bar), but `manifest["meshes"]` is present with
    `written: 0` and every confirmed/included instance id listed in
    `skipped` with a reason — confirms the endpoint wiring, filter
    respect (`confirmed_only`/`include_classes` change which ids appear),
    and manifest shape, without needing the fixture to produce a real hull.
