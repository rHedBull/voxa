// mode-edit.jsx — non-destructive slicing mode.
//
// Each slice is a subset of the previous one, defined by an oriented
// selection box (reuses the Viewer's cuboid + TransformControls gizmo).
// "Original" sits at the top of the chain and is immutable.

import { useState, useMemo, useCallback, useEffect } from 'react';
import { Viewer } from './viewer.jsx';
import { NavModeToggle, HUDChip, CameraPresets, ViewportToolbar, ToolButton } from './viewport-atoms.jsx';

const SEL_BOX_ID = '__edit_sel_box__';
const SEL_COLOR = '#ffd24a';

// Test which active-slice points fall inside an oriented box.
// Box-local frame: subtract center, apply inverse rotation (Euler XYZ),
// check |x| < sx/2 etc. `inside=false` flips the predicate (returns
// points strictly outside the box) — used for the "delete" op.
function pointsInsideOBB(positions, indices, box, inside = true) {
  const [cx, cy, cz] = box.center;
  const [sx, sy, sz] = box.size;
  const [rx, ry, rz] = box.rotation;
  const hx = sx / 2, hy = sy / 2, hz = sz / 2;

  // Build inverse rotation matrix (Rz * Ry * Rx, applied as transpose of XYZ).
  const cxR = Math.cos(rx), sxR = Math.sin(rx);
  const cyR = Math.cos(ry), syR = Math.sin(ry);
  const czR = Math.cos(rz), szR = Math.sin(rz);
  // Forward R = Rz * Ry * Rx (Three.js Euler XYZ default).
  const m00 = cyR * czR;
  const m01 = sxR * syR * czR - cxR * szR;
  const m02 = cxR * syR * czR + sxR * szR;
  const m10 = cyR * szR;
  const m11 = sxR * syR * szR + cxR * czR;
  const m12 = cxR * syR * szR - sxR * czR;
  const m20 = -syR;
  const m21 = sxR * cyR;
  const m22 = cxR * cyR;
  // Inverse = transpose for orthonormal R.

  const out = [];
  const pool = indices || null;
  const N = pool ? pool.length : positions.length / 3;
  for (let k = 0; k < N; k++) {
    const i = pool ? pool[k] : k;
    const px = positions[3 * i] - cx;
    const py = positions[3 * i + 1] - cy;
    const pz = positions[3 * i + 2] - cz;
    // local = R^T * p
    const lx = m00 * px + m10 * py + m20 * pz;
    const ly = m01 * px + m11 * py + m21 * pz;
    const lz = m02 * px + m12 * py + m22 * pz;
    const hit = lx >= -hx && lx <= hx && ly >= -hy && ly <= hy && lz >= -hz && lz <= hz;
    if (hit === inside) out.push(i);
  }
  return Uint32Array.from(out);
}

// Compute axis-aligned bbox of a subset of an original cloud. Returns
// { min: [x,y,z], max: [x,y,z], center: [...], size: [...] } in scene units.
function bboxOfSubset(positions, indices) {
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  const pool = indices || null;
  const N = pool ? pool.length : positions.length / 3;
  for (let k = 0; k < N; k++) {
    const i = pool ? pool[k] : k;
    const x = positions[3 * i], y = positions[3 * i + 1], z = positions[3 * i + 2];
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  if (!Number.isFinite(minX)) {
    return { min: [0,0,0], max: [0,0,0], center: [0,0,0], size: [1,1,1] };
  }
  return {
    min: [minX, minY, minZ],
    max: [maxX, maxY, maxZ],
    center: [(minX+maxX)/2, (minY+maxY)/2, (minZ+maxZ)/2],
    size: [Math.max(maxX-minX, 0.01), Math.max(maxY-minY, 0.01), Math.max(maxZ-minZ, 0.01)],
  };
}

// Build a derived cloud whose positions/colors arrays are restricted to
// the slice's indices. Preserves bbox-shape and metadata fields the Viewer
// actually consumes.
function deriveCloud(original, indices) {
  if (!original) return null;
  if (!indices) return original;
  const N = indices.length;
  const pos = new Float32Array(3 * N);
  const cols = original.colors ? new Float32Array(3 * N) : null;
  for (let k = 0; k < N; k++) {
    const i = indices[k];
    pos[3*k]   = original.positions[3*i];
    pos[3*k+1] = original.positions[3*i+1];
    pos[3*k+2] = original.positions[3*i+2];
    if (cols) {
      cols[3*k]   = original.colors[3*i];
      cols[3*k+1] = original.colors[3*i+1];
      cols[3*k+2] = original.colors[3*i+2];
    }
  }
  const bb = bboxOfSubset(original.positions, indices);
  return {
    ...original,
    positions: pos,
    colors: cols,
    bbox: bb,
    numSubsampled: N,
    numPoints: N,
    classIds: null,
    instanceIds: null,
    subsampleIdx: null,
    meshUrl: null,
  };
}

let _sliceSeq = 0;
const nextSliceId = () => `slice_${++_sliceSeq}_${Date.now().toString(36)}`;

// Build a binary little-endian PLY blob from a slice. If `indices` is null
// the whole cloud is exported. Colors are written as RGB uint8 (0-255).
function buildPlyBlob(cloud, indices) {
  const N = indices ? indices.length : cloud.positions.length / 3;
  const hasColor = !!cloud.colors;
  // Undo the load-time z-up→y-up rotation + recenter so the saved PLY
  // is in the source-file frame, not the Three.js display frame.
  // Voxa backend applies (x, y, z) → (x, z, -y) on load when scene_is_z_up.
  // Inverse: (x, y, z) → (x, -z, y), then add back the recenter offset.
  const zUp = !!cloud.sceneIsZUp;
  const off0 = cloud.recenterOffset || [0, 0, 0];
  const ox = +off0[0] || 0, oy = +off0[1] || 0, oz = +off0[2] || 0;
  const header =
    'ply\n' +
    'format binary_little_endian 1.0\n' +
    `element vertex ${N}\n` +
    'property float x\n' +
    'property float y\n' +
    'property float z\n' +
    (hasColor
      ? 'property uchar red\nproperty uchar green\nproperty uchar blue\n'
      : '') +
    'end_header\n';
  const stride = hasColor ? 15 : 12;
  const buf = new ArrayBuffer(N * stride);
  const dv = new DataView(buf);
  for (let k = 0; k < N; k++) {
    const i = indices ? indices[k] : k;
    const off = k * stride;
    const px = cloud.positions[3*i];
    const py = cloud.positions[3*i+1];
    const pz = cloud.positions[3*i+2];
    let outX, outY, outZ;
    if (zUp) {
      outX =  px;
      outY = -pz;
      outZ =  py;
    } else {
      outX = px; outY = py; outZ = pz;
    }
    dv.setFloat32(off,     outX + ox, true);
    dv.setFloat32(off + 4, outY + oy, true);
    dv.setFloat32(off + 8, outZ + oz, true);
    if (hasColor) {
      const r = Math.max(0, Math.min(255, Math.round(cloud.colors[3*i]   * 255)));
      const g = Math.max(0, Math.min(255, Math.round(cloud.colors[3*i+1] * 255)));
      const b = Math.max(0, Math.min(255, Math.round(cloud.colors[3*i+2] * 255)));
      dv.setUint8(off + 12, r);
      dv.setUint8(off + 13, g);
      dv.setUint8(off + 14, b);
    }
  }
  return new Blob([header, buf], { type: 'application/octet-stream' });
}

// If the browser supports the File System Access API, open a native Save
// dialog so the user picks both location AND filename. Otherwise fall back
// to the standard `<a download>` flow (saves to the default download dir).
async function saveBlob(blob, suggestedName) {
  if (typeof window.showSaveFilePicker === 'function') {
    try {
      const handle = await window.showSaveFilePicker({
        suggestedName,
        types: [{
          description: 'Polygon File Format',
          accept: { 'application/octet-stream': ['.ply'] },
        }],
      });
      const writable = await handle.createWritable();
      await writable.write(blob);
      await writable.close();
      return true;
    } catch (err) {
      // User cancelled (AbortError) — silent. Anything else, fall through.
      if (err && err.name === 'AbortError') return false;
      console.warn('showSaveFilePicker failed, falling back:', err);
    }
  }
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = suggestedName;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  return true;
}

function safeFilename(s) {
  return (s || 'slice').replace(/[^a-z0-9_\-]+/gi, '_').slice(0, 64) || 'slice';
}

export function EditMode({ cloud, theme, viewerRef, navMode, onNavModeChange, onCameraChange, sceneName }) {
  // Slice chain: [{ id, name, parentId, ops }].
  // Each op is { op: 'keep'|'delete', center, size, rotation }, applied
  // in order against the parent's index pool. Original has parentId=null
  // and ops=null (means "all points"). Indices are derived (memo below)
  // so that editing an ancestor automatically reflows descendants.
  const [slices, setSlices] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [transformMode, setTransformMode] = useState('translate');
  const [selBox, setSelBox] = useState(null);

  // Reset chain when scene changes.
  useEffect(() => {
    if (!cloud) {
      setSlices([]);
      setActiveId(null);
      setSelBox(null);
      return;
    }
    _sliceSeq = 0;
    const original = { id: 'original', name: 'Original', parentId: null, ops: null };
    setSlices([original]);
    setActiveId('original');
  }, [cloud]);

  // Derive each slice's index pool from its ancestors' ops. Slices are
  // stored in insertion order, which is topological (children always come
  // after their parent), so a single forward pass suffices.
  const indicesById = useMemo(() => {
    const map = new Map();
    if (!cloud) return map;
    for (const s of slices) {
      if (!s.parentId) { map.set(s.id, null); continue; }
      let cur = map.get(s.parentId);
      if (cur === undefined) cur = null;
      for (const op of (s.ops || [])) {
        cur = pointsInsideOBB(cloud.positions, cur, op, op.op !== 'delete');
      }
      map.set(s.id, cur);
    }
    return map;
  }, [cloud, slices]);

  const activeSlice = useMemo(
    () => slices.find((s) => s.id === activeId) || null,
    [slices, activeId]
  );
  const activeIndices = activeSlice ? (indicesById.get(activeSlice.id) ?? null) : null;

  // The viewer cloud is the active slice's points only.
  const viewerCloud = useMemo(
    () => activeSlice ? deriveCloud(cloud, activeIndices) : null,
    [cloud, activeSlice, activeIndices]
  );

  // Drop the selection box whenever the active slice changes. The user
  // re-arms it via the Select tool below.
  useEffect(() => { setSelBox(null); }, [activeId]);

  const disarmSelectionTool = useCallback(() => setSelBox(null), []);
  const toggleSelectionTool = useCallback(() => {
    setSelBox((b) => {
      if (b) return null;
      if (!viewerCloud) return null;
      const bb = viewerCloud.bbox;
      return {
        id: SEL_BOX_ID,
        label: 'selection',
        cls: 0,
        color: SEL_COLOR,
        center: [(bb.min[0]+bb.max[0])/2, (bb.min[1]+bb.max[1])/2, (bb.min[2]+bb.max[2])/2],
        size: [
          Math.max(bb.max[0]-bb.min[0], 0.05),
          Math.max(bb.max[1]-bb.min[1], 0.05),
          Math.max(bb.max[2]-bb.min[2], 0.05),
        ],
        rotation: [0, 0, 0],
      };
    });
  }, [viewerCloud]);

  // Esc disarms the selection tool. Skip when typing into an input.
  useEffect(() => {
    const onKey = (e) => {
      if (e.key !== 'Escape') return;
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      if (selBox) {
        e.preventDefault();
        disarmSelectionTool();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selBox, disarmSelectionTool]);

  const onCuboidTransform = useCallback((id, patch) => {
    if (id !== SEL_BOX_ID) return;
    setSelBox((b) => b ? { ...b, ...patch } : b);
  }, []);

  const opFromSelBox = (kind) => ({
    op: kind,
    center: [...selBox.center],
    size: [...selBox.size],
    rotation: [...selBox.rotation],
  });

  const saveSliceFromSelection = useCallback(() => {
    if (!cloud || !activeSlice || !selBox) return;
    const survivors = pointsInsideOBB(cloud.positions, activeIndices, selBox);
    if (survivors.length === 0) {
      window.alert('Selection is empty — nothing to slice.');
      return;
    }
    const id = nextSliceId();
    const name = `Slice ${slices.length}`;
    const newSlice = {
      id, name,
      parentId: activeSlice.id,
      // Op chain is replayed root → active by the server export so
      // full-density survivors match what the user sees here.
      ops: [opFromSelBox('keep')],
    };
    setSlices((arr) => [...arr, newSlice]);
    setActiveId(id);
    setSelBox(null);
  }, [cloud, activeSlice, activeIndices, selBox, slices.length]);

  // Box-delete: strip the selection from the *active* slice in place
  // (descendants reflow automatically via indicesById). Original is
  // immutable, so we refuse there — pick or create a slice first.
  const deleteInsideBox = useCallback(() => {
    if (!cloud || !activeSlice || !selBox) return;
    if (activeSlice.id === 'original') {
      window.alert('Original is read-only — pick or create a slice first.');
      return;
    }
    const survivors = pointsInsideOBB(cloud.positions, activeIndices, selBox, false);
    if (activeIndices && survivors.length === activeIndices.length) {
      window.alert('Selection is empty — nothing to delete.');
      return;
    }
    if (survivors.length === 0) {
      const ok = window.confirm('This will remove every point from the active slice. Continue?');
      if (!ok) return;
    }
    const newOp = opFromSelBox('delete');
    setSlices((arr) => arr.map((s) =>
      s.id === activeSlice.id
        ? { ...s, ops: [...(s.ops || []), newOp] }
        : s
    ));
    setSelBox(null);
  }, [cloud, activeSlice, activeIndices, selBox]);

  // Delete a slice + every descendant. Original cannot be deleted.
  const deleteSlice = useCallback((id) => {
    if (id === 'original') return;
    setSlices((arr) => {
      const doomed = new Set([id]);
      // Sweep forward: any slice whose parent is doomed is also doomed.
      for (const s of arr) {
        if (s.parentId && doomed.has(s.parentId)) doomed.add(s.id);
      }
      const next = arr.filter((s) => !doomed.has(s.id));
      return next;
    });
    setActiveId((cur) => {
      if (cur && cur !== id) return cur;
      // Activated one was deleted — fall back to its parent.
      const sl = slices.find((s) => s.id === id);
      return sl?.parentId || 'original';
    });
  }, [slices]);

  const [exportFull, setExportFull] = useState(false);
  const [exportBusy, setExportBusy] = useState(false);

  const exportActiveSlice = useCallback(async () => {
    if (!cloud || !activeSlice) return;
    const fname = `${safeFilename(activeSlice.name)}.ply`;

    if (!exportFull) {
      // Subsampled (in-memory cloud) — same as before.
      const blob = buildPlyBlob(cloud, activeIndices);
      await saveBlob(blob, fname);
      return;
    }

    if (!sceneName) {
      window.alert('Cannot export: no active scene id.');
      return;
    }
    // Walk root → active and concatenate every op along the way. Each
    // op was authored relative to the intermediate slice that came
    // before it, so order matters.
    const byId = new Map(slices.map((s) => [s.id, s]));
    const stack = [];
    let cur = activeSlice;
    while (cur) {
      if (cur.ops && cur.ops.length) stack.push(cur.ops);
      cur = cur.parentId ? byId.get(cur.parentId) : null;
    }
    stack.reverse();
    const ops = stack.flat();

    setExportBusy(true);
    try {
      const r = await fetch('/api/edit/export-ply', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ scene: sceneName, ops }),
      });
      if (!r.ok) {
        const t = await r.text().catch(() => '');
        throw new Error(`server ${r.status}: ${t || r.statusText}`);
      }
      const blob = await r.blob();
      await saveBlob(blob, fname);
    } catch (err) {
      console.error('full-density export failed:', err);
      window.alert(`Export failed: ${err.message || err}`);
    } finally {
      setExportBusy(false);
    }
  }, [cloud, activeSlice, activeIndices, slices, sceneName, exportFull]);

  const renameSlice = useCallback((id, name) => {
    if (id === 'original') return;
    setSlices((arr) => arr.map((s) => s.id === id ? { ...s, name } : s));
  }, []);

  const instances = selBox ? [selBox] : [];
  const highlight = selBox
    ? { center: selBox.center, size: selBox.size, rotation: selBox.rotation, color: SEL_COLOR }
    : null;

  return (
    <div className="mode-root edit">
      <div className="vp-stack">
        <Viewer
          ref={viewerRef}
          cloud={viewerCloud}
          instances={instances}
          showCuboids={true}
          cuboidStyle="wire"
          cuboidOpacity={0.9}
          selectedId={selBox?.id || null}
          transformMode={transformMode}
          onCuboidTransform={onCuboidTransform}
          highlightCuboid={highlight}
          pointSize={0.012}
          showFloor={true}
          background={theme.bg}
          floorColor={theme.floor}
          colorMode="rgb"
          navMode={navMode}
          onCameraChange={onCameraChange}
        />

        <div className="vp-hud-top">
          <div className="hud-group">
            <HUDChip label="Active" value={activeSlice?.name || '—'} />
            <HUDChip
              label="Points"
              value={viewerCloud ? viewerCloud.numSubsampled.toLocaleString() : '—'}
              mono
            />
          </div>
          <div className="hud-group">
            <NavModeToggle navMode={navMode} onChange={onNavModeChange} />
            <CameraPresets onPreset={(p) => viewerRef.current?.preset(p)} />
          </div>
        </div>

        <ViewportToolbar side="left">
          <ToolButton mini icon="↺" label="Reset" hotkey="R"
            onClick={() => viewerRef.current?.preset('iso')} />
        </ViewportToolbar>

        <EditSidePanel
          slices={slices}
          indicesById={indicesById}
          activeId={activeId}
          activeIsOriginal={activeSlice?.id === 'original'}
          onPick={setActiveId}
          onDelete={deleteSlice}
          onRename={renameSlice}
          boxArmed={!!selBox}
          onToggleBox={toggleSelectionTool}
          transformMode={transformMode}
          onTransformMode={setTransformMode}
          onSaveSlice={saveSliceFromSelection}
          onDeleteInBox={deleteInsideBox}
          canSave={!!selBox && !!activeSlice}
          onExport={exportActiveSlice}
          canExport={!!activeSlice && !!cloud && !exportBusy}
          activeName={activeSlice?.name || ''}
          exportFull={exportFull}
          onExportFullChange={setExportFull}
          exportBusy={exportBusy}
          exportCount={activeIndices ? activeIndices.length : 0}
          subsampledTotal={cloud?.numSubsampled ?? 0}
          fullTotal={cloud?.numPointsTotal ?? cloud?.numPoints ?? 0}
        />
      </div>
    </div>
  );
}

function _fmtCount(n) {
  if (n == null || !isFinite(n)) return '—';
  if (n >= 1e7) return `${(n / 1e6).toFixed(0)}M`;
  if (n >= 1e6) return `${(n / 1e6).toFixed(1)}M`;
  if (n >= 1e4) return `${(n / 1e3).toFixed(0)}k`;
  if (n >= 1e3) return `${(n / 1e3).toFixed(1)}k`;
  return String(n);
}

function EditSidePanel({
  slices, indicesById, activeId, activeIsOriginal,
  onPick, onDelete, onRename,
  boxArmed, onToggleBox,
  transformMode, onTransformMode, onSaveSlice, onDeleteInBox, canSave,
  onExport, canExport, activeName,
  exportFull, onExportFullChange, exportBusy,
  exportCount, subsampledTotal, fullTotal,
}) {
  // Subsampled export = exact count (in-memory mask).
  // Full-density export = estimate: scale by source/subsample ratio.
  const subsampleCount = exportCount || 0;
  const fullCountEstimate = (exportCount && subsampledTotal)
    ? Math.round(exportCount * (fullTotal / subsampledTotal))
    : 0;
  const isEstimate = fullTotal > subsampledTotal;
  return (
    <>
      {/* Left: slice list */}
      <div className="inspect-right" style={{ left: 12, right: 'auto' }}>
        <div className="panel">
          <div className="panel-hd">
            <span>Slices</span>
            <span className="badge-soft">{slices.length}</span>
          </div>
          <div className="panel-body">
            {slices.map((s) => (
              <SliceRow key={s.id}
                slice={s}
                count={indicesById?.get(s.id)?.length ?? null}
                active={s.id === activeId}
                onPick={() => onPick(s.id)}
                onDelete={() => onDelete(s.id)}
                onRename={(name) => onRename(s.id, name)} />
            ))}
          </div>
        </div>
      </div>

      {/* Right: tools */}
      <div className="inspect-right" style={{ width: 'auto' }}>
        <div className="panel">
          <div className="panel-hd" style={{ textAlign: 'center' }}>Tool</div>
          <div className="panel-body" style={{ alignItems: 'center' }}>
            <button
              className={'header-btn' + (boxArmed ? ' primary' : '')}
              onClick={onToggleBox}
              title={boxArmed ? 'Cancel (Esc)' : 'Draw a selection box'}>
              {boxArmed ? '◫ Cancel box (Esc)' : '◫ Box select'}
            </button>
            {boxArmed && (
              <div className="ctrl" style={{ marginTop: 8 }}>
                <label>Gizmo mode</label>
                <div className="pill-group">
                  {[
                    ['translate', 'Move'],
                    ['rotate', 'Rotate'],
                    ['scale', 'Resize'],
                  ].map(([k, l]) => (
                    <button key={k}
                      className={'pill' + (transformMode === k ? ' active' : '')}
                      onClick={() => onTransformMode(k)}>{l}</button>
                  ))}
                </div>
              </div>
            )}
            <button className="header-btn primary"
              disabled={!canSave}
              onClick={onSaveSlice}
              title="Create a new child slice from points inside the box"
              style={{ marginTop: 8 }}>
              Save as new slice
            </button>
            <button className="header-btn"
              disabled={!canSave || activeIsOriginal}
              onClick={onDeleteInBox}
              title={activeIsOriginal
                ? 'Original is read-only — pick or create a slice first'
                : 'Remove points inside the box from the active slice (descendants reflow automatically)'}
              style={{ marginTop: 4 }}>
              ✕ Delete in box
            </button>
          </div>
        </div>

        <div className="panel">
          <div className="panel-hd" style={{ textAlign: 'center' }}>Export</div>
          <div className="panel-body" style={{ alignItems: 'center', gap: 6 }}>
            <label style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 12, cursor: 'pointer' }}>
              <input type="checkbox"
                checked={!!exportFull}
                onChange={(e) => onExportFullChange(e.target.checked)} />
              Full density (server)
            </label>
            <button className="header-btn"
              disabled={!canExport}
              onClick={onExport}
              title={canExport
                ? (exportFull
                    ? `Replay slice chain on full-density cloud and download "${activeName}.ply"`
                    : `Download "${activeName}" as PLY (subsampled view)`)
                : 'Pick a slice first'}>
              {exportBusy ? '… exporting' : '⤓ Export active as PLY'}
            </button>
            {canExport && (
              <div style={{
                  fontSize: 11, opacity: 0.85, marginTop: 2,
                  fontVariantNumeric: 'tabular-nums', textAlign: 'center',
                  lineHeight: 1.4,
                }}>
                <div title={`${subsampleCount.toLocaleString()} pts (exact: in-memory mask)`}>
                  subsampled <b>{_fmtCount(subsampleCount)}</b>
                  <span style={{ opacity: 0.55 }}>{` / ${_fmtCount(subsampledTotal)} loaded`}</span>
                </div>
                <div title={`≈ ${fullCountEstimate.toLocaleString()} pts (estimate: ${subsampleCount.toLocaleString()}/${subsampledTotal.toLocaleString()} × ${fullTotal.toLocaleString()})`}>
                  full density <b>{isEstimate ? '~' : ''}{_fmtCount(fullCountEstimate)}</b>
                  <span style={{ opacity: 0.55 }}>{` / ${_fmtCount(fullTotal)} source`}</span>
                </div>
                <div style={{ opacity: 0.6, marginTop: 2 }}>
                  ⤓ will write{' '}
                  <b>{isEstimate && exportFull ? '~' : ''}{_fmtCount(exportFull ? fullCountEstimate : subsampleCount)}</b>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </>
  );
}

function SliceRow({ slice, count, active, onPick, onDelete, onRename }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(slice.name);
  const [confirmingDelete, setConfirmingDelete] = useState(false);
  useEffect(() => { setDraft(slice.name); }, [slice.name]);

  const isOriginal = slice.id === 'original';

  const iconBtnStyle = {
    width: 22, height: 22, padding: 0,
    display: 'inline-flex', alignItems: 'center', justifyContent: 'center',
    fontSize: 12, lineHeight: 1, marginLeft: 4,
    background: 'transparent', border: '1px solid transparent',
    color: 'inherit', opacity: 0.6, cursor: 'pointer',
    borderRadius: 4,
  };

  return (
    <div className={'inst-row' + (active ? ' selected' : '')}
         onClick={onPick}
         style={{ alignItems: 'center' }}>
      <span className="inst-dot" style={{ background: isOriginal ? '#10b981' : '#5b8def' }} />
      <div className="inst-text" style={{ flex: 1, minWidth: 0 }}>
        {editing && !isOriginal ? (
          <input
            autoFocus
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onBlur={() => { onRename(draft.trim() || slice.name); setEditing(false); }}
            onKeyDown={(e) => {
              if (e.key === 'Enter') { onRename(draft.trim() || slice.name); setEditing(false); }
              if (e.key === 'Escape') { setDraft(slice.name); setEditing(false); }
            }}
            onClick={(e) => e.stopPropagation()}
            style={{ width: '100%' }} />
        ) : (
          <b onDoubleClick={(e) => { if (!isOriginal) { e.stopPropagation(); setEditing(true); } }}>
            {slice.name}
          </b>
        )}
        <em>{count != null ? `${count.toLocaleString()} pts` : 'all pts'}</em>
      </div>
      {!isOriginal && !editing && !confirmingDelete && (
        <>
          <button
            title="Rename"
            onClick={(e) => { e.stopPropagation(); setEditing(true); }}
            style={iconBtnStyle}
            onMouseEnter={(e) => { e.currentTarget.style.opacity = 1; e.currentTarget.style.background = 'rgba(255,255,255,0.06)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.opacity = 0.6; e.currentTarget.style.background = 'transparent'; }}>
            ✎
          </button>
          <button
            title="Delete slice"
            onClick={(e) => { e.stopPropagation(); setConfirmingDelete(true); }}
            style={iconBtnStyle}
            onMouseEnter={(e) => { e.currentTarget.style.opacity = 1; e.currentTarget.style.color = 'oklch(0.7 0.18 25)'; e.currentTarget.style.background = 'rgba(255,80,80,0.10)'; }}
            onMouseLeave={(e) => { e.currentTarget.style.opacity = 0.6; e.currentTarget.style.color = 'inherit'; e.currentTarget.style.background = 'transparent'; }}>
            🗑
          </button>
        </>
      )}
      {confirmingDelete && (
        <div onClick={(e) => e.stopPropagation()}
             style={{
               display: 'inline-flex', gap: 4, alignItems: 'center', marginLeft: 6,
               padding: '2px 6px', borderRadius: 4,
               background: 'rgba(255,80,80,0.12)',
               border: '1px solid rgba(255,80,80,0.35)',
               fontSize: 11,
             }}>
          <span style={{ opacity: 0.85 }}>Delete?</span>
          <button
            onClick={(e) => { e.stopPropagation(); setConfirmingDelete(false); onDelete(); }}
            title="Confirm delete (also removes descendants)"
            style={{
              padding: '2px 6px', fontSize: 11, cursor: 'pointer',
              background: 'oklch(0.55 0.18 25)', color: 'white',
              border: 'none', borderRadius: 3,
            }}>OK</button>
          <button
            onClick={(e) => { e.stopPropagation(); setConfirmingDelete(false); }}
            title="Cancel"
            style={{
              padding: '2px 6px', fontSize: 11, cursor: 'pointer',
              background: 'transparent', color: 'inherit',
              border: '1px solid rgba(255,255,255,0.2)', borderRadius: 3,
            }}>Cancel</button>
        </div>
      )}
    </div>
  );
}
