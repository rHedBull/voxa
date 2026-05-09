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
// check |x| < sx/2 etc.
function pointsInsideOBB(positions, indices, box) {
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
    if (lx >= -hx && lx <= hx && ly >= -hy && ly <= hy && lz >= -hz && lz <= hz) {
      out.push(i);
    }
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

export function EditMode({ cloud, theme, viewerRef, navMode, onNavModeChange, onCameraChange }) {
  // Slice chain: [{ id, name, parentId, indices: Uint32Array | null }].
  // Original is the head with indices=null (means all points).
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
    const original = { id: 'original', name: 'Original', parentId: null, indices: null };
    setSlices([original]);
    setActiveId('original');
  }, [cloud]);

  const activeSlice = useMemo(
    () => slices.find((s) => s.id === activeId) || null,
    [slices, activeId]
  );

  // The viewer cloud is the active slice's points only.
  const viewerCloud = useMemo(
    () => activeSlice ? deriveCloud(cloud, activeSlice.indices) : null,
    [cloud, activeSlice]
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

  const saveSliceFromSelection = useCallback(() => {
    if (!cloud || !activeSlice || !selBox) return;
    const survivors = pointsInsideOBB(cloud.positions, activeSlice.indices, selBox);
    if (survivors.length === 0) {
      window.alert('Selection is empty — nothing to slice.');
      return;
    }
    const id = nextSliceId();
    const name = `Slice ${slices.length}`;
    const newSlice = { id, name, parentId: activeSlice.id, indices: survivors };
    setSlices((arr) => [...arr, newSlice]);
    setActiveId(id);
  }, [cloud, activeSlice, selBox, slices.length]);

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
          activeId={activeId}
          onPick={setActiveId}
          onDelete={deleteSlice}
          onRename={renameSlice}
          boxArmed={!!selBox}
          onToggleBox={toggleSelectionTool}
          transformMode={transformMode}
          onTransformMode={setTransformMode}
          onSaveSlice={saveSliceFromSelection}
          canSave={!!selBox && !!activeSlice}
        />
      </div>
    </div>
  );
}

function EditSidePanel({
  slices, activeId, onPick, onDelete, onRename,
  boxArmed, onToggleBox,
  transformMode, onTransformMode, onSaveSlice, canSave,
}) {
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
                active={s.id === activeId}
                onPick={() => onPick(s.id)}
                onDelete={() => onDelete(s.id)}
                onRename={(name) => onRename(s.id, name)} />
            ))}
          </div>
        </div>
      </div>

      {/* Right: tools */}
      <div className="inspect-right">
        <div className="panel">
          <div className="panel-hd">Tool</div>
          <div className="panel-body">
            <button
              className={'header-btn' + (boxArmed ? ' primary' : '')}
              onClick={onToggleBox}
              title={boxArmed ? 'Cancel (Esc)' : 'Draw a selection box'}
              style={{ width: '100%', justifyContent: 'center' }}>
              {boxArmed ? '◫ Box active — click to cancel' : '◫ Box select'}
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
              style={{ width: '100%', justifyContent: 'center', marginTop: 8 }}>
              Save selection as slice
            </button>
          </div>
        </div>
      </div>
    </>
  );
}

function SliceRow({ slice, active, onPick, onDelete, onRename }) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(slice.name);
  useEffect(() => { setDraft(slice.name); }, [slice.name]);

  const isOriginal = slice.id === 'original';
  const count = slice.indices ? slice.indices.length : null;

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
      {!isOriginal && (
        <button className="header-btn icon-only"
          title="Delete slice (and descendants)"
          onClick={(e) => { e.stopPropagation(); onDelete(); }}
          style={{ marginLeft: 4 }}>×</button>
      )}
    </div>
  );
}
