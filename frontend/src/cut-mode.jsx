// cut-mode.jsx — isolated cut modal: mounts a second Viewer over just the
// points belonging to the sources the user right-clicked "Edit selection…"
// on, reuses the Box tool's draw/transform pattern (own selBox/transformMode,
// G/R/Y keys), and round-trips the drawn OBB through POST /api/segment/
// cut-shape. Dumb by design: it never touches the parent's segState directly
// — it only reports the raw cutShape response upward via onCutConfirmed, so
// mode-label.jsx stays the single place that patches segState/instances
// (mirrors confirmSegmentSelection/confirmSamSelection/applyBox).
// See docs/superpowers/specs/2026-07-14-cut-selection-tool-design.md.
import { useState, useEffect, useMemo, useCallback, useRef } from 'react';
import { Viewer } from './viewer.jsx';
import { VoxaAPI } from './api.js';

export const CUT_SEL_BOX_ID = '__cut_sel_box__';
const CUT_SEL_BOX_COLOR = '#ffd24a';

// Pure, testable: given the parent's already-loaded segment state + render
// cloud, produce a filtered/tagged cloud containing only the given sources'
// points. `sources` is [{kind:'preseg'|'sam'|'instance', segId}]. 'preseg'
// and 'instance' both test against segState.instanceFull — presegment ids
// and confirmed instance ids share one namespace there (a preseg id IS an
// instance id until it's promoted; see mode-label.jsx's Ctrl-click handler
// and CLAUDE.md's per-tool-select section) — 'sam' tests segState.samIds.
// `cloud` is the main viewport's (possibly subsampled) render cloud;
// cloud.subsampleIdx (sub -> full index) is used when present. Returns
// { positions, colors, tags, fullIndices } — fullIndices[i] is the full-res
// point index for filtered point i, needed to reconcile a cut-shape response
// (which reports full-res indices) back against this filtered view.
export function buildCutCloud(segState, cloud, sources) {
  const subN = cloud.positions.length / 3;
  const subIdx = cloud.subsampleIdx;
  const kept = [];
  for (let p = 0; p < subN; p++) {
    const f = subIdx ? subIdx[p] : p;
    for (const src of sources) {
      const arr = src.kind === 'sam' ? segState.samIds : segState.instanceFull;
      if (arr[f] === src.segId) {
        kept.push({ p, f, kind: src.kind, segId: src.segId });
        break; // sources never merge — first match wins, matches server partitioning
      }
    }
  }
  const n = kept.length;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  const fullIndices = new Int32Array(n);
  const tags = new Array(n);
  for (let i = 0; i < n; i++) {
    const { p, f, kind, segId } = kept[i];
    positions[i * 3] = cloud.positions[p * 3];
    positions[i * 3 + 1] = cloud.positions[p * 3 + 1];
    positions[i * 3 + 2] = cloud.positions[p * 3 + 2];
    colors[i * 3] = cloud.colors[p * 3];
    colors[i * 3 + 1] = cloud.colors[p * 3 + 1];
    colors[i * 3 + 2] = cloud.colors[p * 3 + 2];
    fullIndices[i] = f;
    tags[i] = { kind, segId };
  }
  return { positions, colors, tags, fullIndices };
}

// Drop every filtered point whose full index is in `removedFullIndices`
// (a Set<number>) — used after a successful cut so the next box in the same
// modal session draws against the remainder, per the spec's "multiple cuts
// per session" requirement. Pure; returns a new filtered object.
export function removeCutPoints(filtered, removedFullIndices) {
  if (!removedFullIndices || removedFullIndices.size === 0) return filtered;
  const keepIdx = [];
  for (let i = 0; i < filtered.fullIndices.length; i++) {
    if (!removedFullIndices.has(filtered.fullIndices[i])) keepIdx.push(i);
  }
  if (keepIdx.length === filtered.fullIndices.length) return filtered;
  const n = keepIdx.length;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  const fullIndices = new Int32Array(n);
  const tags = new Array(n);
  for (let i = 0; i < n; i++) {
    const src = keepIdx[i];
    positions[i * 3] = filtered.positions[src * 3];
    positions[i * 3 + 1] = filtered.positions[src * 3 + 1];
    positions[i * 3 + 2] = filtered.positions[src * 3 + 2];
    colors[i * 3] = filtered.colors[src * 3];
    colors[i * 3 + 1] = filtered.colors[src * 3 + 1];
    colors[i * 3 + 2] = filtered.colors[src * 3 + 2];
    fullIndices[i] = filtered.fullIndices[src];
    tags[i] = filtered.tags[src];
  }
  return { positions, colors, tags, fullIndices };
}

function bboxOf(positions) {
  const n = positions.length / 3;
  if (n === 0) return null;
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let p = 0; p < n; p++) {
    const x = positions[p * 3], y = positions[p * 3 + 1], z = positions[p * 3 + 2];
    if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
    if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;
  }
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

// CutModal({ segState, cloud, sources, protectInstances, theme, onClose,
//            onCutConfirmed }) — see spec's "Isolated modal" + "Cut-confirm
// flow" sections. `onCutConfirmed(response)` receives the raw VoxaAPI.
// cutShape(...) response so the parent can patch its own segState/instances
// (applySamDelta per materialized entry, applyDelta for the instance entry).
export default function CutModal({
  segState, cloud, sources, protectInstances = [], theme, onClose, onCutConfirmed,
}) {
  // Auto-framing already happens on every cloud change via viewerCloud's
  // bbox (Viewer's own cloud-load effect calls controller.frame() from it),
  // so this ref has no imperative caller today — kept for future use (e.g.
  // re-framing on demand without waiting for a cloud swap).
  const viewerRef = useRef(null);
  const [filtered, setFiltered] = useState(() => buildCutCloud(segState, cloud, sources));
  const [selBox, setSelBox] = useState(null);
  const [transformMode, setTransformMode] = useState('translate');
  const [busy, setBusy] = useState(false);

  const bbox = useMemo(() => bboxOf(filtered.positions), [filtered]);
  const viewerCloud = useMemo(
    () => ({ positions: filtered.positions, colors: filtered.colors, bbox }),
    [filtered, bbox],
  );

  // Mirrors mode-label.jsx's toggleBoxSelect: quarter-bbox box, floored at
  // 0.5 per axis, centered on the filtered cloud's bbox.
  const toggleBox = useCallback(() => {
    setSelBox((b) => {
      if (b) return null;
      if (!bbox) return null;
      const c = [
        (bbox.min[0] + bbox.max[0]) / 2,
        (bbox.min[1] + bbox.max[1]) / 2,
        (bbox.min[2] + bbox.max[2]) / 2,
      ];
      const s = [
        Math.max((bbox.max[0] - bbox.min[0]) / 4, 0.5),
        Math.max((bbox.max[1] - bbox.min[1]) / 4, 0.5),
        Math.max((bbox.max[2] - bbox.min[2]) / 4, 0.5),
      ];
      return { id: CUT_SEL_BOX_ID, label: 'cut-box', cls: 0, color: CUT_SEL_BOX_COLOR,
                center: c, size: s, rotation: [0, 0, 0] };
    });
  }, [bbox]);

  // Only ever one synthetic box in this modal — simpler than mode-label's
  // dispatcher, which also has to route to real instances.
  const onCuboidTransform = useCallback((id, patch) => {
    if (id !== CUT_SEL_BOX_ID) return;
    setSelBox((b) => (b ? { ...b, ...patch } : b));
  }, []);

  const confirmCut = useCallback(async () => {
    if (!selBox || busy) return;
    setBusy(true);
    let r;
    try {
      r = await VoxaAPI.cutShape({
        shape: { type: 'obb', center: selBox.center, size: selBox.size, rotation: selBox.rotation },
        sources,
        protectInstances,
      });
    } catch (err) {
      console.error('cut-shape failed:', err);
      setBusy(false);
      return;
    }
    setBusy(false);
    onCutConfirmed(r);
    const removed = new Set();
    for (const m of r.materialized) {
      if (!m.indices) continue;
      for (let k = 0; k < m.indices.length; k++) removed.add(m.indices[k]);
    }
    if (r.instance?.indices) {
      for (let k = 0; k < r.instance.indices.length; k++) removed.add(r.instance.indices[k]);
    }
    setFiltered((f) => removeCutPoints(f, removed));
    setSelBox(null);
  }, [selBox, busy, sources, protectInstances, onCutConfirmed]);

  // G/R/Y transform-mode keys (only while a box exists) + Escape (clear box,
  // else close the modal) + Ctrl/Cmd+Enter (confirm cut) — scoped to this
  // modal only, cleaned up on unmount/close.
  useEffect(() => {
    const onKey = (e) => {
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return;
      if (selBox) {
        if (e.key === 'g' || e.key === 'G') { e.preventDefault(); setTransformMode('translate'); return; }
        if (e.key === 'r' || e.key === 'R') { e.preventDefault(); setTransformMode('rotate'); return; }
        if (e.key === 'y' || e.key === 'Y') { e.preventDefault(); setTransformMode('scale'); return; }
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') { e.preventDefault(); confirmCut(); return; }
      }
      if (e.key === 'Escape') {
        e.preventDefault();
        if (selBox) setSelBox(null);
        else onClose();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selBox, confirmCut, onClose]);

  return (
    <div className="cut-modal-overlay" role="dialog" aria-modal="true">
      <div className="cut-modal">
        <div className="cut-modal-hd">
          <span>Edit selection — {filtered.positions.length / 3} pts remaining</span>
          <button className="ghost-btn" onClick={onClose}>Close</button>
        </div>
        <div className="cut-modal-body">
          <Viewer
            ref={viewerRef}
            cloud={viewerCloud}
            instances={selBox ? [selBox] : []}
            selectedId={selBox ? CUT_SEL_BOX_ID : null}
            transformMode={selBox ? transformMode : null}
            onCuboidTransform={onCuboidTransform}
            navMode="orbit"
            background={theme?.bg}
            floorColor={theme?.floor}
            pointSize={0.012}
            showCuboids
          />
        </div>
        <div className="cut-modal-toolbar">
          {selBox ? (
            <>
              <button className="ghost-btn" onClick={toggleBox}>Clear box</button>
              <div className="tool-opt-toggle">
                <button className={transformMode === 'translate' ? 'active' : ''}
                  onClick={() => setTransformMode('translate')}>Move (G)</button>
                <button className={transformMode === 'rotate' ? 'active' : ''}
                  onClick={() => setTransformMode('rotate')}>Rotate (R)</button>
                <button className={transformMode === 'scale' ? 'active' : ''}
                  onClick={() => setTransformMode('scale')}>Scale (Y)</button>
              </div>
              <button className="ghost-btn active" disabled={busy} onClick={confirmCut}>
                {busy ? 'Cutting…' : 'Confirm cut (Ctrl+Enter)'}
              </button>
            </>
          ) : (
            <button className="ghost-btn active" onClick={toggleBox}>Draw a box</button>
          )}
        </div>
      </div>
    </div>
  );
}
