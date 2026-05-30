// segment-tools.jsx — presegment list (selection panel).
// Selection happens via Ctrl/Cmd-click in the viewport (wired in mode-label).

import { useMemo } from 'react';
import * as THREE from 'three';



// ── Presegment list ─────────────────────────────────────────────────────────
// Sidebar list of every per-point segment, one row per instance id from
// segState.summary. The list is *only* a fast selector — it groups
// points so the user doesn't have to lasso them in 3D. Click a row to
// select that group, shift-click for multi-select. Confirmation lives
// elsewhere (the Instances panel on the right): there is no parallel
// confirm flow here on purpose.
export function PresegmentList({
  segState, setSegState, classes, viewerRef, cloud,
  excludeSegIds = null,
}) {
  const segmentsAll = useMemo(() => {
    if (!segState) return [];
    const out = [];
    for (const [id, info] of segState.summary.entries()) {
      if (excludeSegIds && excludeSegIds.has(id)) continue;
      out.push({ id, classId: info.classId, nPoints: info.nPoints });
    }
    out.sort((a, b) => b.nPoints - a.nPoints);
    return out;
  }, [segState, excludeSegIds]);

  const classesById = useMemo(() => {
    const out = {};
    classes.forEach((c, i) => { out[i] = c; out[c.id] = c; });
    return out;
  }, [classes]);

  if (!segState) return null;

  const onRowClick = (segId, evt) => {
    if (!(evt.ctrlKey || evt.metaKey || evt.shiftKey)) return;
    setSegState((s) => {
      if (!s) return s;
      const next = new Set(s.selection);
      next.has(segId) ? next.delete(segId) : next.add(segId);
      return { ...s, selection: next };
    });
  };

  const focusSegment = (segId) => {
    if (!viewerRef?.current?.frame || !cloud) return;
    const subIdx = cloud.subsampleIdx;
    const pos = cloud.positions;
    if (!pos) return;
    const inst = segState.instanceFull;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    let n = 0;
    const subN = pos.length / 3;
    for (let p = 0; p < subN; p++) {
      const fullIdx = subIdx ? subIdx[p] : p;
      if (inst[fullIdx] !== segId) continue;
      const x = pos[p * 3], y = pos[p * 3 + 1], z = pos[p * 3 + 2];
      if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
      if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;
      n++;
    }
    if (n === 0) return;
    // viewer.frame() calls THREE.Vector3.copy() on the center, which
    // requires a Vector3-like object (not a plain array — copy() is a
    // silent no-op on arrays and the camera ends up looking at NaN).
    const center = new THREE.Vector3(
      (minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);
    const radius = Math.max(maxX - minX, maxY - minY, maxZ - minZ) * 0.6 + 0.05;
    viewerRef.current.frame(center, radius);
  };

  const total = segmentsAll.length;

  return (
    <div className="preseg-panel">
      <div className="side-hd" style={{ marginTop: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
        <span>Presegments</span>
        <span className="badge-soft">{total}</span>
      </div>
      <div className="inst-list" style={{ maxHeight: '40vh', overflowY: 'auto' }}>
        {total === 0 && (
          <div className="sugg-empty" style={{ fontSize: '11px', padding: '6px 4px' }}>
            No presegments. Click ⚙ to choose a mode and run.
          </div>
        )}
        {segmentsAll.map((seg) => {
          const cls = classesById[seg.classId];
          const isSel = segState.selection.has(seg.id);
          const dot = cls?.color || '#6b7280';
          return (
            <div key={seg.id}
              className={'inst-row' + (isSel ? ' selected' : '')}
              onClick={(e) => onRowClick(seg.id, e)}
              title={isSel ? 'Ctrl/Shift-click to deselect'
                           : 'Ctrl/Shift-click to select'}>
              <span className="inst-dot" style={{ background: dot }} />
              <div className="inst-text">
                <b>#{seg.id}</b>
                <em>{cls?.label || (seg.classId < 0 ? 'unlabeled' : `cls ${seg.classId}`)} · {seg.nPoints.toLocaleString()}</em>
              </div>
              <button className="inst-edit-btn"
                onClick={(e) => { e.stopPropagation(); focusSegment(seg.id); }}
                title="Focus camera on segment">◎</button>
            </div>
          );
        })}
      </div>
    </div>
  );
}


