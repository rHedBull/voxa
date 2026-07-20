// segment-tools.jsx — presegment list (selection panel).
// Selection happens via Ctrl/Cmd-click in the viewport (wired in mode-label).

import { useMemo, useState } from 'react';
import * as THREE from 'three';
import { deriveFastQueue } from './fast-label.jsx';
import { ContextMenu } from './context-menu.jsx';
import { cutEligibility } from './cut-eligibility.js';
import { fitEligibility } from './fit-eligibility.js';
import { toggleSamSelection } from './sam-segment-list.jsx';
import { maskColor } from './sam-util.js';


// Center the camera on one segment's bounding box. Shared by the
// PresegmentList focus button and Fast labeling's auto-center-on-step.
export function focusSegment(viewerRef, cloud, segState, segId) {
  if (!viewerRef?.current?.frame || !cloud) return;
  // Prefer the precomputed per-segment boxes (segBoxes) — the fallback scan
  // below walks every subsampled point, which is too slow to run per keypress
  // in fast labeling. segBoxes is null for live-segmented clouds.
  const b = segState?.segBoxes;
  if (b?.segIds) {
    const i = Array.prototype.indexOf.call(b.segIds, segId);
    if (i >= 0) {
      const center = new THREE.Vector3(
        b.segCenters[i * 3], b.segCenters[i * 3 + 1], b.segCenters[i * 3 + 2]);
      const radius = Math.max(
        b.segSizes[i * 3], b.segSizes[i * 3 + 1], b.segSizes[i * 3 + 2]) * 0.6 + 0.05;
      viewerRef.current.frame(center, radius);
      return;
    }
  }
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
}


// ── Presegment list ─────────────────────────────────────────────────────────
// Sidebar list of every per-point segment, one row per instance id from
// segState.summary. The list is *only* a fast selector — it groups
// points so the user doesn't have to lasso them in 3D. Click a row to
// select that group, shift-click for multi-select. Confirmation lives
// elsewhere (the Instances panel on the right): there is no parallel
// confirm flow here on purpose.
export function PresegmentList({
  segState, setSegState, classes, viewerRef, cloud,
  excludeSegIds = null, onEditSelection = null, onFitBox = null,
}) {
  // Same canonical "unpromoted segments, largest first" list fast labeling
  // steps through — one builder so the sidebar and the queue can't drift.
  const [cutMenu, setCutMenu] = useState(null); // {x, y} | null
  const [candCutMenu, setCandCutMenu] = useState(null); // {x, y} | null — cut-candidate rows' own menu

  const segmentsAll = useMemo(
    () => deriveFastQueue(segState?.summary, excludeSegIds),
    [segState, excludeSegIds]);

  // Cut-out chunks of a presegment (source:'preseg' in segState.samSegments)
  // live in the same mutable candidate layer SAM masks use — NOT in
  // segState.instanceFull, so they can't be merged into segmentsAll above.
  // Shown here per the design spec ("source:'preseg' entries render as new
  // rows in the Presegment list"), but selected/classified via the shared
  // samSelection/toggleSamSelection/confirmSamSelection pipeline, never
  // segState.selection — reusing onRowClick for these rows would toggle the
  // wrong Set and silently resolve to zero points on classify.
  const cutCandidates = useMemo(() => {
    if (!segState?.samSegments) return [];
    return Array.from(segState.samSegments.entries())
      .filter(([, meta]) => meta.source === 'preseg')
      .map(([id, meta]) => ({ id, ...meta }))
      .sort((a, b) => b.nPoints - a.nPoints);
  }, [segState]);

  // seg.classId is the canonical numeric class id (classes.yaml `id:`), so
  // key by class_id — never array position, which diverges from it.
  const classesById = useMemo(() => {
    const out = {};
    classes.forEach((c) => { out[c.class_id] = c; });
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

  const onRowContextMenu = (e) => {
    e.preventDefault();
    setCutMenu({ x: e.clientX, y: e.clientY });
  };

  // Cut-candidate rows: toggle samSelection (not segState.selection) and open
  // their own context menu, wired to onEditSelection with kind:'sam' — the
  // candidate's points physically live in sam_ids regardless of its
  // display-routing 'preseg' tag, so a recursive cut must address it there.
  const onCandRowClick = (samSegId, evt) => {
    if (!(evt.ctrlKey || evt.metaKey || evt.shiftKey)) return;
    setSegState((s) => (s ? { ...s, samSelection: toggleSamSelection(s.samSelection, samSegId) } : s));
  };

  const onCandRowContextMenu = (e) => {
    e.preventDefault();
    setCandCutMenu({ x: e.clientX, y: e.clientY });
  };

  const total = segmentsAll.length;

  return (
    <div className="preseg-panel">
      <div className="side-hd" style={{ marginTop: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
        <span>Presegments</span>
        <span className="badge-soft">{total}</span>
      </div>
      <div className="inst-list">
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
              onContextMenu={onRowContextMenu}
              title={isSel ? 'Ctrl/Shift-click to deselect'
                           : 'Ctrl/Shift-click to select'}>
              <span className="inst-dot" style={{ background: dot }} />
              <div className="inst-text">
                <b>#{seg.id}</b>
                <em>{cls?.label || (seg.classId < 0 ? 'unlabeled' : `cls ${seg.classId}`)} · {seg.nPoints.toLocaleString()}</em>
              </div>
              <button className="inst-edit-btn"
                onClick={(e) => { e.stopPropagation(); focusSegment(viewerRef, cloud, segState, seg.id); }}
                title="Focus camera on segment">◎</button>
            </div>
          );
        })}
      </div>
      {cutMenu && (
        <ContextMenu
          x={cutMenu.x}
          y={cutMenu.y}
          onClose={() => setCutMenu(null)}
          items={[{
            label: 'Edit selection…',
            disabled: !cutEligibility({ list: 'preseg', selectionSize: segState.selection.size }).eligible,
            onSelect: () => {
              if (!onEditSelection) return;
              onEditSelection(Array.from(segState.selection).map((segId) => ({ kind: 'preseg', segId })));
            },
          }, {
            label: 'Fit box to selection…',
            disabled: !fitEligibility({ list: 'preseg', selectionSize: segState.selection.size }).eligible,
            onSelect: () => {
              if (!onFitBox) return;
              onFitBox(Array.from(segState.selection).map((segId) => ({ kind: 'preseg', segId })));
            },
          }]}
        />
      )}
      {cutCandidates.length > 0 && (
        <>
          <div className="side-hd" style={{ marginTop: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span>Cut selections</span>
            <span className="badge-soft">{cutCandidates.length}</span>
          </div>
          <div className="inst-list">
            {cutCandidates.map((seg) => {
              const isSel = segState.samSelection?.has(seg.id);
              return (
                <div key={`cut-${seg.id}`}
                  className={'inst-row' + (isSel ? ' selected' : '')}
                  onClick={(e) => onCandRowClick(seg.id, e)}
                  onContextMenu={onCandRowContextMenu}
                  title={isSel ? 'Ctrl/Shift-click to deselect'
                               : 'Ctrl/Shift-click to select'}>
                  <span className="inst-dot" style={{ background: maskColor(seg.id) }} />
                  <div className="inst-text">
                    <b>Cut #{seg.id}</b>
                    <em>unclassified · {seg.nPoints.toLocaleString()}</em>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
      {candCutMenu && (
        <ContextMenu
          x={candCutMenu.x}
          y={candCutMenu.y}
          onClose={() => setCandCutMenu(null)}
          items={[{
            label: 'Edit selection…',
            // Candidate rows are backed by samSelection, not segState.selection —
            // eligibility follows the same rule SamSegmentList uses (list:'sam').
            disabled: !cutEligibility({ list: 'sam', selectionSize: segState.samSelection?.size || 0 }).eligible,
            onSelect: () => {
              if (!onEditSelection) return;
              // kind:'sam' (not 'preseg') — the candidate's points live in
              // sam_ids regardless of its 'preseg' display-routing tag; the
              // backend's kind:'preseg' path checks preseg_ids, which would
              // find nothing for a cut-out chunk.
              onEditSelection(Array.from(segState.samSelection).map((segId) => ({ kind: 'sam', segId })));
            },
          }, {
            label: 'Fit box to selection…',
            disabled: !fitEligibility({ list: 'sam', selectionSize: segState.samSelection?.size || 0 }).eligible,
            onSelect: () => {
              if (!onFitBox) return;
              // kind:'sam' — same reasoning as the "Edit selection…" item above.
              onFitBox(Array.from(segState.samSelection).map((segId) => ({ kind: 'sam', segId })));
            },
          }]}
        />
      )}
    </div>
  );
}


