// sam-segment-list.jsx — bottom-of-panel list of materialized SAM candidate
// segments (accepted masks not yet classified). Sibling of PresegmentList —
// deliberately a SEPARATE component/list; SAM candidates and presegments are
// never mixed in one panel or one selection set.
import { useMemo, useState } from 'react';
import { maskColor } from './sam-util.js';
import { ContextMenu } from './context-menu.jsx';
import { cutEligibility } from './cut-eligibility.js';

export function toggleSamSelection(samSelection, samSegId) {
  const next = new Set(samSelection);
  next.has(samSegId) ? next.delete(samSegId) : next.add(samSegId);
  return next;
}

export function SamSegmentList({ segState, setSegState, onEditSelection = null }) {
  const [cutMenu, setCutMenu] = useState(null); // {x, y} | null

  // Only 'sam'-sourced candidates belong here — 'preseg'-sourced cut
  // candidates (source:'preseg') render in PresegmentList instead (see
  // segment-tools.jsx). `source == null` is kept as a defensive fallback for
  // any pre-existing entry that predates the source tag, though
  // materialize_sam_segment has always required it server-side.
  const segments = useMemo(() => {
    if (!segState) return [];
    return Array.from(segState.samSegments.entries())
      .filter(([, meta]) => meta.source === 'sam' || meta.source == null)
      .map(([id, meta]) => ({ id, ...meta }))
      .sort((a, b) => b.nPoints - a.nPoints);
  }, [segState]);

  if (!segState) return null;

  const onRowClick = (samSegId, evt) => {
    if (!(evt.ctrlKey || evt.metaKey || evt.shiftKey)) return;
    setSegState((s) => (s ? { ...s, samSelection: toggleSamSelection(s.samSelection, samSegId) } : s));
  };

  const onRowContextMenu = (e) => {
    e.preventDefault();
    setCutMenu({ x: e.clientX, y: e.clientY });
  };

  return (
    <div className="preseg-panel">
      <div className="side-hd" style={{ marginTop: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
        <span>SAM segments</span>
        <span className="badge-soft">{segments.length}</span>
      </div>
      <div className="inst-list">
        {segments.length === 0 && (
          <div className="sugg-empty" style={{ fontSize: '11px', padding: '6px 4px' }}>
            No SAM segments yet. Shift-drag a box or run a concept capture.
          </div>
        )}
        {segments.map((seg) => {
          const isSel = segState.samSelection.has(seg.id);
          return (
            <div key={seg.id}
              className={'inst-row' + (isSel ? ' selected' : '')}
              onClick={(e) => onRowClick(seg.id, e)}
              onContextMenu={onRowContextMenu}
              title={isSel ? 'Ctrl/Shift-click to deselect' : 'Ctrl/Shift-click to select'}>
              <span className="inst-dot" style={{ background: maskColor(seg.id) }} />
              <div className="inst-text">
                <b>SAM #{seg.id}</b>
                <em>{seg.nPoints.toLocaleString()} pts</em>
              </div>
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
            disabled: !cutEligibility({ list: 'sam', selectionSize: segState.samSelection.size }).eligible,
            onSelect: () => {
              if (!onEditSelection) return;
              onEditSelection(Array.from(segState.samSelection).map((segId) => ({ kind: 'sam', segId })));
            },
          }]}
        />
      )}
    </div>
  );
}
