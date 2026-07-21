// Class-picker modal — the quick "pick a class for the new instance" overlay
// shared by Box (via mode-label's Ctrl+Enter/Apply path) and Prism (rendered by
// prism-mode after the height is committed). Extracted from mode-label.jsx so
// both can use it without a circular import.
//
// Two-level drill-down over the assignable vocabulary (frozen legacy classes
// are omitted entirely — display-only): level 1 lists the chord groups, click
// a group (or press its key) to open its member list, pick a member by click
// or its key. Esc backs out one level, then closes. Mirrors the two-stroke
// chords of the global handler.

import { useEffect, useState } from 'react';
import { CLASS_GROUPS, groupMembers, chordStep } from './class-chords.js';

export function ClassPickerModal({ classes, counts = {}, onPick, onClose }) {
  const [pendingGroup, setPendingGroup] = useState(null);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        // First Esc backs out of an open group, second closes the modal.
        if (pendingGroup) setPendingGroup(null);
        else onClose();
        return;
      }
      const step = chordStep(pendingGroup, e.key, classes);
      if (step.type === 'group') {
        e.preventDefault();
        e.stopPropagation();
        setPendingGroup(step.group);
      } else if (step.type === 'class') {
        e.preventDefault();
        e.stopPropagation();
        onPick(step.cls);
      } else if (step.type === 'cancel') {
        e.preventDefault();
        e.stopPropagation();
        setPendingGroup(null);
      }
    };
    // Capture phase so we beat any global keydown that would otherwise also
    // handle the chord (e.g. the mode-label handler's own chordStep, or
    // PrismKeys' own handler).
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [classes, onPick, onClose, pendingGroup]);

  const groups = CLASS_GROUPS
    .filter((g) => g.key)
    .map((g) => ({ group: g, members: groupMembers(g.id, classes) }))
    .filter(({ members }) => members.length > 0);

  const openMembers = pendingGroup ? groupMembers(pendingGroup.id, classes) : null;

  return (
    <div className="class-picker-overlay" onClick={onClose}>
      <div className="class-picker-card" onClick={(e) => e.stopPropagation()}>
        <div className="class-picker-title">Pick class for new instance</div>
        <div className="class-picker-list">
          {pendingGroup == null ? (
            groups.map(({ group, members }) => (
              <button key={group.id}
                className="class-picker-row group"
                onClick={() => setPendingGroup(group)}
                title={`Press ${group.key}`}>
                <span className="class-picker-hk">{group.key}</span>
                <span className="class-picker-label">{group.label}</span>
                <span className="class-picker-count">{members.length}</span>
                <span className="class-picker-chevron">›</span>
              </button>
            ))
          ) : (
            <>
              <button className="class-picker-row back"
                onClick={() => setPendingGroup(null)}
                title="Back to groups (Esc)">
                <span className="class-picker-chevron">‹</span>
                <span className="class-picker-label">{pendingGroup.label}</span>
              </button>
              {openMembers.map((c) => (
                <button key={c.id}
                  className="class-picker-row"
                  onClick={() => onPick(c)}
                  title={`Press ${c.hotkey || '–'}`}>
                  <span className="class-swatch" style={{ background: c.color }} />
                  <span className="class-picker-label">{c.label}</span>
                  <span className="class-picker-count">{counts[c.id] || 0}</span>
                  <span className="class-picker-hk">{c.hotkey || '–'}</span>
                </button>
              ))}
            </>
          )}
        </div>
        <div className="class-picker-hint">
          {pendingGroup
            ? 'Press a class key or click · Esc back'
            : 'Press a group key or click · Esc cancel'}
        </div>
      </div>
    </div>
  );
}
