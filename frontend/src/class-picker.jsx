// Class-picker modal — the quick "pick a class for the new instance" overlay
// shared by Box (via mode-label's Ctrl+Enter/Apply path) and Prism (rendered by
// prism-mode after the height is committed). Extracted from mode-label.jsx so
// both can use it without a circular import.
//
// Renders the assignable vocabulary grouped by chord group (frozen legacy
// classes are omitted entirely — they are display-only) and accepts the same
// two-stroke chords as the global handler: group key, then member key.

import { useEffect, useState } from 'react';
import { CLASS_GROUPS, groupMembers, chordStep } from './class-chords.js';

export function ClassPickerModal({ classes, counts = {}, onPick, onClose }) {
  const [pendingGroup, setPendingGroup] = useState(null);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        // First Esc backs out of a pending group, second closes the modal.
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
    .filter((g) => g.key && (!pendingGroup || pendingGroup.id === g.id))
    .map((g) => ({ group: g, members: groupMembers(g.id, classes) }))
    .filter(({ members }) => members.length > 0);

  return (
    <div className="class-picker-overlay" onClick={onClose}>
      <div className="class-picker-card" onClick={(e) => e.stopPropagation()}>
        <div className="class-picker-title">Pick class for new instance</div>
        <div className="class-picker-list">
          {groups.map(({ group, members }) => (
            <div key={group.id}>
              <div className="class-picker-group">
                <span className="class-picker-hk">{group.key}</span> {group.label}
              </div>
              {members.map((c) => (
                <button key={c.id}
                  className="class-picker-row"
                  onClick={() => onPick(c)}
                  title={`Press ${group.key} then ${c.hotkey || '–'}`}>
                  <span className="class-swatch" style={{ background: c.color }} />
                  <span className="class-picker-label">{c.label}</span>
                  <span className="class-picker-count">{counts[c.id] || 0}</span>
                  <span className="class-picker-hk">{c.hotkey || '–'}</span>
                </button>
              ))}
            </div>
          ))}
        </div>
        <div className="class-picker-hint">
          {pendingGroup
            ? `${pendingGroup.label}: press a member key · Esc to back out`
            : 'Group key, then class key · Esc to cancel'}
        </div>
      </div>
    </div>
  );
}
