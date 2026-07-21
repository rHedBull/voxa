// Class-picker modal — the quick "pick a class for the new instance" overlay
// shared by Box (via mode-label's Ctrl+Enter/Apply path) and Prism (rendered by
// prism-mode after the height is committed). Extracted from mode-label.jsx so
// both can use it without a circular import.
//
// Two-column master–detail over the assignable vocabulary (frozen legacy
// classes are omitted entirely — display-only): the left column lists the
// chord groups, the right column shows the selected group's classes, both
// visible at once. Mouse: click a group, then a class. Keyboard: the same
// two strokes as the global handler — a group key "arms" the group, then a
// member key picks; Esc disarms first, then closes.

import { useEffect, useState } from 'react';
import { CLASS_GROUPS, groupMembers, chordStep } from './class-chords.js';

export function ClassPickerModal({ classes, counts = {}, onPick, onClose }) {
  // null = unarmed (next digit is a group key); a group = armed (next digit
  // picks a member). The right column always shows a group: the armed one,
  // else the first.
  const [armedGroup, setArmedGroup] = useState(null);

  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        e.stopPropagation();
        if (armedGroup) setArmedGroup(null);
        else onClose();
        return;
      }
      const step = chordStep(armedGroup, e.key, classes);
      if (step.type === 'group') {
        e.preventDefault();
        e.stopPropagation();
        setArmedGroup(step.group);
      } else if (step.type === 'class') {
        e.preventDefault();
        e.stopPropagation();
        onPick(step.cls);
      } else if (step.type === 'cancel') {
        e.preventDefault();
        e.stopPropagation();
        setArmedGroup(null);
      }
    };
    // Capture phase so we beat any global keydown that would otherwise also
    // handle the chord (e.g. the mode-label handler's own chordStep, or
    // PrismKeys' own handler).
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [classes, onPick, onClose, armedGroup]);

  const groups = CLASS_GROUPS
    .filter((g) => g.key)
    .map((g) => ({ group: g, members: groupMembers(g.id, classes) }))
    .filter(({ members }) => members.length > 0);

  const shownGroup = armedGroup || groups[0]?.group || null;
  const shownMembers = shownGroup ? groupMembers(shownGroup.id, classes) : [];

  return (
    <div className="class-picker-overlay" onClick={onClose}>
      <div className="class-picker-card" onClick={(e) => e.stopPropagation()}>
        <div className="class-picker-title">Pick class for new instance</div>
        <div className="class-picker-cols">
          <div className="class-picker-list groups">
            {groups.map(({ group, members }) => (
              <button key={group.id}
                className={'class-picker-row group'
                  + (shownGroup?.id === group.id ? ' active' : '')
                  + (armedGroup?.id === group.id ? ' armed' : '')}
                onClick={() => setArmedGroup(group)}
                title={`Press ${group.key}`}>
                <span className="class-picker-hk">{group.key}</span>
                <span className="class-picker-label">{group.label}</span>
                <span className="class-picker-count">{members.length}</span>
                <span className="class-picker-chevron">›</span>
              </button>
            ))}
          </div>
          <div className="class-picker-list members">
            {shownMembers.map((c) => (
              <button key={c.id}
                className="class-picker-row"
                onClick={() => onPick(c)}
                title={`Press ${shownGroup.key} then ${c.hotkey || '–'}`}>
                <span className="class-swatch" style={{ background: c.color }} />
                <span className="class-picker-label">{c.label}</span>
                <span className="class-picker-count">{counts[c.id] || 0}</span>
                <span className="class-picker-hk">{c.hotkey || '–'}</span>
              </button>
            ))}
          </div>
        </div>
        <div className="class-picker-hint">
          {armedGroup
            ? `${armedGroup.label}: press a class key or click · Esc back`
            : 'Group key then class key, or click · Esc cancel'}
        </div>
      </div>
    </div>
  );
}
