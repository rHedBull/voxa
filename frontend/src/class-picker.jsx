// Class-picker modal — the quick "pick a class for the new instance" overlay
// shared by Box (via mode-label's Ctrl+Enter/Apply path) and Prism (rendered by
// prism-mode after the height is committed). Extracted from mode-label.jsx so
// both can use it without a circular import.

import { useEffect } from 'react';

export function ClassPickerModal({ classes, counts = {}, onPick, onClose }) {
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
        return;
      }
      const cls = classes.find((c) => c.hotkey === e.key);
      if (cls) {
        e.preventDefault();
        e.stopPropagation();
        onPick(cls);
      }
    };
    // Capture phase so we beat any global keydown that would otherwise also
    // handle the hotkey (e.g. "1" → setActiveClass, or PrismKeys' own handler).
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [classes, onPick, onClose]);

  return (
    <div className="class-picker-overlay" onClick={onClose}>
      <div className="class-picker-card" onClick={(e) => e.stopPropagation()}>
        <div className="class-picker-title">Pick class for new instance</div>
        <div className="class-picker-list">
          {classes.map((c) => (
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
        </div>
        <div className="class-picker-hint">Press a number to assign · Esc to cancel</div>
      </div>
    </div>
  );
}
