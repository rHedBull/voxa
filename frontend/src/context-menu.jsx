import { useEffect, useRef } from 'react';

// Shared floating right-click menu. Generic over its items — callers decide
// what shows up (e.g. Label mode's "Edit selection…" row on preseg/SAM/
// instance list rows). Self-contained dismissal: closes on outside click or
// Escape, so callers don't each need to reimplement it. Wired into
// sam-segment-list.jsx and segment-tools.jsx (the row-level right-click
// triggers), dispatched into mode-label.jsx::openCutModal.
export function ContextMenu({ x, y, items, onClose }) {
  const ref = useRef(null);

  useEffect(() => {
    const handleClick = (e) => {
      if (ref.current && !ref.current.contains(e.target)) onClose();
    };
    const handleKeyDown = (e) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('click', handleClick);
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('click', handleClick);
      document.removeEventListener('keydown', handleKeyDown);
    };
  }, [onClose]);

  return (
    <div ref={ref} className="context-menu" style={{ left: x, top: y }}>
      {items.map((item, i) => (
        <div
          key={item.id ?? `${item.label}-${i}`}
          className={`context-menu-row${item.disabled ? ' disabled' : ''}`}
          onClick={() => {
            if (item.disabled) return;
            item.onSelect();
            onClose();
          }}
        >
          {item.label}
        </div>
      ))}
    </div>
  );
}
