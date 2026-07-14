// Shared floating right-click menu. Generic over its items — callers decide
// what shows up (e.g. Label mode's "Edit selection…" row on preseg/SAM/
// instance list rows). Not wired into anything yet; see label-tools.js /
// mode-label.jsx callers for that.
export function ContextMenu({ x, y, items, onClose }) {
  return (
    <div
      className="context-menu"
      style={{ left: x, top: y }}
      onMouseLeave={onClose}
    >
      {items.map((item) => (
        <div
          key={item.label}
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
