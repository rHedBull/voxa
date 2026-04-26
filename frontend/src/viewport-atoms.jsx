// Shared viewport atoms used by all three modes.

export function ViewportToolbar({ children, side = 'left' }) {
  return <div className="vp-toolbar" data-side={side}>{children}</div>;
}

export function ToolButton({ icon, label, active, badge, onClick, hotkey, mini }) {
  return (
    <button type="button"
      className={'tool-btn' + (active ? ' active' : '') + (mini ? ' mini' : '')}
      onClick={onClick}
      title={label + (hotkey ? `  (${hotkey})` : '')}>
      <span className="tool-ico" aria-hidden>{icon}</span>
      {!mini && <span className="tool-lbl">{label}</span>}
      {hotkey && !mini && <span className="tool-hk">{hotkey}</span>}
      {badge != null && <span className="tool-badge">{badge}</span>}
    </button>
  );
}

export function HUDChip({ label, value, mono, accent }) {
  return (
    <div className={'hud-chip' + (accent ? ' accent' : '')}>
      <span className="hud-lbl">{label}</span>
      <span className={'hud-val' + (mono ? ' mono' : '')}>{value}</span>
    </div>
  );
}

export function CameraPresets({ onPreset }) {
  return (
    <div className="cam-presets">
      {['iso', 'top', 'front', 'side'].map((p) => (
        <button key={p} className="cam-btn" onClick={() => onPreset(p)}>{p}</button>
      ))}
    </div>
  );
}
