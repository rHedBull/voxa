// Shared viewport atoms used by all three modes.

import { useEffect, useRef, useState } from 'react';

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

// Hotkeys / help popover. Lives in the top HUD next to the camera presets.
//
// `sections` is an array of:
//   { title: string, items: [{ keys: string[], desc: string }, ...] }
// The popover toggles open on click or `?`, closes on Esc or outside-click.
// Hosts pass a sections array tailored to the active mode.
export function HelpButton({ sections, hotkey = '?', placement = 'down' }) {
  const [open, setOpen] = useState(false);
  const wrapRef = useRef(null);

  useEffect(() => {
    const onKey = (e) => {
      if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
      if (e.key === hotkey || (e.key === '/' && e.shiftKey)) {
        e.preventDefault();
        setOpen((v) => !v);
      } else if (e.key === 'Escape' && open) {
        setOpen(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [hotkey, open]);

  useEffect(() => {
    if (!open) return;
    const onDoc = (e) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target)) setOpen(false);
    };
    document.addEventListener('mousedown', onDoc);
    return () => document.removeEventListener('mousedown', onDoc);
  }, [open]);

  return (
    <div className="help-wrap" ref={wrapRef}>
      <button
        className={'help-btn' + (open ? ' active' : '')}
        onClick={() => setOpen((v) => !v)}
        title={`Keyboard shortcuts (${hotkey})`}
        aria-label="Keyboard shortcuts"
      >?</button>
      {open && (
        <div className={'help-pop' + (placement === 'up' ? ' up' : '')} role="dialog">
          <div className="help-hd">
            <span>Shortcuts</span>
            <button className="help-close" onClick={() => setOpen(false)}
              aria-label="Close">×</button>
          </div>
          <div className="help-body">
            {sections.map((s, i) => (
              <div className="help-section" key={i}>
                <div className="help-title">{s.title}</div>
                <ul>
                  {s.items.map((it, j) => (
                    <li key={j}>
                      <span className="help-keys">
                        {it.keys.map((k, ki) => (
                          <kbd key={ki}>{k}</kbd>
                        ))}
                      </span>
                      <span className="help-desc">{it.desc}</span>
                    </li>
                  ))}
                </ul>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

// Orbit ↔ Walk toggle. Same chrome as the camera-preset chips so it sits
// naturally next to them in the HUD.
export function NavModeToggle({ navMode, onChange }) {
  return (
    <div className="cam-presets" title={navMode === 'walk' ? 'Walk: WASD move, Q/E up/down, drag to look' : 'Orbit: drag to rotate, scroll to zoom'}>
      <button className={'cam-btn' + (navMode === 'orbit' ? ' active' : '')}
        onClick={() => onChange('orbit')}>orbit</button>
      <button className={'cam-btn' + (navMode === 'walk' ? ' active' : '')}
        onClick={() => onChange('walk')}>walk</button>
    </div>
  );
}
