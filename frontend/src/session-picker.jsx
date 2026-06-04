// session-picker.jsx — labeling-session selector for an annotated scan.
// Compact dropdown: the trigger always shows the ACTIVE session (+ loading
// state); expanding it reveals the session list with rename/delete and the
// create form. Dumb component: all data + handlers arrive via props.

import { useState as useStateSP } from 'react';

// "2026-06-04T13:22:10..." → "2026-06-04 13:22". Returns '' for missing.
export function formatSavedAt(s) {
  return s ? s.slice(0, 16).replace('T', ' ') : '';
}

export default function SessionPicker({ sessions, activeSessionId, presegs, loading, onSelect, onCreate, onRename, onDelete }) {
  const [open, setOpen] = useStateSP(false);
  const [creating, setCreating] = useStateSP(false);
  const [newName, setNewName] = useStateSP('');
  const [newPreseg, setNewPreseg] = useStateSP('');
  const [renamingId, setRenamingId] = useStateSP(null);
  const [renameText, setRenameText] = useStateSP('');

  const active = sessions.find((s) => s.session_id === activeSessionId) || null;
  const empty = sessions.length === 0;
  // The list + create form live in the expanding section. It is forced open
  // while the scan has no sessions (there is nothing to collapse onto).
  const expanded = open || creating || empty;

  const submitCreate = () => {
    const name = newName.trim();
    if (!name) return;
    onCreate({ name, presegId: newPreseg || null });
    setNewName('');
    setNewPreseg('');
    setCreating(false);
    setOpen(false);
  };

  const commitRename = (sid) => {
    const name = renameText.trim();
    if (name) onRename(sid, name);
    setRenamingId(null);
  };

  return (
    <aside className="session-picker">
      <div className="side-hd">
        <span>Session</span>
        <span className="badge-soft">{sessions.length}</span>
      </div>

      {/* Trigger: always shows the active session + open/loading state, even
          while the list is collapsed. */}
      <button className="session-current" onClick={() => setOpen((o) => !o)}
        title={expanded ? 'Collapse session list' : 'Switch session'}>
        <span className="inst-dot"
          style={{ background: active ? '#10b981' : '#6b7280' }} />
        <b>{active ? active.name : (empty ? 'no sessions' : 'no session open')}</b>
        <span className={'session-open-tag' + (loading ? ' loading' : '')}>
          {loading ? 'loading…' : (active ? 'open' : '—')}
        </span>
        <span className="session-chev">{expanded ? '▴' : '▾'}</span>
      </button>

      {expanded && (
        <div className="session-dropdown">
          {empty && (
            <div className="sugg-empty">No sessions yet — create one to start labeling</div>
          )}

          <div className="inst-list">
            {sessions.map((s) => {
              const isActive = s.session_id === activeSessionId;
              const isRenaming = s.session_id === renamingId;
              return (
                <div key={s.session_id}
                  className={'inst-row' + (isActive ? ' selected' : '') + (s.corrupt ? ' corrupt' : '')}
                  title={s.corrupt ? 'Session is corrupt' : 'Click to switch session'}
                  onClick={() => {
                    if (s.corrupt || isRenaming) return;
                    onSelect(s.session_id);
                    setOpen(false);
                  }}>
                  <span className="inst-dot" style={{ background: s.corrupt ? '#6b7280' : '#10b981' }} />
                  {isRenaming ? (
                    <input className="ins-input" autoFocus
                      value={renameText}
                      onClick={(e) => e.stopPropagation()}
                      onChange={(e) => setRenameText(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === 'Enter') { e.preventDefault(); commitRename(s.session_id); }
                        else if (e.key === 'Escape') { e.preventDefault(); setRenamingId(null); }
                      }} />
                  ) : (
                    <div className="inst-text">
                      <b>{s.name}</b>
                      <em>
                        <span className="badge-soft">{s.preseg_id ?? 'blank'}</span>
                        {s.saved_at ? ` · ${formatSavedAt(s.saved_at)}` : ''}
                        {s.corrupt ? ' · corrupt' : ''}
                      </em>
                    </div>
                  )}
                  {isActive && !isRenaming && (
                    <span className={'session-open-tag' + (loading ? ' loading' : '')}
                      title={loading ? 'Loading session…' : 'This session is open'}>
                      {loading ? 'loading…' : 'open'}
                    </span>
                  )}
                  {s.dirty && !s.corrupt && (
                    <span title="Unsaved changes" style={{ color: 'oklch(0.75 0.18 60)' }}>●</span>
                  )}
                  {!s.corrupt && !isRenaming && (
                    <button className="inst-edit-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        setRenamingId(s.session_id);
                        setRenameText(s.name);
                      }}
                      title="Rename">✎</button>
                  )}
                  {!isRenaming && (
                    <button className="inst-edit-btn"
                      onClick={(e) => {
                        e.stopPropagation();
                        if (window.confirm(`Delete session "${s.name}"? This removes its labels.`)) {
                          onDelete(s.session_id);
                        }
                      }}
                      title="Delete">⌫</button>
                  )}
                </div>
              );
            })}
          </div>

          {(creating || empty) ? (
            <div className="inst-edit-panel">
              <div className="ins-row">
                <label>Name</label>
                <input className="ins-input" autoFocus
                  placeholder="Session name"
                  value={newName}
                  onChange={(e) => setNewName(e.target.value)}
                  onKeyDown={(e) => { if (e.key === 'Enter') { e.preventDefault(); submitCreate(); } }} />
              </div>
              <div className="ins-row">
                <label>Preseg</label>
                <select className="ins-input"
                  value={newPreseg}
                  onChange={(e) => setNewPreseg(e.target.value)}>
                  <option value="">blank</option>
                  {(presegs || []).map((p) => (
                    <option key={p.preseg_id} value={p.preseg_id}>
                      {p.preseg_id} ({p.n_segments} segs)
                    </option>
                  ))}
                </select>
              </div>
              <div className="ins-actions">
                <button className="ghost-btn" onClick={submitCreate} disabled={!newName.trim()}>Create</button>
                {!empty && (
                  <button className="ghost-btn" onClick={() => setCreating(false)}>Cancel</button>
                )}
              </div>
            </div>
          ) : (
            <button className="ghost-btn" onClick={() => setCreating(true)}>+ New session</button>
          )}
        </div>
      )}
    </aside>
  );
}
