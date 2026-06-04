// session-picker.jsx — labeling-session list + create/rename/delete for an
// annotated scan. Dumb component: all data + handlers arrive via props.

import { useState as useStateSP } from 'react';

// "2026-06-04T13:22:10..." → "2026-06-04 13:22". Returns '' for missing.
export function formatSavedAt(s) {
  return s ? s.slice(0, 16).replace('T', ' ') : '';
}

export default function SessionPicker({ sessions, activeSessionId, presegs, onSelect, onCreate, onRename, onDelete }) {
  const [creating, setCreating] = useStateSP(sessions.length === 0);
  const [newName, setNewName] = useStateSP('');
  const [newPreseg, setNewPreseg] = useStateSP('');
  const [renamingId, setRenamingId] = useStateSP(null);
  const [renameText, setRenameText] = useStateSP('');

  const submitCreate = () => {
    const name = newName.trim();
    if (!name) return;
    onCreate({ name, presegId: newPreseg || null });
    setNewName('');
    setNewPreseg('');
    setCreating(false);
  };

  const commitRename = (sid) => {
    const name = renameText.trim();
    if (name) onRename(sid, name);
    setRenamingId(null);
  };

  return (
    <aside className="session-picker">
      <div className="side-hd">
        <span>Sessions</span>
        <span className="badge-soft">{sessions.length}</span>
      </div>

      {sessions.length === 0 && !creating && (
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
              onClick={() => { if (!s.corrupt && !isRenaming) onSelect(s.session_id); }}>
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

      {creating ? (
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
            {sessions.length > 0 && (
              <button className="ghost-btn" onClick={() => setCreating(false)}>Cancel</button>
            )}
          </div>
        </div>
      ) : (
        <button className="ghost-btn" onClick={() => setCreating(true)}>+ New session</button>
      )}
    </aside>
  );
}
