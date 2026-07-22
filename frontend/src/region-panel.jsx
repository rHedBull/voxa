// region-panel.jsx — the Regions tab of Label mode's right panel (eval-
// labeling phase 1). Presentational: regions + stats + instances come in as
// props; every mutation goes out through a callback so mode-label.jsx owns
// the API calls and state. Membership = majority-inside (region-utils.js).

import { useState } from 'react';
import { majorityInstances, unlabeledPct, regionCssColor } from './region-utils.js';

export default function RegionPanel({
  regions, stats, instances, classes, eyes,
  onToggleEye, onRename, onDelete, onFlipStatus, onSelectInstance,
}) {
  const [expanded, setExpanded] = useState(null);   // region id
  const [editingName, setEditingName] = useState(null); // {id, value}
  const [rowError, setRowError] = useState(null);       // {id, message}

  // Every row mutation goes through here. The callbacks in mode-label.jsx
  // deliberately do NOT catch — a rejected rename/delete/flip (409 no session,
  // 422 locked/gated) has to reach the row that triggered it, not the console.
  const run = (id, result) => {
    setRowError(null);
    Promise.resolve(result).catch((err) => setRowError({ id, message: err.detail || err.message }));
  };

  if (!regions.length) {
    return <div className="sugg-empty">No regions yet. Pick the Region tool and trace a footprint.</div>;
  }
  return (
    <div className="region-list">
      {regions.map((region) => {
        const stat = stats[region.id];
        const pct = unlabeledPct(stat);
        const isOpen = expanded === region.id;
        const evalGrade = region.status === 'eval_grade';
        const members = isOpen ? majorityInstances(stat, instances) : [];
        return (
          <div key={region.id} className={'region-row' + (isOpen ? ' open' : '')}>
            <div className="region-row-hd" onClick={() => setExpanded(isOpen ? null : region.id)}>
              <span className="class-swatch" style={{ background: regionCssColor(region.status) }} />
              {editingName?.id === region.id ? (
                <input className="ins-input" autoFocus value={editingName.value}
                  onClick={(e) => e.stopPropagation()}
                  onChange={(e) => setEditingName({ id: region.id, value: e.target.value })}
                  onBlur={() => { run(region.id, onRename(region.id, editingName.value)); setEditingName(null); }}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') e.target.blur();
                    if (e.key === 'Escape') setEditingName(null);
                  }} />
              ) : (
                <span className="region-name"
                  title="Double-click to rename"
                  onDoubleClick={(e) => { e.stopPropagation(); setEditingName({ id: region.id, value: region.name }); }}>
                  {region.name}
                </span>
              )}
              <button className="ghost-btn" title={eyes.has(region.id) ? 'Hide overlay' : 'Show overlay while other tools are active'}
                onClick={(e) => { e.stopPropagation(); onToggleEye(region.id); }}>
                {eyes.has(region.id) ? '●' : '◌'}
              </button>
              {!evalGrade && (
                <button className="ghost-btn danger" title="Delete region"
                  onClick={(e) => {
                    e.stopPropagation();
                    // Regions are not on the undo stack — a mis-click would
                    // silently destroy a hand-traced footprint.
                    if (!window.confirm(`Delete region "${region.name}"? This cannot be undone.`)) return;
                    run(region.id, onDelete(region.id));
                  }}>×</button>
              )}
            </div>
            {/* Status/coverage badges share the second line with the flip
                button — the header line is too narrow for them without
                shrinking the name to a few characters. Flip actions are
                ALWAYS visible (not behind expansion) — the jsdom test clicks
                them on two different rows in one test. */}
            <div className="region-actions">
              <span className="badge-soft">{evalGrade ? 'eval-grade' : 'draft'}</span>
              {pct != null && <span className="badge-soft">{Math.round(pct)}% unlabeled</span>}
              {evalGrade ? (
                <button className="ghost-btn"
                  onClick={() => run(region.id, onFlipStatus(region.id, 'draft'))}>Back to draft</button>
              ) : (
                <button className="ghost-btn"
                  onClick={() => run(region.id, onFlipStatus(region.id, 'eval_grade'))}>Mark eval-grade</button>
              )}
              {rowError?.id === region.id && <p className="tool-opt-hint danger">{rowError.message}</p>}
            </div>
            {isOpen && (
              <div className="region-detail">
                {evalGrade && region.accuracy && (
                  <p className="tool-opt-hint">
                    p50 {(region.accuracy.p50 * 1000).toFixed(1)} mm · p90 {(region.accuracy.p90 * 1000).toFixed(1)} mm · {region.accuracy.loa} — geometry locked
                  </p>
                )}
                <div className="region-members">
                  {members.length === 0 && <p className="tool-opt-hint">No confirmed instance is majority-inside this region.</p>}
                  {members.map(({ inst, frac }) => {
                    const cls = classes.find((c) => c.id === inst.cls);
                    return (
                      <div key={inst.id} className="inst-row region-member" onClick={() => onSelectInstance(inst)}>
                        <span className="class-swatch" style={{ background: cls?.color }} />
                        <span className="region-member-name">{inst.label || cls?.label || inst.cls}</span>
                        <span className="badge-soft">{Math.round(frac * 100)} %</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
