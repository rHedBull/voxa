// draw-mode.jsx — Draw (centerline) sub-mode of Label mode. Pipes/tanks are
// labeled by drawing centerline paths; the backend extracts points within a
// tube radius. State machine in draw-paths.js; spec in
// docs/superpowers/specs/2026-06-04-centerline-pipe-labeling-design.md.

import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { VoxaAPI } from './api.js';
import { applyDelta } from './segment-state.js';
import {
  initDrawState, addPoint, movePoint, removeLastPoint, endActive,
  selectPath, clearSelection, setRadius, nudgeRadius, setClass, toggleSmooth,
  deleteSelected, mergeSelection, buildApplyCalls, markApplied, seedFromServer,
} from './draw-paths.js';

// Capture-phase keyboard driver (same trick as FastLabelKeys: beat the
// LabelMode global keydown). classes[i] ↔ palette index i.
export function DrawKeys({ active, classes, onKey }) {
  useEffect(() => {
    if (!active) return undefined;
    const handler = (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;   // leave Ctrl+S/Z/click alone
      const clsIdx = classes.findIndex((c) => c.hotkey === e.key);
      let handled = true;
      if (clsIdx >= 0) onKey({ type: 'class', clsIdx });
      else if (e.key === 'Enter') onKey({ type: 'apply' });
      else if (e.key === 'Escape') onKey({ type: 'escape' });
      else if (e.key === 'Backspace' || e.key === 'Delete') onKey({ type: 'backspace' });
      else if (e.key === 'm' || e.key === 'M') onKey({ type: 'merge' });
      else if (e.key === 'c' || e.key === 'C') onKey({ type: 'smooth' });
      else if (e.key === '+' || e.key === '=') onKey({ type: 'radius', dir: +1 });
      else if (e.key === '-' || e.key === '_') onKey({ type: 'radius', dir: -1 });
      else handled = false;
      if (handled) { e.preventDefault(); e.stopPropagation(); }
    };
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [active, classes, onKey]);
  return null;
}

function DrawHUD({ state, classes, toast }) {
  const drawing = !!state.active;
  const nSel = state.selection.size;
  return (
    <div style={{
      position: 'fixed', bottom: 16, left: '50%', transform: 'translateX(-50%)',
      background: 'rgba(17, 24, 39, 0.92)', color: '#e5e7eb', borderRadius: 8,
      padding: '8px 14px', fontSize: 12, display: 'flex', gap: 14,
      alignItems: 'center', pointerEvents: 'none', zIndex: 30,
      border: '1px solid rgba(96,165,250,0.5)',
    }}>
      {toast ? <b style={{ color: '#fbbf24' }}>{toast}</b> : (
        <>
          <b style={{ color: '#60a5fa' }}>
            {drawing ? 'Drawing path…' : nSel ? `${nSel} path${nSel > 1 ? 's' : ''} selected` : 'Draw centerlines'}
          </b>
          <span style={{ opacity: 0.65 }}>
            {drawing
              ? 'Ctrl+click add · ⌫ undo pt · Esc end · Enter end+apply'
              : 'Ctrl+click start · click tube select · scroll/± radius · C smooth · M merge · Enter apply · Esc exit'}
          </span>
        </>
      )}
    </div>
  );
}

// Stub replaced in the next task.
function DrawOverlay() { return null; }

export default function DrawMode({
  viewerRef, classes, segState, setSegState, onExit,
}) {
  const [draw, setDraw] = useState(() => initDrawState());
  const [defaultClsIdx, setDefaultClsIdx] = useState(0);
  const [toast, setToast] = useState(null);
  const toastTimer = useRef(null);
  const showToast = useCallback((msg) => {
    clearTimeout(toastTimer.current);
    setToast(msg);
    toastTimer.current = setTimeout(() => setToast(null), 2500);
  }, []);

  // Load stored centerlines once on open so applied paths render + re-edit.
  useEffect(() => {
    let gone = false;
    VoxaAPI.getCenterlines()
      .then((doc) => { if (!gone) setDraw((s) => seedFromServer(s, doc)); })
      .catch((err) => { if (!gone) showToast(`centerlines load failed: ${err.message}`); });
    return () => { gone = true; };
  }, [showToast]);

  const applySelection = useCallback(async () => {
    // Enter while drawing = end + apply that path (spec shortcut).
    let s = draw;
    if (s.active) {
      const key = s.active;
      s = endActive(s);
      if (s.paths.some((p) => p.key === key)) s = selectPath(s, key);
    }
    const calls = buildApplyCalls(s);
    if (calls.length === 0) { setDraw(s); return; }
    for (const call of calls) {
      let r;
      try {
        r = await VoxaAPI.centerlineApply({
          paths: call.paths,
          targetClass: call.classId,
          targetInst: call.targetInst,
          mergedFrom: call.mergedFrom,
        });
      } catch (err) {
        showToast(`apply failed: ${err.message}`);
        continue;                       // surface and move on; state unchanged for this group
      }
      if (r.nAffected === 0) {
        showToast('no points in tube');
        continue;
      }
      s = markApplied(s, call.instKey, r.instanceId);
      setSegState((st) => st ? applyDelta(st, {
        indices: r.indices,
        after_class: r.afterClass,
        after_instance: r.afterInstance,
      }) : st);
    }
    setDraw(clearSelection(s));
  }, [draw, setSegState, showToast]);

  const onKey = useCallback((action) => {
    switch (action.type) {
      case 'class':
        setDefaultClsIdx(action.clsIdx);
        setDraw((s) => setClass(s, action.clsIdx));
        break;
      case 'apply':
        applySelection();
        break;
      case 'escape':
        setDraw((s) => {
          if (s.active) return endActive(s);
          if (s.selection.size) return clearSelection(s);
          onExit();
          return s;
        });
        break;
      case 'backspace':
        setDraw((s) => s.active ? removeLastPoint(s) : deleteSelected(s));
        break;
      case 'merge':
        setDraw((s) => mergeSelection(s));
        break;
      case 'smooth':
        setDraw((s) => toggleSmooth(s));
        break;
      case 'radius':
        setDraw((s) => nudgeRadius(s, action.dir));
        break;
      default:
    }
  }, [applySelection, onExit]);

  return (
    <>
      <DrawKeys active classes={classes} onKey={onKey} />
      <DrawHUD state={draw} classes={classes} toast={toast} />
      <DrawOverlay
        viewerRef={viewerRef}
        draw={draw}
        setDraw={setDraw}
        classes={classes}
        defaultClsIdx={defaultClsIdx}
      />
      <DrawPanel
        draw={draw}
        setDraw={setDraw}
        classes={classes}
        onApply={applySelection}
      />
    </>
  );
}

// Side-panel section: path list + radius field + actions. Rendered by
// LabelMode inside the left sidebar (portal-free: this component returns
// plain divs; LabelMode places it).
export function DrawPanel({ draw, setDraw, classes, onApply }) {
  const selected = draw.paths.filter((p) => draw.selection.has(p.key));
  const radiusValue = selected[0]?.radius
    ?? draw.paths.find((p) => p.key === draw.active)?.radius
    ?? draw.lastRadius;
  return (
    <div className="draw-panel" style={{ marginTop: 10 }}>
      <div className="side-hd"><span>Centerline paths</span>
        <span className="badge-soft">{draw.paths.length}</span></div>
      <div className="ins-row">
        <label>Radius</label>
        <input className="ins-input" type="number" step="0.01" min="0.005"
          value={Number(radiusValue.toFixed(4))}
          onChange={(e) => {
            const v = parseFloat(e.target.value);
            if (Number.isFinite(v) && v > 0) setDraw((s) => setRadius(s, v));
          }} />
      </div>
      <div style={{ maxHeight: 180, overflowY: 'auto' }}>
        {draw.paths.map((p) => {
          const cls = classes[p.classId];
          const applied = draw.instanceIds[p.instKey] != null;
          const isSel = draw.selection.has(p.key);
          return (
            <div key={p.key}
              className={'inst-row' + (isSel ? ' selected' : '')}
              onClick={(e) => setDraw((s) =>
                selectPath(s, p.key, { additive: e.shiftKey }))}>
              <span className="inst-dot" style={{ background: cls?.color }} />
              <div className="inst-text">
                <b>{cls?.label || '?'} {applied ? `#${draw.instanceIds[p.instKey]}` : '(staged)'}</b>
                <em>{p.points.length} pts · r={p.radius.toFixed(3)}{p.smooth ? ' · smooth' : ''}</em>
              </div>
            </div>
          );
        })}
      </div>
      <div className="ins-actions">
        <button className="ghost-btn" disabled={draw.selection.size < 2}
          onClick={() => setDraw((s) => mergeSelection(s))}>M Merge</button>
        <button className="ghost-btn" disabled={!draw.selection.size && !draw.active}
          onClick={onApply}>↵ Apply</button>
      </div>
    </div>
  );
}
