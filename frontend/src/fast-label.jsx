// fast-label.jsx — Fast labeling sub-mode of Label mode.
// Iterate preseg segments largest-first, classify each with one keypress and
// an Enter-confirm. Queue/navigation logic is exported as pure functions so it
// can be unit-tested; the components are thin views over mode-label's state.
// See docs/superpowers/specs/2026-06-04-fast-labeling-design.md.

import { useEffect, useState } from 'react';
import { chordStep } from './class-chords.js';
import { ChordOverlay } from './chord-overlay.jsx';

// Orange — distinct from the yellow click-selection overlay.
export const FAST_HIGHLIGHT_COLOR = 0xffa500;

// Queue of unpromoted preseg segments, largest first (id tiebreak for a
// stable order). `summary` is segState.summary (Map id -> {classId, nPoints}).
export function deriveFastQueue(summary, promotedSegIds) {
  if (!summary) return [];
  const out = [];
  for (const [id, info] of summary.entries()) {
    if (id < 0) continue;
    if (promotedSegIds && promotedSegIds.has(id)) continue;
    out.push({ id, classId: info.classId, nPoints: info.nPoints });
  }
  out.sort((a, b) => b.nPoints - a.nPoints || a.id - b.id);
  return out;
}

// Step with wrap-around. Clamps into range first so a queue that shrank
// (segments confirmed away) never strands the cursor out of bounds.
export function stepIndex(len, pos, delta) {
  if (len <= 0) return 0;
  const clamped = Math.min(Math.max(pos, 0), len - 1);
  return (clamped + (delta % len) + len) % len;
}

const NEXT_KEYS = new Set(['ArrowRight', 'ArrowDown', 'd', 'D', 's', 'S']);
const PREV_KEYS = new Set(['ArrowLeft', 'ArrowUp', 'a', 'A', 'w', 'W']);

// Capture-phase keyboard driver (same trick as ClassPickerModal: beat the
// LabelMode global keydown). Inert while the confirm modal is open — the
// modal owns the keyboard then.
export function FastLabelKeys({ active, classes, onStep, onPickClass, onExit }) {
  // Two-stroke chord state (class-chords.js). Escape with a pending group
  // cancels the chord, NOT rapid mode — so the chord check runs before the
  // exit branch.
  const [pendingGroup, setPendingGroup] = useState(null);
  useEffect(() => {
    if (!active) { setPendingGroup(null); return undefined; }
    const onKey = (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;   // leave Ctrl+S/Z alone
      const step = chordStep(pendingGroup, e.key, classes);
      if (step.type === 'group') {
        setPendingGroup(step.group);
      } else if (step.type === 'class') {
        setPendingGroup(null);
        onPickClass(step.cls);
      } else if (pendingGroup) {          // cancel: Esc/invalid second key
        setPendingGroup(null);
      } else if (NEXT_KEYS.has(e.key)) {
        onStep(1);
      } else if (PREV_KEYS.has(e.key)) {
        onStep(-1);
      } else if (e.key === 'Escape') {
        onExit();
      } else {
        return;
      }
      e.preventDefault();
      e.stopPropagation();
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [active, classes, onStep, onPickClass, onExit, pendingGroup]);
  return pendingGroup ? <ChordOverlay group={pendingGroup} classes={classes} /> : null;
}

// Bottom-center status strip while fast mode is active.
export function FastLabelHUD({ queue, pos, classes }) {
  const seg = queue[pos];
  // summary classIds are canonical numeric class ids (deriveSummary), -1 = unlabeled.
  const presegCls = seg && seg.classId >= 0
    ? classes.find((c) => c.class_id === seg.classId) : null;
  return (
    <div style={{
      position: 'fixed', bottom: 16, left: '50%', transform: 'translateX(-50%)',
      background: 'rgba(17, 24, 39, 0.92)', color: '#e5e7eb', borderRadius: 8,
      padding: '8px 14px', fontSize: 12, display: 'flex', gap: 14,
      alignItems: 'center', pointerEvents: 'none', zIndex: 30,
      border: '1px solid rgba(255,165,0,0.5)',
    }}>
      {queue.length === 0 ? (
        <b>Fast labeling — queue empty 🎉</b>
      ) : (
        <>
          <b style={{ color: '#fbbf24' }}>#{pos + 1} / {queue.length}</b>
          <span>seg {seg.id} · {seg.nPoints.toLocaleString()} pts
            {presegCls ? ` · preseg: ${presegCls.label}` : ''}</span>
          <span style={{ opacity: 0.65 }}>
            ←/→ · WASD move &nbsp;·&nbsp; 1–9 class &nbsp;·&nbsp; Esc exit
          </span>
        </>
      )}
    </div>
  );
}

// Confirm popup: Enter labels + confirms, Esc cancels back to the queue.
export function FastConfirmModal({ seg, cls, onConfirm, onCancel }) {
  useEffect(() => {
    const onKey = (e) => {
      if (e.key === 'Enter') {
        e.preventDefault(); e.stopPropagation();
        onConfirm();
      } else if (e.key === 'Escape') {
        e.preventDefault(); e.stopPropagation();
        onCancel();
      }
    };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [onConfirm, onCancel]);

  return (
    <div className="class-picker-overlay" onClick={onCancel}>
      <div className="class-picker-card" onClick={(e) => e.stopPropagation()}>
        <div className="class-picker-title">
          Label {seg.nPoints.toLocaleString()} pts as{' '}
          <span style={{ color: cls.color }}>{cls.label}</span>?
        </div>
        <div className="class-picker-hint">Enter to confirm · Esc to cancel</div>
      </div>
    </div>
  );
}
