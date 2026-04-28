// segment-tools.jsx — tool strip + Pick tool for per-point segment editing.

import { useEffect, useMemo } from 'react';
import { VoxaAPI } from './api.js';

// ── Tool strip ──────────────────────────────────────────────────────────────
// Three-button row: Cuboid / Pick / Brush.
// Pick and Brush are disabled when no segState is present (no segment data loaded).
export function SegmentToolStrip({ activeTool, onChange, hasSegState }) {
  const tools = [
    { id: 'cuboid', icon: '◫', label: 'Cuboid', hotkey: 'C' },
    { id: 'pick',   icon: '⊕', label: 'Pick',   hotkey: 'P' },
    { id: 'brush',  icon: '◉', label: 'Brush',   hotkey: 'B' },
  ];
  return (
    <div className="seg-toolstrip">
      {tools.map((t) => {
        const disabled = (t.id === 'pick' || t.id === 'brush') && !hasSegState;
        return (
          <button
            key={t.id}
            type="button"
            className={'tool-btn mini' + (activeTool === t.id ? ' active' : '')}
            disabled={disabled}
            title={t.label + `  (${t.hotkey})`}
            onClick={() => !disabled && onChange(t.id)}
          >
            <span className="tool-ico" aria-hidden>{t.icon}</span>
          </button>
        );
      })}
    </div>
  );
}

// ── Pick tool ───────────────────────────────────────────────────────────────
// Headless component — returns null. Registers pointer-pick callbacks on the
// viewer and routes hotkeys to segment operations.
export function PickTool({ viewerRef, segState, onApply, classes }) {
  // Map hotkey → class id, driven entirely by classes prop (never hardcoded labels).
  const keyToClassId = useMemo(() => Object.fromEntries(
    classes.filter((c) => c.hotkey).map((c) => [c.hotkey.toLowerCase(), c.id]),
  ), [classes]);

  // Subscribe to the viewer's pointer-pick stream.
  useEffect(() => {
    const viewer = viewerRef?.current;
    if (!viewer?.onPointerPick) return;
    const unsub = viewer.onPointerPick((fullIndex, evt) => {
      const instId = segState.instanceFull[fullIndex];
      if (instId < 0) return;
      // Shift = multi-select, plain click = replace selection with single segment.
      const next = new Set(segState.selection);
      if (evt.shiftKey) {
        next.has(instId) ? next.delete(instId) : next.add(instId);
      } else {
        if (next.size === 1 && next.has(instId)) next.clear();
        else { next.clear(); next.add(instId); }
      }
      // Propagate selection update via a synthetic "selection-only" path.
      // onApply handles actual mutations; selection is stored on segState.
      onApply('__select__', next);
    });
    return unsub;
  }, [viewerRef, segState, onApply]);

  // Hotkey routing.
  useEffect(() => {
    const onKey = (e) => {
      if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
      const k = e.key.toLowerCase();
      const sel = segState.selection;

      if (k === 't') {
        // T = detach selected segments into a new instance.
        if (sel.size === 0) return;
        const indices = collectIndices(segState.instanceFull, sel);
        onApply('detach', { indices, payload: {} });
      } else if (k === 'e') {
        // E = expand selection to include all points of each selected instance.
        // No-op here — expansion is implicit since selection is already by instance.
      } else if (k === 's') {
        // S = split: same as detach when a single segment is selected.
        if (sel.size !== 1) return;
        const indices = collectIndices(segState.instanceFull, sel);
        onApply('split', { indices, payload: {} });
      } else if (k === 'd') {
        // D = mark selected as unlabeled (class=-1, instance=-1).
        if (sel.size === 0) return;
        const indices = collectIndices(segState.instanceFull, sel);
        onApply('delete', { indices, payload: {} });
      } else if (k === 'm') {
        // M = merge selected instances into the lowest id.
        if (sel.size < 2) return;
        const targetInst = Math.min(...sel);
        const targetClass = segState.summary.get(targetInst)?.classId ?? -1;
        const indices = collectIndices(segState.instanceFull, sel);
        onApply('reassign', {
          indices,
          payload: { target_inst: targetInst, target_class: targetClass },
        });
      } else if (k === 'r') {
        // R = unlabeled sentinel: reassign to class=-1, instance=-1.
        if (sel.size === 0) return;
        const indices = collectIndices(segState.instanceFull, sel);
        onApply('reassign', {
          indices,
          payload: { target_inst: null, target_class: null },
        });
      } else {
        // Class hotkeys: reassign to that class, keep instances.
        const classId = keyToClassId[k];
        if (classId == null) return;
        if (sel.size === 0) return;
        const indices = collectIndices(segState.instanceFull, sel);
        onApply('reassign', {
          indices,
          payload: { target_inst: null, target_class: classId },
        });
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [segState, onApply, keyToClassId]);

  return null;
}

// ── Helper ──────────────────────────────────────────────────────────────────
// Returns an Int32Array of full-resolution point indices belonging to the
// given set of instance ids.
export function collectIndices(instanceFull, selection) {
  const out = [];
  for (let i = 0; i < instanceFull.length; i++) {
    if (selection.has(instanceFull[i])) out.push(i);
  }
  return new Int32Array(out);
}
