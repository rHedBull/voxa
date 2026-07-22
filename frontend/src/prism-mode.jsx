// prism-mode.jsx — Prism sub-mode of Label mode. An arbitrary object is
// selected by tracing a footprint whose corners SNAP to the cloud surface,
// then aiming an extrusion height, then classifying through the shared
// apply-shape pipeline (kind:'pointset' instance, no gizmo).
//
// The interaction mirrors engine/product/demo's Measure tools: the footprint
// is the Surface tool (click to snap a corner, dashed rubber-band to the
// cursor, double-click/Enter to close) and the height is the Volume tool (a
// vertical plane through the camera-nearest footprint edge; mouse aims the
// height, a click commits it). The earlier fixed-plane + scroll model read as
// disconnected from the geometry — see docs/superpowers/specs/2026-07-20-...
//
// The emitted shape is unchanged: {type:'prism', polygon:[[x,z],...], y0, height}
// (backend/labeling/shapes.py). prism-geom.js::prismShapeFromCorners maps the
// snapped corners + aimed top Y to it (XZ projection, y0 = min corner Y,
// signed-height normalization).

import { useCallback, useEffect, useRef, useState } from 'react';
import { VoxaAPI } from './api.js';
import { applyDelta } from './segment-state.js';
import { prismShapeFromCorners, footprintBaseY } from './prism-geom.js';
import { ClassPickerModal } from './class-picker.jsx';
import { chordStep } from './class-chords.js';
import { ChordOverlay } from './chord-overlay.jsx';
import {
  EMPTY_PRISM, liveHeight, nearestEdge,
  PrismOverlay, PrismRubberBand,
} from './prism-draw.jsx';

// Capture-phase keyboard driver (like BeamKeys) — runs before mode-label.jsx's
// handler, which early-returns while a sub-mode owns input.
function PrismKeys({ prism, setPrism, classes, pickerOpen, onApply, onOpenPicker, onExit, onClose, onBackToFootprint }) {
  const prismRef = useRef(prism);
  prismRef.current = prism;
  // Two-stroke chord state (class-chords.js); chording engages only once the
  // volume is ready (height committed) — digits stay inert before that.
  const [pendingGroup, setPendingGroup] = useState(null);
  useEffect(() => {
    const handler = (e) => {
      // While the class picker is open it owns the keyboard (hotkey → pick,
      // Esc → close). Standing down here avoids double-handling the same key.
      if (pickerOpen) return;
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      const p = prismRef.current;
      const canClose = p.phase === 'footprint' && p.corners.length >= 3;
      const ready = p.phase === 'height' && p.committed;

      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (ready) { e.preventDefault(); e.stopPropagation(); onOpenPicker(); }
        else if (canClose) { e.preventDefault(); e.stopPropagation(); onClose(); }
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;

      let handled = true;
      if (e.key === 'Enter') {
        if (ready) onOpenPicker();
        else if (canClose) onClose();
        else handled = false;
      } else if (e.key === 'Escape') {
        if (pendingGroup) setPendingGroup(null);   // cancel the chord first
        else if (p.corners.length > 0) setPrism(EMPTY_PRISM);
        else onExit();
      } else if (e.key === 'Backspace' || e.key === 'Delete') {
        if (p.phase === 'height' && p.committed) setPrism((s) => ({ ...s, committed: false }));  // re-aim
        else if (p.phase === 'height') onBackToFootprint();                                       // edit footprint
        else if (p.corners.length > 0) setPrism((s) => ({ ...s, corners: s.corners.slice(0, -1) }));
        else handled = false;
      } else {
        // Class chord applies only once the height is committed.
        const step = ready ? chordStep(pendingGroup, e.key, classes) : { type: 'pass' };
        if (step.type === 'group') setPendingGroup(step.group);
        else if (step.type === 'class') { setPendingGroup(null); onApply(step.cls.class_id); }
        else if (step.type === 'cancel') setPendingGroup(null);
        else handled = false;
      }
      if (handled) { e.preventDefault(); e.stopPropagation(); }
    };
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [classes, pickerOpen, onApply, onOpenPicker, onExit, onClose, onBackToFootprint, setPrism, pendingGroup]);
  return pendingGroup ? <ChordOverlay group={pendingGroup} classes={classes} /> : null;
}

function PrismPanel({ prism, onClear, onApply }) {
  const placing = prism.phase === 'footprint';
  const aiming = prism.phase === 'height' && !prism.committed;
  const ready = prism.phase === 'height' && prism.committed;
  const h = liveHeight(prism);
  return (
    <div className="tool-options tool-options-prism">
      {placing && prism.corners.length === 0 && (
        <p className="tool-opt-hint">Click points on the surface to trace a footprint. Double-click or Enter to close (min 3).</p>
      )}
      {placing && prism.corners.length > 0 && (
        <p className="tool-opt-hint">
          {prism.corners.length} corner{prism.corners.length === 1 ? '' : 's'} · double-click or Enter to close · Backspace undo · Esc cancel
        </p>
      )}
      {aiming && (
        <p className="tool-opt-hint">Move the mouse to set the height, then click to commit. Height: {h.toFixed(2)} m</p>
      )}
      {ready && (
        <>
          <div className="ins-row">
            <label>Height</label>
            <div className="ins-input" style={{ display: 'flex', alignItems: 'center' }}>{h.toFixed(2)} m</div>
          </div>
          <p className="tool-opt-hint">Press a class key to label, or Ctrl+Enter / the button to pick a class · Backspace to re-aim · Esc to cancel</p>
          <div className="tool-opt-toggle">
            <button onClick={onClear}>Clear</button>
            <button className="active" onClick={onApply}>Pick class… (Ctrl+Enter)</button>
          </div>
        </>
      )}
    </div>
  );
}

export default function PrismMode({
  viewerRef, classes, counts, setSegState, onExit,
  onApplied, protectInstances = [],
}) {
  const [prism, setPrism] = useState(EMPTY_PRISM);
  const [pickerOpen, setPickerOpen] = useState(false);
  const prismRef = useRef(prism);
  prismRef.current = prism;
  const protectInstancesRef = useRef(protectInstances);
  protectInstancesRef.current = protectInstances;
  const reset = useCallback(() => { setPickerOpen(false); setPrism(EMPTY_PRISM); }, []);

  // Classify goes through the shared class picker (same modal as Box) so the
  // user chooses the label — the old default-class apply silently produced
  // "Exclude / Review". Only valid once the height is committed.
  const requestClassify = useCallback(() => {
    const p = prismRef.current;
    if (p.phase === 'height' && p.committed) setPickerOpen(true);
  }, []);

  // Close the footprint → enter the height stage. `dropLast` drops the extra
  // corner a closing double-click added. Fixes the base plane + the height-aim
  // edge from the current camera.
  const closeFootprint = useCallback((dropLast = false) => {
    const camera = viewerRef.current?.getCamera?.();
    if (!camera) return;
    setPrism((s) => {
      if (s.phase !== 'footprint') return s;
      const corners = dropLast ? s.corners.slice(0, -1) : s.corners;
      if (corners.length < 3) return { ...s, corners };
      const baseY = footprintBaseY(corners);
      return { phase: 'height', corners, baseY, heightEdge: nearestEdge(corners, baseY, camera), topY: baseY, committed: false };
    });
  }, [viewerRef]);

  // Stable identity so PrismKeys' keydown effect doesn't re-subscribe every
  // render — PrismMode re-renders on every pointermove during the height aim.
  const closeNow = useCallback(() => closeFootprint(false), [closeFootprint]);

  const backToFootprint = useCallback(() => {
    setPrism((s) => (s.phase === 'height'
      ? { ...EMPTY_PRISM, phase: 'footprint', corners: s.corners } : s));
  }, []);

  const applyPrism = useCallback(async (classId) => {
    const p = prismRef.current;
    if (!(p.phase === 'height' && p.committed)) return;
    const shape = prismShapeFromCorners(p.corners, p.topY);
    if (!shape) return;
    const cls = classes.find((c) => c.class_id === classId);
    if (!cls) return;
    let r;
    try {
      r = await VoxaAPI.applyShape({
        shape: { type: 'prism', ...shape },
        targetClass: cls.id,
        protectInstances: protectInstancesRef.current,
      });
    } catch (err) { console.error('prism apply failed:', err); return; }
    if (!r.indices || r.nAffected === 0) {
      console.warn(r.nProtected > 0
        ? `prism apply: ${r.nProtected} point(s) skipped — inside a confirmed instance (un-confirm to re-label)`
        : 'prism apply: no points inside the prism');
      return;
    }
    const segId = Number.isFinite(r.instanceId) ? r.instanceId : -1;
    if (segId >= 0) {
      onApplied?.({
        instanceId: segId, classId: cls.class_id, source: 'prism',
        prism: { polygon: shape.polygon.map((pt) => [...pt]), y0: shape.y0, height: shape.height },
      });
    }
    setSegState((s) => (s ? {
      ...applyDelta(s, { indices: r.indices, after_class: r.afterClass, after_instance: r.afterInstance }),
      selection: new Set(),
    } : s));
    reset();
  }, [classes, onApplied, setSegState, reset]);

  return (
    <>
      <PrismKeys prism={prism} setPrism={setPrism} classes={classes} pickerOpen={pickerOpen}
        onApply={applyPrism} onOpenPicker={requestClassify} onExit={onExit}
        onClose={closeNow} onBackToFootprint={backToFootprint} />
      <PrismOverlay viewerRef={viewerRef} prism={prism} setPrism={setPrism} onClose={closeFootprint} />
      <PrismRubberBand viewerRef={viewerRef} prism={prism} />
      <PrismPanel prism={prism} onClear={reset} onApply={requestClassify} />
      {pickerOpen && (
        <ClassPickerModal classes={classes} counts={counts}
          onPick={(cls) => { setPickerOpen(false); applyPrism(cls.class_id); }}
          onClose={() => setPickerOpen(false)} />
      )}
    </>
  );
}
