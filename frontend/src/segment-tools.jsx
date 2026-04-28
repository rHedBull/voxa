// segment-tools.jsx — tool strip + Pick tool + Brush tool for per-point segment editing.

import { useEffect, useRef, useMemo } from 'react';
import * as THREE from 'three';
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

// ── Brush tool ──────────────────────────────────────────────────────────────
// Headless component — returns null. Mounts a sphere gizmo in the viewer,
// follows the cursor, and paints per-point reassignments on left-drag.
export function BrushTool({ viewerRef, segState, classes, activeClassId, onApply }) {
  const radiusRef = useRef(segState?.brush?.radius ?? 0.05);
  const gizmoMeshRef = useRef(null);
  const strokeRef = useRef(null); // { indices: Set<number>, altHeld: bool } | null

  const activeClassColor = useMemo(() => {
    const cls = classes.find((c) => c.id === activeClassId);
    return cls?.color || '#ffffff';
  }, [classes, activeClassId]);

  // Mount gizmo on activate; remove on unmount or color change.
  useEffect(() => {
    const viewer = viewerRef?.current;
    if (!viewer?.attachBrushGizmo) return;
    const { remove, mesh } = viewer.attachBrushGizmo({
      radius: radiusRef.current,
      color: activeClassColor,
    });
    gizmoMeshRef.current = mesh;
    return () => {
      gizmoMeshRef.current = null;
      remove();
    };
  }, [viewerRef, activeClassColor]);

  // Cursor-follow: move gizmo to the hovered point.
  useEffect(() => {
    const viewer = viewerRef?.current;
    if (!viewer?.onPointerMove) return;
    const unsub = viewer.onPointerMove((_idx, evt) => {
      const mesh = gizmoMeshRef.current;
      if (!mesh) return;
      const hit = viewer.firstHitUnderCursor(evt);
      viewer.setBrushPosition(hit ? hit.world : null, mesh);
    });
    return unsub;
  }, [viewerRef]);

  // Wheel on the canvas adjusts brush radius (×1.2 / ÷1.2, clamped [0.005, 5.0]).
  useEffect(() => {
    const viewer = viewerRef?.current;
    if (!viewer?.domElement) return;
    const dom = viewer.domElement();
    if (!dom) return;
    const onWheel = (e) => {
      const mesh = gizmoMeshRef.current;
      if (!mesh) return;
      e.preventDefault();
      const factor = e.deltaY > 0 ? 1 / 1.2 : 1.2;
      radiusRef.current = Math.max(0.005, Math.min(5.0, radiusRef.current * factor));
      mesh.scale.setScalar(radiusRef.current);
    };
    dom.addEventListener('wheel', onWheel, { passive: false });
    return () => dom.removeEventListener('wheel', onWheel);
  }, [viewerRef]);

  // Stroke: pointerdown starts accumulation → pointermove samples brush-query
  // → pointerup commits one segApply call with all collected indices.
  useEffect(() => {
    const viewer = viewerRef?.current;
    if (!viewer?.domElement) return;
    const dom = viewer.domElement();
    if (!dom) return;

    const onDown = (e) => {
      if (e.button !== 0) return;
      strokeRef.current = { indices: new Set(), altHeld: e.altKey };
    };

    const onMove = async (e) => {
      if (!strokeRef.current) return;
      if (!(e.buttons & 1)) return;
      const hit = viewer.firstHitUnderCursor(e);
      if (!hit) return;
      const center = [hit.world.x, hit.world.y, hit.world.z];
      const fwd = viewer.cameraForward();
      try {
        const res = await VoxaAPI.segBrushQuery({
          center,
          radius: radiusRef.current,
          cameraRay: fwd,
          depthCull: radiusRef.current * 2,
        });
        if (!strokeRef.current) return;
        for (let i = 0; i < res.indices.length; i++) {
          strokeRef.current.indices.add(res.indices[i]);
        }
      } catch (_) {
        // Non-fatal — skip this sample.
      }
    };

    const onUp = async (e) => {
      if (e.button !== 0) return;
      const stroke = strokeRef.current;
      strokeRef.current = null;
      if (!stroke || stroke.indices.size === 0) return;

      const indices = new Int32Array(stroke.indices);
      let payload;
      if (stroke.altHeld) {
        payload = { target_inst: null, target_class: null };
      } else if (segState.selection.size === 1) {
        const targetInst = [...segState.selection][0];
        const targetClass = segState.summary.get(targetInst)?.classId ?? -1;
        payload = { target_inst: targetInst, target_class: targetClass };
      } else {
        payload = { target_inst: -1, target_class: activeClassId };
      }

      try {
        const r = await VoxaAPI.segApply('reassign', { indices, payload });
        onApply('__delta__', r);
      } catch (err) {
        console.error('brush segApply failed:', err);
      }
    };

    dom.addEventListener('pointerdown', onDown);
    dom.addEventListener('pointermove', onMove);
    dom.addEventListener('pointerup', onUp);
    return () => {
      dom.removeEventListener('pointerdown', onDown);
      dom.removeEventListener('pointermove', onMove);
      dom.removeEventListener('pointerup', onUp);
    };
  }, [viewerRef, segState, activeClassId, onApply]);

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
