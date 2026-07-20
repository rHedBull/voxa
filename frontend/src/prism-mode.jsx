// prism-mode.jsx — Prism sub-mode of Label mode. An arbitrary object is
// selected by drawing a footprint polygon on a horizontal plane at the first
// click's height (`y0`), then extruding it UP to `y0 + height`. The enclosed
// points are classified through the shared apply-shape pipeline, producing a
// kind:'pointset' instance (no gizmo). Mirrors beam-mode.jsx's three-part
// shape: PrismKeys (capture-phase keys) + PrismOverlay (Three.js overlay +
// pointer/scroll) + PrismPanel (sidebar). Backend resolves the shape via
// {type:'prism', polygon:[[x,z],...], y0, height} (backend/labeling/shapes.py).

import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { evtToNdc } from './viewer.jsx';
import { VoxaAPI } from './api.js';
import { applyDelta } from './segment-state.js';

const EMPTY_PRISM = { vertices: [], y0: null, closed: false, height: 2.0 };
const MIN_HEIGHT = 0.05;
const HEIGHT_STEP = 0.1;     // metres per wheel notch
const CLOSE_PX = 12;         // screen-space radius to click-close on the first vertex
const ACCENT = 0x38bdf8;     // sky-400 — footprint + extrusion preview

// Capture-phase keyboard driver (same trick as BeamKeys / DrawKeys). Attaches
// in the capture phase so it runs before mode-label.jsx's handler, which
// early-returns while a sub-mode owns input.
function PrismKeys({ prism, setPrism, classes, defaultClassId, onApply, onExit }) {
  useEffect(() => {
    const handler = (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      const canClose = prism.vertices.length >= 3 && !prism.closed;
      // Ctrl+Enter: close if still drawing, otherwise apply. Every other
      // Ctrl/Meta combo (Ctrl+S/Z…) passes through untouched.
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        if (prism.closed) { e.preventDefault(); e.stopPropagation(); onApply(defaultClassId); }
        else if (canClose) { e.preventDefault(); e.stopPropagation(); setPrism((s) => ({ ...s, closed: true })); }
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      let handled = true;
      if (e.key === 'Enter') {
        if (prism.closed) onApply(defaultClassId);
        else if (canClose) setPrism((s) => ({ ...s, closed: true }));
        else handled = false;
      } else if (e.key === 'Escape') {
        if (prism.vertices.length > 0) setPrism(EMPTY_PRISM);
        else onExit();
      } else if (e.key === 'Backspace' || e.key === 'Delete') {
        if (!prism.closed && prism.vertices.length > 0) {
          setPrism((s) => {
            const vertices = s.vertices.slice(0, -1);
            return { ...s, vertices, y0: vertices.length ? s.y0 : null };
          });
        } else handled = false;
      } else {
        // Class hotkey only applies once closed (mirrors beam: a hotkey while
        // still placing corners would apply an incomplete polygon).
        const cls = prism.closed ? classes.find((c) => c.hotkey === e.key) : null;
        if (cls) onApply(cls.class_id);
        else handled = false;
      }
      if (handled) { e.preventDefault(); e.stopPropagation(); }
    };
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [prism, classes, defaultClassId, onApply, onExit, setPrism]);
  return null;
}

// Bounding-box-scaled vertex marker radius so the corner spheres read at any
// scene scale (scenes range from ~unit to tens of metres). Falls back to a
// fixed radius before the polygon has any extent.
function markerRadius(vertices) {
  if (vertices.length < 2) return 0.05;
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const [x, z] of vertices) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const diag = Math.hypot(maxX - minX, maxZ - minZ);
  return Math.min(0.5, Math.max(0.02, diag * 0.02));
}

function PrismOverlay({ viewerRef, prism, setPrism }) {
  const layerRef = useRef(null);        // { group, remove }
  const prismRef = useRef(prism);
  prismRef.current = prism;

  // One overlay group for the lifetime of the sub-mode.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v?.attachOverlayGroup) return undefined;
    layerRef.current = v.attachOverlayGroup();
    return () => { layerRef.current?.remove(); layerRef.current = null; };
  }, [viewerRef]);

  // Rebuild overlay children whenever the prism changes. The polygon is tiny
  // (a handful of corners) so dispose-and-rebuild beats bookkeeping.
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer?.group) return;
    const group = layer.group;
    while (group.children.length) {
      const c = group.children.pop();
      c.geometry?.dispose?.(); c.material?.dispose?.();
      group.remove(c);
    }
    const { vertices, y0, closed, height } = prism;
    if (y0 == null || vertices.length === 0) return;

    const lineMat = (opacity) => new THREE.LineBasicMaterial({
      color: ACCENT, transparent: true, opacity, depthWrite: false,
    });
    const bottom = vertices.map(([x, z]) => new THREE.Vector3(x, y0, z));

    // Footprint: LineLoop once closed, open Line while placing corners.
    if (bottom.length >= 2) {
      const geo = new THREE.BufferGeometry().setFromPoints(bottom);
      const line = closed
        ? new THREE.LineLoop(geo, lineMat(0.95))
        : new THREE.Line(geo, lineMat(0.95));
      line.raycast = () => {};
      group.add(line);
    }

    // Corner markers.
    const r = markerRadius(vertices);
    for (const bp of bottom) {
      const sph = new THREE.Mesh(
        new THREE.SphereGeometry(r, 10, 8),
        new THREE.MeshBasicMaterial({ color: ACCENT }));
      sph.raycast = () => {};
      sph.position.copy(bp);
      group.add(sph);
    }

    // Extrusion preview: the top polygon + the vertical edges linking bottom↔top.
    if (closed) {
      const yTop = y0 + height;
      const top = vertices.map(([x, z]) => new THREE.Vector3(x, yTop, z));
      const topLine = new THREE.LineLoop(
        new THREE.BufferGeometry().setFromPoints(top), lineMat(0.95));
      topLine.raycast = () => {};
      group.add(topLine);

      const seg = [];
      for (let i = 0; i < vertices.length; i++) { seg.push(bottom[i], top[i]); }
      const vLine = new THREE.LineSegments(
        new THREE.BufferGeometry().setFromPoints(seg), lineMat(0.5));
      vLine.raycast = () => {};
      group.add(vLine);
    }
  }, [prism]);

  // Pointer (place/close corners) + wheel (adjust height once closed).
  useEffect(() => {
    const v = viewerRef.current;
    const dom = v?.domElement?.();
    if (!dom) return undefined;
    const raycaster = new THREE.Raycaster();

    const onPointerDown = (evt) => {
      if (evt.button !== 0) return;
      const p = prismRef.current;
      if (p.closed) return;                     // closed → no more corners
      const camera = v.getCamera();
      if (!camera) return;
      const rect = dom.getBoundingClientRect();
      const ndc = evtToNdc(evt, rect);

      // The first corner seeds the base plane from the point under the cursor;
      // without a surface hit there is no height to anchor the footprint to.
      let y0 = p.y0;
      if (p.vertices.length === 0) {
        const hit = v.firstHitUnderCursor(evt);
        if (!hit) return;
        y0 = hit.world.y;
      }

      // Click near the first corner (≥3 placed) closes the polygon.
      if (p.vertices.length >= 3) {
        const first = new THREE.Vector3(p.vertices[0][0], y0, p.vertices[0][1]).project(camera);
        const sx = (first.x * 0.5 + 0.5) * rect.width;
        const sy = (-first.y * 0.5 + 0.5) * rect.height;
        if (Math.hypot(sx - (evt.clientX - rect.left), sy - (evt.clientY - rect.top)) < CLOSE_PX) {
          setPrism((s) => ({ ...s, closed: true }));
          v.setOrbitEnabled(true);
          evt.stopPropagation();
          return;
        }
      }

      // Every corner (incl. the first) lands where the camera ray meets the
      // horizontal base plane y = y0.
      raycaster.setFromCamera(ndc, camera);
      const out = new THREE.Vector3();
      const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -y0);
      if (!raycaster.ray.intersectPlane(plane, out)) return;

      // Suppress orbit while drawing so a drag places corners instead of
      // spinning the camera; re-enabled on close and on unmount.
      v.setOrbitEnabled(false);
      setPrism((s) => ({
        ...s,
        y0: s.y0 == null ? y0 : s.y0,
        vertices: [...s.vertices, [out.x, out.z]],
      }));
      evt.stopPropagation();
    };

    // Wheel-resize height beats orbit-zoom via a CAPTURE listener on the parent
    // (same trick beam/draw document). Only active once the polygon is closed.
    const wheelHost = dom.parentElement || dom;
    const onWheel = (evt) => {
      if (!prismRef.current.closed) return;
      evt.preventDefault();
      evt.stopPropagation();
      setPrism((s) => ({
        ...s,
        height: Math.max(MIN_HEIGHT, s.height + Math.sign(evt.deltaY) * HEIGHT_STEP),
      }));
    };

    dom.addEventListener('pointerdown', onPointerDown, true);
    wheelHost.addEventListener('wheel', onWheel, { capture: true, passive: false });
    return () => {
      dom.removeEventListener('pointerdown', onPointerDown, true);
      wheelHost.removeEventListener('wheel', onWheel, { capture: true });
      v.setOrbitEnabled(true);
    };
  }, [viewerRef, setPrism]);

  return null;
}

// Side-panel section (rendered by LabelMode inside the left sidebar, like
// BeamPanel/BoxOptions).
function PrismPanel({ prism, setPrism, onClear, onApply }) {
  const drawing = prism.vertices.length > 0 && !prism.closed;
  return (
    <div className="tool-options tool-options-prism">
      {prism.vertices.length === 0 && (
        <p className="tool-opt-hint">
          Click a point to start the footprint. Each click adds a corner on the base plane.
        </p>
      )}
      {drawing && (
        <p className="tool-opt-hint">
          {prism.vertices.length} corner{prism.vertices.length === 1 ? '' : 's'} · Enter or click
          the first corner to close · Backspace undo · Esc cancel
        </p>
      )}
      {prism.closed && (
        <>
          <div className="ins-row">
            <label>Height</label>
            <input className="ins-input" type="number" step="0.05" min={MIN_HEIGHT}
              value={Number(prism.height.toFixed(3))}
              onChange={(e) => {
                const v = parseFloat(e.target.value);
                if (Number.isFinite(v)) setPrism((s) => ({ ...s, height: Math.max(MIN_HEIGHT, v) }));
              }} />
          </div>
          <p className="tool-opt-hint">Scroll in the viewport to adjust height.</p>
          <div className="tool-opt-toggle">
            <button onClick={onClear}>Clear</button>
            <button className="active" onClick={onApply}>Apply (Ctrl+Enter)</button>
          </div>
        </>
      )}
    </div>
  );
}

export default function PrismMode({
  viewerRef, classes, setSegState, onExit, defaultClassId,
  onClassChange, onApplied, protectInstances = [],
}) {
  const [prism, setPrism] = useState(EMPTY_PRISM);
  const prismRef = useRef(prism);
  prismRef.current = prism;
  // Latest confirmed-lock set, read at apply time (async), not closure-stale.
  const protectInstancesRef = useRef(protectInstances);
  protectInstancesRef.current = protectInstances;
  const reset = useCallback(() => setPrism(EMPTY_PRISM), []);

  const applyPrism = useCallback(async (classId) => {
    const p = prismRef.current;
    if (!p.closed || p.vertices.length < 3) return;
    const cls = classes.find((c) => c.class_id === classId);
    if (!cls) return;
    const shape = { type: 'prism', polygon: p.vertices, y0: p.y0, height: p.height };
    let r;
    try {
      r = await VoxaAPI.applyShape({
        shape,
        targetClass: cls.id,
        protectInstances: protectInstancesRef.current,
      });
    } catch (err) { console.error('prism apply failed:', err); return; }
    // Empty prism: no delta when the volume encloses zero full-res points
    // (parity with Box/Draw). Keep the polygon so the user can adjust it.
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
        prism: { polygon: p.vertices.map((pt) => [...pt]), y0: p.y0, height: p.height },
      });
    }
    setSegState((s) => (s ? {
      ...applyDelta(s, {
        indices: r.indices, after_class: r.afterClass, after_instance: r.afterInstance,
      }),
      selection: new Set(),
    } : s));
    reset();
  }, [classes, onApplied, setSegState, reset]);

  return (
    <>
      <PrismKeys prism={prism} setPrism={setPrism} classes={classes}
        defaultClassId={defaultClassId} onApply={applyPrism} onExit={onExit} />
      <PrismOverlay viewerRef={viewerRef} prism={prism} setPrism={setPrism} />
      <PrismPanel prism={prism} setPrism={setPrism} onClear={reset}
        onApply={() => applyPrism(defaultClassId)} />
    </>
  );
}
