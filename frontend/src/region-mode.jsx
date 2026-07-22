// region-mode.jsx — Region sub-mode of Label mode (eval-labeling phase 1).
// Drawing reuses the shared prism-draw machinery: trace a footprint, aim a
// height, and the commit click immediately POSTs a draft region — no class,
// no point selection, nothing on the undo stack (geometry, like Draw/Beam).
// RegionsOverlay renders the PERSISTED regions (status-colored translucent
// volumes); visibility = all while the Region tool is active, else the
// per-region eye set from the Regions tab.

import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { VoxaAPI } from './api.js';
import { prismShapeFromCorners, footprintBaseY } from './prism-geom.js';
import {
  EMPTY_PRISM, liveHeight, nearestEdge, fanTris, wallTris, fillMesh,
  PrismOverlay, PrismRubberBand,
} from './prism-draw.jsx';
import { REGION_COLORS } from './region-utils.js';

// Capture-phase keys like PrismKeys, minus chords/classify: Enter closes the
// footprint, Backspace steps back, Esc clears-then-exits.
function RegionKeys({ prism, setPrism, onExit, onClose }) {
  const prismRef = useRef(prism);
  prismRef.current = prism;
  useEffect(() => {
    const handler = (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const p = prismRef.current;
      const canClose = p.phase === 'footprint' && p.corners.length >= 3;
      let handled = true;
      if (e.key === 'Enter') {
        if (canClose) onClose(); else handled = false;
      } else if (e.key === 'Escape') {
        if (p.corners.length > 0) setPrism(EMPTY_PRISM); else onExit();
      } else if (e.key === 'Backspace' || e.key === 'Delete') {
        if (p.phase === 'height') {
          setPrism((s) => ({ ...EMPTY_PRISM, phase: 'footprint', corners: s.corners }));
        } else if (p.corners.length > 0) {
          setPrism((s) => ({ ...s, corners: s.corners.slice(0, -1) }));
        } else handled = false;
      } else handled = false;
      if (handled) { e.preventDefault(); e.stopPropagation(); }
    };
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [setPrism, onExit, onClose]);
  return null;
}

function RegionDrawPanel({ prism, onClear }) {
  const placing = prism.phase === 'footprint';
  const aiming = prism.phase === 'height' && !prism.committed;
  const h = liveHeight(prism);
  return (
    <div className="tool-options tool-options-region">
      {placing && prism.corners.length === 0 && (
        <p className="tool-opt-hint">Click points on the surface to trace a region footprint. Double-click or Enter to close (min 3).</p>
      )}
      {placing && prism.corners.length > 0 && (
        <p className="tool-opt-hint">
          {prism.corners.length} corner{prism.corners.length === 1 ? '' : 's'} · double-click or Enter to close · Backspace undo · Esc cancel
        </p>
      )}
      {aiming && (
        <p className="tool-opt-hint">
          Move the mouse to set the height, then click to create the region. Height: {h.toFixed(2)} m · Backspace to edit footprint
        </p>
      )}
      {prism.corners.length > 0 && (
        <div className="tool-opt-toggle"><button onClick={onClear}>Clear</button></div>
      )}
      <p className="tool-opt-hint">Regions appear in the Regions tab (right panel). Drawing labels nothing.</p>
    </div>
  );
}

export default function RegionMode({ viewerRef, onExit, onCreated }) {
  const [prism, setPrism] = useState(EMPTY_PRISM);
  const busyRef = useRef(false);

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
  const closeNow = useCallback(() => closeFootprint(false), [closeFootprint]);

  // The commit click (PrismOverlay sets committed:true) IS the create action.
  useEffect(() => {
    if (!(prism.phase === 'height' && prism.committed) || busyRef.current) return;
    const shape = prismShapeFromCorners(prism.corners, prism.topY);
    if (!shape) { setPrism(EMPTY_PRISM); return; }
    busyRef.current = true;
    VoxaAPI.regionCreate({ prism: shape })
      .then((region) => { onCreated?.(region); })
      .catch((err) => console.error('region create failed:', err))
      .finally(() => { busyRef.current = false; setPrism(EMPTY_PRISM); });
  }, [prism, onCreated]);

  return (
    <>
      <RegionKeys prism={prism} setPrism={setPrism} onExit={onExit} onClose={closeNow} />
      <PrismOverlay viewerRef={viewerRef} prism={prism} setPrism={setPrism} onClose={closeFootprint} />
      <PrismRubberBand viewerRef={viewerRef} prism={prism} />
      <RegionDrawPanel prism={prism} onClear={() => setPrism(EMPTY_PRISM)} />
    </>
  );
}

// Persisted-region volumes: translucent fill + outline per visible region,
// colored by status. Dispose-and-rebuild on change, like PrismOverlay.
export function RegionsOverlay({ viewerRef, regions, visibleIds }) {
  const layerRef = useRef(null);
  useEffect(() => {
    const v = viewerRef.current;
    if (!v?.attachOverlayGroup) return undefined;
    layerRef.current = v.attachOverlayGroup();
    return () => { layerRef.current?.remove(); layerRef.current = null; };
  }, [viewerRef]);

  useEffect(() => {
    const group = layerRef.current?.group;
    if (!group) return;
    while (group.children.length) {
      const c = group.children.pop();
      c.geometry?.dispose?.(); c.material?.dispose?.();
    }
    const noRay = (o) => { o.raycast = () => {}; return o; };
    for (const region of regions) {
      if (!visibleIds.has(region.id)) continue;
      const color = REGION_COLORS[region.status] ?? REGION_COLORS.draft;
      const { polygon, y0, height } = region.prism;
      const ring = (y) => polygon.map(([x, z]) => new THREE.Vector3(x, y, z));
      const bot = ring(y0), top = ring(y0 + height);
      group.add(fillMesh([...fanTris(bot), ...fanTris(top), ...wallTris(bot, top)], 0.08, color));
      const mat = () => new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.9, depthWrite: false });
      group.add(noRay(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(bot), mat())));
      group.add(noRay(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(top), mat())));
      const seg = [];
      for (let i = 0; i < polygon.length; i++) seg.push(bot[i], top[i]);
      group.add(noRay(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(seg), mat())));
    }
  }, [regions, visibleIds]);
  return null;
}
