// prism-draw.jsx — the shared footprint+height draw machinery (state shape,
// geometry helpers, viewport overlay + rubber band), extracted from
// prism-mode.jsx so the Region tool can reuse the exact Prism interaction
// with a different commit handler (eval-regions spec §3). Classify-specific
// logic (chords, class picker, apply) stays in prism-mode.jsx.

import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { evtToNdc } from './viewer.jsx';
import { prismShapeFromCorners } from './prism-geom.js';

// phase 'footprint': placing snapped corners. phase 'height': footprint closed;
// `committed` false = aiming the height live, true = height locked, awaiting a
// class. corners are [x,y,z] snapped to the surface; topY is the aimed top.
export const EMPTY_PRISM = { phase: 'footprint', corners: [], baseY: null, heightEdge: null, topY: null, committed: false };
export const MIN_HEIGHT = 0.05;
export const CLICK_PX = 5;          // pointerup within this of pointerdown = a click (else a camera drag)
export const ACCENT = 0x38bdf8;     // sky-400

export const liveHeight = (p) => (p.topY == null || p.baseY == null ? 0 : Math.abs(p.topY - p.baseY));

// Vertical plane through a footprint edge (normal horizontal, perpendicular to
// the edge in XZ) — the surface the cursor rides to set the height.
export function edgePlane(edge) {
  const [p1, p2] = [edge.p1, edge.p2];
  const normal = new THREE.Vector3(p2[2] - p1[2], 0, -(p2[0] - p1[0])).normalize();
  return new THREE.Plane().setFromNormalAndCoplanarPoint(normal, new THREE.Vector3(...p1));
}

// The footprint edge (at the flat base) whose midpoint is nearest the camera in
// XZ — gives the height-aim plane the most head-on angle (Volume-tool rule).
export function nearestEdge(corners, baseY, camera) {
  const cam = camera.position;
  let best = null, bestD = Infinity;
  for (let i = 0; i < corners.length; i++) {
    const a = corners[i], b = corners[(i + 1) % corners.length];
    const p1 = [a[0], baseY, a[2]], p2 = [b[0], baseY, b[2]];
    const d = Math.hypot(cam.x - (p1[0] + p2[0]) / 2, cam.z - (p1[2] + p2[2]) / 2);
    if (d < bestD) { bestD = d; best = { p1, p2 }; }
  }
  return best;
}

// Fan-triangulate a closed ring of Vector3 into a flat position array (from the
// centroid), and the side walls between a bottom and top ring — the translucent
// fills that make the footprint/prism read as solid surfaces, mirroring the
// Measure Surface (buildSurfaceMesh) and Volume (buildVolumeMesh) tools.
export function fanTris(ring) {
  const c = new THREE.Vector3();
  ring.forEach((p) => c.add(p));
  c.multiplyScalar(1 / ring.length);
  const pos = [];
  for (let i = 0; i < ring.length; i++) {
    const a = ring[i], b = ring[(i + 1) % ring.length];
    pos.push(c.x, c.y, c.z, a.x, a.y, a.z, b.x, b.y, b.z);
  }
  return pos;
}
export function wallTris(bot, top) {
  const pos = [];
  for (let i = 0; i < bot.length; i++) {
    const b1 = bot[i], b2 = bot[(i + 1) % bot.length], t1 = top[i], t2 = top[(i + 1) % top.length];
    pos.push(b1.x, b1.y, b1.z, b2.x, b2.y, b2.z, t2.x, t2.y, t2.z);
    pos.push(b1.x, b1.y, b1.z, t2.x, t2.y, t2.z, t1.x, t1.y, t1.z);
  }
  return pos;
}
export function fillMesh(positions, opacity, color = ACCENT) {
  const g = new THREE.BufferGeometry();
  g.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  const m = new THREE.Mesh(g, new THREE.MeshBasicMaterial({
    color, transparent: true, opacity, depthWrite: false, side: THREE.DoubleSide,
  }));
  m.raycast = () => {};
  return m;
}

// Bounding-box-scaled marker radius so corner spheres read at any scene scale.
export function markerRadius(corners) {
  if (corners.length < 2) return 0.05;
  let minX = Infinity, maxX = -Infinity, minZ = Infinity, maxZ = -Infinity;
  for (const [x, , z] of corners) {
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  return Math.min(0.5, Math.max(0.02, Math.hypot(maxX - minX, maxZ - minZ) * 0.02));
}

// Dashed rubber-band from the last placed corner to the cursor, footprint phase
// only. Driven imperatively (its own overlay group) so it never re-runs the
// state-driven rebuild effect at pointermove rate. The band rides the SAME
// horizontal base plane the next corner will land on (y = baseY = anchor Y), so
// the preview matches the actual landing point exactly.
export function PrismRubberBand({ viewerRef, prism }) {
  const anchor = prism.phase === 'footprint' && prism.corners.length > 0
    ? prism.corners[prism.corners.length - 1] : null;
  useEffect(() => {
    const v = viewerRef.current;
    if (!anchor || !v?.attachOverlayGroup || !v.domElement || !v.getCamera) return undefined;
    const camera = v.getCamera();
    const dom = v.domElement();
    if (!camera || !dom) return undefined;

    const handle = v.attachOverlayGroup();
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(6), 3));
    const band = new THREE.Line(geom, new THREE.LineDashedMaterial({
      color: ACCENT, transparent: true, opacity: 0.9, dashSize: 0.2, gapSize: 0.12, depthWrite: false,
    }));
    band.frustumCulled = false;
    band.raycast = () => {};
    band.visible = false;
    handle.group.add(band);

    const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -anchor[1]);  // horizontal base plane
    const raycaster = new THREE.Raycaster();
    const target = new THREE.Vector3();
    const onMove = (e) => {
      if (e.buttons !== 0) { band.visible = false; return; }   // camera drag, not aiming
      raycaster.setFromCamera(evtToNdc(e, dom.getBoundingClientRect()), camera);
      if (!raycaster.ray.intersectPlane(plane, target)) return;
      const pos = band.geometry.getAttribute('position');
      pos.setXYZ(0, anchor[0], anchor[1], anchor[2]);
      pos.setXYZ(1, target.x, target.y, target.z);
      pos.needsUpdate = true;
      band.computeLineDistances();
      band.visible = true;
    };
    const onLeave = () => { band.visible = false; };
    dom.addEventListener('pointermove', onMove);
    dom.addEventListener('pointerleave', onLeave);
    return () => {
      dom.removeEventListener('pointermove', onMove);
      dom.removeEventListener('pointerleave', onLeave);
      handle.remove();
    };
  }, [viewerRef, anchor]);
  return null;
}

export function PrismOverlay({ viewerRef, prism, setPrism, onClose }) {
  const layerRef = useRef(null);
  const prismRef = useRef(prism);
  prismRef.current = prism;
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  // Lifetime overlay group.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v?.attachOverlayGroup) return undefined;
    layerRef.current = v.attachOverlayGroup();
    return () => { layerRef.current?.remove(); layerRef.current = null; };
  }, [viewerRef]);

  // Rebuild overlay children on every state change (footprint + height are both
  // small, so dispose-and-rebuild beats bookkeeping — updates at aim rate).
  useEffect(() => {
    const group = layerRef.current?.group;
    if (!group) return;
    while (group.children.length) {
      const c = group.children.pop();
      c.geometry?.dispose?.(); c.material?.dispose?.();
    }
    const { phase, corners } = prism;
    if (corners.length === 0) return;
    const mat = (opacity) => new THREE.LineBasicMaterial({ color: ACCENT, transparent: true, opacity, depthWrite: false });
    const noRay = (o) => { o.raycast = () => {}; return o; };

    // Corner markers at their true snapped positions.
    const r = markerRadius(corners);
    for (const c of corners) {
      const sph = noRay(new THREE.Mesh(new THREE.SphereGeometry(r, 10, 8),
        new THREE.MeshBasicMaterial({ color: ACCENT })));
      sph.position.set(c[0], c[1], c[2]);
      group.add(sph);
    }

    if (phase === 'footprint') {
      const pts = corners.map((c) => new THREE.Vector3(...c));
      // Translucent fill of the traced surface (once it's an area), like the
      // Measure Surface tool; open outline through the snapped corners.
      if (corners.length >= 3) group.add(fillMesh(fanTris(pts), 0.18));
      if (corners.length >= 2) {
        group.add(noRay(new THREE.Line(new THREE.BufferGeometry().setFromPoints(pts), mat(0.95))));
      }
      return;
    }

    // Height phase: the flat prism the selection will actually use — filled
    // (top/bottom faces + side walls) like the Measure Volume tool.
    const shape = prismShapeFromCorners(corners, prism.topY ?? prism.baseY);
    const lo = shape ? shape.y0 : prism.baseY;
    const hi = shape ? shape.y0 + shape.height : prism.baseY;
    const poly = corners.map((c) => [c[0], c[2]]);
    const ring = (y) => poly.map(([x, z]) => new THREE.Vector3(x, y, z));
    const bot = ring(lo), top = ring(hi);
    group.add(fillMesh([...fanTris(bot), ...fanTris(top), ...wallTris(bot, top)], 0.13));
    group.add(noRay(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(bot), mat(0.95))));
    group.add(noRay(new THREE.LineLoop(new THREE.BufferGeometry().setFromPoints(top), mat(0.95))));
    const seg = [];
    for (let i = 0; i < poly.length; i++) { seg.push(bot[i], top[i]); }
    group.add(noRay(new THREE.LineSegments(new THREE.BufferGeometry().setFromPoints(seg), mat(0.5))));
  }, [prism]);

  // Pointer interaction: place corners (footprint), aim/commit height.
  // Orbit stays enabled throughout; BOTH placing a corner AND committing a
  // height go through the same pointerdown→pointerup distance test, so an
  // orbit-drag (moved > CLICK_PX) never places or commits — only a real click
  // does. (Committing on a raw native `click` would let a drag-release lock in
  // a stale height; distinguishing click-from-drag here is what prevents that.)
  useEffect(() => {
    const v = viewerRef.current;
    const dom = v?.domElement?.();
    const camera = v?.getCamera?.();
    if (!dom || !camera) return undefined;
    const raycaster = new THREE.Raycaster();
    let down = null;             // pointerdown pos, for the click-vs-drag test

    const aimTopY = (e) => {
      const p = prismRef.current;
      if (!p.heightEdge) return null;
      raycaster.setFromCamera(evtToNdc(e, dom.getBoundingClientRect()), camera);
      const t = new THREE.Vector3();
      return raycaster.ray.intersectPlane(edgePlane(p.heightEdge), t) ? t.y : null;
    };
    // Ray ∩ the horizontal base plane at y = baseY. Every corner after the
    // first lands here, so the footprint is coplanar and its XZ polygon can't
    // self-intersect from per-corner surface-depth scramble (that produced a
    // bowtie selection). Matches the Measure Volume tool's plane footprint.
    const onBasePlane = (e, baseY) => {
      raycaster.setFromCamera(evtToNdc(e, dom.getBoundingClientRect()), camera);
      const t = new THREE.Vector3();
      const plane = new THREE.Plane(new THREE.Vector3(0, 1, 0), -baseY);
      return raycaster.ray.intersectPlane(plane, t) ? t : null;
    };

    const onDown = (e) => {
      if (e.button !== 0 || e.ctrlKey || e.metaKey) return;
      down = { x: e.clientX, y: e.clientY };
    };
    const onUp = (e) => {
      if (!down) return;
      const { x, y } = down; down = null;
      if (Math.hypot(e.clientX - x, e.clientY - y) > CLICK_PX) return;   // drag = orbit, not a click
      const p = prismRef.current;
      if (p.phase === 'footprint') {
        if (p.corners.length === 0) {
          // First corner snaps to the surface — its Y fixes the base plane.
          const hit = v.firstHitUnderCursor?.(e);
          if (!hit) return;                                              // must start on the surface
          const baseY = hit.world.y;
          setPrism((s) => (s.phase === 'footprint'
            ? { ...s, baseY, corners: [[hit.world.x, baseY, hit.world.z]] } : s));
        } else {
          // Later corners ride the horizontal base plane (can extend past the
          // object / over empty space, unlike a surface-only pick).
          const t = onBasePlane(e, p.baseY);
          if (!t) return;
          setPrism((s) => (s.phase === 'footprint'
            ? { ...s, corners: [...s.corners, [t.x, s.baseY, t.z]] } : s));
        }
      } else if (p.phase === 'height' && !p.committed) {
        // Commit the aimed height. topY sits at baseY right after close (and
        // during the closing dblclick's own clicks), so the height-0 guard
        // drops those — no explicit click-suppressor needed.
        if (p.topY == null || Math.abs(p.topY - p.baseY) < MIN_HEIGHT) return;
        setPrism((s) => (s.phase === 'height' ? { ...s, committed: true } : s));
      }
    };
    const onDblClick = (e) => {
      // Both constituent clicks of the dblclick already ran onUp (adding two
      // corners); drop the last so the double-click nets one final corner then
      // closes. Same fix the Measure Surface tool documents.
      if (prismRef.current.phase !== 'footprint') return;
      e.preventDefault();
      onCloseRef.current(true);
    };
    const onMove = (e) => {
      const p = prismRef.current;
      if (p.phase !== 'height' || p.committed || e.buttons !== 0) return;  // held button = drag, freeze aim
      const topY = aimTopY(e);
      if (topY == null) return;
      setPrism((s) => (s.phase === 'height' && !s.committed ? { ...s, topY } : s));
    };

    dom.addEventListener('pointerdown', onDown);
    dom.addEventListener('pointerup', onUp);
    dom.addEventListener('pointermove', onMove);
    dom.addEventListener('dblclick', onDblClick);
    return () => {
      dom.removeEventListener('pointerdown', onDown);
      dom.removeEventListener('pointerup', onUp);
      dom.removeEventListener('pointermove', onMove);
      dom.removeEventListener('dblclick', onDblClick);
    };
  }, [viewerRef, setPrism]);

  return null;
}
