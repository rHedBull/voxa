// draw-mode.jsx — Draw (centerline) sub-mode of Label mode. Pipes/tanks are
// labeled by drawing centerline paths; the backend extracts points within a
// tube radius. State machine in draw-paths.js; spec in
// docs/superpowers/specs/2026-06-04-centerline-pipe-labeling-design.md.

import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { evtToNdc } from './viewer.jsx';
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

function DrawHUD({ state, toast }) {
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

function DrawOverlay({ viewerRef, draw, setDraw, classes, defaultClsIdx }) {
  const layerRef = useRef(null);        // { group, remove }
  const dragRef = useRef(null);         // { pathKey, pointIdx, plane, mesh, last }
  const drawRef = useRef(draw);
  drawRef.current = draw;
  const defaultClsIdxRef = useRef(defaultClsIdx);
  defaultClsIdxRef.current = defaultClsIdx;

  // One overlay group for the lifetime of the sub-mode.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v?.attachOverlayGroup) return undefined;
    layerRef.current = v.attachOverlayGroup();
    return () => { layerRef.current?.remove(); layerRef.current = null; };
  }, [viewerRef]);

  // Rebuild overlay children whenever paths/selection change. Path counts
  // are tiny (dozens), so dispose-and-rebuild beats incremental bookkeeping.
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer?.group) return;
    const group = layer.group;
    while (group.children.length) {
      const c = group.children.pop();
      c.geometry?.dispose?.(); c.material?.dispose?.();
      group.remove(c);
    }
    for (const p of draw.paths) {
      const cls = classes[p.classId];
      const color = new THREE.Color(cls?.color || '#60a5fa');
      const isSel = draw.selection.has(p.key) || draw.active === p.key;
      const applied = draw.instanceIds[p.instKey] != null;
      const baseOpacity = applied ? 0.10 : 0.25;
      const tubeMatCfg = {
        color, transparent: true, depthWrite: false,
        opacity: isSel ? baseOpacity + 0.15 : baseOpacity,
      };
      // Tube: smooth → one TubeGeometry; straight → cylinder per segment.
      // Each mesh gets its own material (no orphan to leak on rebuild).
      const pts3 = p.points.map(([x, y, z]) => new THREE.Vector3(x, y, z));
      if (p.smooth && pts3.length >= 3) {
        const curve = new THREE.CatmullRomCurve3(pts3);
        const tube = new THREE.Mesh(
          new THREE.TubeGeometry(curve, pts3.length * 8, p.radius, 12, false),
          new THREE.MeshBasicMaterial(tubeMatCfg));
        tube.userData.drawPath = p.key;
        group.add(tube);
      } else {
        for (let i = 0; i < pts3.length - 1; i++) {
          const a = pts3[i], b = pts3[i + 1];
          const len = a.distanceTo(b);
          if (len < 1e-6) continue;
          const cyl = new THREE.Mesh(
            new THREE.CylinderGeometry(p.radius, p.radius, len, 12, 1, true),
            new THREE.MeshBasicMaterial(tubeMatCfg));
          cyl.position.copy(a).lerp(b, 0.5);
          cyl.quaternion.setFromUnitVectors(
            new THREE.Vector3(0, 1, 0), b.clone().sub(a).normalize());
          cyl.userData.drawPath = p.key;
          group.add(cyl);
        }
      }
      // Control points — pick priority targets (spec: point > tube > cloud).
      const sphereR = Math.max(0.02, p.radius * 0.3);
      p.points.forEach((pt, i) => {
        const sph = new THREE.Mesh(
          new THREE.SphereGeometry(sphereR, 12, 8),
          new THREE.MeshBasicMaterial({ color: isSel ? 0xffffff : color }));
        sph.position.set(pt[0], pt[1], pt[2]);
        sph.userData.drawPoint = { pathKey: p.key, pointIdx: i };
        group.add(sph);
      });
    }
  }, [draw, classes]);

  // Pointer interactions.
  useEffect(() => {
    const v = viewerRef.current;
    const dom = v?.domElement?.();
    if (!dom) return undefined;
    const raycaster = new THREE.Raycaster();

    const castOverlay = (evt) => {
      const camera = v.getCamera();
      const group = layerRef.current?.group;
      if (!camera || !group) return [];
      const rect = dom.getBoundingClientRect();
      raycaster.setFromCamera(evtToNdc(evt, rect), camera);
      return raycaster.intersectObjects(group.children, false);
    };

    const onPointerDown = (evt) => {
      if (evt.button !== 0) return;
      const hits = castOverlay(evt);
      const sphereHit = hits.find((h) => h.object.userData.drawPoint);
      if (sphereHit && !evt.ctrlKey) {
        // Drag start: move the point on a camera-parallel plane through it.
        const camera = v.getCamera();
        const normal = new THREE.Vector3();
        camera.getWorldDirection(normal);
        const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(
          normal, sphereHit.object.position.clone());
        dragRef.current = {
          ...sphereHit.object.userData.drawPoint, plane,
          mesh: sphereHit.object, last: null,
        };
        v.setOrbitEnabled(false);
        evt.stopPropagation();
        return;
      }
      if (evt.ctrlKey || evt.metaKey) {
        // Place a control point at the picked cloud point.
        const hit = v.firstHitUnderCursor(evt);
        if (hit) {
          setDraw((s) => {
            const activePath = s.active && s.paths.find((p) => p.key === s.active);
            const classId = activePath ? activePath.classId : defaultClsIdxRef.current;
            return addPoint(s, [hit.world.x, hit.world.y, hit.world.z], classId);
          });
        }
        evt.stopPropagation();
        return;
      }
      const tubeHit = hits.find((h) => h.object.userData.drawPath);
      if (tubeHit) {
        setDraw((s) => selectPath(s, tubeHit.object.userData.drawPath,
          { additive: evt.shiftKey }));
        return;     // don't stop: orbit-from-tube is harmless and feels natural
      }
      // Plain click on empty space / cloud clears the selection (keymap).
      if (drawRef.current.selection.size) setDraw((s) => clearSelection(s));
    };

    const onPointerMove = (evt) => {
      const drag = dragRef.current;
      if (!drag) return;
      const camera = v.getCamera();
      if (!camera) return;
      const rect = dom.getBoundingClientRect();
      raycaster.setFromCamera(evtToNdc(evt, rect), camera);
      const pt = new THREE.Vector3();
      if (raycaster.ray.intersectPlane(drag.plane, pt)) {
        // Sphere tracks live; the tube re-renders once on release. A full
        // overlay rebuild at pointer rate would thrash geometry alloc/dispose.
        drag.mesh.position.copy(pt);
        drag.last = [pt.x, pt.y, pt.z];
      }
    };

    const onPointerUp = () => {
      const drag = dragRef.current;
      if (!drag) return;
      if (drag.last) setDraw((cur) => movePoint(cur, drag.pathKey, drag.pointIdx, drag.last));
      dragRef.current = null;
      v.setOrbitEnabled(true);
    };

    // Wheel-resize beats orbit-zoom: orbit's wheel listener bubbles on the
    // canvas, so a CAPTURE listener on the PARENT runs first and can stop
    // propagation before the canvas ever sees it. (Capture vs bubble on the
    // same node would NOT reorder — the parent hop is the trick.)
    const wheelHost = dom.parentElement || dom;
    const onWheel = (evt) => {
      const s = drawRef.current;
      if (!s.active && s.selection.size === 0) return;   // fall through to zoom
      evt.preventDefault();
      evt.stopPropagation();
      setDraw((cur) => nudgeRadius(cur, -Math.sign(evt.deltaY)));
    };

    dom.addEventListener('pointerdown', onPointerDown, true);
    window.addEventListener('pointermove', onPointerMove);
    window.addEventListener('pointerup', onPointerUp);
    wheelHost.addEventListener('wheel', onWheel, { capture: true, passive: false });
    return () => {
      dom.removeEventListener('pointerdown', onPointerDown, true);
      window.removeEventListener('pointermove', onPointerMove);
      window.removeEventListener('pointerup', onPointerUp);
      wheelHost.removeEventListener('wheel', onWheel, { capture: true });
      v.setOrbitEnabled(true);
    };
  }, [viewerRef, setDraw]);

  return null;
}

export default function DrawMode({
  viewerRef, classes, setSegState, onExit,
}) {
  const [draw, setDraw] = useState(() => initDrawState());
  const drawLiveRef = useRef(draw);
  drawLiveRef.current = draw;
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
    // The press-time `draw` decides which calls to send — intentional
    // snapshot semantics (Enter applies what was selected at press time).
    // All state WRITES go through functional updaters so user edits during
    // the network round-trips survive. Don't read state back out of a
    // setDraw updater: React only invokes updaters eagerly as an
    // optimization, not as a contract.
    let snapshot = drawLiveRef.current;
    if (snapshot.active) {
      const key = snapshot.active;
      snapshot = endActive(snapshot);
      if (snapshot.paths.some((p) => p.key === key)) snapshot = selectPath(snapshot, key);
      // Commit the same end+select pre-step to the live state — guarded so a
      // raced edit (e.g. Esc already ended the path) wins over the replay.
      setDraw((cur) => {
        if (cur.active !== key) return cur;
        let s2 = endActive(cur);
        if (s2.paths.some((p) => p.key === key)) s2 = selectPath(s2, key);
        return s2;
      });
    }
    const calls = buildApplyCalls(snapshot);
    if (calls.length === 0) return;
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
      setDraw((cur) => markApplied(cur, call.instKey, r.instanceId));
      setSegState((st) => st ? applyDelta(st, {
        indices: r.indices,
        after_class: r.afterClass,
        after_instance: r.afterInstance,
      }) : st);
    }
    // Clear selection after Enter (spec) — even if some calls failed. Route
    // through a functional updater so concurrent edits in cur survive.
    setDraw((cur) => clearSelection(cur));
  }, [setSegState, showToast]);

  const onKey = useCallback((action) => {
    switch (action.type) {
      case 'class':
        setDefaultClsIdx(action.clsIdx);
        setDraw((s) => setClass(s, action.clsIdx));
        break;
      case 'apply':
        applySelection();
        break;
      case 'escape': {
        // Decide on the live state OUTSIDE the updater — calling onExit()
        // (a LabelMode setState) from inside setDraw runs during render and
        // trips React's setState-while-rendering warning.
        const s = drawLiveRef.current;
        if (s.active) setDraw((cur) => endActive(cur));
        else if (s.selection.size) setDraw((cur) => clearSelection(cur));
        else onExit();
        break;
      }
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
      <DrawHUD state={draw} toast={toast} />
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
function DrawPanel({ draw, setDraw, classes, onApply }) {
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
