// beam-mode.jsx — Beam sub-mode of Label mode. Steel members (beams/pillars)
// are labeled by building a node/edge graph: joints are placed on the cloud,
// edges between joints are square-section boxes applied through the shared
// apply-shape pipeline, one instance per beam. State machine in beam-graph.js;
// spec: docs/superpowers/specs/2026-07-10-beam-structure-labeling-design.md.

import { useCallback, useEffect, useRef, useState } from 'react';
import * as THREE from 'three';
import { evtToNdc } from './viewer.jsx';
import { VoxaAPI } from './api.js';
import { applyDelta } from './segment-state.js';
import {
  initBeamState, addNode, clickNode, selectNode, selectEdge, clearSelection,
  moveNode, deleteSelected, setWidth, nudgeWidth, setClass,
  applyTargets, markApplied, commitAll, obbForEdge, toStructureDoc,
  seedFromServer, nodePos,
} from './beam-graph.js';

// Capture-phase keyboard driver (same trick as DrawKeys / FastLabelKeys).
export function BeamKeys({ active, classes, onKey }) {
  useEffect(() => {
    if (!active) return undefined;
    const handler = (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      // Ctrl+Enter commits the batch; every other Ctrl/Meta/Alt combo
      // (Ctrl+S/Z, Ctrl+click…) passes through untouched.
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault(); e.stopPropagation();
        onKey({ type: 'commit' });
        return;
      }
      if (e.ctrlKey || e.metaKey || e.altKey) return;
      const cls = classes.find((c) => c.hotkey === e.key);
      let handled = true;
      if (cls) onKey({ type: 'class', classId: cls.class_id });
      else if (e.key === 'Enter') onKey({ type: 'apply' });
      else if (e.key === 'Escape') onKey({ type: 'escape' });
      else if (e.key === 'Backspace' || e.key === 'Delete') onKey({ type: 'delete' });
      else if (e.key === '+' || e.key === '=') onKey({ type: 'width', dir: +1 });
      else if (e.key === '-' || e.key === '_') onKey({ type: 'width', dir: -1 });
      else handled = false;
      if (handled) { e.preventDefault(); e.stopPropagation(); }
    };
    window.addEventListener('keydown', handler, true);
    return () => window.removeEventListener('keydown', handler, true);
  }, [active, classes, onKey]);
  return null;
}

function BeamHUD({ state, toast }) {
  const sel = state.selection;
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
          <b style={{ color: sel?.kind === 'node' ? '#fb923c' : '#60a5fa' }}>
            {sel?.kind === 'node' ? 'Joint selected'
              : sel?.kind === 'edge' ? 'Beam selected'
              : 'Build beam graph'}
          </b>
          <span style={{ opacity: 0.65 }}>
            {sel?.kind === 'node'
              ? 'Ctrl+click another joint to connect · drag move · ⌫ delete · Esc deselect'
              : sel?.kind === 'edge'
              ? 'scroll/± width · class hotkey · ⌫ delete · Enter apply · Esc deselect'
              : 'Ctrl+click cloud place joint · Ctrl+click joint select · Enter apply · Ctrl+Enter commit · Esc exit'}
          </span>
        </>
      )}
    </div>
  );
}

// Oriented box mesh for a beam a→b (shared by active edges + committed layer).
function makeBeamBox(a, b, width, matCfg) {
  const av = new THREE.Vector3(...a), bv = new THREE.Vector3(...b);
  const len = av.distanceTo(bv);
  if (len < 1e-6) return null;
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(len, width, width),
    new THREE.MeshBasicMaterial(matCfg));
  mesh.position.copy(av).lerp(bv, 0.5);
  // Box local x is the beam axis; any roll about it is fine (square section).
  mesh.quaternion.setFromUnitVectors(
    new THREE.Vector3(1, 0, 0), bv.clone().sub(av).normalize());
  return mesh;
}

function BeamOverlay({ viewerRef, beam, setBeam, classes, defaultClassId, showCommitted }) {
  const layerRef = useRef(null);        // { group, remove }
  const dragRef = useRef(null);         // { nodeId, plane, mesh, last }
  const beamRef = useRef(beam);
  beamRef.current = beam;
  const defaultClassIdRef = useRef(defaultClassId);
  defaultClassIdRef.current = defaultClassId;

  // One overlay group for the lifetime of the sub-mode.
  useEffect(() => {
    const v = viewerRef.current;
    if (!v?.attachOverlayGroup) return undefined;
    layerRef.current = v.attachOverlayGroup();
    return () => { layerRef.current?.remove(); layerRef.current = null; };
  }, [viewerRef]);

  // Rebuild overlay children whenever the graph/selection changes. Graphs are
  // tiny (dozens of members), so dispose-and-rebuild beats bookkeeping.
  useEffect(() => {
    const layer = layerRef.current;
    if (!layer?.group) return;
    const group = layer.group;
    while (group.children.length) {
      const c = group.children.pop();
      c.geometry?.dispose?.(); c.material?.dispose?.();
      group.remove(c);
    }
    const addRimShell = (mesh, grow, color = 0xffffff, opacity = 0.55) => {
      // White back-side shell = selection rim (borrowed from draw-mode).
      // Never swallows picks: raycast is a no-op.
      const p = mesh.geometry.parameters;
      const shell = new THREE.Mesh(
        new THREE.BoxGeometry(p.width + grow, p.height + grow, p.depth + grow),
        new THREE.MeshBasicMaterial({
          color, side: THREE.BackSide, transparent: true, opacity, depthWrite: false,
        }));
      shell.raycast = () => {};
      shell.position.copy(mesh.position);
      shell.quaternion.copy(mesh.quaternion);
      group.add(shell);
    };
    // Committed layer: faded, read-only, unpickable.
    if (showCommitted) {
      for (const c of beam.committed) {
        const cls = classes.find((k) => k.class_id === c.classId);
        const mesh = makeBeamBox(c.a, c.b, c.width, {
          color: new THREE.Color(cls?.color || '#9ca3af'),
          transparent: true, opacity: 0.08, depthWrite: false,
        });
        if (!mesh) continue;
        mesh.raycast = () => {};
        group.add(mesh);
      }
    }
    // Active edges.
    for (const e of beam.edges) {
      const cls = classes.find((k) => k.class_id === e.classId);
      const isSel = beam.selection?.kind === 'edge' && beam.selection.id === e.id;
      const mesh = makeBeamBox(
        nodePos(beam, e.a), nodePos(beam, e.b), e.width, {
          color: new THREE.Color(cls?.color || '#60a5fa'),
          transparent: true, depthWrite: false,
          opacity: isSel ? 0.40 : 0.25,
        });
      if (!mesh) continue;
      mesh.userData.beamEdge = e.id;
      group.add(mesh);
      if (isSel) addRimShell(mesh, Math.max(0.02, e.width * 0.15));
    }
    // Nodes. Pick priority (node sphere > beam box > cloud) comes from the
    // pointer handler checking beamNode hits before beamEdge hits — not from
    // add order.
    const sphereR = Math.max(0.03, beam.lastWidth * 0.3);
    for (const n of beam.nodes) {
      const isSel = beam.selection?.kind === 'node' && beam.selection.id === n.id;
      const sph = new THREE.Mesh(
        new THREE.SphereGeometry(sphereR, 12, 8),
        new THREE.MeshBasicMaterial({ color: 0x60a5fa }));
      sph.position.set(n.pos[0], n.pos[1], n.pos[2]);
      sph.userData.beamNode = n.id;
      group.add(sph);
      if (isSel) {
        // Opaque orange ring — the connect anchor, same affordance as Draw's
        // extend anchor.
        const shell = new THREE.Mesh(
          new THREE.SphereGeometry(sphereR * 1.5, 12, 8),
          new THREE.MeshBasicMaterial({
            color: 0xfb923c, side: THREE.BackSide, transparent: true,
            opacity: 1, depthWrite: false,
          }));
        shell.raycast = () => {};
        shell.position.copy(sph.position);
        group.add(shell);
      }
    }
  }, [beam, classes, showCommitted]);

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
      const nodeHit = hits.find((h) => h.object.userData.beamNode != null);
      if (nodeHit && !evt.ctrlKey && !evt.metaKey) {
        // Drag start: move the joint on a camera-parallel plane through it.
        const camera = v.getCamera();
        const normal = new THREE.Vector3();
        camera.getWorldDirection(normal);
        const plane = new THREE.Plane().setFromNormalAndCoplanarPoint(
          normal, nodeHit.object.position.clone());
        dragRef.current = {
          nodeId: nodeHit.object.userData.beamNode, plane,
          mesh: nodeHit.object, last: null,
        };
        v.setOrbitEnabled(false);
        evt.stopPropagation();
        return;
      }
      if (evt.ctrlKey || evt.metaKey) {
        // Ctrl is the "edit graph" modifier; pick priority node > beam > cloud.
        if (nodeHit) {
          setBeam((s) => clickNode(s, nodeHit.object.userData.beamNode,
            defaultClassIdRef.current));
          evt.stopPropagation();
          return;
        }
        const edgeHit = hits.find((h) => h.object.userData.beamEdge != null);
        if (edgeHit) {
          setBeam((s) => selectEdge(s, edgeHit.object.userData.beamEdge));
          evt.stopPropagation();
          return;
        }
        const hit = v.firstHitUnderCursor(evt);
        if (hit) setBeam((s) => addNode(s, [hit.world.x, hit.world.y, hit.world.z]));
        evt.stopPropagation();
        return;
      }
      // Plain click: select a beam under the cursor (doesn't block orbit,
      // same feel as Draw's tube click); empty click clears the selection.
      const edgeHit = hits.find((h) => h.object.userData.beamEdge != null);
      if (edgeHit) {
        setBeam((s) => selectEdge(s, edgeHit.object.userData.beamEdge));
        return;
      }
      if (beamRef.current.selection) setBeam((s) => clearSelection(s));
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
        // Sphere tracks live; incident boxes re-render once on release —
        // full rebuilds at pointer rate would thrash geometry alloc/dispose.
        drag.mesh.position.copy(pt);
        drag.last = [pt.x, pt.y, pt.z];
      }
    };

    const onPointerUp = () => {
      const drag = dragRef.current;
      if (!drag) return;
      if (drag.last) setBeam((cur) => moveNode(cur, drag.nodeId, drag.last));
      // No movement → it was a click: select the joint (connect anchor).
      // Race-tolerant: the node may have been deleted mid-drag.
      else setBeam((cur) => cur.nodes.some((n) => n.id === drag.nodeId)
        ? selectNode(cur, drag.nodeId) : cur);
      dragRef.current = null;
      v.setOrbitEnabled(true);
    };

    // Wheel-resize beats orbit-zoom via a CAPTURE listener on the PARENT
    // (the same trick draw-mode documents). Falls through to zoom unless a
    // beam is selected.
    const wheelHost = dom.parentElement || dom;
    const onWheel = (evt) => {
      if (beamRef.current.selection?.kind !== 'edge') return;
      evt.preventDefault();
      evt.stopPropagation();
      setBeam((cur) => nudgeWidth(cur, -Math.sign(evt.deltaY)));
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
  }, [viewerRef, setBeam]);

  return null;
}

export default function BeamMode({
  viewerRef, classes, setSegState, onExit, pointSize, setPointSize,
  defaultClassId, onClassChange, onApplied, sessionId, protectInstances = [],
}) {
  const [beam, setBeam] = useState(() => initBeamState());
  const beamLiveRef = useRef(beam);
  beamLiveRef.current = beam;
  // Latest confirmed-lock set, read at apply time (async), not closure-stale.
  const protectInstancesRef = useRef(protectInstances);
  protectInstancesRef.current = protectInstances;
  const [showCommitted, setShowCommitted] = useState(true);
  const [toast, setToast] = useState(null);
  const toastTimer = useRef(null);
  const showToast = useCallback((msg) => {
    clearTimeout(toastTimer.current);
    setToast(msg);
    toastTimer.current = setTimeout(() => setToast(null), 2500);
  }, []);
  const seededRef = useRef(false);
  const persistPendingRef = useRef(false);
  const skipPersistRef = useRef(false);

  // Load stored structure once on open (active graph + committed layer).
  // On failure (incl. a 409 from the session pin) seededRef stays false → we
  // never PUT → a load hiccup can't wipe the server-side doc with an empty
  // graph.
  useEffect(() => {
    let gone = false;
    VoxaAPI.getStructure(sessionId)
      .then((doc) => {
        if (gone) return;
        skipPersistRef.current = true;
        setBeam((s) => seedFromServer(s, doc));
        seededRef.current = true;
      })
      .catch((err) => { if (!gone) showToast(`structure load failed: ${err.message}`); });
    return () => { gone = true; };
  }, [showToast, sessionId]);

  // Persist graph geometry (debounced) after every graph change post-seed.
  // Covers apply/commit/edits per the spec; the session pin makes a session
  // switch mid-debounce a loud 409, never a cross-session write. Not on the
  // undo stack (matching centerlines.json). Depends on the three persisted
  // arrays — the beam-graph ops preserve their identity when untouched — so
  // selection clicks don't trigger writes.
  useEffect(() => {
    if (!seededRef.current) return undefined;
    if (skipPersistRef.current) {
      // The seed run: the arrays were just replaced by what we loaded —
      // writing them back is a redundant echo (and, in a remount race, the
      // vector for cross-session clobber).
      skipPersistRef.current = false;
      return undefined;
    }
    persistPendingRef.current = true;
    const t = setTimeout(() => {
      persistPendingRef.current = false;
      VoxaAPI.putStructure(toStructureDoc(beamLiveRef.current), sessionId)
        .catch((err) => showToast(`structure save failed: ${err.message}`));
    }, 800);
    return () => clearTimeout(t);
  }, [beam.nodes, beam.edges, beam.committed, sessionId, showToast]);

  // Flush a pending debounced write on unmount — otherwise the last <800ms of
  // graph changes (incl. fresh instanceIds from an apply) silently vanish.
  // The session pin turns a race with a session switch into a 409, never a
  // cross-session write.
  useEffect(() => () => {
    if (seededRef.current && persistPendingRef.current) {
      VoxaAPI.putStructure(toStructureDoc(beamLiveRef.current), sessionId)
        .catch((err) => console.error('beam structure flush failed:', err));
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Apply every never-applied/edited beam: one apply-shape call per beam,
  // sequential (same loop pattern as draw-mode's applySelection). Press-time
  // snapshot decides the calls; all state WRITES go through functional
  // updaters so user edits during the round-trips survive.
  const applyAll = useCallback(async () => {
    const snapshot = beamLiveRef.current;
    const targets = applyTargets(snapshot);
    if (targets.length === 0) return true;
    let allOk = true;
    for (const edge of targets) {
      const obb = obbForEdge(snapshot, edge);
      let r;
      try {
        r = await VoxaAPI.applyShape({
          shape: { type: 'obb', ...obb },
          targetClass: edge.classId,
          targetInst: edge.instanceId ?? -1,
          protectInstances: protectInstancesRef.current,
        });
      } catch (err) {
        showToast(`apply failed: ${err.message}`);
        allOk = false;
        continue;                     // surface and move on; edge stays dirty
      }
      if (r.nAffected === 0) {
        showToast(r.nProtected > 0
          ? `${r.nProtected} point(s) locked in a confirmed instance`
          : 'no points in beam box');
        allOk = false;
        continue;                     // no instance allocated for empty beams
      }
      setBeam((cur) => markApplied(cur, edge.id, r.instanceId, edge));
      setSegState((st) => st ? applyDelta(st, {
        indices: r.indices,
        after_class: r.afterClass,
        after_instance: r.afterInstance,
      }) : st);
      onApplied?.({ instanceId: r.instanceId, classId: edge.classId,
                    source: 'beam', obb });
    }
    return allOk;
  }, [setSegState, showToast, onApplied]);

  const commitBatch = useCallback(async () => {
    // Decide on live state OUTSIDE any updater — window.confirm must never
    // run inside a React updater (same rule as draw-mode Esc/Backspace).
    const s = beamLiveRef.current;
    if (s.edges.length === 0) { showToast('nothing to commit'); return; }
    const n = s.edges.length;
    if (!window.confirm(`Commit ${n} beam${n > 1 ? 's' : ''} to unconfirmed instances?`)) return;
    const ok = await applyAll();
    if (!ok) showToast('some beams failed to apply — they stay in the active graph');
    setBeam((cur) => commitAll(cur));
  }, [applyAll, showToast]);

  const onKey = useCallback((action) => {
    switch (action.type) {
      case 'class':
        onClassChange(action.classId);
        setBeam((s) => setClass(s, action.classId));
        break;
      case 'apply':
        applyAll();
        break;
      case 'commit':
        commitBatch();
        break;
      case 'width':
        setBeam((s) => nudgeWidth(s, action.dir));
        break;
      case 'delete': {
        const s = beamLiveRef.current;
        if (s.selection) setBeam((cur) => deleteSelected(cur));
        break;
      }
      case 'escape': {
        // Live-state decision outside the updater (onExit is a LabelMode
        // setState — calling it inside setBeam would run during render).
        const s = beamLiveRef.current;
        if (s.selection) setBeam((cur) => clearSelection(cur));
        else onExit();
        break;
      }
      default:
    }
  }, [applyAll, commitBatch, onExit, onClassChange]);

  // The sidebar class list is the single source of the beam class — clicking
  // a class row (or its hotkey) re-targets the selected beam.
  useEffect(() => {
    setBeam((s) => (s.selection?.kind === 'edge') ? setClass(s, defaultClassId) : s);
  }, [defaultClassId]);

  return (
    <>
      <BeamKeys active classes={classes} onKey={onKey} />
      <BeamHUD state={beam} toast={toast} />
      <BeamOverlay
        viewerRef={viewerRef}
        beam={beam}
        setBeam={setBeam}
        classes={classes}
        defaultClassId={defaultClassId}
        showCommitted={showCommitted}
      />
      <BeamPanel
        beam={beam}
        setBeam={setBeam}
        classes={classes}
        onApply={applyAll}
        onCommit={commitBatch}
        pointSize={pointSize}
        setPointSize={setPointSize}
        showCommitted={showCommitted}
        setShowCommitted={setShowCommitted}
      />
    </>
  );
}

// Side-panel section: beam list + width field + actions (rendered by
// LabelMode inside the left sidebar, like DrawPanel).
function BeamPanel({
  beam, setBeam, classes, onApply, onCommit, pointSize, setPointSize,
  showCommitted, setShowCommitted,
}) {
  const selEdge = beam.selection?.kind === 'edge'
    ? beam.edges.find((e) => e.id === beam.selection.id) : null;
  const widthValue = selEdge?.width ?? beam.lastWidth;
  const nCommitted = beam.committed.length;
  return (
    <div className="beam-panel" style={{ marginTop: 10 }}>
      <div className="side-hd"><span>Beams</span>
        <div className="side-hd-actions">
          {nCommitted > 0 && (
            <button className="hide-labeled-btn"
              onClick={() => setShowCommitted((v) => !v)}
              title={showCommitted
                ? `Hide ${nCommitted} committed beam${nCommitted === 1 ? '' : 's'}`
                : `Show ${nCommitted} committed beam${nCommitted === 1 ? '' : 's'}`}>
              {showCommitted ? '●' : '◌'} {nCommitted} committed
            </button>
          )}
          <span className="badge-soft">{beam.edges.length}</span>
        </div></div>
      <div className="ins-row">
        <label>Width</label>
        <input className="ins-input" type="number" step="0.01" min="0.01"
          value={Number(widthValue.toFixed(4))}
          onChange={(e) => {
            const v = parseFloat(e.target.value);
            if (Number.isFinite(v) && v > 0) setBeam((s) => setWidth(s, v));
          }} />
      </div>
      <div className="ctrl" style={{ margin: '6px 0' }}>
        <label>Point size <span className="mono">{pointSize.toFixed(3)}</span></label>
        <input type="range" min={0.002} max={1.5} step={0.005}
          value={pointSize} className="slider"
          onChange={(e) => setPointSize(Number(e.target.value))} />
      </div>
      <div style={{ maxHeight: 180, overflowY: 'auto' }}>
        {beam.edges.map((e) => {
          const cls = classes.find((c) => c.class_id === e.classId);
          const isSel = beam.selection?.kind === 'edge' && beam.selection.id === e.id;
          const status = e.instanceId == null ? '(staged)'
            : e.dirty ? `#${e.instanceId} (edited)` : `#${e.instanceId}`;
          return (
            <div key={e.id}
              className={'inst-row' + (isSel ? ' selected' : '')}
              onClick={() => setBeam((s) => selectEdge(s, e.id))}>
              <span className="inst-dot" style={{ background: cls?.color }} />
              <div className="inst-text">
                <b>{cls?.label || '?'} {status}</b>
                <em>w={e.width.toFixed(3)}</em>
              </div>
            </div>
          );
        })}
      </div>
      <div className="ins-actions">
        <button className="ghost-btn" disabled={beam.edges.length === 0}
          onClick={onApply}>↵ Apply</button>
        <button className="ghost-btn" disabled={beam.edges.length === 0}
          onClick={onCommit}>⌃↵ Commit</button>
      </div>
    </div>
  );
}
