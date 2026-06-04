// mode-label.jsx — class palette (left), instance list (right), cuboid editing.

import { useState as useStateLabel, useMemo as useMemoLabel,
         useEffect as useEffectLabel, useCallback as useCallbackLabel,
         useRef as useRefLabel } from 'react';
import * as THREE from 'three';
import { Viewer } from './viewer.jsx';
import { ViewportToolbar, ToolButton, HUDChip, CameraPresets, NavModeToggle, HelpButton } from './viewport-atoms.jsx';
import { VoxaAPI, newId } from './api.js';
import { PresegmentList } from './segment-tools.jsx';
import SessionPicker from './session-picker.jsx';
import { applyDelta, computeDiffMask } from './segment-state.js';

// "30k", "1.2M", "523" — keeps the HUD chip narrow regardless of scene size.
function formatPointCount(n) {
  if (n < 1000) return String(n);
  if (n < 1e6) return `${(n / 1e3).toFixed(n < 10000 ? 1 : 0)}k`;
  return `${(n / 1e6).toFixed(n < 1e7 ? 2 : 1)}M`;
}

const LABEL_SEL_BOX_ID = '__label_sel_box__';
const LABEL_SEL_BOX_COLOR = '#ffd24a';

// Mirrors mode-edit.jsx::pointsInsideOBB — kept local to avoid refactoring
// that file while voxa is live. Returns Uint32Array of subRow indices into
// `positions` whose points lie inside the oriented box.
function pointsInsideOBBLabel(positions, box) {
  const [cx, cy, cz] = box.center;
  const [sx, sy, sz] = box.size;
  const [rx, ry, rz] = box.rotation;
  const hx = sx / 2, hy = sy / 2, hz = sz / 2;
  const cxR = Math.cos(rx), sxR = Math.sin(rx);
  const cyR = Math.cos(ry), syR = Math.sin(ry);
  const czR = Math.cos(rz), szR = Math.sin(rz);
  const m00 = cyR * czR;
  const m01 = sxR * syR * czR - cxR * szR;
  const m02 = cxR * syR * czR + sxR * szR;
  const m10 = cyR * szR;
  const m11 = sxR * syR * szR + cxR * czR;
  const m12 = cxR * syR * szR - sxR * czR;
  const m20 = -syR;
  const m21 = sxR * cyR;
  const m22 = cxR * cyR;
  const out = [];
  const N = positions.length / 3;
  for (let i = 0; i < N; i++) {
    const px = positions[3 * i]     - cx;
    const py = positions[3 * i + 1] - cy;
    const pz = positions[3 * i + 2] - cz;
    const lx = m00 * px + m10 * py + m20 * pz;
    const ly = m01 * px + m11 * py + m21 * pz;
    const lz = m02 * px + m12 * py + m22 * pz;
    if (lx >= -hx && lx <= hx && ly >= -hy && ly <= hy && lz >= -hz && lz <= hz) out.push(i);
  }
  return out;
}

export function LabelMode({ cloud, theme, viewerRef, classes, instances, onChange, cloudBBox, navMode, onNavModeChange, segState, setSegState, prelabelRef, onCameraChange, hasMesh, isAnnotated, sessions, activeSessionId, presegs, onSelectSession, onCreateSession, onRenameSession, onDeleteSession, sessionLoading }) {
  const meshPopupRef = useRefLabel(null);
  const [activeClass, setActiveClass] = useStateLabel(classes[0]?.id || 'unknown');
  const [selectedId, setSelectedId] = useStateLabel(null);
  const [hiddenClasses, setHiddenClasses] = useStateLabel(new Set());
  const activeTool = 'cuboid';
  // Stateful so PresegmentButton can flip to 'instance' after a RANSAC
  // run — wildly different hues per segment make the grouping legible.
  const [colorMode] = useStateLabel('class');
  const [showDiff, setShowDiff] = useStateLabel(false);
  // Gizmo mode for the selected cuboid. null = no gizmo (edges only).
  const [transformMode, setTransformMode] = useStateLabel('translate');
  // Free-text filter for the instance list (matches label + class name).
  const [instFilter, setInstFilter] = useStateLabel('');
  // Which instance row is currently expanded for inline edit.
  const [editingId, setEditingId] = useStateLabel(null);
  // When true, points inside any confirmed cuboid are hidden from the main
  // viewport (NaN'd in the position buffer). Default on so the labeling
  // workflow naturally reveals what's left to label.
  const [hideConfirmed, setHideConfirmed] = useStateLabel(true);
  // 3D box-select: a transformable OBB the user drags via the existing
  // gizmo. On Confirm, every preseg with any point inside the box is toggled
  // into segState.selection. Same UX as the box in Edit mode.
  const [selBox, setSelBox] = useStateLabel(null);
  const showSegHulls = false;
  const [sideRCollapsed, setSideRCollapsed] = useStateLabel(() => {
    try { return localStorage.getItem('voxa.label.sideRCollapsed') === '1'; }
    catch { return false; }
  });
  useEffectLabel(() => {
    try { localStorage.setItem('voxa.label.sideRCollapsed', sideRCollapsed ? '1' : '0'); }
    catch { /* quota / private mode */ }
  }, [sideRCollapsed]);

  const diffMask = useMemoLabel(() => {
    if (!showDiff || !segState) return null;
    const pre = prelabelRef?.current;
    if (!pre?.classFull || !pre?.instanceFull) return null;
    return computeDiffMask(
      segState.classFull, pre.classFull,
      segState.instanceFull, pre.instanceFull,
    );
  }, [showDiff, segState, prelabelRef]);

  // Keep activeClass valid as the class list streams in.
  useEffectLabel(() => {
    if (classes.length && !classes.find((c) => c.id === activeClass)) {
      setActiveClass(classes[0].id);
    }
  }, [classes, activeClass]);

  // Yellow overlay for selected presegments. Recompute the per-subrow
  // mask whenever the selection or the underlying instance assignment
  // changes, then push it to the viewer's segSelection buffer.
  useEffectLabel(() => {
    const v = viewerRef?.current;
    if (!v?.setSelectedSegmentMask) return;
    if (!segState || !cloud?.positions) {
      v.setSelectedSegmentMask(null);
      return;
    }
    const sel = segState.selection;
    if (sel.size === 0) {
      v.setSelectedSegmentMask(null);
      return;
    }
    const inst = segState.instanceFull;
    const subIdx = cloud.subsampleIdx;
    const subN = cloud.positions.length / 3;
    const mask = new Uint8Array(subN);
    for (let p = 0; p < subN; p++) {
      const f = subIdx ? subIdx[p] : p;
      if (sel.has(inst[f])) mask[p] = 1;
    }
    v.setSelectedSegmentMask(mask);
  }, [segState?.selection, segState?.instanceFull, cloud, viewerRef]);

  // Ctrl/Cmd-click in the 3D viewport toggles selection of the presegment
  // under the cursor. Active in any tool mode whenever segment data exists.
  useEffectLabel(() => {
    if (!segState) return;
    const viewer = viewerRef?.current;
    if (!viewer?.onPointerPick) return;
    return viewer.onPointerPick((fullIndex, evt) => {
      if (!evt.ctrlKey && !evt.metaKey) return;
      const instId = segState.instanceFull[fullIndex];
      if (instId < 0) return;
      setSegState((s) => {
        if (!s) return s;
        const next = new Set(s.selection);
        next.has(instId) ? next.delete(instId) : next.add(instId);
        return { ...s, selection: next };
      });
    });
  }, [segState, viewerRef, setSegState]);

  // Hull-click selection: clicking directly on a hull face (Ctrl or plain click)
  // selects the segment. Works in all tool modes.
  useEffectLabel(() => {
    if (!segState) return;
    const viewer = viewerRef?.current;
    if (!viewer?.onHullPick) return;
    return viewer.onHullPick((segId, evt) => {
      if (!(evt.ctrlKey || evt.metaKey || evt.shiftKey)) return false;
      setSegState((s) => {
        if (!s) return s;
        const next = new Set(s.selection);
        next.has(segId) ? next.delete(segId) : next.add(segId);
        return { ...s, selection: next };
      });
      return true;
    });
  }, [segState, viewerRef, setSegState]);

  // Drop the selection box whenever the scene changes.
  useEffectLabel(() => { setSelBox(null); }, [cloud]);

  // Toggle the box select tool. Initializes the box to a quarter of the cloud
  // bbox centered at the cloud's center — small enough to be reachable, large
  // enough to grab with the gizmo handles.
  const toggleBoxSelect = useCallbackLabel(() => {
    setSelBox((b) => {
      if (b) return null;
      if (!cloudBBox) return null;
      const c = [
        (cloudBBox.min[0] + cloudBBox.max[0]) / 2,
        (cloudBBox.min[1] + cloudBBox.max[1]) / 2,
        (cloudBBox.min[2] + cloudBBox.max[2]) / 2,
      ];
      const s = [
        Math.max((cloudBBox.max[0] - cloudBBox.min[0]) / 4, 0.5),
        Math.max((cloudBBox.max[1] - cloudBBox.min[1]) / 4, 0.5),
        Math.max((cloudBBox.max[2] - cloudBBox.min[2]) / 4, 0.5),
      ];
      return {
        id: LABEL_SEL_BOX_ID,
        label: 'box-select',
        cls: 0,
        color: LABEL_SEL_BOX_COLOR,
        center: c, size: s, rotation: [0, 0, 0],
      };
    });
  }, [cloudBBox]);

  // Per-segment centroid memo. Sums positions per segment id and divides by
  // count → one centroid per preseg. Recomputed only when the cloud or the
  // segment assignment changes (not on selBox movement).
  const segCentroids = useMemoLabel(() => {
    if (!cloud?.positions || !segState?.instanceFull) return null;
    const positions = cloud.positions;
    const inst = segState.instanceFull;
    const subIdx = cloud.subsampleIdx;
    const subN = positions.length / 3;
    // First pass: find max segment id so we can size the accumulators.
    let maxId = -1;
    for (let p = 0; p < subN; p++) {
      const f = subIdx ? subIdx[p] : p;
      const id = inst[f];
      if (id > maxId) maxId = id;
    }
    if (maxId < 0) return null;
    const n = maxId + 1;
    const sx = new Float64Array(n);
    const sy = new Float64Array(n);
    const sz = new Float64Array(n);
    const cnt = new Uint32Array(n);
    for (let p = 0; p < subN; p++) {
      const x = positions[3 * p];
      // NaN sentinels (used by hideConfirmedPoints) get rejected here.
      if (!Number.isFinite(x)) continue;
      const f = subIdx ? subIdx[p] : p;
      const id = inst[f];
      if (id < 0) continue;
      sx[id] += x;
      sy[id] += positions[3 * p + 1];
      sz[id] += positions[3 * p + 2];
      cnt[id] += 1;
    }
    const cents = new Float32Array(n * 3);
    for (let id = 0; id < n; id++) {
      if (cnt[id] === 0) {
        // mark inactive with NaN so the OBB test rejects it
        cents[3 * id] = NaN; cents[3 * id + 1] = NaN; cents[3 * id + 2] = NaN;
      } else {
        cents[3 * id] = sx[id] / cnt[id];
        cents[3 * id + 1] = sy[id] / cnt[id];
        cents[3 * id + 2] = sz[id] / cnt[id];
      }
    }
    return cents;
  }, [cloud, segState?.instanceFull]);

  // Commit: select every preseg whose CENTROID falls inside the box. Centroid
  // test matches the visual mental model "is this segment in the box" much
  // better than the any-point test (a 100k-point segment poking one point
  // into the box no longer drags the whole segment in).
  const confirmBoxSelect = useCallbackLabel(() => {
    if (!selBox || !segCentroids) return;
    const inSet = pointsInsideOBBLabel(segCentroids, selBox);
    if (inSet.length === 0) { setSelBox(null); return; }
    setSegState((s) => {
      if (!s) return s;
      const next = new Set(s.selection);
      for (const segId of inSet) {
        next.has(segId) ? next.delete(segId) : next.add(segId);
      }
      return { ...s, selection: next };
    });
    setSelBox(null);
  }, [selBox, segCentroids, setSegState]);

  // Esc cancels; Enter commits. Skip when typing into an input.
  useEffectLabel(() => {
    if (!selBox) return;
    const onKey = (e) => {
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return;
      if (e.key === 'Escape') { e.preventDefault(); setSelBox(null); }
      else if (e.key === 'Enter') { e.preventDefault(); confirmBoxSelect(); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selBox, confirmBoxSelect]);

  // Only the selected cuboid renders in the viewer — keeps the scene readable
  // when there are dozens/hundreds of prelabel instances. Hidden classes still
  // hide everything in their class.
  const visibleIds = useMemoLabel(() => {
    if (!selectedId) return [];
    const sel = instances.find((i) => i.id === selectedId);
    if (!sel || hiddenClasses.has(sel.cls)) return [];
    // Pointset instances never get a cuboid drawn — they're a group of
    // points labeled with the same instance id and nothing more.
    if (sel.kind === 'pointset') return [];
    return [selectedId];
  }, [instances, hiddenClasses, selectedId]);

  const counts = useMemoLabel(() => {
    const c = {};
    instances.forEach((i) => { c[i.cls] = (c[i.cls] || 0) + 1; });
    return c;
  }, [instances]);

  // Set of segment ids absorbed into a right-side instance, so the left
  // PresegmentList can hide them once they've been promoted.
  const promotedSegIds = useMemoLabel(() => {
    const s = new Set();
    for (const i of instances) {
      if (i.kind === 'pointset' && Number.isFinite(i.segId)) s.add(i.segId);
    }
    return s;
  }, [instances]);

  // Hulls and per-segment AABBs for promoted segIds shouldn't render —
  // those points belong to a confirmed instance now, not a presegment.
  // Drop their faces / box entries before handing the data to the viewer.
  const segHullsFiltered = useMemoLabel(() => {
    const h = segState?.segHulls;
    if (!h) return null;
    if (promotedSegIds.size === 0) return h;
    const { vertices, faces, faceSeg } = h;
    const keepFaces = [];
    const keepSeg = [];
    for (let f = 0; f < faceSeg.length; f++) {
      if (promotedSegIds.has(faceSeg[f])) continue;
      keepFaces.push(faces[f * 3], faces[f * 3 + 1], faces[f * 3 + 2]);
      keepSeg.push(faceSeg[f]);
    }
    return {
      vertices,
      faces: new Uint32Array(keepFaces),
      faceSeg: new Int32Array(keepSeg),
    };
  }, [segState?.segHulls, promotedSegIds]);

  const segBoxesFiltered = useMemoLabel(() => {
    const b = segState?.segBoxes;
    if (!b) return null;
    if (promotedSegIds.size === 0) return b;
    const { segIds, segCenters, segSizes } = b;
    const keepIdx = [];
    for (let i = 0; i < segIds.length; i++) {
      if (!promotedSegIds.has(segIds[i])) keepIdx.push(i);
    }
    if (keepIdx.length === segIds.length) return b;
    const newIds = new Int32Array(keepIdx.length);
    const newCenters = new Float32Array(keepIdx.length * 3);
    const newSizes = new Float32Array(keepIdx.length * 3);
    for (let k = 0; k < keepIdx.length; k++) {
      const i = keepIdx[k];
      newIds[k] = segIds[i];
      newCenters[k * 3]     = segCenters[i * 3];
      newCenters[k * 3 + 1] = segCenters[i * 3 + 1];
      newCenters[k * 3 + 2] = segCenters[i * 3 + 2];
      newSizes[k * 3]     = segSizes[i * 3];
      newSizes[k * 3 + 1] = segSizes[i * 3 + 1];
      newSizes[k * 3 + 2] = segSizes[i * 3 + 2];
    }
    return { segIds: newIds, segCenters: newCenters, segSizes: newSizes };
  }, [segState?.segBoxes, promotedSegIds]);

  const selected = instances.find((i) => i.id === selectedId);
  const activeClassDef = classes.find((c) => c.id === activeClass);
  // Confirmed instances are read-only: no gizmo, no auto-fit, no rename, no
  // class change, no delete. The user reopens (toggles confirmed off) first.
  const isLocked = !!selected?.confirmed;

  // Pass-through for the viewer to highlight points inside the currently
  // selected cuboid. Updates as the box is dragged because `selected` is
  // re-derived from `instances` on every render.
  // Dense overlay: full-density LAZ points inside the selected cuboid.
  // Manually triggered (D hotkey) so the user controls when to pay the
  // load cost. Auto-clears whenever the selected cuboid moves/resizes so
  // a stale overlay never stays "stuck" beside the box after a drag.
  const [denseOverlay, setDenseOverlay] = useStateLabel(null);
  // Bumping this token causes the fetch effect to refire with the current
  // cuboid bounds. Token-based (not bounds-based) so we don't thrash the
  // backend on every gizmo tick.
  const [denseTrigger, setDenseTrigger] = useStateLabel(0);

  // Stable key for the selected cuboid's geometry. Whenever this changes
  // (drag, resize, rotate, deselect, switch to another instance) we drop
  // the overlay so it can't visibly disconnect from the box.
  const selectedBoundsKey = useMemoLabel(() => {
    if (!selected) return null;
    if (selected.kind === 'pointset' || !selected.center || !selected.size) return null;
    const c = selected.center, sz = selected.size, r = selected.rotation || [0, 0, 0];
    return `${selected.id}|${c[0]},${c[1]},${c[2]}|${sz[0]},${sz[1]},${sz[2]}|${r[0]},${r[1]},${r[2]}`;
  }, [selected]);

  useEffectLabel(() => { setDenseOverlay(null); }, [selectedBoundsKey]);

  useEffectLabel(() => {
    if (!denseTrigger) return;
    if (!selected) return;
    if (!selected.center || !selected.size) return;
    const center = selected.center;
    const size = selected.size;
    const rot = selected.rotation || [0, 0, 0];
    const hx = size[0] / 2, hy = size[1] / 2, hz = size[2] / 2;
    const m = new THREE.Matrix4().makeRotationFromEuler(
      new THREE.Euler(rot[0], rot[1], rot[2], 'XYZ')
    );
    const v = new THREE.Vector3();
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const sx of [-hx, hx]) for (const sy of [-hy, hy]) for (const sz of [-hz, hz]) {
      v.set(sx, sy, sz).applyMatrix4(m);
      if (v.x < minX) minX = v.x; if (v.y < minY) minY = v.y; if (v.z < minZ) minZ = v.z;
      if (v.x > maxX) maxX = v.x; if (v.y > maxY) maxY = v.y; if (v.z > maxZ) maxZ = v.z;
    }
    // Small margin so points right at the box edge are visible.
    const dx = (maxX - minX) * 0.10, dy = (maxY - minY) * 0.10, dz = (maxZ - minZ) * 0.10;
    const aabbMin = [center[0] + minX - dx, center[1] + minY - dy, center[2] + minZ - dz];
    const aabbMax = [center[0] + maxX + dx, center[1] + maxY + dy, center[2] + maxZ + dz];

    let cancelled = false;
    VoxaAPI.loadRegion(aabbMin, aabbMax, { maxPoints: 500_000 })
      .then((res) => {
        if (cancelled) return;
        setDenseOverlay({ positions: res.positions, colors: res.colors });
      })
      .catch(() => { if (!cancelled) setDenseOverlay(null); });
    return () => { cancelled = true; };
    // Token-based: only fires when the user explicitly hits D.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [denseTrigger]);

  // Mask of sub-cloud points belonging to the selected instance. Viewer dims
  // points where mask=0 so the selection visually pops from the rest of the
  // cloud. Pointset instances use segState.instanceFull membership; cuboid
  // instances test inside-OBB on subsampled positions. Recomputes whenever
  // selection (or, for cuboids, the box) changes.
  const selectionMask = useMemoLabel(() => {
    if (!selected || !cloud?.positions) return null;
    const subN = cloud.positions.length / 3;
    const mask = new Uint8Array(subN);
    // Prefer per-point membership whenever we have it. A pointset instance
    // always has segId; cuboid instances may also have one if they were
    // promoted from a presegment, in which case we still want point-accurate
    // highlight rather than box-filtered.
    if (Number.isFinite(selected.segId) && segState?.instanceFull) {
      const subIdx = cloud.subsampleIdx;
      const inst = segState.instanceFull;
      const target = selected.segId;
      for (let p = 0; p < subN; p++) {
        const f = subIdx ? subIdx[p] : p;
        if (inst[f] === target) mask[p] = 1;
      }
      return mask;
    }
    // Annotated SCHEMA scans expose per-point gt instance ids on the cloud
    // itself. If the instance carries a matching gt instance id, use it.
    if (Number.isFinite(selected.gtInstanceId) && cloud.instanceIds) {
      const ids = cloud.instanceIds;
      const target = selected.gtInstanceId;
      for (let p = 0; p < subN; p++) {
        if (ids[p] === target) mask[p] = 1;
      }
      return mask;
    }
    if (selected.kind !== 'pointset' && selected.center && selected.size) {
      const pos = cloud.positions;
      const cx = selected.center[0], cy = selected.center[1], cz = selected.center[2];
      const sz = selected.size, r = selected.rotation || [0, 0, 0];
      const hx = sz[0] / 2, hy = sz[1] / 2, hz = sz[2] / 2;
      const isAA = !r[0] && !r[1] && !r[2];
      if (isAA) {
        for (let p = 0; p < subN; p++) {
          const dx = pos[p * 3] - cx, dy = pos[p * 3 + 1] - cy, dz = pos[p * 3 + 2] - cz;
          if (dx > -hx && dx < hx && dy > -hy && dy < hy && dz > -hz && dz < hz) mask[p] = 1;
        }
      } else {
        const m = new THREE.Matrix4().makeRotationFromEuler(
          new THREE.Euler(r[0], r[1], r[2], 'XYZ')
        ).invert();
        const e = m.elements;
        for (let p = 0; p < subN; p++) {
          const dx = pos[p * 3] - cx, dy = pos[p * 3 + 1] - cy, dz = pos[p * 3 + 2] - cz;
          const lx = e[0] * dx + e[4] * dy + e[8]  * dz;
          const ly = e[1] * dx + e[5] * dy + e[9]  * dz;
          const lz = e[2] * dx + e[6] * dy + e[10] * dz;
          if (lx > -hx && lx < hx && ly > -hy && ly < hy && lz > -hz && lz < hz) mask[p] = 1;
        }
      }
      return mask;
    }
    return null;
  }, [selected, cloud, segState?.instanceFull]);  // cloud carries instanceIds; selected.segId/gtInstanceId fold into `selected`

  const highlightCuboid = useMemoLabel(() => {
    if (!selected) return null;
    if (selected.kind === 'pointset' || !selected.center || !selected.size) return null;
    const cls = classes.find((c) => c.id === selected.cls);
    return {
      center: selected.center,
      size: selected.size,
      rotation: selected.rotation || [0, 0, 0],
      color: cls?.color || selected.color,
    };
  }, [selected, classes]);

  // Stable string key for the confirmed subset. Used to short-circuit
  // hideCuboids' identity on gizmo drags of UNCONFIRMED instances (which
  // mutate `instances` every tick but leave the confirmed subset alone).
  const confirmedKey = useMemoLabel(() => {
    let s = '';
    for (const i of instances) {
      if (!i.confirmed) continue;
      // Pointset instances have no cuboid; skip them in confirmed-cuboid
      // bookkeeping so the viewer's hide/labeled-stats logic stays cuboid-only.
      if (i.kind === 'pointset' || !i.center || !i.size) continue;
      const c = i.center, sz = i.size, r = i.rotation || [0, 0, 0];
      s += `${i.id}|${c[0]},${c[1]},${c[2]}|${sz[0]},${sz[1]},${sz[2]}|${r[0]},${r[1]},${r[2]};`;
    }
    return s;
  }, [instances]);

  const confirmedCount = useMemoLabel(
    () => instances.reduce((n, i) => n + (i.confirmed ? 1 : 0), 0),
    [instances],
  );

  // Always populated when there are confirmed instances, regardless of the
  // hide toggle. The Viewer uses it to compute "points labeled / left" stats
  // as well as to optionally NaN positions.
  const confirmedCuboids = useMemoLabel(() => {
    if (!confirmedKey) return [];
    return instances
      // Confirmed-and-selected: temporarily un-hide so the user can re-inspect.
      .filter((i) => i.confirmed && i.id !== selectedId
        && i.kind !== 'pointset' && i.center && i.size)
      .map((i) => ({
        center: i.center,
        size: i.size,
        rotation: i.rotation || [0, 0, 0],
      }));
    // confirmedKey transitively covers `instances`; eslint can't see that.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [confirmedKey, selectedId]);

  // Hide mask for confirmed POINTSET instances (cuboid confirmed instances
  // are handled by confirmedCuboids' inside-test in the Viewer). Points
  // belonging to confirmed pointsets get NaN'd just like inside-cuboid points
  // when hideConfirmed is on. Selected confirmed pointset is excluded so
  // re-selecting it brings the points back.
  const confirmedPointsetHideMask = useMemoLabel(() => {
    if (!cloud?.positions) return null;
    const inst = segState?.instanceFull;
    if (!inst) return null;
    const targets = new Set();
    for (const i of instances) {
      if (!i.confirmed) continue;
      if (i.id === selectedId) continue;
      if (Number.isFinite(i.segId)) targets.add(i.segId);
    }
    if (targets.size === 0) return null;
    const subN = cloud.positions.length / 3;
    const subIdx = cloud.subsampleIdx;
    const mask = new Uint8Array(subN);
    for (let p = 0; p < subN; p++) {
      const f = subIdx ? subIdx[p] : p;
      if (targets.has(inst[f])) mask[p] = 1;
    }
    return mask;
  }, [instances, selectedId, cloud, segState?.instanceFull]);

  // Stats from Viewer's confirmed-mask pass: how many points fall inside any
  // confirmed cuboid (unique), regardless of show/hide toggle.
  const [labelStats, setLabelStats] = useStateLabel({ total: 0, labeled: 0, left: 0 });

  const filteredInstances = useMemoLabel(() => {
    const q = instFilter.trim().toLowerCase();
    if (!q) return instances;
    return instances.filter((inst) => {
      const cls = classes.find((c) => c.id === inst.cls);
      return (
        (inst.label || '').toLowerCase().includes(q) ||
        (cls?.label || inst.cls || '').toLowerCase().includes(q) ||
        (inst.id || '').toLowerCase().includes(q)
      );
    });
  }, [instances, classes, instFilter]);

  const helpSections = useMemoLabel(() => ([
    {
      title: 'Cuboid',
      items: [
        { keys: ['A'], desc: 'Add cuboid for active class' },
        { keys: ['G'], desc: 'Move (translate gizmo)' },
        { keys: ['R'], desc: 'Rotate gizmo' },
        { keys: ['Y'], desc: 'Scale gizmo' },
        { keys: ['F'], desc: 'Frame selection' },
        { keys: ['⌫'], desc: 'Delete selected' },
        { keys: ['Ctrl', '↵'], desc: 'Confirm selected (hides interior pts)' },
        { keys: ['⌘', 'S'], desc: 'Save annotations' },
      ],
    },
    {
      title: 'Class assignment',
      items: classes.length
        ? classes.map((c) => ({ keys: [c.hotkey], desc: c.label }))
        : [{ keys: ['—'], desc: 'No classes configured' }],
    },
    {
      title: 'Camera',
      items: navMode === 'walk'
        ? [
            { keys: ['W', 'A', 'S', 'D'], desc: 'Move (XZ plane)' },
            { keys: ['Q', 'E'], desc: 'Down / up' },
            { keys: ['Shift'], desc: 'Hold to sprint' },
            { keys: ['Drag'], desc: 'Look around' },
            { keys: ['Scroll'], desc: 'Step forward / back' },
          ]
        : [
            { keys: ['Drag'], desc: 'Orbit' },
            { keys: ['Shift', 'Drag'], desc: 'Pan' },
            { keys: ['Right', 'Drag'], desc: 'Pan' },
            { keys: ['Scroll'], desc: 'Zoom' },
          ],
    },
    {
      title: 'Mouse',
      items: [
        { keys: ['Dbl-click'], desc: 'Select cuboid in right list (shows box)' },
        { keys: ['✎'], desc: 'Edit button selects + opens panel' },
        { keys: ['Drag', 'gizmo'], desc: 'Move / rotate / scale selected' },
      ],
    },
    {
      title: 'Other',
      items: [
        { keys: ['?'], desc: 'Toggle this panel' },
        { keys: ['Esc'], desc: 'Close panel' },
      ],
    },
  ]), [classes, navMode]);

  const toggleClass = (cls) => {
    setHiddenClasses((s) => {
      const n = new Set(s);
      n.has(cls) ? n.delete(cls) : n.add(cls);
      return n;
    });
  };

  // Add a cuboid centered on the loaded scene's bbox center, sized as a small
  // cube. The user then nudges it via the inspector or auto-fit.
  const addCuboid = useCallbackLabel(async () => {
    if (!cloudBBox || !activeClassDef) return;
    const cx = (cloudBBox.min[0] + cloudBBox.max[0]) / 2;
    const cy = (cloudBBox.min[1] + cloudBBox.max[1]) / 2;
    const cz = (cloudBBox.min[2] + cloudBBox.max[2]) / 2;
    const ext = Math.max(
      cloudBBox.max[0] - cloudBBox.min[0],
      cloudBBox.max[1] - cloudBBox.min[1],
      cloudBBox.max[2] - cloudBBox.min[2],
    );
    const s = Math.max(0.05, ext * 0.1);
    const inst = {
      id: newId(),
      cls: activeClassDef.id,
      label: `${activeClassDef.label} ${(counts[activeClassDef.id] || 0) + 1}`,
      color: activeClassDef.color,
      center: [cx, cy, cz],
      size: [s, s, s],
      rotation: [0, 0, 0],
      conf: 1.0,
      source: 'manual',
    };
    const next = [...instances, inst];
    onChange(next);
    setSelectedId(inst.id);
  }, [activeClassDef, instances, cloudBBox, counts, onChange]);

  const updateSelected = (patch) => {
    const next = instances.map((i) => i.id === selectedId ? { ...i, ...patch } : i);
    onChange(next);
  };

  const updateInstance = (id, patch) => {
    onChange(instances.map((i) => i.id === id ? { ...i, ...patch } : i));
  };
  const deleteInstance = (id) => {
    onChange(instances.filter((i) => i.id !== id));
    if (selectedId === id) setSelectedId(null);
    if (editingId === id) setEditingId(null);
  };
  const autoFitInstance = async (inst) => {
    const half = inst.size.map((v) => v / 2);
    const cmin = [inst.center[0] - half[0], inst.center[1] - half[1], inst.center[2] - half[2]];
    const cmax = [inst.center[0] + half[0], inst.center[1] + half[1], inst.center[2] + half[2]];
    const cls = classes.find((c) => c.id === inst.cls);
    const fitted = await VoxaAPI.autoFit(cmin, cmax, inst.cls, cls?.color || inst.color, inst.label);
    updateInstance(inst.id, { center: fitted.center, size: fitted.size });
  };
  const focusInstance = (inst) => {
    if (!inst) return;
    // Pointset instance: derive bbox from the points labeled with segId.
    if (inst.kind === 'pointset' && Number.isFinite(inst.segId)
        && segState && cloud?.positions) {
      const pos = cloud.positions;
      const subIdx = cloud.subsampleIdx;
      const instArr = segState.instanceFull;
      let minX = Infinity, minY = Infinity, minZ = Infinity;
      let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
      const subN = pos.length / 3;
      for (let p = 0; p < subN; p++) {
        const f = subIdx ? subIdx[p] : p;
        if (instArr[f] !== inst.segId) continue;
        const x = pos[p * 3], y = pos[p * 3 + 1], z = pos[p * 3 + 2];
        if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
        if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;
      }
      if (!isFinite(minX)) return;
      const center = new THREE.Vector3(
        (minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);
      const radius = Math.max(maxX - minX, maxY - minY, maxZ - minZ) * 0.6 + 0.05;
      viewerRef.current?.frame(center, radius);
      return;
    }
    if (!inst.center || !inst.size) return;
    viewerRef.current?.frame(
      new THREE.Vector3(...inst.center),
      Math.max(...inst.size) / 2,
    );
  };
  // Toggle confirmed state. When transitioning to confirmed, clear the
  // selection so the cuboid edges + gizmo disappear alongside the (now
  // hidden) interior points — the visual signal that the instance is "done".
  const toggleConfirm = (id) => {
    const target = instances.find((i) => i.id === id);
    if (!target) return;
    const willConfirm = !target.confirmed;
    onChange(instances.map((i) => i.id === id ? { ...i, confirmed: willConfirm } : i));
    if (willConfirm) {
      if (selectedId === id) setSelectedId(null);
      if (editingId === id) setEditingId(null);
    }
  };
  // Ctrl+Enter on a presegment selection: collapse the selected segments
  // into one new point-set instance. The backend reassigns the union of
  // their points to a fresh instance id (target_class = active class) and
  // returns the allocated id in r.afterInstance. The instance is added
  // to the right-side list as a *pointset* (no center/size — never drawn
  // as a cuboid) and the segments it absorbed disappear from the left
  // PresegmentList (filtered via promotedSegIds). Selection clears.
  // Class picker modal state. When set, Ctrl+Enter on a selection deferred
  // to a class choice instead of using activeClass; pressing the class
  // hotkey picks that class and creates the (unconfirmed) pointset.
  const [classPickerOpen, setClassPickerOpen] = useStateLabel(false);

  const confirmSegmentSelection = useCallbackLabel(async (clsDef) => {
    const targetCls = clsDef || activeClassDef;
    if (!segState || segState.selection.size === 0) return;
    if (!targetCls) return;
    const inst = segState.instanceFull;
    const sel = segState.selection;
    const idx = [];
    for (let p = 0; p < inst.length; p++) {
      if (sel.has(inst[p])) idx.push(p);
    }
    if (idx.length === 0) return;
    const indices = new Int32Array(idx);

    let r;
    try {
      r = await VoxaAPI.segApply('reassign', {
        indices,
        payload: { target_inst: -1, target_class: targetCls.id },
      });
    } catch (err) {
      console.error('confirm reassign failed:', err);
      return;
    }

    const newSegId = r.afterInstance && r.afterInstance.length > 0
      ? r.afterInstance[0] : -1;
    if (newSegId >= 0) {
      const newInst = {
        id: newId(),
        segId: newSegId,
        kind: 'pointset',
        cls: targetCls.id,
        label: `${targetCls.label} ${(counts[targetCls.id] || 0) + 1}`,
        color: targetCls.color,
        source: 'preseg',
      };
      onChange([...instances, newInst]);
    }

    setSegState((s) => {
      if (!s) return s;
      const next = applyDelta(s, {
        indices: r.indices,
        after_class: r.afterClass,
        after_instance: r.afterInstance,
      });
      return { ...next, selection: new Set() };
    });
  }, [segState, activeClassDef, instances, counts, onChange, setSegState]);

  const toggleConfirmSelected = useCallbackLabel(() => {
    if (!selectedId) return;
    const target = instances.find((i) => i.id === selectedId);
    if (!target) return;
    const willConfirm = !target.confirmed;
    onChange(instances.map((i) => i.id === selectedId ? { ...i, confirmed: willConfirm } : i));
    if (willConfirm) {
      setSelectedId(null);
      if (editingId === selectedId) setEditingId(null);
    }
  }, [selectedId, editingId, instances, onChange]);

  // Gizmo drag callback. Patches the targeted instance by id (not by selectedId)
  // since the viewer dispatches based on its own gizmoTargetIdRef snapshot.
  const onCuboidTransform = useCallbackLabel((id, patch) => {
    if (id === LABEL_SEL_BOX_ID) {
      setSelBox((b) => b ? { ...b, ...patch } : b);
      return;
    }
    const next = instances.map((i) => i.id === id ? { ...i, ...patch } : i);
    onChange(next);
  }, [instances, onChange]);

  const deleteSelected = () => {
    if (!selectedId) return;
    onChange(instances.filter((i) => i.id !== selectedId));
    setSelectedId(null);
  };

  const autoFitSelected = async () => {
    if (!selected) return;
    const half = selected.size.map((v) => v / 2);
    const cmin = [selected.center[0] - half[0], selected.center[1] - half[1], selected.center[2] - half[2]];
    const cmax = [selected.center[0] + half[0], selected.center[1] + half[1], selected.center[2] + half[2]];
    const fitted = await VoxaAPI.autoFit(cmin, cmax, selected.cls,
      activeClassDef?.color || selected.color, selected.label);
    updateSelected({ center: fitted.center, size: fitted.size });
  };

  // Hotkeys: 0–9 assign class, ⌫ delete, A add, F frame, ⌘S save.
  // In walk mode the viewer owns WASD/QE; bail on those keys here so we
  // don't double-fire (e.g. 'A' is both walk-left and add-cuboid).
  // Gated on activeTool === 'cuboid' so Pick/Brush tools own their own hotkeys.
  useEffectLabel(() => {
    const onKey = (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      // Ctrl/Cmd+Enter is tool-agnostic: with a presegment selection it
      // collapses the selection into a new instance; otherwise it toggles
      // the confirmed flag on the active cuboid (the legacy behaviour).
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (segState && segState.selection.size > 0) {
          // Open the class picker so the user can quick-pick the class for
          // the new (unconfirmed) pointset instead of falling back on the
          // activeClass. The picker has its own keydown handler.
          setClassPickerOpen(true);
        } else if (activeTool === 'cuboid') {
          toggleConfirmSelected();
        }
        return;
      }
      if (activeTool !== 'cuboid') return;
      if (navMode === 'walk' && /^[wasdqeWASDQE]$/.test(e.key)) return;
      const cls = classes.find((c) => c.hotkey === e.key);
      if (cls) {
        setActiveClass(cls.id);
        // Class change is an edit — block it for confirmed instances.
        if (selected && !isLocked) updateSelected({ cls: cls.id, color: cls.color });
      } else if (e.key === 'Backspace' || e.key === 'Delete') {
        if (selected && !isLocked) { e.preventDefault(); deleteSelected(); }
      } else if (e.key === 'a' || e.key === 'A') {
        addCuboid();
      } else if (e.key === 'f' || e.key === 'F') {
        if (selected) {
          viewerRef.current?.frame(
            new THREE.Vector3(...selected.center),
            Math.max(...selected.size) / 2,
          );
        }
      } else if ((!isLocked || !!selBox) && (e.key === 'g' || e.key === 'G')) {
        setTransformMode('translate');
      } else if ((!isLocked || !!selBox) && (e.key === 'r' || e.key === 'R')) {
        setTransformMode('rotate');
      } else if ((!isLocked || !!selBox) && (e.key === 'y' || e.key === 'Y')) {
        setTransformMode('scale');
      } else if (e.key === 'd' || e.key === 'D') {
        // Densify: pop full-density LAZ points inside the selected cuboid.
        // Manual trigger so we don't refetch on every gizmo drag tick. The
        // overlay auto-clears when the box moves or the cuboid is deselected.
        if (selected) setDenseTrigger((t) => t + 1);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line
  }, [classes, selected, isLocked, instances, activeTool, navMode, segState, confirmSegmentSelection]);

  return (
    <div className="mode-root label">
      {classPickerOpen && (
        <ClassPickerModal
          classes={classes}
          counts={counts}
          onPick={(cls) => {
            setClassPickerOpen(false);
            confirmSegmentSelection(cls);
          }}
          onClose={() => setClassPickerOpen(false)}
        />
      )}

      {/* Left: class palette */}
      <aside className="side-l">
        {isAnnotated && (
          <SessionPicker
            sessions={sessions}
            activeSessionId={activeSessionId}
            presegs={presegs}
            loading={sessionLoading}
            onSelect={onSelectSession}
            onCreate={onCreateSession}
            onRename={onRenameSession}
            onDelete={onDeleteSession}
          />
        )}
        <div className="side-hd">
          <span>Classes</span>
          <span className="badge-soft">{instances.length}</span>
        </div>
        <div className="class-list">
          {classes.map((c) => {
            const hidden = hiddenClasses.has(c.id);
            return (
              <div key={c.id}
                className={'class-row' + (activeClass === c.id ? ' active' : '') + (hidden ? ' hidden' : '')}
                onClick={() => setActiveClass(c.id)}>
                <span className="class-swatch" style={{ background: c.color }} />
                <span className="class-name">{c.label}</span>
                <span className="class-count">{counts[c.id] || 0}</span>
                <button className="class-eye" onClick={(e) => { e.stopPropagation(); toggleClass(c.id); }}
                  title={hidden ? 'Show' : 'Hide'}>{hidden ? '◌' : '●'}</button>
                <span className="class-hk">{c.hotkey}</span>
              </div>
            );
          })}
        </div>

        <PresegmentList
          segState={segState}
          setSegState={setSegState}
          classes={classes}
          viewerRef={viewerRef}
          cloud={cloud}
          excludeSegIds={promotedSegIds}
        />
      </aside>

      {/* Center: viewport */}
      <div className="vp-stack">
        <Viewer
          ref={viewerRef}
          cloud={cloud}
          instances={selBox ? [...instances, selBox] : instances}
          visibleInstanceIds={selBox ? [LABEL_SEL_BOX_ID] : visibleIds}
          selectedId={selBox ? LABEL_SEL_BOX_ID : selectedId}
          showCuboids
          background={theme.bg}
          floorColor={theme.floor}
          navMode={navMode}
          colorMode={colorMode}
          pointSize={0.012}
          diffMask={diffMask}
          showDiff={showDiff}
          transformMode={selBox ? (transformMode || 'translate') : (activeTool === 'cuboid' && !isLocked ? transformMode : null)}
          onCuboidTransform={onCuboidTransform}
          highlightCuboid={highlightCuboid}
          selectionMask={selectionMask}
          denseOverlay={denseOverlay}
          confirmedCuboids={confirmedCuboids}
          confirmedPointsetHideMask={confirmedPointsetHideMask}
          hideConfirmedPoints={hideConfirmed}
          onLabelStats={setLabelStats}
          onCameraChange={onCameraChange}
          segBoxes={segBoxesFiltered
            ? { ...segBoxesFiltered, selection: segState.selection }
            : null}
          segHulls={segHullsFiltered
            ? { ...segHullsFiltered, selection: segState.selection }
            : null}
          showSegHulls={showSegHulls}
        />

        <div className="vp-hud-top">
          <div className="hud-group">
            {labelStats.total > 0 && (
              <HUDChip label="Points left"
                value={`${formatPointCount(labelStats.left)} / ${formatPointCount(labelStats.total)}`}
                mono />
            )}
          </div>
          <div className="hud-group">
            <NavModeToggle navMode={navMode} onChange={onNavModeChange} />
            <CameraPresets onPreset={(p) => viewerRef.current?.preset(p)} />
            <button className="hud-chip-btn"
              onClick={() => {
                // Reuse a tracked popup ref instead of relying on window.name
                // dedupe. Some browsers (Brave with strict popup rules,
                // Firefox under certain blockers) react to a denied popup
                // by stamping the requested name onto the *calling* tab —
                // every subsequent click then navigates the main app to
                // ?mesh=1 instead of opening a popup.
                const w = meshPopupRef.current;
                if (w && !w.closed) { w.focus(); return; }
                meshPopupRef.current = window.open(
                  window.location.pathname + '?mesh=1',
                  '_blank',
                  'popup=yes,width=960,height=720',
                );
              }}
              disabled={!hasMesh}
              title={hasMesh
                ? 'Open synced mesh-only companion window'
                : 'No mesh available for this scene'}>
              ▦ Mesh window
            </button>
          </div>
        </div>

        <div className="vp-help-corner">
          <HelpButton sections={helpSections} placement="up" />
        </div>

        <ViewportToolbar side="left">
          {activeTool === 'cuboid' && (
            <>
              {(!isLocked || !!selBox) && (
                <>
                  <ToolButton mini icon="⇄" label="Move (G)"
                    onClick={() => setTransformMode('translate')}
                    active={transformMode === 'translate'} />
                  <ToolButton mini icon="↻" label="Rotate (R)"
                    onClick={() => setTransformMode('rotate')}
                    active={transformMode === 'rotate'} />
                  <ToolButton mini icon="⇲" label="Scale (Y)"
                    onClick={() => setTransformMode('scale')}
                    active={transformMode === 'scale'} />
                </>
              )}
              {selected && (
                <>
                  <div className="tool-sep" />
                  <ToolButton mini icon="◎" label="Focus selection (F)"
                    onClick={() => focusInstance(selected)} />
                  {!isLocked && (
                    <>
                      <ToolButton mini icon="✦" label="Auto-fit selection" onClick={autoFitSelected} />
                      <ToolButton mini icon="⌫" label="Delete selection" onClick={deleteSelected} />
                    </>
                  )}
                </>
              )}
            </>
          )}
          {segState?.isFromPrelabel && (
            <ToolButton mini
              icon="Δ"
              label={showDiff ? 'Hide diff' : 'Diff vs prelabel'}
              onClick={() => setShowDiff((v) => !v)}
              active={showDiff}
            />
          )}
          {segState && (
            <>
              <ToolButton mini
                icon="◫"
                label={selBox ? 'Cancel box (Esc)' : 'Box-select segments'}
                onClick={toggleBoxSelect}
                active={!!selBox}
              />
              {selBox && (
                <ToolButton mini
                  icon="✓"
                  label="Confirm box (Enter)"
                  onClick={confirmBoxSelect}
                />
              )}
            </>
          )}
          <ToolButton mini icon="↺" label="Reset cam" onClick={() => viewerRef.current?.preset('iso')} />
        </ViewportToolbar>
      </div>

      {/* Right: filterable instance list + slim inspector */}
      <aside className={'side-r' + (sideRCollapsed ? ' collapsed' : '')}>
        {sideRCollapsed ? (
          <button className="side-collapse-handle"
            onClick={() => setSideRCollapsed(false)}
            title={`Show instances panel (${instances.length})`}>
            <span className="side-collapse-chev">‹</span>
            <span className="side-collapse-label">Instances</span>
            <span className="badge-soft">{instances.length}</span>
          </button>
        ) : (
        <>
        <div className="side-hd">
          <button className="side-collapse-btn"
            onClick={() => setSideRCollapsed(true)}
            title="Collapse panel">›</button>
          <span>Instances</span>
          <div className="side-hd-actions">
            {confirmedCount > 0 && (
              <button className="hide-labeled-btn"
                onClick={() => setHideConfirmed((v) => !v)}
                title={hideConfirmed
                  ? `Show ${confirmedCount} labeled instance${confirmedCount === 1 ? '' : 's'}`
                  : `Hide ${confirmedCount} labeled instance${confirmedCount === 1 ? '' : 's'}`}>
                {hideConfirmed ? '◌' : '●'} {confirmedCount} done
              </button>
            )}
            <span className="badge-soft">
              {instFilter ? `${filteredInstances.length} / ${instances.length}` : instances.length}
            </span>
          </div>
        </div>
        <div className="inst-filter">
          <input className="ins-input"
            placeholder="Filter by label, class, or id"
            value={instFilter}
            onChange={(e) => setInstFilter(e.target.value)} />
          {instFilter && (
            <button className="inst-filter-clear"
              onClick={() => setInstFilter('')}
              title="Clear filter">×</button>
          )}
        </div>
        <div className="inst-list">
          {instances.length === 0 && (
            <div className="sugg-empty">No instances yet. Press <kbd>A</kbd> to add.</div>
          )}
          {instances.length > 0 && filteredInstances.length === 0 && (
            <div className="sugg-empty">No matches for "{instFilter}".</div>
          )}
          {filteredInstances.map((inst) => {
            const cls = classes.find((c) => c.id === inst.cls);
            const isSel = inst.id === selectedId;
            const isEditing = inst.id === editingId;
            return (
              <div key={inst.id} className={'inst-item' + (isEditing ? ' editing' : '')}>
                <div className={'inst-row' + (isSel ? ' selected' : '') + (inst.confirmed ? ' confirmed' : '')}
                  onDoubleClick={() => setSelectedId(isSel ? null : inst.id)}
                  title={isSel ? 'Double-click to deselect' : 'Double-click to select (shows bounding box)'}>
                  <span className="inst-dot" style={{ background: cls?.color || inst.color }} />
                  <div className="inst-text">
                    <b>{inst.label}</b>
                    <em>{cls?.label || inst.cls}</em>
                  </div>
                  <button className={'inst-edit-btn' + (inst.confirmed ? ' is-confirmed' : '')}
                    onClick={(e) => { e.stopPropagation(); toggleConfirm(inst.id); }}
                    title={inst.confirmed ? 'Reopen (mark as unlabeled)' : 'Confirm (Ctrl+Enter)'}>
                    {inst.confirmed ? '✓' : '○'}
                  </button>
                  <button className="inst-edit-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedId(inst.id);
                      focusInstance(inst);
                    }}
                    title="Focus camera on this instance">◎</button>
                  <button className="inst-edit-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (isEditing) {
                        setEditingId(null);
                      } else {
                        setSelectedId(inst.id);
                        setEditingId(inst.id);
                      }
                    }}
                    title={isEditing ? 'Close' : 'Edit (selects + opens panel)'}>{isEditing ? '×' : '✎'}</button>
                </div>
                {isEditing && (
                  <div className="inst-edit-panel">
                    {inst.confirmed && (
                      <div className="locked-banner">🔒 Confirmed — Reopen to edit</div>
                    )}
                    <div className="ins-row">
                      <label>Name</label>
                      <input className="ins-input"
                        value={inst.label}
                        autoFocus
                        disabled={inst.confirmed}
                        onChange={(e) => updateInstance(inst.id, { label: e.target.value })} />
                    </div>
                    <div className="ins-row">
                      <label>Class</label>
                      <div className="class-pills">
                        {classes.map((c) => (
                          <button key={c.id}
                            className={'class-pill' + (c.id === inst.cls ? ' active' : '')}
                            disabled={inst.confirmed}
                            onClick={() => updateInstance(inst.id, { cls: c.id, color: c.color })}
                            title={`${c.label}${c.hotkey ? `  (${c.hotkey})` : ''}`}>
                            <span className="class-swatch" style={{ background: c.color }} />
                            <span>{c.label}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="ins-actions">
                      <button className="ghost-btn" onClick={() => focusInstance(inst)}>◎ Focus</button>
                      {!inst.confirmed && inst.kind !== 'pointset' && (
                        <button className="ghost-btn" onClick={() => autoFitInstance(inst)}>↻ Auto-fit</button>
                      )}
                      <button className="ghost-btn" onClick={() => toggleConfirm(inst.id)}
                        title={inst.confirmed ? 'Reopen' : 'Confirm (Ctrl+Enter)'}>
                        {inst.confirmed ? '✓ Reopen' : '✓ Confirm'}
                      </button>
                      {!inst.confirmed && (
                        <button className="ghost-btn danger" onClick={() => deleteInstance(inst.id)}>Delete</button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
        </>
        )}
      </aside>
    </div>
  );
}

// Centered modal shown after Ctrl+Enter on a presegment selection. Lists all
// classes with their hotkeys; pressing the hotkey (or clicking) commits the
// selection as a new unconfirmed pointset instance with that class. Esc
// dismisses without creating anything. Selection survives a cancel so the
// user can pick again or hit Ctrl+Enter once more.
function ClassPickerModal({ classes, counts, onPick, onClose }) {
  useEffectLabel(() => {
    const onKey = (e) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
        return;
      }
      const cls = classes.find((c) => c.hotkey === e.key);
      if (cls) {
        e.preventDefault();
        e.stopPropagation();
        onPick(cls);
      }
    };
    // Capture phase so we beat the LabelMode global keydown that would
    // otherwise also handle the hotkey (e.g. "1" → setActiveClass).
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [classes, onPick, onClose]);

  return (
    <div className="class-picker-overlay" onClick={onClose}>
      <div className="class-picker-card" onClick={(e) => e.stopPropagation()}>
        <div className="class-picker-title">Pick class for new instance</div>
        <div className="class-picker-list">
          {classes.map((c) => (
            <button key={c.id}
              className="class-picker-row"
              onClick={() => onPick(c)}
              title={`Press ${c.hotkey || '–'}`}>
              <span className="class-swatch" style={{ background: c.color }} />
              <span className="class-picker-label">{c.label}</span>
              <span className="class-picker-count">{counts[c.id] || 0}</span>
              <span className="class-picker-hk">{c.hotkey || '–'}</span>
            </button>
          ))}
        </div>
        <div className="class-picker-hint">Press a number to assign · Esc to cancel</div>
      </div>
    </div>
  );
}

