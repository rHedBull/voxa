// mode-label.jsx — class palette (left), instance list (right), cuboid editing.

import { useState as useStateLabel, useMemo as useMemoLabel,
         useEffect as useEffectLabel, useCallback as useCallbackLabel,
         useRef as useRefLabel } from 'react';
import * as THREE from 'three';
import { Viewer } from './viewer.jsx';
import { ViewportToolbar, ToolButton, HUDChip, CameraPresets, NavModeToggle, HelpButton } from './viewport-atoms.jsx';
import { VoxaAPI, newId } from './api.js';
import { focusSegment } from './segment-tools.jsx';
import { deriveFastQueue, stepIndex, FastLabelKeys, FastLabelHUD,
         FastConfirmModal, FAST_HIGHLIGHT_COLOR } from './fast-label.jsx';
import SessionPicker from './session-picker.jsx';
import { applyDelta, computeDiffMask } from './segment-state.js';
import { toolAvailable, defaultTool } from './label-tools.js';
import ToolRail from './tool-rail.jsx';
import ToolOptions from './tool-options.jsx';

// "30k", "1.2M", "523" — keeps the HUD chip narrow regardless of scene size.
function formatPointCount(n) {
  if (n < 1000) return String(n);
  if (n < 1e6) return `${(n / 1e3).toFixed(n < 10000 ? 1 : 0)}k`;
  return `${(n / 1e6).toFixed(n < 1e7 ? 2 : 1)}M`;
}

const LABEL_SEL_BOX_ID = '__label_sel_box__';
const LABEL_SEL_BOX_COLOR = '#ffd24a';

export function LabelMode({ cloud, theme, viewerRef, classes, instances, onChange, cloudBBox, navMode, onNavModeChange, segState, setSegState, prelabelRef, onCameraChange, hasMesh, isAnnotated, sessions, activeSessionId, presegs, onSelectSession, onCreateSession, onRenameSession, onDeleteSession, sessionLoading }) {
  const meshPopupRef = useRefLabel(null);
  const [activeClass, setActiveClass] = useStateLabel(classes[0]?.id || 'unknown');
  const [selectedId, setSelectedId] = useStateLabel(null);
  const [hiddenClasses, setHiddenClasses] = useStateLabel(new Set());
  const [activeTool, setActiveTool] = useStateLabel(() =>
    defaultTool({ segState, isAnnotated }));
  // Presegment "rapid" = the old fast-labeling queue.
  const [presegRapid, setPresegRapid] = useStateLabel(false);

  // Derived legacy flags — keep the existing body working during the refactor.
  const fastMode = activeTool === 'presegment' && presegRapid;
  const drawMode = activeTool === 'draw';
  // Presegmentation is a way to *select* points; its segments (hulls, boxes,
  // per-segment hue coloring) only show while the Presegment tool is active.
  // Every other tool works on the raw RGB cloud.
  const isPreseg = activeTool === 'presegment';

  // Per-tool auto-confirm (added here to avoid a forward reference in Tasks 8/9;
  // threaded into apply paths in Task 10).
  const [autoConfirm, setAutoConfirm] = useStateLabel({ box: false, draw: false, presegment: false });
  const autoConfirmFor = (tool) =>
    tool === 'presegment' ? (presegRapid || autoConfirm.presegment) : !!autoConfirm[tool];
  // Stateful so PresegmentButton can flip to 'instance' after a RANSAC
  // run — wildly different hues per segment make the grouping legible.
  const [colorMode] = useStateLabel('class');
  // Draw works on the raw RGB cloud, where bumping the point size makes the
  // sparse subsample read denser (same slider as Inspect).
  const [pointSize, setPointSize] = useStateLabel(0.012);
  const [showDiff, setShowDiff] = useStateLabel(false);
  // Gizmo mode for the selected cuboid. null = no gizmo (edges only).
  const [transformMode, setTransformMode] = useStateLabel('translate');
  // Free-text filter for the instance list (matches label + class name).
  const [instFilter, setInstFilter] = useStateLabel('');
  const [instStatus, setInstStatus] = useStateLabel('all');
  // Which instance row is currently expanded for inline edit.
  const [editingId, setEditingId] = useStateLabel(null);
  // When true, points inside any confirmed cuboid are hidden from the main
  // viewport (NaN'd in the position buffer). Default on so the labeling
  // workflow naturally reveals what's left to label.
  const [hideConfirmed, setHideConfirmed] = useStateLabel(true);
  // At most one Label sub-mode is active. Fast labeling steps through
  // unpromoted presegments largest-first and confirms a class per segment;
  // Draw labels pipes/tanks via centerline tubes. Declared here because the
  // selection-overlay effect needs fastMode for the orange highlight. Queue/
  // handlers live further down (they need promotedSegIds + confirmSegmentSelection).
  const [fastPos, setFastPos] = useStateLabel(0);
  const [fastPendingCls, setFastPendingCls] = useStateLabel(null);
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

  // Never leave an unavailable tool active (e.g. after switching to a scene
  // with no segState or a non-annotated scan).
  useEffectLabel(() => {
    if (!toolAvailable(activeTool, { segState, isAnnotated })) {
      setActiveTool(defaultTool({ segState, isAnnotated }));
    }
  }, [segState, isAnnotated, activeTool]);

  // The box-select gizmo belongs to the Box tool only; leaving Box must clear it
  // so a stale box + gizmo can't render over / hijack Presegment or Draw.
  useEffectLabel(() => {
    if (activeTool !== 'box') setSelBox(null);
  }, [activeTool]);

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
    v.setSelectedSegmentMask(mask, fastMode ? FAST_HIGHLIGHT_COLOR : undefined);
  }, [segState?.selection, segState?.instanceFull, cloud, viewerRef, fastMode]);

  // Ctrl/Cmd-click in the 3D viewport toggles selection of the presegment
  // under the cursor. Active in any tool mode whenever segment data exists.
  // Draw mode owns all pointer events; skip registration while it's on.
  useEffectLabel(() => {
    if (!segState) return;
    if (drawMode) return;
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
  }, [segState, viewerRef, setSegState, drawMode]);

  // Hull-click selection: clicking directly on a hull face (Ctrl or plain click)
  // selects the segment. Works in all tool modes.
  useEffectLabel(() => {
    if (!segState) return;
    if (drawMode) return;
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
  }, [segState, viewerRef, setSegState, drawMode]);

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

  // Esc cancels the box-select. (Enter no longer centroid-selects — the Box
  // tool supersedes that gesture; apply is Ctrl+Enter or a class hotkey.)
  useEffectLabel(() => {
    if (!selBox) return;
    const onKey = (e) => {
      const t = e.target;
      if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA')) return;
      if (e.key === 'Escape') { e.preventDefault(); setSelBox(null); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [selBox]);

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
    return instances.filter((inst) => {
      if (instStatus === 'unconfirmed' && inst.confirmed) return false;
      if (instStatus === 'confirmed' && !inst.confirmed) return false;
      if (!q) return true;
      const cls = classes.find((c) => c.id === inst.cls);
      return (
        (inst.label || '').toLowerCase().includes(q) ||
        (cls?.label || inst.cls || '').toLowerCase().includes(q) ||
        (inst.id || '').toLowerCase().includes(q)
      );
    });
  }, [instances, classes, instFilter, instStatus]);

  const helpSections = useMemoLabel(() => ([
    {
      title: 'Tools',
      items: [
        { keys: ['Rail'], desc: 'Switch Presegment / Box / Draw' },
        { keys: ['Ctrl', '↵'], desc: 'Apply selection (pick class)' },
        { keys: ['0–9'], desc: 'Apply selection with that class' },
        { keys: ['✓'], desc: 'Confirm instance (row button)' },
        { keys: ['G'], desc: 'Move box (translate gizmo)' },
        { keys: ['R'], desc: 'Rotate box' },
        { keys: ['Y'], desc: 'Scale box' },
        { keys: ['F'], desc: 'Frame selection' },
        { keys: ['⌫'], desc: 'Delete selected' },
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

  const confirmSegmentSelection = useCallbackLabel(async (clsDef, opts) => {
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
        // Fast labeling promotes + confirms in one step (no review pass).
        confirmed: !!(opts?.confirmed ?? autoConfirmFor('presegment')),
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
  }, [segState, activeClassDef, instances, counts, onChange, setSegState, autoConfirm, presegRapid]);

  // Box tool apply: send the OBB to the backend, which labels every FULL-RES
  // point inside it (never a client-side subsample test) and returns a delta.
  // Mirrors confirmSegmentSelection's pointset-creation + applyDelta flow, then
  // clears the box so the outline vanishes.
  const applyBox = useCallbackLabel(async (clsDef) => {
    const targetCls = clsDef || activeClassDef;
    if (!selBox || !targetCls) return;
    let r;
    try {
      r = await VoxaAPI.applyShape({
        shape: { type: 'obb', center: selBox.center, size: selBox.size, rotation: selBox.rotation },
        targetClass: targetCls.id,
      });
    } catch (err) {
      console.error('box apply failed:', err);
      return;
    }
    // Empty box: the backend returns no delta when the OBB encloses zero
    // full-res points (parity with the Draw tool's empty-tube guard). Keep the
    // box so the user can reposition it, and don't create an empty instance.
    if (!r.indices || r.nAffected === 0) {
      console.warn('box apply: no points inside the box');
      return;
    }
    const segId = Number.isFinite(r.instanceId) ? r.instanceId : -1;
    if (segId >= 0) {
      onChange([...instances, {
        id: newId(),
        segId,
        kind: 'pointset',
        cls: targetCls.id,
        label: `${targetCls.label} ${(counts[targetCls.id] || 0) + 1}`,
        color: targetCls.color,
        source: 'box',
        confirmed: !!autoConfirmFor('box'),
        // Persist the selection OBB (display frame) so a future export can
        // rasterize this box at any density. Stays kind:'pointset' -> no gizmo,
        // no cuboid edges (gated on kind !== 'pointset'). Spec section 1.
        center: [...selBox.center],
        size: [...selBox.size],
        rotation: [...selBox.rotation],
      }]);
    }
    // Refresh working arrays AND clear any stale preseg selection so it can't
    // resurface after a box apply (confirmSegmentSelection clears it likewise).
    setSegState((s) => (s ? { ...applyDelta(s, {
      indices: r.indices,
      after_class: r.afterClass,
      after_instance: r.afterInstance,
    }), selection: new Set() } : s));
    setSelBox(null);
  }, [selBox, activeClassDef, instances, counts, onChange, setSegState, setSelBox, autoConfirm, presegRapid]);

  // Surface each applied centerline instance in the right Instances panel as
  // a pointset row (parity with fast-label promotion). Re-applies refresh the
  // class; instances absorbed by a merge drop their row. Reads the latest
  // instances through a ref — one Enter can apply several groups back to
  // back, faster than the prop re-renders.
  const instancesRef = useRefLabel(instances);
  instancesRef.current = instances;
  const onDrawApplied = useCallbackLabel(({ instanceId, classId, mergedFrom }) => {
    const cls = classes.find((c) => c.class_id === classId);
    if (!cls) return;
    const absorbed = new Set(mergedFrom);
    const kept = instancesRef.current.filter(
      (i) => !(i.kind === 'pointset' && absorbed.has(i.segId)));
    const existing = kept.find((i) => i.kind === 'pointset' && i.segId === instanceId);
    const next = existing
      ? kept.map((i) => i === existing ? { ...i, cls: cls.id, color: cls.color } : i)
      : [...kept, {
        id: newId(),
        segId: instanceId,
        kind: 'pointset',
        cls: cls.id,
        label: `${cls.label} #${instanceId}`,
        color: cls.color,
        source: 'draw',
        confirmed: autoConfirmFor('draw'),
      }];
    instancesRef.current = next;
    onChange(next);
  }, [classes, onChange, autoConfirm, presegRapid]);

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

  // ── Fast labeling sub-mode ────────────────────────────────────────────────
  // Queue of unpromoted presegments, largest first. Recomputes after every
  // confirm (applyDelta refreshes summary, the new instance extends
  // promotedSegIds), so the confirmed segment drops out and the same index
  // lands on the next-largest one.
  const fastQueue = useMemoLabel(
    () => (fastMode ? deriveFastQueue(segState?.summary, promotedSegIds) : []),
    [fastMode, segState?.summary, promotedSegIds]);
  // Single clamped cursor — the queue shrinks on confirm, so fastPos may
  // point past the end until the next step.
  const fastIdx = Math.min(fastPos, Math.max(fastQueue.length - 1, 0));
  const fastSeg = fastMode && fastQueue.length > 0 ? fastQueue[fastIdx] : null;

  // Highlight the current queue segment (drives the orange overlay via the
  // selection effect above) and center the camera on it.
  useEffectLabel(() => {
    if (!fastSeg) return;
    setSegState((s) => {
      if (!s) return s;
      if (s.selection.size === 1 && s.selection.has(fastSeg.id)) return s;
      return { ...s, selection: new Set([fastSeg.id]) };
    });
    focusSegment(viewerRef, cloud, segState, fastSeg.id);
    // eslint-disable-next-line
  }, [fastSeg?.id]);

  // Leaving fast mode clears the queue highlight and any pending confirm.
  useEffectLabel(() => {
    if (fastMode) return;
    setFastPendingCls(null);
    setSegState((s) => (s && s.selection.size ? { ...s, selection: new Set() } : s));
  }, [fastMode, setSegState]);

  // WASD steps the fast queue — force orbit nav so the walk controller
  // can't swallow those keys. Draw binds no WASD keys, so walking while
  // drawing is allowed.
  useEffectLabel(() => {
    if (fastMode && navMode === 'walk') onNavModeChange?.('orbit');
  }, [fastMode, navMode, onNavModeChange]);

  const fastStep = useCallbackLabel((delta) => {
    setFastPos((p) => stepIndex(fastQueue.length, p, delta));
  }, [fastQueue.length]);

  const fastConfirm = useCallbackLabel(async () => {
    if (!fastSeg || !fastPendingCls) return;
    const cls = fastPendingCls;
    setFastPendingCls(null);
    await confirmSegmentSelection(cls, { confirmed: true });
  }, [fastSeg, fastPendingCls, confirmSegmentSelection]);

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

  // Snap the box-select OBB to the points inside its AABB (rotation ignored,
  // matching the old cuboid auto-fit). Drives the Box tool's Auto-fit button.
  const autoFitBox = useCallbackLabel(async () => {
    if (!selBox) return;
    const half = selBox.size.map((v) => v / 2);
    const cmin = [selBox.center[0] - half[0], selBox.center[1] - half[1], selBox.center[2] - half[2]];
    const cmax = [selBox.center[0] + half[0], selBox.center[1] + half[1], selBox.center[2] + half[2]];
    const fitted = await VoxaAPI.autoFit(cmin, cmax, activeClass,
      activeClassDef?.color, 'box-select');
    setSelBox((b) => (b ? { ...b, center: fitted.center, size: fitted.size } : b));
  }, [selBox, activeClass, activeClassDef]);

  // Hotkeys: class key applies/labels the active selection, ⌫ delete, F frame,
  // G/R/Y transform the box, ⌘S save. In walk mode the viewer owns WASD/QE.
  useEffectLabel(() => {
    const onKey = (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      // Fast labeling / Draw sub-modes own the keyboard while active.
      // DrawMode's DrawKeys runs in capture phase like FastLabelKeys.
      if (fastMode || drawMode) return;
      // Ctrl/Cmd+Enter is tool-agnostic: with a tool selection it opens the
      // class picker to apply; otherwise (Box tool) it toggles the confirmed
      // flag on the selected instance.
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if ((segState && segState.selection.size > 0) || (activeTool === 'box' && selBox)) {
          // Open the class picker so the user can quick-pick the class for the
          // new (unconfirmed) pointset. In Box mode this routes to applyBox;
          // otherwise to confirmSegmentSelection.
          setClassPickerOpen(true);
        } else if (activeTool === 'box') {
          toggleConfirmSelected();
        }
        return;
      }
      // In walk mode the viewer owns WASD/QE — bail on those keys so we don't
      // double-fire (several classes also bind w/e/r/q as hotkeys).
      if (navMode === 'walk' && /^[wasdqeWASDQE]$/.test(e.key)) return;
      // Class hotkey. Runs before the Box-only gate so it works for a preseg
      // selection in any tool. With an active tool selection it applies+labels
      // (honoring auto-confirm inside confirmSegmentSelection / applyBox);
      // with no selection it just sets the active class.
      const cls = classes.find((c) => c.hotkey === e.key);
      if (cls) {
        if (segState && segState.selection.size > 0) {
          e.preventDefault();
          confirmSegmentSelection(cls);
        } else if (activeTool === 'box' && selBox) {
          e.preventDefault();
          applyBox(cls);
        } else {
          setActiveClass(cls.id);
        }
        return;
      }
      // Frame + delete work on any selected instance, in any tool (the help
      // text advertises them as tool-agnostic).
      if (e.key === 'f' || e.key === 'F') {
        if (selected) focusInstance(selected);
        return;
      }
      if (e.key === 'Backspace' || e.key === 'Delete') {
        if (selected && !isLocked) { e.preventDefault(); deleteSelected(); }
        return;
      }
      // Below here: Box-tool gizmo interactions only (G/R/Y transform selBox).
      if (activeTool !== 'box') return;
      if ((!isLocked || !!selBox) && (e.key === 'g' || e.key === 'G')) {
        setTransformMode('translate');
      } else if ((!isLocked || !!selBox) && (e.key === 'r' || e.key === 'R')) {
        setTransformMode('rotate');
      } else if ((!isLocked || !!selBox) && (e.key === 'y' || e.key === 'Y')) {
        setTransformMode('scale');
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line
  }, [classes, selected, isLocked, instances, activeTool, navMode, segState, selBox, confirmSegmentSelection, applyBox, fastMode, drawMode]);

  return (
    <div className="mode-root label">
      <FastLabelKeys
        active={fastMode && !fastPendingCls}
        classes={classes}
        onStep={fastStep}
        onPickClass={setFastPendingCls}
        onExit={() => setPresegRapid(false)}
      />
      {fastMode && (
        <FastLabelHUD queue={fastQueue} pos={fastIdx} classes={classes} />
      )}
      {fastPendingCls && fastSeg && (
        <FastConfirmModal
          seg={fastSeg}
          cls={fastPendingCls}
          onConfirm={fastConfirm}
          onCancel={() => setFastPendingCls(null)}
        />
      )}
      {classPickerOpen && (
        <ClassPickerModal
          classes={classes}
          counts={counts}
          onPick={(cls) => {
            setClassPickerOpen(false);
            if (activeTool === 'box' && selBox) applyBox(cls);
            else confirmSegmentSelection(cls);
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

        <ToolOptions
          activeTool={activeTool}
          presegRapid={presegRapid} setPresegRapid={setPresegRapid} setFastPos={setFastPos}
          autoConfirm={autoConfirm} setAutoConfirm={setAutoConfirm}
          segState={segState} setSegState={setSegState} classes={classes}
          viewerRef={viewerRef} cloud={cloud} promotedSegIds={promotedSegIds}
          pointSize={pointSize} setPointSize={setPointSize}
          activeClass={activeClass} setActiveClass={setActiveClass}
          onExit={() => setActiveTool('presegment')}
          onDrawApplied={onDrawApplied}
          hasBox={!!selBox} onDrawBox={toggleBoxSelect}
          transformMode={transformMode} setTransformMode={setTransformMode}
          onAutoFit={autoFitBox}
          onApply={() => setClassPickerOpen(true)} />
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
          colorMode={isPreseg ? colorMode : 'rgb'}
          pointSize={pointSize}
          diffMask={diffMask}
          showDiff={showDiff}
          transformMode={selBox ? (transformMode || 'translate') : null}
          onCuboidTransform={onCuboidTransform}
          highlightCuboid={highlightCuboid}
          selectionMask={selectionMask}
          confirmedCuboids={confirmedCuboids}
          confirmedPointsetHideMask={confirmedPointsetHideMask}
          hideConfirmedPoints={hideConfirmed}
          onLabelStats={setLabelStats}
          onCameraChange={onCameraChange}
          segBoxes={isPreseg && segBoxesFiltered
            ? { ...segBoxesFiltered, selection: segState.selection }
            : null}
          segHulls={isPreseg && segHullsFiltered
            ? { ...segHullsFiltered, selection: segState.selection }
            : null}
          showSegHulls={showSegHulls}
        />

        <div className="vp-hud-top">
          <div className="hud-group">
            <ToolRail activeTool={activeTool} onSelect={setActiveTool}
              ctx={{ segState, isAnnotated }} />
          </div>
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
          {segState?.isFromPrelabel && (
            <ToolButton mini
              icon="Δ"
              label={showDiff ? 'Hide diff' : 'Diff vs prelabel'}
              onClick={() => setShowDiff((v) => !v)}
              active={showDiff}
            />
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
              {(instFilter || instStatus !== 'all')
                ? `${filteredInstances.length} / ${instances.length}`
                : instances.length}
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
        <div className="tool-opt-toggle inst-status-toggle">
          <button
            className={instStatus === 'all' ? 'active' : ''}
            onClick={() => setInstStatus('all')}
          >all</button>
          <button
            className={instStatus === 'unconfirmed' ? 'active' : ''}
            onClick={() => setInstStatus('unconfirmed')}
          >unconfirmed</button>
          <button
            className={instStatus === 'confirmed' ? 'active' : ''}
            onClick={() => setInstStatus('confirmed')}
          >confirmed</button>
        </div>
        <div className="inst-list">
          {instances.length === 0 && (
            <div className="sugg-empty">No instances yet. Pick a tool, select points, and apply.</div>
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

