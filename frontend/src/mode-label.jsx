// mode-label.jsx — class palette (left), instance list (right), cuboid editing.

import { useState as useStateLabel, useMemo as useMemoLabel,
         useEffect as useEffectLabel, useCallback as useCallbackLabel,
         useRef as useRefLabel } from 'react';
import * as THREE from 'three';
import { Viewer } from './viewer.jsx';
import { CollapsiblePanel } from './mode-inspect.jsx';
import { ViewportToolbar, ToolButton, HUDChip, CameraPresets, NavModeToggle, HelpButton } from './viewport-atoms.jsx';
import { VoxaAPI, newId } from './api.js';
import { focusSegment } from './segment-tools.jsx';
import { deriveFastQueue, stepIndex, FastLabelKeys, FastLabelHUD,
         FastConfirmModal, FAST_HIGHLIGHT_COLOR } from './fast-label.jsx';
import SessionPicker from './session-picker.jsx';
import ExportWizard from './export-wizard.jsx';
import { applyDelta, applySamDelta, computeDiffMask, reconcileSamAfterApply, filterSamSelectionOnToolSwitch, retireSamIdsForIndices } from './segment-state.js';
import { toolAvailable, defaultTool } from './label-tools.js';
import { maskColorRGB } from './sam-util.js';
import ToolRail from './tool-rail.jsx';
import ToolOptions from './tool-options.jsx';
import { ContextMenu } from './context-menu.jsx';
import { ClassPickerModal } from './class-picker.jsx';
import { chordStep, CLASS_GROUPS } from './class-chords.js';
import { ChordOverlay } from './chord-overlay.jsx';
import { cutEligibility } from './cut-eligibility.js';
import { removeOutliersEligibility } from './outlier-eligibility.js';
import { fitEligibility } from './fit-eligibility.js';
import CutModal from './cut-mode.jsx';

// "30k", "1.2M", "523" — keeps the HUD chip narrow regardless of scene size.
function formatPointCount(n) {
  if (n < 1000) return String(n);
  if (n < 1e6) return `${(n / 1e3).toFixed(n < 10000 ? 1 : 0)}k`;
  return `${(n / 1e6).toFixed(n < 1e7 ? 2 : 1)}M`;
}

const LABEL_SEL_BOX_ID = '__label_sel_box__';
const LABEL_SEL_BOX_COLOR = '#ffd24a';
// Selected SAM candidates switch to this flat color — the same yellow
// setSelectedSegmentMask defaults to for every other tool's selection
// overlay (0xfacc15) — rather than a lightened version of their own hue,
// so "selected" reads identically across presegment/box/draw/beam/SAM.
const SAM_SELECTED_COLOR = new THREE.Color(0xfacc15).toArray();

export function LabelMode({ cloud, theme, viewerRef, classes, instances, onChange, cloudBBox, navMode, onNavModeChange, segState, setSegState, prelabelRef, onCameraChange, hasMesh, isAnnotated, sessions, activeSessionId, presegs, onSelectSession, onCreateSession, onRenameSession, onDeleteSession, sessionLoading }) {
  const meshPopupRef = useRefLabel(null);
  // Default to the first ASSIGNABLE class — 'unknown' (the old fallback) is
  // frozen now, and a frozen default would 422 on the first apply.
  const [activeClass, setActiveClass] = useStateLabel(
    classes.find((c) => !c.frozen)?.id || classes[0]?.id || 'unknown');
  const [selectedId, setSelectedId] = useStateLabel(null);
  const [instCutMenu, setInstCutMenu] = useStateLabel(null); // {x, y, instId} | null
  const [hiddenClasses, setHiddenClasses] = useStateLabel(new Set());
  const [activeTool, setActiveTool] = useStateLabel(() =>
    defaultTool({ segState, isAnnotated }));
  // Presegment "rapid" = the old fast-labeling queue.
  const [presegRapid, setPresegRapid] = useStateLabel(false);

  // Derived legacy flags — keep the existing body working during the refactor.
  const fastMode = activeTool === 'presegment' && presegRapid;
  const drawMode = activeTool === 'draw';
  const beamMode = activeTool === 'beam';
  const prismMode = activeTool === 'prism';
  // Sub-modes whose overlay owns viewport input (capture-phase keys +
  // pointer): global pick/hotkey handlers stand down. A future 5th tool
  // adds one term here instead of touching every gate.
  const subModeOwnsInput = drawMode || beamMode || prismMode;
  // Presegmentation is a way to *select* points; its segments (hulls, boxes,
  // per-segment hue coloring) only show while the Presegment tool is active.
  // Every other tool works on the raw RGB cloud.
  const isPreseg = activeTool === 'presegment';

  // Per-tool auto-confirm (added here to avoid a forward reference in Tasks 8/9;
  // threaded into apply paths in Task 10).
  const [autoConfirm, setAutoConfirm] = useStateLabel({ box: false, draw: false, presegment: false, beam: false, sam: false, prism: false });
  const autoConfirmFor = (tool) =>
    tool === 'presegment' ? (presegRapid || autoConfirm.presegment) : !!autoConfirm[tool];
  // Color mode picked in the Display panel; null = automatic — class colors
  // while Presegment is active (so existing labels stay legible during
  // selection; per-segment hues override per-point colors once hulls exist
  // anyway), raw RGB for every other tool. Picking a pill pins the mode.
  const [colorModeChoice, setColorModeChoice] = useStateLabel(null);
  const colorMode = colorModeChoice ?? (isPreseg ? 'class' : 'rgb');
  // Which color channels this scene offers (mirrors Inspect's Display panel).
  const colorChannels = useMemoLabel(() => ({
    rgb: !!cloud,
    height: !!cloud,
    class: !!cloud && !!cloud.classIds && !!cloud.classPalette,
    instance: !!cloud && !!cloud.instanceIds,
  }), [cloud]);
  // Drop a pinned mode the scene can't honor — back to automatic.
  useEffectLabel(() => {
    if (cloud && colorModeChoice && !colorChannels[colorModeChoice]) setColorModeChoice(null);
  }, [cloud, colorChannels, colorModeChoice]);
  // Point size lives in the Display panel (same slider as Inspect); bumping
  // it makes the sparse subsample read denser for Draw/Beam work.
  const [pointSize, setPointSize] = useStateLabel(0.012);
  const [showFloor, setShowFloor] = useStateLabel(true);
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

  // Keep activeClass valid as the class list streams in — and never let it
  // rest on a frozen legacy class (the pre-config fallback 'unknown' is
  // frozen now; a frozen active class would 422 on the first apply).
  useEffectLabel(() => {
    if (!classes.length) return;
    const cur = classes.find((c) => c.id === activeClass);
    if (!cur || cur.frozen) {
      setActiveClass((classes.find((c) => !c.frozen) || classes[0]).id);
    }
  }, [classes, activeClass]);

  // Never leave an unavailable tool active (e.g. after switching to a scene
  // with no segState or a non-annotated scan).
  useEffectLabel(() => {
    const ctx = { segState, isAnnotated, rawSourceAvailable: cloud?.rawSourceAvailable };
    if (!toolAvailable(activeTool, ctx)) {
      setActiveTool(defaultTool(ctx));
    }
  }, [segState, isAnnotated, activeTool, cloud?.rawSourceAvailable]);

  // The box-select gizmo belongs to the Box tool only; leaving Box must clear it
  // so a stale box + gizmo can't render over / hijack Presegment or Draw.
  useEffectLabel(() => {
    if (activeTool !== 'box') setSelBox(null);
  }, [activeTool]);

  // SAM candidate selection belongs to the SAM tool AND the Presegment tool
  // (source:'preseg'-tagged cut candidates render/select there too — see
  // segment-tools.jsx::PresegmentList). Leaving both must clear it so a stale
  // selection can't silently get confirmed if the user returns to either tool
  // and hits Ctrl+Enter / a class hotkey without re-selecting. Switching
  // directly between SAM and Presegment is narrower and symmetric: a real
  // SAM candidate (source:'sam') must be dropped on entering Presegment, and
  // a source:'preseg' cut candidate must be dropped on entering SAM — either
  // one surviving into the wrong tool would otherwise get silently
  // classified ahead of whatever the user actually selects there (see
  // filterSamSelectionOnToolSwitch).
  const prevToolRef = useRefLabel(activeTool);
  useEffectLabel(() => {
    const prevTool = prevToolRef.current;
    prevToolRef.current = activeTool;
    setSegState((s) => {
      if (!s || s.samSelection.size === 0) return s;
      if (activeTool !== 'sam' && activeTool !== 'presegment') {
        return { ...s, samSelection: new Set() };
      }
      const filtered = filterSamSelectionOnToolSwitch(s.samSelection, s.samSegments, prevTool, activeTool);
      return filtered === s.samSelection ? s : { ...s, samSelection: filtered };
    });
  }, [activeTool, setSegState]);

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
    const subIdx = cloud.subsampleIdx;
    const subN = cloud.positions.length / 3;
    if (activeTool === 'sam') {
      const samIds = segState.samIds;
      const samSelection = segState.samSelection;
      const mask = new Uint8Array(subN);
      // One RGB triplet per rendered point: an unselected candidate gets
      // its own hue (same palette as the SAM segment list's row dot and
      // the review modal's mask swatches, sam-util.js::maskColorRGB, keyed
      // by sam_seg_id) so segments are distinguishable; a SELECTED one
      // switches to the same flat yellow/orange every other tool's
      // selection overlay uses (setSelectedSegmentMask's default / fast
      // mode's FAST_HIGHLIGHT_COLOR) — the actual "selected" cue elsewhere
      // in this app is that flat highlight color, not a lightened hue.
      const selColor = SAM_SELECTED_COLOR;
      const colors = new Float32Array(subN * 3);
      const colorCache = new Map();
      let any = false;
      for (let p = 0; p < subN; p++) {
        const f = subIdx ? subIdx[p] : p;
        const samId = samIds[f];
        if (samId < 0) continue;
        mask[p] = 1;
        any = true;
        const isSel = samSelection.has(samId);
        const cacheKey = isSel ? -1 : samId; // every selected point shares one color
        let rgb = colorCache.get(cacheKey);
        if (!rgb) {
          rgb = isSel ? selColor : maskColorRGB(samId);
          colorCache.set(cacheKey, rgb);
        }
        const o = p * 3;
        colors[o] = rgb[0]; colors[o + 1] = rgb[1]; colors[o + 2] = rgb[2];
      }
      v.setSelectedSegmentMask(any ? mask : null, undefined, any ? colors : null);
      return;
    }
    const sel = segState.selection;
    if (sel.size === 0) {
      v.setSelectedSegmentMask(null);
      return;
    }
    const inst = segState.instanceFull;
    const mask = new Uint8Array(subN);
    for (let p = 0; p < subN; p++) {
      const f = subIdx ? subIdx[p] : p;
      if (sel.has(inst[f])) mask[p] = 1;
    }
    v.setSelectedSegmentMask(mask, fastMode ? FAST_HIGHLIGHT_COLOR : undefined);
  }, [segState?.selection, segState?.samIds, segState?.samSelection, segState?.instanceFull, cloud, viewerRef, fastMode, activeTool]);

  // Ctrl/Cmd-click in the 3D viewport toggles selection of the presegment
  // under the cursor. Active in any tool mode whenever segment data exists.
  // Draw mode owns all pointer events; skip registration while it's on.
  useEffectLabel(() => {
    if (!segState) return;
    if (subModeOwnsInput) return;
    const viewer = viewerRef?.current;
    if (!viewer?.onPointerPick) return;
    return viewer.onPointerPick((fullIndex, evt) => {
      if (activeTool === 'sam') {
        if (!evt.ctrlKey && !evt.metaKey && !evt.shiftKey) return;
        const samId = segState.samIds[fullIndex];
        if (samId < 0) return;
        setSegState((s) => {
          if (!s) return s;
          const next = new Set(s.samSelection);
          next.has(samId) ? next.delete(samId) : next.add(samId);
          return { ...s, samSelection: next };
        });
        return;
      }
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
  }, [segState, viewerRef, setSegState, subModeOwnsInput, activeTool]);

  // Hull-click selection: clicking directly on a hull face (Ctrl or plain click)
  // selects the segment. Works in all tool modes.
  useEffectLabel(() => {
    if (!segState) return;
    if (subModeOwnsInput) return;
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
  }, [segState, viewerRef, setSegState, subModeOwnsInput]);

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

  const [exportOpen, setExportOpen] = useStateLabel(false);

  const [denoiseRatio, setDenoiseRatio] = useStateLabel(2.0);
  const [denoiseInstId, setDenoiseInstId] = useStateLabel(null);  // backend instance id of the live denoise result
  const [denoiseBusy, setDenoiseBusy] = useStateLabel(false);

  // Per-class point counts (class_id → n_points) for the export wizard's
  // "~0 after filters" guard. Derived from the load-time segment_summary;
  // null when the session wasn't loaded from a prelabel (then the wizard
  // falls back to a nLabeledPoints check).
  const perClassPointCounts = useMemoLabel(() => {
    const ss = cloud?.segmentSummary;
    if (!ss) return null;
    const c = {};
    for (const k in ss) {
      const cid = ss[k]?.class_id;
      if (cid == null || cid < 0) continue;
      c[cid] = (c[cid] || 0) + (ss[k].n_points || 0);
    }
    return Object.keys(c).length ? c : null;
  }, [cloud]);

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

  // "Confirmed = locked": instance ids a volume apply (Box/Draw/Beam) must not
  // overwrite. Sent to apply-shape so an overextended box can't steal points
  // that already belong to a confirmed instance. Un-confirm to re-label them.
  const protectedSegIds = useMemoLabel(
    () => instances.filter((i) => i.confirmed && Number.isFinite(i.segId)).map((i) => i.segId),
    [instances],
  );

  // Cut-selection tool (Task 11): {sources, instanceClassId} | null.
  // instanceClassId is only set for a single-instance cut — the source
  // instance's class (a string id matching classes[].id, same convention as
  // inst.cls elsewhere in this file) is already known here from the
  // Instances-panel row that was right-clicked, so the modal/backend never
  // need to round-trip it; the cut-shape response has no class field for the
  // instance case (it inherits the source's class server-side, but doesn't
  // echo it back) and this is how that gap is closed without touching the
  // backend response shape.
  const [cutModal, setCutModal] = useStateLabel(null);

  const openCutModal = useCallbackLabel((sources, instanceClassId = null) => {
    setCutModal({ sources, instanceClassId });
  }, []);

  // Fit a gravity-aligned Box selection volume around the current selection's
  // points, then hand it to the Box tool (staged, unconfirmed). Capture happens
  // here — BEFORE setActiveTool('box'), which clears the preseg/SAM selection.
  const fitBoxToSelection = useCallbackLabel(async (sources) => {
    try {
      const obb = await VoxaAPI.fitBox(sources);
      // FULL selBox shape (mirror toggleBoxSelect) — id is mandatory or the box
      // is invisible and non-transformable (Viewer gates on LABEL_SEL_BOX_ID).
      setSelBox({
        id: LABEL_SEL_BOX_ID,
        label: 'box-select',
        cls: 0,
        color: LABEL_SEL_BOX_COLOR,
        center: obb.center, size: obb.size, rotation: obb.rotation,
      });
      setActiveTool('box');
    } catch (e) {
      console.error('fit-box failed', e);
    }
  }, [setSelBox, setActiveTool]);

  // Reports the raw VoxaAPI.cutShape(...) response upward from CutModal.
  // Mirrors confirmSegmentSelection/confirmSamSelection/applyBox: this is the
  // one place that patches segState + the Instances panel's `instances` rows,
  // CutModal itself never touches segState directly.
  const onCutConfirmedHandler = useCallbackLabel((resp) => {
    if (resp.materialized.length > 0) {
      setSegState((s) => {
        if (!s) return s;
        let next = s;
        for (const m of resp.materialized) {
          if (!m.indices || m.indices.length === 0) continue;
          next = applySamDelta(next, { indices: m.indices, samSegId: m.samSegId, source: m.source });
        }
        return next;
      });
    }
    if (resp.instance && resp.instance.indices && resp.instance.indices.length > 0) {
      const clsId = cutModal?.instanceClassId;
      const cls = classes.find((c) => c.id === clsId);
      if (!cls) {
        console.error('cut-shape: instance entry returned but no source class id known', resp.instance);
        return;
      }
      const n = resp.instance.indices.length;
      const afterClass = new Int8Array(n).fill(cls.class_id);
      const afterInstance = new Int32Array(n).fill(resp.instance.instId);
      setSegState((s) => (s ? applyDelta(s, {
        indices: resp.instance.indices, after_class: afterClass, after_instance: afterInstance,
      }) : s));
      onChange([...instances, {
        id: newId(),
        segId: resp.instance.instId,
        kind: 'pointset',
        cls: cls.id,
        label: `${cls.label} ${(counts[cls.id] || 0) + 1}`,
        color: cls.color,
        source: 'cut',
        confirmed: false,
      }]);
    }
  }, [cutModal, classes, instances, counts, onChange, setSegState]);

  // Feature C: global "Detect outliers". Runs cloud-wide statistical outlier
  // removal on the backend and stages the outliers as one unconfirmed Exclude
  // pointset. Re-runs pass the tracked denoiseInstId as replaceInst so the
  // backend erases the prior result first; we then drop the old row and add the
  // new one (re-running never stacks Exclude instances). Mirrors
  // onCutConfirmedHandler's segState patch + Instances-panel row append.
  const runDenoise = useCallbackLabel(async () => {
    if (!segState || denoiseBusy) return;
    setDenoiseBusy(true);
    try {
      const resp = await VoxaAPI.denoise({
        stdRatio: denoiseRatio,
        replaceInst: denoiseInstId,           // erase the prior result first
        protectInstances: protectedSegIds,
      });
      // Drop the previous denoise row (its points were just erased server-side).
      // Look up the Exclude class by its stable string key (like everywhere
      // else in this file), not a hardcoded numeric id.
      const cls = classes.find((c) => c.id === 'unknown');   // Exclude / Review
      if (!cls) { console.error('denoise: no "unknown" (Exclude) class in config'); return; }
      const kept = instances.filter((i) => i.segId !== denoiseInstId);
      if (resp.instance_id == null) {
        onChange(kept);
        setDenoiseInstId(null);
      } else {
        const idx = resp.indices;                            // Int32Array (decoded in api.js)
        const afterClass = new Int8Array(idx.length).fill(cls.class_id);
        const afterInstance = new Int32Array(idx.length).fill(resp.instance_id);
        setSegState((s) => (s ? applyDelta(s, {
          indices: idx, after_class: afterClass, after_instance: afterInstance,
        }) : s));
        onChange([...kept, {
          id: newId(),
          segId: resp.instance_id,
          kind: 'pointset',
          cls: cls.id,
          label: `${cls.label} ${(counts[cls.id] || 0) + 1}`,
          color: cls.color,
          source: 'denoise',
          confirmed: false,
        }]);
        setDenoiseInstId(resp.instance_id);
      }
    } catch (e) {
      console.error('denoise failed', e);
    } finally {
      setDenoiseBusy(false);
    }
  }, [segState, denoiseBusy, denoiseRatio, denoiseInstId, protectedSegIds,
      classes, instances, counts, onChange, setSegState]);

  // Feature B: right-click "Remove outliers" strips a selection's spatial
  // strays back to unlabeled (instance) or drops their SAM candidacy (sam).
  const removeOutliers = useCallbackLabel(async ({ source, id }) => {
    if (!segState) return;
    try {
      const resp = await VoxaAPI.denoiseSelection({ source, id, stdRatio: denoiseRatio });
      if (!resp.n_removed || !resp.indices) return;
      const idx = resp.indices;                       // Int32Array (decoded in api.js)
      setSegState((s) => {
        if (!s) return s;
        if (source === 'sam') {
          // Strays lose SAM candidacy; candidate shrinks (mirrors backend
          // remove_sam_points -> _retire_sam_ids).
          return retireSamIdsForIndices(s, idx);
        }
        // instance: strays back to unlabeled
        const afterClass = new Int8Array(idx.length).fill(-1);
        const afterInstance = new Int32Array(idx.length).fill(-1);
        return applyDelta(s, { indices: idx, after_class: afterClass, after_instance: afterInstance });
      });
    } catch (e) {
      console.error('removeOutliers failed', e);
    }
  }, [segState, denoiseRatio, setSegState]);

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
        { keys: ['Rail'], desc: 'Switch Presegment / Box / Draw / Beam' },
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
      title: 'Class assignment (two-stroke chords)',
      items: classes.length
        ? CLASS_GROUPS.filter((g) => g.key).map((g) => (
            { keys: [g.key, '…'], desc: `${g.label} (then member key)` }))
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

  // Ctrl+Enter (or a class hotkey) on a SAM candidate selection: same
  // reassign→pointset→applyDelta pipeline as confirmSegmentSelection, plus
  // reconcileSamAfterApply to retire the absorbed SAM candidates (their
  // samIds entries go back to -1 so the cyan overlay/candidate list drop them).
  const confirmSamSelection = useCallbackLabel(async (clsDef) => {
    const targetCls = clsDef || activeClassDef;
    if (!segState || segState.samSelection.size === 0) return;
    if (!targetCls) return;
    const samIds = segState.samIds;
    const sel = segState.samSelection;
    const idx = [];
    for (let p = 0; p < samIds.length; p++) {
      if (sel.has(samIds[p])) idx.push(p);
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
      console.error('confirm sam reassign failed:', err);
      return;
    }

    const newSegId = r.afterInstance && r.afterInstance.length > 0
      ? r.afterInstance[0] : -1;
    const appliedSamSegIds = new Set(sel);
    if (newSegId >= 0) {
      const newInst = {
        id: newId(),
        segId: newSegId,
        kind: 'pointset',
        cls: targetCls.id,
        label: `${targetCls.label} ${(counts[targetCls.id] || 0) + 1}`,
        color: targetCls.color,
        source: 'sam',
        confirmed: !!autoConfirmFor('sam'),
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
      return reconcileSamAfterApply(next, appliedSamSegIds);
    });
  }, [segState, activeClassDef, instances, counts, onChange, setSegState, autoConfirm]);

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
        protectInstances: protectedSegIds,
      });
    } catch (err) {
      console.error('box apply failed:', err);
      return;
    }
    // Empty box: the backend returns no delta when the OBB encloses zero
    // full-res points (parity with the Draw tool's empty-tube guard). Keep the
    // box so the user can reposition it, and don't create an empty instance.
    if (!r.indices || r.nAffected === 0) {
      // Distinguish a geometrically-empty box from one that only covered
      // confirmed (locked) points, so the user knows to un-confirm to re-label.
      console.warn(r.nProtected > 0
        ? `box apply: ${r.nProtected} point(s) skipped — inside a confirmed instance (un-confirm to re-label)`
        : 'box apply: no points inside the box');
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
  }, [selBox, activeClassDef, instances, counts, onChange, setSegState, setSelBox, autoConfirm, presegRapid, protectedSegIds]);

  // Surface each applied Draw/Beam instance in the right Instances panel as a
  // pointset row. Re-applies refresh the class AND (for beams) the persisted
  // OBB selection volume — a stale box would replay into raw exports;
  // instances absorbed by a Draw merge drop their row. Reads the latest
  // instances through a ref — one Enter can apply several groups back to
  // back, faster than the prop re-renders.
  const instancesRef = useRefLabel(instances);
  instancesRef.current = instances;
  const onToolApplied = useCallbackLabel(({
    instanceId, classId, mergedFrom = [], source = 'draw', obb = null, prism = null,
  }) => {
    const cls = classes.find((c) => c.class_id === classId);
    if (!cls) return;
    const absorbed = new Set(mergedFrom);
    const kept = instancesRef.current.filter(
      (i) => !(i.kind === 'pointset' && absorbed.has(i.segId)));
    const existing = kept.find((i) => i.kind === 'pointset' && i.segId === instanceId);
    const volume = obb
      ? { center: [...obb.center], size: [...obb.size], rotation: [...obb.rotation] }
      : prism
        ? { prism: { polygon: prism.polygon.map((v) => [...v]), y0: prism.y0, height: prism.height } }
        : {};
    const next = existing
      ? kept.map((i) => i === existing ? { ...i, cls: cls.id, color: cls.color, ...volume } : i)
      : [...kept, {
        id: newId(),
        segId: instanceId,
        kind: 'pointset',
        cls: cls.id,
        label: `${cls.label} #${instanceId}`,
        color: cls.color,
        source,
        confirmed: autoConfirmFor(source),
        ...volume,
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

  // Two-stroke class chord state: null | a CLASS_GROUPS entry picked by the
  // first stroke (class-chords.js). Reset on tool switch so a stale pending
  // group can't linger.
  const [pendingGroup, setPendingGroup] = useStateLabel(null);
  useEffectLabel(() => { setPendingGroup(null); }, [activeTool]);
  // Class-rail groups are collapsed by default; clicking a header toggles
  // its member list open. (Set of open group ids; legacy included.)
  const [openGroups, setOpenGroups] = useStateLabel(() => new Set());
  const toggleGroup = (gid) => setOpenGroups((prev) => {
    const next = new Set(prev);
    if (next.has(gid)) next.delete(gid); else next.add(gid);
    return next;
  });

  // Hotkeys: class chords apply/label the active selection, ⌫ delete, F frame,
  // G/R/Y transform the box, ⌘S save. In walk mode the viewer owns WASD/QE.
  useEffectLabel(() => {
    const onKey = (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      // Fast labeling / Draw / Beam sub-modes own the keyboard while active.
      // DrawMode's DrawKeys / BeamMode's BeamKeys run in capture phase like
      // FastLabelKeys.
      if (fastMode || subModeOwnsInput) return;
      // Ctrl/Cmd+Enter is tool-agnostic: with a tool selection it opens the
      // class picker to apply; otherwise (Box tool) it toggles the confirmed
      // flag on the selected instance.
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if ((segState && segState.selection.size > 0)
          || (activeTool === 'box' && selBox)
          || ((activeTool === 'sam' || activeTool === 'presegment') && segState && segState.samSelection.size > 0)) {
          // Open the class picker so the user can quick-pick the class for the
          // new (unconfirmed) pointset. In Box mode this routes to applyBox;
          // in SAM mode (or a Presegment-tool cut-candidate selection) to
          // confirmSamSelection; otherwise to confirmSegmentSelection.
          setClassPickerOpen(true);
        } else if (activeTool === 'box') {
          toggleConfirmSelected();
        }
        return;
      }
      // In walk mode the viewer owns WASD/QE — bail on those keys so we don't
      // double-fire (several classes also bind w/e/r/q as hotkeys).
      if (navMode === 'walk' && /^[wasdqeWASDQE]$/.test(e.key)) return;
      // Two-stroke class chord (class-chords.js). Runs before the Box-only
      // gate so it works for a preseg selection in any tool. The second
      // stroke applies+labels through the same dispatch the old single-key
      // hotkeys used (honoring auto-confirm inside confirmSegmentSelection /
      // applyBox); with no selection it just sets the active class.
      const step = chordStep(pendingGroup, e.key, classes);
      if (step.type === 'group') { e.preventDefault(); setPendingGroup(step.group); return; }
      if (step.type === 'cancel') { e.preventDefault(); setPendingGroup(null); return; }
      if (step.type === 'class') {
        e.preventDefault();
        setPendingGroup(null);
        const cls = step.cls;
        if ((activeTool === 'sam' || activeTool === 'presegment')
          && segState && segState.samSelection.size > 0) {
          confirmSamSelection(cls);
        } else if (segState && segState.selection.size > 0) {
          confirmSegmentSelection(cls);
        } else if (activeTool === 'box' && selBox) {
          applyBox(cls);
        } else {
          setActiveClass(cls.id);
        }
        return;
      }
      // step.type === 'pass' → fall through to the non-class hotkeys below.
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
  }, [classes, selected, isLocked, instances, activeTool, navMode, segState, selBox, confirmSegmentSelection, confirmSamSelection, applyBox, fastMode, subModeOwnsInput, pendingGroup]);

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
      <ChordOverlay group={pendingGroup} classes={classes} />
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
            if ((activeTool === 'sam' || activeTool === 'presegment')
              && segState && segState.samSelection.size > 0) confirmSamSelection(cls);
            else if (activeTool === 'box' && selBox) applyBox(cls);
            else confirmSegmentSelection(cls);
          }}
          onClose={() => setClassPickerOpen(false)}
        />
      )}
      {exportOpen && cloud && (
        <ExportWizard
          scene={cloud.scene}
          sessionId={activeSessionId}
          classes={classes}
          scanCount={cloud.numPointsTotal ?? cloud.numPoints}
          rawSourceAvailable={cloud.rawSourceAvailable}
          perClassPointCounts={perClassPointCounts}
          nLabeledPoints={cloud.nLabeledPoints}
          onClose={() => setExportOpen(false)}
        />
      )}
      {cutModal && segState && cloud && (
        <CutModal
          segState={segState}
          cloud={cloud}
          sources={cutModal.sources}
          protectInstances={protectedSegIds}
          theme={theme}
          onClose={() => setCutModal(null)}
          onCutConfirmed={onCutConfirmedHandler}
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
        {isAnnotated && (
          <button className="ghost-btn ew-open" disabled={!activeSessionId || !cloud}
            title={activeSessionId ? 'Export labeled cloud from this session' : 'Open a session first'}
            onClick={() => setExportOpen(true)}>
            ⬇ Export…
          </button>
        )}
        {isAnnotated && (
          <div className="denoise-row" style={{ display: 'flex', gap: 6, alignItems: 'center' }}>
            <button
              className="ghost-btn"
              disabled={!activeSessionId || denoiseBusy}
              title={activeSessionId ? 'Detect stray outlier points and stage them as Exclude'
                                     : 'Open a session first'}
              onClick={runDenoise}>
              {denoiseBusy ? '… Detecting' : '✧ Detect outliers'}
            </button>
            <input type="range" min="1" max="3" step="0.1"
              value={denoiseRatio}
              onChange={(e) => setDenoiseRatio(parseFloat(e.target.value))}
              title={`Aggressiveness (σ=${denoiseRatio.toFixed(1)}; lower = greedier)`} />
          </div>
        )}
        <div className="side-hd">
          <span>Classes</span>
          <span className="badge-soft">{instances.length}</span>
        </div>
        <div className="class-list">
          {(() => {
            // Grouped render: chord groups in order, then a trailing section
            // for any class with an unknown/absent group (defaults path).
            // Legacy is collapsed by default; frozen rows are display-only
            // (visibility + counts work, activation doesn't).
            const known = new Set(CLASS_GROUPS.map((g) => g.id));
            const ungrouped = classes.filter((c) => !known.has(c.group));
            const sections = CLASS_GROUPS
              .map((g) => ({ g, members: classes.filter((c) => c.group === g.id) }))
              .filter(({ members }) => members.length > 0);
            if (ungrouped.length) sections.push({ g: { id: '_other', key: null, label: 'Ungrouped' }, members: ungrouped });
            const row = (c) => {
              const hidden = hiddenClasses.has(c.id);
              return (
                <div key={c.id}
                  className={'class-row' + (activeClass === c.id ? ' active' : '') + (hidden ? ' hidden' : '') + (c.frozen ? ' frozen' : '')}
                  onClick={() => { if (!c.frozen) setActiveClass(c.id); }}
                  title={c.frozen ? 'Legacy class — display-only, no new labels' : undefined}>
                  <span className="class-swatch" style={{ background: c.color }} />
                  <span className="class-name">{c.label}</span>
                  <span className="class-count">{counts[c.id] || 0}</span>
                  <button className="class-eye" onClick={(e) => { e.stopPropagation(); toggleClass(c.id); }}
                    title={hidden ? 'Show' : 'Hide'}>{hidden ? '◌' : '●'}</button>
                  <span className="class-hk">{c.frozen ? '—' : c.hotkey}</span>
                </div>
              );
            };
            return sections.map(({ g, members }) => {
              const open = openGroups.has(g.id);
              const hasActive = members.some((c) => c.id === activeClass);
              return (
                <div key={g.id}>
                  <div className={'class-group-hd toggle' + (g.id === 'legacy' ? ' legacy' : '')}
                    onClick={() => toggleGroup(g.id)}>
                    {open ? '▾' : '▸'} {g.key ? `${g.key} · ` : ''}{g.label}
                    {/* Active-class swatch stays visible while the group is
                        collapsed, so the current class is never invisible. */}
                    {!open && hasActive && (
                      <span className="class-swatch hd-active"
                        style={{ background: members.find((c) => c.id === activeClass)?.color }} />
                    )}
                    <span className="class-group-n">{members.length}</span>
                  </div>
                  {open && members.map(row)}
                </div>
              );
            });
          })()}
        </div>

        <ToolOptions
          activeTool={activeTool}
          presegRapid={presegRapid} setPresegRapid={setPresegRapid} setFastPos={setFastPos}
          autoConfirm={autoConfirm} setAutoConfirm={setAutoConfirm}
          segState={segState} setSegState={setSegState} classes={classes} counts={counts}
          viewerRef={viewerRef} cloud={cloud} promotedSegIds={promotedSegIds}
          activeClass={activeClass} setActiveClass={setActiveClass}
          onExit={() => setActiveTool('presegment')}
          onToolApplied={onToolApplied}
          protectInstances={protectedSegIds}
          activeSessionId={activeSessionId}
          hasBox={!!selBox} onDrawBox={toggleBoxSelect}
          transformMode={transformMode} setTransformMode={setTransformMode}
          onAutoFit={autoFitBox}
          onApply={() => setClassPickerOpen(true)}
          onEditSelection={openCutModal}
          onRemoveOutliers={removeOutliers}
          onFitBox={fitBoxToSelection} />
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
          pointSize={pointSize}
          showFloor={showFloor}
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
              ctx={{ segState, isAnnotated, rawSourceAvailable: cloud?.rawSourceAvailable }} />
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

        {/* Same Display controls as Inspect (minus mesh — that's the
            companion window above), collapsed by default so the labeling
            viewport stays clear. */}
        <div className="inspect-right">
          <CollapsiblePanel title="Display">
            <div className="ctrl">
              <label>Color by</label>
              <div className="pill-group">
                {[
                  ['rgb', 'RGB'],
                  ['height', 'Height'],
                  ['class', 'Class'],
                  ['instance', 'Instance'],
                ].map(([k, l]) => {
                  const enabled = !!colorChannels[k];
                  return (
                    <button key={k}
                      className={'pill' + (colorMode === k ? ' active' : '') + (enabled ? '' : ' disabled')}
                      disabled={!enabled}
                      title={enabled ? '' : `not available — scene has no ${k} channel`}
                      onClick={() => enabled && setColorModeChoice(k)}>{l}</button>
                  );
                })}
              </div>
            </div>
            <div className="ctrl">
              <label>Point size <span className="mono">{pointSize.toFixed(3)}</span></label>
              <input type="range" min={0.002} max={1.5} step={0.005}
                value={pointSize} className="slider"
                onChange={(e) => setPointSize(Number(e.target.value))} />
            </div>
            <div className="ctrl row">
              <label>Floor & grid</label>
              <button className={'sw' + (showFloor ? ' on' : '')}
                onClick={() => setShowFloor(!showFloor)}><i /></button>
            </div>
          </CollapsiblePanel>
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
                  onContextMenu={(e) => { e.preventDefault(); setInstCutMenu({ x: e.clientX, y: e.clientY, instId: inst.id }); }}
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
                        {/* Assignable only — re-classing to a frozen legacy
                            class is impossible (backend would 422 anyway). */}
                        {classes.filter((c) => !c.frozen).map((c) => (
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
                    {/* Eval-labeling phase-0 instance metadata (spec §4):
                        inert pass-through persisted via the normal
                        annotation autosave (updateInstance). */}
                    <div className="ins-row">
                      <label>Flags</label>
                      <div className="ins-flags">
                        {['boundary_uncertain', 'incomplete'].map((f) => (
                          <label key={f} className="ins-check">
                            <input type="checkbox"
                              checked={(inst.flags || []).includes(f)}
                              disabled={inst.confirmed}
                              onChange={(e) => {
                                const cur = new Set(inst.flags || []);
                                if (e.target.checked) cur.add(f); else cur.delete(f);
                                updateInstance(inst.id, { flags: [...cur] });
                              }} />
                            {f.replace('_', ' ')}
                          </label>
                        ))}
                        <label className="ins-check">
                          <input type="checkbox"
                            checked={inst.insulated === true}
                            disabled={inst.confirmed}
                            onChange={(e) => updateInstance(inst.id, { insulated: e.target.checked })} />
                          insulated
                        </label>
                      </div>
                    </div>
                    <div className="ins-row">
                      <label>Subtype</label>
                      <input className="ins-input" placeholder="e.g. ball valve"
                        value={inst.subtype || ''}
                        disabled={inst.confirmed}
                        onChange={(e) => updateInstance(inst.id, { subtype: e.target.value || null })} />
                    </div>
                    <div className="ins-row">
                      <label>Note</label>
                      <textarea className="ins-input ins-note" rows={2}
                        value={inst.note || ''}
                        disabled={inst.confirmed}
                        onChange={(e) => updateInstance(inst.id, { note: e.target.value })} />
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
        {/* No jsdom wiring test for this Instances-panel menu (Task 10 fix-up):
            mode-label.jsx has no existing test harness and pulls in the full
            instances/confirmed/selectedId state to render, unlike the small
            self-contained SamSegmentList/PresegmentList (both covered by
            sam-segment-list.jsdom.test.jsx / segment-tools.jsdom.test.jsx).
            The shared cutEligibility({list:'instance', ...}) call below is
            unit-tested in cut-eligibility.test.js. */}
        {instCutMenu && (() => {
          const target = instances.find((i) => i.id === instCutMenu.instId);
          const elig = cutEligibility({
            list: 'instance',
            isSelected: instCutMenu.instId === selectedId,
            confirmed: !!target?.confirmed,
            classFrozen: !!classes.find((c) => c.id === target?.cls)?.frozen,
          });
          return (
            <ContextMenu
              x={instCutMenu.x}
              y={instCutMenu.y}
              onClose={() => setInstCutMenu(null)}
              items={[{
                label: elig.reason === 'confirmed'
                  ? 'Edit selection… (un-confirm first)'
                  : elig.reason === 'frozen-class'
                    ? 'Edit selection… (legacy class — re-label with a primitive first)'
                    : 'Edit selection…',
                disabled: !elig.eligible,
                onSelect: () => {
                  if (!target || !Number.isFinite(target.segId)) return;
                  openCutModal([{ kind: 'instance', segId: target.segId }], target.cls);
                },
              },
              {
                label: 'Remove outliers',
                disabled: !removeOutliersEligibility({
                  list: 'instance',
                  isSelected: instCutMenu.instId === selectedId,
                  confirmed: !!target?.confirmed,
                }).eligible,
                onSelect: () => {
                  if (!target || !Number.isFinite(target.segId)) return;
                  removeOutliers({ source: 'instance', id: target.segId });
                },
              }, {
                // Fitting only READS the source points — a confirmed instance is
                // still eligible (see fit-eligibility.js).
                label: 'Fit box to selection…',
                disabled: !fitEligibility({
                  list: 'instance',
                  isSelected: instCutMenu.instId === selectedId,
                }).eligible,
                onSelect: () => {
                  if (!target || !Number.isFinite(target.segId)) return;
                  fitBoxToSelection([{ kind: 'instance', segId: target.segId }]);
                },
              }]}
            />
          );
        })()}
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

