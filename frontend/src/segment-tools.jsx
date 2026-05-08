// segment-tools.jsx — tool strip + Pick tool + Brush tool for per-point segment editing.

import { useEffect, useRef, useMemo, useState } from 'react';
import * as THREE from 'three';
import { VoxaAPI } from './api.js';
import { initSegState } from './segment-state.js';

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

// ── Presegment button ───────────────────────────────────────────────────────
// Click ⚙ to open a small popover that picks the voxel resolution and
// whether to preserve already-classified points, then re-runs the
// supervoxel presegmentation on the active scene. Slow on large clouds
// (~10–60s), so the button shows a busy state while in flight.
// RANSAC defaults — must mirror RANSAC_DEFAULTS in backend/presegment_ransac.py.
// Group order = popover layout (plane → curvature → cylinder → merge).
const RANSAC_KNOBS = [
  { key: 'plane_distance_threshold', label: 'Plane dist (m)',     def: 0.025, step: 0.005, min: 0.001 },
  { key: 'plane_min_inliers',        label: 'Plane min inliers',  def: 80,    step: 10,    min: 1     },
  { key: 'max_planes',               label: 'Max planes',         def: 25,    step: 1,     min: 1     },
  { key: 'plane_cluster_eps',        label: 'Plane gap eps (m)',  def: 0.15,  step: 0.05,  min: 0     },
  { key: 'leftover_cluster_eps',     label: 'Leftover eps (m)',   def: 0.10,  step: 0.02,  min: 0     },
  { key: 'leftover_min_pts',         label: 'Leftover min pts',   def: 30,    step: 5,     min: 1     },
  { key: 'flat_thresh',              label: 'Flat thresh',        def: 0.5,   step: 0.05,  min: 0     },
  { key: 'cylinder_ratio_thresh',    label: 'Cyl ratio thresh',   def: 3.0,   step: 0.1,   min: 1     },
  { key: 'cyl_search_radius',        label: 'Cyl radius (m)',     def: 0.12,  step: 0.01,  min: 0.001 },
  { key: 'cyl_axis_thresh',          label: 'Cyl axis dot',       def: 0.92,  step: 0.01,  min: 0, max: 1 },
  { key: 'cyl_radius_ratio',         label: 'Cyl radius ratio',   def: 1.8,   step: 0.05,  min: 1     },
  { key: 'cyl_distance_threshold',   label: 'Cyl dist (m)',       def: 0.03,  step: 0.005, min: 0.001 },
  { key: 'merge_axis_dot',           label: 'Merge axis dot',     def: 0.95,  step: 0.01,  min: 0, max: 1 },
  { key: 'merge_radius_ratio',       label: 'Merge radius ratio', def: 1.4,   step: 0.05,  min: 1     },
];

export function PresegmentButton({ segState, setSegState, prelabelRef, cloud, setCloud }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const [open, setOpen] = useState(false);
  const [resolution, setResolution] = useState(0.3);
  const [preserveLabeled, setPreserveLabeled] = useState(true);
  const [mode, setMode] = useState('voxel');
  const [ransacKnobs, setRansacKnobs] = useState(() => {
    const o = {};
    for (const k of RANSAC_KNOBS) o[k.key] = k.def;
    return o;
  });
  const [labelerStrict, setLabelerStrict] = useState(false);
  const [stats, setStats] = useState(null);  // { nSegments, meanSize } after run
  const [optStatus, setOptStatus] = useState('idle'); // 'idle' | 'running' | 'done' | 'aborted' | 'error'
  const [optInfo, setOptInfo] = useState(null);
  const optTimerRef = useRef(null);
  const popRef = useRef(null);
  const btnRef = useRef(null);
  const disabled = !cloud || busy;

  // Polling interval (not setTimeout chain) so abort/unmount can clearInterval
  // synchronously without racing an in-flight fetch resolution.
  useEffect(() => () => {
    if (optTimerRef.current) { clearInterval(optTimerRef.current); optTimerRef.current = null; }
  }, []);

  useEffect(() => {
    if (!open) return;
    function onDoc(e) {
      if (popRef.current?.contains(e.target)) return;
      if (btnRef.current?.contains(e.target)) return;
      setOpen(false);
    }
    function onKey(e) { if (e.key === 'Escape') setOpen(false); }
    document.addEventListener('mousedown', onDoc);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDoc);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  async function run() {
    if (!cloud) return;
    // Only the destructive (full-replace) path can drop unsaved edits.
    if (!preserveLabeled && segState?.dirty) {
      const ok = window.confirm(
        'Presegmenting without preserve will discard your unsaved edits. Continue?');
      if (!ok) return;
    }
    setBusy(true);
    setError(null);
    setStats(null);
    // Keep the popover open during the request so the spinner is visible.
    try {
      // Only send overrides for ransac mode; only include knobs that differ from defaults.
      let ransacParams = null;
      if (mode === 'ransac') {
        ransacParams = {};
        for (const k of RANSAC_KNOBS) {
          const v = ransacKnobs[k.key];
          if (v !== k.def && Number.isFinite(v)) ransacParams[k.key] = v;
        }
      }
      const res = await VoxaAPI.segPresegment({
        mode, resolution, preserveLabeled, ransacParams,
        labelerStrict: mode === 'ransac' && labelerStrict,
      });
      setStats({ nSegments: res.nSegments, meanSize: res.meanSegSize ?? 0 });
      if (prelabelRef) {
        prelabelRef.current = {
          classFull: res.fullClassIds.slice(),
          instanceFull: res.fullInstanceIds.slice(),
        };
      }
      setSegState(initSegState({
        classFull: res.fullClassIds,
        instanceFull: res.fullInstanceIds,
        isFromPrelabel: true,
        segBoxes: (res.segIds && res.segCenters && res.segSizes)
          ? { segIds: res.segIds, segCenters: res.segCenters, segSizes: res.segSizes }
          : null,
        segHulls: (res.hullVertices && res.hullFaces && res.hullFaceSeg)
          ? { vertices: res.hullVertices, faces: res.hullFaces, faceSeg: res.hullFaceSeg }
          : null,
      }));
      // Project the freshly-computed full arrays onto the subsampled
      // cloud so the viewer's recolor effect (which reads cloud.classIds /
      // cloud.instanceIds) reflects the presegmentation. Replacing cloud
      // with a shallow copy triggers that effect cleanly.
      if (setCloud) {
        const subIdx = cloud.subsampleIdx;
        const subN = (cloud.positions?.length || 0) / 3;
        const subClass = new Int8Array(subN);
        const subInst = new Int32Array(subN);
        for (let p = 0; p < subN; p++) {
          const f = subIdx ? subIdx[p] : p;
          subClass[p] = res.fullClassIds[f];
          subInst[p] = res.fullInstanceIds[f];
        }
        setCloud({
          ...cloud,
          classIds: subClass,
          instanceIds: subInst,
          isFromPrelabel: true,
        });
      }
      // Point colouring while preseg is active is handled inside Viewer
      // (it overrides ``colorMode`` to use the same segment-id hue as the
      // hull mesh). Nothing to switch here.
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      setBusy(false);
    }
  }

  function stopPolling() {
    if (optTimerRef.current) { clearInterval(optTimerRef.current); optTimerRef.current = null; }
  }

  async function hydrateFromSession() {
    const s = await VoxaAPI.segState();
    if (!s) return;
    setStats({ nSegments: s.nSegments, meanSize: s.nAssigned / Math.max(s.nSegments, 1) });
    if (prelabelRef) {
      prelabelRef.current = {
        classFull: s.fullClassIds.slice(),
        instanceFull: s.fullInstanceIds.slice(),
      };
    }
    setSegState(initSegState({
      classFull: s.fullClassIds,
      instanceFull: s.fullInstanceIds,
      isFromPrelabel: true,
      segBoxes: (s.segIds && s.segCenters && s.segSizes)
        ? { segIds: s.segIds, segCenters: s.segCenters, segSizes: s.segSizes }
        : null,
      segHulls: (s.hullVertices && s.hullFaces && s.hullFaceSeg)
        ? { vertices: s.hullVertices, faces: s.hullFaces, faceSeg: s.hullFaceSeg }
        : null,
    }));
    if (setCloud && cloud) {
      const subIdx = cloud.subsampleIdx;
      const subN = (cloud.positions?.length || 0) / 3;
      const subClass = new Int8Array(subN);
      const subInst = new Int32Array(subN);
      for (let p = 0; p < subN; p++) {
        const f = subIdx ? subIdx[p] : p;
        subClass[p] = s.fullClassIds[f];
        subInst[p] = s.fullInstanceIds[f];
      }
      setCloud({
        ...cloud,
        classIds: subClass,
        instanceIds: subInst,
        isFromPrelabel: true,
      });
    }
  }

  async function runOptimize() {
    if (!cloud || mode !== 'ransac') return;
    setBusy(true);
    setError(null);
    setStats(null);
    setOptStatus('running');
    setOptInfo(null);
    try {
      const { jobId, total } = await VoxaAPI.segPresegOptimizeStart({
        nTrials: 20,
        subsampleN: 200_000,
        preserveLabeled,
      });
      setOptInfo({ jobId, trial: 0, total, bestScore: null, bestParams: null });
      optTimerRef.current = setInterval(async () => {
        try {
          const s = await VoxaAPI.segPresegOptimizeStatus(jobId);
          setOptInfo({ jobId, trial: s.trial, total: s.total, bestScore: s.bestScore, bestParams: s.bestParams });
          if (s.status === 'done' || s.status === 'aborted' || s.status === 'error') {
            stopPolling();
            setBusy(false);
            setOptStatus(s.status);
            if (s.status === 'done' && s.bestParams) {
              setRansacKnobs((prev) => ({ ...prev, ...s.bestParams }));
              await hydrateFromSession();
            } else if (s.status === 'error') {
              setError(s.error || 'optimize failed');
            }
          }
        } catch (e) {
          stopPolling();
          setBusy(false);
          setOptStatus('error');
          setError(String(e.message || e));
        }
      }, 1500);
    } catch (e) {
      stopPolling();
      setBusy(false);
      setOptStatus('error');
      setError(String(e.message || e));
    }
  }

  async function abortOptimize() {
    if (!optInfo?.jobId) return;
    try {
      await VoxaAPI.segPresegOptimizeAbort(optInfo.jobId);
    } catch (e) {
      setError(String(e.message || e));
    }
  }

  const title = busy
    ? 'Running presegmentation…'
    : error
      ? `Presegment failed: ${error}`
      : !cloud
        ? 'Load a scene first'
        : 'Configure & run voxel presegmentation';

  return (
    <span style={{ position: 'relative', display: 'inline-block' }}>
      <button
        ref={btnRef}
        type="button"
        className={'tool-btn mini' + (busy ? ' busy' : '') + (error ? ' error' : '') + (open ? ' active' : '')}
        disabled={disabled}
        title={title}
        onClick={() => !disabled && setOpen((v) => !v)}
      >
        <span className="tool-ico" aria-hidden>{busy ? '…' : '⚙'}</span>
      </button>
      {open && (
        <div
          ref={popRef}
          className="preseg-popover"
          style={{
            position: 'absolute',
            top: 'calc(100% + 6px)',
            right: 0,
            zIndex: 1000,
            minWidth: 220,
            padding: 10,
            background: 'var(--panel-bg, #1d1d1d)',
            border: '1px solid var(--panel-border, #333)',
            borderRadius: 6,
            boxShadow: '0 6px 18px rgba(0,0,0,0.4)',
            display: 'flex', flexDirection: 'column', gap: 8,
            fontSize: 12, color: 'var(--text, #ddd)',
          }}
        >
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            <span style={{ opacity: 0.7 }}>Mode</span>
            {[
              { id: 'voxel',  label: 'Voxel (uniform spatial)',         disabled: false },
              { id: 'ransac', label: 'RANSAC (curvature primitives)',   disabled: false },
              { id: 'model',  label: 'Model (learned merge — disabled)', disabled: true  },
            ].map((opt) => (
              <label
                key={opt.id}
                style={{
                  display: 'flex', alignItems: 'center', gap: 6,
                  opacity: opt.disabled ? 0.4 : 1,
                  cursor: opt.disabled ? 'not-allowed' : 'pointer',
                }}
              >
                <input
                  type="radio"
                  name="preseg-mode"
                  value={opt.id}
                  checked={mode === opt.id}
                  disabled={opt.disabled}
                  onChange={() => setMode(opt.id)}
                />
                <span>{opt.label}</span>
              </label>
            ))}
          </div>
          <label
            style={{
              display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8,
              opacity: mode === 'voxel' ? 1 : 0.4,
            }}
          >
            <span>Voxel resolution (m)</span>
            <input
              type="number"
              step="0.05"
              min="0.01"
              max="5"
              value={resolution}
              disabled={mode !== 'voxel'}
              onChange={(e) => setResolution(Math.max(0.01, parseFloat(e.target.value) || 0.05))}
              style={{ width: 70, background: '#111', color: '#eee', border: '1px solid #444', borderRadius: 3, padding: '2px 4px' }}
            />
          </label>
          <div
            style={{
              display: 'flex', flexDirection: 'column', gap: 4,
              opacity: mode === 'ransac' ? 1 : 0.4,
              borderTop: '1px solid #2a2a2a',
              paddingTop: 6,
            }}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span style={{ opacity: 0.7 }}>RANSAC params</span>
              <button
                type="button"
                className="tool-btn"
                style={{ width: 'auto', padding: '1px 6px', fontSize: 10 }}
                disabled={mode !== 'ransac'}
                title="Reset RANSAC params to defaults"
                onClick={() => {
                  const o = {};
                  for (const k of RANSAC_KNOBS) o[k.key] = k.def;
                  setRansacKnobs(o);
                }}
              >reset</button>
            </div>
            <label
              style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11 }}
              title="Run the bit-for-bit industrial_point_labeler pipeline: no plane CC split, no nearest-neighbour catchall, labeler-style cylinder merge. Use to A/B against the labeler."
            >
              <input
                type="checkbox"
                checked={labelerStrict}
                disabled={mode !== 'ransac'}
                onChange={(e) => setLabelerStrict(e.target.checked)}
              />
              <span>Labeler-strict (match industrial_point_labeler)</span>
            </label>
            {RANSAC_KNOBS.map((k) => (
              <label
                key={k.key}
                style={{
                  display: 'flex', justifyContent: 'space-between', alignItems: 'center', gap: 8,
                }}
              >
                <span style={{ fontSize: 11 }}>{k.label}</span>
                <input
                  type="number"
                  step={k.step}
                  min={k.min}
                  {...(k.max !== undefined ? { max: k.max } : {})}
                  value={ransacKnobs[k.key]}
                  disabled={mode !== 'ransac'}
                  onChange={(e) => {
                    const v = parseFloat(e.target.value);
                    setRansacKnobs((prev) => ({ ...prev, [k.key]: Number.isFinite(v) ? v : k.def }));
                  }}
                  style={{ width: 70, background: '#111', color: '#eee', border: '1px solid #444', borderRadius: 3, padding: '2px 4px', fontSize: 11 }}
                />
              </label>
            ))}
          </div>
          <label style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
            <input
              type="checkbox"
              checked={preserveLabeled}
              onChange={(e) => setPreserveLabeled(e.target.checked)}
            />
            <span>Preserve classified points</span>
          </label>
          {stats && (
            <div style={{ opacity: 0.8, fontSize: 11 }}>
              {stats.nSegments} segments · mean size {stats.meanSize.toFixed(0)} pts
            </div>
          )}
          {optStatus === 'running' ? (
            <div style={{ display: 'flex', gap: 8, justifyContent: 'space-between', alignItems: 'center', marginTop: 2 }}>
              <span style={{ display: 'inline-flex', alignItems: 'center', gap: 8, fontSize: 11, opacity: 0.85 }}>
                <span className="preseg-spinner" aria-hidden />
                Trial {optInfo?.trial ?? 0}/{optInfo?.total ?? 0}
                {optInfo?.bestScore != null ? ` · best ${optInfo.bestScore.toFixed(4)}` : ''}
              </span>
              <button
                type="button"
                className="tool-btn"
                style={{ width: 'auto', padding: '2px 10px' }}
                onClick={abortOptimize}
              >Abort</button>
            </div>
          ) : busy ? (
            <div style={{ display: 'flex', gap: 8, alignItems: 'center', marginTop: 2 }}>
              <span className="preseg-spinner" aria-hidden />
              <span style={{ fontSize: 11, opacity: 0.85 }}>
                Running {mode === 'ransac' ? 'RANSAC' : 'voxel'} preseg…
              </span>
            </div>
          ) : (
            <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end', marginTop: 2 }}>
              <button
                type="button"
                className="tool-btn"
                style={{ width: 'auto', padding: '2px 10px' }}
                onClick={() => setOpen(false)}
              >Cancel</button>
              <button
                type="button"
                className="tool-btn"
                style={{ width: 'auto', padding: '2px 10px' }}
                disabled={mode !== 'ransac' || busy}
                title={mode !== 'ransac' ? 'Optimize is only available in RANSAC mode' : 'Search for best RANSAC params'}
                onClick={runOptimize}
              >Optimize</button>
              <button
                type="button"
                className="tool-btn active"
                style={{ width: 'auto', padding: '2px 10px' }}
                onClick={run}
              >Run</button>
            </div>
          )}
          {error && <div style={{ color: '#f88' }}>{error}</div>}
        </div>
      )}
    </span>
  );
}


// ── Presegment list ─────────────────────────────────────────────────────────
// Sidebar list of every per-point segment, one row per instance id from
// segState.summary. The list is *only* a fast selector — it groups
// points so the user doesn't have to lasso them in 3D. Click a row to
// select that group, shift-click for multi-select. Confirmation lives
// elsewhere (the Instances panel on the right): there is no parallel
// confirm flow here on purpose.
export function PresegmentList({
  segState, setSegState, classes, viewerRef, cloud,
  showSegHulls = true, setShowSegHulls = null,
  excludeSegIds = null,
}) {
  const segmentsAll = useMemo(() => {
    if (!segState) return [];
    const out = [];
    for (const [id, info] of segState.summary.entries()) {
      if (excludeSegIds && excludeSegIds.has(id)) continue;
      out.push({ id, classId: info.classId, nPoints: info.nPoints });
    }
    out.sort((a, b) => b.nPoints - a.nPoints);
    return out;
  }, [segState, excludeSegIds]);

  const classesById = useMemo(() => {
    const out = {};
    classes.forEach((c, i) => { out[i] = c; out[c.id] = c; });
    return out;
  }, [classes]);

  if (!segState) return null;

  const onRowClick = (segId, evt) => {
    if (!(evt.ctrlKey || evt.metaKey || evt.shiftKey)) return;
    setSegState((s) => {
      if (!s) return s;
      const next = new Set(s.selection);
      next.has(segId) ? next.delete(segId) : next.add(segId);
      return { ...s, selection: next };
    });
  };

  const focusSegment = (segId) => {
    if (!viewerRef?.current?.frame || !cloud) return;
    const subIdx = cloud.subsampleIdx;
    const pos = cloud.positions;
    if (!pos) return;
    const inst = segState.instanceFull;
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    let n = 0;
    const subN = pos.length / 3;
    for (let p = 0; p < subN; p++) {
      const fullIdx = subIdx ? subIdx[p] : p;
      if (inst[fullIdx] !== segId) continue;
      const x = pos[p * 3], y = pos[p * 3 + 1], z = pos[p * 3 + 2];
      if (x < minX) minX = x; if (y < minY) minY = y; if (z < minZ) minZ = z;
      if (x > maxX) maxX = x; if (y > maxY) maxY = y; if (z > maxZ) maxZ = z;
      n++;
    }
    if (n === 0) return;
    // viewer.frame() calls THREE.Vector3.copy() on the center, which
    // requires a Vector3-like object (not a plain array — copy() is a
    // silent no-op on arrays and the camera ends up looking at NaN).
    const center = new THREE.Vector3(
      (minX + maxX) / 2, (minY + maxY) / 2, (minZ + maxZ) / 2);
    const radius = Math.max(maxX - minX, maxY - minY, maxZ - minZ) * 0.6 + 0.05;
    viewerRef.current.frame(center, radius);
  };

  const total = segmentsAll.length;

  return (
    <div className="preseg-panel">
      <div className="side-hd" style={{ marginTop: 14, display: 'flex', alignItems: 'center', gap: 6 }}>
        <span>Presegments</span>
        <span className="badge-soft">{total}</span>
        {setShowSegHulls && (
          <button
            type="button"
            className={'tool-btn mini' + (showSegHulls ? ' active' : '')}
            onClick={() => setShowSegHulls(!showSegHulls)}
            title={showSegHulls
              ? 'Hide hull overlay (points stay coloured by segment)'
              : 'Show hull overlay'}
            style={{ marginLeft: 'auto', padding: '2px 6px', fontSize: 11 }}
          >
            {showSegHulls ? '👁 Hulls' : '◌ Hulls'}
          </button>
        )}
      </div>
      <div className="inst-list" style={{ maxHeight: '40vh', overflowY: 'auto' }}>
        {total === 0 && (
          <div className="sugg-empty" style={{ fontSize: '11px', padding: '6px 4px' }}>
            No presegments. Click ⚙ to choose a mode and run.
          </div>
        )}
        {segmentsAll.map((seg) => {
          const cls = classesById[seg.classId];
          const isSel = segState.selection.has(seg.id);
          const dot = cls?.color || '#6b7280';
          return (
            <div key={seg.id}
              className={'inst-row' + (isSel ? ' selected' : '')}
              onClick={(e) => onRowClick(seg.id, e)}
              title={isSel ? 'Ctrl/Shift-click to deselect'
                           : 'Ctrl/Shift-click to select'}>
              <span className="inst-dot" style={{ background: dot }} />
              <div className="inst-text">
                <b>#{seg.id}</b>
                <em>{cls?.label || (seg.classId < 0 ? 'unlabeled' : `cls ${seg.classId}`)} · {seg.nPoints.toLocaleString()}</em>
              </div>
              <button className="inst-edit-btn"
                onClick={(e) => { e.stopPropagation(); focusSegment(seg.id); }}
                title="Focus camera on segment">◎</button>
            </div>
          );
        })}
      </div>
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
      // ctrl/cmd/shift = multi-select, plain click = replace.
      const next = new Set(segState.selection);
      const additive = evt.shiftKey || evt.ctrlKey || evt.metaKey;
      if (additive) {
        next.has(instId) ? next.delete(instId) : next.add(instId);
      } else {
        if (next.size === 1 && next.has(instId)) next.clear();
        else { next.clear(); next.add(instId); }
      }
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
