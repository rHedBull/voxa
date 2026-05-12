// segment-tools.jsx — Presegment trigger + presegment list (selection panel).
// Selection happens via Ctrl/Cmd-click in the viewport (wired in mode-label).

import { useEffect, useRef, useMemo, useState } from 'react';
import * as THREE from 'three';
import { VoxaAPI } from './api.js';
import { initSegState } from './segment-state.js';

// ── Presegment button ───────────────────────────────────────────────────────
// Click ⚙ to open a small popover that picks the voxel resolution and
// whether to preserve already-classified points, then re-runs the
// supervoxel presegmentation on the active scene. Slow on large clouds
// (~10–60s), so the button shows a busy state while in flight.
// RANSAC defaults — must mirror RANSAC_DEFAULTS in backend/presegment_ransac.py.
// Group order = popover layout (plane → curvature → cylinder → merge).
const RANSAC_KNOBS = [
  { key: 'plane_distance_threshold', label: 'Plane dist (m)',     def: 0.025, step: 0.005, min: 0.001 },
  { key: 'plane_min_inliers',        label: 'Plane min inliers',  def: 30,    step: 5,     min: 1     },
  { key: 'max_planes',               label: 'Max planes',         def: 40,    step: 1,     min: 1     },
  { key: 'plane_cluster_eps',        label: 'Plane gap eps (m)',  def: 0.15,  step: 0.05,  min: 0     },
  { key: 'leftover_cluster_eps',     label: 'Leftover eps (m)',   def: 0.10,  step: 0.02,  min: 0     },
  { key: 'leftover_min_pts',         label: 'Leftover min pts',   def: 30,    step: 5,     min: 1     },
  { key: 'flat_thresh',              label: 'Flat thresh',        def: 0.9,   step: 0.05,  min: 0     },
  { key: 'cylinder_ratio_thresh',    label: 'Cyl ratio thresh',   def: 4.0,   step: 0.1,   min: 1     },
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
  // Run-new state (SAM3 feature-aware sub-segmentation)
  const [runMode, setRunMode] = useState('ransac');
  const [sam3Info, setSam3Info] = useState(null);    // {scene, root, root_exists, runs}
  const [sam3Picked, setSam3Picked] = useState(() => new Set());  // Set<absolutePath>
  const [forceRecompute, setForceRecompute] = useState(false);
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
    refreshSam3();
  }, [open, cloud?.recenterOffset]);

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

  async function refreshSam3() {
    try {
      const info = await VoxaAPI.sam3ListRenders();
      setSam3Info(info);
      // Default: select all runs newest-first; user can uncheck.
      setSam3Picked(new Set((info.runs || []).map((r) => r.path)));
    } catch {
      setSam3Info(null);
      setSam3Picked(new Set());
    }
  }

  function toggleSam3Pick(path) {
    setSam3Picked((prev) => {
      const next = new Set(prev);
      if (next.has(path)) next.delete(path); else next.add(path);
      return next;
    });
  }

  function selectAllSam3(yes) {
    if (!sam3Info) return;
    setSam3Picked(yes ? new Set(sam3Info.runs.map((r) => r.path)) : new Set());
  }

  async function runNewPreseg() {
    if (!cloud || busy) return;
    setBusy(true);
    setError(null);
    try {
      const sam3 = (runMode === 'ransac' && sam3Picked.size > 0)
        ? {
            render_dirs: Array.from(sam3Picked),
            force_recompute: forceRecompute,
          }
        : null;
      const res = await VoxaAPI.segPresegment({
        mode: runMode,
        preserveLabeled: true,
        sam3,
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
          ? { segIds: res.segIds, segCenters: res.segCenters, segSizes: res.segSizes } : null,
        segHulls: (res.hullVertices && res.hullFaces && res.hullFaceSeg)
          ? { vertices: res.hullVertices, faces: res.hullFaces, faceSeg: res.hullFaceSeg } : null,
      }));
      if (setCloud && cloud) {
        const subIdx = cloud.subsampleIdx;
        const subN = (cloud.positions?.length || 0) / 3;
        const subClass = new Int8Array(subN);
        const subInst = new Int32Array(subN);
        for (let p = 0; p < subN; p++) {
          const f = subIdx ? subIdx[p] : p;
          subClass[p] = res.fullClassIds[f];
          subInst[p] = res.fullInstanceIds[f];
        }
        setCloud({ ...cloud, classIds: subClass, instanceIds: subInst, isFromPrelabel: true });
      }
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
          {/* ── Run new preseg (SAM3 feature-aware) ─────────── */}
          <div style={{ borderBottom: '1px solid var(--panel-border, #333)', paddingBottom: 8, display: 'flex', flexDirection: 'column', gap: 6 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
              <span style={{ opacity: 0.7, flex: 1 }}>Run new preseg</span>
              <select
                value={runMode}
                disabled={busy}
                onChange={(e) => setRunMode(e.target.value)}
                style={{ background: '#222', color: '#ddd', border: '1px solid #444', borderRadius: 3, fontSize: 11, padding: '1px 3px' }}
              >
                <option value="voxel">voxel</option>
                <option value="ransac">ransac</option>
              </select>
            </div>
            {runMode === 'ransac' && (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 4, marginLeft: 2 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6 }}>
                  <span style={{ opacity: 0.6, fontSize: 10, flex: 1 }}>
                    SAM3 renders {sam3Info ? `(${sam3Info.runs?.length ?? 0} found)` : ''}
                  </span>
                  <button type="button" className="tool-btn" disabled={busy}
                    style={{ width: 'auto', padding: '0 6px', fontSize: 10 }}
                    onClick={() => selectAllSam3(true)} title="Select all">all</button>
                  <button type="button" className="tool-btn" disabled={busy}
                    style={{ width: 'auto', padding: '0 6px', fontSize: 10 }}
                    onClick={() => selectAllSam3(false)} title="Clear selection">none</button>
                  <button type="button" className="tool-btn" disabled={busy}
                    style={{ width: 'auto', padding: '0 6px', fontSize: 10 }}
                    onClick={refreshSam3} title="Refresh">↻</button>
                </div>
                {sam3Info && !sam3Info.root_exists && (
                  <div style={{ fontSize: 10, color: '#f88' }}>
                    VOXA_RENDERS_ROOT missing: {sam3Info.root}
                  </div>
                )}
                {sam3Info && sam3Info.root_exists && (sam3Info.runs?.length ?? 0) === 0 && (
                  <div style={{ fontSize: 10, opacity: 0.6 }}>
                    No runs found for scene "{sam3Info.scene}" under {sam3Info.root}
                  </div>
                )}
                {sam3Info && (sam3Info.runs?.length ?? 0) > 0 && (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: 2, maxHeight: '24vh', overflowY: 'auto', fontSize: 11 }}>
                    {sam3Info.runs.map((r) => {
                      const ageS = Math.max(0, Date.now() / 1000 - r.mtime);
                      const age = ageS < 60 ? `${Math.round(ageS)}s`
                        : ageS < 3600 ? `${Math.round(ageS / 60)}m`
                        : ageS < 86400 ? `${Math.round(ageS / 3600)}h`
                        : `${Math.round(ageS / 86400)}d`;
                      return (
                        <label key={r.path} style={{ display: 'flex', alignItems: 'center', gap: 6, cursor: 'pointer' }} title={r.path}>
                          <input type="checkbox" checked={sam3Picked.has(r.path)}
                            disabled={busy}
                            onChange={() => toggleSam3Pick(r.path)} />
                          <span style={{ flex: 1, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                            {r.name}
                          </span>
                          <span style={{ opacity: 0.55, fontSize: 10 }}>
                            {r.n_frames}f · {r.has_orbit_target ? 'orbit' : 'route'} · {age}
                          </span>
                        </label>
                      );
                    })}
                  </div>
                )}
                {sam3Info && sam3Picked.size > 0 && (
                  <label style={{ fontSize: 10, opacity: 0.75, display: 'flex', alignItems: 'center', gap: 4 }}>
                    <input type="checkbox" checked={forceRecompute}
                      disabled={busy}
                      onChange={(e) => setForceRecompute(e.target.checked)} />
                    force recompute features
                  </label>
                )}
              </div>
            )}
            <button
              type="button"
              className="tool-btn"
              style={{ width: 'auto', padding: '3px 10px', fontSize: 11, alignSelf: 'flex-start' }}
              disabled={disabled}
              onClick={runNewPreseg}
              title={runMode === 'ransac' && sam3Picked.size > 0
                ? `Run RANSAC + SAM3 features (${sam3Picked.size} render run${sam3Picked.size === 1 ? '' : 's'})`
                : `Run ${runMode} preseg`}
            >Run {runMode}{runMode === 'ransac' && sam3Picked.size > 0 ? ` + SAM3 ×${sam3Picked.size}` : ''}</button>
          </div>

          {/* ── Mode picker ─────────────────────────────────── */}
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


