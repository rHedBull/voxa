// mode-compare.jsx — per-point comparison of two finished labelings (session
// outputs / presegs): synced split view colored by class per side, agreement %,
// and a per-class IoU/precision/recall table with confusion pairs.

import { useState as useStateCmp, useRef as useRefCmp,
         useEffect as useEffectCmp, useMemo as useMemoCmp } from 'react';
import { Viewer } from './viewer.jsx';
import { CameraPresets, NavModeToggle, HelpButton } from './viewport-atoms.jsx';
import { VoxaAPI } from './api.js';

const COLOR_A = '#10b981';
const COLOR_B = '#5b8def';

// Build the selectable source list from App's session/preseg state.
// Sessions sort saved_at desc and disable when they have no saved output;
// presegs are always comparable.
function buildSources(sessions, presegs) {
  const ses = [...(sessions || [])]
    .sort((x, y) => String(y.saved_at || '').localeCompare(String(x.saved_at || '')))
    .map((s) => ({
      kind: 'session', id: s.session_id, label: s.name || s.session_id,
      disabled: !s.has_output, hint: s.has_output ? null : 'no output',
    }));
  const pre = (presegs || []).map((p) => ({
    kind: 'preseg', id: p.preseg_id, label: `${p.preseg_id} · ${p.generator || '?'}`,
    disabled: false, hint: null,
  }));
  return [...ses, ...pre];
}

const srcKey = (s) => (s ? `${s.kind}:${s.id}` : '');

function SourceSelect({ side, color, sources, value, onChange }) {
  return (
    <select
      className="cmp-source-select"
      style={{ borderLeftColor: color }}
      value={srcKey(value)}
      onChange={(e) => {
        const s = sources.find((x) => srcKey(x) === e.target.value);
        if (s) onChange(s);
      }}>
      {sources.map((s) => (
        <option key={srcKey(s)} value={srcKey(s)} disabled={s.disabled}>
          {side} · {s.label}{s.hint ? ` (${s.hint})` : ''}
        </option>
      ))}
    </select>
  );
}

function ComparePanel({ title, color, viewerProps, viewerRef }) {
  return (
    <div className="cmp-panel">
      <div className="cmp-hd">
        <span className="cmp-tag"
          style={{ background: color + '22', color, borderColor: color + '55' }}>
          {title}
        </span>
      </div>
      <div className="cmp-vp">
        <Viewer ref={viewerRef} {...viewerProps} />
      </div>
    </div>
  );
}

export function CompareMode({ cloud, theme, sceneName, isAnnotated,
                             sessions, presegs, activeSessionId,
                             navMode, onNavModeChange }) {
  const leftRef = useRefCmp();
  const rightRef = useRefCmp();
  const [syncCameras, setSyncCameras] = useStateCmp(true);
  const [srcA, setSrcA] = useStateCmp(null);
  const [srcB, setSrcB] = useStateCmp(null);
  const [cmp, setCmp] = useStateCmp(null);
  const [error, setError] = useStateCmp(null);

  const sources = useMemoCmp(() => buildSources(sessions, presegs), [sessions, presegs]);
  const enabled = useMemoCmp(() => sources.filter((s) => !s.disabled), [sources]);

  // Defaults: A = active session if it has output, else first enabled source;
  // B = the next distinct enabled source. Re-derive whenever the scene or
  // source set changes.
  useEffectCmp(() => {
    if (enabled.length < 2) { setSrcA(null); setSrcB(null); return; }
    const active = enabled.find((s) => s.kind === 'session' && s.id === activeSessionId);
    const a = active || enabled[0];
    const b = enabled.find((s) => srcKey(s) !== srcKey(a));
    setSrcA(a);
    setSrcB(b || null);
  }, [sceneName, sources, activeSessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  // Fetch metrics + both class arrays whenever the scene or either source changes.
  useEffectCmp(() => {
    if (!isAnnotated || !sceneName || !srcA || !srcB) { setCmp(null); return; }
    let cancel = false;
    setError(null);
    VoxaAPI.comparePoints(sceneName,
      { kind: srcA.kind, id: srcA.id }, { kind: srcB.kind, id: srcB.id })
      .then((r) => { if (!cancel) { setCmp(r); setError(null); } })
      .catch((e) => { if (!cancel) { setCmp(null); setError(e.message || String(e)); } });
    return () => { cancel = true; };
  }, [sceneName, isAnnotated, srcKey(srcA), srcKey(srcB)]); // eslint-disable-line react-hooks/exhaustive-deps

  const onLeftMove = () => {
    if (!syncCameras) return;
    const s = leftRef.current?.getCameraState();
    if (s) rightRef.current?.setCameraState(s);
  };
  const onRightMove = () => {
    if (!syncCameras) return;
    const s = rightRef.current?.getCameraState();
    if (s) leftRef.current?.setCameraState(s);
  };

  // Project a full-res class array onto the subsampled cloud (same loop as
  // App.jsx) so the viewer's colorMode='class' path can color each side.
  const projectClassIds = (full) => {
    if (!cloud || !full) return null;
    const subIdx = cloud.subsampleIdx;
    const subN = (cloud.positions?.length || 0) / 3;
    const out = new Int8Array(subN);
    for (let p = 0; p < subN; p++) out[p] = full[subIdx ? subIdx[p] : p];
    return out;
  };
  const classA = useMemoCmp(() => projectClassIds(cmp?.aClassIds), [cmp, cloud]);
  const classB = useMemoCmp(() => projectClassIds(cmp?.bClassIds), [cmp, cloud]);

  const m = cmp?.metrics;
  const palette = cmp?.palette || [];
  const nameFor = (id) => palette.find((p) => p.id === id)?.label || `cls ${id}`;
  const colorFor = (id) => palette.find((p) => p.id === id)?.color || '#7c8088';

  const rows = useMemoCmp(() => {
    if (!m?.per_class) return [];
    return [...m.per_class].sort((x, y) => {
      const ax = x.iou == null ? Infinity : x.iou;
      const ay = y.iou == null ? Infinity : y.iou;
      return ax - ay; // ascending; nulls last
    });
  }, [m]);

  const agr = m?.agreement;
  const agrTip = m?.agreement_all != null
    ? `over points labeled in at least one source; all-points: ${(m.agreement_all * 100).toFixed(1)}%`
    : undefined;

  const helpSections = [
    {
      title: 'Compare',
      items: [
        { keys: ['A / B'], desc: 'Pick two finished labelings (session output or preseg)' },
        { keys: ['Sync'], desc: 'Both viewports share camera while on' },
        { keys: ['Drag'], desc: 'Either side; the other follows when synced' },
      ],
    },
    {
      title: 'Camera',
      items: navMode === 'walk'
        ? [
            { keys: ['W', 'A', 'S', 'D'], desc: 'Move (XZ plane)' },
            { keys: ['Q', 'E'], desc: 'Down / up' },
            { keys: ['Shift'], desc: 'Hold to sprint' },
            { keys: ['Drag'], desc: 'Look around' },
          ]
        : [
            { keys: ['Drag'], desc: 'Orbit' },
            { keys: ['Shift', 'Drag'], desc: 'Pan' },
            { keys: ['Scroll'], desc: 'Zoom' },
          ],
    },
    {
      title: 'Metrics',
      items: [
        { keys: ['Agree'], desc: 'Fraction of points where A and B agree (labeled in either side)' },
        { keys: ['IoU'], desc: 'Per-class point overlap: tp / (A∪B)' },
        { keys: ['P / R'], desc: 'Precision (of B’s claims) / recall (of A’s points)' },
      ],
    },
    {
      title: 'Other',
      items: [
        { keys: ['?'], desc: 'Toggle this panel' },
        { keys: ['Esc'], desc: 'Close panel' },
      ],
    },
  ];

  const emptyState = (text) => (
    <div className="mode-root compare">
      <div style={{ flex: 1, display: 'flex', alignItems: 'center',
                    justifyContent: 'center', color: 'var(--text-faint)',
                    fontSize: 13, textAlign: 'center', padding: 24 }}>
        {text}
      </div>
    </div>
  );

  if (!isAnnotated) return emptyState('Compare needs an annotated scan');
  if (enabled.length < 2) {
    return emptyState('Need two finished labelings — save a second session or register a model preseg.');
  }

  return (
    <div className="mode-root compare">
      <div className="cmp-bar">
        <div className="cmp-bar-metrics">
          <SourceSelect side="A" color={COLOR_A} sources={sources}
            value={srcA} onChange={setSrcA} />
          <SourceSelect side="B" color={COLOR_B} sources={sources}
            value={srcB} onChange={setSrcB} />
          {error
            ? <span className="cmp-metric" style={{ color: 'oklch(0.72 0.17 30)' }}>{error}</span>
            : (
              <>
                <span className="cmp-metric" title={agrTip}><label>Agree</label>
                  <b className="mono accent">{agr != null ? `${(agr * 100).toFixed(1)}%` : '—'}</b></span>
                <span className="cmp-metric"><label>Pts</label>
                  <b className="mono">{m
                    ? `A: ${m.n_labeled_a.toLocaleString()} · B: ${m.n_labeled_b.toLocaleString()}`
                    : '—'}</b></span>
              </>
            )}
        </div>
        <div className="cmp-bar-controls">
          <span className="cmp-toggle">
            <NavModeToggle navMode={navMode} onChange={onNavModeChange} /></span>
          <span className="cmp-toggle"><label>Sync</label>
            <button className={'sw' + (syncCameras ? ' on' : '')}
              onClick={() => setSyncCameras(!syncCameras)}><i /></button></span>
          <span className="cmp-toggle"><label>View</label>
            <CameraPresets onPreset={(p) => {
              leftRef.current?.preset(p);
              rightRef.current?.preset(p);
            }} /></span>
          <span className="cmp-toggle"><HelpButton sections={helpSections} /></span>
        </div>
      </div>

      <div className="cmp-grid">
        <ComparePanel
          title={srcA?.label || 'Source A'}
          color={COLOR_A}
          viewerRef={leftRef}
          viewerProps={{
            cloud: { ...cloud, classIds: classA, classPalette: palette },
            colorMode: 'class', showCuboids: false,
            background: theme.bg, floorColor: theme.floor, navMode,
            onCameraChange: onLeftMove,
          }}
        />
        <ComparePanel
          title={srcB?.label || 'Source B'}
          color={COLOR_B}
          viewerRef={rightRef}
          viewerProps={{
            cloud: { ...cloud, classIds: classB, classPalette: palette },
            colorMode: 'class', showCuboids: false,
            background: theme.bg, floorColor: theme.floor, navMode,
            onCameraChange: onRightMove,
          }}
        />
      </div>

      <div className="cmp-table">
        <div className="cmp-table-hd">
          <div>Class</div>
          <div>IoU</div>
          <div>P</div>
          <div>R</div>
          <div>pts A</div>
          <div>pts B</div>
        </div>
        {rows.map((r) => (
          <div key={r.class_id} className="cmp-table-row">
            <div className="cmp-class-name">
              <i className="cmp-dot" style={{ background: colorFor(r.class_id) }} />
              {nameFor(r.class_id)}
            </div>
            <div className="mono">{r.iou != null ? r.iou.toFixed(3) : '—'}</div>
            <div className="mono">{r.precision != null ? r.precision.toFixed(3) : '—'}</div>
            <div className="mono">{r.recall != null ? r.recall.toFixed(3) : '—'}</div>
            <div className="mono">{r.n_a.toLocaleString()}</div>
            <div className="mono">{r.n_b.toLocaleString()}</div>
          </div>
        ))}
        {rows.length === 0 && (
          <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-faint)', fontSize: 12 }}>
            {error ? error : 'No overlapping classes to compare yet.'}
          </div>
        )}
        {m?.confusion?.length > 0 && (
          <div className="cmp-confusion">
            <div className="cmp-confusion-hd">Top confusions</div>
            {m.confusion.map((c, i) => (
              <div key={i} className="cmp-confusion-row">
                <span style={{ color: colorFor(c.a_class) }}>{nameFor(c.a_class)}</span>
                {' → '}
                <span style={{ color: colorFor(c.b_class) }}>{nameFor(c.b_class)}</span>
                {' · '}
                <span className="mono">{c.n.toLocaleString()} pts</span>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
