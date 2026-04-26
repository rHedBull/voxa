// mode-compare.jsx — synced split view: GT (solid) vs prediction (dashed),
// with a server-computed diff table.

import { useState as useStateCmp, useRef as useRefCmp,
         useEffect as useEffectCmp } from 'react';
import { Viewer } from './viewer.jsx';
import { CameraPresets, NavModeToggle } from './viewport-atoms.jsx';
import { VoxaAPI } from './api.js';

function ComparePanel({ title, badge, badgeColor, viewerProps, viewerRef, stats }) {
  return (
    <div className="cmp-panel">
      <div className="cmp-hd">
        <span className="cmp-title">{title}</span>
        <span className="cmp-badge"
          style={{ background: badgeColor + '22', color: badgeColor, borderColor: badgeColor + '55' }}>
          {badge}
        </span>
        <div className="cmp-cam"><CameraPresets onPreset={(p) => viewerRef.current?.preset(p)} /></div>
      </div>
      <div className="cmp-vp">
        <Viewer ref={viewerRef} {...viewerProps} />
      </div>
      <div className="cmp-stats">
        {stats.map((s, i) => (
          <div key={i} className="cmp-stat">
            <span className="cmp-stat-lbl">{s.label}</span>
            <b className="mono">{s.value}</b>
          </div>
        ))}
      </div>
    </div>
  );
}

export function CompareMode({ cloud, theme, sceneName, gtInstances, predInstances, navMode, onNavModeChange }) {
  const leftRef = useRefCmp();
  const rightRef = useRefCmp();
  const [syncCameras, setSyncCameras] = useStateCmp(true);
  const [diff, setDiff] = useStateCmp(null);
  const [loading, setLoading] = useStateCmp(false);

  useEffectCmp(() => {
    if (!sceneName) return;
    setLoading(true);
    VoxaAPI.compare(sceneName).then((d) => {
      setDiff(d);
      setLoading(false);
    });
  }, [sceneName, gtInstances, predInstances]);

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

  const meanConf = predInstances.length
    ? (predInstances.reduce((a, p) => a + (p.conf || 0), 0) / predInstances.length).toFixed(2)
    : '—';

  return (
    <div className="mode-root compare">
      <div className="cmp-bar">
        <div className="cmp-bar-l">
          <span className="cmp-bar-title">Diff overview</span>
          <span className="cmp-bar-sub mono">{sceneName || '—'} · ground truth vs prediction</span>
        </div>
        <div className="cmp-bar-r">
          <div className="cmp-metric"><label>Precision</label>
            <b className="mono">{diff ? diff.precision.toFixed(3) : '—'}</b></div>
          <div className="cmp-metric"><label>Recall</label>
            <b className="mono">{diff ? diff.recall.toFixed(3) : '—'}</b></div>
          <div className="cmp-metric"><label>F1</label>
            <b className="mono accent">{diff ? diff.f1.toFixed(3) : '—'}</b></div>
          <div className="cmp-metric"><label>IoU<sub>μ</sub></label>
            <b className="mono">{diff ? diff.iou_mean.toFixed(3) : '—'}</b></div>
          <div className="cmp-metric"><label>TP / FP / FN</label>
            <span className="diff-pills">
              <i style={{ background: 'oklch(0.65 0.15 150 / 0.2)', color: 'oklch(0.78 0.15 150)' }}>
                {diff?.tp ?? '—'}</i>
              <i style={{ background: 'oklch(0.65 0.18 30 / 0.2)',  color: 'oklch(0.78 0.16 30)' }}>
                {diff?.fp ?? '—'}</i>
              <i style={{ background: 'oklch(0.65 0.18 60 / 0.2)',  color: 'oklch(0.82 0.14 70)' }}>
                {diff?.fn ?? '—'}</i>
            </span>
          </div>
          <div className="cmp-toggle">
            <label>Nav</label>
            <NavModeToggle navMode={navMode} onChange={onNavModeChange} />
          </div>
          <div className="cmp-toggle">
            <label>Sync cameras</label>
            <button className={'sw' + (syncCameras ? ' on' : '')}
              onClick={() => setSyncCameras(!syncCameras)}><i /></button>
          </div>
        </div>
      </div>

      <div className="cmp-grid">
        <ComparePanel
          title="Ground truth"
          badge="GT"
          badgeColor="#10b981"
          viewerRef={leftRef}
          viewerProps={{
            cloud, instances: gtInstances, showCuboids: true, cuboidStyle: 'solid',
            background: theme.bg, floorColor: theme.floor, navMode,
            onCameraChange: onLeftMove,
          }}
          stats={[
            { label: 'Instances', value: gtInstances.length },
            { label: 'Classes', value: new Set(gtInstances.map((i) => i.cls)).size },
            { label: 'Scene', value: sceneName || '—' },
          ]}
        />
        <ComparePanel
          title="Prediction"
          badge="PRED"
          badgeColor="#5b8def"
          viewerRef={rightRef}
          viewerProps={{
            cloud, instances: predInstances, showCuboids: true, cuboidStyle: 'dashed',
            background: theme.bg, floorColor: theme.floor, navMode,
            onCameraChange: onRightMove,
          }}
          stats={[
            { label: 'Predictions', value: predInstances.length },
            { label: 'Mean conf', value: meanConf },
            { label: 'Status', value: loading ? 'computing…' : 'ready' },
          ]}
        />
      </div>

      <div className="cmp-table">
        <div className="cmp-table-hd">
          <div>Instance</div>
          <div>Class</div>
          <div>Status</div>
          <div>IoU</div>
          <div>Δ position</div>
          <div>Δ size</div>
          <div>Conf</div>
        </div>
        {diff?.rows.map((r, i) => {
          const id = r.gt_id || r.pred_id || `row-${i}`;
          return (
            <div key={id + '-' + i} className="cmp-table-row">
              <div className="mono">{id}</div>
              <div>{r.cls}</div>
              <div><span className={'status-pill ' + r.status.toLowerCase()}>{r.status}</span></div>
              <div className="mono">{r.iou != null ? r.iou.toFixed(3) : '—'}</div>
              <div className="mono">{r.dpos != null ? r.dpos.toFixed(3) : '—'}</div>
              <div className="mono">{r.dsize != null ? `${(r.dsize * 100).toFixed(1)}%` : '—'}</div>
              <div className="mono">{r.conf != null ? r.conf.toFixed(2) : '—'}</div>
            </div>
          );
        })}
        {(!diff || diff.rows.length === 0) && (
          <div style={{ padding: 24, textAlign: 'center', color: 'var(--text-faint)', fontSize: 12 }}>
            {sceneName
              ? 'No GT or prediction annotations for this scene yet.'
              : 'Pick a scene to compute the diff.'}
          </div>
        )}
      </div>
    </div>
  );
}

