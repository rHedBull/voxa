// mode-compare.jsx — synced split view: GT (solid) vs prediction (dashed),
// with a server-computed diff table.

import { useState as useStateCmp, useRef as useRefCmp,
         useEffect as useEffectCmp } from 'react';
import * as THREE from 'three';
import { Viewer } from './viewer.jsx';
import { CameraPresets, NavModeToggle, HelpButton } from './viewport-atoms.jsx';
import { VoxaAPI } from './api.js';

function ComparePanel({ title, badge, badgeColor, viewerProps, viewerRef }) {
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
    </div>
  );
}

export function CompareMode({ cloud, theme, sceneName, gtInstances, predInstances, navMode, onNavModeChange }) {
  const leftRef = useRefCmp();
  const rightRef = useRefCmp();
  const [syncCameras, setSyncCameras] = useStateCmp(true);
  const [diff, setDiff] = useStateCmp(null);
  const [selectedRowKey, setSelectedRowKey] = useStateCmp(null);

  useEffectCmp(() => {
    if (!sceneName) return;
    VoxaAPI.compare(sceneName).then(setDiff);
  }, [sceneName, gtInstances, predInstances]);

  // Clear selection when the underlying diff changes (scene swap, new prediction
  // load) — the old row key may not exist in the new rows.
  useEffectCmp(() => { setSelectedRowKey(null); }, [sceneName]);

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

  // Selected-row → which instance ids are visible per side. Empty list means
  // hide all cuboids on that side. By design, no row selected → no boxes
  // anywhere; users opt-in via double-click.
  const rowKey = (r, i) => (r.gt_id || r.pred_id || `row-${i}`) + '-' + i;
  const selectedRow = selectedRowKey && diff?.rows
    ? diff.rows.find((r, i) => rowKey(r, i) === selectedRowKey)
    : null;
  const gtVisible = selectedRow?.gt_id ? [selectedRow.gt_id] : [];
  const predVisible = selectedRow?.pred_id ? [selectedRow.pred_id] : [];

  // Frame the cameras on a row's box(es). Selects the row first so the box is
  // visible even if the user clicked Focus without double-clicking the row.
  const focusRow = (r, i) => {
    setSelectedRowKey(rowKey(r, i));
    const frame = (ref, inst) => {
      if (!inst) return;
      ref.current?.frame(
        new THREE.Vector3(...inst.center),
        Math.sqrt(inst.size.reduce((a, v) => a + v * v, 0)) / 2,
      );
    };
    frame(leftRef,  r.gt_id   ? gtInstances.find((x) => x.id === r.gt_id)   : null);
    frame(rightRef, r.pred_id ? predInstances.find((x) => x.id === r.pred_id) : null);
  };

  const helpSections = [
    {
      title: 'Compare',
      items: [
        { keys: ['Dbl-click row'], desc: 'Select row → show its box(es); again to deselect' },
        { keys: ['◎'], desc: 'Per-row button: select + frame camera on that box' },
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
      title: 'Diff metrics',
      items: [
        { keys: ['IoU'], desc: 'Axis-aligned overlap (rotation ignored)' },
        { keys: ['F1'], desc: 'Harmonic mean of precision & recall' },
        { keys: ['TP/FP/FN'], desc: 'Matched / spurious / missed instances' },
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

  return (
    <div className="mode-root compare">
      <div className="cmp-bar">
        <div className="cmp-bar-l">
          <span className="cmp-bar-title">Diff overview</span>
          <span className="cmp-bar-sub mono">{sceneName || '—'} · ground truth vs prediction</span>
        </div>
        <div className="cmp-bar-r">
          <div className="cmp-metric"><label>Precision</label>
            <b className="mono">{diff?.precision != null ? diff.precision.toFixed(3) : '—'}</b></div>
          <div className="cmp-metric"><label>Recall</label>
            <b className="mono">{diff?.recall != null ? diff.recall.toFixed(3) : '—'}</b></div>
          <div className="cmp-metric"><label>F1</label>
            <b className="mono accent">{diff?.f1 != null ? diff.f1.toFixed(3) : '—'}</b></div>
          <div className="cmp-metric"><label>IoU<sub>μ</sub></label>
            <b className="mono">{diff?.iou_mean != null ? diff.iou_mean.toFixed(3) : '—'}</b></div>
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
          <div className="cmp-toggle">
            <HelpButton sections={helpSections} />
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
            cloud, instances: gtInstances, visibleInstanceIds: gtVisible,
            showCuboids: true, cuboidStyle: 'solid',
            background: theme.bg, floorColor: theme.floor, navMode,
            onCameraChange: onLeftMove,
          }}
        />
        <ComparePanel
          title="Prediction"
          badge="PRED"
          badgeColor="#5b8def"
          viewerRef={rightRef}
          viewerProps={{
            cloud, instances: predInstances, visibleInstanceIds: predVisible,
            showCuboids: true, cuboidStyle: 'dashed',
            background: theme.bg, floorColor: theme.floor, navMode,
            onCameraChange: onRightMove,
          }}
        />
      </div>

      <div className="cmp-table">
        <div className="cmp-table-hd">
          <div>GT id</div>
          <div>Pred id</div>
          <div>Class</div>
          <div>Status</div>
          <div>IoU</div>
          <div>Δ position</div>
          <div>Δ size</div>
          <div>Conf</div>
          <div />
        </div>
        {diff?.rows?.map((r, i) => {
          const key = rowKey(r, i);
          const sel = key === selectedRowKey;
          return (
            <div
              key={key}
              className={'cmp-table-row' + (sel ? ' selected' : '')}
              onDoubleClick={() => setSelectedRowKey(sel ? null : key)}
              title={sel ? 'Double-click to deselect' : 'Double-click to show this row\'s box'}>
              <div className="mono">{r.gt_id ?? '—'}</div>
              <div className="mono">{r.pred_id ?? '—'}</div>
              <div>{r.cls}</div>
              <div><span className={'status-pill ' + r.status.toLowerCase()}>{r.status}</span></div>
              <div className="mono">{r.iou != null ? r.iou.toFixed(3) : '—'}</div>
              <div className="mono">{r.dpos != null ? r.dpos.toFixed(3) : '—'}</div>
              <div className="mono">{r.dsize != null ? `${(r.dsize * 100).toFixed(1)}%` : '—'}</div>
              <div className="mono">{r.conf != null ? r.conf.toFixed(2) : '—'}</div>
              <button
                className="inst-edit-btn"
                onClick={(e) => { e.stopPropagation(); focusRow(r, i); }}
                title="Select & frame camera on this box">◎</button>
            </div>
          );
        })}
        {(!diff?.rows || diff.rows.length === 0) && (
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

