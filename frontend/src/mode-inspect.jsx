// mode-inspect.jsx — full-bleed scrubby viewer for fast scan review.

import { useEffect as useEffectInspect,
         useState as useStateInspect,
         useMemo as useMemoInspect } from 'react';
import { Viewer } from './viewer.jsx';
import { ViewportToolbar, ToolButton, HUDChip, CameraPresets, NavModeToggle, HelpButton } from './viewport-atoms.jsx';

// Compact point-count label: ≥1M → "X.X M", otherwise "Nk".
function fmtCount(n) {
  return n >= 1e6 ? `${(n / 1e6).toFixed(1)}M` : `${(n / 1e3).toFixed(0)}k`;
}

// A right-rail panel whose body collapses when its header is clicked.
function CollapsiblePanel({ title, defaultOpen = false, children }) {
  const [open, setOpen] = useStateInspect(defaultOpen);
  return (
    <div className={'panel' + (open ? '' : ' collapsed')}>
      <button className="panel-hd panel-hd-btn" aria-expanded={open}
        onClick={() => setOpen((o) => !o)}>
        <span>{title}</span>
        <span className="panel-chev">{open ? '▾' : '▸'}</span>
      </button>
      {open && <div className="panel-body">{children}</div>}
    </div>
  );
}

export function InspectMode({ cloud, loading, theme, viewerRef, sceneName, navMode, onNavModeChange, onCameraChange }) {
  const [pointSize, setPointSize] = useStateInspect(0.012);
  const [colorMode, setColorMode] = useStateInspect('rgb');
  const [showFloor, setShowFloor] = useStateInspect(true);
  const [showMesh, setShowMesh] = useStateInspect(false);
  const [meshBrightness, setMeshBrightness] = useStateInspect(1.0);
  const [meshProgress, setMeshProgress] = useStateInspect(null);

  // Reset mesh toggle when switching to a scene without a mesh, so the
  // user doesn't carry a stale "on" state into a scene that can't honor it.
  useEffectInspect(() => {
    if (cloud && !cloud.meshUrl && showMesh) setShowMesh(false);
    if (!cloud) setMeshProgress(null);
  }, [cloud, showMesh]);

  // Derived stats from the loaded cloud.
  const stats = useMemoInspect(() => {
    if (!cloud) return null;
    const extN = cloud.bbox.max[0] - cloud.bbox.min[0];
    const depN = cloud.bbox.max[2] - cloud.bbox.min[2];
    const hgtN = cloud.bbox.max[1] - cloud.bbox.min[1];
    return {
      ext: extN.toFixed(2), dep: depN.toFixed(2), hgt: hgtN.toFixed(2),
      area: (extN * depN).toFixed(0),
    };
  }, [cloud]);

  const channels = useMemoInspect(() => ({
    rgb: !!cloud,
    height: !!cloud,
    class: !!cloud && !!cloud.classIds && !!cloud.classPalette,
    instance: !!cloud && !!cloud.instanceIds,
  }), [cloud]);

  // Auto-fall-back if the previously-selected color mode is no longer
  // available on this scene (e.g. we switched from a labeled scene to a
  // raw-LAZ one and the user had Class active).
  useEffectInspect(() => {
    if (cloud && !channels[colorMode]) setColorMode('rgb');
  }, [cloud, channels, colorMode]);

  const helpSections = useMemoInspect(() => ([
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
      title: 'Camera presets',
      items: [
        { keys: ['iso'], desc: 'Isometric' },
        { keys: ['top'], desc: 'Top-down' },
        { keys: ['front'], desc: 'Front' },
        { keys: ['side'], desc: 'Side' },
      ],
    },
    {
      title: 'Display',
      items: [
        { keys: ['RGB'], desc: 'Per-point color from the source' },
        { keys: ['Height'], desc: 'Y-axis gradient' },
        { keys: ['Class'], desc: 'Per-point class palette (annotated scans)' },
        { keys: ['Instance'], desc: 'Per-point instance hash hue' },
      ],
    },
    {
      title: 'Other',
      items: [
        { keys: ['?'], desc: 'Toggle this panel' },
        { keys: ['Esc'], desc: 'Close panel' },
      ],
    },
  ]), [navMode]);

  return (
    <div className="mode-root inspect">
      <div className="vp-stack">
        <Viewer
          ref={viewerRef}
          cloud={cloud}
          instances={[]}
          showCuboids={false}
          pointSize={pointSize}
          showFloor={showFloor}
          background={theme.bg}
          floorColor={theme.floor}
          colorMode={colorMode}
          navMode={navMode}
          meshUrl={cloud?.meshUrl || null}
          meshIsZUp={!!cloud?.meshIsZUp}
          showMesh={showMesh}
          meshBrightness={meshBrightness}
          onMeshLoadProgress={setMeshProgress}
          onCameraChange={onCameraChange}
        />

        <div className="vp-hud-top">
          <div className="hud-group">
            <HUDChip label="Scene" value={sceneName || '—'} mono />
            <HUDChip label="Points" value={cloud ? `${fmtCount(cloud.numSubsampled)} / ${fmtCount(cloud.numPointsTotal ?? cloud.numPoints)}` : '—'} mono />
            {stats && <HUDChip label="Extent" value={`${stats.ext}×${stats.dep}×${stats.hgt}m · ${stats.area} sqm`} mono />}
          </div>
          <div className="hud-group">
            <NavModeToggle navMode={navMode} onChange={onNavModeChange} />
            <CameraPresets onPreset={(p) => viewerRef.current?.preset(p)} />
          </div>
        </div>

        <div className="vp-help-corner">
          <HelpButton sections={helpSections} placement="up" />
        </div>

        <ViewportToolbar side="left">
          <ToolButton mini icon="↺" label="Reset" hotkey="R"
            onClick={() => viewerRef.current?.preset('iso')} />
        </ViewportToolbar>

        <div className="inspect-right">
          <CollapsiblePanel title="Scene">
              <div className="kv"><span>name</span><b>{sceneName || '—'}</b></div>
              {cloud && <>
                <div className="kv"><span>points</span><b className="mono">{(cloud.numPointsTotal ?? cloud.numPoints).toLocaleString()}{cloud.numPointsTotal ? ` (loaded ${cloud.numPoints.toLocaleString()})` : ''}</b></div>
                <div className="kv"><span>shown</span><b className="mono">{cloud.numSubsampled.toLocaleString()}</b></div>
                <div className="kv"><span>x-min/max</span><b className="mono">{cloud.bbox.min[0].toFixed(2)}/{cloud.bbox.max[0].toFixed(2)}</b></div>
                <div className="kv"><span>y-min/max</span><b className="mono">{cloud.bbox.min[1].toFixed(2)}/{cloud.bbox.max[1].toFixed(2)}</b></div>
                <div className="kv"><span>z-min/max</span><b className="mono">{cloud.bbox.min[2].toFixed(2)}/{cloud.bbox.max[2].toFixed(2)}</b></div>
                {cloud.nInstances != null && (
                  <div className="kv"><span>segments</span>
                    <b className="mono">
                      {cloud.nInstances} · {(cloud.nLabeledPoints || 0).toLocaleString()} / {cloud.numPoints.toLocaleString()} labeled
                    </b>
                  </div>
                )}
              </>}
              {loading && <div className="kv"><span>status</span><b>loading…</b></div>}
          </CollapsiblePanel>
          <CollapsiblePanel title="Display">
              <div className="ctrl">
                <label>Color by</label>
                <div className="pill-group">
                  {[
                    ['rgb','RGB'],
                    ['height','Height'],
                    ['class','Class'],
                    ['instance','Instance'],
                  ].map(([k, l]) => {
                    const enabled = !!channels[k];
                    return (
                      <button key={k}
                        className={'pill' + (colorMode === k ? ' active' : '') + (enabled ? '' : ' disabled')}
                        disabled={!enabled}
                        title={enabled ? '' : `not available — scene has no ${k} channel`}
                        onClick={() => enabled && setColorMode(k)}>{l}</button>
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
              <div className="ctrl row">
                <label>
                  Mesh{cloud?.meshUrl ? '' : ' (none)'}
                  {meshProgress && meshProgress.total > 0 && meshProgress.loaded < meshProgress.total && (
                    <span className="mono dim">
                      {' '}{((meshProgress.loaded / meshProgress.total) * 100).toFixed(0)}%
                    </span>
                  )}
                </label>
                <button className={'sw' + (showMesh ? ' on' : '') + (cloud?.meshUrl ? '' : ' disabled')}
                  disabled={!cloud?.meshUrl}
                  title={cloud?.meshUrl ? '' : 'No mesh.glb for this scene'}
                  onClick={() => cloud?.meshUrl && setShowMesh(!showMesh)}><i /></button>
              </div>
              {showMesh && (
                <div className="ctrl row">
                  <label>Brightness <span className="mono dim">{meshBrightness.toFixed(2)}×</span></label>
                  <input type="range" min={0} max={2} step={0.05}
                    value={meshBrightness}
                    onChange={(e) => setMeshBrightness(parseFloat(e.target.value))} />
                </div>
              )}
          </CollapsiblePanel>
        </div>
      </div>
    </div>
  );
}
