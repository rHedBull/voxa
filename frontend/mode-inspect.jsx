// mode-inspect.jsx — full-bleed scrubby viewer for fast scan review.

const { useState: useStateInspect, useMemo: useMemoInspect } = React;

function ViewportToolbar({ children, side = 'left' }) {
  return <div className="vp-toolbar" data-side={side}>{children}</div>;
}

function ToolButton({ icon, label, active, badge, onClick, hotkey, mini }) {
  return (
    <button type="button"
      className={'tool-btn' + (active ? ' active' : '') + (mini ? ' mini' : '')}
      onClick={onClick}
      title={label + (hotkey ? `  (${hotkey})` : '')}>
      <span className="tool-ico" aria-hidden>{icon}</span>
      {!mini && <span className="tool-lbl">{label}</span>}
      {hotkey && !mini && <span className="tool-hk">{hotkey}</span>}
      {badge != null && <span className="tool-badge">{badge}</span>}
    </button>
  );
}

function HUDChip({ label, value, mono, accent }) {
  return (
    <div className={'hud-chip' + (accent ? ' accent' : '')}>
      <span className="hud-lbl">{label}</span>
      <span className={'hud-val' + (mono ? ' mono' : '')}>{value}</span>
    </div>
  );
}

function CameraPresets({ onPreset }) {
  return (
    <div className="cam-presets">
      {['iso', 'top', 'front', 'side'].map((p) => (
        <button key={p} className="cam-btn" onClick={() => onPreset(p)}>{p}</button>
      ))}
    </div>
  );
}

function InspectMode({ cloud, loading, theme, viewerRef, sceneName }) {
  const [pointSize, setPointSize] = useStateInspect(0.012);
  const [colorMode, setColorMode] = useStateInspect('rgb');
  const [showFloor, setShowFloor] = useStateInspect(true);

  // Derived stats from the loaded cloud.
  const stats = useMemoInspect(() => {
    if (!cloud) return null;
    const ext = (cloud.bbox.max[0] - cloud.bbox.min[0]).toFixed(2);
    const dep = (cloud.bbox.max[2] - cloud.bbox.min[2]).toFixed(2);
    const hgt = (cloud.bbox.max[1] - cloud.bbox.min[1]).toFixed(2);
    return { ext, dep, hgt };
  }, [cloud]);

  // Color override for non-RGB modes uses the per-point color tint approach
  // baked into the viewer.
  const overrideColor = colorMode === 'height' ? '#5b8def'
    : colorMode === 'intensity' ? '#a3a3a3'
    : colorMode === 'flat' ? '#7c8088'
    : null;

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
          overrideColor={overrideColor}
        />

        <div className="vp-hud-top">
          <div className="hud-group">
            <HUDChip label="Scene" value={sceneName || '—'} mono />
            <HUDChip label="Points" value={cloud ? `${(cloud.numSubsampled / 1000).toFixed(0)}k / ${(cloud.numPoints / 1000).toFixed(0)}k` : '—'} mono />
            {stats && <HUDChip label="Extent" value={`${stats.ext}×${stats.dep}×${stats.hgt}m`} mono />}
          </div>
          <div className="hud-group">
            <CameraPresets onPreset={(p) => viewerRef.current?.preset(p)} />
          </div>
        </div>

        <ViewportToolbar side="left">
          <ToolButton mini icon="◔" label="Orbit" active hotkey="1" />
          <ToolButton mini icon="✥" label="Pan" hotkey="2" />
          <ToolButton mini icon="⊕" label="Zoom" hotkey="3" />
          <div className="tool-sep" />
          <ToolButton mini icon="↺" label="Reset" hotkey="R"
            onClick={() => viewerRef.current?.preset('iso')} />
        </ViewportToolbar>

        <div className="inspect-right">
          <div className="panel">
            <div className="panel-hd">Scene</div>
            <div className="panel-body">
              <div className="kv"><span>name</span><b>{sceneName || '—'}</b></div>
              {cloud && <>
                <div className="kv"><span>points</span><b className="mono">{cloud.numPoints.toLocaleString()}</b></div>
                <div className="kv"><span>shown</span><b className="mono">{cloud.numSubsampled.toLocaleString()}</b></div>
                <div className="kv"><span>x-min/max</span><b className="mono">{cloud.bbox.min[0].toFixed(2)}/{cloud.bbox.max[0].toFixed(2)}</b></div>
                <div className="kv"><span>y-min/max</span><b className="mono">{cloud.bbox.min[1].toFixed(2)}/{cloud.bbox.max[1].toFixed(2)}</b></div>
                <div className="kv"><span>z-min/max</span><b className="mono">{cloud.bbox.min[2].toFixed(2)}/{cloud.bbox.max[2].toFixed(2)}</b></div>
              </>}
              {loading && <div className="kv"><span>status</span><b>loading…</b></div>}
            </div>
          </div>
          <div className="panel">
            <div className="panel-hd">Display</div>
            <div className="panel-body">
              <div className="ctrl">
                <label>Color by</label>
                <div className="pill-group">
                  {[['rgb','RGB'],['height','Height'],['intensity','Intensity'],['flat','Flat']].map(([k, l]) => (
                    <button key={k}
                      className={'pill' + (colorMode === k ? ' active' : '')}
                      onClick={() => setColorMode(k)}>{l}</button>
                  ))}
                </div>
              </div>
              <div className="ctrl">
                <label>Point size <span className="mono">{pointSize.toFixed(3)}</span></label>
                <input type="range" min={0.002} max={0.05} step={0.001}
                  value={pointSize} className="slider"
                  onChange={(e) => setPointSize(Number(e.target.value))} />
              </div>
              <div className="ctrl row">
                <label>Floor & grid</label>
                <button className={'sw' + (showFloor ? ' on' : '')}
                  onClick={() => setShowFloor(!showFloor)}><i /></button>
              </div>
            </div>
          </div>
        </div>

        <div className="vp-hud-bottom">
          <div className="kbd-strip">
            <span><kbd>Drag</kbd> orbit</span>
            <span><kbd>Shift</kbd>+<kbd>Drag</kbd> pan</span>
            <span><kbd>Scroll</kbd> zoom</span>
            <span><kbd>R</kbd> reset view</span>
          </div>
        </div>
      </div>
    </div>
  );
}

window.InspectMode = InspectMode;
