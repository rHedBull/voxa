// App.jsx — Voxa shell. Owns the scene/cloud/annotations/config state and
// dispatches to mode components.

import { useState as useStateApp, useRef as useRefApp,
         useEffect as useEffectApp, useCallback as useCallbackApp } from 'react';
import { VoxaAPI } from './api.js';
import { initSegState } from './segment-state.js';
import { InspectMode } from './mode-inspect.jsx';
import { LabelMode } from './mode-label.jsx';
import { CompareMode } from './mode-compare.jsx';
import { useTweaks, TweaksPanel, TweakSection, TweakRadio } from './tweaks-panel.jsx';

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "dark",
  "mode": "inspect"
}/*EDITMODE-END*/;

const MODE_META = {
  inspect: {
    label: 'Inspect',
    sub: 'Lightweight scrubby viewer for fast scan review',
    color: 'oklch(0.72 0.14 250)',
  },
  label: {
    label: 'Label',
    sub: 'Cuboid annotation with auto-fit assistance',
    color: 'oklch(0.74 0.15 150)',
  },
  compare: {
    label: 'Compare',
    sub: 'Side-by-side ground truth vs prediction diff',
    color: 'oklch(0.75 0.16 60)',
  },
};

export default function App() {
  const [t, setTweak] = useTweaks(TWEAK_DEFAULTS);
  const viewerRef = useRefApp();

  const [scenes, setScenes] = useStateApp([]);
  const [activeScene, setActiveScene] = useStateApp(null);
  const [cloud, setCloud] = useStateApp(null);
  const [loading, setLoading] = useStateApp(false);
  const [loadError, setLoadError] = useStateApp(null);
  const [classes, setClasses] = useStateApp([]);
  const [gtInstances, setGtInstances] = useStateApp([]);
  const [predInstances, setPredInstances] = useStateApp([]);
  const [savedAt, setSavedAt] = useStateApp(null);
  const [scenePickerOpen, setScenePickerOpen] = useStateApp(false);
  const [segState, setSegState] = useStateApp(null);
  // Camera nav mode is shared across the three modes so toggling Inspect →
  // Label preserves whether the user was orbiting or walking.
  const [navMode, setNavMode] = useStateApp('orbit');

  // Initial config + scene list.
  useEffectApp(() => {
    VoxaAPI.config().then((c) => setClasses(c.classes || []));
    VoxaAPI.scenes().then((s) => {
      setScenes(s);
      if (s.length && !activeScene) setActiveScene(s[0].id || s[0].name);
    });
    // eslint-disable-next-line
  }, []);

  // Load cloud + annotations whenever the active scene or mode changes.
  useEffectApp(() => {
    if (!activeScene) return;
    let cancel = false;
    setLoading(true);
    setLoadError(null);
    const activeSceneObj = scenes.find((s) => (s.id || s.name) === activeScene);
    const wantFullLabels = t.mode === 'label' && activeSceneObj?.tier === 'annotated';
    VoxaAPI.load(activeScene, { wantFullLabels })
      .then((c) => {
        if (cancel) return;
        setCloud(c);
        if (c.fullClassIds && c.fullInstanceIds) {
          setSegState(initSegState({
            classFull: c.fullClassIds,
            instanceFull: c.fullInstanceIds,
            isFromPrelabel: c.isFromPrelabel,
          }));
        } else {
          setSegState(null);
        }
        setLoading(false);
      })
      .catch((e) => {
        if (cancel) return;
        setLoadError(String(e.message || e));
        setLoading(false);
      });
    VoxaAPI.getAnnotation(activeScene, 'gt')
      .then((d) => !cancel && setGtInstances(d.instances || []));
    VoxaAPI.getAnnotation(activeScene, 'pred')
      .then((d) => !cancel && setPredInstances(d.instances || []));
    return () => { cancel = true; };
  }, [activeScene, t.mode]); // eslint-disable-line react-hooks/exhaustive-deps

  const theme = t.theme === 'dark'
    ? { bg: '#0a0b0e', floor: '#15171c' }
    : { bg: '#ebedf1', floor: '#d9dde3' };
  const themeClass = t.theme === 'dark' ? 'theme-dark' : 'theme-light';

  useEffectApp(() => { document.body.className = themeClass; }, [themeClass]);

  const meta = MODE_META[t.mode] || MODE_META.inspect;

  const saveGt = useCallbackApp(async (instances) => {
    if (!activeScene) return;
    setGtInstances(instances);
    await VoxaAPI.putAnnotation(activeScene, 'gt', { instances });
    setSavedAt(new Date().toLocaleTimeString());
  }, [activeScene]);

  // Cmd/Ctrl+S → save
  useEffectApp(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault();
        saveGt(gtInstances);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [gtInstances, saveGt]);

  return (
    <div className={'app-shell ' + themeClass}>
      <header className="app-header">
        <div className="app-brand">
          <div className="logo">⊞</div>
          <span>Voxa</span>
          <span className="brand-sub">3D scan studio</span>
        </div>

        <div className="mode-switcher">
          {Object.entries(MODE_META).map(([k, m]) => (
            <button key={k}
              className={t.mode === k ? 'active' : ''}
              onClick={() => setTweak('mode', k)}>
              <span className="mode-dot" style={{ background: m.color }} />
              {m.label}
            </button>
          ))}
        </div>

        <div className="header-spacer" />

        <div className="header-meta">
          {loading
            ? <><span className="dot" style={{ background: 'oklch(0.75 0.15 60)' }} /> Loading…</>
            : loadError
              ? <><span className="dot" style={{ background: 'oklch(0.65 0.18 25)' }} /> {loadError}</>
              : cloud
                ? <><span className="dot" /> {cloud.numSubsampled.toLocaleString()} pts loaded</>
                : <><span className="dot" style={{ background: 'oklch(0.6 0.02 250)' }} /> No scene</>
          }
        </div>
        <button className="header-btn" onClick={() => setScenePickerOpen((o) => !o)}>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}>◧</span>
          {activeScene || 'Pick scene'}
        </button>
        <button className="header-btn icon-only"
          onClick={() => setTweak('theme', t.theme === 'dark' ? 'light' : 'dark')}
          title={t.theme === 'dark' ? 'Switch to bright mode' : 'Switch to dark mode'}>
          {t.theme === 'dark' ? '☀' : '☾'}
        </button>
        {savedAt && <span className="header-meta" style={{ fontSize: 11 }}>saved {savedAt}</span>}
        <button className="header-btn primary" onClick={() => saveGt(gtInstances)} title="Save (⌘S)">
          ⌘S Save
        </button>
      </header>

      <div className="mode-banner">
        <div className="stripe" style={{ background: meta.color }} />
        <b>{meta.label} mode</b>
        <span>·</span>
        <span>{meta.sub}</span>
      </div>

      {scenePickerOpen && (
        <ScenePicker
          scenes={scenes}
          activeScene={activeScene}
          onPick={(name) => { setActiveScene(name); setScenePickerOpen(false); }}
          onClose={() => setScenePickerOpen(false)}
        />
      )}

      <div className="app-body">
        {t.mode === 'inspect' && (
          <InspectMode key="i" cloud={cloud} loading={loading} theme={theme}
            viewerRef={viewerRef} sceneName={activeScene}
            navMode={navMode} onNavModeChange={setNavMode} />
        )}
        {t.mode === 'label' && (
          <LabelMode key="l" cloud={cloud} theme={theme} viewerRef={viewerRef}
            classes={classes} instances={gtInstances} sceneName={activeScene}
            cloudBBox={cloud?.bbox}
            navMode={navMode} onNavModeChange={setNavMode}
            onChange={setGtInstances} onSave={saveGt}
            segState={segState} setSegState={setSegState} />
        )}
        {t.mode === 'compare' && (
          <CompareMode key="c" cloud={cloud} theme={theme}
            sceneName={activeScene}
            navMode={navMode} onNavModeChange={setNavMode}
            gtInstances={gtInstances} predInstances={predInstances} />
        )}
      </div>

      <TweaksPanel title="Tweaks">
        <TweakSection label="Mode" />
        <TweakRadio label="Active mode" value={t.mode}
          options={[
            { value: 'inspect', label: 'Inspect' },
            { value: 'label',   label: 'Label' },
            { value: 'compare', label: 'Compare' },
          ]}
          onChange={(v) => setTweak('mode', v)} />
        <TweakSection label="Appearance" />
        <TweakRadio label="Theme" value={t.theme}
          options={[
            { value: 'dark',  label: 'Dark' },
            { value: 'light', label: 'Light' },
          ]}
          onChange={(v) => setTweak('theme', v)} />
      </TweaksPanel>
    </div>
  );
}

const TIER_LABEL = {
  legacy:    'Legacy (voxa/data/scenes)',
  annotated: 'Annotated (lidar/annotated)',
  decimated: 'Decimated (lidar/ply_viewer)',
  raw:       'Raw LAZ (lidar/laz)',
};
const TIER_DOT = {
  legacy:    '#6b7280',
  annotated: '#10b981',
  decimated: '#5b8def',
  raw:       '#f5a524',
};

function ScenePicker({ scenes, activeScene, onPick, onClose }) {
  // Group by tier (preserve the order returned by the backend).
  const groups = [];
  const seen = new Map();
  for (const s of scenes) {
    const tier = s.tier || 'legacy';
    if (!seen.has(tier)) {
      const g = { tier, items: [] };
      seen.set(tier, g);
      groups.push(g);
    }
    seen.get(tier).items.push(s);
  }

  return (
    <div className="scene-picker" onClick={onClose}>
      <div className="scene-picker-card" onClick={(e) => e.stopPropagation()}>
        <div className="side-hd">
          <span>Scenes</span>
          <span className="badge-soft">{scenes.length}</span>
        </div>
        {scenes.length === 0 && (
          <div className="sugg-empty">
            No scenes found. Drop a folder under <span className="mono">data/scenes/&lt;name&gt;/source.ply</span>,
            or set <span className="mono">VOXA_LIDAR_ROOT</span> to your lidar archive.
          </div>
        )}
        {groups.map((g) => (
          <div key={g.tier} className="scene-group">
            {groups.length > 1 && (
              <div className="scene-group-hd">
                <span className="inst-dot" style={{ background: TIER_DOT[g.tier] || '#6b7280' }} />
                <span>{TIER_LABEL[g.tier] || g.tier}</span>
                <span className="badge-soft">{g.items.length}</span>
              </div>
            )}
            {g.items.map((s) => {
              const id = s.id || s.name;
              const fmt = s.source_format || s.source_type;
              return (
                <div key={id}
                     className={'inst-row' + (id === activeScene ? ' selected' : '')}
                     onClick={() => onPick(id)}>
                  <span className="inst-dot" style={{ background: TIER_DOT[g.tier] || '#5b8def' }} />
                  <div className="inst-text">
                    <b>{s.name}</b>
                    <em>
                      {fmt ? `.${fmt}` : 'no source'}
                      {s.n_points ? ` · ${(s.n_points / 1000).toFixed(0)}k pts` : ''}
                      {s.has_labels ? ' · labels' : ''}
                      {s.has_intensity ? ' · intensity' : ''}
                      {s.has_ground_truth ? ' · GT' : ''}
                      {s.has_predictions ? ' · pred' : ''}
                    </em>
                  </div>
                </div>
              );
            })}
          </div>
        ))}
      </div>
    </div>
  );
}

