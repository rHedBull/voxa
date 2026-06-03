// App.jsx — Voxa shell. Owns the scene/cloud/annotations/config state and
// dispatches to mode components.

import { useState as useStateApp, useRef as useRefApp,
         useEffect as useEffectApp, useCallback as useCallbackApp } from 'react';
import { VoxaAPI, getSegmentState } from './api.js';
import { initSegState, applyDelta, hydrateFromServerState } from './segment-state.js';
import { InspectMode } from './mode-inspect.jsx';
import { LabelMode } from './mode-label.jsx';
import { CompareMode } from './mode-compare.jsx';
import { EditMode } from './mode-edit.jsx';
import { useTweaks, TweaksPanel, TweakSection, TweakRadio } from './tweaks-panel.jsx';
import { MeshCompanion } from './mesh-companion.jsx';
import { openChannel, postState, postCamera, isMeshCompanion } from './mesh-sync.js';

const TWEAK_DEFAULTS = /*EDITMODE-BEGIN*/{
  "theme": "dark",
  "mode": "inspect"
}/*EDITMODE-END*/;

// Pull persisted overrides at module-load so the first paint already shows
// the right mode — avoids a flash from default → restored. Only `mode` is
// browser-persisted; `theme` lives in the on-disk EDITMODE block.
const INITIAL_TWEAKS = (() => {
  try {
    const m = localStorage.getItem('voxa.mode');
    if (m && ['inspect', 'label', 'compare', 'edit'].includes(m)) {
      return { ...TWEAK_DEFAULTS, mode: m };
    }
  } catch { /* private mode / no localStorage */ }
  return TWEAK_DEFAULTS;
})();

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
  edit: {
    label: 'Edit',
    sub: 'Per-point editing surface',
    color: 'oklch(0.72 0.16 320)',
  },
};

export default function App() {
  // The mesh companion is the same bundle, served at `/?mesh=1`. Branching
  // here keeps it cheap (no separate Vite entry) and lets the companion reuse
  // the existing Viewer + CSS.
  if (isMeshCompanion()) return <MeshCompanion />;
  return <MainApp />;
}

function MainApp() {
  const [t, setTweak] = useTweaks(INITIAL_TWEAKS);
  const viewerRef = useRefApp();
  const prelabelRef = useRefApp({ classFull: null, instanceFull: null });

  const [scenes, setScenes] = useStateApp([]);
  // Lazy-init from localStorage so refresh keeps you on the scene you were
  // working on. Validated against the scenes list once it loads.
  const [activeScene, setActiveScene] = useStateApp(() => {
    try { return localStorage.getItem('voxa.activeScene') || null; } catch { return null; }
  });
  const [cloud, setCloud] = useStateApp(null);
  const [loading, setLoading] = useStateApp(false);
  const [loadError, setLoadError] = useStateApp(null);
  const [classes, setClasses] = useStateApp([]);
  const [gtInstances, setGtInstances] = useStateApp([]);
  const [predInstances, setPredInstances] = useStateApp([]);
  const [savedAt, setSavedAt] = useStateApp(null);
  const [cuboidDirty, setCuboidDirty] = useStateApp(false);
  const [scenePickerOpen, setScenePickerOpen] = useStateApp(false);
  const [segState, setSegState] = useStateApp(null);
  // Multi-session (scan-schema v2): the load response carries the available
  // sessions + which one resumed. `explicitSessionRef` makes /api/load send a
  // session_id ONLY when the user explicitly picked one — otherwise the
  // backend resumes the last-worked session for the scan.
  const [sessions, setSessions] = useStateApp([]);
  const [activeSessionId, setActiveSessionId] = useStateApp(null);
  const [presegs, setPresegs] = useStateApp([]);
  const [pinError, setPinError] = useStateApp(null);
  const explicitSessionRef = useRefApp(null);
  // Scene the load effect last ran for. The effect keys on activeScene, t.mode,
  // and activeSessionId, but activeSessionId is also set BY the effect (to the
  // resumed id) and reset to null on scene change — both would re-fire it
  // pointlessly. So we only honour an activeSessionId change when it came from
  // an explicit pick (ref set); otherwise a re-run with the same scene+mode and
  // no pick is a self-echo and is skipped.
  const loadedSceneRef = useRefApp(null);
  const loadedModeRef = useRefApp(null);
  // Camera nav mode is shared across the three modes so toggling Inspect →
  // Label preserves whether the user was orbiting or walking.
  const [navMode, setNavMode] = useStateApp('orbit');

  // Initial config + scene list. If a persisted scene id no longer exists in
  // the list, fall back to the first scene rather than firing a 404 load.
  useEffectApp(() => {
    VoxaAPI.config().then((c) => setClasses(c.classes || []));
    VoxaAPI.scenes().then((s) => {
      setScenes(s);
      setActiveScene((cur) => {
        if (cur && s.some((x) => (x.id || x.name) === cur)) return cur;
        return s[0]?.id || s[0]?.name || null;
      });
    });
    // eslint-disable-next-line
  }, []);

  // Persist active scene across refreshes.
  useEffectApp(() => {
    if (!activeScene) return;
    try { localStorage.setItem('voxa.activeScene', activeScene); } catch { /* quota / private mode */ }
  }, [activeScene]);

  // Each scan resumes its OWN last-worked session, so clear the explicit pick
  // and reset the resumed id whenever the scene changes. Also (re)fetch the
  // preseg list for annotated scenes — the create form needs it. Real preseg
  // errors surface in the console + the picker's inline message; non-annotated
  // scenes simply have no presegs and skip the fetch.
  useEffectApp(() => {
    explicitSessionRef.current = null;
    setActiveSessionId(null);
    setSessions([]);
    setPresegs([]);
    if (!activeScene) return;
    const sceneObj = scenes.find((s) => (s.id || s.name) === activeScene);
    if (sceneObj?.tier !== 'annotated') return;
    let cancel = false;
    VoxaAPI.listPresegs(activeScene)
      .then((ps) => { if (!cancel) setPresegs(ps || []); })
      .catch((e) => { if (!cancel) console.error('listPresegs failed:', e); });
    return () => { cancel = true; };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeScene]);

  // Persist active mode across refreshes (paired with INITIAL_TWEAKS lazy-init).
  useEffectApp(() => {
    try { localStorage.setItem('voxa.mode', t.mode); } catch { /* quota / private mode */ }
  }, [t.mode]);

  // ── Mesh companion sync ─────────────────────────────────────────────────
  // BroadcastChannel link to any open `?mesh=1` window. Main publishes the
  // scene-shape state (mesh URL, cuboids, theme) and the live camera; companion
  // publishes only its camera back. The viewer's setFromState already silences
  // its onChange while applying, so the loop terminates after one hop.
  const meshChannelRef = useRefApp(null);
  const themeBg = t.theme === 'light' ? '#f5f5f5' : '#0a0b0e';
  const themeFloor = t.theme === 'light' ? '#e7e8ec' : '#15171c';
  useEffectApp(() => {
    const ch = openChannel();
    meshChannelRef.current = ch;
    if (!ch) return;
    ch.onmessage = (ev) => {
      const m = ev.data;
      if (!m || typeof m !== 'object') return;
      if (m.type === 'camera' && m.camera) {
        viewerRef.current?.setCameraState(m.camera);
      } else if (m.type === 'request-state') {
        // Companion just opened — re-broadcast latest. We rely on the next
        // effect's broadcast to fire on the next render cycle, so simulate
        // an explicit broadcast here too.
        meshBroadcastRef.current?.();
      }
    };
    return () => { ch.close(); meshChannelRef.current = null; };
    // eslint-disable-next-line
  }, []);

  // Latest broadcast closure, kept in a ref so the channel-open effect (which
  // runs once) can call the freshest version on demand.
  const meshBroadcastRef = useRefApp(null);
  useEffectApp(() => {
    meshBroadcastRef.current = () => {
      const ch = meshChannelRef.current;
      if (!ch || !cloud) return;
      postState(ch, {
        scene: activeScene,
        meshUrl: cloud.meshUrl ? new URL(cloud.meshUrl, window.location.origin).toString() : null,
        meshIsZUp: !!cloud.meshIsZUp,
        meshOffset: cloud.recenterOffset || null,
        instances: gtInstances,
        bbox: cloud.bbox || null,
        background: themeBg,
        floorColor: themeFloor,
      });
    };
    meshBroadcastRef.current();
  }, [cloud, gtInstances, activeScene, themeBg, themeFloor]);

  const onMainCameraChange = useCallbackApp((cam) => {
    postCamera(meshChannelRef.current, cam);
  }, []);

  // Load cloud + annotations whenever the active scene or mode changes.
  useEffectApp(() => {
    if (!activeScene) return;
    // Skip self-echoes: a re-run for the same scene+mode with no explicit pick
    // pending was triggered by our own setActiveSessionId (resume / scene-reset
    // to null), not by user intent. Re-fetching the cloud then is wasteful.
    if (!explicitSessionRef.current
        && activeScene === loadedSceneRef.current
        && t.mode === loadedModeRef.current) {
      return;
    }
    loadedSceneRef.current = activeScene;
    loadedModeRef.current = t.mode;
    let cancel = false;
    setLoading(true);
    setLoadError(null);
    setPinError(null);
    const activeSceneObj = scenes.find((s) => (s.id || s.name) === activeScene);
    const wantFullLabels = t.mode === 'label' && activeSceneObj?.tier === 'annotated';
    // Only send a session_id when the user explicitly picked one; clear the
    // ref immediately so a subsequent scene/mode change defaults to last-worked.
    const explicitSessionId = explicitSessionRef.current;
    explicitSessionRef.current = null;
    // Run /api/load first, then fetch /api/segment/state. They MUST be
    // sequential: /api/load is what swaps the backend's in-memory seg
    // session over to the new scene; if segState() races ahead it can
    // come back with the previous scene's data (e.g. 482k smart_ais
    // segments hydrated onto a 16k industrial_scan cloud).
    Promise.all([
      VoxaAPI.load(activeScene, { wantFullLabels, sessionId: explicitSessionId }),
      VoxaAPI.getAnnotation(activeScene, 'gt'),
    ]).then(async ([c, gtDoc]) => {
      const segLive = await VoxaAPI.segState().catch(() => null);
      return [c, gtDoc, segLive];
    }).then(([c, gtDoc, segLive]) => {
      if (cancel) return;
      setCloud(c);
      setSessions(c.sessions);
      setActiveSessionId(c.sessionId);
      setCuboidDirty(false);
      if (segLive) {
        // Live seg session wins — includes hulls + any unsaved preseg edits.
        prelabelRef.current = c.isFromPrelabel
          ? { classFull: segLive.fullClassIds.slice(), instanceFull: segLive.fullInstanceIds.slice() }
          : { classFull: null, instanceFull: null };
        setSegState(initSegState({
          classFull: segLive.fullClassIds,
          instanceFull: segLive.fullInstanceIds,
          isFromPrelabel: !!c.isFromPrelabel,
          segBoxes: (segLive.segIds && segLive.segCenters && segLive.segSizes)
            ? { segIds: segLive.segIds, segCenters: segLive.segCenters, segSizes: segLive.segSizes }
            : null,
          segHulls: (segLive.hullVertices && segLive.hullFaces && segLive.hullFaceSeg)
            ? { vertices: segLive.hullVertices, faces: segLive.hullFaces, faceSeg: segLive.hullFaceSeg }
            : null,
        }));
        // Project full-res labels onto the subsampled cloud so points pick
        // up segment colours immediately.
        const subIdx = c.subsampleIdx;
        const subN = (c.positions?.length || 0) / 3;
        const subClass = new Int8Array(subN);
        const subInst = new Int32Array(subN);
        for (let p = 0; p < subN; p++) {
          const f = subIdx ? subIdx[p] : p;
          subClass[p] = segLive.fullClassIds[f];
          subInst[p]  = segLive.fullInstanceIds[f];
        }
        setCloud({ ...c, classIds: subClass, instanceIds: subInst, isFromPrelabel: !!c.isFromPrelabel });
      } else if (c.fullClassIds && c.fullInstanceIds) {
        if (c.isFromPrelabel) {
          prelabelRef.current = {
            classFull: c.fullClassIds.slice(),
            instanceFull: c.fullInstanceIds.slice(),
          };
        } else {
          prelabelRef.current = { classFull: null, instanceFull: null };
        }
        setSegState(initSegState({
          classFull: c.fullClassIds,
          instanceFull: c.fullInstanceIds,
          isFromPrelabel: c.isFromPrelabel,
          segBoxes: (c.segIds && c.segCenters && c.segSizes)
            ? { segIds: c.segIds, segCenters: c.segCenters, segSizes: c.segSizes }
            : null,
        }));
      } else {
        prelabelRef.current = { classFull: null, instanceFull: null };
        setSegState(null);
      }

      // Hydrate hidden-inst-ids + preseg/source fingerprints from server state.
      // This is separate from the segLive payload (which carries hulls/labels)
      // and merges scalar fields onto the local segState owned by the FE.
      getSegmentState().then((srv) => {
        if (cancel) return;
        setSegState((s) => (s ? hydrateFromServerState(s, srv) : s));
      }).catch(() => {});

      // Cuboid recommendations from prelabels are disabled — the right
      // Instances panel only ever holds user-authored cuboids. Presegments
      // live in the left list and feed into cuboids manually.
      setGtInstances(gtDoc.instances || []);

      setLoading(false);
    }).catch((e) => {
      if (cancel) return;
      // Pin mismatch / corrupt session: surface a blocking banner and leave
      // the scene unloaded so the user can pick another session. The failed
      // session must NOT stay "active" — otherwise switchSession's equality
      // guard would silently swallow a retry of the same session.
      if (e.status === 409 && e.detail) {
        setPinError(e.detail);
        setActiveSessionId(null);
      } else {
        setLoadError(String(e.message || e));
      }
      setLoading(false);
    });
    VoxaAPI.getAnnotation(activeScene, 'pred')
      .then((d) => !cancel && setPredInstances(d.instances || []));
    return () => { cancel = true; };
    // activeSessionId is in the deps so an explicit session pick (which sets
    // the ref and bumps activeSessionId) re-runs this effect and reloads.
  }, [activeScene, t.mode, activeSessionId]); // eslint-disable-line react-hooks/exhaustive-deps

  const theme = t.theme === 'dark'
    ? { bg: '#0a0b0e', floor: '#15171c' }
    : { bg: '#ebedf1', floor: '#d9dde3' };
  const themeClass = t.theme === 'dark' ? 'theme-dark' : 'theme-light';

  useEffectApp(() => { document.body.className = themeClass; }, [themeClass]);

  const saveGt = useCallbackApp(async (instances) => {
    if (!activeScene) return;
    setGtInstances(instances);
    await VoxaAPI.putAnnotation(activeScene, 'gt', { instances });
    setSavedAt(new Date().toLocaleTimeString());
    setCuboidDirty(false);
  }, [activeScene]);

  // Auto-save: debounce cuboid edits to the backend so a refresh never loses
  // unconfirmed work. Refs let the debounced save read the LATEST scene id
  // even after a scene switch interleaves with the timer, and let the
  // beforeunload flush below send whatever's pending without rebuilding the
  // closure chain.
  const activeSceneRef = useRefApp(activeScene);
  useEffectApp(() => { activeSceneRef.current = activeScene; }, [activeScene]);
  const autosaveTimerRef = useRefApp(null);
  const pendingSaveRef = useRefApp(null);  // { scene, instances } | null

  const onCuboidChange = useCallbackApp((instances) => {
    setGtInstances(instances);
    setCuboidDirty(true);
    const sceneAtChange = activeSceneRef.current;
    if (!sceneAtChange) return;
    pendingSaveRef.current = { scene: sceneAtChange, instances };
    if (autosaveTimerRef.current) clearTimeout(autosaveTimerRef.current);
    autosaveTimerRef.current = setTimeout(async () => {
      autosaveTimerRef.current = null;
      const pending = pendingSaveRef.current;
      pendingSaveRef.current = null;
      if (!pending || pending.scene !== activeSceneRef.current) return;
      try {
        await VoxaAPI.putAnnotation(pending.scene, 'gt', { instances: pending.instances });
        setSavedAt(new Date().toLocaleTimeString());
        setCuboidDirty(false);
      } catch (err) {
        console.error('autosave failed:', err);
      }
    }, 600);
  }, []);

  // Cancel any pending autosave when the scene changes; the load effect will
  // re-fetch persisted annotations for the new scene.
  useEffectApp(() => {
    return () => {
      if (autosaveTimerRef.current) {
        clearTimeout(autosaveTimerRef.current);
        autosaveTimerRef.current = null;
      }
    };
  }, [activeScene]);

  // Flush any pending autosave when the user closes/refreshes the tab.
  // `keepalive: true` lets the PUT complete past the navigation so a fast
  // Ctrl+Enter → refresh doesn't lose the just-confirmed state.
  useEffectApp(() => {
    const onBeforeUnload = () => {
      const pending = pendingSaveRef.current;
      if (!pending) return;
      if (autosaveTimerRef.current) {
        clearTimeout(autosaveTimerRef.current);
        autosaveTimerRef.current = null;
      }
      pendingSaveRef.current = null;
      try {
        fetch(`/api/annotations/gt/${pending.scene}`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            scene: pending.scene, kind: 'gt',
            instances: pending.instances, meta: {},
          }),
          keepalive: true,
        });
      } catch { /* best effort */ }
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, []);

  // Auto-save segment (per-point) edits ~600 ms after a change so closing
  // the tab without Ctrl+S no longer drops the work. The endpoint reads the
  // server-side seg session, so the request body is empty; we just need to
  // fire it and clear the dirty flag on success.
  const segAutosaveTimerRef = useRefApp(null);
  useEffectApp(() => {
    if (!segState?.dirty) return;
    if (segAutosaveTimerRef.current) clearTimeout(segAutosaveTimerRef.current);
    segAutosaveTimerRef.current = setTimeout(async () => {
      segAutosaveTimerRef.current = null;
      try {
        await VoxaAPI.segSave();
        setSegState((s) => (s ? { ...s, dirty: false } : s));
        setSavedAt(new Date().toLocaleTimeString());
      } catch (err) {
        console.error('seg autosave failed:', err);
      }
    }, 600);
    return () => {
      if (segAutosaveTimerRef.current) {
        clearTimeout(segAutosaveTimerRef.current);
        segAutosaveTimerRef.current = null;
      }
    };
  }, [segState?.dirty]);

  // beforeunload: best-effort flush of pending segment edits. The save is
  // server-resident, so a keepalive PUT is enough — no payload to ferry.
  useEffectApp(() => {
    const onBeforeUnload = () => {
      if (!segState?.dirty) return;
      if (segAutosaveTimerRef.current) {
        clearTimeout(segAutosaveTimerRef.current);
        segAutosaveTimerRef.current = null;
      }
      try {
        fetch('/api/segment/save', { method: 'PUT', keepalive: true });
      } catch { /* best effort */ }
    };
    window.addEventListener('beforeunload', onBeforeUnload);
    return () => window.removeEventListener('beforeunload', onBeforeUnload);
  }, [segState?.dirty]);

  // Cmd/Ctrl+Z / Shift+Z → segment undo / redo (only when segState active).
  useEffectApp(() => {
    const onKey = async (e) => {
      if (!segState) return;
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      if (!(e.metaKey || e.ctrlKey) || e.key.toLowerCase() !== 'z') return;
      e.preventDefault();
      try {
        const r = e.shiftKey ? await VoxaAPI.segRedo() : await VoxaAPI.segUndo();
        if (r === null) return;
        setSegState((s) => {
          if (!s) return s;
          const next = applyDelta(s, {
            indices: r.indices,
            after_class: r.afterClass,
            after_instance: r.afterInstance,
          });
          viewerRef.current?.recolorByEdit({
            affectedFullIndices: r.indices,
            classFull: next.classFull,
            instanceFull: next.instanceFull,
            colorMode: 'class',
            palette: cloud?.classPalette ?? null,
          });
          return next;
        });
      } catch (err) {
        console.error('undo/redo failed:', err);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [segState, viewerRef, cloud]);

  // Full save: persist per-point segments (into the scan dir) first, then
  // cuboids. Both the Ctrl/Cmd+S shortcut and the header Save button call this
  // so they do identical work.
  const handleSave = useCallbackApp(async () => {
    if (segState?.dirty) {
      try {
        await VoxaAPI.segSave();
        setSegState((s) => s ? { ...s, dirty: false } : s);
      } catch (err) {
        console.error('segSave failed, skipping cuboid save:', err);
        return;
      }
    }
    saveGt(gtInstances);
  }, [gtInstances, saveGt, segState]);

  // Cmd/Ctrl+S → same work as the Save button.
  useEffectApp(() => {
    const onKey = (e) => {
      if ((e.metaKey || e.ctrlKey) && e.key === 's') {
        e.preventDefault();
        handleSave();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [handleSave]);

  // ── Session picker handlers ─────────────────────────────────────────────
  // Switching activates a session by marking the explicit pick (so the load
  // effect sends its session_id) and bumping activeSessionId to re-run the
  // effect. Edits are autosaved, so we only confirm rather than block.
  const switchSession = useCallbackApp((sid) => {
    if (sid === activeSessionId) return;
    explicitSessionRef.current = sid;
    setActiveSessionId(sid);
  }, [activeSessionId]);

  const onSelectSession = useCallbackApp((sid) => {
    if ((cuboidDirty || segState?.dirty)
        && !window.confirm('Unsaved changes are autosaved; switch session?')) {
      return;
    }
    switchSession(sid);
  }, [cuboidDirty, segState?.dirty, switchSession]);

  const onCreateSession = useCallbackApp(async ({ name, presegId }) => {
    try {
      // The create route fingerprints the loaded cloud, so the scene must be
      // loaded first (it is — the picker only shows once a scene is active).
      const created = await VoxaAPI.createSession(activeScene, { name, presegId });
      switchSession(created.session_id);
    } catch (err) {
      console.error('createSession failed:', err);
      window.alert(`Create session failed: ${err.message || err}`);
    }
  }, [activeScene, switchSession]);

  const onRenameSession = useCallbackApp(async (sid, name) => {
    try {
      await VoxaAPI.renameSession(activeScene, sid, name);
      setSessions(await VoxaAPI.listSessions(activeScene));
    } catch (err) {
      console.error('renameSession failed:', err);
      window.alert(`Rename session failed: ${err.message || err}`);
    }
  }, [activeScene]);

  const onDeleteSession = useCallbackApp(async (sid) => {
    try {
      await VoxaAPI.deleteSession(activeScene, sid);
      if (sid === activeSessionId) {
        // Deleting the active session: force the load effect to re-run (clear
        // the loaded-scene marker so the guard doesn't treat it as an echo)
        // and let the backend resume whatever it now considers last-worked.
        explicitSessionRef.current = null;
        loadedSceneRef.current = null;
        setActiveSessionId(null);
      } else {
        setSessions(await VoxaAPI.listSessions(activeScene));
      }
    } catch (err) {
      console.error('deleteSession failed:', err);
      window.alert(`Delete session failed: ${err.message || err}`);
    }
  }, [activeScene, activeSessionId]);

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
                ? <><span className="dot" /> {cloud.numSubsampled.toLocaleString()} / {(cloud.numPointsTotal ?? cloud.numPoints).toLocaleString()} pts</>
                : <><span className="dot" style={{ background: 'oklch(0.6 0.02 250)' }} /> No scene</>
          }
        </div>
        <button className="header-btn" onClick={() => setScenePickerOpen((o) => !o)}>
          <span style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: 11 }}>◧</span>
          {activeScene || 'Pick scene'}
          {(cuboidDirty || segState?.dirty) && (
            <span title="Unsaved changes" style={{ color: 'oklch(0.75 0.18 60)', marginLeft: 4 }}>●</span>
          )}
        </button>
        <button className="header-btn icon-only"
          onClick={() => setTweak('theme', t.theme === 'dark' ? 'light' : 'dark')}
          title={t.theme === 'dark' ? 'Switch to bright mode' : 'Switch to dark mode'}>
          {t.theme === 'dark' ? '☀' : '☾'}
        </button>
        {savedAt && <span className="header-meta" style={{ fontSize: 11 }}>saved {savedAt}</span>}
        <button className="header-btn primary" onClick={handleSave} title="Save (Ctrl+S)">
          Ctrl+S Save
        </button>
      </header>

      {scenePickerOpen && (
        <ScenePicker
          scenes={scenes}
          activeScene={activeScene}
          onPick={(name) => {
            // Switching scenes drops the in-memory cloud, the per-point
            // segState (presegments + selection), and resets selection in
            // the instance list. Saved-to-disk annotations survive, but any
            // unsaved labels and the active selection do not — so warn.
            if (name && activeScene && name !== activeScene) {
              const ok = window.confirm(
                'Switch scene?\n\n'
                + 'Any unsaved instances, selections, and presegmentation '
                + 'state for the current scene may be lost.');
              if (!ok) return;
            }
            setActiveScene(name);
            setScenePickerOpen(false);
          }}
          onClose={() => setScenePickerOpen(false)}
        />
      )}

      {pinError && (
        <div className="pin-error-banner">
          <span className="dot" style={{ background: 'oklch(0.65 0.18 25)' }} />
          <div className="pin-error-text">
            <b>{pinError.message || 'Session could not be loaded'}</b>
            {pinError.diverged && (
              <em>diverged pin: {pinError.diverged}</em>
            )}
          </div>
          <button className="ghost-btn" onClick={() => setPinError(null)}>Dismiss</button>
        </div>
      )}

      <div className="app-body">
        {t.mode === 'inspect' && (
          <InspectMode key="i" cloud={cloud} loading={loading} theme={theme}
            viewerRef={viewerRef} sceneName={activeScene}
            navMode={navMode} onNavModeChange={setNavMode}
            onCameraChange={onMainCameraChange} />
        )}
        {t.mode === 'label' && (
          <LabelMode key="l" cloud={cloud} setCloud={setCloud} theme={theme} viewerRef={viewerRef}
            classes={classes} instances={gtInstances}
            cloudBBox={cloud?.bbox}
            navMode={navMode} onNavModeChange={setNavMode}
            onChange={onCuboidChange} onSave={saveGt}
            segState={segState} setSegState={setSegState}
            prelabelRef={prelabelRef}
            onCameraChange={onMainCameraChange}
            isAnnotated={scenes.find((s) => (s.id || s.name) === activeScene)?.tier === 'annotated'}
            sessions={sessions} activeSessionId={activeSessionId} presegs={presegs}
            onSelectSession={onSelectSession} onCreateSession={onCreateSession}
            onRenameSession={onRenameSession} onDeleteSession={onDeleteSession}
            hasMesh={!!cloud?.meshUrl} />
        )}
        {t.mode === 'compare' && (
          <CompareMode key="c" cloud={cloud} theme={theme}
            sceneName={activeScene}
            navMode={navMode} onNavModeChange={setNavMode}
            gtInstances={gtInstances} predInstances={predInstances} />
        )}
        {t.mode === 'edit' && (
          <EditMode key="e" cloud={cloud} theme={theme} viewerRef={viewerRef}
            sceneName={activeScene}
            navMode={navMode} onNavModeChange={setNavMode}
            onCameraChange={onMainCameraChange} />
        )}
      </div>

      <TweaksPanel title="Tweaks">
        <TweakSection label="Mode" />
        <TweakRadio label="Active mode" value={t.mode}
          options={[
            { value: 'inspect', label: 'Inspect' },
            { value: 'label',   label: 'Label' },
            { value: 'compare', label: 'Compare' },
            { value: 'edit',    label: 'Edit' },
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

