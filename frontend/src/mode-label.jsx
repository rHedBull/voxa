// mode-label.jsx — class palette (left), instance list (right), cuboid editing.

import { useState as useStateLabel, useMemo as useMemoLabel,
         useEffect as useEffectLabel, useCallback as useCallbackLabel } from 'react';
import * as THREE from 'three';
import { Viewer } from './viewer.jsx';
import { ViewportToolbar, ToolButton, HUDChip, CameraPresets, NavModeToggle } from './viewport-atoms.jsx';
import { VoxaAPI, newId } from './api.js';
import { SegmentToolStrip, PickTool, BrushTool } from './segment-tools.jsx';
import { applyDelta, computeDiffMask } from './segment-state.js';

export function LabelMode({ cloud, theme, viewerRef, classes, instances, onChange, onSave, sceneName, cloudBBox, navMode, onNavModeChange, segState, setSegState, prelabelRef }) {
  const [activeClass, setActiveClass] = useStateLabel(classes[0]?.id || 'unknown');
  const [selectedId, setSelectedId] = useStateLabel(null);
  const [hiddenClasses, setHiddenClasses] = useStateLabel(new Set());
  const [activeTool, setActiveTool] = useStateLabel('cuboid');
  const [colorMode] = useStateLabel('class');
  const [showDiff, setShowDiff] = useStateLabel(false);

  const diffMask = useMemoLabel(() => {
    if (!showDiff || !segState) return null;
    const pre = prelabelRef?.current;
    if (!pre?.classFull || !pre?.instanceFull) return null;
    return computeDiffMask(
      segState.classFull, pre.classFull,
      segState.instanceFull, pre.instanceFull,
    );
  }, [showDiff, segState, prelabelRef]);

  // Keep activeClass valid as the class list streams in.
  useEffectLabel(() => {
    if (classes.length && !classes.find((c) => c.id === activeClass)) {
      setActiveClass(classes[0].id);
    }
  }, [classes, activeClass]);

  const visibleIds = useMemoLabel(() => (
    instances.filter((i) => !hiddenClasses.has(i.cls)).map((i) => i.id)
  ), [instances, hiddenClasses]);

  const counts = useMemoLabel(() => {
    const c = {};
    instances.forEach((i) => { c[i.cls] = (c[i.cls] || 0) + 1; });
    return c;
  }, [instances]);

  const selected = instances.find((i) => i.id === selectedId);
  const activeClassDef = classes.find((c) => c.id === activeClass);

  const toggleClass = (cls) => {
    setHiddenClasses((s) => {
      const n = new Set(s);
      n.has(cls) ? n.delete(cls) : n.add(cls);
      return n;
    });
  };

  // Add a cuboid centered on the loaded scene's bbox center, sized as a small
  // cube. The user then nudges it via the inspector or auto-fit.
  const addCuboid = useCallbackLabel(async () => {
    if (!cloudBBox || !activeClassDef) return;
    const cx = (cloudBBox.min[0] + cloudBBox.max[0]) / 2;
    const cy = (cloudBBox.min[1] + cloudBBox.max[1]) / 2;
    const cz = (cloudBBox.min[2] + cloudBBox.max[2]) / 2;
    const ext = Math.max(
      cloudBBox.max[0] - cloudBBox.min[0],
      cloudBBox.max[1] - cloudBBox.min[1],
      cloudBBox.max[2] - cloudBBox.min[2],
    );
    const s = Math.max(0.05, ext * 0.1);
    const inst = {
      id: newId(),
      cls: activeClassDef.id,
      label: `${activeClassDef.label} ${(counts[activeClassDef.id] || 0) + 1}`,
      color: activeClassDef.color,
      center: [cx, cy, cz],
      size: [s, s, s],
      rotation: [0, 0, 0],
      conf: 1.0,
      source: 'manual',
    };
    const next = [...instances, inst];
    onChange(next);
    setSelectedId(inst.id);
  }, [activeClassDef, instances, cloudBBox, counts, onChange]);

  const updateSelected = (patch) => {
    const next = instances.map((i) => i.id === selectedId ? { ...i, ...patch } : i);
    onChange(next);
  };

  const deleteSelected = () => {
    if (!selectedId) return;
    onChange(instances.filter((i) => i.id !== selectedId));
    setSelectedId(null);
  };

  const autoFitSelected = async () => {
    if (!selected) return;
    const half = selected.size.map((v) => v / 2);
    const cmin = [selected.center[0] - half[0], selected.center[1] - half[1], selected.center[2] - half[2]];
    const cmax = [selected.center[0] + half[0], selected.center[1] + half[1], selected.center[2] + half[2]];
    const fitted = await VoxaAPI.autoFit(cmin, cmax, selected.cls,
      activeClassDef?.color || selected.color, selected.label);
    updateSelected({ center: fitted.center, size: fitted.size });
  };

  // Pick-tool apply: handles selection updates (__select__) and actual segment ops.
  const onPickApply = useCallbackLabel(async (op, { indices, payload } = {}) => {
    if (op === '__select__') {
      // indices is actually the new Set of selected instance ids in this path.
      setSegState((s) => s ? { ...s, selection: indices } : s);
      return;
    }
    try {
      const r = await VoxaAPI.segApply(op, { indices, payload });
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
          colorMode,
          palette: cloud?.classPalette ?? null,
        });
        return next;
      });
    } catch (err) {
      console.error('segApply failed:', err);
    }
  }, [setSegState, colorMode, cloud, viewerRef]);

  // Brush-tool apply: receives a pre-resolved apply response (op === '__delta__').
  const onBrushApply = useCallbackLabel((_op, r) => {
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
        colorMode,
        palette: cloud?.classPalette ?? null,
      });
      return next;
    });
  }, [setSegState, colorMode, cloud, viewerRef]);

  // Hotkeys: 0–9 assign class, ⌫ delete, A add, F frame, ⌘S save.
  // In walk mode the viewer owns WASD/QE; bail on those keys here so we
  // don't double-fire (e.g. 'A' is both walk-left and add-cuboid).
  // Gated on activeTool === 'cuboid' so Pick/Brush tools own their own hotkeys.
  useEffectLabel(() => {
    const onKey = (e) => {
      if (e.target && (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA')) return;
      if (activeTool !== 'cuboid') return;
      if (navMode === 'walk' && /^[wasdqeWASDQE]$/.test(e.key)) return;
      const cls = classes.find((c) => c.hotkey === e.key);
      if (cls) {
        setActiveClass(cls.id);
        if (selected) updateSelected({ cls: cls.id, color: cls.color });
      } else if (e.key === 'Backspace' || e.key === 'Delete') {
        if (selected) { e.preventDefault(); deleteSelected(); }
      } else if (e.key === 'a' || e.key === 'A') {
        addCuboid();
      } else if (e.key === 'f' || e.key === 'F') {
        if (selected) {
          viewerRef.current?.frame(
            new THREE.Vector3(...selected.center),
            Math.max(...selected.size) / 2,
          );
        }
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line
  }, [classes, selected, instances, activeTool, navMode]);

  return (
    <div className="mode-root label">
      {activeTool === 'pick' && segState && (
        <PickTool
          viewerRef={viewerRef}
          segState={segState}
          onApply={onPickApply}
          classes={classes}
        />
      )}
      {activeTool === 'brush' && segState && (
        <BrushTool
          viewerRef={viewerRef}
          segState={segState}
          classes={classes}
          activeClassId={activeClass}
          onApply={onBrushApply}
        />
      )}

      {/* Left: class palette */}
      <aside className="side-l">
        <div className="side-hd">
          <span>Classes</span>
          <span className="badge-soft">{instances.length}</span>
        </div>
        <div className="class-list">
          {classes.map((c) => {
            const hidden = hiddenClasses.has(c.id);
            return (
              <div key={c.id}
                className={'class-row' + (activeClass === c.id ? ' active' : '') + (hidden ? ' hidden' : '')}
                onClick={() => setActiveClass(c.id)}>
                <span className="class-swatch" style={{ background: c.color }} />
                <span className="class-name">{c.label}</span>
                <span className="class-count">{counts[c.id] || 0}</span>
                <button className="class-eye" onClick={(e) => { e.stopPropagation(); toggleClass(c.id); }}
                  title={hidden ? 'Show' : 'Hide'}>{hidden ? '◌' : '●'}</button>
                <span className="class-hk">{c.hotkey}</span>
              </div>
            );
          })}
        </div>

        <div className="side-hd" style={{ marginTop: 14 }}>
          <span>Quick add</span>
        </div>
        <button className="ghost-btn" onClick={addCuboid} disabled={!cloud}>
          + Cuboid for <b style={{ marginLeft: 4 }}>{activeClassDef?.label || '—'}</b> &nbsp;<kbd>A</kbd>
        </button>
        <button className="ghost-btn" style={{ marginTop: 6 }} onClick={() => onSave(instances)}>
          ⌘S Save annotations
        </button>
      </aside>

      {/* Center: viewport */}
      <div className="vp-stack">
        <Viewer
          ref={viewerRef}
          cloud={cloud}
          instances={instances}
          visibleInstanceIds={visibleIds}
          selectedId={selectedId}
          showCuboids
          background={theme.bg}
          floorColor={theme.floor}
          navMode={navMode}
          colorMode={colorMode}
          diffMask={diffMask}
          showDiff={showDiff}
        />

        <div className="vp-hud-top">
          <div className="hud-group">
            <HUDChip label="Scene" value={sceneName || '—'} mono />
            <HUDChip label="Annotations" value={instances.length} mono />
            <HUDChip label="Active class" value={activeClassDef?.label || '—'} accent />
          </div>
          <div className="hud-group">
            <NavModeToggle navMode={navMode} onChange={onNavModeChange} />
            <CameraPresets onPreset={(p) => viewerRef.current?.preset(p)} />
          </div>
        </div>

        <ViewportToolbar side="left">
          <SegmentToolStrip
            activeTool={activeTool}
            onChange={setActiveTool}
            hasSegState={!!segState}
          />
          <div className="tool-sep" />
          {activeTool === 'cuboid' && (
            <>
              <ToolButton mini icon="✦" label="Auto-fit" onClick={autoFitSelected} />
              <ToolButton mini icon="⌫" label="Delete" onClick={deleteSelected} />
            </>
          )}
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

        <div className="vp-hud-bottom">
          <div className="kbd-strip">
            {navMode === 'walk' ? (
              <>
                <span><kbd>W</kbd><kbd>A</kbd><kbd>S</kbd><kbd>D</kbd> move</span>
                <span><kbd>Q</kbd>/<kbd>E</kbd> down/up</span>
                <span><kbd>Drag</kbd> look</span>
                <span style={{ opacity: 0.5 }}>(toggle off walk to use label hotkeys)</span>
              </>
            ) : (
              <>
                <span><kbd>A</kbd> add cuboid</span>
                {classes.length > 0 && <span><kbd>{classes[0].hotkey}</kbd>–<kbd>{classes[classes.length - 1].hotkey}</kbd> assign class</span>}
                <span><kbd>F</kbd> frame selection</span>
                <span><kbd>⌫</kbd> delete</span>
                <span><kbd>⌘S</kbd> save</span>
              </>
            )}
          </div>
        </div>
      </div>

      {/* Right: instance list + inspector */}
      <aside className="side-r">
        <div className="side-hd">
          <span>Instances</span>
          <span className="badge-soft">{instances.length}</span>
        </div>
        <div className="inst-list">
          {instances.length === 0 && (
            <div className="sugg-empty">No instances yet. Press <kbd>A</kbd> to add.</div>
          )}
          {instances.map((inst) => {
            const cls = classes.find((c) => c.id === inst.cls);
            const isSel = inst.id === selectedId;
            return (
              <div key={inst.id}
                className={'inst-row' + (isSel ? ' selected' : '')}
                onClick={() => setSelectedId(inst.id)}>
                <span className="inst-dot" style={{ background: cls?.color || inst.color }} />
                <div className="inst-text">
                  <b>{inst.label}</b>
                  <em><span className="mono">{inst.id}</span> · {cls?.label || inst.cls}</em>
                </div>
                <div className="inst-conf" title={`Confidence ${inst.conf}`}>
                  <i style={{ width: `${(inst.conf || 1) * 100}%`, background: cls?.color || inst.color }} />
                </div>
              </div>
            );
          })}
        </div>

        {selected && (
          <>
            <div className="side-hd" style={{ marginTop: 14 }}>
              <span>Inspector</span>
              <span className="mono dim">{selected.id}</span>
            </div>
            <div className="inspector">
              <div className="ins-row">
                <label>Label</label>
                <input className="ins-input" value={selected.label}
                       onChange={(e) => updateSelected({ label: e.target.value })} />
              </div>
              <div className="ins-row">
                <label>Class</label>
                <div className="ins-class">
                  <span className="class-swatch"
                    style={{ background: classes.find((c) => c.id === selected.cls)?.color || selected.color }} />
                  {classes.find((c) => c.id === selected.cls)?.label || selected.cls}
                </div>
              </div>
              <NumGrid label3={['cx','cy','cz']} values={selected.center}
                onChange={(i, v) => {
                  const c = [...selected.center]; c[i] = v; updateSelected({ center: c });
                }} />
              <NumGrid label3={['w','h','d']} values={selected.size}
                onChange={(i, v) => {
                  const sz = [...selected.size]; sz[i] = Math.max(0.005, v); updateSelected({ size: sz });
                }} />
              <div className="ins-row" style={{ flexDirection: 'row', alignItems: 'center', gap: 8 }}>
                <label style={{ width: 80 }}>Confidence</label>
                <div className="conf-bar"><i style={{ width: `${(selected.conf || 1) * 100}%` }} /></div>
                <span className="mono dim">{(selected.conf || 1).toFixed(2)}</span>
              </div>
              <div className="ins-actions">
                <button className="ghost-btn" onClick={autoFitSelected}>↻ Auto-fit</button>
                <button className="ghost-btn danger" onClick={deleteSelected}>Delete</button>
              </div>
            </div>
          </>
        )}
      </aside>
    </div>
  );
}

function NumGrid({ label3, values, onChange }) {
  return (
    <div className="ins-grid">
      {label3.map((l, i) => (
        <div key={l}>
          <label>{l}</label>
          <input className="ins-input mono" type="number" step="0.01"
                 value={Number(values[i]).toFixed(3)}
                 onChange={(e) => onChange(i, Number(e.target.value))}
                 style={{ height: 22, padding: '0 6px', fontSize: 11 }} />
        </div>
      ))}
    </div>
  );
}

