// mode-label.jsx — class palette (left), instance list (right), cuboid editing.

import { useState as useStateLabel, useMemo as useMemoLabel,
         useEffect as useEffectLabel, useCallback as useCallbackLabel } from 'react';
import * as THREE from 'three';
import { Viewer } from './viewer.jsx';
import { ViewportToolbar, ToolButton, HUDChip, CameraPresets, NavModeToggle, HelpButton } from './viewport-atoms.jsx';
import { VoxaAPI, newId } from './api.js';
import { SegmentToolStrip, PickTool, BrushTool } from './segment-tools.jsx';
import { applyDelta, computeDiffMask } from './segment-state.js';

// "30k", "1.2M", "523" — keeps the HUD chip narrow regardless of scene size.
function formatPointCount(n) {
  if (n < 1000) return String(n);
  if (n < 1e6) return `${(n / 1e3).toFixed(n < 10000 ? 1 : 0)}k`;
  return `${(n / 1e6).toFixed(n < 1e7 ? 2 : 1)}M`;
}

export function LabelMode({ cloud, theme, viewerRef, classes, instances, onChange, onSave, cloudBBox, navMode, onNavModeChange, segState, setSegState, prelabelRef, onCameraChange, hasMesh }) {
  const [activeClass, setActiveClass] = useStateLabel(classes[0]?.id || 'unknown');
  const [selectedId, setSelectedId] = useStateLabel(null);
  const [hiddenClasses, setHiddenClasses] = useStateLabel(new Set());
  const [activeTool, setActiveTool] = useStateLabel('cuboid');
  const [colorMode] = useStateLabel('class');
  const [showDiff, setShowDiff] = useStateLabel(false);
  // Gizmo mode for the selected cuboid. null = no gizmo (edges only).
  const [transformMode, setTransformMode] = useStateLabel('translate');
  // Free-text filter for the instance list (matches label + class name).
  const [instFilter, setInstFilter] = useStateLabel('');
  // Which instance row is currently expanded for inline edit.
  const [editingId, setEditingId] = useStateLabel(null);
  // When true, points inside any confirmed cuboid are hidden from the main
  // viewport (NaN'd in the position buffer). Default on so the labeling
  // workflow naturally reveals what's left to label.
  const [hideConfirmed, setHideConfirmed] = useStateLabel(true);
  const [sideRCollapsed, setSideRCollapsed] = useStateLabel(() => {
    try { return localStorage.getItem('voxa.label.sideRCollapsed') === '1'; }
    catch { return false; }
  });
  useEffectLabel(() => {
    try { localStorage.setItem('voxa.label.sideRCollapsed', sideRCollapsed ? '1' : '0'); }
    catch { /* quota / private mode */ }
  }, [sideRCollapsed]);

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

  // Only the selected cuboid renders in the viewer — keeps the scene readable
  // when there are dozens/hundreds of prelabel instances. Hidden classes still
  // hide everything in their class.
  const visibleIds = useMemoLabel(() => {
    if (!selectedId) return [];
    const sel = instances.find((i) => i.id === selectedId);
    if (!sel || hiddenClasses.has(sel.cls)) return [];
    return [selectedId];
  }, [instances, hiddenClasses, selectedId]);

  const counts = useMemoLabel(() => {
    const c = {};
    instances.forEach((i) => { c[i.cls] = (c[i.cls] || 0) + 1; });
    return c;
  }, [instances]);

  const selected = instances.find((i) => i.id === selectedId);
  const activeClassDef = classes.find((c) => c.id === activeClass);
  // Confirmed instances are read-only: no gizmo, no auto-fit, no rename, no
  // class change, no delete. The user reopens (toggles confirmed off) first.
  const isLocked = !!selected?.confirmed;

  // Pass-through for the viewer to highlight points inside the currently
  // selected cuboid. Updates as the box is dragged because `selected` is
  // re-derived from `instances` on every render.
  // Dense overlay: full-density LAZ points inside the selected cuboid.
  // Manually triggered (D hotkey) so the user controls when to pay the
  // load cost. Auto-clears whenever the selected cuboid moves/resizes so
  // a stale overlay never stays "stuck" beside the box after a drag.
  const [denseOverlay, setDenseOverlay] = useStateLabel(null);
  // Bumping this token causes the fetch effect to refire with the current
  // cuboid bounds. Token-based (not bounds-based) so we don't thrash the
  // backend on every gizmo tick.
  const [denseTrigger, setDenseTrigger] = useStateLabel(0);

  // Stable key for the selected cuboid's geometry. Whenever this changes
  // (drag, resize, rotate, deselect, switch to another instance) we drop
  // the overlay so it can't visibly disconnect from the box.
  const selectedBoundsKey = useMemoLabel(() => {
    if (!selected) return null;
    const c = selected.center, sz = selected.size, r = selected.rotation || [0, 0, 0];
    return `${selected.id}|${c[0]},${c[1]},${c[2]}|${sz[0]},${sz[1]},${sz[2]}|${r[0]},${r[1]},${r[2]}`;
  }, [selected]);

  useEffectLabel(() => { setDenseOverlay(null); }, [selectedBoundsKey]);

  useEffectLabel(() => {
    if (!denseTrigger) return;
    if (!selected) return;
    const center = selected.center;
    const size = selected.size;
    const rot = selected.rotation || [0, 0, 0];
    const hx = size[0] / 2, hy = size[1] / 2, hz = size[2] / 2;
    const m = new THREE.Matrix4().makeRotationFromEuler(
      new THREE.Euler(rot[0], rot[1], rot[2], 'XYZ')
    );
    const v = new THREE.Vector3();
    let minX = Infinity, minY = Infinity, minZ = Infinity;
    let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
    for (const sx of [-hx, hx]) for (const sy of [-hy, hy]) for (const sz of [-hz, hz]) {
      v.set(sx, sy, sz).applyMatrix4(m);
      if (v.x < minX) minX = v.x; if (v.y < minY) minY = v.y; if (v.z < minZ) minZ = v.z;
      if (v.x > maxX) maxX = v.x; if (v.y > maxY) maxY = v.y; if (v.z > maxZ) maxZ = v.z;
    }
    // Small margin so points right at the box edge are visible.
    const dx = (maxX - minX) * 0.10, dy = (maxY - minY) * 0.10, dz = (maxZ - minZ) * 0.10;
    const aabbMin = [center[0] + minX - dx, center[1] + minY - dy, center[2] + minZ - dz];
    const aabbMax = [center[0] + maxX + dx, center[1] + maxY + dy, center[2] + maxZ + dz];

    let cancelled = false;
    VoxaAPI.loadRegion(aabbMin, aabbMax, { maxPoints: 500_000 })
      .then((res) => {
        if (cancelled) return;
        setDenseOverlay({ positions: res.positions, colors: res.colors });
      })
      .catch(() => { if (!cancelled) setDenseOverlay(null); });
    return () => { cancelled = true; };
    // Token-based: only fires when the user explicitly hits D.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [denseTrigger]);

  const highlightCuboid = useMemoLabel(() => {
    if (!selected) return null;
    const cls = classes.find((c) => c.id === selected.cls);
    return {
      center: selected.center,
      size: selected.size,
      rotation: selected.rotation || [0, 0, 0],
      color: cls?.color || selected.color,
    };
  }, [selected, classes]);

  // Stable string key for the confirmed subset. Used to short-circuit
  // hideCuboids' identity on gizmo drags of UNCONFIRMED instances (which
  // mutate `instances` every tick but leave the confirmed subset alone).
  const confirmedKey = useMemoLabel(() => {
    let s = '';
    for (const i of instances) {
      if (!i.confirmed) continue;
      const c = i.center, sz = i.size, r = i.rotation || [0, 0, 0];
      s += `${i.id}|${c[0]},${c[1]},${c[2]}|${sz[0]},${sz[1]},${sz[2]}|${r[0]},${r[1]},${r[2]};`;
    }
    return s;
  }, [instances]);

  const confirmedCount = useMemoLabel(
    () => instances.reduce((n, i) => n + (i.confirmed ? 1 : 0), 0),
    [instances],
  );

  // Always populated when there are confirmed instances, regardless of the
  // hide toggle. The Viewer uses it to compute "points labeled / left" stats
  // as well as to optionally NaN positions.
  const confirmedCuboids = useMemoLabel(() => {
    if (!confirmedKey) return [];
    return instances
      .filter((i) => i.confirmed)
      .map((i) => ({
        center: i.center,
        size: i.size,
        rotation: i.rotation || [0, 0, 0],
      }));
    // confirmedKey transitively covers `instances`; eslint can't see that.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [confirmedKey]);

  // Stats from Viewer's confirmed-mask pass: how many points fall inside any
  // confirmed cuboid (unique), regardless of show/hide toggle.
  const [labelStats, setLabelStats] = useStateLabel({ total: 0, labeled: 0, left: 0 });

  const filteredInstances = useMemoLabel(() => {
    const q = instFilter.trim().toLowerCase();
    if (!q) return instances;
    return instances.filter((inst) => {
      const cls = classes.find((c) => c.id === inst.cls);
      return (
        (inst.label || '').toLowerCase().includes(q) ||
        (cls?.label || inst.cls || '').toLowerCase().includes(q) ||
        (inst.id || '').toLowerCase().includes(q)
      );
    });
  }, [instances, classes, instFilter]);

  const helpSections = useMemoLabel(() => ([
    {
      title: 'Cuboid',
      items: [
        { keys: ['A'], desc: 'Add cuboid for active class' },
        { keys: ['G'], desc: 'Move (translate gizmo)' },
        { keys: ['R'], desc: 'Rotate gizmo' },
        { keys: ['Y'], desc: 'Scale gizmo' },
        { keys: ['F'], desc: 'Frame selection' },
        { keys: ['⌫'], desc: 'Delete selected' },
        { keys: ['Ctrl', '↵'], desc: 'Confirm selected (hides interior pts)' },
        { keys: ['⌘', 'S'], desc: 'Save annotations' },
      ],
    },
    {
      title: 'Class assignment',
      items: classes.length
        ? classes.map((c) => ({ keys: [c.hotkey], desc: c.label }))
        : [{ keys: ['—'], desc: 'No classes configured' }],
    },
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
      title: 'Mouse',
      items: [
        { keys: ['Dbl-click'], desc: 'Select cuboid in right list (shows box)' },
        { keys: ['✎'], desc: 'Edit button selects + opens panel' },
        { keys: ['Drag', 'gizmo'], desc: 'Move / rotate / scale selected' },
      ],
    },
    {
      title: 'Other',
      items: [
        { keys: ['?'], desc: 'Toggle this panel' },
        { keys: ['Esc'], desc: 'Close panel' },
      ],
    },
  ]), [classes, navMode]);

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

  const updateInstance = (id, patch) => {
    onChange(instances.map((i) => i.id === id ? { ...i, ...patch } : i));
  };
  const deleteInstance = (id) => {
    onChange(instances.filter((i) => i.id !== id));
    if (selectedId === id) setSelectedId(null);
    if (editingId === id) setEditingId(null);
  };
  const autoFitInstance = async (inst) => {
    const half = inst.size.map((v) => v / 2);
    const cmin = [inst.center[0] - half[0], inst.center[1] - half[1], inst.center[2] - half[2]];
    const cmax = [inst.center[0] + half[0], inst.center[1] + half[1], inst.center[2] + half[2]];
    const cls = classes.find((c) => c.id === inst.cls);
    const fitted = await VoxaAPI.autoFit(cmin, cmax, inst.cls, cls?.color || inst.color, inst.label);
    updateInstance(inst.id, { center: fitted.center, size: fitted.size });
  };
  const focusInstance = (inst) => {
    if (!inst) return;
    viewerRef.current?.frame(
      new THREE.Vector3(...inst.center),
      Math.max(...inst.size) / 2,
    );
  };
  // Toggle confirmed state. When transitioning to confirmed, clear the
  // selection so the cuboid edges + gizmo disappear alongside the (now
  // hidden) interior points — the visual signal that the instance is "done".
  const toggleConfirm = (id) => {
    const target = instances.find((i) => i.id === id);
    if (!target) return;
    const willConfirm = !target.confirmed;
    onChange(instances.map((i) => i.id === id ? { ...i, confirmed: willConfirm } : i));
    if (willConfirm) {
      if (selectedId === id) setSelectedId(null);
      if (editingId === id) setEditingId(null);
    }
  };
  const toggleConfirmSelected = useCallbackLabel(() => {
    if (!selectedId) return;
    const target = instances.find((i) => i.id === selectedId);
    if (!target) return;
    const willConfirm = !target.confirmed;
    onChange(instances.map((i) => i.id === selectedId ? { ...i, confirmed: willConfirm } : i));
    if (willConfirm) {
      setSelectedId(null);
      if (editingId === selectedId) setEditingId(null);
    }
  }, [selectedId, editingId, instances, onChange]);

  // Gizmo drag callback. Patches the targeted instance by id (not by selectedId)
  // since the viewer dispatches based on its own gizmoTargetIdRef snapshot.
  const onCuboidTransform = useCallbackLabel((id, patch) => {
    const next = instances.map((i) => i.id === id ? { ...i, ...patch } : i);
    onChange(next);
  }, [instances, onChange]);

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
      // Ctrl/Cmd+Enter: confirm/unconfirm the selected instance. Runs before
      // class-hotkey lookup so Enter never doubles as a class hotkey.
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        toggleConfirmSelected();
        return;
      }
      if (navMode === 'walk' && /^[wasdqeWASDQE]$/.test(e.key)) return;
      const cls = classes.find((c) => c.hotkey === e.key);
      if (cls) {
        setActiveClass(cls.id);
        // Class change is an edit — block it for confirmed instances.
        if (selected && !isLocked) updateSelected({ cls: cls.id, color: cls.color });
      } else if (e.key === 'Backspace' || e.key === 'Delete') {
        if (selected && !isLocked) { e.preventDefault(); deleteSelected(); }
      } else if (e.key === 'a' || e.key === 'A') {
        addCuboid();
      } else if (e.key === 'f' || e.key === 'F') {
        if (selected) {
          viewerRef.current?.frame(
            new THREE.Vector3(...selected.center),
            Math.max(...selected.size) / 2,
          );
        }
      } else if (!isLocked && (e.key === 'g' || e.key === 'G')) {
        setTransformMode('translate');
      } else if (!isLocked && (e.key === 'r' || e.key === 'R')) {
        setTransformMode('rotate');
      } else if (!isLocked && (e.key === 'y' || e.key === 'Y')) {
        setTransformMode('scale');
      } else if (e.key === 'd' || e.key === 'D') {
        // Densify: pop full-density LAZ points inside the selected cuboid.
        // Manual trigger so we don't refetch on every gizmo drag tick. The
        // overlay auto-clears when the box moves or the cuboid is deselected.
        if (selected) setDenseTrigger((t) => t + 1);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
    // eslint-disable-next-line
  }, [classes, selected, isLocked, instances, activeTool, navMode]);

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
          transformMode={activeTool === 'cuboid' && !isLocked ? transformMode : null}
          onCuboidTransform={onCuboidTransform}
          highlightCuboid={highlightCuboid}
          denseOverlay={denseOverlay}
          confirmedCuboids={confirmedCuboids}
          hideConfirmedPoints={hideConfirmed}
          onLabelStats={setLabelStats}
          onCameraChange={onCameraChange}
        />

        <div className="vp-hud-top">
          <div className="hud-group">
            {labelStats.total > 0 && (
              <HUDChip label="Points left"
                value={`${formatPointCount(labelStats.left)} / ${formatPointCount(labelStats.total)}`}
                mono />
            )}
          </div>
          <div className="hud-group">
            <NavModeToggle navMode={navMode} onChange={onNavModeChange} />
            <CameraPresets onPreset={(p) => viewerRef.current?.preset(p)} />
            <button className="hud-chip-btn"
              onClick={() => window.open(window.location.pathname + '?mesh=1', 'voxa-mesh',
                'popup=yes,width=960,height=720')}
              disabled={!hasMesh}
              title={hasMesh
                ? 'Open synced mesh-only companion window'
                : 'No mesh available for this scene'}>
              ▦ Mesh window
            </button>
          </div>
        </div>

        <div className="vp-help-corner">
          <HelpButton sections={helpSections} placement="up" />
        </div>

        <ViewportToolbar side="left">
          {segState && (
            <>
              <SegmentToolStrip
                activeTool={activeTool}
                onChange={setActiveTool}
                hasSegState={!!segState}
              />
              <div className="tool-sep" />
            </>
          )}
          {activeTool === 'cuboid' && (
            <>
              {!isLocked && (
                <>
                  <ToolButton mini icon="⇄" label="Move (G)"
                    onClick={() => setTransformMode('translate')}
                    active={transformMode === 'translate'} />
                  <ToolButton mini icon="↻" label="Rotate (R)"
                    onClick={() => setTransformMode('rotate')}
                    active={transformMode === 'rotate'} />
                  <ToolButton mini icon="⇲" label="Scale (Y)"
                    onClick={() => setTransformMode('scale')}
                    active={transformMode === 'scale'} />
                </>
              )}
              {selected && (
                <>
                  <div className="tool-sep" />
                  <ToolButton mini icon="◎" label="Focus selection (F)"
                    onClick={() => focusInstance(selected)} />
                  {!isLocked && (
                    <>
                      <ToolButton mini icon="✦" label="Auto-fit selection" onClick={autoFitSelected} />
                      <ToolButton mini icon="⌫" label="Delete selection" onClick={deleteSelected} />
                    </>
                  )}
                </>
              )}
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

      </div>

      {/* Right: filterable instance list + slim inspector */}
      <aside className={'side-r' + (sideRCollapsed ? ' collapsed' : '')}>
        {sideRCollapsed ? (
          <button className="side-collapse-handle"
            onClick={() => setSideRCollapsed(false)}
            title={`Show instances panel (${instances.length})`}>
            <span className="side-collapse-chev">‹</span>
            <span className="side-collapse-label">Instances</span>
            <span className="badge-soft">{instances.length}</span>
          </button>
        ) : (
        <>
        <div className="side-hd">
          <button className="side-collapse-btn"
            onClick={() => setSideRCollapsed(true)}
            title="Collapse panel">›</button>
          <span>Instances</span>
          <div className="side-hd-actions">
            {confirmedCount > 0 && (
              <button className="hide-labeled-btn"
                onClick={() => setHideConfirmed((v) => !v)}
                title={hideConfirmed
                  ? `Show ${confirmedCount} labeled instance${confirmedCount === 1 ? '' : 's'}`
                  : `Hide ${confirmedCount} labeled instance${confirmedCount === 1 ? '' : 's'}`}>
                {hideConfirmed ? '◌' : '●'} {confirmedCount} done
              </button>
            )}
            <span className="badge-soft">
              {instFilter ? `${filteredInstances.length} / ${instances.length}` : instances.length}
            </span>
          </div>
        </div>
        <div className="inst-filter">
          <input className="ins-input"
            placeholder="Filter by label, class, or id"
            value={instFilter}
            onChange={(e) => setInstFilter(e.target.value)} />
          {instFilter && (
            <button className="inst-filter-clear"
              onClick={() => setInstFilter('')}
              title="Clear filter">×</button>
          )}
        </div>
        <div className="inst-list">
          {instances.length === 0 && (
            <div className="sugg-empty">No instances yet. Press <kbd>A</kbd> to add.</div>
          )}
          {instances.length > 0 && filteredInstances.length === 0 && (
            <div className="sugg-empty">No matches for "{instFilter}".</div>
          )}
          {filteredInstances.map((inst) => {
            const cls = classes.find((c) => c.id === inst.cls);
            const isSel = inst.id === selectedId;
            const isEditing = inst.id === editingId;
            return (
              <div key={inst.id} className={'inst-item' + (isEditing ? ' editing' : '')}>
                <div className={'inst-row' + (isSel ? ' selected' : '') + (inst.confirmed ? ' confirmed' : '')}
                  onDoubleClick={() => setSelectedId(isSel ? null : inst.id)}
                  title={isSel ? 'Double-click to deselect' : 'Double-click to select (shows bounding box)'}>
                  <span className="inst-dot" style={{ background: cls?.color || inst.color }} />
                  <div className="inst-text">
                    <b>{inst.label}</b>
                    <em>{cls?.label || inst.cls}</em>
                  </div>
                  <button className={'inst-edit-btn' + (inst.confirmed ? ' is-confirmed' : '')}
                    onClick={(e) => { e.stopPropagation(); toggleConfirm(inst.id); }}
                    title={inst.confirmed ? 'Reopen (mark as unlabeled)' : 'Confirm (Ctrl+Enter)'}>
                    {inst.confirmed ? '✓' : '○'}
                  </button>
                  <button className="inst-edit-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      setSelectedId(inst.id);
                      focusInstance(inst);
                    }}
                    title="Focus camera on this instance">◎</button>
                  <button className="inst-edit-btn"
                    onClick={(e) => {
                      e.stopPropagation();
                      if (isEditing) {
                        setEditingId(null);
                      } else {
                        setSelectedId(inst.id);
                        setEditingId(inst.id);
                      }
                    }}
                    title={isEditing ? 'Close' : 'Edit (selects + opens panel)'}>{isEditing ? '×' : '✎'}</button>
                </div>
                {isEditing && (
                  <div className="inst-edit-panel">
                    {inst.confirmed && (
                      <div className="locked-banner">🔒 Confirmed — Reopen to edit</div>
                    )}
                    <div className="ins-row">
                      <label>Name</label>
                      <input className="ins-input"
                        value={inst.label}
                        autoFocus
                        disabled={inst.confirmed}
                        onChange={(e) => updateInstance(inst.id, { label: e.target.value })} />
                    </div>
                    <div className="ins-row">
                      <label>Class</label>
                      <div className="class-pills">
                        {classes.map((c) => (
                          <button key={c.id}
                            className={'class-pill' + (c.id === inst.cls ? ' active' : '')}
                            disabled={inst.confirmed}
                            onClick={() => updateInstance(inst.id, { cls: c.id, color: c.color })}
                            title={`${c.label}${c.hotkey ? `  (${c.hotkey})` : ''}`}>
                            <span className="class-swatch" style={{ background: c.color }} />
                            <span>{c.label}</span>
                          </button>
                        ))}
                      </div>
                    </div>
                    <div className="ins-actions">
                      <button className="ghost-btn" onClick={() => focusInstance(inst)}>◎ Focus</button>
                      {!inst.confirmed && (
                        <button className="ghost-btn" onClick={() => autoFitInstance(inst)}>↻ Auto-fit</button>
                      )}
                      <button className="ghost-btn" onClick={() => toggleConfirm(inst.id)}
                        title={inst.confirmed ? 'Reopen' : 'Confirm (Ctrl+Enter)'}>
                        {inst.confirmed ? '✓ Reopen' : '✓ Confirm'}
                      </button>
                      {!inst.confirmed && (
                        <button className="ghost-btn danger" onClick={() => deleteInstance(inst.id)}>Delete</button>
                      )}
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
        </>
        )}
      </aside>
    </div>
  );
}

