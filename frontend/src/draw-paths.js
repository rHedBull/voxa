// draw-paths.js — pure state machine for the Draw (centerline) sub-mode.
// All functions take state and return a new state; React owns the object.
// See docs/superpowers/specs/2026-06-04-centerline-pipe-labeling-design.md.
//
// State shape:
// {
//   paths: [{ key, points: [[x,y,z],...], radius, smooth, classId, instKey }],
//   active: string|null,          // key of the path being drawn
//   selection: Set<string>,       // selected path keys
//   selectedPoint: { pathKey, pointIdx } | null,  // single control point (extend anchor)
//   instanceIds: { [instKey]: number },        // backend ids, applied groups only
//   pendingMergedFrom: { [instKey]: number[] },// absorbed backend ids awaiting next apply
//   lastRadius: number,
//   nextKey: number,              // monotonic key source (deterministic, testable)
// }
// classId is the canonical numeric class id (ClassDef.class_id from
// /api/config, matching engine/data/lidar/classes.json) — NOT the
// position in the classes array.
// instKey groups paths into one instance; starts unique per path, M merges.

export function initDrawState({ defaultRadius = 0.15 } = {}) {
  return {
    paths: [],
    active: null,
    selection: new Set(),
    selectedPoint: null,
    instanceIds: {},
    pendingMergedFrom: {},
    lastRadius: defaultRadius,
    nextKey: 1,
  };
}

function freshKey(s) {
  return [`p${s.nextKey}`, { ...s, nextKey: s.nextKey + 1 }];
}

export function addPoint(state, xyz, classId) {
  if (state.active) {
    const paths = state.paths.map((p) =>
      p.key === state.active ? { ...p, points: [...p.points, xyz] } : p);
    return { ...state, paths };
  }
  const [key, s] = freshKey(state);
  const path = {
    key,
    points: [xyz],
    radius: s.lastRadius,
    smooth: false,
    classId,
    instKey: key,        // unique until merged
  };
  return {
    ...s, paths: [...s.paths, path], active: key,
    selection: new Set(), selectedPoint: null,
  };
}

// No-op if pathKey is unknown — pointer drags can race a delete, and a
// missed move is harmless (unlike the structural ops below, which throw).
export function movePoint(state, pathKey, pointIdx, xyz) {
  const paths = state.paths.map((p) => {
    if (p.key !== pathKey) return p;
    const points = p.points.slice();
    points[pointIdx] = xyz;
    return { ...p, points };
  });
  return { ...state, paths };
}

// Looks up a path by key, failing loudly on inconsistent state — a missing
// key here means a wiring bug upstream, and a clear throw beats a cryptic
// TypeError deep in a render.
function mustFind(state, key) {
  const p = state.paths.find((x) => x.key === key);
  if (!p) throw new Error(`draw-paths: no path with key ${key}`);
  return p;
}

export function removeLastPoint(state) {
  if (!state.active) return state;
  const p = mustFind(state, state.active);
  if (p.points.length <= 1) {
    return {
      ...state,
      paths: state.paths.filter((x) => x.key !== state.active),
      active: null,
    };
  }
  return popLastPoint(state, p);
}

// Pops the last control point from path p (helper for removeLastPoint).
function popLastPoint(state, p) {
  const paths = state.paths.map((x) =>
    x.key === p.key ? { ...x, points: x.points.slice(0, -1) } : x);
  return { ...state, paths };
}

export function endActive(state) {
  if (!state.active) return state;
  const p = mustFind(state, state.active);
  if (p.points.length < 2) {
    // Can't confirm < 2 points (spec: Error handling) — discard.
    return { ...state, paths: state.paths.filter((x) => x.key !== p.key), active: null };
  }
  return { ...state, active: null };
}

export function selectPath(state, pathKey, { additive = false } = {}) {
  const selection = additive ? new Set(state.selection) : new Set();
  if (additive && selection.has(pathKey)) selection.delete(pathKey);
  else selection.add(pathKey);
  return { ...state, selection, selectedPoint: null };
}

export function clearSelection(state) {
  return { ...state, selection: new Set(), selectedPoint: null };
}

// Single-point selection: the anchor for extending a path. Also selects the
// owning path so radius/class/smooth ops target it. Ends any active draw
// first — otherwise a later Ctrl+click appends to the active path's tail
// instead of extending from the anchor. endActive discards 1-point paths,
// which can orphan the clicked point; that's a user action, not a wiring
// bug, so degrade to "nothing selected" rather than throwing.
export function selectPoint(state, pathKey, pointIdx) {
  const s = state.active ? endActive(state) : state;
  if (!s.paths.some((p) => p.key === pathKey)) {
    return { ...s, selection: new Set(), selectedPoint: null };
  }
  return {
    ...s,
    selection: new Set([pathKey]),
    selectedPoint: { pathKey, pointIdx },
  };
}

// Grow the selected point's path. Endpoints extend the run (before the head,
// after the tail) and the new point becomes the anchor so repeated extends
// keep walking outward. A middle point already has 2 connections — inserting
// there would reroute the run out and back through the new point, so branch
// a fresh path off the junction instead; it starts active so further
// Ctrl+clicks chain like normal drawing.
export function extendFromPoint(state, xyz) {
  const sel = state.selectedPoint;
  if (!sel) return state;
  const p = mustFind(state, sel.pathKey);
  if (sel.pointIdx > 0 && sel.pointIdx < p.points.length - 1) {
    const [key, s] = freshKey(state);
    const branch = {
      key,
      points: [[...p.points[sel.pointIdx]], xyz],
      radius: p.radius,
      smooth: false,
      classId: p.classId,
      instKey: key,        // own instance until merged, like any new path
    };
    return {
      ...s, paths: [...s.paths, branch], active: key,
      selection: new Set(), selectedPoint: null,
    };
  }
  const insertAt = sel.pointIdx === 0 ? 0 : sel.pointIdx + 1;
  const points = [...p.points.slice(0, insertAt), xyz, ...p.points.slice(insertAt)];
  const paths = state.paths.map((x) => x.key === p.key ? { ...x, points } : x);
  return { ...state, paths, selectedPoint: { pathKey: p.key, pointIdx: insertAt } };
}

// Radius/class/smooth target the active path while drawing, else the selection.
function targetKeys(state) {
  if (state.active) return new Set([state.active]);
  return state.selection;
}

const MIN_RADIUS = 0.005;

export function setRadius(state, radius) {
  const r = Math.max(radius, MIN_RADIUS);
  const keys = targetKeys(state);
  if (keys.size === 0) return state;
  const paths = state.paths.map((p) => keys.has(p.key) ? { ...p, radius: r } : p);
  return { ...state, paths, lastRadius: r };
}

export function nudgeRadius(state, dir) {
  const keys = targetKeys(state);
  const first = state.paths.find((p) => keys.has(p.key));
  if (!first) return state;
  // Multiplicative steps feel uniform across pipe sizes (8% like orbit zoom).
  return setRadius(state, first.radius * (1 + Math.sign(dir) * 0.08));
}

export function setClass(state, classId) {
  const keys = targetKeys(state);
  if (keys.size === 0) return state;
  const paths = state.paths.map((p) => keys.has(p.key) ? { ...p, classId } : p);
  return { ...state, paths };
}

export function toggleSmooth(state) {
  const keys = targetKeys(state);
  if (keys.size === 0) return state;
  const anyOff = state.paths.some((p) => keys.has(p.key) && !p.smooth);
  const paths = state.paths.map((p) => keys.has(p.key) ? { ...p, smooth: anyOff } : p);
  return { ...state, paths };
}

export function deleteSelected(state) {
  const paths = state.paths.filter((p) => !state.selection.has(p.key));
  return { ...state, paths, selection: new Set(), selectedPoint: null };
}

// Remove just the anchor point. A path can't survive on one point, so
// deleting from a 2-point path drops the whole path (same rule as
// removeLastPoint / endActive). The path stays selected so a follow-up
// click-⌫ sequence keeps working on it.
export function deleteSelectedPoint(state) {
  const sel = state.selectedPoint;
  if (!sel) return state;
  const p = mustFind(state, sel.pathKey);
  if (p.points.length <= 2) {
    return {
      ...state,
      paths: state.paths.filter((x) => x.key !== p.key),
      selection: new Set(),
      selectedPoint: null,
    };
  }
  const points = p.points.filter((_, i) => i !== sel.pointIdx);
  const paths = state.paths.map((x) => x.key === p.key ? { ...x, points } : x);
  return { ...state, paths, selectedPoint: null };
}

export function mergeSelection(state) {
  if (state.selection.size < 2) return state;
  const selectedGroups = [];          // ordered, unique instKeys of the selection
  for (const p of state.paths) {
    if (state.selection.has(p.key) && !selectedGroups.includes(p.instKey)) {
      selectedGroups.push(p.instKey);
    }
  }
  if (selectedGroups.length < 2) return state;
  // Survivor: lowest applied backend id wins (spec); else first selected group.
  const appliedGroups = selectedGroups.filter((g) => state.instanceIds[g] != null);
  const survivor = appliedGroups.length
    ? appliedGroups.reduce((m, g) => state.instanceIds[g] < state.instanceIds[m] ? g : m)
    : selectedGroups[0];
  const survivorClass = state.paths.find((p) => p.instKey === survivor).classId;
  const absorbed = selectedGroups.filter((g) => g !== survivor);
  const absorbedSet = new Set(absorbed);
  const absorbedIds = absorbed
    .map((g) => state.instanceIds[g])
    .filter((id) => id != null);
  const paths = state.paths.map((p) =>
    absorbedSet.has(p.instKey)
      ? { ...p, instKey: survivor, classId: survivorClass }
      : p);
  const instanceIds = { ...state.instanceIds };
  const pendingMergedFrom = { ...state.pendingMergedFrom };
  const carried = absorbed.flatMap((g) => pendingMergedFrom[g] ?? []);
  for (const g of absorbed) { delete instanceIds[g]; delete pendingMergedFrom[g]; }
  if (absorbedIds.length || carried.length) {
    pendingMergedFrom[survivor] = [
      ...(pendingMergedFrom[survivor] ?? []), ...absorbedIds, ...carried,
    ];
  }
  return { ...state, paths, instanceIds, pendingMergedFrom };
}

export function buildApplyCalls(state) {
  // Enter applies whole instances only (spec, workflow item 7).
  const groups = [];
  for (const p of state.paths) {
    if (state.selection.has(p.key) && !groups.includes(p.instKey)) groups.push(p.instKey);
  }
  return groups.map((g) => {
    const members = state.paths.filter((p) => p.instKey === g);
    return {
      instKey: g,
      paths: members.map((p) => ({ points: p.points, radius: p.radius, smooth: p.smooth })),
      // One class per instance — mergeSelection harmonizes classIds, so any
      // member's classId is the group's.
      classId: members[0].classId,
      targetInst: state.instanceIds[g] ?? -1,
      mergedFrom: state.pendingMergedFrom[g] ?? [],
    };
  });
}

export function markApplied(state, instKey, instanceId) {
  const pendingMergedFrom = { ...state.pendingMergedFrom };
  delete pendingMergedFrom[instKey];
  return {
    ...state,
    instanceIds: { ...state.instanceIds, [instKey]: instanceId },
    pendingMergedFrom,
  };
}

export function seedFromServer(state, doc) {
  let s = state;
  const groupByInstance = {};
  for (const sp of doc.paths ?? []) {
    let instKey = groupByInstance[sp.instance_id];
    const [key, next] = freshKey(s);
    s = next;
    if (!instKey) {
      instKey = key;
      groupByInstance[sp.instance_id] = instKey;
      s = { ...s, instanceIds: { ...s.instanceIds, [instKey]: sp.instance_id } };
    }
    s = {
      ...s,
      paths: [...s.paths, {
        key, points: sp.points, radius: sp.radius,
        smooth: !!sp.smooth, classId: sp.class_id, instKey,
      }],
    };
  }
  return s;
}
