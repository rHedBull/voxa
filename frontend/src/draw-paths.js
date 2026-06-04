// draw-paths.js — pure state machine for the Draw (centerline) sub-mode.
// All functions take state and return a new state; React owns the object.
// See docs/superpowers/specs/2026-06-04-centerline-pipe-labeling-design.md.
//
// State shape:
// {
//   paths: [{ key, points: [[x,y,z],...], radius, smooth, classId, instKey }],
//   active: string|null,          // key of the path being drawn
//   selection: Set<string>,       // selected path keys
//   instanceIds: { [instKey]: number },        // backend ids, applied groups only
//   pendingMergedFrom: { [instKey]: number[] },// absorbed backend ids awaiting next apply
//   lastRadius: number,
//   nextKey: number,              // monotonic key source (deterministic, testable)
// }
// classId is the int palette index (== position in the classes array).
// instKey groups paths into one instance; starts unique per path, M merges.

export function initDrawState({ defaultRadius = 0.15 } = {}) {
  return {
    paths: [],
    active: null,
    selection: new Set(),
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
  return { ...s, paths: [...s.paths, path], active: key, selection: new Set() };
}

export function movePoint(state, pathKey, pointIdx, xyz) {
  const paths = state.paths.map((p) => {
    if (p.key !== pathKey) return p;
    const points = p.points.slice();
    points[pointIdx] = xyz;
    return { ...p, points };
  });
  return { ...state, paths };
}

export function removeLastPoint(state) {
  if (!state.active) return state;
  const p = state.paths.find((x) => x.key === state.active);
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
  const p = state.paths.find((x) => x.key === state.active);
  if (p.points.length < 2) {
    // Can't confirm < 2 points (spec: Error handling) — discard.
    return { ...state, paths: state.paths.filter((x) => x.key !== p.key), active: null };
  }
  return { ...state, active: null };
}
