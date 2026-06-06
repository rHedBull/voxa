import { describe, expect, it } from 'vitest';
import {
  initDrawState, addPoint, movePoint, removeLastPoint, endActive,
  selectPath, clearSelection, selectPoint, extendFromPoint, deleteSelectedPoint,
  setRadius, nudgeRadius, setClass,
  toggleSmooth, deleteSelected,
  mergeSelection, buildApplyCalls, markApplied, seedFromServer,
} from './draw-paths.js';

const P = (x = 0) => [x, 0, 0];

describe('drawing core', () => {
  it('first addPoint starts a path; subsequent ones chain', () => {
    let s = initDrawState({ defaultRadius: 0.2 });
    s = addPoint(s, P(0), 1);
    expect(s.active).not.toBeNull();
    expect(s.paths).toHaveLength(1);
    expect(s.paths[0].classId).toBe(1);
    expect(s.paths[0].radius).toBe(0.2);
    s = addPoint(s, P(1), 1);
    expect(s.paths).toHaveLength(1);
    expect(s.paths[0].points).toEqual([P(0), P(1)]);
  });

  it('movePoint replaces a control point', () => {
    let s = addPoint(initDrawState(), P(0), 0);
    s = addPoint(s, P(1), 0);
    s = movePoint(s, s.paths[0].key, 1, [1, 2, 3]);
    expect(s.paths[0].points[1]).toEqual([1, 2, 3]);
  });

  it('removeLastPoint pops; removing the only point discards the path', () => {
    let s = addPoint(initDrawState(), P(0), 0);
    s = addPoint(s, P(1), 0);
    s = removeLastPoint(s);
    expect(s.paths[0].points).toEqual([P(0)]);
    s = removeLastPoint(s);
    expect(s.paths).toHaveLength(0);
    expect(s.active).toBeNull();
  });

  it('endActive stages a valid path, discards a 1-point path', () => {
    let s = addPoint(initDrawState(), P(0), 0);
    s = endActive(s);                       // 1 point → discard
    expect(s.paths).toHaveLength(0);
    s = addPoint(s, P(0), 0);
    s = addPoint(s, P(1), 0);
    s = endActive(s);
    expect(s.active).toBeNull();
    expect(s.paths).toHaveLength(1);
  });

  it('new path after endActive reuses lastRadius and gets a fresh instKey', () => {
    let s = addPoint(initDrawState({ defaultRadius: 0.1 }), P(0), 0);
    s = addPoint(s, P(1), 0);
    s = endActive(s);
    s = addPoint(s, P(5), 0);
    expect(s.paths[1].radius).toBe(0.1);
    expect(s.paths[1].instKey).not.toBe(s.paths[0].instKey);
  });
});

function staged2() {
  let s = initDrawState();
  s = addPoint(s, P(0), 0); s = addPoint(s, P(1), 0); s = endActive(s);
  s = addPoint(s, P(5), 1); s = addPoint(s, P(6), 1); s = endActive(s);
  return s;
}

describe('selection + path edits', () => {
  it('selectPath replaces; additive=true toggles', () => {
    let s = staged2();
    const [a, b] = s.paths.map((p) => p.key);
    s = selectPath(s, a);
    expect([...s.selection]).toEqual([a]);
    s = selectPath(s, b, { additive: true });
    expect(s.selection.size).toBe(2);
    s = selectPath(s, b, { additive: true });   // shift-click again removes
    expect([...s.selection]).toEqual([a]);
    s = selectPath(s, b);                       // plain click replaces
    expect([...s.selection]).toEqual([b]);
    expect(clearSelection(s).selection.size).toBe(0);
  });

  it('setRadius / nudgeRadius hit selected paths (or active) and update lastRadius', () => {
    let s = staged2();
    const a = s.paths[0].key;
    s = selectPath(s, a);
    s = setRadius(s, 0.5);
    expect(s.paths[0].radius).toBe(0.5);
    expect(s.paths[1].radius).not.toBe(0.5);
    expect(s.lastRadius).toBe(0.5);
    s = nudgeRadius(s, +1);
    expect(s.paths[0].radius).toBeGreaterThan(0.5);
    // While drawing, the active path is implicitly targeted.
    let d = addPoint(initDrawState({ defaultRadius: 0.2 }), P(0), 0);
    d = nudgeRadius(d, -1);
    expect(d.paths[0].radius).toBeLessThan(0.2);
    expect(d.paths[0].radius).toBeGreaterThan(0);   // floor > 0
  });

  it('setClass and toggleSmooth target the selection', () => {
    let s = staged2();
    s = selectPath(s, s.paths[0].key);
    s = setClass(s, 3);
    expect(s.paths[0].classId).toBe(3);
    expect(s.paths[1].classId).toBe(1);
    s = toggleSmooth(s);
    expect(s.paths[0].smooth).toBe(true);
  });

  it('deleteSelected removes staged paths and clears selection', () => {
    let s = staged2();
    s = selectPath(s, s.paths[0].key);
    s = deleteSelected(s);
    expect(s.paths).toHaveLength(1);
    expect(s.selection.size).toBe(0);
  });
});

describe('point selection + extend', () => {
  it('selectPoint selects the point and its path; selectPath/clearSelection drop it', () => {
    let s = staged2();
    const a = s.paths[0].key;
    s = selectPoint(s, a, 1);
    expect(s.selectedPoint).toEqual({ pathKey: a, pointIdx: 1 });
    expect([...s.selection]).toEqual([a]);
    expect(selectPath(s, s.paths[1].key).selectedPoint).toBeNull();
    expect(clearSelection(s).selectedPoint).toBeNull();
  });

  it('extendFromPoint appends after a tail point and reselects the new point', () => {
    let s = staged2();
    const a = s.paths[0].key;
    s = selectPoint(s, a, 1);             // last point of a 2-point path
    s = extendFromPoint(s, P(2));
    expect(s.paths[0].points).toEqual([P(0), P(1), P(2)]);
    expect(s.selectedPoint).toEqual({ pathKey: a, pointIdx: 2 });
    s = extendFromPoint(s, P(3));         // chained extends keep walking the tail
    expect(s.paths[0].points).toEqual([P(0), P(1), P(2), P(3)]);
  });

  it('extendFromPoint prepends before the head point', () => {
    let s = staged2();
    const a = s.paths[0].key;
    s = selectPoint(s, a, 0);
    s = extendFromPoint(s, P(-1));
    expect(s.paths[0].points).toEqual([P(-1), P(0), P(1)]);
    expect(s.selectedPoint).toEqual({ pathKey: a, pointIdx: 0 });
  });

  it('extendFromPoint branches from a middle point instead of rerouting the run', () => {
    let s = initDrawState({ defaultRadius: 0.2 });
    s = addPoint(s, P(0), 2); s = addPoint(s, P(1), 2); s = addPoint(s, P(2), 2);
    s = endActive(s);
    s = setRadius(selectPath(s, s.paths[0].key), 0.4);
    s = selectPoint(s, s.paths[0].key, 1);   // middle point — 2 connections
    s = extendFromPoint(s, P(9));
    // Trunk untouched; a new active branch runs junction → click.
    expect(s.paths[0].points).toEqual([P(0), P(1), P(2)]);
    expect(s.paths).toHaveLength(2);
    const branch = s.paths[1];
    expect(branch.points).toEqual([P(1), P(9)]);
    expect(branch.classId).toBe(2);
    expect(branch.radius).toBe(0.4);
    expect(s.active).toBe(branch.key);       // further Ctrl+clicks keep drawing it
    expect(s.selectedPoint).toBeNull();
  });

  it('selectPoint while drawing ends the active path so the anchor wins over append', () => {
    let s = addPoint(initDrawState(), P(0), 0);
    s = addPoint(s, P(1), 0);
    s = selectPoint(s, s.paths[0].key, 0);   // click the head while still drawing
    expect(s.active).toBeNull();
    s = extendFromPoint(s, P(-1));           // Ctrl+click left of the head
    expect(s.paths[0].points).toEqual([P(-1), P(0), P(1)]);
  });

  it('selectPoint on the only point of an active path discards it gracefully', () => {
    let s = addPoint(initDrawState(), P(0), 0);
    s = selectPoint(s, s.paths[0].key, 0);   // endActive discards 1-pt paths
    expect(s.paths).toHaveLength(0);
    expect(s.selectedPoint).toBeNull();
    expect(s.active).toBeNull();
  });

  it('extendFromPoint without a selected point is a no-op; new path clears it', () => {
    let s = staged2();
    expect(extendFromPoint(s, P(9))).toBe(s);
    s = selectPoint(s, s.paths[0].key, 0);
    s = addPoint(s, P(9), 0);             // starting a new path drops point selection
    expect(s.selectedPoint).toBeNull();
  });

  it('deleteSelected clears the selected point with its path', () => {
    let s = staged2();
    s = selectPoint(s, s.paths[0].key, 0);
    s = deleteSelected(s);
    expect(s.selectedPoint).toBeNull();
  });

  it('deleteSelectedPoint removes only the anchor point, keeping the path', () => {
    let s = initDrawState();
    s = addPoint(s, P(0), 0); s = addPoint(s, P(1), 0); s = addPoint(s, P(2), 0);
    s = endActive(s);
    const a = s.paths[0].key;
    s = selectPoint(s, a, 1);
    s = deleteSelectedPoint(s);
    expect(s.paths[0].points).toEqual([P(0), P(2)]);
    expect(s.selectedPoint).toBeNull();
    expect([...s.selection]).toEqual([a]);    // path stays selected
  });

  it('deleteSelectedPoint drops a 2-point path entirely (cannot survive 1 point)', () => {
    let s = staged2();
    s = selectPoint(s, s.paths[0].key, 0);
    s = deleteSelectedPoint(s);
    expect(s.paths).toHaveLength(1);
    expect(s.selection.size).toBe(0);
    expect(s.selectedPoint).toBeNull();
  });

  it('deleteSelectedPoint without an anchor is a no-op', () => {
    const s = staged2();
    expect(deleteSelectedPoint(s)).toBe(s);
  });
});

function applied(state, instKeyToId) {
  let s = state;
  for (const [k, id] of Object.entries(instKeyToId)) s = markApplied(s, k, id);
  return s;
}

describe('merge + apply calls', () => {
  it('buildApplyCalls: one call per selected instance group, whole-instance expansion', () => {
    let s = staged2();
    const [a, b] = s.paths.map((p) => p.key);
    s = selectPath(s, a);
    s = selectPath(s, b, { additive: true });
    // Unmerged → two independent calls (spec: M is the only merge trigger).
    const calls = buildApplyCalls(s);
    expect(calls).toHaveLength(2);
    expect(calls[0].targetInst).toBe(-1);
    expect(calls[0].mergedFrom).toEqual([]);
    expect(calls[0].paths[0].points).toEqual([P(0), P(1)]);
  });

  it('merge of two staged paths → one call, one shared class', () => {
    let s = staged2();
    const [a, b] = s.paths.map((p) => p.key);
    s = selectPath(s, a); s = selectPath(s, b, { additive: true });
    s = mergeSelection(s);
    expect(s.paths[0].instKey).toBe(s.paths[1].instKey);
    expect(s.paths[1].classId).toBe(s.paths[0].classId);   // survivor's class
    const calls = buildApplyCalls(s);
    expect(calls).toHaveLength(1);
    expect(calls[0].paths).toHaveLength(2);
  });

  it('selecting one path of a multi-path instance expands to all its paths', () => {
    let s = staged2();
    const [a, b] = s.paths.map((p) => p.key);
    s = selectPath(s, a); s = selectPath(s, b, { additive: true });
    s = mergeSelection(s);
    s = clearSelection(s);
    s = selectPath(s, a);                       // only one sibling selected
    const calls = buildApplyCalls(s);
    expect(calls[0].paths).toHaveLength(2);     // whole instance anyway
  });

  it('applied-applied merge: lowest id survives, others go to mergedFrom', () => {
    let s = staged2();
    const [a, b] = s.paths.map((p) => p.key);
    s = applied(s, { [a]: 9, [b]: 4 });
    s = selectPath(s, a); s = selectPath(s, b, { additive: true });
    s = mergeSelection(s);
    const calls = buildApplyCalls(s);
    expect(calls).toHaveLength(1);
    expect(calls[0].targetInst).toBe(4);        // lowest survives
    expect(calls[0].mergedFrom).toEqual([9]);
  });

  it('markApplied records id and clears pendingMergedFrom', () => {
    let s = staged2();
    const [a, b] = s.paths.map((p) => p.key);
    s = applied(s, { [a]: 9, [b]: 4 });
    s = selectPath(s, a); s = selectPath(s, b, { additive: true });
    s = mergeSelection(s);
    const survivor = s.paths[0].instKey;
    s = markApplied(s, survivor, 4);
    expect(s.instanceIds[survivor]).toBe(4);
    expect(s.pendingMergedFrom[survivor] ?? []).toEqual([]);
    // Re-apply after edit: same target, no mergedFrom this time.
    s = selectPath(clearSelection(s), s.paths[0].key);
    const again = buildApplyCalls(s);
    expect(again[0].targetInst).toBe(4);
    expect(again[0].mergedFrom).toEqual([]);
  });

  it('staged + applied merge adopts the applied id', () => {
    let s = staged2();
    const [a, b] = s.paths.map((p) => p.key);
    s = applied(s, { [a]: 7 });
    s = selectPath(s, a); s = selectPath(s, b, { additive: true });
    s = mergeSelection(s);
    const calls = buildApplyCalls(s);
    expect(calls[0].targetInst).toBe(7);
    expect(calls[0].mergedFrom).toEqual([]);
  });

  it('seedFromServer rebuilds applied groups keyed by instance_id', () => {
    const doc = { paths: [
      { points: [[0, 0, 0], [1, 0, 0]], radius: 0.2, smooth: false, class_id: 0, instance_id: 7 },
      { points: [[2, 0, 0], [3, 0, 0]], radius: 0.2, smooth: true,  class_id: 0, instance_id: 7 },
      { points: [[9, 0, 0], [9, 1, 0]], radius: 0.4, smooth: false, class_id: 2, instance_id: 8 },
    ] };
    const s = seedFromServer(initDrawState(), doc);
    expect(s.paths).toHaveLength(3);
    const groups = new Set(s.paths.map((p) => p.instKey));
    expect(groups.size).toBe(2);
    const g7 = s.paths.filter((p) => s.instanceIds[p.instKey] === 7);
    expect(g7).toHaveLength(2);
    expect(g7[1].smooth).toBe(true);
  });
});
