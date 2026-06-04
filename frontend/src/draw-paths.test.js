import { describe, expect, it } from 'vitest';
import {
  initDrawState, addPoint, movePoint, removeLastPoint, endActive,
  selectPath, clearSelection, setRadius, nudgeRadius, setClass,
  toggleSmooth, deleteSelected,
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
