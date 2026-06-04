import { describe, expect, it } from 'vitest';
import {
  initDrawState, addPoint, movePoint, removeLastPoint, endActive,
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
