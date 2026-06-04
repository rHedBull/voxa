import { describe, it, expect } from 'vitest';
import { deriveFastQueue, stepIndex } from './fast-label.jsx';

const summary = (entries) => new Map(entries);

describe('deriveFastQueue', () => {
  it('sorts largest first with id tiebreak', () => {
    const q = deriveFastQueue(summary([
      [3, { classId: -1, nPoints: 10 }],
      [1, { classId: 0, nPoints: 500 }],
      [7, { classId: -1, nPoints: 500 }],
      [2, { classId: 2, nPoints: 99 }],
    ]), new Set());
    expect(q.map((s) => s.id)).toEqual([1, 7, 2, 3]);
  });

  it('excludes promoted segIds and negative ids', () => {
    const q = deriveFastQueue(summary([
      [-1, { classId: -1, nPoints: 9999 }],
      [4, { classId: 0, nPoints: 50 }],
      [5, { classId: 0, nPoints: 40 }],
    ]), new Set([4]));
    expect(q.map((s) => s.id)).toEqual([5]);
  });

  it('handles missing summary / empty promoted set', () => {
    expect(deriveFastQueue(null, new Set())).toEqual([]);
    expect(deriveFastQueue(summary([[1, { classId: 0, nPoints: 5 }]]), null))
      .toHaveLength(1);
  });

  it('carries nPoints and classId through', () => {
    const q = deriveFastQueue(summary([[9, { classId: 3, nPoints: 123 }]]), new Set());
    expect(q[0]).toEqual({ id: 9, classId: 3, nPoints: 123 });
  });
});

describe('stepIndex', () => {
  it('steps forward and back', () => {
    expect(stepIndex(5, 0, 1)).toBe(1);
    expect(stepIndex(5, 3, -1)).toBe(2);
  });

  it('wraps at both ends', () => {
    expect(stepIndex(5, 4, 1)).toBe(0);
    expect(stepIndex(5, 0, -1)).toBe(4);
  });

  it('clamps an out-of-range cursor before stepping (shrunken queue)', () => {
    // pos 9 in a queue that shrank to 4: clamp to 3, then step.
    expect(stepIndex(4, 9, 1)).toBe(0);   // 3 -> wrap to 0
    expect(stepIndex(4, 9, -1)).toBe(2);
  });

  it('is safe on an empty queue', () => {
    expect(stepIndex(0, 3, 1)).toBe(0);
  });
});
