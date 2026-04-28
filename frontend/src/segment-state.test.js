import { describe, it, expect } from 'vitest';
import { initSegState, applyDelta, recomputeSummary } from './segment-state.js';

const seed = () => initSegState({
  classFull: new Int8Array([-1, 0, 0, 1, 1, 2, -1, 2]),
  instanceFull: new Int32Array([-1, 0, 0, 1, 1, 2, -1, 2]),
  isFromPrelabel: true,
});

describe('segment-state', () => {
  it('applyDelta patches arrays at given indices', () => {
    const s = seed();
    const next = applyDelta(s, {
      indices: new Int32Array([1, 2]),
      after_class: new Int8Array([2, 2]),
      after_instance: new Int32Array([0, 0]),
    });
    expect(next.classFull[1]).toBe(2);
    expect(next.classFull[2]).toBe(2);
    expect(next.instanceFull[1]).toBe(0);
    expect(next.dirty).toBe(true);
  });

  it('recomputeSummary derives per-instance counts', () => {
    const s = seed();
    const sum = recomputeSummary(s);
    expect(sum.get(0).nPoints).toBe(2);
    expect(sum.get(1).classId).toBe(1);
  });
});
