import { describe, it, expect } from 'vitest';
import { initSegState, applyDelta, recomputeSummary, computeDiffMask } from './segment-state.js';

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

describe('computeDiffMask', () => {
  it('marks points where class changed', () => {
    const cls     = new Int8Array([0, 1, 2, 0]);
    const pre_cls = new Int8Array([0, 0, 2, 0]);
    const inst     = new Int32Array([0, 1, 2, 3]);
    const pre_inst = new Int32Array([0, 1, 2, 3]);
    const mask = computeDiffMask(cls, pre_cls, inst, pre_inst);
    expect(mask[0]).toBe(0);
    expect(mask[1]).toBe(1);
    expect(mask[2]).toBe(0);
    expect(mask[3]).toBe(0);
  });

  it('marks points where instance changed', () => {
    const cls      = new Int8Array([1, 1, 1]);
    const pre_cls  = new Int8Array([1, 1, 1]);
    const inst     = new Int32Array([0, 99, 2]);
    const pre_inst = new Int32Array([0,  1, 2]);
    const mask = computeDiffMask(cls, pre_cls, inst, pre_inst);
    expect(mask[0]).toBe(0);
    expect(mask[1]).toBe(1);
    expect(mask[2]).toBe(0);
  });

  it('returns all-zero mask when nothing changed', () => {
    const cls  = new Int8Array([0, 1, 2]);
    const inst = new Int32Array([0, 1, 2]);
    const mask = computeDiffMask(cls, cls, inst, inst);
    expect(Array.from(mask)).toEqual([0, 0, 0]);
  });

  it('returns a Uint8Array of the same length as the input', () => {
    const cls  = new Int8Array(5).fill(0);
    const inst = new Int32Array(5).fill(0);
    const mask = computeDiffMask(cls, cls, inst, inst);
    expect(mask).toBeInstanceOf(Uint8Array);
    expect(mask.length).toBe(5);
  });
});
