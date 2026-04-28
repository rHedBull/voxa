import { describe, it, expect } from 'vitest';
import { evtToNdc, buildFullToSubMap } from './viewer.jsx';

describe('evtToNdc', () => {
  it('maps centre of rect to (0, 0)', () => {
    const rect = { left: 100, top: 50, width: 800, height: 600 };
    const evt  = { clientX: 500, clientY: 350 };
    const ndc  = evtToNdc(evt, rect);
    expect(ndc.x).toBeCloseTo(0, 5);
    expect(ndc.y).toBeCloseTo(0, 5);
  });

  it('maps top-left corner to (-1, +1)', () => {
    const rect = { left: 0, top: 0, width: 400, height: 300 };
    const evt  = { clientX: 0, clientY: 0 };
    const ndc  = evtToNdc(evt, rect);
    expect(ndc.x).toBeCloseTo(-1, 5);
    expect(ndc.y).toBeCloseTo(1, 5);
  });

  it('maps bottom-right corner to (+1, -1)', () => {
    const rect = { left: 0, top: 0, width: 400, height: 300 };
    const evt  = { clientX: 400, clientY: 300 };
    const ndc  = evtToNdc(evt, rect);
    expect(ndc.x).toBeCloseTo(1, 5);
    expect(ndc.y).toBeCloseTo(-1, 5);
  });
});

describe('buildFullToSubMap', () => {
  it('maps each subsampled full index to its sub row', () => {
    // subsampleIdx: [fullIdx0, fullIdx1, ...] — sub row i points to full index subsampleIdx[i]
    // full cloud has 10 points; 4 were sampled at rows 2, 5, 7, 9
    const subsampleIdx = new Int32Array([2, 5, 7, 9]);
    const map = buildFullToSubMap(subsampleIdx, 10);
    expect(map[2]).toBe(0);
    expect(map[5]).toBe(1);
    expect(map[7]).toBe(2);
    expect(map[9]).toBe(3);
  });

  it('fills non-sampled indices with -1', () => {
    const subsampleIdx = new Int32Array([1, 4]);
    const map = buildFullToSubMap(subsampleIdx, 6);
    expect(map[0]).toBe(-1);
    expect(map[2]).toBe(-1);
    expect(map[3]).toBe(-1);
    expect(map[5]).toBe(-1);
  });

  it('returns an Int32Array of length fullN', () => {
    const subsampleIdx = new Int32Array([0, 2]);
    const map = buildFullToSubMap(subsampleIdx, 5);
    expect(map).toBeInstanceOf(Int32Array);
    expect(map.length).toBe(5);
  });

  it('handles identity (no subsampling) when all indices are included', () => {
    const subsampleIdx = new Int32Array([0, 1, 2]);
    const map = buildFullToSubMap(subsampleIdx, 3);
    expect(Array.from(map)).toEqual([0, 1, 2]);
  });
});
