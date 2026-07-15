import { describe, it, expect } from 'vitest';
import { buildCutCloud, removeCutPoints } from './cut-mode.jsx';

// 6 full-res points: instanceFull tags 0/0/1/1/-1/-1, samIds tags -1/-1/-1/-1/7/7.
function makeSegState() {
  return {
    instanceFull: new Int32Array([0, 0, 1, 1, -1, -1]),
    samIds: new Int32Array([-1, -1, -1, -1, 7, 7]),
  };
}

function makeCloud(subsampleIdx = null) {
  const n = 6;
  const positions = new Float32Array(n * 3);
  const colors = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    positions[i * 3] = i; positions[i * 3 + 1] = i * 10; positions[i * 3 + 2] = i * 100;
    colors[i * 3] = i / 10; colors[i * 3 + 1] = i / 10; colors[i * 3 + 2] = i / 10;
  }
  return { positions, colors, subsampleIdx };
}

describe('buildCutCloud', () => {
  it('filters to only the given single preseg source', () => {
    const out = buildCutCloud(makeSegState(), makeCloud(), [{ kind: 'preseg', segId: 0 }]);
    expect(out.positions.length / 3).toBe(2);
    expect(Array.from(out.fullIndices)).toEqual([0, 1]);
    expect(out.tags).toEqual([{ kind: 'preseg', segId: 0 }, { kind: 'preseg', segId: 0 }]);
  });

  it('filters to only the given sam source', () => {
    const out = buildCutCloud(makeSegState(), makeCloud(), [{ kind: 'sam', segId: 7 }]);
    expect(Array.from(out.fullIndices)).toEqual([4, 5]);
    expect(out.tags.every((t) => t.kind === 'sam' && t.segId === 7)).toBe(true);
  });

  it('handles a mix of preseg + sam sources without merging tags', () => {
    const out = buildCutCloud(makeSegState(), makeCloud(),
      [{ kind: 'preseg', segId: 0 }, { kind: 'sam', segId: 7 }]);
    expect(Array.from(out.fullIndices)).toEqual([0, 1, 4, 5]);
    expect(out.tags).toEqual([
      { kind: 'preseg', segId: 0 },
      { kind: 'preseg', segId: 0 },
      { kind: 'sam', segId: 7 },
      { kind: 'sam', segId: 7 },
    ]);
  });

  it('handles a single instance source (instanceFull namespace)', () => {
    const out = buildCutCloud(makeSegState(), makeCloud(), [{ kind: 'instance', segId: 1 }]);
    expect(Array.from(out.fullIndices)).toEqual([2, 3]);
    expect(out.tags.every((t) => t.kind === 'instance' && t.segId === 1)).toBe(true);
  });

  it('excludes points belonging to sources not in the list', () => {
    const out = buildCutCloud(makeSegState(), makeCloud(), [{ kind: 'preseg', segId: 0 }]);
    expect(Array.from(out.fullIndices)).not.toContain(2);
    expect(Array.from(out.fullIndices)).not.toContain(4);
  });

  it('honors cloud.subsampleIdx when the render cloud is subsampled', () => {
    // Sub-cloud has 3 points mapping to full indices [1, 3, 5].
    const cloud = makeCloud(new Int32Array([1, 3, 5]));
    cloud.positions = new Float32Array([1, 10, 100, 3, 30, 300, 5, 50, 500]);
    cloud.colors = new Float32Array(9);
    const out = buildCutCloud(makeSegState(), cloud, [{ kind: 'preseg', segId: 0 }]);
    // Only sub-point 0 (full idx 1) belongs to preseg 0.
    expect(Array.from(out.fullIndices)).toEqual([1]);
  });

  it('preserves point position/color values for kept points', () => {
    const out = buildCutCloud(makeSegState(), makeCloud(), [{ kind: 'preseg', segId: 0 }]);
    expect(Array.from(out.positions)).toEqual([0, 0, 0, 1, 10, 100]);
  });
});

describe('removeCutPoints', () => {
  it('drops points whose full index is in the removed set', () => {
    const filtered = buildCutCloud(makeSegState(), makeCloud(),
      [{ kind: 'preseg', segId: 0 }, { kind: 'preseg', segId: 1 }]);
    const next = removeCutPoints(filtered, new Set([0]));
    expect(Array.from(next.fullIndices)).toEqual([1, 2, 3]);
  });

  it('is a no-op (same reference) when nothing to remove', () => {
    const filtered = buildCutCloud(makeSegState(), makeCloud(), [{ kind: 'preseg', segId: 0 }]);
    expect(removeCutPoints(filtered, new Set())).toBe(filtered);
    expect(removeCutPoints(filtered, null)).toBe(filtered);
  });

  it('removes multiple points across sources without touching the rest', () => {
    const filtered = buildCutCloud(makeSegState(), makeCloud(),
      [{ kind: 'preseg', segId: 0 }, { kind: 'sam', segId: 7 }]);
    const next = removeCutPoints(filtered, new Set([1, 5]));
    expect(Array.from(next.fullIndices)).toEqual([0, 4]);
    expect(next.tags).toEqual([{ kind: 'preseg', segId: 0 }, { kind: 'sam', segId: 7 }]);
  });
});
