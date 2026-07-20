import { describe, it, expect } from 'vitest';
import { pointsInsidePrism, pointInPolygonXZ } from './prism-geom.js';

// Shared parity fixture — MUST match backend test_shapes.py exactly. Edges at
// 0.5/3.5 keep the x,z in {1,2,3} grid block strictly interior (on-edge points
// are excluded by even-odd ray-cast, so a [1,3] square would give 4, not 9).
const SQUARE = { polygon: [[0.5, 0.5], [3.5, 0.5], [3.5, 3.5], [0.5, 3.5]], y0: 0, height: 1 };

function gridXZ(ys) {
  const out = [];
  for (const y of ys) for (let x = 0; x < 5; x++) for (let z = 0; z < 5; z++) out.push(x, y, z);
  return new Float32Array(out);
}

describe('pointsInsidePrism', () => {
  it('selects the 3x3 block inside a square footprint within the Y-band', () => {
    const pos = gridXZ([0, 5]);
    const idx = pointsInsidePrism(pos, null, SQUARE);
    const expect9 = [];
    for (const x of [1, 2, 3]) for (const z of [1, 2, 3]) expect9.push(x * 5 + z);
    expect([...idx].sort((a, b) => a - b)).toEqual(expect9);
  });

  it('supports concave (L) footprints', () => {
    const pos = gridXZ([0]);
    const L = { polygon: [[0, 0], [4, 0], [4, 2], [2, 2], [2, 4], [0, 4]], y0: -0.5, height: 1 };
    const idx = new Set(pointsInsidePrism(pos, null, L));
    expect(idx.has(3 * 5 + 3)).toBe(false); // notch excluded
    expect(idx.has(1 * 5 + 3)).toBe(true);
    expect(idx.has(3 * 5 + 1)).toBe(true);
  });

  it('returns nothing for < 3 vertices or zero height', () => {
    const pos = gridXZ([0]);
    expect(pointsInsidePrism(pos, null, { polygon: [[1, 1], [2, 2]], y0: 0, height: 1 })).toEqual([]);
    expect(pointsInsidePrism(pos, null, { ...SQUARE, height: 0 })).toEqual([]);
  });

  it('pointInPolygonXZ matches the ray-cast rule', () => {
    const sq = [[0, 0], [4, 0], [4, 4], [0, 4]];
    expect(pointInPolygonXZ(2, 2, sq)).toBe(true);
    expect(pointInPolygonXZ(5, 2, sq)).toBe(false);
  });
});
