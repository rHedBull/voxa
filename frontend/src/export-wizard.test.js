import { describe, it, expect } from 'vitest';
import { remapToTaxonomy, estimatePoints, pointsAfterFilters } from './export-wizard-util.js';

const CLASSES = [
  { class_id: 0, label: 'Pipe', color: '#22c55e' },
  { class_id: 1, label: 'Wall', color: '#8b5cf6' },
  { class_id: 2, label: 'Floor', color: '#64748b' },
];

describe('remapToTaxonomy', () => {
  it('merges classes into one target', () => {
    const tax = remapToTaxonomy(CLASSES, [{ from: [1, 2], to: { id: 9, label: 'Building', color: '#000' } }], null);
    expect(tax[9]).toEqual({ label: 'Building', color: '#000' });
    expect(tax[0]).toEqual({ label: 'Pipe', color: '#22c55e' }); // unmapped pass-through
    expect(tax[1]).toBeUndefined();  // merged away
    expect(tax[2]).toBeUndefined();
  });
  it('omits excluded classes (includeSet)', () => {
    const tax = remapToTaxonomy(CLASSES, [], new Set([0]));  // only Pipe
    expect(Object.keys(tax)).toEqual(['0']);
  });
});

describe('estimatePoints', () => {
  it('returns per-kind counts', () => {
    expect(estimatePoints({ kind: 'scan' }, 3_000_000, 156_000_000)).toBe(3_000_000);
    expect(estimatePoints({ kind: 'subsample', n: 500_000 }, 3_000_000, 156_000_000)).toBe(500_000);
    expect(estimatePoints({ kind: 'raw' }, 3_000_000, 156_000_000)).toBe(156_000_000);
  });
});

describe('pointsAfterFilters', () => {
  it('sums included classes, scaled to target density', () => {
    const counts = { 0: 1000, 1: 2000, 2: 3000 };  // 6000 at scan density
    // include only classes 0,1 => 3000 of 6000; target = 2x scan => ~ (3000)*2 ... but scale is target/scan on the SUM
    expect(pointsAfterFilters(counts, new Set([0, 1]), 12000, 6000)).toBe(6000); // 3000 * (12000/6000)
    expect(pointsAfterFilters(counts, null, 6000, 6000)).toBe(6000);  // all, no scaling
    expect(pointsAfterFilters(counts, new Set([]), 6000, 6000)).toBe(0); // none
  });
});
