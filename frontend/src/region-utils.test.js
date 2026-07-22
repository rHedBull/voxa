import { describe, expect, it } from 'vitest';
import { majorityInstances, unlabeledPct, regionCssColor, reviewPct, REGION_COLORS, REVIEW_BUDGET_PCT } from './region-utils.js';

const stat = {
  id: 1, n_points: 200, n_unlabeled: 30,
  instances: { 7: { inside: 90, total: 100 }, 8: { inside: 40, total: 100 }, 9: { inside: 60, total: 100 } },
};
const instances = [
  { id: 'a', segId: 7, confirmed: true, label: 'pump' },
  { id: 'b', segId: 8, confirmed: true, label: 'pipe' },   // 40% — not majority
  { id: 'c', segId: 9, confirmed: false, label: 'tank' },  // unconfirmed — excluded
];

describe('majorityInstances', () => {
  it('keeps only confirmed instances with >50% of points inside, sorted by fraction', () => {
    const out = majorityInstances(stat, instances);
    expect(out.map((o) => o.inst.segId)).toEqual([7]);
    expect(out[0].frac).toBeCloseTo(0.9);
  });
  it('handles missing stats and empty inputs', () => {
    expect(majorityInstances(null, instances)).toEqual([]);
    expect(majorityInstances(stat, [])).toEqual([]);
    expect(majorityInstances({ ...stat, instances: {} }, instances)).toEqual([]);
  });
  it('ignores instances without a finite segId (legacy cuboids)', () => {
    expect(majorityInstances(stat, [{ id: 'x', segId: null, confirmed: true }])).toEqual([]);
  });
});

describe('unlabeledPct', () => {
  it('returns the unlabeled percentage', () => {
    expect(unlabeledPct(stat)).toBeCloseTo(15);
  });
  it('returns null for empty or missing regions', () => {
    expect(unlabeledPct(null)).toBeNull();
    expect(unlabeledPct({ n_points: 0, n_unlabeled: 0 })).toBeNull();
  });
});

describe('regionCssColor', () => {
  it('maps statuses to css colors matching REGION_COLORS', () => {
    expect(regionCssColor('draft')).toBe('#f59e0b');
    expect(regionCssColor('eval_grade')).toBe('#22c55e');
    expect(REGION_COLORS.draft).toBe(0xf59e0b);
    expect(REGION_COLORS.eval_grade).toBe(0x22c55e);
  });
});

describe('reviewPct (phase 2 budget)', () => {
  it('is null without stats and 0 when nothing is marked', () => {
    expect(reviewPct(null)).toBeNull();
    expect(reviewPct({ n_points: 0, n_review: 0 })).toBeNull();
    expect(reviewPct({ n_points: 100 })).toBe(0);
  });

  it('reports the excluded-review share', () => {
    expect(reviewPct({ n_points: 200, n_review: 8 })).toBe(4);
    expect(REVIEW_BUDGET_PCT).toBe(3);
  });
});
