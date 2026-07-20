import { describe, it, expect } from 'vitest';
import { fitEligibility } from './fit-eligibility.js';

describe('fitEligibility', () => {
  it('preseg/sam eligible iff selection non-empty', () => {
    expect(fitEligibility({ list: 'preseg', selectionSize: 2 }).eligible).toBe(true);
    expect(fitEligibility({ list: 'preseg', selectionSize: 0 }).eligible).toBe(false);
    expect(fitEligibility({ list: 'sam', selectionSize: 1 }).eligible).toBe(true);
    expect(fitEligibility({ list: 'sam', selectionSize: 0 }).eligible).toBe(false);
  });
  it('instance eligible when selected, even if confirmed (diverges from cut)', () => {
    expect(fitEligibility({ list: 'instance', isSelected: true, confirmed: false }).eligible).toBe(true);
    expect(fitEligibility({ list: 'instance', isSelected: true, confirmed: true }).eligible).toBe(true);
    expect(fitEligibility({ list: 'instance', isSelected: false }).eligible).toBe(false);
  });
  it('throws on unknown list', () => {
    expect(() => fitEligibility({ list: 'nope' })).toThrow();
  });
});
