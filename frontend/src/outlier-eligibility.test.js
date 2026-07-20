// frontend/src/outlier-eligibility.test.js
import { describe, it, expect } from 'vitest';
import { removeOutliersEligibility } from './outlier-eligibility.js';

describe('removeOutliersEligibility', () => {
  it('sam: eligible iff exactly one candidate is selected', () => {
    expect(removeOutliersEligibility({ list: 'sam', selectionSize: 1 }).eligible).toBe(true);
    expect(removeOutliersEligibility({ list: 'sam', selectionSize: 0 }).eligible).toBe(false);
    expect(removeOutliersEligibility({ list: 'sam', selectionSize: 3 }).eligible).toBe(false);
  });

  it('instance: eligible iff selected and not confirmed', () => {
    expect(removeOutliersEligibility({ list: 'instance', isSelected: true, confirmed: false }).eligible).toBe(true);
    expect(removeOutliersEligibility({ list: 'instance', isSelected: true, confirmed: true }).eligible).toBe(false);
    expect(removeOutliersEligibility({ list: 'instance', isSelected: false, confirmed: false }).eligible).toBe(false);
  });

  it('throws on unknown list', () => {
    expect(() => removeOutliersEligibility({ list: 'preseg' })).toThrow();
  });
});
