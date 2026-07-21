import { describe, it, expect } from 'vitest';
import { cutEligibility } from './cut-eligibility.js';

describe('cutEligibility', () => {
  it('preseg multi-select is eligible', () => {
    expect(cutEligibility({ list: 'preseg', selectionSize: 2 })).toEqual({ eligible: true });
  });

  it('preseg single-select is eligible', () => {
    expect(cutEligibility({ list: 'preseg', selectionSize: 1 })).toEqual({ eligible: true });
  });

  it('empty preseg selection is not eligible', () => {
    expect(cutEligibility({ list: 'preseg', selectionSize: 0 }))
      .toEqual({ eligible: false, reason: 'empty' });
  });

  it('sam multi-select is eligible', () => {
    expect(cutEligibility({ list: 'sam', selectionSize: 3 })).toEqual({ eligible: true });
  });

  it('empty sam selection is not eligible', () => {
    expect(cutEligibility({ list: 'sam', selectionSize: 0 }))
      .toEqual({ eligible: false, reason: 'empty' });
  });

  it('single unconfirmed selected instance is eligible', () => {
    expect(cutEligibility({ list: 'instance', isSelected: true, confirmed: false }))
      .toEqual({ eligible: true });
  });

  it('confirmed instance is not eligible even if selected', () => {
    expect(cutEligibility({ list: 'instance', isSelected: true, confirmed: true }))
      .toEqual({ eligible: false, reason: 'confirmed' });
  });

  it('unselected instance row is not eligible', () => {
    expect(cutEligibility({ list: 'instance', isSelected: false, confirmed: false }))
      .toEqual({ eligible: false, reason: 'not-selected' });
  });

  it('unselected confirmed instance row is not eligible (not-selected wins)', () => {
    expect(cutEligibility({ list: 'instance', isSelected: false, confirmed: true }))
      .toEqual({ eligible: false, reason: 'not-selected' });
  });

  it('frozen-class instance is not eligible (cut inherits the class)', () => {
    expect(cutEligibility({
      list: 'instance', isSelected: true, confirmed: false, classFrozen: true,
    })).toEqual({ eligible: false, reason: 'frozen-class' });
  });

  it('live-class instance stays eligible with classFrozen false', () => {
    expect(cutEligibility({
      list: 'instance', isSelected: true, confirmed: false, classFrozen: false,
    })).toEqual({ eligible: true });
  });

  it('throws on an unknown list', () => {
    expect(() => cutEligibility({ list: 'bogus' })).toThrow();
  });
});
