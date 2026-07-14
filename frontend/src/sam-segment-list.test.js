import { describe, it, expect } from 'vitest';
import { toggleSamSelection } from './sam-segment-list.jsx';

describe('toggleSamSelection', () => {
  it('adds an id not yet in the selection', () => {
    const next = toggleSamSelection(new Set([1]), 2);
    expect(Array.from(next)).toEqual([1, 2]);
  });

  it('removes an id already in the selection', () => {
    const next = toggleSamSelection(new Set([1, 2]), 2);
    expect(Array.from(next)).toEqual([1]);
  });

  it('does not mutate the input set', () => {
    const input = new Set([1]);
    toggleSamSelection(input, 2);
    expect(Array.from(input)).toEqual([1]);
  });
});
