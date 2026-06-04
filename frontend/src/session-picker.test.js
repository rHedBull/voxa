import { describe, expect, it } from 'vitest';
import { formatSavedAt } from './session-picker.jsx';

describe('formatSavedAt', () => {
  it('trims an ISO timestamp to minute precision and swaps the T for a space', () => {
    expect(formatSavedAt('2026-06-04T13:22:10.123456Z')).toBe('2026-06-04 13:22');
  });

  it('returns empty string for null/undefined/empty', () => {
    expect(formatSavedAt(null)).toBe('');
    expect(formatSavedAt(undefined)).toBe('');
    expect(formatSavedAt('')).toBe('');
  });
});
