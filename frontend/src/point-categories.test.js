import { describe, it, expect } from 'vitest';
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import {
  CATEGORY_ARTIFACT, CATEGORY_EXCLUDED_REVIEW, CATEGORY_NONE, CATEGORY_TRANSIENT,
  POINT_CATEGORIES, buildCategoryOverlay, categoryByKey, categoryCounts,
} from './point-categories.js';

describe('point categories', () => {
  it('mirrors the backend constants (pinned like classes.yaml)', () => {
    // backend/labeling/categories.py is the other half of this contract.
    const src = readFileSync(
      fileURLToPath(new URL('../../backend/labeling/categories.py', import.meta.url)),
      'utf8');
    const value = (name) => {
      const m = src.match(new RegExp(`^${name} = (\\d+)`, 'm'));
      expect(m, `${name} missing from categories.py`).toBeTruthy();
      return Number(m[1]);
    };
    expect(value('CATEGORY_NONE')).toBe(CATEGORY_NONE);
    expect(value('CATEGORY_ARTIFACT')).toBe(CATEGORY_ARTIFACT);
    expect(value('CATEGORY_TRANSIENT')).toBe(CATEGORY_TRANSIENT);
    expect(value('CATEGORY_EXCLUDED_REVIEW')).toBe(CATEGORY_EXCLUDED_REVIEW);
    // every wire name the picker can send must exist in the backend mapping
    for (const c of POINT_CATEGORIES) {
      expect(src).toContain(`"${c.name}"`);
    }
  });

  it('resolves picker hotkeys case-insensitively, and only known ones', () => {
    expect(categoryByKey('a').name).toBe('artifact');
    expect(categoryByKey('R').name).toBe('excluded_review');
    expect(categoryByKey('x').name).toBe('none');
    expect(categoryByKey('q')).toBeNull();
  });

  it('counts marks, ignoring none', () => {
    const cats = Int8Array.from([0, 1, 1, 2, 3, 0]);
    expect(categoryCounts(cats)).toEqual({ artifact: 2, transient: 1, excluded_review: 1 });
    expect(categoryCounts(null)).toEqual({ artifact: 0, transient: 0, excluded_review: 0 });
  });
});

describe('buildCategoryOverlay', () => {
  it('returns null when nothing is marked', () => {
    expect(buildCategoryOverlay(Int8Array.from([0, 0, 0]), null, 3)).toBeNull();
    expect(buildCategoryOverlay(null, null, 3)).toBeNull();
  });

  it('masks marked points and colors them per category', () => {
    const cats = Int8Array.from([0, 1, 3]);
    const { mask, colors } = buildCategoryOverlay(cats, null, 3);
    expect(Array.from(mask)).toEqual([0, 1, 1]);
    expect(colors.slice(0, 3)).toEqual(new Float32Array([0, 0, 0]));
    // artifact (#ff4dd2) and review (#9aa0a6) get distinct colors
    expect(colors.slice(3, 6)).not.toEqual(colors.slice(6, 9));
    expect(colors[3]).toBeCloseTo(1.0, 5);          // ff
  });

  it('maps rendered rows through subsampleIdx', () => {
    const cats = Int8Array.from([0, 0, 2, 0]);      // full-res index 2 is transient
    const subIdx = Int32Array.from([1, 2]);         // renders full-res 1 and 2
    const { mask } = buildCategoryOverlay(cats, subIdx, 2);
    expect(Array.from(mask)).toEqual([0, 1]);
  });
});
