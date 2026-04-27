import { describe, expect, it } from 'vitest';
import { newId } from './api.js';

describe('newId', () => {
  it('uses the supplied prefix', () => {
    expect(newId('cuboid')).toMatch(/^cuboid-[a-z0-9]+$/);
  });

  it('defaults to "inst"', () => {
    expect(newId()).toMatch(/^inst-[a-z0-9]+$/);
  });

  it('returns distinct ids on successive calls', () => {
    const a = newId();
    const b = newId();
    expect(a).not.toBe(b);
  });
});
