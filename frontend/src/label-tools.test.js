import { describe, it, expect } from 'vitest';
import { TOOLS, toolAvailable, defaultTool } from './label-tools.js';

describe('label-tools', () => {
  it('lists the three selection tools in rail order', () => {
    expect(TOOLS.map((t) => t.id)).toEqual(['presegment', 'box', 'draw']);
  });
  it('gates presegment on segState and draw on annotated', () => {
    const raw = { segState: null, isAnnotated: false };
    expect(toolAvailable('box', raw)).toBe(true);
    expect(toolAvailable('presegment', raw)).toBe(false);
    expect(toolAvailable('draw', raw)).toBe(false);
    const ann = { segState: {}, isAnnotated: true };
    expect(toolAvailable('presegment', ann)).toBe(true);
    expect(toolAvailable('draw', ann)).toBe(true);
  });
  it('defaults to presegment when available, else box', () => {
    expect(defaultTool({ segState: {}, isAnnotated: true })).toBe('presegment');
    expect(defaultTool({ segState: null, isAnnotated: false })).toBe('box');
  });
});
