import { describe, it, expect } from 'vitest';
import { TOOLS, toolAvailable, defaultTool } from './label-tools.js';

describe('label-tools', () => {
  it('lists the seven selection tools in rail order', () => {
    expect(TOOLS.map((t) => t.id)).toEqual(['presegment', 'box', 'prism', 'draw', 'beam', 'sam', 'region']);
  });
  it('gates all tools on a segment session (apply needs one); draw/beam/sam also on annotated', () => {
    const raw = { segState: null, isAnnotated: false };
    // No session → nothing can apply, including Box.
    expect(toolAvailable('box', raw)).toBe(false);
    expect(toolAvailable('prism', raw)).toBe(false);
    expect(toolAvailable('presegment', raw)).toBe(false);
    expect(toolAvailable('draw', raw)).toBe(false);
    expect(toolAvailable('beam', raw)).toBe(false);
    expect(toolAvailable('sam', raw)).toBe(false);
    // Session but not annotated tier → Box + Presegment work, Draw/Beam/SAM don't.
    const sessionOnly = { segState: {}, isAnnotated: false };
    expect(toolAvailable('box', sessionOnly)).toBe(true);
    expect(toolAvailable('prism', sessionOnly)).toBe(true);
    expect(toolAvailable('presegment', sessionOnly)).toBe(true);
    expect(toolAvailable('draw', sessionOnly)).toBe(false);
    expect(toolAvailable('beam', sessionOnly)).toBe(false);
    expect(toolAvailable('sam', sessionOnly)).toBe(false);
    const ann = { segState: {}, isAnnotated: true };
    expect(toolAvailable('presegment', ann)).toBe(true);
    expect(toolAvailable('draw', ann)).toBe(true);
    expect(toolAvailable('beam', ann)).toBe(true);
    // SAM additionally needs rawSourceAvailable.
    expect(toolAvailable('sam', ann)).toBe(false);
    expect(toolAvailable('sam', { ...ann, rawSourceAvailable: true })).toBe(true);
  });
  it('defaults to presegment when available, else box', () => {
    expect(defaultTool({ segState: {}, isAnnotated: true })).toBe('presegment');
    expect(defaultTool({ segState: null, isAnnotated: false })).toBe('box');
  });
  it('region tool is annotated-tier + session gated, like draw/beam', () => {
    expect(TOOLS.some((t) => t.id === 'region')).toBe(true);
    expect(toolAvailable('region', { segState: {}, isAnnotated: true })).toBe(true);
    expect(toolAvailable('region', { segState: {}, isAnnotated: false })).toBe(false);
    expect(toolAvailable('region', { segState: null, isAnnotated: true })).toBe(false);
  });
});
