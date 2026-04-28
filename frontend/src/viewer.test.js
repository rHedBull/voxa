import { describe, it, expect } from 'vitest';
import { evtToNdc } from './viewer.jsx';

describe('evtToNdc', () => {
  it('maps centre of rect to (0, 0)', () => {
    const rect = { left: 100, top: 50, width: 800, height: 600 };
    const evt  = { clientX: 500, clientY: 350 };
    const ndc  = evtToNdc(evt, rect);
    expect(ndc.x).toBeCloseTo(0, 5);
    expect(ndc.y).toBeCloseTo(0, 5);
  });

  it('maps top-left corner to (-1, +1)', () => {
    const rect = { left: 0, top: 0, width: 400, height: 300 };
    const evt  = { clientX: 0, clientY: 0 };
    const ndc  = evtToNdc(evt, rect);
    expect(ndc.x).toBeCloseTo(-1, 5);
    expect(ndc.y).toBeCloseTo(1, 5);
  });

  it('maps bottom-right corner to (+1, -1)', () => {
    const rect = { left: 0, top: 0, width: 400, height: 300 };
    const evt  = { clientX: 400, clientY: 300 };
    const ndc  = evtToNdc(evt, rect);
    expect(ndc.x).toBeCloseTo(1, 5);
    expect(ndc.y).toBeCloseTo(-1, 5);
  });
});
