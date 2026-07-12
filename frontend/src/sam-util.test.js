import { describe, it, expect } from 'vitest';
import { normalizeBox, capturePayload } from './sam-util.js';

describe('normalizeBox', () => {
  it('CSS px rect → normalized [cx,cy,w,h] in canvas buffer space', () => {
    const b = normalizeBox({ x0: 100, y0: 100, x1: 300, y1: 300 },
                           { clientWidth: 500, clientHeight: 400, width: 1000, height: 800 });
    expect(b).toEqual([0.4, 0.5, 0.4, 0.5]); // cx=(100+300)/2/500=0.4 ; w=200/500=0.4 ...
  });
});

describe('capturePayload', () => {
  it('assembles camera pose from a viewer view', () => {
    const view = { position: {toArray:()=>[1,2,3]}, getPivot:()=>({toArray:()=>[4,5,6]}) };
    const p = capturePayload({ view, fov: 60, canvas: { width: 1000, height: 800 },
                               mode: 'box', box: [0.4,0.5,0.4,0.5], text: null });
    expect(p.camera).toEqual({ pos:[1,2,3], target:[4,5,6], fov:60, W:1000, H:800 });
    expect(p.mode).toBe('box'); expect(p.box).toEqual([0.4,0.5,0.4,0.5]);
  });
});
