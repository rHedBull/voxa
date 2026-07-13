import { describe, it, expect } from 'vitest';
import { normalizeBox, capturePayload, maskColor, containPixel } from './sam-util.js';

describe('normalizeBox', () => {
  it('CSS px rect → normalized [cx,cy,w,h] in canvas buffer space', () => {
    const b = normalizeBox({ x0: 100, y0: 100, x1: 300, y1: 300 },
                           { clientWidth: 500, clientHeight: 400, width: 1000, height: 800 });
    expect(b).toEqual([0.4, 0.5, 0.4, 0.5]); // cx=(100+300)/2/500=0.4 ; w=200/500=0.4 ...
  });
});

describe('capturePayload', () => {
  it('assembles camera pose from a plain viewer pose', () => {
    const pose = { pos: [1, 2, 3], target: [4, 5, 6], fov: 60 };
    const p = capturePayload({ pose, canvas: { width: 1000, height: 800 },
                               mode: 'box', box: [0.4,0.5,0.4,0.5], text: null });
    expect(p.camera).toEqual({ pos:[1,2,3], target:[4,5,6], fov:60, W:1000, H:800 });
    expect(p.mode).toBe('box'); expect(p.box).toEqual([0.4,0.5,0.4,0.5]);
  });
});

describe('maskColor', () => {
  it('matches sam_sidecar/main.py::_palette golden-ratio hues', () => {
    // Values captured from colorsys.hsv_to_rgb((i*0.61803398875) % 1.0, 0.65, 1.0)
    // for i in 0..5 — kept in sync with the sidecar so list swatches match the overlay.
    expect(maskColor(0)).toBe('rgb(255, 89, 89)');
    expect(maskColor(1)).toBe('rgb(89, 138, 255)');
    expect(maskColor(2)).toBe('rgb(186, 255, 89)');
    expect(maskColor(3)).toBe('rgb(255, 89, 234)');
    expect(maskColor(4)).toBe('rgb(89, 255, 227)');
    expect(maskColor(5)).toBe('rgb(255, 179, 89)');
  });
});

describe('containPixel', () => {
  it('maps a click to source pixels when the box is wider than the image (horizontal letterbox)', () => {
    // 200x100 box, 100x100 image -> scale 1, letterboxed 50px each side horizontally
    const p = containPixel({ boxW: 200, boxH: 100, natW: 100, natH: 100, x: 100, y: 50 });
    expect(p).toEqual([50, 50]);
  });

  it('returns null for a click on the letterbox padding', () => {
    const p = containPixel({ boxW: 200, boxH: 100, natW: 100, natH: 100, x: 10, y: 50 });
    expect(p).toBeNull();
  });

  it('scales down a larger image to fit the box', () => {
    // 400x400 image in a 100x100 box -> scale 0.25
    const p = containPixel({ boxW: 100, boxH: 100, natW: 400, natH: 400, x: 50, y: 25 });
    expect(p).toEqual([200, 100]);
  });
});
