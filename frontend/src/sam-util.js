// Pure helpers for the SAM tool. No DOM/Three imports (unit-testable).
export function normalizeBox({ x0, y0, x1, y1 }, canvas) {
  // drag rect is in CSS px; normalize against CSS size (buffer scaling cancels out).
  const W = canvas.clientWidth, H = canvas.clientHeight;
  const lx = Math.min(x0, x1), hx = Math.max(x0, x1);
  const ly = Math.min(y0, y1), hy = Math.max(y0, y1);
  return [ (lx + hx) / 2 / W, (ly + hy) / 2 / H, (hx - lx) / W, (hy - ly) / H ];
}

export function capturePayload({ pose, canvas, mode, box, text }) {
  return {
    camera: { pos: pose.pos, target: pose.target, fov: pose.fov,
              W: canvas.width, H: canvas.height },
    mode, box: box ?? null, text: text ?? null,
  };
}

// Maps a click point (CSS px, relative to the <img> box) to source-pixel
// coords under `object-fit: contain` letterboxing, or null if the click
// landed on the letterbox padding rather than the image itself.
export function containPixel({ boxW, boxH, natW, natH, x, y }) {
  const scale = Math.min(boxW / natW, boxH / natH);
  const dispW = natW * scale, dispH = natH * scale;
  const offX = (boxW - dispW) / 2, offY = (boxH - dispH) / 2;
  const px = x - offX, py = y - offY;
  if (px < 0 || py < 0 || px >= dispW || py >= dispH) return null;
  return [Math.floor(px / scale), Math.floor(py / scale)];
}

// Mirrors sam_sidecar/main.py::_palette (golden-ratio hue spacing) so the
// mask list's color swatches match the wash the sidecar bakes into
// overlay_png_b64 — mask #i is always the same color in the list and image.
export function maskColor(i) {
  const h = (i * 0.61803398875) % 1.0, s = 0.65, v = 1.0;
  const k = (n) => (n + h * 6) % 6;
  const f = (n) => v - v * s * Math.max(0, Math.min(k(n), 4 - k(n), 1));
  const r = Math.round(f(5) * 255), g = Math.round(f(3) * 255), b = Math.round(f(1) * 255);
  return `rgb(${r}, ${g}, ${b})`;
}
