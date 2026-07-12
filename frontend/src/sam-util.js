// Pure helpers for the SAM tool. No DOM/Three imports (unit-testable).
export function normalizeBox({ x0, y0, x1, y1 }, canvas) {
  // drag rect is in CSS px; normalize against CSS size (buffer scaling cancels out).
  const W = canvas.clientWidth, H = canvas.clientHeight;
  const lx = Math.min(x0, x1), hx = Math.max(x0, x1);
  const ly = Math.min(y0, y1), hy = Math.max(y0, y1);
  return [ (lx + hx) / 2 / W, (ly + hy) / 2 / H, (hx - lx) / W, (hy - ly) / H ];
}

export function capturePayload({ view, fov, canvas, mode, box, text }) {
  return {
    camera: { pos: view.position.toArray(), target: view.getPivot().toArray(),
              fov, W: canvas.width, H: canvas.height },
    mode, box: box ?? null, text: text ?? null,
  };
}
