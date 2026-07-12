// sam-mode.jsx — SAM sub-mode of Label mode. A tool is only a way to *select
// points*: Shift-drag a screen box (or type a concept prompt) → the server
// renders the cloud from the current camera pose and runs SAM → the returned
// masks are reviewed → chosen masks project back to a point selection that
// flows through the shared apply pipeline (via /api/sam/project). Modeled on
// beam-mode.jsx (overlay listeners on the viewer canvas + a side panel).

import { useCallback, useEffect, useRef, useState } from 'react';
import { VoxaAPI } from './api.js';
import { normalizeBox, capturePayload } from './sam-util.js';

// Shift-drag rubber-band capture over the viewer canvas. Shift is the "select"
// modifier — without it the mousedown falls through so the viewer orbits.
function SamOverlay({ viewerRef, mode, busyRef, onBox }) {
  const [rect, setRect] = useState(null); // page-space {left,top,w,h} while dragging
  const dragRef = useRef(null);           // { canvasRect, x0, y0 }
  const onBoxRef = useRef(onBox);
  onBoxRef.current = onBox;
  const modeRef = useRef(mode);
  modeRef.current = mode;

  useEffect(() => {
    const v = viewerRef.current;
    const dom = v?.domElement?.();
    if (!dom) return undefined;

    const onDown = (e) => {
      if (e.button !== 0 || !e.shiftKey || modeRef.current !== 'box') return;
      if (busyRef.current) return;   // a SAM request is already in flight
      // Shift-drag is ours: suppress orbit and start the rubber-band.
      e.preventDefault();
      e.stopPropagation();
      v.setOrbitEnabled?.(false);
      const canvasRect = dom.getBoundingClientRect();
      dragRef.current = { canvasRect, x0: e.clientX, y0: e.clientY };
      setRect({ left: e.clientX, top: e.clientY, w: 0, h: 0 });
    };

    const onMove = (e) => {
      const d = dragRef.current;
      if (!d) return;
      setRect({
        left: Math.min(d.x0, e.clientX),
        top: Math.min(d.y0, e.clientY),
        w: Math.abs(e.clientX - d.x0),
        h: Math.abs(e.clientY - d.y0),
      });
    };

    const onUp = (e) => {
      const d = dragRef.current;
      if (!d) return;
      dragRef.current = null;
      setRect(null);
      v.setOrbitEnabled?.(true);
      const r = d.canvasRect;
      // CSS-pixel offsets relative to the canvas bounding rect.
      const x0 = d.x0 - r.left, y0 = d.y0 - r.top;
      const x1 = e.clientX - r.left, y1 = e.clientY - r.top;
      if (Math.abs(x1 - x0) < 4 || Math.abs(y1 - y0) < 4) return; // trivial rect
      const box = normalizeBox({ x0, y0, x1, y1 }, dom);
      onBoxRef.current?.(box);
    };

    dom.addEventListener('pointerdown', onDown, true);
    window.addEventListener('pointermove', onMove);
    window.addEventListener('pointerup', onUp);
    return () => {
      dom.removeEventListener('pointerdown', onDown, true);
      window.removeEventListener('pointermove', onMove);
      window.removeEventListener('pointerup', onUp);
      v.setOrbitEnabled?.(true);
    };
  }, [viewerRef, busyRef]);

  if (!rect) return null;
  return (
    <div style={{
      position: 'fixed', left: rect.left, top: rect.top,
      width: rect.w, height: rect.h,
      border: '1.5px solid #60a5fa', background: 'rgba(96,165,250,0.15)',
      pointerEvents: 'none', zIndex: 30,
    }} />
  );
}

export default function SamMode({
  viewerRef, classes, defaultClassId, onApplied,
  protectInstances = [], autoConfirm,
}) {
  const [mode, setMode] = useState('box');      // 'box' | 'concept'
  const [text, setText] = useState('');
  const [capture, setCapture] = useState(null); // { captureId, overlayPng, masks }
  const [chosen, setChosen] = useState(() => new Set());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const protectInstancesRef = useRef(protectInstances);
  protectInstancesRef.current = protectInstances;
  const textRef = useRef(text);
  textRef.current = text;
  // At most one in-flight SAM request at a time: read synchronously (not
  // through the async-stale `busy` state) at each entry point.
  const busyRef = useRef(false);
  busyRef.current = busy;

  // Switching mode drops any pending review (a box mask doesn't belong to a
  // concept prompt and vice versa).
  useEffect(() => {
    setCapture(null);
    setChosen(new Set());
  }, [mode]);

  const doCapture = useCallback(async (capMode, box) => {
    if (busyRef.current) return;
    const v = viewerRef.current;
    const pose = v?.cameraPose?.();
    const canvas = v?.domElement?.();
    if (!pose || !canvas) { setError('viewer not ready'); return; }
    busyRef.current = true;
    setBusy(true);
    setError(null);
    const payload = capturePayload({
      pose, canvas, mode: capMode, box, text: textRef.current || null,
    });
    try {
      const cap = await VoxaAPI.samCapture(payload);
      const masks = cap.masks || [];
      setCapture({
        captureId: cap.capture_id,
        overlayPng: cap.overlay_png_b64,
        masks,
      });
      setChosen(new Set(
        capMode === 'box' && masks.length ? [masks[0].mask_id] : []));
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [viewerRef]);

  const doProject = useCallback(async () => {
    if (busyRef.current) return;
    if (!capture || chosen.size === 0) return;
    busyRef.current = true;
    setBusy(true);
    setError(null);
    try {
      const res = await VoxaAPI.samProject({
        captureId: capture.captureId,
        maskIds: [...chosen],
        targetClass: defaultClassId,
        protectInstances: protectInstancesRef.current,
      });
      for (const inst of res.instances || []) {
        if (inst.new_instance_id != null) {
          onApplied?.({ instanceId: inst.new_instance_id,
                        classId: defaultClassId, source: 'sam' });
        }
      }
      setCapture(null);
      setChosen(new Set());
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [capture, chosen, defaultClassId, onApplied]);

  const toggleMask = useCallback((maskId) => {
    setChosen((prev) => {
      const next = new Set(prev);
      if (next.has(maskId)) next.delete(maskId); else next.add(maskId);
      return next;
    });
  }, []);

  return (
    <>
      <SamOverlay viewerRef={viewerRef} mode={mode} busyRef={busyRef}
        onBox={(box) => doCapture('box', box)} />
      <div className="sam-panel" style={{ marginTop: 10 }}>
        {error && (
          <div className="sam-error" style={{
            background: 'rgba(220,38,38,0.15)', border: '1px solid #dc2626',
            color: '#fca5a5', borderRadius: 6, padding: '6px 8px', fontSize: 12,
            marginBottom: 8, display: 'flex', gap: 8, alignItems: 'center',
          }}>
            <span style={{ flex: 1 }}>{error}</span>
            <button className="ghost-btn" onClick={() => setError(null)}>Dismiss</button>
          </div>
        )}

        <div className="tool-opt-toggle">
          <button className={mode === 'box' ? 'active' : ''}
            onClick={() => setMode('box')}>Box</button>
          <button className={mode === 'concept' ? 'active' : ''}
            onClick={() => setMode('concept')}>Concept</button>
        </div>

        {mode === 'box' ? (
          <p className="tool-opt-hint">Shift-drag a rectangle over the cloud to segment it.</p>
        ) : (
          <div className="ins-row" style={{ marginTop: 6 }}>
            <input className="ins-input" type="text" placeholder="prompt (e.g. pipe)"
              value={text} onChange={(e) => setText(e.target.value)} />
            <button className="ghost-btn" disabled={busy || !text.trim()}
              onClick={() => doCapture('concept', null)}>Segment all</button>
          </div>
        )}

        {busy && <p className="tool-opt-hint">Working…</p>}

        {capture && (
          <div style={{ marginTop: 10 }}>
            <img src={capture.overlayPng} alt="SAM masks"
              style={{ width: '100%', borderRadius: 6, display: 'block' }} />
            <div style={{ maxHeight: 180, overflowY: 'auto', marginTop: 6 }}>
              {capture.masks.map((m) => (
                <label key={m.mask_id} className="inst-row"
                  style={{ display: 'flex', gap: 8, alignItems: 'center', cursor: 'pointer' }}>
                  <input type="checkbox" checked={chosen.has(m.mask_id)}
                    onChange={() => toggleMask(m.mask_id)} />
                  <span className="inst-text">
                    <b>mask #{m.mask_id}</b>{' '}
                    <em>{typeof m.score === 'number' ? m.score.toFixed(3) : ''}</em>
                  </span>
                </label>
              ))}
            </div>
            <div className="ins-actions">
              <button className="ghost-btn" disabled={chosen.size === 0 || busy}
                onClick={doProject}>Project selected</button>
            </div>
          </div>
        )}
      </div>
    </>
  );
}
