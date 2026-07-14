// sam-mode.jsx — SAM sub-mode of Label mode. A tool is only a way to *select
// points*: Shift-drag a screen box (or type a concept prompt) → the server
// renders the cloud from the current camera pose and runs SAM → the returned
// masks are reviewed → chosen masks project back to a point selection that
// flows through the shared apply pipeline (via /api/sam/project). Modeled on
// beam-mode.jsx (overlay listeners on the viewer canvas + a side panel).

import { useCallback, useEffect, useRef, useState } from 'react';
import { VoxaAPI } from './api.js';
import { normalizeBox, capturePayload, maskColor, containPixel } from './sam-util.js';
import { applySamDelta } from './segment-state.js';

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

// Large centered review modal — the box-drag/concept capture returns an
// overlay image the size of the viewport and N masks; squeezing that into
// the 280px tool-options rail made it unreadable and gave no way to tell
// which list row was which colored blob. Shown big, with swatches that
// mirror the sidecar's per-mask palette (see sam-util.js::maskColor).
function SamReviewModal({ capture, chosen, busy, onToggle, onSelectOnly, onProject, onCancel }) {
  const imgRef = useRef(null);
  const indexCanvasRef = useRef(null);   // offscreen decode of mask_index_png_b64

  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape') onCancel(); };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [onCancel]);

  // Decode the per-pixel mask-index map once per capture so a click on the
  // (baked, flat) overlay image can be hit-tested against a specific mask_id —
  // color-matching 20 palette hues by eye doesn't scale, clicking the object does.
  useEffect(() => {
    indexCanvasRef.current = null;
    if (!capture.indexPng) return undefined;
    let cancelled = false;
    const img = new Image();
    img.onload = () => {
      if (cancelled) return;
      const c = document.createElement('canvas');
      c.width = img.naturalWidth; c.height = img.naturalHeight;
      c.getContext('2d').drawImage(img, 0, 0);
      indexCanvasRef.current = c;
    };
    img.src = capture.indexPng;
    return () => { cancelled = true; };
  }, [capture.indexPng]);

  const onImageClick = useCallback((e) => {
    const idxCanvas = indexCanvasRef.current;
    const imgEl = imgRef.current;
    if (!idxCanvas || !imgEl) return;
    const rect = imgEl.getBoundingClientRect();
    const px = containPixel({
      boxW: rect.width, boxH: rect.height,
      natW: imgEl.naturalWidth, natH: imgEl.naturalHeight,
      x: e.clientX - rect.left, y: e.clientY - rect.top,
    });
    if (!px) return;
    const [sx, sy] = px;
    const val = idxCanvas.getContext('2d').getImageData(sx, sy, 1, 1).data[0];
    if (val === 0) return;              // background — no mask under the click
    const maskId = val - 1;             // index PNG is 1-based; mask_id is 0-based
    // Plain click = select just this mask; Ctrl/Cmd/Shift+click = add/remove it from
    // the current selection — matches the additive-select convention used elsewhere
    // (draw-mode.jsx, mode-label.jsx: e.shiftKey || e.ctrlKey || e.metaKey).
    if (e.ctrlKey || e.metaKey || e.shiftKey) onToggle(maskId);
    else onSelectOnly(maskId);
  }, [onToggle, onSelectOnly]);

  return (
    <div className="sam-review-overlay" onClick={onCancel}>
      <div className="sam-review-card" onClick={(e) => e.stopPropagation()}>
        <div className="sam-review-head">
          <b>SAM masks — {capture.masks.length} found</b>
          <button className="ew-x" onClick={onCancel} title="Close (Esc)">✕</button>
        </div>
        <div className="sam-review-body">
          <div className="sam-review-imgcol">
            <img ref={imgRef} src={capture.overlayPng} alt="SAM masks"
              className="sam-review-img" onClick={onImageClick} />
            <p className="tool-opt-hint">
              Click a mask to select it; Ctrl/Cmd/Shift+click to add or remove it from the selection.
            </p>
          </div>
          <div className="sam-review-list">
            {capture.masks.map((m) => (
              <label key={m.mask_id} className="sam-mask-row">
                <input type="checkbox" checked={chosen.has(m.mask_id)}
                  onChange={() => onToggle(m.mask_id)} />
                <span className="sam-mask-swatch" style={{ background: maskColor(m.mask_id) }} />
                <span className="inst-text">
                  <b>mask #{m.mask_id}</b>{' '}
                  <em>{typeof m.score === 'number' ? m.score.toFixed(3) : ''}</em>
                </span>
              </label>
            ))}
          </div>
        </div>
        <div className="sam-review-actions">
          <button className="ghost-btn" onClick={onCancel}>Cancel</button>
          <button className="ghost-btn" disabled={chosen.size === 0 || busy}
            onClick={onProject}>Add to SAM segments</button>
        </div>
      </div>
    </div>
  );
}

export default function SamMode({
  viewerRef, setSegState, protectInstances = [],
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
        indexPng: cap.mask_index_png_b64 || null,
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
        protectInstances: protectInstancesRef.current,
      });
      let nProtected = 0;
      for (const seg of res.segments || []) {
        nProtected += seg.nProtected || 0;
        if (seg.indices) {
          setSegState?.((s) => (s ? applySamDelta(s, {
            indices: seg.indices,
            samSegId: seg.samSegId,
          }) : s));
        }
      }
      // Mirrors draw-mode.jsx/beam-mode.jsx: a mask that landed entirely (or
      // partly) on already-confirmed points is silently narrowed by
      // protect_instances — "confirmed = locked" must fail loud, not silent.
      if (nProtected > 0) {
        setError(`${nProtected} point(s) locked in a confirmed instance`);
      }
      setCapture(null);
      setChosen(new Set());
    } catch (e) {
      setError(String(e.message || e));
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }, [capture, chosen, setSegState]);

  const toggleMask = useCallback((maskId) => {
    setChosen((prev) => {
      const next = new Set(prev);
      if (next.has(maskId)) next.delete(maskId); else next.add(maskId);
      return next;
    });
  }, []);

  const selectOnlyMask = useCallback((maskId) => setChosen(new Set([maskId])), []);

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

        {/* Text is visible in BOTH modes: in concept it's the prompt, in box it's
            an optional SAM refinement. Always shown so a box capture never carries
            hidden/stale text (the sidecar forwards req.text in box mode too). */}
        <div className="ins-row" style={{ marginTop: 6 }}>
          <input className="ins-input" type="text"
            placeholder={mode === 'concept' ? 'prompt (e.g. pipe)' : 'optional text refinement'}
            value={text} onChange={(e) => setText(e.target.value)} />
          {mode === 'concept' && (
            <button className="ghost-btn" disabled={busy || !text.trim()}
              onClick={() => doCapture('concept', null)}>Segment all</button>
          )}
        </div>
        {mode === 'box' && (
          <p className="tool-opt-hint">
            Shift-drag a rectangle over the cloud to segment it
            {text.trim() ? ' (refined by the text above)' : ''}.
          </p>
        )}

        {busy && <p className="tool-opt-hint">Working…</p>}
      </div>

      {capture && (
        <SamReviewModal capture={capture} chosen={chosen} busy={busy}
          onToggle={toggleMask} onSelectOnly={selectOnlyMask} onProject={doProject}
          onCancel={() => { setCapture(null); setChosen(new Set()); }} />
      )}
    </>
  );
}
