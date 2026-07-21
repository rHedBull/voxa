// export-wizard.jsx — 3-step modal that turns the active session's labels
// into a downloadable zip (labeled .ply + manifest.json) via
// POST /api/labels/export. Read-only: never mutates stored labels.
//
// Steps: 1 Resolution · 2 Classes (include/exclude + merge/rename) · 3 Review.
// The correctness-critical taxonomy/estimate logic lives in the unit-tested
// pure helpers in export-wizard-util.js; this file is presentation + wiring.

import { useState, useEffect, useMemo, useRef } from 'react';
import { VoxaAPI } from './api.js';
import { remapToTaxonomy, estimatePoints, pointsAfterFilters } from './export-wizard-util.js';

// Format a backend error (422 {errors:[...]} or 409 string) for inline display.
function errText(err) {
  const d = err?.detail;
  if (d && Array.isArray(d.errors)) return d.errors.join('; ');
  if (typeof d === 'string') return d;
  return err?.message || 'Export failed';
}

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function ExportWizard({
  scene, sessionId, classes, scanCount, rawSourceAvailable,
  perClassPointCounts, nLabeledPoints, onClose,
}) {
  const [step, setStep] = useState(1);

  // Resolution
  const [kind, setKind] = useState('scan');
  const [subN, setSubN] = useState(() => Math.min(500000, scanCount || 500000));

  // Classes
  const [confirmedOnly, setConfirmedOnly] = useState(false);
  // includeSet holds the class_ids currently kept. Starts as "all".
  const [includeSet, setIncludeSet] = useState(() => new Set(classes.map((c) => c.class_id)));
  // Merge rows: { key, from:Set<class_id>, label, color }. `key` is a stable
  // unique id from a monotonic counter (survives middle-row removal).
  const [rows, setRows] = useState([]);
  const nextRowKey = useRef(0);

  // Review
  const [accuracy, setAccuracy] = useState(null);
  const [accuracyErr, setAccuracyErr] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);

  const consumed = useMemo(() => {
    const s = new Set();
    rows.forEach((r) => r.from.forEach((id) => s.add(id)));
    return s;
  }, [rows]);

  const rawCount = null;  // raw n_points not in LoadResponse; shown qualitatively.
  const targetPoints = estimatePoints({ kind, n: subN }, scanCount, rawCount);

  // Target ids for merge rows: above every palette id so they never collide
  // with a kept-through class (backend validation rejects collisions).
  const baseTargetId = useMemo(
    () => Math.max(-1, ...classes.map((c) => c.class_id)) + 1, [classes]);
  const activeRows = rows.filter((r) => r.from.size > 0);

  // include_classes payload: null when everything is kept, else the array.
  const includeSetOrNull = includeSet.size === classes.length ? null : includeSet;
  const includeArr = includeSetOrNull && [...includeSetOrNull];

  // Live taxonomy preview (mirrors backend build_taxonomy).
  const previewRows = activeRows.map((r, i) => ({
    from: [...r.from],
    to: { id: baseTargetId + i, label: r.label || `group_${i + 1}`, color: r.color },
  }));
  const taxonomy = remapToTaxonomy(classes, previewRows, includeSetOrNull);

  // ~0 disable-guard, computed at scan density (resolution-independent —
  // scaling to a denser target can't turn a non-zero count into zero, and
  // raw's target count is unknown client-side). Uses per-class point counts
  // when available (prelabel sessions); otherwise falls back to "any labeled
  // point + a non-empty include set". confirmed-only is ignored here (safe
  // lower bound).
  const labeledAfter = perClassPointCounts
    ? pointsAfterFilters(perClassPointCounts, includeSetOrNull, scanCount, scanCount)
    : null;
  const emptyAfter = perClassPointCounts
    ? labeledAfter < 1
    : !((nLabeledPoints || 0) > 0 && (includeSetOrNull === null || includeSetOrNull.size > 0));

  // A merge row with sources but a blank label would export an unlabeled
  // taxonomy entry — block until named.
  const blankLabelRow = activeRows.some((r) => !r.label.trim());
  const subInvalid = kind === 'subsample' && (!(subN >= 1) || subN > (scanCount || 0));

  const canExport = !busy && !emptyAfter && !blankLabelRow && !subInvalid;

  // Fetch the real p50/p90 once — it's fixed for the loaded scan, so there's
  // no need to refetch on each Review re-entry (the Review render just gates
  // display on step === 3).
  useEffect(() => {
    let alive = true;
    VoxaAPI.getAccuracy(scene, sessionId)
      .then((a) => { if (alive) setAccuracy(a); })
      .catch((e) => { if (alive) setAccuracyErr(e.message || String(e)); });
    return () => { alive = false; };
  }, [scene, sessionId]);

  // Esc closes.
  useEffect(() => {
    const onKey = (e) => { if (e.key === 'Escape' && !busy) { e.preventDefault(); onClose(); } };
    window.addEventListener('keydown', onKey, true);
    return () => window.removeEventListener('keydown', onKey, true);
  }, [busy, onClose]);

  const toggleInclude = (cid) => {
    if (consumed.has(cid)) return;  // consumed classes are forced-kept
    setIncludeSet((prev) => {
      const next = new Set(prev);
      if (next.has(cid)) next.delete(cid); else next.add(cid);
      return next;
    });
  };

  const addRow = () => setRows((prev) => [
    ...prev,
    { key: `row-${nextRowKey.current++}`, from: new Set(), label: '', color: '#8b5cf6' },
  ]);
  const removeRow = (key) => setRows((prev) => prev.filter((r) => r.key !== key));
  const patchRow = (key, patch) => setRows((prev) => prev.map((r) => (r.key === key ? { ...r, ...patch } : r)));
  const toggleRowSource = (key, cid) => setRows((prev) => prev.map((r) => {
    if (r.key !== key) return r;
    const from = new Set(r.from);
    if (from.has(cid)) from.delete(cid); else from.add(cid);
    return { ...r, from };
  }));

  const doExport = async () => {
    setBusy(true);
    setError(null);
    try {
      const cfg = {
        scene,
        session_id: sessionId,
        resolution: { kind, ...(kind === 'subsample' ? { n: subN } : {}) },
        confirmed_only: confirmedOnly,
        include_classes: includeArr,
        remap: previewRows.map((r) => ({ from: r.from, to: r.to })),
        drop_unlabeled: false,
      };
      const blob = await VoxaAPI.exportLabels(cfg);
      downloadBlob(blob, `scan_labeled_${kind}.zip`);
      onClose();
    } catch (e) {
      setError(errText(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="ew-overlay" onClick={() => !busy && onClose()}>
      <div className="ew-card" onClick={(e) => e.stopPropagation()}>
        <div className="ew-head">
          <b>Export labeled cloud</b>
          <div className="ew-steps">
            {['Resolution', 'Classes', 'Review'].map((label, i) => (
              <span key={label} className={'ew-step' + (step === i + 1 ? ' active' : '')}>
                {i + 1}. {label}
              </span>
            ))}
          </div>
          <button className="ew-x" onClick={() => !busy && onClose()} title="Close (Esc)">✕</button>
        </div>

        <div className="ew-body">
          {step === 1 && (
            <div className="ew-pane">
              <label className="ew-radio">
                <input type="radio" checked={kind === 'scan'} onChange={() => setKind('scan')} />
                <span><b>scan.ply native</b> — {(scanCount || 0).toLocaleString()} pts</span>
              </label>
              <label className="ew-radio">
                <input type="radio" checked={kind === 'subsample'} onChange={() => setKind('subsample')} />
                <span><b>subsample to N</b>
                  <input className="ew-num" type="number" min={1} max={scanCount || undefined}
                    value={subN} disabled={kind !== 'subsample'}
                    onChange={(e) => setSubN(Math.max(1, Math.floor(Number(e.target.value) || 0)))} />
                  <em>down-sample only (≤ scan.ply)</em>
                </span>
              </label>
              {kind === 'subsample' && subInvalid && (
                <div className="ew-warn">N must be between 1 and {(scanCount || 0).toLocaleString()}.</div>
              )}
              <label className={'ew-radio' + (rawSourceAvailable ? '' : ' disabled')}
                title={rawSourceAvailable ? '' : 'No raw source linked for this scan'}>
                <input type="radio" checked={kind === 'raw'} disabled={!rawSourceAvailable}
                  onChange={() => setKind('raw')} />
                <span><b>full raw</b> — full-density source cloud</span>
              </label>
              {kind === 'raw' && (
                <div className="ew-warn">⚠ Raw export is multi-GB and can take minutes to build and download.</div>
              )}
            </div>
          )}

          {step === 2 && (
            <div className="ew-pane">
              <label className="ew-check">
                <input type="checkbox" checked={confirmedOnly}
                  onChange={(e) => setConfirmedOnly(e.target.checked)} />
                <span>Confirmed instances only <em>(unconfirmed points become unlabeled)</em></span>
              </label>

              <div className="ew-sub">Include classes</div>
              <div className="ew-classlist">
                {classes.map((c) => {
                  const merged = consumed.has(c.class_id);
                  return (
                    <label key={c.class_id}
                      className={'ew-classrow' + (merged ? ' locked' : '')}
                      title={merged ? 'In a merge group below — kept and remapped' : ''}>
                      <input type="checkbox"
                        checked={merged || includeSet.has(c.class_id)}
                        disabled={merged}
                        onChange={() => toggleInclude(c.class_id)} />
                      <span className="class-swatch" style={{ background: c.color }} />
                      <span className="ew-classname">{c.label}</span>
                      {merged && <span className="ew-tag">→ merged</span>}
                    </label>
                  );
                })}
              </div>

              <div className="ew-sub">Merge / rename groups</div>
              {rows.map((r) => (
                <div key={r.key} className="ew-merge">
                  <div className="ew-merge-hd">
                    <input className="ew-num ew-label" type="text" placeholder="Target label"
                      value={r.label} onChange={(e) => patchRow(r.key, { label: e.target.value })} />
                    <input className="ew-color" type="color"
                      value={r.color} onChange={(e) => patchRow(r.key, { color: e.target.value })} />
                    <button className="ew-x" onClick={() => removeRow(r.key)} title="Remove group">✕</button>
                  </div>
                  <div className="ew-merge-src">
                    {classes.map((c) => {
                      const inThis = r.from.has(c.class_id);
                      const elsewhere = consumed.has(c.class_id) && !inThis;
                      const excluded = !includeSet.has(c.class_id);
                      const disabled = elsewhere || excluded;
                      return (
                        <label key={c.class_id}
                          className={'ew-chip' + (inThis ? ' on' : '') + (disabled ? ' disabled' : '')}
                          title={excluded ? 'Excluded above' : elsewhere ? 'In another group' : ''}>
                          <input type="checkbox" checked={inThis} disabled={disabled}
                            onChange={() => toggleRowSource(r.key, c.class_id)} />
                          <span className="class-swatch" style={{ background: c.color }} />
                          {c.label}
                        </label>
                      );
                    })}
                  </div>
                </div>
              ))}
              <button className="ghost-btn" onClick={addRow}>+ Add merge group</button>
            </div>
          )}

          {step === 3 && (
            <div className="ew-pane">
              <div className="ew-review">
                <span>Resolution</span>
                <b>{kind}{kind === 'subsample' ? ` (${subN.toLocaleString()})` : ''}</b>
                <span>Estimated points</span>
                <b>{kind === 'raw' ? 'full-density (multi-GB)' : (targetPoints || 0).toLocaleString()}</b>
                <span>Labeled after filters</span>
                <b>{perClassPointCounts ? `~${labeledAfter.toLocaleString()} (at scan density)` : (emptyAfter ? '~0' : 'some')}</b>
                <span>Confirmed only</span>
                <b>{confirmedOnly ? 'yes' : 'no'}</b>
              </div>

              <div className="ew-sub">Target taxonomy</div>
              <div className="ew-classlist">
                {Object.entries(taxonomy).map(([id, t]) => (
                  <div key={id} className="ew-classrow">
                    <span className="class-swatch" style={{ background: t.color }} />
                    <span className="ew-classname">{t.label}</span>
                    <span className="ew-tag">id {id}</span>
                  </div>
                ))}
              </div>

              <div className="ew-accuracy">
                {accuracyErr
                  ? <span className="ew-warn">Accuracy unavailable: {accuracyErr}</span>
                  : accuracy
                    ? <>Semantic boundary accuracy {accuracy.loa} — ±~{(accuracy.p90 * 100).toFixed(1)} cm
                        (p90 sample spacing; p50 {(accuracy.p50 * 100).toFixed(1)} cm; set by
                        labeling density, unchanged by export resolution).
                        Box/pipe boundaries: exact.</>
                    : <span className="ew-faint">Computing accuracy…</span>}
              </div>

              {emptyAfter && (
                <div className="ew-warn">No labeled points survive these filters — nothing to export.</div>
              )}
              {error && <div className="ew-warn">Export error: {error}</div>}
            </div>
          )}
        </div>

        <div className="ew-foot">
          <button className="ghost-btn" onClick={() => (step === 1 ? onClose() : setStep(step - 1))}
            disabled={busy}>
            {step === 1 ? 'Cancel' : 'Back'}
          </button>
          {step < 3
            ? <button className="header-btn primary" onClick={() => setStep(step + 1)}
                disabled={subInvalid}>Next</button>
            : <button className="header-btn primary" onClick={doExport} disabled={!canExport}>
                {busy ? 'Exporting…' : 'Export'}
              </button>}
        </div>
      </div>
    </div>
  );
}
