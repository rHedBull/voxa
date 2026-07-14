import { describe, it, expect } from 'vitest';
import { initSegState, applyDelta, recomputeSummary, computeDiffMask, hydrateFromServerState, reconcilePointsetRows, applyUndoRedoDelta, applySamDelta, reconcileSamAfterApply } from './segment-state.js';

const seed = () => initSegState({
  classFull: new Int8Array([-1, 0, 0, 1, 1, 2, -1, 2]),
  instanceFull: new Int32Array([-1, 0, 0, 1, 1, 2, -1, 2]),
  isFromPrelabel: true,
});

describe('segment-state', () => {
  it('applyDelta patches arrays at given indices', () => {
    const s = seed();
    const next = applyDelta(s, {
      indices: new Int32Array([1, 2]),
      after_class: new Int8Array([2, 2]),
      after_instance: new Int32Array([0, 0]),
    });
    expect(next.classFull[1]).toBe(2);
    expect(next.classFull[2]).toBe(2);
    expect(next.instanceFull[1]).toBe(0);
    expect(next.dirty).toBe(true);
  });

  it('applyDelta retires SAM candidacy for points it labels (any tool, not just SAM confirm)', () => {
    let s = seed();
    s = applySamDelta(s, { indices: [0, 6], samSegId: 5 });
    expect(Array.from(s.samIds)).toEqual([5, -1, -1, -1, -1, -1, 5, -1]);
    // A non-SAM apply (e.g. Box/Draw/Beam/Presegment confirm) labels point 6 —
    // its SAM candidacy must be retired even though this isn't confirmSamSelection.
    const next = applyDelta(s, {
      indices: new Int32Array([6]),
      after_class: new Int8Array([1]),
      after_instance: new Int32Array([3]),
    });
    expect(next.samIds[6]).toBe(-1);
    expect(next.samIds[0]).toBe(5); // untouched point keeps its candidacy
    expect(next.samSegments.get(5)).toEqual({ nPoints: 1, maskScore: null });
  });

  it('applyDelta is a no-op on samIds/samSegments/samSelection when no touched point has SAM candidacy', () => {
    const s = seed();
    const next = applyDelta(s, {
      indices: new Int32Array([1]),
      after_class: new Int8Array([2]),
      after_instance: new Int32Array([0]),
    });
    expect(next.samIds).toBe(s.samIds);
    expect(next.samSegments).toBe(s.samSegments);
    expect(next.samSelection).toBe(s.samSelection);
  });

  it('recomputeSummary derives per-instance counts', () => {
    const s = seed();
    const sum = recomputeSummary(s);
    expect(sum.get(0).nPoints).toBe(2);
    expect(sum.get(1).classId).toBe(1);
  });
});

describe('computeDiffMask', () => {
  it('marks points where class changed', () => {
    const cls     = new Int8Array([0, 1, 2, 0]);
    const pre_cls = new Int8Array([0, 0, 2, 0]);
    const inst     = new Int32Array([0, 1, 2, 3]);
    const pre_inst = new Int32Array([0, 1, 2, 3]);
    const mask = computeDiffMask(cls, pre_cls, inst, pre_inst);
    expect(mask[0]).toBe(0);
    expect(mask[1]).toBe(1);
    expect(mask[2]).toBe(0);
    expect(mask[3]).toBe(0);
  });

  it('marks points where instance changed', () => {
    const cls      = new Int8Array([1, 1, 1]);
    const pre_cls  = new Int8Array([1, 1, 1]);
    const inst     = new Int32Array([0, 99, 2]);
    const pre_inst = new Int32Array([0,  1, 2]);
    const mask = computeDiffMask(cls, pre_cls, inst, pre_inst);
    expect(mask[0]).toBe(0);
    expect(mask[1]).toBe(1);
    expect(mask[2]).toBe(0);
  });

  it('returns all-zero mask when nothing changed', () => {
    const cls  = new Int8Array([0, 1, 2]);
    const inst = new Int32Array([0, 1, 2]);
    const mask = computeDiffMask(cls, cls, inst, inst);
    expect(Array.from(mask)).toEqual([0, 0, 0]);
  });

  it('returns a Uint8Array of the same length as the input', () => {
    const cls  = new Int8Array(5).fill(0);
    const inst = new Int32Array(5).fill(0);
    const mask = computeDiffMask(cls, cls, inst, inst);
    expect(mask).toBeInstanceOf(Uint8Array);
    expect(mask.length).toBe(5);
  });
});

describe('hydrateFromServerState', () => {
  it('pulls preseg fields', () => {
    const state = { presegRunId: null,
                    presegFingerprint: null, sourceFingerprint: null,
                    dirty: false };
    const out = hydrateFromServerState(state, {
      has_seg: true, n_points: 100,
      preseg_id: 'r1', preseg_fingerprint: 'sha256:x',
      source_fingerprint: 'sha256:y',
      is_from_prelabel: false, dirty: true,
    });
    expect(out.presegRunId).toBe('r1');
    expect(out.presegFingerprint).toBe('sha256:x');
    expect(out.dirty).toBe(true);
  });

  it('is a no-op when has_seg is false', () => {
    const state = { hiddenInstIds: new Set([42]), presegRunId: 'old' };
    const out = hydrateFromServerState(state, { has_seg: false });
    expect(out).toBe(state);
  });
});

describe('reconcilePointsetRows', () => {
  const row = (segId, extra = {}) => ({ id: `i${segId}`, kind: 'pointset', segId, ...extra });
  const summaryOf = (...ids) => new Map(ids.map((id) => [id, { nPoints: 1 }]));

  it('prunes a pointset row whose touched segId vanished (undo)', () => {
    const rows = [row(3), row(4)];
    const dormant = new Map();
    const out = reconcilePointsetRows(rows, summaryOf(4), new Set([3]), dormant);
    expect(out.map((r) => r.segId)).toEqual([4]);
    expect(dormant.get(3)).toBe(rows[0]);
  });

  it('returns null when the touched id still has points (partial undo)', () => {
    const rows = [row(3)];
    expect(reconcilePointsetRows(rows, summaryOf(3), new Set([3]), new Map())).toBeNull();
  });

  it('revives a dormant row when its id reappears (redo)', () => {
    const undone = row(3);
    const dormant = new Map([[3, undone]]);
    const rows = [row(4)];
    const out = reconcilePointsetRows(rows, summaryOf(3, 4), new Set([3]), dormant);
    expect(out).toEqual([rows[0], undone]);
    expect(dormant.has(3)).toBe(false);
  });

  it('never touches rows outside touchedIds, even with zero points', () => {
    // An instance fully overwritten by a later apply is NOT an undone one:
    // its OBB volume must survive for raw-density replay.
    const rows = [row(3)];
    expect(reconcilePointsetRows(rows, summaryOf(7), new Set([7]), new Map())).toBeNull();
  });

  it('ignores non-pointset (legacy cuboid) rows', () => {
    const rows = [{ id: 'c1', kind: 'cuboid', segId: 3 }];
    expect(reconcilePointsetRows(rows, summaryOf(), new Set([3]), new Map())).toBeNull();
  });

  it('is a no-op for an empty touched set', () => {
    expect(reconcilePointsetRows([row(3)], summaryOf(), new Set(), new Map())).toBeNull();
  });
});

describe('applyUndoRedoDelta', () => {
  it('applies the delta and prunes the undone row, then revives it on redo', () => {
    // Point 7 was labeled by apply #3 (a fresh box). Undo erases it.
    const state = initSegState({
      classFull: new Int8Array([-1, 0, 0, 1, 1, 2, -1, 2]),
      instanceFull: new Int32Array([-1, 0, 0, 1, 1, 2, -1, 3]),
    });
    const boxRow = { id: 'i3', kind: 'pointset', segId: 3, center: [0, 0, 0] };
    const rows = [{ id: 'i1', kind: 'pointset', segId: 1 }, boxRow];

    const undo = applyUndoRedoDelta(state, {
      indices: new Int32Array([7]),
      after_class: new Int8Array([-1]),
      after_instance: new Int32Array([-1]),
    }, rows);
    expect(undo.next.instanceFull[7]).toBe(-1);
    expect(undo.rows.map((r) => r.segId)).toEqual([1]);
    expect(state.dormant.get(3)).toBe(boxRow);

    const redo = applyUndoRedoDelta(undo.next, {
      indices: new Int32Array([7]),
      after_class: new Int8Array([2]),
      after_instance: new Int32Array([3]),
    }, undo.rows);
    expect(redo.next.instanceFull[7]).toBe(3);
    expect(redo.rows.map((r) => r.segId)).toEqual([1, 3]);
    expect(state.dormant.has(3)).toBe(false);
  });

  it('returns rows=null when the delta does not change row liveness', () => {
    const state = initSegState({
      classFull: new Int8Array([0, 0]),
      instanceFull: new Int32Array([1, 1]),
    });
    const out = applyUndoRedoDelta(state, {
      indices: new Int32Array([0]),
      after_class: new Int8Array([2]),
      after_instance: new Int32Array([1]),
    }, [{ id: 'i1', kind: 'pointset', segId: 1 }]);
    expect(out.rows).toBeNull();
  });
});

describe('SAM candidate layer', () => {
  it('initSegState defaults samIds to all -1 and samSegments/samSelection empty', () => {
    const classFull = new Int8Array([-1, -1, -1]);
    const instanceFull = new Int32Array([-1, -1, -1]);
    const s = initSegState({ classFull, instanceFull });
    expect(Array.from(s.samIds)).toEqual([-1, -1, -1]);
    expect(s.samSegments.size).toBe(0);
    expect(s.samSelection.size).toBe(0);
  });

  it('initSegState hydrates samIds/samSegments when provided', () => {
    const classFull = new Int8Array([-1, -1]);
    const instanceFull = new Int32Array([-1, -1]);
    const samIds = new Int32Array([3, 3]);
    const s = initSegState({
      classFull, instanceFull, samIds,
      samSegments: [{ id: 3, n_points: 2, mask_score: 0.5 }],
    });
    expect(Array.from(s.samIds)).toEqual([3, 3]);
    expect(s.samSegments.get(3)).toEqual({ nPoints: 2, maskScore: 0.5 });
  });

  it('applySamDelta writes the new sam id at the given indices', () => {
    const classFull = new Int8Array([-1, -1, -1]);
    const instanceFull = new Int32Array([-1, -1, -1]);
    const s = initSegState({ classFull, instanceFull });
    const next = applySamDelta(s, { indices: [0, 2], samSegId: 5 });
    expect(Array.from(next.samIds)).toEqual([5, -1, 5]);
    expect(next.samSegments.get(5)).toEqual({ nPoints: 2, maskScore: null });
  });

  it('applySamDelta shrinks an overlapping older candidate', () => {
    const classFull = new Int8Array([-1, -1]);
    const instanceFull = new Int32Array([-1, -1]);
    let s = initSegState({ classFull, instanceFull });
    s = applySamDelta(s, { indices: [0, 1], samSegId: 1 });
    s = applySamDelta(s, { indices: [1], samSegId: 2 });   // overlaps id 1 at index 1
    expect(Array.from(s.samIds)).toEqual([1, 2]);
    expect(s.samSegments.get(1)).toEqual({ nPoints: 1, maskScore: null });
    expect(s.samSegments.get(2)).toEqual({ nPoints: 1, maskScore: null });
  });

  it('applySamDelta drops a fully-overlapped older candidate', () => {
    const classFull = new Int8Array([-1]);
    const instanceFull = new Int32Array([-1]);
    let s = initSegState({ classFull, instanceFull });
    s = applySamDelta(s, { indices: [0], samSegId: 1 });
    s = applySamDelta(s, { indices: [0], samSegId: 2 });
    expect(s.samSegments.has(1)).toBe(false);
    expect(s.samSegments.get(2)).toEqual({ nPoints: 1, maskScore: null });
  });

  it('reconcileSamAfterApply clears samIds and removes samSegments/samSelection entries', () => {
    const classFull = new Int8Array([-1, -1]);
    const instanceFull = new Int32Array([-1, -1]);
    let s = initSegState({ classFull, instanceFull });
    s = applySamDelta(s, { indices: [0, 1], samSegId: 4 });
    s = { ...s, samSelection: new Set([4]) };
    const next = reconcileSamAfterApply(s, new Set([4]));
    expect(Array.from(next.samIds)).toEqual([-1, -1]);
    expect(next.samSegments.has(4)).toBe(false);
    expect(next.samSelection.has(4)).toBe(false);
  });

  it('reconcileSamAfterApply is a no-op for an empty id set', () => {
    const classFull = new Int8Array([-1]);
    const instanceFull = new Int32Array([-1]);
    const s = initSegState({ classFull, instanceFull });
    expect(reconcileSamAfterApply(s, new Set())).toBe(s);
  });

  it('initSegState threads the source tag through hydrated samSegments', () => {
    const classFull = new Int8Array([-1, -1]);
    const instanceFull = new Int32Array([-1, -1]);
    const samIds = new Int32Array([3, 3]);
    const s = initSegState({
      classFull, instanceFull, samIds,
      samSegments: [{ id: 3, n_points: 2, mask_score: 0.5, source: 'preseg' }],
    });
    expect(s.samSegments.get(3)).toEqual({ nPoints: 2, maskScore: 0.5, source: 'preseg' });
  });

  it('applySamDelta stores the source tag on the samSegments entry', () => {
    const classFull = new Int8Array([-1, -1, -1]);
    const instanceFull = new Int32Array([-1, -1, -1]);
    const s = initSegState({ classFull, instanceFull });
    const next = applySamDelta(s, { indices: [0, 1], samSegId: 5, source: 'preseg' });
    expect(next.samSegments.get(5).source).toBe('preseg');
  });
});
