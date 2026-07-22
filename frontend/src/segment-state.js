export function initSegState({
  classFull, instanceFull, isFromPrelabel = false,
  segBoxes = null, segHulls = null,
  samIds = null, samSegments = [], categories = null,
}) {
  return {
    classFull,
    instanceFull,
    // Annotation-status axis (phase 2): 0 = none, else artifact/transient/
    // excluded_review. Defaults to zeros so a pre-phase-2 session (or an
    // older backend that sends no categories) degrades to "nothing marked".
    categoryFull: categories || new Int8Array(classFull.length),
    summary: deriveSummary(classFull, instanceFull),
    dirty: false,
    selection: new Set(),
    activeTool: 'cuboid',
    brush: { radius: 0.05, mode: 'create', destInstance: null, destClass: 0 },
    isFromPrelabel,
    segBoxes,  // { segIds, segCenters, segSizes } — kept for fallback / metrics
    segHulls,  // { vertices: Float32Array, faces: Int32Array, faceSeg: Int32Array }
    // Pointset rows removed by an undo, keyed by segId, so a redo revives the
    // row (with its OBB/centerline volume + seq) instead of orphaning points.
    // Living on the state keeps it session-scoped structurally: a scene or
    // session switch rebuilds the state and drops these with it.
    dormant: new Map(),
    // SAM candidate layer — parallel to instanceFull, never merged into it.
    // A point can carry a sam id with no class (unlike instanceFull, which
    // always pairs with a real classFull entry once >= 0).
    samIds: samIds || new Int32Array(classFull.length).fill(-1),
    samSegments: new Map(samSegments.map((s) =>
      [s.id, { nPoints: s.n_points, maskScore: s.mask_score ?? null, source: s.source }])),
    samSelection: new Set(),
  };
}

export function applyDelta(state, { indices, after_class, after_instance, after_category }) {
  for (let k = 0; k < indices.length; k++) {
    state.classFull[indices[k]] = after_class[k];
    state.instanceFull[indices[k]] = after_instance[k];
    // Absent only when talking to a pre-phase-2 backend; the working arrays
    // then simply keep whatever categories they had.
    if (after_category && state.categoryFull) {
      state.categoryFull[indices[k]] = after_category[k];
    }
  }
  const retired = retireSamIdsForIndices(state, indices);
  return { ...retired, summary: deriveSummary(state.classFull, state.instanceFull), dirty: true };
}

// Any apply (box/draw/beam/presegment/SAM-confirm, and undo/redo replaying a
// delta) retires SAM candidacy for the points it touches — mirrors the
// backend's SegmentSession._apply calling _retire_sam_ids unconditionally.
// Without this, segState.samIds/samSegments would go stale (phantom
// candidates in the cyan overlay/SAM segments list) the moment another tool
// labels over points a SAM mask had claimed. No-ops (returns state as-is,
// no copy) when none of the indices currently carry a live sam id.
export function retireSamIdsForIndices(state, indices) {
  const shrink = new Map();
  for (let k = 0; k < indices.length; k++) {
    const old = state.samIds[indices[k]];
    if (old >= 0) shrink.set(old, (shrink.get(old) || 0) + 1);
  }
  if (shrink.size === 0) return state;
  const samIds = state.samIds.slice();
  for (let k = 0; k < indices.length; k++) {
    if (samIds[indices[k]] >= 0) samIds[indices[k]] = -1;
  }
  const samSegments = new Map(state.samSegments);
  const samSelection = new Set(state.samSelection);
  for (const [oldId, removed] of shrink) {
    const entry = samSegments.get(oldId);
    if (!entry) continue;
    const nPoints = entry.nPoints - removed;
    if (nPoints <= 0) { samSegments.delete(oldId); samSelection.delete(oldId); }
    else samSegments.set(oldId, { ...entry, nPoints });
  }
  return { ...state, samIds, samSegments, samSelection };
}

// The SAM-layer analogue of applyDelta: writes samSegId at each index and
// shrinks/drops any older candidate those indices used to belong to
// (last-materialize-wins, mirrors SegmentSession._retire_sam_ids).
export function applySamDelta(state, { indices, samSegId, source }) {
  const samIds = state.samIds.slice();
  const shrink = new Map();
  for (let k = 0; k < indices.length; k++) {
    const old = samIds[indices[k]];
    if (old >= 0 && old !== samSegId) shrink.set(old, (shrink.get(old) || 0) + 1);
    samIds[indices[k]] = samSegId;
  }
  const samSegments = new Map(state.samSegments);
  for (const [oldId, removed] of shrink) {
    const entry = samSegments.get(oldId);
    if (!entry) continue;
    const nPoints = entry.nPoints - removed;
    if (nPoints <= 0) samSegments.delete(oldId);
    else samSegments.set(oldId, { ...entry, nPoints });
  }
  samSegments.set(samSegId, { nPoints: indices.length, maskScore: null, source });
  return { ...state, samIds, samSegments };
}

// After classifying a SAM selection, retire the absorbed candidates: clear
// their samIds entries and drop them from samSegments/samSelection so they
// vanish from the SAM segment list (mirrors presegments disappearing via
// promotedSegIds once absorbed into an instance).
export function reconcileSamAfterApply(state, appliedSamSegIds) {
  if (!appliedSamSegIds || appliedSamSegIds.size === 0) return state;
  const samIds = state.samIds.slice();
  for (let p = 0; p < samIds.length; p++) {
    if (appliedSamSegIds.has(samIds[p])) samIds[p] = -1;
  }
  const samSegments = new Map(state.samSegments);
  const samSelection = new Set(state.samSelection);
  for (const id of appliedSamSegIds) {
    samSegments.delete(id);
    samSelection.delete(id);
  }
  return { ...state, samIds, samSegments, samSelection };
}

// samSelection is shared by the SAM tool and the Presegment tool (cut
// candidates tagged source:'preseg' render/select from either), but a
// candidate whose source tag doesn't match the tool being entered must NOT
// survive the switch — otherwise a forgotten selection from the OTHER tool
// silently gets classified alongside whatever the user actually selects
// after switching (mode-label.jsx's classify-gating checks samSelection
// before segState.selection, and both confirmSamSelection and the viewport
// recolor/pick logic operate on the raw, unfiltered samSelection). This is
// symmetric: SAM -> Presegment drops source:'sam' ids, and Presegment -> SAM
// drops source:'preseg' ids. Every other tool-switch case already fully
// clears samSelection (see mode-label.jsx's tool-switch effect).
export function filterSamSelectionOnToolSwitch(samSelection, samSegments, fromTool, toTool) {
  const samToPreseg = fromTool === 'sam' && toTool === 'presegment';
  const presegToSam = fromTool === 'presegment' && toTool === 'sam';
  if (!samToPreseg && !presegToSam) return samSelection;
  const keepSource = samToPreseg ? 'preseg' : 'sam';
  const next = new Set();
  for (const id of samSelection) {
    if (samSegments.get(id)?.source === keepSource) next.add(id);
  }
  return next;
}

export function recomputeSummary(state) {
  return deriveSummary(state.classFull, state.instanceFull);
}

export function computeDiffMask(classFull, prelabelClass, instanceFull, prelabelInstance) {
  const n = classFull.length;
  const mask = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    if (classFull[i] !== prelabelClass[i] || instanceFull[i] !== prelabelInstance[i]) {
      mask[i] = 1;
    }
  }
  return mask;
}

export function hydrateFromServerState(state, payload) {
  if (!payload || !payload.has_seg) return state;
  return {
    ...state,
    presegRunId: payload.preseg_id ?? null,
    presegFingerprint: payload.preseg_fingerprint ?? null,
    sourceFingerprint: payload.source_fingerprint ?? null,
    dirty: !!payload.dirty,
  };
}

// After an undo/redo delta, sync the Instances-panel rows with the working
// arrays: a pointset row whose instance id vanished (its apply was undone)
// moves into `dormant` — otherwise its persisted OBB/centerline volume would
// replay into raw-density exports the user can't see — and comes back from
// `dormant` when a redo restores the id. Only ids in `touchedIds` (the ones
// the delta wrote) are considered: an id merely overwritten by a later apply
// is not an undone one, and its volume must survive for export replay.
// `summary` (from deriveSummary) answers liveness in O(1); returns the new
// rows array, or null when nothing changed.
export function reconcilePointsetRows(rows, summary, touchedIds, dormant) {
  if (!touchedIds || touchedIds.size === 0) return null;
  let changed = false;
  const kept = [];
  for (const row of rows) {
    if (row.kind === 'pointset' && touchedIds.has(row.segId) && !summary.has(row.segId)) {
      dormant.set(row.segId, row);
      changed = true;
    } else {
      kept.push(row);
    }
  }
  for (const id of touchedIds) {
    if (summary.has(id) && dormant.has(id)) {
      kept.push(dormant.get(id));
      dormant.delete(id);
      changed = true;
    }
  }
  return changed ? kept : null;
}

// One undo/redo step: apply the delta to the working arrays AND reconcile the
// Instances-panel rows. Bundled because the touched-set must read the
// instance values BEFORE applyDelta mutates them in place — an ordering
// invariant no caller should have to know. Returns { next, rows } where
// `rows` is the reconciled row list or null when the rows are unchanged.
export function applyUndoRedoDelta(state, { indices, after_class, after_instance, after_category }, rows) {
  const touched = new Set();
  for (let k = 0; k < indices.length; k++) {
    touched.add(state.instanceFull[indices[k]]);
    touched.add(after_instance[k]);
  }
  touched.delete(-1);
  const next = applyDelta(state, { indices, after_class, after_instance, after_category });
  return { next, rows: reconcilePointsetRows(rows, next.summary, touched, state.dormant) };
}

function deriveSummary(cls, inst) {
  const m = new Map();
  for (let i = 0; i < inst.length; i++) {
    const id = inst[i];
    if (id < 0) continue;
    const e = m.get(id);
    if (e === undefined) m.set(id, { classId: cls[i], nPoints: 1 });
    else e.nPoints += 1;
  }
  return m;
}
