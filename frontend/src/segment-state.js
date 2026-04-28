export function initSegState({ classFull, instanceFull, isFromPrelabel = false }) {
  return {
    classFull,
    instanceFull,
    summary: deriveSummary(classFull, instanceFull),
    dirty: false,
    selection: new Set(),
    activeTool: 'cuboid',
    brush: { radius: 0.05, mode: 'create', destInstance: null, destClass: 0 },
    isFromPrelabel,
  };
}

export function applyDelta(state, { indices, after_class, after_instance }) {
  for (let k = 0; k < indices.length; k++) {
    state.classFull[indices[k]] = after_class[k];
    state.instanceFull[indices[k]] = after_instance[k];
  }
  return { ...state, summary: deriveSummary(state.classFull, state.instanceFull), dirty: true };
}

export function recomputeSummary(state) {
  return deriveSummary(state.classFull, state.instanceFull);
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
