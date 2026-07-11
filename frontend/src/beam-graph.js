// beam-graph.js — pure state machine for the Beam sub-mode of Label mode.
// No Three.js, no React (same testability contract as draw-paths.js).
// Spec: docs/superpowers/specs/2026-07-10-beam-structure-labeling-design.md.
//
// State shape:
// {
//   nodes:     [{ id, pos: [x,y,z] }],                        // recentered frame
//   edges:     [{ id, a, b, width, classId, instanceId|null, dirty }],
//   committed: [{ a:[x,y,z], b:[x,y,z], width, classId, instanceId }],
//   selection: { kind: 'node'|'edge', id } | null,
//   lastWidth: number,             // a new edge inherits the last-used width
//   nextId:    number,             // monotonic id source (deterministic, testable)
// }
// classId is the canonical numeric class id (ClassDef.class_id), never the
// array position. `dirty` marks an edge whose geometry/width/class changed
// since its last apply — Enter re-applies only dirty or never-applied edges.

export const MIN_WIDTH = 0.01;
const DEGENERATE_EPS = 1e-6;

export function initBeamState({ defaultWidth = 0.2 } = {}) {
  return {
    nodes: [], edges: [], committed: [],
    selection: null, lastWidth: defaultWidth, nextId: 1,
  };
}

export function nodePos(state, id) {
  const n = state.nodes.find((x) => x.id === id);
  if (!n) throw new Error(`beam-graph: no node with id ${id}`);
  return n.pos;
}

export function addNode(state, pos) {
  // A new node is NOT auto-selected (spec: workflow step 2) — placing a run
  // of joints must not chain accidental beams.
  const node = { id: state.nextId, pos };
  return { ...state, nodes: [...state.nodes, node], nextId: state.nextId + 1 };
}

export function selectNode(state, id) {
  return { ...state, selection: { kind: 'node', id } };
}

export function selectEdge(state, id) {
  return { ...state, selection: { kind: 'edge', id } };
}

export function clearSelection(state) {
  return { ...state, selection: null };
}

// Ctrl+click on an existing node: with a *different* node selected, connect
// them into a beam (selecting it); otherwise select the clicked node.
export function clickNode(state, nodeId, defaultClassId) {
  const sel = state.selection;
  if (sel?.kind === 'node' && sel.id !== nodeId) {
    return addEdge(state, sel.id, nodeId, defaultClassId);
  }
  return selectNode(state, nodeId);
}

function addEdge(state, aId, bId, classId) {
  // Connecting an already-connected pair is a select, not a duplicate.
  const existing = state.edges.find(
    (e) => (e.a === aId && e.b === bId) || (e.a === bId && e.b === aId));
  if (existing) return selectEdge(state, existing.id);
  const edge = {
    id: state.nextId, a: aId, b: bId, width: state.lastWidth,
    classId, instanceId: null, dirty: true,
  };
  return {
    ...state, edges: [...state.edges, edge],
    selection: { kind: 'edge', id: edge.id }, nextId: state.nextId + 1,
  };
}

export function moveNode(state, nodeId, pos) {
  const nodes = state.nodes.map((n) => (n.id === nodeId ? { ...n, pos } : n));
  // Incident beams follow the joint; they need a re-apply to re-extract.
  const edges = state.edges.map((e) =>
    e.a === nodeId || e.b === nodeId ? { ...e, dirty: true } : e);
  return { ...state, nodes, edges };
}

function deleteNode(state, nodeId) {
  return {
    ...state,
    nodes: state.nodes.filter((n) => n.id !== nodeId),
    edges: state.edges.filter((e) => e.a !== nodeId && e.b !== nodeId),
    selection: null,
  };
}

function deleteEdge(state, edgeId) {
  return {
    ...state,
    edges: state.edges.filter((e) => e.id !== edgeId),
    selection: null,
  };
}

export function deleteSelected(state) {
  const sel = state.selection;
  if (!sel) return state;
  return sel.kind === 'node' ? deleteNode(state, sel.id) : deleteEdge(state, sel.id);
}

export function setWidth(state, w) {
  const width = Math.max(w, MIN_WIDTH);
  if (state.selection?.kind !== 'edge') {
    // No beam selected: the field still sets the width new edges inherit.
    return { ...state, lastWidth: width };
  }
  const edges = state.edges.map((e) =>
    e.id === state.selection.id ? { ...e, width, dirty: true } : e);
  return { ...state, edges, lastWidth: width };
}

export function nudgeWidth(state, dir) {
  if (state.selection?.kind !== 'edge') return state;
  const e = state.edges.find((x) => x.id === state.selection.id);
  // Multiplicative steps feel uniform across member sizes (8%, like Draw).
  return setWidth(state, e.width * (1 + Math.sign(dir) * 0.08));
}

export function setClass(state, classId) {
  if (state.selection?.kind !== 'edge') return state;
  const edges = state.edges.map((e) =>
    e.id === state.selection.id ? { ...e, classId, dirty: true } : e);
  return { ...state, edges };
}

function isDegenerate(state, edge) {
  const a = nodePos(state, edge.a), b = nodePos(state, edge.b);
  return Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]) < DEGENERATE_EPS;
}

// Edges Enter should apply: never applied, or edited since the last apply.
// Degenerate (coincident endpoints) edges are skipped — the client-side guard
// from the spec's error handling.
export function applyTargets(state) {
  return state.edges.filter(
    (e) => (e.dirty || e.instanceId == null) && !isDegenerate(state, e));
}

export function markApplied(state, edgeId, instanceId) {
  const edges = state.edges.map((e) =>
    e.id === edgeId ? { ...e, instanceId, dirty: false } : e);
  return { ...state, edges };
}

// Ctrl+Enter: retire every applied edge into the committed layer (bake its
// endpoint positions — the graph structure is not needed after commit) and
// drop it from the active graph. Edges that never got an instance (apply
// failed / 0 points) stay active so the user can fix them; nodes with no
// remaining edges are dropped (canvas clears).
export function commitAll(state) {
  const committed = [...state.committed];
  const remaining = [];
  for (const e of state.edges) {
    if (e.instanceId == null) { remaining.push(e); continue; }
    committed.push({
      a: [...nodePos(state, e.a)], b: [...nodePos(state, e.b)],
      width: e.width, classId: e.classId, instanceId: e.instanceId,
    });
  }
  const used = new Set(remaining.flatMap((e) => [e.a, e.b]));
  return {
    ...state,
    nodes: state.nodes.filter((n) => used.has(n.id)),
    edges: remaining, committed, selection: null,
  };
}

// ── OBB math ────────────────────────────────────────────────────────────────
// A beam's swept square box is expressed as the codebase-standard OBB
// {center, size, rotation} so the existing apply-shape endpoint, viewer
// preview conventions, and export replay (materialize.collect_volumes) all
// consume it unchanged. Rotation is Euler XYZ whose Rx·Ry·Rz composition is
// the frame matrix — the one convention every containment test shares (see
// CLAUDE.md "Coordinate system"). Roll about the axis is unspecified by
// design (square section): any stable (v, w) pair is acceptable.

const sub = (p, q) => [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
const cross = (p, q) => [
  p[1] * q[2] - p[2] * q[1],
  p[2] * q[0] - p[0] * q[2],
  p[0] * q[1] - p[1] * q[0],
];
const norm = (p) => Math.hypot(p[0], p[1], p[2]);
const scale = (p, s) => [p[0] * s, p[1] * s, p[2] * s];

export function beamFrame(a, b) {
  const d = sub(b, a);
  const len = norm(d);
  const u = scale(d, 1 / len);
  // Reference: the world axis least aligned with u — the cross product can't
  // degenerate. Mirrored by _beam_frame in backend/tests/test_shapes.py.
  const abs = [Math.abs(u[0]), Math.abs(u[1]), Math.abs(u[2])];
  const ref = abs[0] <= abs[1] && abs[0] <= abs[2] ? [1, 0, 0]
    : abs[1] <= abs[2] ? [0, 1, 0] : [0, 0, 1];
  const v0 = cross(ref, u);
  const v = scale(v0, 1 / norm(v0));
  const w = cross(u, v);
  return { u, v, w, len };
}

// Euler XYZ (radians) whose Rx·Ry·Rz composition equals the rotation matrix
// with COLUMNS (u, v, w). Same algorithm as
// THREE.Euler.setFromRotationMatrix(m, 'XYZ') — parity-tested against THREE
// in beam-graph.test.js and against obb_indices in test_shapes.py.
export function eulerXYZFromBasis(u, v, w) {
  const m11 = u[0], m12 = v[0], m13 = w[0];
  const m22 = v[1], m23 = w[1];
  const m32 = v[2], m33 = w[2];
  const y = Math.asin(Math.min(1, Math.max(-1, m13)));
  if (Math.abs(m13) < 0.9999999) {
    return [Math.atan2(-m23, m33), y, Math.atan2(-m12, m11)];
  }
  return [Math.atan2(m32, m22), y, 0];
}

export function obbForEdge(state, edge) {
  const a = nodePos(state, edge.a);
  const b = nodePos(state, edge.b);
  const { u, v, w, len } = beamFrame(a, b);
  return {
    center: [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2],
    size: [len, edge.width, edge.width],
    rotation: eulerXYZFromBasis(u, v, w),
  };
}

// ── Persistence (structure.json) ────────────────────────────────────────────

export function toStructureDoc(state) {
  return {
    nodes: state.nodes.map((n) => ({ id: n.id, pos: n.pos })),
    edges: state.edges.map((e) => ({
      id: e.id, a: e.a, b: e.b, width: e.width,
      class_id: e.classId, instance_id: e.instanceId, dirty: e.dirty,
    })),
    committed_beams: state.committed.map((c) => ({
      a: c.a, b: c.b, width: c.width,
      class_id: c.classId, instance_id: c.instanceId,
    })),
  };
}

export function seedFromServer(state, doc) {
  let maxId = 0;
  const nodes = (doc.nodes ?? []).map((n) => {
    maxId = Math.max(maxId, n.id);
    return { id: n.id, pos: n.pos };
  });
  const edges = (doc.edges ?? []).map((e) => {
    maxId = Math.max(maxId, e.id);
    return {
      id: e.id, a: e.a, b: e.b, width: e.width, classId: e.class_id,
      instanceId: e.instance_id ?? null,
      // Older docs without a dirty flag: an unapplied edge needs an apply.
      dirty: e.dirty ?? e.instance_id == null,
    };
  });
  const committed = (doc.committed_beams ?? []).map((c) => ({
    a: c.a, b: c.b, width: c.width, classId: c.class_id, instanceId: c.instance_id,
  }));
  return { ...state, nodes, edges, committed, nextId: Math.max(state.nextId, maxId + 1) };
}
