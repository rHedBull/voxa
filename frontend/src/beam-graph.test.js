import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import {
  initBeamState, addNode, clickNode, selectNode, selectEdge, clearSelection,
  moveNode, deleteSelected, setWidth, nudgeWidth, setClass,
  applyTargets, markApplied, commitAll, beamFrame, eulerXYZFromBasis,
  obbForEdge, toStructureDoc, seedFromServer, MIN_WIDTH,
} from './beam-graph.js';

const CLS = 10;

function twoNodes() {
  let s = initBeamState();
  s = addNode(s, [0, 0, 0]);
  s = addNode(s, [2, 0, 0]);
  return s;
}

function oneEdge() {
  let s = twoNodes();
  s = clickNode(s, 1, CLS);          // select node 1
  s = clickNode(s, 2, CLS);          // connect 1-2 -> edge selected
  return s;
}

describe('beam-graph: nodes & edges', () => {
  it('addNode appends without selecting (spec: new node not auto-selected)', () => {
    const s = addNode(initBeamState(), [1, 2, 3]);
    expect(s.nodes).toEqual([{ id: 1, pos: [1, 2, 3] }]);
    expect(s.selection).toBeNull();
  });

  it('clickNode selects; clicking a different node connects and selects the edge', () => {
    let s = twoNodes();
    s = clickNode(s, 1, CLS);
    expect(s.selection).toEqual({ kind: 'node', id: 1 });
    s = clickNode(s, 2, CLS);
    expect(s.edges).toHaveLength(1);
    expect(s.edges[0]).toMatchObject({ a: 1, b: 2, classId: CLS, instanceId: null, dirty: true });
    expect(s.edges[0].width).toBe(s.lastWidth);
    expect(s.selection).toEqual({ kind: 'edge', id: s.edges[0].id });
  });

  it('clicking the selected node again keeps it selected (no self-edge)', () => {
    let s = clickNode(twoNodes(), 1, CLS);
    s = clickNode(s, 1, CLS);
    expect(s.edges).toHaveLength(0);
    expect(s.selection).toEqual({ kind: 'node', id: 1 });
  });

  it('connecting an already-connected pair selects the existing edge (de-dup, both orders)', () => {
    let s = oneEdge();
    const edgeId = s.edges[0].id;
    s = clickNode(s, 2, CLS);          // select node 2
    s = clickNode(s, 1, CLS);          // reverse order
    expect(s.edges).toHaveLength(1);
    expect(s.selection).toEqual({ kind: 'edge', id: edgeId });
  });

  it('moveNode updates pos and dirties incident edges only', () => {
    let s = oneEdge();
    // Nodes and edges share one nextId counter: after oneEdge() (nodes 1, 2 +
    // edge 3) the next node is id 4.
    s = addNode(s, [0, 5, 0]);
    s = selectNode(s, 4);
    s = clickNode(s, 1, CLS);          // second edge 4-1
    s = markApplied(markApplied(s, s.edges[0].id, 100), s.edges[1].id, 101);
    s = moveNode(s, 2, [3, 0, 0]);
    expect(s.nodes.find((n) => n.id === 2).pos).toEqual([3, 0, 0]);
    expect(s.edges.find((e) => e.a === 1 && e.b === 2).dirty).toBe(true);
    expect(s.edges.find((e) => e.a === 4).dirty).toBe(false);
  });

  it('deleteSelected on a node cascades incident edges; on an edge removes just it', () => {
    let s = oneEdge();
    s = selectNode(s, 1);
    s = deleteSelected(s);
    expect(s.nodes.map((n) => n.id)).toEqual([2]);
    expect(s.edges).toHaveLength(0);
    expect(s.selection).toBeNull();

    let t = oneEdge();                 // edge already selected
    t = deleteSelected(t);
    expect(t.edges).toHaveLength(0);
    expect(t.nodes).toHaveLength(2);   // nodes survive edge deletion
  });
});

describe('beam-graph: width & class', () => {
  it('setWidth targets the selected edge, updates lastWidth, clamps to MIN_WIDTH', () => {
    let s = setWidth(oneEdge(), 0.5);
    expect(s.edges[0].width).toBe(0.5);
    expect(s.lastWidth).toBe(0.5);
    s = setWidth(s, 0);
    expect(s.edges[0].width).toBe(MIN_WIDTH);
  });

  it('setWidth with no edge selected only updates lastWidth (next-edge default)', () => {
    let s = setWidth(clearSelection(oneEdge()), 0.7);
    expect(s.edges[0].width).not.toBe(0.7);
    expect(s.lastWidth).toBe(0.7);
    s = addNode(s, [0, 3, 0]);                   // node 4 (shared id counter)
    s = clickNode(selectNode(s, 1), 4, CLS);
    expect(s.edges.at(-1).width).toBe(0.7);
  });

  it('nudgeWidth is multiplicative on the selected edge, no-op otherwise', () => {
    let s = oneEdge();
    const w0 = s.edges[0].width;
    s = nudgeWidth(s, +1);
    expect(s.edges[0].width).toBeCloseTo(w0 * 1.08);
    const t = nudgeWidth(clearSelection(s), +1);
    expect(t.edges[0].width).toBe(s.edges[0].width);
  });

  it('setClass retargets the selected edge and dirties it', () => {
    let s = markApplied(oneEdge(), oneEdge().edges[0].id, 100);
    s = setClass(s, 11);
    expect(s.edges[0]).toMatchObject({ classId: 11, dirty: true });
  });
});

describe('beam-graph: apply & commit', () => {
  it('applyTargets returns unapplied or dirty edges, skipping degenerate ones', () => {
    let s = oneEdge();
    expect(applyTargets(s).map((e) => e.id)).toEqual([s.edges[0].id]);
    s = markApplied(s, s.edges[0].id, 100);
    expect(applyTargets(s)).toEqual([]);
    s = moveNode(s, 2, [4, 0, 0]);
    expect(applyTargets(s)).toHaveLength(1);
    // Degenerate: both endpoints coincide.
    s = moveNode(s, 2, [0, 0, 0]);
    expect(applyTargets(s)).toEqual([]);
  });

  it('markApplied stores the instance id and clears dirty', () => {
    let s = oneEdge();
    s = markApplied(s, s.edges[0].id, 42);
    expect(s.edges[0]).toMatchObject({ instanceId: 42, dirty: false });
  });

  it('commitAll retires applied edges (baked endpoints), keeps failed ones + their nodes', () => {
    let s = oneEdge();
    s = addNode(s, [0, 5, 0]);                   // node 4 (shared id counter)
    s = clickNode(selectNode(s, 1), 4, CLS);     // second edge 1-4, unapplied
    s = markApplied(s, s.edges[0].id, 100);
    s = commitAll(s);
    expect(s.committed).toEqual([{
      a: [0, 0, 0], b: [2, 0, 0], width: s.lastWidth, classId: CLS, instanceId: 100,
    }]);
    expect(s.edges).toHaveLength(1);              // the unapplied one stays
    expect(s.nodes.map((n) => n.id).sort()).toEqual([1, 4]);  // node 2 dropped
    expect(s.selection).toBeNull();
  });
});

describe('beam-graph: OBB math', () => {
  it('beamFrame is orthonormal with u along the axis, for skew and axis-aligned axes', () => {
    for (const [a, b] of [
      [[0.5, -0.2, 1.0], [2.0, 1.3, -0.4]],
      [[0, 0, 0], [1, 0, 0]],
      [[0, 0, 0], [0, 3, 0]],
      [[1, 1, 1], [1, 1, 5]],
    ]) {
      const { u, v, w, len } = beamFrame(a, b);
      const dot = (p, q) => p[0] * q[0] + p[1] * q[1] + p[2] * q[2];
      expect(len).toBeCloseTo(Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]));
      for (const vec of [u, v, w]) expect(dot(vec, vec)).toBeCloseTo(1);
      expect(dot(u, v)).toBeCloseTo(0);
      expect(dot(u, w)).toBeCloseTo(0);
      expect(dot(v, w)).toBeCloseTo(0);
      const d = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
      expect(dot(u, d)).toBeCloseTo(len);
    }
  });

  it('eulerXYZFromBasis reconstructs the basis via THREE Euler XYZ (Rx·Ry·Rz parity)', () => {
    for (const [a, b] of [
      [[0.5, -0.2, 1.0], [2.0, 1.3, -0.4]],
      [[0, 0, 0], [1, 0, 0]],
      [[0, 0, 0], [0, 0, -2]],       // near the m13 = ±1 gimbal branch
      [[-1, 2, 0.3], [0.7, 2.1, 4]],
    ]) {
      const { u, v, w } = beamFrame(a, b);
      const [rx, ry, rz] = eulerXYZFromBasis(u, v, w);
      const rebuilt = new THREE.Matrix4().makeRotationFromEuler(
        new THREE.Euler(rx, ry, rz, 'XYZ'));
      const direct = new THREE.Matrix4().makeBasis(
        new THREE.Vector3(...u), new THREE.Vector3(...v), new THREE.Vector3(...w));
      rebuilt.elements.forEach((el, i) => expect(el).toBeCloseTo(direct.elements[i], 6));
    }
  });

  it('obbForEdge: center is the midpoint, size is [len, width, width]', () => {
    const s = setWidth(oneEdge(), 0.3);
    const obb = obbForEdge(s, s.edges[0]);
    expect(obb.center).toEqual([1, 0, 0]);
    expect(obb.size[0]).toBeCloseTo(2);
    expect(obb.size[1]).toBe(0.3);
    expect(obb.size[2]).toBe(0.3);
    expect(obb.rotation).toHaveLength(3);
  });
});

describe('beam-graph: serialization', () => {
  it('toStructureDoc <-> seedFromServer round-trip preserves graph + committed + dirty', () => {
    let s = oneEdge();
    s = addNode(s, [0, 5, 0]);
    s = markApplied(s, s.edges[0].id, 100);
    s = moveNode(s, 1, [0, 1, 0]);               // applied edge now dirty
    s = commitAll(s);                            // applied edge → committed; isolated nodes dropped
    s = addNode(s, [7, 7, 7]);
    const doc = toStructureDoc(s);
    expect(doc.committed_beams).toHaveLength(1);
    expect(doc.committed_beams[0].class_id).toBe(CLS);

    const seeded = seedFromServer(initBeamState(), doc);
    expect(toStructureDoc(seeded)).toEqual(doc);
    // nextId continues past every seeded id (no collisions).
    const ids = [...seeded.nodes.map((n) => n.id), ...seeded.edges.map((e) => e.id)];
    expect(seeded.nextId).toBeGreaterThan(Math.max(0, ...ids));
  });

  it('seedFromServer defaults dirty from instance_id when absent', () => {
    const doc = {
      nodes: [{ id: 1, pos: [0, 0, 0] }, { id: 2, pos: [1, 0, 0] }],
      edges: [
        { id: 3, a: 1, b: 2, width: 0.2, class_id: CLS, instance_id: 9 },
        { id: 4, a: 1, b: 2, width: 0.2, class_id: CLS, instance_id: null },
      ],
      committed_beams: [],
    };
    const s = seedFromServer(initBeamState(), doc);
    expect(s.edges[0].dirty).toBe(false);
    expect(s.edges[1].dirty).toBe(true);
  });
});
