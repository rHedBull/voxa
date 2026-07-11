import { describe, it, expect } from 'vitest';
import * as THREE from 'three';
import { pointsInsideOBB } from './mode-edit.jsx';

// pointsInsideOBB must use the exact rotation the viewer renders the gizmo
// box with — Three.js Euler 'XYZ'. A multi-axis rotation catches a reversed
// (Rz·Ry·Rx) composition that single-axis rotations can't distinguish.
describe('pointsInsideOBB', () => {
  it('matches THREE Euler XYZ containment for a multi-axis rotation', () => {
    const box = { center: [0.2, -0.1, 0.3], size: [2.0, 1.0, 0.5], rotation: [0.4, -0.7, 0.3] };

    let seed = 42;
    const rand = () => {
      seed = (seed * 1103515245 + 12345) % 2147483648;
      return seed / 2147483648;
    };
    const n = 2000;
    const positions = new Float32Array(n * 3);
    for (let i = 0; i < n * 3; i++) positions[i] = rand() * 4 - 2;

    const inv = new THREE.Matrix4()
      .makeRotationFromEuler(new THREE.Euler(...box.rotation, 'XYZ'))
      .invert();
    const expected = [];
    const v = new THREE.Vector3();
    for (let i = 0; i < n; i++) {
      v.set(positions[3 * i] - box.center[0],
            positions[3 * i + 1] - box.center[1],
            positions[3 * i + 2] - box.center[2]).applyMatrix4(inv);
      if (Math.abs(v.x) <= box.size[0] / 2 &&
          Math.abs(v.y) <= box.size[1] / 2 &&
          Math.abs(v.z) <= box.size[2] / 2) expected.push(i);
    }
    expect(expected.length).toBeGreaterThan(0); // guard: non-trivial fixture

    const got = Array.from(pointsInsideOBB(positions, null, box));
    expect(got).toEqual(expected);
  });
});
