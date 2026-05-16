// mesh-loading.test.js — verify GLTFLoader actually surfaces a Mesh from
// the files the registry hands us. This is the frontend half of the
// `mesh.optimized.glb` bug: backend/tests/test_scene_registry.py guards
// the registry against picking the broken file; this test guards the
// FALLBACK assumption — that a "valid" GLB does deliver a Mesh once
// parsed, while the broken one yields a scene with no Mesh descendants
// (which is exactly what the companion window saw: empty Group → black).
//
// We do not boot the React tree or a real browser here. The bug shows up
// at GLTFLoader.parse() — same parser the viewer uses — so loading the
// fixture bytes through it directly catches the failure mode without
// needing jsdom / @testing-library / a headless Chrome.

import { describe, it, expect, beforeAll } from 'vitest';
import { readFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';

const __dirname = dirname(fileURLToPath(import.meta.url));
const fixtures = join(__dirname, '__fixtures__');

// GLTFLoader.parse needs an ArrayBuffer; node's fs returns a Buffer. Trim
// to the exact byte range so we don't pass the underlying allocator's slack.
function bufferToArrayBuffer(buf) {
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength);
}

function parseGlb(arrayBuffer) {
  return new Promise((resolve, reject) => {
    new GLTFLoader().parse(arrayBuffer, '', resolve, reject);
  });
}

function countMeshes(root) {
  let n = 0;
  root.traverse((node) => { if (node.isMesh) n += 1; });
  return n;
}

describe('GLTFLoader on registry-eligible GLBs', () => {
  let validBuf, brokenBuf;

  beforeAll(async () => {
    validBuf = bufferToArrayBuffer(await readFile(join(fixtures, 'valid_mesh.glb')));
    brokenBuf = bufferToArrayBuffer(await readFile(join(fixtures, 'broken_mesh.glb')));
  });

  it('valid GLB delivers at least one Mesh in the scene root', async () => {
    const gltf = await parseGlb(validBuf);
    expect(gltf.scene).toBeTruthy();
    // The companion / main viewer wires `gltf.scene` directly into the
    // meshGroup; if this is empty, the viewport renders nothing.
    expect(countMeshes(gltf.scene)).toBeGreaterThan(0);
  });

  it('broken GLB parses without throwing but yields an empty scene', async () => {
    // This is the *symptom* of the mesh.optimized.glb bug. Parse succeeds
    // (the file is well-formed glTF), but the scene root has no children
    // that reference the mesh node, so there's nothing to render. The
    // registry must avoid handing this file to the viewer.
    const gltf = await parseGlb(brokenBuf);
    expect(gltf.scene).toBeTruthy();
    expect(countMeshes(gltf.scene)).toBe(0);
  });
});
