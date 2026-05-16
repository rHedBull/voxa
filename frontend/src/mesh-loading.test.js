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
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';

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

describe('EXT_meshopt_compression handling', () => {
  // The voxa build pipeline writes EXT_meshopt_compression on every
  // production GLB (mesh.r05.glb / mesh.r05.small.glb / mesh.optimized
  // .glb). GLTFLoader requires MeshoptDecoder to be registered up-front
  // — when it isn't, the loader throws *inside the Promise* and the
  // viewer's onError handler logs it but the meshGroup stays empty, so
  // the symptom is "mesh window is black with floor + axes." Fixture
  // built with `gltfpack -i valid_mesh.glb -o meshopt_compressed.glb -cc`
  // so it matches the bit-for-bit shape the pipeline emits.
  let meshoptBuf;

  beforeAll(async () => {
    const __dirname = dirname(fileURLToPath(import.meta.url));
    meshoptBuf = bufferToArrayBuffer(
      await readFile(join(__dirname, '__fixtures__', 'meshopt_compressed.glb')),
    );
  });

  it('vanilla GLTFLoader (no decoder) cannot load a meshopt-compressed GLB', async () => {
    // Regression guard: prove the failure mode exists. If a future three
    // release ships a built-in decoder this test will start failing —
    // that would be a good time to revisit viewer.jsx and drop the
    // explicit setMeshoptDecoder wiring.
    const loader = new GLTFLoader();
    await expect(parseGlbWith(loader, meshoptBuf)).rejects.toThrow(/setMeshoptDecoder|meshopt/i);
  });

  it('GLTFLoader with MeshoptDecoder loads the compressed GLB', async () => {
    await MeshoptDecoder.ready;
    const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
    const gltf = await parseGlbWith(loader, meshoptBuf);
    expect(countMeshes(gltf.scene)).toBeGreaterThan(0);
  });
});

function parseGlbWith(loader, arrayBuffer) {
  return new Promise((resolve, reject) => {
    loader.parse(arrayBuffer, '', resolve, reject);
  });
}
