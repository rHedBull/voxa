import { describe, expect, it } from 'vitest';
import { b64ToFloat32, b64ToInt8, b64ToInt32, newId, decodeLoadResponse } from './api.js';

// Encode helpers matching the backend's little-endian binary layout.
function encodeFloat32(floats) {
  const u8 = new Uint8Array(new Float32Array(floats).buffer);
  let s = '';
  for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
  return btoa(s);
}
function encodeInt8(vals) {
  const u8 = new Uint8Array(new Int8Array(vals).buffer);
  let s = '';
  for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
  return btoa(s);
}
function encodeInt32(vals) {
  const u8 = new Uint8Array(new Int32Array(vals).buffer);
  let s = '';
  for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
  return btoa(s);
}

function makeFakeLoadResponse({ withFull = false } = {}) {
  const base = {
    scene: 'test_scene',
    num_points: 3,
    num_subsampled: 3,
    bbox_min: [0, 0, 0],
    bbox_max: [1, 1, 1],
    positions: encodeFloat32([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    colors: encodeFloat32([1, 0, 0, 0, 1, 0, 0, 0, 1]),
    recenter_offset: [0, 0, 0],
  };
  if (!withFull) return base;
  return {
    ...base,
    full_class_ids: encodeInt8([-1, 0, 1]),
    full_instance_ids: encodeInt32([-1, 0, 1]),
    full_positions: encodeFloat32([0, 0, 0, 1, 0, 0, 0, 1, 0]),
    full_n: 3,
    is_from_prelabel: true,
    segment_summary: { n_instances: 2 },
  };
}

describe('newId', () => {
  it('uses the supplied prefix', () => {
    expect(newId('cuboid')).toMatch(/^cuboid-[a-z0-9]+$/);
  });

  it('defaults to "inst"', () => {
    expect(newId()).toMatch(/^inst-[a-z0-9]+$/);
  });

  it('returns distinct ids on successive calls', () => {
    const a = newId();
    const b = newId();
    expect(a).not.toBe(b);
  });
});

describe('b64ToFloat32', () => {
  const encode = (floats) => {
    const u8 = new Uint8Array(new Float32Array(floats).buffer);
    let s = '';
    for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
    return btoa(s);
  };

  it('roundtrips a known Float32 array', () => {
    const src = [0.0, -1.5, 3.25, 1e-3, 1234.5];
    const decoded = b64ToFloat32(encode(src));
    expect(decoded).toBeInstanceOf(Float32Array);
    expect(decoded.length).toBe(src.length);
    for (let i = 0; i < src.length; i++) {
      expect(decoded[i]).toBeCloseTo(src[i], 6);
    }
  });

  it('returns an empty Float32Array for an empty payload', () => {
    const decoded = b64ToFloat32('');
    expect(decoded).toBeInstanceOf(Float32Array);
    expect(decoded.length).toBe(0);
  });
});

describe('b64ToInt8 / b64ToInt32', () => {
  const encodeBytes = (typedArray) => {
    const u8 = new Uint8Array(typedArray.buffer);
    let s = '';
    for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
    return btoa(s);
  };

  it('roundtrips Int8 (including -1 sentinel for unlabeled)', () => {
    const src = new Int8Array([-1, 0, 0, 1, 2, -1, 4, 127, -128]);
    const decoded = b64ToInt8(encodeBytes(src));
    expect(decoded).toBeInstanceOf(Int8Array);
    expect(Array.from(decoded)).toEqual(Array.from(src));
  });

  it('roundtrips Int32 instance ids', () => {
    const src = new Int32Array([-1, 0, 1, 12345, 2147483647, -2147483648]);
    const decoded = b64ToInt32(encodeBytes(src));
    expect(decoded).toBeInstanceOf(Int32Array);
    expect(Array.from(decoded)).toEqual(Array.from(src));
  });
});

describe('decodeLoadResponse', () => {
  it('decodes full_* fields when present', () => {
    const j = makeFakeLoadResponse({ withFull: true });
    const out = decodeLoadResponse(j);
    expect(out.fullClassIds).toBeInstanceOf(Int8Array);
    expect(out.fullClassIds.length).toBe(3);
    expect(out.fullInstanceIds).toBeInstanceOf(Int32Array);
    expect(out.fullInstanceIds.length).toBe(3);
    expect(out.fullPositions).toBeInstanceOf(Float32Array);
    expect(out.fullPositions.length).toBe(9);
    expect(out.fullN).toBe(3);
    expect(out.isFromPrelabel).toBe(true);
    expect(out.segmentSummary).toEqual({ n_instances: 2 });
  });

  it('returns null fullClassIds / fullInstanceIds / fullPositions when absent', () => {
    const j = makeFakeLoadResponse({ withFull: false });
    const out = decodeLoadResponse(j);
    expect(out.fullClassIds).toBeNull();
    expect(out.fullInstanceIds).toBeNull();
    expect(out.fullPositions).toBeNull();
    expect(out.fullN).toBeNull();
    expect(out.isFromPrelabel).toBe(false);
    expect(out.segmentSummary).toBeNull();
  });

  it('always decodes base positions and colors', () => {
    const j = makeFakeLoadResponse({ withFull: false });
    const out = decodeLoadResponse(j);
    expect(out.positions).toBeInstanceOf(Float32Array);
    expect(out.positions.length).toBe(9);
    expect(out.colors).toBeInstanceOf(Float32Array);
    expect(out.scene).toBe('test_scene');
    expect(out.numPoints).toBe(3);
  });
});
