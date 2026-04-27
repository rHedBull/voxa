import { describe, expect, it } from 'vitest';
import { b64ToFloat32, b64ToInt8, b64ToInt32, newId } from './api.js';

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
  // Helper: encode a Float32Array → base64 the same way the backend does.
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
