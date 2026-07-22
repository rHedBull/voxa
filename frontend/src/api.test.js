import { describe, expect, it, vi, afterEach } from 'vitest';
import { b64ToFloat32, b64ToInt8, b64ToInt32, newId, decodeLoadResponse, decodeCompareResponse, VoxaAPI } from './api.js';

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

  it('decodes subsampleIdx as Int32Array when present', () => {
    const j = { ...makeFakeLoadResponse(), subsample_idx: encodeInt32([0, 2, 4]) };
    const out = decodeLoadResponse(j);
    expect(out.subsampleIdx).toBeInstanceOf(Int32Array);
    expect(Array.from(out.subsampleIdx)).toEqual([0, 2, 4]);
  });

  it('returns null subsampleIdx when absent', () => {
    const j = makeFakeLoadResponse();
    const out = decodeLoadResponse(j);
    expect(out.subsampleIdx).toBeNull();
  });
});

describe('decodeLoadResponse — session/preseg fields (scan-schema v2)', () => {
  it('maps session_id and sessions when present', () => {
    const j = {
      ...makeFakeLoadResponse(),
      session_id: 'abc-123',
      sessions: [
        { session_id: 'abc-123', name: 'main', preseg_id: null, created_at: 't', saved_at: null, dirty: false, has_output: true, corrupt: false },
      ],
    };
    const out = decodeLoadResponse(j);
    expect(out.sessionId).toBe('abc-123');
    expect(out.sessions).toHaveLength(1);
    expect(out.sessions[0].session_id).toBe('abc-123');
    expect(out.sessions[0].name).toBe('main');
  });

  it('defaults sessionId to null when absent', () => {
    const out = decodeLoadResponse(makeFakeLoadResponse());
    expect(out.sessionId).toBeNull();
  });

  it('defaults sessions to [] when absent', () => {
    const out = decodeLoadResponse(makeFakeLoadResponse());
    expect(out.sessions).toEqual([]);
  });

  it('passes through a multi-session list unchanged', () => {
    const sessions = [
      { session_id: 's1', name: 'alpha', preseg_id: 'p1', created_at: 't1', saved_at: 't2', dirty: true, has_output: false, corrupt: false },
      { session_id: 's2', name: 'beta',  preseg_id: null, created_at: 't3', saved_at: null,  dirty: false, has_output: true,  corrupt: false },
    ];
    const out = decodeLoadResponse({ ...makeFakeLoadResponse(), session_id: 's1', sessions });
    expect(out.sessions).toHaveLength(2);
    expect(out.sessions[1].name).toBe('beta');
  });
});

describe('VoxaAPI.load — 409 detail attachment', () => {
  afterEach(() => { vi.unstubAllGlobals(); });

  it('attaches detail and status=409 on a pin-mismatch 409', async () => {
    const detail = { error: 'session_pin_mismatch', diverged: 'preseg', session_id: 'sid1', message: 'preseg changed' };
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: false,
      status: 409,
      json: async () => ({ detail }),
    })));

    let thrown = null;
    try { await VoxaAPI.load('annotated/demo', {}); } catch (e) { thrown = e; }

    expect(thrown).not.toBeNull();
    expect(thrown.status).toBe(409);
    expect(thrown.detail).toEqual(detail);
    expect(thrown.message).toBe('preseg changed');
  });

  it('falls back to generic message when 409 body has no message', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: false,
      status: 409,
      json: async () => ({ detail: { error: 'session_unreadable' } }),
    })));

    let thrown = null;
    try { await VoxaAPI.load('annotated/demo', {}); } catch (e) { thrown = e; }

    expect(thrown.status).toBe(409);
    expect(thrown.message).toBe('load failed: 409');
  });
});

describe('VoxaAPI.getAnnotation — fail loudly', () => {
  afterEach(() => { vi.unstubAllGlobals(); });

  it('throws with status + detail on a non-OK response (never an empty doc)', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: false,
      status: 500,
      json: async () => ({ detail: 'instances_gt.json is corrupt' }),
    })));

    let thrown = null;
    try { await VoxaAPI.getAnnotation('annotated/demo', 'gt', 's1'); } catch (e) { thrown = e; }

    expect(thrown).not.toBeNull();
    expect(thrown.status).toBe(500);
    expect(thrown.message).toBe('instances_gt.json is corrupt');
  });

  it('throws a generic message when the error body is not JSON', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: false,
      status: 502,
      json: async () => { throw new Error('not json'); },
    })));

    let thrown = null;
    try { await VoxaAPI.getAnnotation('annotated/demo', 'gt', 's1'); } catch (e) { thrown = e; }

    expect(thrown.status).toBe(502);
    expect(thrown.message).toBe('annotation fetch failed: 502');
  });

  it('returns the parsed doc on OK', async () => {
    const doc = { scene: 'annotated/demo', kind: 'gt', instances: [{ id: 'i1' }], meta: {} };
    vi.stubGlobal('fetch', vi.fn(async () => ({ ok: true, json: async () => doc })));
    expect(await VoxaAPI.getAnnotation('annotated/demo', 'gt', 's1')).toEqual(doc);
  });
});

describe('decodeCompareResponse', () => {
  it('decodes metrics, arrays and palette', () => {
    const enc = (arr) => Buffer.from(Int8Array.from(arr).buffer).toString('base64');
    const out = decodeCompareResponse({
      metrics: { agreement: 0.5, per_class: [], confusion: [] },
      a_class_ids: enc([-1, 0, 1]),
      b_class_ids: enc([0, 0, 1]),
      palette: [{ id: 0, label: 'Pipe', color: '#5b8def' }],
    });
    expect(out.metrics.agreement).toBe(0.5);
    expect(Array.from(out.aClassIds)).toEqual([-1, 0, 1]);
    expect(Array.from(out.bClassIds)).toEqual([0, 0, 1]);
    expect(out.palette[0].label).toBe('Pipe');
  });
});

describe('centerline API', () => {
  afterEach(() => { vi.unstubAllGlobals(); });

  it('centerlineApply posts a tube shape to apply-shape and decodes the delta', async () => {
    let capturedUrl, capturedOpts;
    vi.stubGlobal('fetch', vi.fn(async (url, opts) => {
      capturedUrl = url;
      capturedOpts = opts;
      return { ok: true, json: async () => ({ op: 'reassign', n_affected: 0, dirty: true }) };
    }));
    await VoxaAPI.centerlineApply({
      paths: [{ points: [[0, 0, 0], [1, 0, 0]], radius: 0.15, smooth: false }],
      targetClass: 'pipe', targetInst: -1, mergedFrom: [4],
    });
    // centerlineApply now delegates to the generic apply-shape endpoint.
    expect(capturedUrl).toBe('/api/segment/apply-shape');
    const body = JSON.parse(capturedOpts.body);
    expect(body.target_class).toBe('pipe');
    expect(body.target_inst).toBe(-1);
    expect(body.merged_from).toEqual([4]);
    expect(body.shape.type).toBe('tube');
    expect(body.shape.paths[0].radius).toBe(0.15);
  });

  it('centerlineApply surfaces instance_id on the decoded response', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      json: async () => ({ op: 'reassign', n_affected: 3, dirty: true,
                           new_instance_id: 9, instance_id: 9 }),
    })));
    const r = await VoxaAPI.centerlineApply({ paths: [], targetClass: 0 });
    expect(r.instanceId).toBe(9);
    expect(r.nAffected).toBe(3);
  });

  it('applyShape forwards protectInstances as body.protect_instances', async () => {
    let capturedOpts;
    vi.stubGlobal('fetch', vi.fn(async (url, opts) => {
      capturedOpts = opts;
      return { ok: true, json: async () => ({ op: 'reassign', n_affected: 0, dirty: true }) };
    }));
    await VoxaAPI.applyShape({
      shape: { type: 'obb', center: [0, 0, 0], size: [1, 1, 1], rotation: [0, 0, 0] },
      targetClass: 'pipe', protectInstances: [3, 7],
    });
    expect(JSON.parse(capturedOpts.body).protect_instances).toEqual([3, 7]);
  });

  it('applyShape decodes n_protected onto the response', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true, json: async () => ({ op: 'reassign', n_affected: 0, n_protected: 5, dirty: true }),
    })));
    const r = await VoxaAPI.applyShape({
      shape: { type: 'obb', center: [0, 0, 0], size: [1, 1, 1], rotation: [0, 0, 0] },
      targetClass: 0,
    });
    expect(r.nProtected).toBe(5);
  });

  it('cutShape posts shape+sources and decodes materialized/instance response', async () => {
    let capturedOpts;
    vi.stubGlobal('fetch', vi.fn(async (url, opts) => {
      capturedOpts = opts;
      return {
        ok: true,
        json: async () => ({
          materialized: [{ sam_seg_id: 7, source: 'preseg', n_points: 12 }],
          instance: null,
          n_protected: 0,
        }),
      };
    }));
    const result = await VoxaAPI.cutShape({
      shape: { type: 'obb', center: [0, 0, 0], size: [1, 1, 1], rotation: [0, 0, 0] },
      sources: [{ kind: 'preseg', segId: 3 }],
    });
    expect(result.materialized[0]).toEqual({ samSegId: 7, source: 'preseg', nPoints: 12, indices: null });
    expect(result.instance).toBeNull();
    expect(JSON.parse(capturedOpts.body).sources).toEqual([{ kind: 'preseg', seg_id: 3 }]);
  });

  it('cutShape decodes a non-null instance entry', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      json: async () => ({ materialized: [], instance: { instance_id: 9, n_points: 4 }, n_protected: 1 }),
    })));
    const result = await VoxaAPI.cutShape({
      shape: { type: 'obb', center: [0, 0, 0], size: [1, 1, 1], rotation: [0, 0, 0] },
      sources: [{ kind: 'instance', segId: 5 }],
    });
    expect(result.instance).toEqual({ instId: 9, nPoints: 4, indices: null });
    expect(result.nProtected).toBe(1);
  });

  it('cutShape decodes scan_indices_b64 into materialized[].indices and instance.indices', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      json: async () => ({
        materialized: [{
          sam_seg_id: 7, source: 'preseg', n_points: 3,
          scan_indices_b64: encodeInt32([1, 2, 3]),
        }],
        instance: {
          instance_id: 9, n_points: 3,
          scan_indices_b64: encodeInt32([1, 2, 3]),
        },
        n_protected: 0,
      }),
    })));
    const result = await VoxaAPI.cutShape({
      shape: { type: 'obb', center: [0, 0, 0], size: [1, 1, 1], rotation: [0, 0, 0] },
      sources: [{ kind: 'preseg', segId: 3 }],
    });
    expect(result.materialized[0].indices).toBeInstanceOf(Int32Array);
    expect(result.materialized[0].indices).toEqual(Int32Array.from([1, 2, 3]));
    expect(result.instance.indices).toBeInstanceOf(Int32Array);
    expect(result.instance.indices).toEqual(Int32Array.from([1, 2, 3]));
  });

  it('fitBox posts sources and returns the OBB', async () => {
    const obb = { center: [0, 0, 0], size: [1, 2, 3], rotation: [0, 0.5, 0] };
    let capturedUrl, capturedOpts;
    vi.stubGlobal('fetch', vi.fn(async (url, opts) => {
      capturedUrl = url;
      capturedOpts = opts;
      return { ok: true, json: async () => obb };
    }));
    const result = await VoxaAPI.fitBox([{ kind: 'preseg', segId: 7 }]);
    expect(capturedUrl).toBe('/api/segment/fit-box');
    expect(JSON.parse(capturedOpts.body)).toEqual({ sources: [{ kind: 'preseg', seg_id: 7 }] });
    expect(result).toEqual(obb);
  });

  it('samProject decodes each sam segment so segment-state can be patched', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      json: async () => ({
        segments: [{
          mask_id: 0, sam_seg_id: 7, n_affected: 2, n_protected: 0,
          scan_indices_b64: encodeInt32([10, 20]),
        }],
      }),
    })));
    const r = await VoxaAPI.samProject({
      captureId: 'c1', maskIds: [0],
    });
    expect(r.segments).toHaveLength(1);
    const seg = r.segments[0];
    expect(seg.maskId).toBe(0);
    expect(seg.samSegId).toBe(7);
    expect(seg.nAffected).toBe(2);
    expect(seg.nProtected).toBe(0);
    expect(seg.indices).toBeInstanceOf(Int32Array);
    expect(Array.from(seg.indices)).toEqual([10, 20]);
  });

  it('getCenterlines returns the stored paths', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true, json: async () => ({ paths: [{ instance_id: 7 }] }),
    })));
    const j = await VoxaAPI.getCenterlines();
    expect(j.paths).toHaveLength(1);
  });
});

describe('VoxaAPI regions', () => {
  afterEach(() => vi.unstubAllGlobals());

  it('regionPatch surfaces the backend detail on 422', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify({ detail: 'measured p90 spacing 23.0 mm exceeds the 10 mm eval-grade bar' }),
      { status: 422 },
    )));
    await expect(VoxaAPI.regionPatch(1, { status: 'eval_grade' }))
      .rejects.toMatchObject({ status: 422, detail: expect.stringContaining('p90') });
  });

  it('regionsList unwraps the regions array', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify({ regions: [{ id: 1, name: 'Region 1', status: 'draft' }] }),
      { status: 200 },
    )));
    expect(await VoxaAPI.regionsList()).toEqual([{ id: 1, name: 'Region 1', status: 'draft' }]);
  });

  it('regionCreate/regionDelete/regionStats hit the expected URL, method, and body', async () => {
    const prism = { polygon: [[0, 0], [1, 0], [1, 1]], y0: 0, height: 2 };
    let capturedUrl, capturedOpts;
    const fakeFetch = vi.fn(async (url, opts) => {
      capturedUrl = url;
      capturedOpts = opts;
      return new Response(JSON.stringify({ regions: [] }), { status: 200 });
    });
    vi.stubGlobal('fetch', fakeFetch);

    await VoxaAPI.regionCreate({ prism, name: 'Region 1' });
    expect(capturedUrl).toBe('/api/regions');
    expect(capturedOpts.method).toBe('POST');
    expect(JSON.parse(capturedOpts.body)).toEqual({ prism, name: 'Region 1' });

    await VoxaAPI.regionDelete(3);
    expect(capturedUrl).toBe('/api/regions/3');
    expect(capturedOpts.method).toBe('DELETE');

    await VoxaAPI.regionStats();
    expect(capturedUrl).toBe('/api/regions/stats');
    expect(capturedOpts).toBeUndefined();
  });
});

describe('VoxaAPI.segApply wire shape', () => {
  afterEach(() => { vi.unstubAllGlobals(); });

  it('nests payload under body.payload, not at top level', async () => {
    let captured = null;
    const fakeFetch = vi.fn(async (url, opts) => {
      captured = { url, body: JSON.parse(opts.body) };
      return { ok: true, json: async () => ({ op: 'set_class', n_affected: 0, dirty: true }) };
    });
    vi.stubGlobal('fetch', fakeFetch);

    await VoxaAPI.segApply('set_class', { payload: { class_id: 2 } });

    expect(captured.url).toBe('/api/segment/apply');
    expect(captured.body.op).toBe('set_class');
    expect(captured.body.payload).toEqual({ class_id: 2 });
    expect(captured.body.class_id).toBeUndefined();
  });
});

describe('point categories on the wire (phase 2)', () => {
  afterEach(() => { vi.unstubAllGlobals(); });

  it('applyShape sends target_category instead of target_class', async () => {
    let capturedOpts;
    vi.stubGlobal('fetch', vi.fn(async (url, opts) => {
      capturedOpts = opts;
      return { ok: true, json: async () => ({ op: 'set_category', n_affected: 2, dirty: true }) };
    }));
    await VoxaAPI.applyShape({
      shape: { type: 'obb', center: [0, 0, 0], size: [1, 1, 1], rotation: [0, 0, 0] },
      targetCategory: 'artifact',
    });
    const body = JSON.parse(capturedOpts.body);
    expect(body.target_category).toBe('artifact');
    expect('target_class' in body).toBe(false);
  });

  it('segApply decodes after_category into the response', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => ({
      ok: true,
      json: async () => ({
        op: 'set_category', n_affected: 2, dirty: true,
        indices: encodeInt32([4, 5]),
        after_class: encodeInt8([-1, -1]),
        after_instance: encodeInt32([-1, -1]),
        after_category: encodeInt8([1, 1]),
      }),
    })));
    const r = await VoxaAPI.segApply('set_category', {
      indices: new Int32Array([4, 5]), payload: { category: 'artifact' },
    });
    expect(Array.from(r.afterCategory)).toEqual([1, 1]);
  });

  it('decodeLoadResponse surfaces full_categories', () => {
    const j = {
      positions: encodeFloat32([0, 0, 0]), colors: encodeFloat32([1, 1, 1]),
      full_categories: encodeInt8([0, 3]),
    };
    expect(Array.from(decodeLoadResponse(j).fullCategories)).toEqual([0, 3]);
  });
});
