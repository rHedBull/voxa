// api.js — thin client for the FastAPI backend.

export function decodeLoadResponse(j) {
  return {
    scene: j.scene,
    numPoints: j.num_points,
    numPointsTotal: j.num_points_total ?? null,
    numSubsampled: j.num_subsampled,
    bbox: { min: j.bbox_min, max: j.bbox_max },
    positions: b64ToFloat32(j.positions),
    colors: b64ToFloat32(j.colors),
    intensity: j.intensity ? b64ToFloat32(j.intensity) : null,
    classIds: j.class_ids ? b64ToInt8(j.class_ids) : null,
    instanceIds: j.instance_ids ? b64ToInt32(j.instance_ids) : null,
    classPalette: j.class_palette || null,
    nClasses: j.n_classes ?? null,
    nInstances: j.n_instances ?? null,
    nLabeledPoints: j.n_labeled_points ?? null,
    recenterOffset: j.recenter_offset || [0, 0, 0],
    meshUrl: j.mesh_url || null,
    meshIsZUp: !!j.mesh_is_z_up,
    sceneIsZUp: !!j.scene_is_z_up,
    fullClassIds: j.full_class_ids ? b64ToInt8(j.full_class_ids) : null,
    fullInstanceIds: j.full_instance_ids ? b64ToInt32(j.full_instance_ids) : null,
    fullPositions: j.full_positions ? b64ToFloat32(j.full_positions) : null,
    fullN: j.full_n ?? null,
    isFromPrelabel: !!j.is_from_prelabel,
    segmentSummary: j.segment_summary || null,
    subsampleIdx: j.subsample_idx ? b64ToInt32(j.subsample_idx) : null,
    segIds: j.seg_ids ? b64ToInt32(j.seg_ids) : null,
    segCenters: j.seg_centers ? b64ToFloat32(j.seg_centers) : null,
    segSizes: j.seg_sizes ? b64ToFloat32(j.seg_sizes) : null,
    sessionId: j.session_id ?? null,
    sessions: j.sessions || [],
    rawSourceAvailable: !!j.raw_source_available,
  };
}

export function decodeCompareResponse(j) {
  return {
    metrics: j.metrics,
    aClassIds: b64ToInt8(j.a_class_ids),
    bClassIds: b64ToInt8(j.b_class_ids),
    palette: j.palette || [],
  };
}

// Shared non-OK handler: surfaces the backend's `detail` (message when it's a
// string, attached as err.detail either way) plus err.status. Every endpoint
// that fails loudly funnels through here so the shapes can't drift.
async function throwApiError(r, label) {
  let detail = null;
  try { detail = (await r.json()).detail; } catch { /* non-JSON body */ }
  const err = new Error(typeof detail === 'string' ? detail : `${label} failed: ${r.status}`);
  err.status = r.status;
  err.detail = detail;
  throw err;
}

export const VoxaAPI = {
  async health() {
    const r = await fetch('/api/health');
    return r.json();
  },
  async config() {
    const r = await fetch('/api/config');
    return r.json();
  },
  async scenes() {
    const r = await fetch('/api/scenes');
    return r.json();
  },
  async load(name, { maxPoints = null, wantFullLabels = false, sessionId = null } = {}) {
    const body = {
      name,
      ...(maxPoints != null ? { max_points: maxPoints } : {}),
      ...(wantFullLabels ? { want_full_labels: true } : {}),
      ...(sessionId != null ? { session_id: sessionId } : {}),
    };
    const r = await fetch('/api/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      if (r.status === 409) {
        let detail = null;
        try { detail = (await r.json()).detail; } catch { /* non-JSON body */ }
        const err = new Error(detail?.message || `load failed: 409`);
        err.status = 409;
        err.detail = detail;
        throw err;
      }
      throw new Error(`load failed: ${r.status} ${await r.text()}`);
    }
    return decodeLoadResponse(await r.json());
  },
  async loadRegion(aabbMin, aabbMax, { maxPoints = null } = {}) {
    const body = {
      aabb_min: aabbMin,
      aabb_max: aabbMax,
      ...(maxPoints != null ? { max_points: maxPoints } : {}),
    };
    const r = await fetch('/api/load-region', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`loadRegion failed: ${r.status} ${await r.text()}`);
    const j = await r.json();
    return {
      numPoints: j.num_points,
      numInRegionTotal: j.num_in_region_total,
      positions: b64ToFloat32(j.positions),
      colors: j.colors ? b64ToFloat32(j.colors) : null,
    };
  },
  async getAnnotation(scene, kind, sessionId = null) {
    // Tier-prefixed ids contain `/` which Starlette decodes during routing,
    // so the route puts `kind` first and matches `scene` greedily as a path.
    // sessionId scopes the doc to one labeling session (annotated tier) —
    // without it, every session of a scan would share one instance list.
    const q = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
    const r = await fetch(`/api/annotations/${kind}/${scene}${q}`);
    // Never fall back to an empty doc: the caller autosaves whatever this
    // returns, so masking a failure here would overwrite the session's
    // instance doc (OBB volumes, confirmed flags, seqs) with emptiness.
    // A missing doc is not an error — the backend returns an empty doc.
    if (!r.ok) await throwApiError(r, 'annotation fetch');
    return r.json();
  },
  async putAnnotation(scene, kind, doc, sessionId = null) {
    const q = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
    const body = { scene, kind, instances: doc.instances || [], meta: doc.meta || {} };
    const r = await fetch(`/api/annotations/${kind}/${scene}${q}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`save failed: ${r.status}`);
    return r.json();
  },
  async comparePoints(scene, a, b) {
    const r = await fetch(`/api/compare-points/${scene}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ a, b }),
    });
    if (!r.ok) await throwApiError(r, 'compare');
    return decodeCompareResponse(await r.json());
  },
  async autoFit(bboxMin, bboxMax, cls, color, label) {
    const r = await fetch('/api/auto-fit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bbox_min: bboxMin, bbox_max: bboxMax, cls, color, label }),
    });
    return r.json();
  },
  async segApply(op, { indices = null, payload = {} } = {}) {
    const body = {
      op,
      payload,
      ...(indices != null ? { indices: _int32ToB64(indices) } : {}),
    };
    const r = await fetch('/api/segment/apply', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`segApply failed: ${r.status} ${await r.text()}`);
    const j = await r.json();
    return _decodeApplyResponse(j);
  },
  async segUndo() {
    const r = await fetch('/api/segment/undo', { method: 'POST' });
    if (r.status === 204) return null;
    if (!r.ok) throw new Error(`segUndo failed: ${r.status} ${await r.text()}`);
    return _decodeApplyResponse(await r.json());
  },
  async segRedo() {
    const r = await fetch('/api/segment/redo', { method: 'POST' });
    if (r.status === 204) return null;
    if (!r.ok) throw new Error(`segRedo failed: ${r.status} ${await r.text()}`);
    return _decodeApplyResponse(await r.json());
  },
  async segSave() {
    const r = await fetch('/api/segment/save', { method: 'PUT' });
    if (!r.ok) throw new Error(`segSave failed: ${r.status} ${await r.text()}`);
    return r.json();
  },
  async segState() {
    const r = await fetch('/api/segment/state');
    if (!r.ok) throw new Error(`segState failed: ${r.status} ${await r.text()}`);
    const j = await r.json();
    if (!j.has_state) return null;
    return {
      nAssigned: j.n_assigned,
      nSegments: j.n_segments,
      fullClassIds: b64ToInt8(j.full_class_ids),
      fullInstanceIds: b64ToInt32(j.full_instance_ids),
      segIds: j.seg_ids ? b64ToInt32(j.seg_ids) : null,
      segCenters: j.seg_centers ? b64ToFloat32(j.seg_centers) : null,
      segSizes: j.seg_sizes ? b64ToFloat32(j.seg_sizes) : null,
      hullVertices: j.hull_vertices ? b64ToFloat32(j.hull_vertices) : null,
      hullFaces: j.hull_faces ? b64ToInt32(j.hull_faces) : null,
      hullFaceSeg: j.hull_face_seg ? b64ToInt32(j.hull_face_seg) : null,
    };
  },
  async listSessions(scene) {
    const r = await fetch(`/api/scenes/${scene}/sessions`);
    if (!r.ok) throw new Error(`listSessions failed: ${r.status} ${await r.text()}`);
    const j = await r.json();
    return j.sessions;
  },
  async createSession(scene, { name, presegId = null } = {}) {
    const body = { name, ...(presegId ? { preseg_id: presegId } : {}) };
    const r = await fetch(`/api/scenes/${scene}/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`createSession failed: ${r.status} ${await r.text()}`);
    return r.json();
  },
  async renameSession(scene, sid, name) {
    const r = await fetch(`/api/scenes/${scene}/sessions/${sid}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name }),
    });
    if (!r.ok) throw new Error(`renameSession failed: ${r.status} ${await r.text()}`);
    return r.json();
  },
  async deleteSession(scene, sid) {
    const r = await fetch(`/api/scenes/${scene}/sessions/${sid}`, { method: 'DELETE' });
    if (!r.ok) throw new Error(`deleteSession failed: ${r.status} ${await r.text()}`);
    return r.json();
  },
  async listPresegs(scene) {
    const r = await fetch(`/api/scenes/${scene}/presegs`);
    if (!r.ok) throw new Error(`listPresegs failed: ${r.status} ${await r.text()}`);
    const j = await r.json();
    return j.presegs;
  },
  async applyShape({ shape, targetClass, targetInst = -1, mergedFrom = [], protectInstances = [] }) {
    const body = {
      shape,
      target_class: targetClass,
      target_inst: targetInst,
      merged_from: mergedFrom,
      protect_instances: protectInstances,
    };
    const r = await fetch('/api/segment/apply-shape', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`applyShape failed: ${r.status} ${await r.text()}`);
    const j = await r.json();
    return { ..._decodeApplyResponse(j), instanceId: j.instance_id ?? null };
  },
  async centerlineApply({ paths, targetClass, targetInst = -1, mergedFrom = [], protectInstances = [] }) {
    return this.applyShape({ shape: { type: 'tube', paths }, targetClass, targetInst, mergedFrom, protectInstances });
  },
  async getCenterlines() {
    const r = await fetch('/api/segment/centerlines');
    if (!r.ok) throw new Error(`getCenterlines failed: ${r.status} ${await r.text()}`);
    return r.json();
  },
  // sessionId pins the read for the same reason putStructure pins the write:
  // a remount racing a session switch must fail loudly, never seed from the
  // wrong session.
  async getStructure(sessionId) {
    const q = sessionId ? `?session_id=${encodeURIComponent(sessionId)}` : '';
    const r = await fetch(`/api/segment/structure${q}`);
    if (!r.ok) throw new Error(`getStructure failed: ${r.status} ${await r.text()}`);
    return r.json();
  },
  // sessionId pins the write: the backend 409s if the active session changed
  // between the edit and this (debounced) write — never write cross-session.
  async putStructure(doc, sessionId) {
    const r = await fetch('/api/segment/structure', {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ ...doc, session_id: sessionId }),
    });
    if (!r.ok) throw new Error(`putStructure failed: ${r.status} ${await r.text()}`);
    return r.json();
  },
  // Export wizard Review step: real p50/p90 sample spacing for the loaded scan.
  async getAccuracy(scene, sessionId) {
    const q = `?scene=${encodeURIComponent(scene)}&session_id=${encodeURIComponent(sessionId)}`;
    const r = await fetch(`/api/labels/accuracy${q}`);
    if (!r.ok) throw new Error(`getAccuracy failed: ${r.status} ${await r.text()}`);
    return r.json();
  },
  // Export the active session's labels → zip blob. Surfaces backend 422/409
  // detail on the thrown error (err.status / err.detail) so the wizard can
  // render it inline instead of alerting.
  async exportLabels(cfg) {
    const r = await fetch('/api/labels/export', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(cfg),
    });
    if (!r.ok) await throwApiError(r, 'export');
    return r.blob();
  },
};

function _decodeApplyResponse(j) {
  return {
    op: j.op,
    nAffected: j.n_affected,
    nProtected: j.n_protected ?? 0,
    dirty: j.dirty,
    newInstanceId: j.new_instance_id ?? null,
    indices: j.indices ? b64ToInt32(j.indices) : null,
    afterClass: j.after_class ? b64ToInt8(j.after_class) : null,
    afterInstance: j.after_instance ? b64ToInt32(j.after_instance) : null,
    direction: j.direction ?? null,
  };
}

function _int32ToB64(arr) {
  const u8 = new Uint8Array(arr.buffer, arr.byteOffset, arr.byteLength);
  let s = '';
  for (let i = 0; i < u8.length; i++) s += String.fromCharCode(u8[i]);
  return btoa(s);
}

function b64ToBuf(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return buf;
}

export function b64ToFloat32(b64) { return new Float32Array(b64ToBuf(b64)); }
export function b64ToInt8(b64)    { return new Int8Array(b64ToBuf(b64)); }
export function b64ToInt32(b64)   { return new Int32Array(b64ToBuf(b64)); }

export function newId(prefix = 'inst') {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

export async function getSegmentState() {
  const r = await fetch('/api/segment/state');
  if (!r.ok) throw new Error(`segment/state ${r.status}`);
  return r.json();
}
