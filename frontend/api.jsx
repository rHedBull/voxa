// api.jsx — thin client for the FastAPI backend.

const API = {
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
  async load(name, maxPoints = 300000) {
    const r = await fetch('/api/load', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, max_points: maxPoints }),
    });
    if (!r.ok) throw new Error(`load failed: ${r.status} ${await r.text()}`);
    const j = await r.json();
    return {
      scene: j.scene,
      numPoints: j.num_points,
      numSubsampled: j.num_subsampled,
      bbox: { min: j.bbox_min, max: j.bbox_max },
      positions: b64ToFloat32(j.positions),
      colors: b64ToFloat32(j.colors),
    };
  },
  async getAnnotation(scene, kind) {
    const r = await fetch(`/api/annotations/${encodeURIComponent(scene)}/${kind}`);
    if (!r.ok) return { scene, kind, instances: [], meta: {} };
    return r.json();
  },
  async putAnnotation(scene, kind, doc) {
    const body = { scene, kind, instances: doc.instances || [], meta: doc.meta || {} };
    const r = await fetch(`/api/annotations/${encodeURIComponent(scene)}/${kind}`, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`save failed: ${r.status}`);
    return r.json();
  },
  async compare(scene, iouThreshold = 0.3) {
    const r = await fetch(`/api/compare/${encodeURIComponent(scene)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ scene, iou_threshold: iouThreshold }),
    });
    return r.json();
  },
  async autoFit(bboxMin, bboxMax, cls, color, label) {
    const r = await fetch('/api/auto-fit', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ bbox_min: bboxMin, bbox_max: bboxMax, cls, color, label }),
    });
    return r.json();
  },
};

function b64ToFloat32(b64) {
  const bin = atob(b64);
  const buf = new ArrayBuffer(bin.length);
  const view = new Uint8Array(buf);
  for (let i = 0; i < bin.length; i++) view[i] = bin.charCodeAt(i);
  return new Float32Array(buf);
}

function newId(prefix = 'inst') {
  return `${prefix}-${Math.random().toString(36).slice(2, 10)}`;
}

window.VoxaAPI = API;
window.VoxaUtil = { newId };
