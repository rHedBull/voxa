// mesh-sync.js — cross-tab channel for the mesh-companion window.
//
// The main Voxa window broadcasts `state` (scene + cuboids + mesh URL) and
// `camera` messages; the companion window broadcasts `camera` back. The
// viewer's setFromState already silences its onChange while applying, so
// no application-level loop guard is needed.

export const CHANNEL_NAME = 'voxa-mesh-sync';

export function openChannel() {
  if (typeof BroadcastChannel === 'undefined') return null;
  return new BroadcastChannel(CHANNEL_NAME);
}

export function postState(ch, state) {
  if (!ch) return;
  ch.postMessage({ type: 'state', ...state });
}

export function postCamera(ch, camera) {
  if (!ch || !camera) return;
  // THREE.Vector3 instances DO NOT survive structured-clone with their
  // x/y/z values intact (they round-trip as `{}`), so we hand-serialize
  // any Vector3-shaped fields. Without this, the receiver's
  // `target.copy(payload.target)` reads undefined coords and nukes the
  // camera position to NaN.
  const v3 = (v) => v && typeof v === 'object'
    ? { x: +v.x || 0, y: +v.y || 0, z: +v.z || 0 } : v;
  const out = { type: 'camera', camera: {} };
  if (camera.spherical) {
    out.camera.spherical = {
      r: +camera.spherical.r, phi: +camera.spherical.phi, theta: +camera.spherical.theta,
    };
  }
  if (camera.target) out.camera.target = v3(camera.target);
  if (camera.position) out.camera.position = v3(camera.position);
  if (camera.yaw != null) out.camera.yaw = +camera.yaw;
  if (camera.pitch != null) out.camera.pitch = +camera.pitch;
  ch.postMessage(out);
}

export function postRequestState(ch) {
  if (!ch) return;
  ch.postMessage({ type: 'request-state' });
}

export function isMeshCompanion() {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).get('mesh') === '1';
}
