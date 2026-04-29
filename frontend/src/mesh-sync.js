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
  // Vector3 / Spherical clones from getState() are plain {x,y,z} / numeric
  // bags after structured-clone; no special serialization needed.
  ch.postMessage({ type: 'camera', camera });
}

export function postRequestState(ch) {
  if (!ch) return;
  ch.postMessage({ type: 'request-state' });
}

export function isMeshCompanion() {
  if (typeof window === 'undefined') return false;
  return new URLSearchParams(window.location.search).get('mesh') === '1';
}
