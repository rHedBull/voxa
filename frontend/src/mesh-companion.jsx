// mesh-companion.jsx — secondary window that mirrors the main Voxa scene as
// a mesh-only view. Receives scene + cuboids + camera over BroadcastChannel
// from the main window, and broadcasts its own camera back so orbiting in
// either window keeps the two cameras in sync.

import { useEffect, useRef, useState } from 'react';
import { Viewer } from './viewer.jsx';
import { openChannel, postCamera, postRequestState } from './mesh-sync.js';

export function MeshCompanion() {
  const viewerRef = useRef(null);
  const channelRef = useRef(null);
  const lastSeenSceneRef = useRef(null);
  const [state, setState] = useState({
    scene: null,
    meshUrl: null,
    meshIsZUp: false,
    instances: [],
    bbox: null,
    background: '#0a0b0e',
    floorColor: '#15171c',
  });

  useEffect(() => {
    document.title = 'Voxa · Mesh companion';
  }, []);

  useEffect(() => {
    const ch = openChannel();
    channelRef.current = ch;
    if (!ch) return;
    ch.onmessage = (ev) => {
      const m = ev.data;
      if (!m || typeof m !== 'object') return;
      if (m.type === 'state') {
        setState({
          scene: m.scene ?? null,
          meshUrl: m.meshUrl ?? null,
          meshIsZUp: !!m.meshIsZUp,
          instances: m.instances ?? [],
          bbox: m.bbox ?? null,
          background: m.background ?? '#0a0b0e',
          floorColor: m.floorColor ?? '#15171c',
        });
      } else if (m.type === 'camera' && m.camera) {
        viewerRef.current?.setCameraState(m.camera);
      }
    };
    // Ask the main window to send a fresh state in case the companion was
    // opened after the main window's last broadcast.
    postRequestState(ch);
    return () => { ch.close(); channelRef.current = null; };
  }, []);

  // Frame the camera once per scene change so the companion lands on the
  // mesh instead of the origin if no camera msg has arrived yet.
  useEffect(() => {
    if (!state.bbox || state.scene === lastSeenSceneRef.current) return;
    lastSeenSceneRef.current = state.scene;
    const min = state.bbox.min, max = state.bbox.max;
    const cx = (min[0] + max[0]) / 2;
    const cy = (min[1] + max[1]) / 2;
    const cz = (min[2] + max[2]) / 2;
    const dx = max[0] - min[0];
    const dy = max[1] - min[1];
    const dz = max[2] - min[2];
    const radius = Math.max(dx, dy, dz, 1) * 0.6;
    viewerRef.current?.frame?.([cx, cy, cz], radius);
  }, [state.scene, state.bbox]);

  const onCameraChange = (cam) => {
    postCamera(channelRef.current, cam);
  };

  return (
    <div className="mesh-companion-root">
      <header className="mesh-companion-hd">
        <span className="dot" />
        <b>Mesh companion</b>
        <span className="muted">{state.scene || '— waiting for scene —'}</span>
        <span className="muted right">{state.instances.length} cuboid{state.instances.length === 1 ? '' : 's'}</span>
      </header>
      <div className="mesh-companion-vp">
        <Viewer
          ref={viewerRef}
          cloud={null}
          instances={state.instances}
          showCuboids
          background={state.background}
          floorColor={state.floorColor}
          showFloor
          showAxes
          meshUrl={state.meshUrl}
          meshIsZUp={state.meshIsZUp}
          showMesh={!!state.meshUrl}
          onCameraChange={onCameraChange}
        />
        {!state.meshUrl && (
          <div className="mesh-companion-empty">
            <div>
              <b>No mesh available</b>
              <p>This scene has no GLB. The companion mirrors the main window only when the active scene has a mesh.</p>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
