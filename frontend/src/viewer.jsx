// viewer.jsx
// Three.js viewport. Renders a point cloud, supports orbit/pan/zoom, draws
// cuboid overlays for instances, supports per-instance highlight, and exposes
// camera-state ref handles used by Compare mode for sync.
//
// Two camera control schemes:
//   navMode='orbit' — attachOrbit, the default, orbit/pan/zoom around a target
//   navMode='walk'  — attachWalk, FPS-style WASD on the XZ plane + Q/E for Y
//                     and mouse-drag to look around. W always moves you in the
//                     direction you're facing projected to horizontal, so
//                     looking down doesn't translate downward.

import { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import * as THREE from 'three';
import { GLTFLoader } from 'three/examples/jsm/loaders/GLTFLoader.js';
import { MeshoptDecoder } from 'three/examples/jsm/libs/meshopt_decoder.module.js';
import { TransformControls } from 'three/examples/jsm/controls/TransformControls.js';

// Converts a pointer event + bounding rect to normalized device coords [-1,1].
// Exported for unit testing.
export function evtToNdc(evt, rect) {
  return {
    x:  ((evt.clientX - rect.left) / rect.width)  * 2 - 1,
    y: -((evt.clientY - rect.top)  / rect.height) * 2 + 1,
  };
}

// Pick the index of the point under the cursor. Mirrors
// industrial-point-labeler: small fixed world-space threshold, then take
// the closest hit along the ray (Three.js sorts by along-ray distance,
// so hits[0] is the front-most point in the pick cylinder — i.e. the one
// not occluded by anything closer).
function pickPointSubRow(pointsObj, raycaster) {
  raycaster.params.Points = { threshold: 0.05 };
  const hits = raycaster.intersectObject(pointsObj);
  return hits.length ? hits[0].index : null;
}

// Inline HSL→RGB without THREE.Color allocation — used in the hot box-overlay loop.
function _hue2rgb(p, q, t) {
  if (t < 0) t += 1; if (t > 1) t -= 1;
  if (t < 1/6) return p + (q - p) * 6 * t;
  if (t < 1/2) return q;
  if (t < 2/3) return p + (q - p) * (2/3 - t) * 6;
  return p;
}

// Resolve a class palette (list of {id, color, label}) into a dense RGB lookup
// indexed by class id. Out-of-range ids fall through to the unlabeled grey.
function buildPaletteRGB(palette) {
  const out = [];
  const tmp = new THREE.Color();
  for (const entry of palette) {
    const id = entry.id | 0;
    if (id < 0) continue;
    tmp.set(entry.color);
    out[id] = [tmp.r, tmp.g, tmp.b];
  }
  return out;
}

// Build an inverse map: fullIdx → subsampled row index (or -1 if not rendered).
// subsampleIdx: Int32Array of length numSubsampled, value = full-res index.
// fullN: total number of full-resolution points.
export function buildFullToSubMap(subsampleIdx, fullN) {
  const map = new Int32Array(fullN).fill(-1);
  for (let sub = 0; sub < subsampleIdx.length; sub++) {
    map[subsampleIdx[sub]] = sub;
  }
  return map;
}

function attachOrbit(camera, dom, target, onChange) {
  const state = {
    dragging: false, mode: null, lx: 0, ly: 0,
    spherical: { r: 0, phi: 0, theta: 0 },
    target: target.clone(),
    silent: false,   // when true, apply() skips the onChange callback
    enabled: true,   // when false, ignore mouse + wheel (gizmo is dragging)
  };
  const off = camera.position.clone().sub(target);
  state.spherical.r = off.length();
  state.spherical.phi = Math.acos(Math.max(-1, Math.min(1, off.y / state.spherical.r)));
  state.spherical.theta = Math.atan2(off.x, off.z);

  const apply = () => {
    const { r, phi, theta } = state.spherical;
    camera.position.set(
      state.target.x + r * Math.sin(phi) * Math.sin(theta),
      state.target.y + r * Math.cos(phi),
      state.target.z + r * Math.sin(phi) * Math.cos(theta),
    );
    camera.lookAt(state.target);
    if (!state.silent) onChange && onChange(state);
  };
  state.silent = true; apply(); state.silent = false;

  const onDown = (e) => {
    if (!state.enabled) return;
    state.dragging = true;
    state.mode = e.button === 2 || e.shiftKey ? 'pan' : 'orbit';
    state.lx = e.clientX; state.ly = e.clientY;
    e.preventDefault();
  };
  const onMove = (e) => {
    if (!state.dragging) return;
    const dx = e.clientX - state.lx, dy = e.clientY - state.ly;
    state.lx = e.clientX; state.ly = e.clientY;
    if (state.mode === 'orbit') {
      state.spherical.theta -= dx * 0.005;
      state.spherical.phi = Math.max(0.05, Math.min(Math.PI - 0.05,
        state.spherical.phi - dy * 0.005));
    } else {
      const right = new THREE.Vector3();
      const up = new THREE.Vector3();
      camera.matrixWorld.extractBasis(right, up, new THREE.Vector3());
      const k = state.spherical.r * 0.0015;
      state.target.addScaledVector(right, -dx * k);
      state.target.addScaledVector(up, dy * k);
    }
    apply();
  };
  const onUp = () => { state.dragging = false; state.mode = null; };
  const onWheel = (e) => {
    if (!state.enabled) return;
    e.preventDefault();
    state.spherical.r = Math.max(0.1, Math.min(2000,
      state.spherical.r * (1 + Math.sign(e.deltaY) * 0.08)));
    apply();
  };
  const onCtx = (e) => e.preventDefault();

  dom.addEventListener('mousedown', onDown);
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);
  dom.addEventListener('wheel', onWheel, { passive: false });
  dom.addEventListener('contextmenu', onCtx);

  return {
    setFromState(s) {
      // Programmatic update — silence the onChange so synced viewports don't
      // ping-pong each other into a stack overflow. Shape-check guards against
      // cross-mode sync (walk-state arriving here during a navMode flip) and
      // malformed cross-window payloads (e.g. Vector3 lost across structured
      // clone) — applying NaN coords would warp the camera off the cloud.
      if (!s || !s.spherical || !s.target) return;
      const sp = s.spherical;
      if (!Number.isFinite(sp.r) || !Number.isFinite(sp.phi) || !Number.isFinite(sp.theta)) return;
      const t = s.target;
      if (!Number.isFinite(t.x) || !Number.isFinite(t.y) || !Number.isFinite(t.z)) return;
      state.silent = true;
      state.spherical = { r: sp.r, phi: sp.phi, theta: sp.theta };
      state.target.set(t.x, t.y, t.z);
      apply();
      state.silent = false;
    },
    getState() { return { spherical: { ...state.spherical }, target: state.target.clone() }; },
    setEnabled(on) {
      state.enabled = !!on;
      if (!on) { state.dragging = false; state.mode = null; }
    },
    frame(center, radius) {
      // Accept either a [x,y,z] tuple or a Vector3-shaped object — the
      // mesh companion frames the bbox via an array literal.
      if (Array.isArray(center)) state.target.set(center[0], center[1], center[2]);
      else if (center) state.target.copy(center);
      state.spherical.r = Math.max(0.4, radius * 2.4);
      apply();
    },
    preset(name, center, radius) {
      const r = radius != null ? radius * 2.4 : state.spherical.r;
      if (name === 'top')   { state.spherical.phi = 0.05;       state.spherical.theta = 0; }
      if (name === 'front') { state.spherical.phi = Math.PI / 2; state.spherical.theta = 0; }
      if (name === 'side')  { state.spherical.phi = Math.PI / 2; state.spherical.theta = Math.PI / 2; }
      if (name === 'iso')   { state.spherical.phi = 1.0;        state.spherical.theta = 0.7; }
      state.spherical.r = r;
      if (center) state.target.copy(center);
      apply();
    },
    dispose() {
      dom.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      dom.removeEventListener('wheel', onWheel);
      dom.removeEventListener('contextmenu', onCtx);
    },
  };
}

// FPS-style walkthrough controller.
// Movement keys (WASD) translate the camera in the XZ plane regardless of
// where it's looking — pitch only changes the look direction. Q/E are the
// only way to change Y. Mouse drag rotates yaw + pitch.
function attachWalk(camera, dom, sceneRadius, onChange) {
  const state = {
    yaw: 0, pitch: 0,
    keys: { w: false, a: false, s: false, d: false, q: false, e: false, shift: false },
    dragging: false, lx: 0, ly: 0,
    silent: false,
    enabled: true,
  };
  // Seed yaw/pitch from the camera's current look direction so toggling
  // doesn't snap the view.
  const dir = new THREE.Vector3();
  camera.getWorldDirection(dir);
  state.yaw = Math.atan2(dir.x, dir.z);
  state.pitch = Math.asin(Math.max(-1, Math.min(1, dir.y)));

  const apply = () => {
    const cy = Math.cos(state.yaw), sy = Math.sin(state.yaw);
    const cp = Math.cos(state.pitch), sp = Math.sin(state.pitch);
    camera.lookAt(
      camera.position.x + sy * cp,
      camera.position.y + sp,
      camera.position.z + cy * cp,
    );
    if (!state.silent) onChange && onChange(state);
  };
  state.silent = true; apply(); state.silent = false;

  // Drag with any mouse button (left, middle, right) rotates look. Right
  // button needs the contextmenu listener to keep the OS menu from popping.
  const onDown = (e) => {
    if (!state.enabled) return;
    state.dragging = true;
    state.lx = e.clientX; state.ly = e.clientY;
    e.preventDefault();
    e.stopPropagation();
  };
  const onMove = (e) => {
    if (!state.dragging) return;
    const dx = e.clientX - state.lx, dy = e.clientY - state.ly;
    state.lx = e.clientX; state.ly = e.clientY;
    state.yaw -= dx * 0.005;
    state.pitch = Math.max(-Math.PI / 2 + 0.01,
      Math.min(Math.PI / 2 - 0.01, state.pitch - dy * 0.005));
    apply();
  };
  const onUp = () => { state.dragging = false; };
  const onCtx = (e) => e.preventDefault();
  const onWheel = (e) => {
    if (!state.enabled) return;
    e.preventDefault();
    // Scroll in walk mode steps you forward/back in the horizontal plane.
    const fwd = new THREE.Vector3(Math.sin(state.yaw), 0, Math.cos(state.yaw));
    camera.position.addScaledVector(fwd, sceneRadius * 0.04 * -Math.sign(e.deltaY));
    apply();
  };

  const KEY_MAP = { w: 'w', a: 'a', s: 's', d: 'd', q: 'q', e: 'e' };
  const onKeyDown = (e) => {
    if (e.target && /INPUT|TEXTAREA|SELECT/.test(e.target.tagName)) return;
    const k = e.key.toLowerCase();
    if (k === 'shift') state.keys.shift = true;
    if (KEY_MAP[k] !== undefined) { state.keys[KEY_MAP[k]] = true; e.preventDefault(); }
  };
  const onKeyUp = (e) => {
    const k = e.key.toLowerCase();
    if (k === 'shift') state.keys.shift = false;
    if (KEY_MAP[k] !== undefined) state.keys[KEY_MAP[k]] = false;
  };

  // Movement loop — rAF-driven so multiple keys held at once compose smoothly
  // and the speed is frame-rate independent.
  let raf, last = performance.now();
  const tick = () => {
    const now = performance.now();
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;

    const baseSpeed = sceneRadius * (state.keys.shift ? 1.6 : 0.6);
    const v = baseSpeed * dt;
    // Three.js is right-handed, Y up. For a person facing direction
    // F=(sin(yaw),0,cos(yaw)) their right side (90° clockwise viewed from
    // above) is R=(-cos(yaw),0,sin(yaw)). The inverted form was making D
    // strafe left and A strafe right.
    const fwd = new THREE.Vector3(Math.sin(state.yaw), 0, Math.cos(state.yaw));
    const right = new THREE.Vector3(-Math.cos(state.yaw), 0, Math.sin(state.yaw));

    let moved = false;
    if (state.keys.w) { camera.position.addScaledVector(fwd,    v); moved = true; }
    if (state.keys.s) { camera.position.addScaledVector(fwd,   -v); moved = true; }
    if (state.keys.d) { camera.position.addScaledVector(right,  v); moved = true; }
    if (state.keys.a) { camera.position.addScaledVector(right, -v); moved = true; }
    if (state.keys.e) { camera.position.y += v;                     moved = true; }
    if (state.keys.q) { camera.position.y -= v;                     moved = true; }
    if (moved) apply();

    raf = requestAnimationFrame(tick);
  };
  tick();

  dom.addEventListener('mousedown', onDown);
  window.addEventListener('mousemove', onMove);
  window.addEventListener('mouseup', onUp);
  dom.addEventListener('wheel', onWheel, { passive: false });
  dom.addEventListener('contextmenu', onCtx);
  window.addEventListener('keydown', onKeyDown);
  window.addEventListener('keyup', onKeyUp);

  return {
    setFromState(s) {
      // Shape-check guards against cross-mode sync (orbit-state arriving here
      // during a navMode flip) and malformed cross-window payloads.
      if (!s || !s.position || s.yaw == null || s.pitch == null) return;
      const p = s.position;
      if (!Number.isFinite(p.x) || !Number.isFinite(p.y) || !Number.isFinite(p.z)) return;
      if (!Number.isFinite(s.yaw) || !Number.isFinite(s.pitch)) return;
      state.silent = true;
      camera.position.set(p.x, p.y, p.z);
      state.yaw = s.yaw; state.pitch = s.pitch;
      apply();
      state.silent = false;
    },
    getState() {
      return { position: camera.position.clone(), yaw: state.yaw, pitch: state.pitch };
    },
    setEnabled(on) {
      state.enabled = !!on;
      if (!on) {
        state.dragging = false;
        // Release any held movement keys so the camera doesn't keep drifting
        // while the gizmo has the input.
        Object.keys(state.keys).forEach((k) => { state.keys[k] = false; });
      }
    },
    frame() { /* no-op in walk mode — user navigates manually */ },
    preset() { /* no-op in walk mode — orbit presets don't apply */ },
    dispose() {
      cancelAnimationFrame(raf);
      dom.removeEventListener('mousedown', onDown);
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mouseup', onUp);
      dom.removeEventListener('wheel', onWheel);
      dom.removeEventListener('contextmenu', onCtx);
      window.removeEventListener('keydown', onKeyDown);
      window.removeEventListener('keyup', onKeyUp);
    },
  };
}

export const Viewer = forwardRef(function Viewer(props, ref) {
  const {
    cloud,                   // { positions, colors, bbox? } | null
    instances = [],
    visibleInstanceIds = null,
    highlightedId = null,
    selectedId = null,
    pointSize = 0.012,
    background = '#0a0b0e',
    floorColor = '#15171c',
    showFloor = true,
    showAxes = true,
    showCuboids = true,
    cuboidStyle = 'solid',
    cuboidOpacity = 1,
    colorMode = 'rgb',          // 'rgb' | 'height' | 'intensity' | 'class' | 'instance' | 'flat'
    navMode = 'orbit',          // 'orbit' | 'walk'
    meshUrl = null,             // GLB streaming URL
    meshOffset = null,          // [x,y,z] in pre-rotation world units; subtracted from mesh group position so a recentered cloud and the original-frame mesh overlay. Auto-derived from cloud.recenterOffset on the main viewer; the mesh-companion window passes it explicitly because it has no cloud.
    meshIsZUp = false,          // rotate GLB by -π/2 around X when its source frame is Z-up
    showMesh = false,           // when false, mesh stays unloaded (saves a 100MB+ fetch)
    meshBrightness = 1.0,       // multiplier on every loaded mesh material's color (0..2)
    onMeshLoadProgress = null,  // ({ loaded, total }) — wire-progress callback
    onCameraChange = null,
    diffMask = null,            // Uint8Array, length = num full-res points; 1 = changed vs prelabel
    showDiff = false,           // when true, tint diff points red using diffMask
    transformMode = null,       // 'translate' | 'rotate' | 'scale' | null — gizmo for selected cuboid
    onCuboidTransform = null,   // (id, { center, size, rotation }) => void; called on gizmo drag
    highlightCuboid = null,     // { center, size, rotation, color } — points inside this oriented box are tinted
    confirmedCuboids = null,    // [{ center, size, rotation }] — confirmed cuboids; always present (drives stats)
    confirmedPointsetHideMask = null, // Uint8Array sub-cloud length: 1 = belongs to a confirmed pointset; NaN'd alongside confirmed-cuboid points
    hideConfirmedPoints = true, // when true, points inside any confirmedCuboid are NaN'd in the position buffer
    onLabelStats = null,        // ({ total, labeled, left }) => void — reported after each confirmed-set change
    denseOverlay = null,        // { positions: Float32Array, colors: Float32Array | null } — full-density points to show inside the selected cuboid; takes precedence over the base cloud where they overlap
    segBoxes = null,            // { segIds, segCenters, segSizes, selection } — fallback bbox overlay per presegment
    segHulls = null,            // { vertices, faces, faceSeg, selection } — merged convex-hull overlay per presegment
    showSegHulls = true,        // hide the hull mesh when false; points still get segment-coloured
    selectionMask = null,       // Uint8Array sub-cloud length: 1 = belongs to selected instance; non-1 points are dimmed
  } = props;

  const mountRef = useRef(null);
  const stateRef = useRef({});
  // Keep onCameraChange callable from inside the long-lived controller
  // closure without recreating the controller every render.
  const onCameraChangeRef = useRef(onCameraChange);
  onCameraChangeRef.current = onCameraChange;
  // Same trick for the gizmo: TransformControls listeners are wired once at
  // mount, but the callback identity changes per render.
  const onCuboidTransformRef = useRef(onCuboidTransform);
  onCuboidTransformRef.current = onCuboidTransform;
  // Latest onLabelStats; read inside the hide-effect without forcing a re-run
  // when only the callback identity changes.
  const onLabelStatsRef = useRef(onLabelStats);
  onLabelStatsRef.current = onLabelStats;
  // ID of the cuboid the gizmo is currently attached to. Read inside the
  // gizmo's objectChange listener so we know which instance to patch.
  const gizmoTargetIdRef = useRef(null);
  // Set true while the user is dragging the gizmo. Used to suppress the
  // anchor-sync effect (which would otherwise overwrite the in-progress drag).
  const gizmoDraggingRef = useRef(false);

  // ── Mount once ─────────────────────────────────────────────────────────
  useEffect(() => {
    const mount = mountRef.current;
    const w = mount.clientWidth, h = mount.clientHeight;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
    // Survive context loss when a sibling tab/popup opens its own WebGL
    // context (Brave under snap/AppArmor evicts older contexts aggressively).
    // preventDefault on `lost` is required for the browser to fire `restored`
    // afterwards; the continuous rAF loop redraws automatically once the
    // GPU resources are re-uploaded by Three.js's internal handlers.
    const onCtxLost = (e) => { e.preventDefault(); };
    const onCtxRestored = () => {
      try { renderer.setSize(renderer.domElement.clientWidth,
                             renderer.domElement.clientHeight, false); }
      catch { /* renderer disposed mid-restore */ }
    };
    renderer.domElement.addEventListener('webglcontextlost', onCtxLost, false);
    renderer.domElement.addEventListener('webglcontextrestored', onCtxRestored, false);
    // Linear tone mapping lets per-material color > 1 actually brighten
    // pixels instead of clamping. Used by the mesh-brightness slider.
    renderer.toneMapping = THREE.LinearToneMapping;
    renderer.toneMappingExposure = 1.0;
    mount.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color(background);

    const camera = new THREE.PerspectiveCamera(40, w / h, 0.01, 5000);
    camera.position.set(1.6, 1.2, 1.8);

    const floor = new THREE.Mesh(
      new THREE.CircleGeometry(2.0, 64),
      new THREE.MeshBasicMaterial({ color: floorColor }),
    );
    floor.rotation.x = -Math.PI / 2;
    floor.position.y = -0.002;
    floor.visible = showFloor;
    scene.add(floor);

    const grid = new THREE.GridHelper(4, 40, 0x222630, 0x1a1c22);
    grid.position.y = -0.001;
    grid.material.opacity = 0.5;
    grid.material.transparent = true;
    grid.visible = showFloor;
    scene.add(grid);

    const axes = new THREE.AxesHelper(0.18);
    axes.position.set(-0.7, 0.005, -0.35);
    axes.visible = showAxes;
    scene.add(axes);

    const pointsMat = new THREE.PointsMaterial({
      size: pointSize, vertexColors: true,
      sizeAttenuation: true, transparent: false,
    });
    const pointsGeom = new THREE.BufferGeometry();
    const points = new THREE.Points(pointsGeom, pointsMat);
    scene.add(points);

    const cuboidGroup = new THREE.Group();
    scene.add(cuboidGroup);

    // Highlight overlay: a second Points object that re-renders only the
    // points falling inside the selected cuboid, in yellow and at a larger
    // size. Position buffer is sized to the active cloud (capped) and the
    // draw range shrinks to the actual count each update.
    const highlightGeom = new THREE.BufferGeometry();
    const highlightMat = new THREE.PointsMaterial({
      size: pointSize * 2.4,
      color: 0xfacc15, // tailwind yellow-400
      sizeAttenuation: true,
      transparent: false,
      depthWrite: true,
    });
    const highlightPoints = new THREE.Points(highlightGeom, highlightMat);
    highlightPoints.frustumCulled = false;
    highlightPoints.visible = false;
    scene.add(highlightPoints);

    // Selected-segment overlay. Same yellow + 2.4× size as cuboid highlight,
    // but driven independently by setSelectedSegmentMask() — populated when
    // the user clicks rows in the Presegment list or picks segments in 3D.
    const segSelectionGeom = new THREE.BufferGeometry();
    const segSelectionMat = new THREE.PointsMaterial({
      size: pointSize * 2.4,
      color: 0xfacc15,
      sizeAttenuation: true,
      transparent: false,
      depthWrite: true,
    });
    const segSelectionPoints = new THREE.Points(segSelectionGeom, segSelectionMat);
    segSelectionPoints.frustumCulled = false;
    segSelectionPoints.visible = false;
    scene.add(segSelectionPoints);

    // Dense overlay: full-density points fetched for the AABB of the selected
    // cuboid. Drawn on top of the base cloud so small geometry (pipes, valves)
    // is legible even when the base is strided to ~1M for performance. Uses
    // vertex colors from the LAZ source; a slight size bump distinguishes it
    // visually without being garish.
    const denseGeom = new THREE.BufferGeometry();
    const denseMat = new THREE.PointsMaterial({
      size: pointSize * 1.1,
      vertexColors: true,
      sizeAttenuation: true,
      transparent: false,
    });
    const densePoints = new THREE.Points(denseGeom, denseMat);
    densePoints.frustumCulled = false;
    densePoints.visible = false;
    scene.add(densePoints);

    // Instanced box overlay — one box per presegment, sized to its bounding box.
    // InstancedMesh is allocated to MAX_SEGS capacity; actual count is set via .count.
    const MAX_SEGS = 100000;
    const boxGeom = new THREE.BoxGeometry(1, 1, 1);
    const boxMat = new THREE.MeshBasicMaterial({
      transparent: true,
      opacity: 0.28,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const boxMesh = new THREE.InstancedMesh(boxGeom, boxMat, MAX_SEGS);
    boxMesh.count = 0;
    boxMesh.frustumCulled = false;
    boxMesh.visible = false;
    // Pre-allocate instanceColor so the box-overlay loop can write directly
    // to the underlying Float32Array without per-iteration setColorAt overhead.
    boxMesh.instanceColor = new THREE.InstancedBufferAttribute(
      new Float32Array(MAX_SEGS * 3), 3);
    scene.add(boxMesh);

    // Merged hull overlay — single BufferGeometry built from all segment
    // convex hulls (3d-labeler style). Replaces boxMesh visually when the
    // backend ships hull data; boxMesh stays as the click-pick fallback.
    const hullGeom = new THREE.BufferGeometry();
    const hullMat = new THREE.MeshBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.25,
      side: THREE.DoubleSide,
      depthWrite: false,
    });
    const hullMesh = new THREE.Mesh(hullGeom, hullMat);
    hullMesh.frustumCulled = false;
    hullMesh.visible = false;
    scene.add(hullMesh);

    // Gizmo anchor. Persistent Object3D the TransformControls is attached to —
    // we sync its position/rotation/scale from the selected instance, and
    // read them back on each gizmo drag tick. Decoupling the gizmo from the
    // edge-line meshes (which get rebuilt every instances change) keeps the
    // gizmo stable across re-renders.
    const transformAnchor = new THREE.Object3D();
    scene.add(transformAnchor);

    const transformControls = new TransformControls(camera, renderer.domElement);
    transformControls.setSpace('local');
    transformControls.setTranslationSnap(null);
    transformControls.setRotationSnap(null);
    transformControls.setScaleSnap(null);
    transformControls.visible = false;
    transformControls.enabled = false;
    scene.add(transformControls);

    // While the gizmo is being dragged, suspend orbit/walk pointer handling
    // so camera and gizmo don't fight for the same drag.
    transformControls.addEventListener('dragging-changed', (e) => {
      gizmoDraggingRef.current = !!e.value;
      const ctrl = stateRef.current?.controller;
      if (ctrl?.setEnabled) ctrl.setEnabled(!e.value);
    });

    // Read transform off the anchor on every change tick and patch the
    // selected instance. Scale is the size (anchor.scale === inst.size since
    // we drive scale directly from inst.size on attach).
    transformControls.addEventListener('objectChange', () => {
      const id = gizmoTargetIdRef.current;
      const cb = onCuboidTransformRef.current;
      if (!id || !cb) return;
      const c = transformAnchor.position;
      const r = transformAnchor.rotation;
      const sc = transformAnchor.scale;
      cb(id, {
        center: [c.x, c.y, c.z],
        rotation: [r.x, r.y, r.z],
        size: [Math.max(0.005, sc.x), Math.max(0.005, sc.y), Math.max(0.005, sc.z)],
      });
    });

    // Mesh container. The optional Z-up → Y-up rotation is applied when the
    // backend reports the mesh source is Z-up; the group also carries the
    // recenter offset so mesh and cloud overlay correctly.
    const meshGroup = new THREE.Group();
    meshGroup.visible = false;
    scene.add(meshGroup);

    // Lights so PBR/standard glTF materials don't render pure black. Cheap
    // and benign for the textured munich GLB; required for vertex-colored
    // BPA meshes whose default PBR material has no emissive.
    scene.add(new THREE.HemisphereLight(0xffffff, 0x222233, 1.0));
    const sunLight = new THREE.DirectionalLight(0xffffff, 0.8);
    sunLight.position.set(2, 4, 2);
    scene.add(sunLight);

    let raf;
    const tick = () => {
      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    tick();

    const onResize = () => {
      const r = mount.getBoundingClientRect();
      const W = Math.max(1, Math.floor(r.width || mount.clientWidth));
      const H = Math.max(1, Math.floor(r.height || mount.clientHeight));
      renderer.setSize(W, H, true);
      camera.aspect = W / H;
      camera.updateProjectionMatrix();
    };
    const ro = new ResizeObserver(onResize);
    ro.observe(mount);
    onResize();
    requestAnimationFrame(onResize);
    setTimeout(onResize, 50);
    setTimeout(onResize, 200);

    // Pointer-pick subscriber list: { cb } entries added by onPointerPick/onPointerMove.
    const pickSubs = [];
    const moveSubs = [];
    // Hull-pick subscriber list: cb(segId, evt) — fires when a hull face is clicked.
    const hullPickSubs = [];

    const raycaster = new THREE.Raycaster();

    const onPointerDown = (e) => {
      // Left button only — right/middle drags drive the camera and must
      // not trigger pick/select callbacks (otherwise orbiting silently
      // toggles whatever segment was under the cursor).
      if (e.button !== 0) return;
      const rect = renderer.domElement.getBoundingClientRect();
      const ndc = evtToNdc(e, rect);
      raycaster.setFromCamera(ndc, camera);

      // Hull hit check. Fires hullPickSubs when a segment hull face is
      // clicked; callbacks return truthy to consume the event (suppressing
      // the points-pick fallback). Plain clicks on a hull therefore fall
      // through to point-pick — important for volumetric "leftover" hulls
      // that envelope their points and would otherwise swallow every click.
      const fireHullPick = (segId) => {
        let consumed = false;
        hullPickSubs.forEach(({ cb }) => { if (cb(segId, e)) consumed = true; });
        return consumed;
      };
      const hm = stateRef.current?.hullMesh;
      if (hm?.visible && hullPickSubs.length > 0) {
        const hullHits = raycaster.intersectObject(hm);
        if (hullHits.length > 0) {
          const faceIdx = hullHits[0].faceIndex;
          const faceSeg = stateRef.current.hullFaceSeg;
          if (faceIdx !== undefined && faceSeg) {
            if (fireHullPick(faceSeg[faceIdx])) return;
          }
        }
      }
      // Box hit check (fallback when hulls aren't available).
      const bm = stateRef.current?.boxMesh;
      if (bm?.visible && hullPickSubs.length > 0) {
        const boxHits = raycaster.intersectObject(bm);
        if (boxHits.length > 0) {
          const instIdx = boxHits[0].instanceId;
          const segIds = stateRef.current.boxSegIds;
          if (instIdx !== undefined && segIds) {
            if (fireHullPick(segIds[instIdx])) return;
          }
        }
      }

      if (pickSubs.length === 0) return;
      const m = stateRef.current?.points;
      if (!m) return;
      const subRow = pickPointSubRow(m, raycaster, camera, ndc, rect);
      if (subRow == null) return;
      const subsampleIdx = m.userData.subsampleIdx;
      // subsampleIdx maps subsampled row → full-res index.
      // Falls back to subRow when subsampleIdx is not yet available (Task 20).
      const fullIndex = subsampleIdx ? subsampleIdx[subRow] : subRow;
      pickSubs.forEach(({ cb }) => cb(fullIndex, e));
    };

    const onPointerMove = (e) => {
      if (moveSubs.length === 0) return;
      const rect = renderer.domElement.getBoundingClientRect();
      const ndc = evtToNdc(e, rect);
      raycaster.setFromCamera(ndc, camera);
      const m = stateRef.current?.points;
      if (!m) return;
      const subRow = pickPointSubRow(m, raycaster, camera, ndc, rect);
      const subsampleIdx = m.userData.subsampleIdx;
      const fullIndex = subRow != null ? (subsampleIdx ? subsampleIdx[subRow] : subRow) : null;
      moveSubs.forEach(({ cb }) => cb(fullIndex, e));
    };

    renderer.domElement.addEventListener('pointerdown', onPointerDown);
    renderer.domElement.addEventListener('pointermove', onPointerMove);

    stateRef.current = {
      renderer, scene, camera, pointsGeom, pointsMat, points,
      controller: null,
      cuboidGroup, meshGroup, meshUrlLoaded: null, meshLoadAbort: null,
      floor, grid, axes, mount,
      pickSubs, moveSubs, hullPickSubs,
      transformAnchor, transformControls,
      highlightGeom, highlightMat, highlightPoints,
      segSelectionGeom, segSelectionMat, segSelectionPoints,
      denseGeom, denseMat, densePoints,
      boxGeom, boxMat, boxMesh, boxSegIds: null,
      hullGeom, hullMat, hullMesh, hullFaceSeg: null,
    };

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      stateRef.current.controller?.dispose();
      transformControls.detach();
      transformControls.dispose();
      scene.remove(transformControls);
      scene.remove(transformAnchor);
      renderer.domElement.removeEventListener('pointerdown', onPointerDown);
      renderer.domElement.removeEventListener('pointermove', onPointerMove);
      renderer.domElement.removeEventListener('webglcontextlost', onCtxLost);
      renderer.domElement.removeEventListener('webglcontextrestored', onCtxRestored);
      pointsGeom.dispose();
      pointsMat.dispose();
      highlightGeom.dispose();
      highlightMat.dispose();
      segSelectionGeom.dispose();
      segSelectionMat.dispose();
      boxGeom.dispose();
      boxMat.dispose();
      hullGeom.dispose();
      hullMat.dispose();
      renderer.forceContextLoss();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line
  }, []);

  // ── Camera controller (orbit ↔ walk) ────────────────────────────────────
  // Disposed and re-created when navMode flips. Seeds the new controller from
  // the prior camera state so toggling doesn't snap the view.
  useEffect(() => {
    const s = stateRef.current;
    if (!s.camera) return;
    s.controller?.dispose();
    const onChange = (st) => onCameraChangeRef.current && onCameraChangeRef.current(st);
    if (navMode === 'walk') {
      const sceneRadius = s._lastRadius || 1;
      s.controller = attachWalk(s.camera, s.renderer.domElement, sceneRadius, onChange);
    } else {
      const target = s._lastCenter ? s._lastCenter.clone() : new THREE.Vector3(0, 0.2, 0);
      s.controller = attachOrbit(s.camera, s.renderer.domElement, target, onChange);
    }
  }, [navMode]);

  // ── Cloud upload (rebuilds buffers when scene changes) ──────────────────
  useEffect(() => {
    const s = stateRef.current;
    if (!s.pointsGeom || !cloud) return;
    s.pointsGeom.setAttribute('position',
      new THREE.BufferAttribute(cloud.positions.slice(), 3));
    s.pointsGeom.setAttribute('color',
      new THREE.BufferAttribute(cloud.colors.slice(), 3));
    s.pointsGeom.computeBoundingSphere();
    s.points.userData.subsampleIdx = cloud.subsampleIdx ?? null;

    // Pre-size the highlight overlay buffer to match the rendered cloud.
    // Each update only fills the leading subset and uses setDrawRange.
    if (s.highlightGeom) {
      const N = cloud.positions.length / 3;
      const buf = new Float32Array(N * 3);
      const attr = new THREE.BufferAttribute(buf, 3);
      attr.setUsage(THREE.DynamicDrawUsage);
      s.highlightGeom.setAttribute('position', attr);
      s.highlightGeom.setDrawRange(0, 0);
      s.highlightPoints.visible = false;
    }

    // Same for the segment-selection overlay (independent of cuboid highlight).
    if (s.segSelectionGeom) {
      const N = cloud.positions.length / 3;
      const buf = new Float32Array(N * 3);
      const attr = new THREE.BufferAttribute(buf, 3);
      attr.setUsage(THREE.DynamicDrawUsage);
      s.segSelectionGeom.setAttribute('position', attr);
      s.segSelectionGeom.setDrawRange(0, 0);
      s.segSelectionPoints.visible = false;
    }

    // Apply the recenter offset to the mesh group so a co-located mesh
    // overlays the cloud. The group's rotation already brings GLB Z-up
    // into Y-up; the offset is in the post-rotation Y-up frame, applied
    // at the world level via group.position.
    if (cloud.recenterOffset) {
      const o = cloud.recenterOffset;
      s.meshGroup.position.set(-o[0], -o[1], -o[2]);
    } else {
      s.meshGroup.position.set(0, 0, 0);
    }

    if (cloud.bbox) {
      const c = new THREE.Vector3(
        (cloud.bbox.min[0] + cloud.bbox.max[0]) / 2,
        (cloud.bbox.min[1] + cloud.bbox.max[1]) / 2,
        (cloud.bbox.min[2] + cloud.bbox.max[2]) / 2,
      );
      const dx = cloud.bbox.max[0] - cloud.bbox.min[0];
      const dy = cloud.bbox.max[1] - cloud.bbox.min[1];
      const dz = cloud.bbox.max[2] - cloud.bbox.min[2];
      const radius = Math.sqrt(dx * dx + dy * dy + dz * dz) / 2;
      s.controller?.frame?.(c, radius);
      s._lastCenter = c;
      s._lastRadius = radius;

      // Resize floor + grid to match the cloud's xz footprint instead of a
      // fixed 4m square. The grid lives at the cloud's y-min so points sit
      // *on* it, not floating above an arbitrary plane.
      const horizExtent = Math.max(dx, dz);
      const floorRadius = Math.max(0.3, horizExtent * 0.7);
      const gridSize = floorRadius * 2;
      // Aim for ~25-40 divisions regardless of scale.
      const divisions = Math.max(10, Math.min(60, Math.round(gridSize / Math.max(0.1, horizExtent / 30))));

      s.floor.geometry.dispose();
      s.floor.geometry = new THREE.CircleGeometry(floorRadius, 64);
      s.floor.position.set(c.x, cloud.bbox.min[1] - 0.002, c.z);

      s.scene.remove(s.grid);
      s.grid.geometry.dispose();
      s.grid.material.dispose();
      const newGrid = new THREE.GridHelper(gridSize, divisions, 0x222630, 0x1a1c22);
      newGrid.position.set(c.x, cloud.bbox.min[1] - 0.001, c.z);
      newGrid.material.opacity = 0.5;
      newGrid.material.transparent = true;
      newGrid.visible = s.grid.visible;
      s.grid = newGrid;
      s.scene.add(newGrid);

      // Axes anchored at the bbox corner near the camera-facing front.
      s.axes.position.set(cloud.bbox.min[0], cloud.bbox.min[1] + 0.005, cloud.bbox.min[2]);
      s.axes.scale.setScalar(Math.max(0.1, horizExtent * 0.08));
    }
  }, [cloud]);

  // ── Cheap appearance updates ────────────────────────────────────────────
  // Kept off the recolor effect so dragging the point-size slider doesn't
  // trigger the per-point color loop on multi-million-point clouds.
  useEffect(() => {
    const s = stateRef.current;
    if (!s.pointsMat) return;
    s.pointsMat.size = pointSize;
    if (s.highlightMat) s.highlightMat.size = pointSize * 2.4;
    s.scene.background = new THREE.Color(background);
    s.floor.material.color = new THREE.Color(floorColor);
    s.floor.visible = showFloor;
    s.grid.visible = showFloor;
    s.axes.visible = showAxes;
  }, [pointSize, background, floorColor, showFloor, showAxes]);

  // ── Confirmed-cuboid mask + hide ────────────────────────────────────────
  // Build a per-point mask of points inside ANY confirmed cuboid (unique
  // count regardless of overlap). When `hideConfirmedPoints` is true, NaN
  // those positions so they vanish from rendering AND raycasting. Always
  // emit { total, labeled, left } via onLabelStats so the HUD can show the
  // unlabeled count even when hide is toggled off.
  useEffect(() => {
    const s = stateRef.current;
    if (!s.pointsGeom || !cloud) return;
    const posAttr = s.pointsGeom.getAttribute('position');
    if (!posAttr) return;
    const out = posAttr.array;
    out.set(cloud.positions);
    const N = out.length;
    const numPts = N / 3;

    let labeled = 0;
    const hasCuboids = confirmedCuboids && confirmedCuboids.length > 0;
    const hasPointsetMask = confirmedPointsetHideMask
      && confirmedPointsetHideMask.length === numPts;
    if (hasCuboids || hasPointsetMask) {
      const mask = new Uint8Array(numPts);
      if (hasPointsetMask) {
        for (let i = 0; i < numPts; i++) if (confirmedPointsetHideMask[i]) mask[i] = 1;
      }
      for (const cub of (confirmedCuboids || [])) {
        const cx = cub.center[0], cy = cub.center[1], cz = cub.center[2];
        const hx = cub.size[0] / 2, hy = cub.size[1] / 2, hz = cub.size[2] / 2;
        const rx = cub.rotation?.[0] || 0;
        const ry = cub.rotation?.[1] || 0;
        const rz = cub.rotation?.[2] || 0;
        const isAA = rx === 0 && ry === 0 && rz === 0;
        if (isAA) {
          for (let p = 0, i = 0; p < N; p += 3, i++) {
            if (mask[i]) continue;
            const dx = out[p] - cx, dy = out[p + 1] - cy, dz = out[p + 2] - cz;
            if (dx > -hx && dx < hx && dy > -hy && dy < hy && dz > -hz && dz < hz) {
              mask[i] = 1;
            }
          }
        } else {
          const mtx = new THREE.Matrix4().makeRotationFromEuler(
            new THREE.Euler(rx, ry, rz, 'XYZ')
          ).invert();
          const e = mtx.elements;
          const e0 = e[0], e1 = e[1], e2 = e[2];
          const e4 = e[4], e5 = e[5], e6 = e[6];
          const e8 = e[8], e9 = e[9], e10 = e[10];
          for (let p = 0, i = 0; p < N; p += 3, i++) {
            if (mask[i]) continue;
            const dx = out[p] - cx, dy = out[p + 1] - cy, dz = out[p + 2] - cz;
            const lx = e0 * dx + e4 * dy + e8 * dz;
            const ly = e1 * dx + e5 * dy + e9 * dz;
            const lz = e2 * dx + e6 * dy + e10 * dz;
            if (lx > -hx && lx < hx && ly > -hy && ly < hy && lz > -hz && lz < hz) {
              mask[i] = 1;
            }
          }
        }
      }
      if (hideConfirmedPoints) {
        for (let p = 0, i = 0; p < N; p += 3, i++) {
          if (mask[i]) {
            labeled++;
            out[p] = NaN; out[p + 1] = NaN; out[p + 2] = NaN;
          }
        }
      } else {
        for (let i = 0; i < numPts; i++) if (mask[i]) labeled++;
      }
    }
    posAttr.needsUpdate = true;
    onLabelStatsRef.current?.({ total: numPts, labeled, left: numPts - labeled });
  }, [confirmedCuboids, confirmedPointsetHideMask, hideConfirmedPoints, cloud]);

  // ── Per-point color recompute ───────────────────────────────────────────
  useEffect(() => {
    const s = stateRef.current;
    if (!s.pointsGeom) return;
    const colorAttr = s.pointsGeom.getAttribute('color');
    const posAttr = s.pointsGeom.getAttribute('position');
    if (!colorAttr || !cloud) return;
    const orig = cloud.colors;

    // While a presegmentation is active, points get the same segment-id hue
    // the hull mesh uses, regardless of the global ``colorMode`` setting.
    // The user is grouping points, not labelling them, so class/RGB modes
    // would just hide the grouping. Same _hue2rgb math as the hull effect
    // so points and hulls are exact colour matches.
    const segActive = !!(segHulls && segHulls.faces && segHulls.faces.length && cloud.instanceIds);
    if (segActive) {
      const ids = cloud.instanceIds;
      const sel = segHulls.selection;
      const G = 0.6180339887;
      const arr = colorAttr.array;
      for (let i = 0, p = 0; i < arr.length; i += 3, p++) {
        const iid = ids[p];
        if (iid < 0) {
          arr[i] = 0.42; arr[i + 1] = 0.44; arr[i + 2] = 0.48;
          continue;
        }
        const hue = ((iid * G) % 1 + 1) % 1;
        const isSel = sel && sel.has(iid);
        const l = isSel ? 0.82 : 0.52;
        const pp = l <= 0.5 ? l * 1.7 : l + 0.7 - l * 0.7;
        const q = 2 * l - pp;
        arr[i]     = _hue2rgb(q, pp, hue + 1 / 3);
        arr[i + 1] = _hue2rgb(q, pp, hue);
        arr[i + 2] = _hue2rgb(q, pp, hue - 1 / 3);
      }
    } else if (colorMode === 'height' && cloud.bbox) {
      // Per-point gradient by Y, normalized to the cloud's own Y range.
      const yMin = cloud.bbox.min[1];
      const yMax = cloud.bbox.max[1];
      const yRange = Math.max(yMax - yMin, 1e-6);
      const lo = new THREE.Color('#1d4ed8');   // deep blue (low)
      const mid = new THREE.Color('#10b981');  // green (mid)
      const hi = new THREE.Color('#f59e0b');   // amber (high)
      const tmp = new THREE.Color();
      for (let i = 0, p = 0; i < colorAttr.array.length; i += 3, p += 3) {
        const t = (posAttr.array[p + 1] - yMin) / yRange;
        if (t < 0.5) tmp.copy(lo).lerp(mid, t * 2);
        else         tmp.copy(mid).lerp(hi, (t - 0.5) * 2);
        colorAttr.array[i + 0] = tmp.r;
        colorAttr.array[i + 1] = tmp.g;
        colorAttr.array[i + 2] = tmp.b;
      }
    } else if (colorMode === 'intensity') {
      // Real per-point intensity when the loader provided it (LAS/LAZ),
      // otherwise pseudo-intensity from RGB luminance.
      if (cloud.intensity && cloud.intensity.length * 3 === colorAttr.array.length) {
        for (let i = 0, p = 0; i < colorAttr.array.length; i += 3, p++) {
          const t = cloud.intensity[p];
          colorAttr.array[i + 0] = t;
          colorAttr.array[i + 1] = t;
          colorAttr.array[i + 2] = t;
        }
      } else {
        for (let i = 0; i < orig.length; i += 3) {
          const lum = orig[i] * 0.299 + orig[i + 1] * 0.587 + orig[i + 2] * 0.114;
          colorAttr.array[i + 0] = lum;
          colorAttr.array[i + 1] = lum;
          colorAttr.array[i + 2] = lum;
        }
      }
    } else if (colorMode === 'class' && cloud.classIds && cloud.classPalette) {
      // Per-point class color from the palette. Unlabeled (-1) → muted grey.
      const paletteRgb = buildPaletteRGB(cloud.classPalette);
      const grey = [0.42, 0.44, 0.48];
      const ids = cloud.classIds;
      for (let i = 0, p = 0; i < colorAttr.array.length; i += 3, p++) {
        const cid = ids[p];
        const rgb = cid >= 0 ? (paletteRgb[cid] || grey) : grey;
        colorAttr.array[i + 0] = rgb[0];
        colorAttr.array[i + 1] = rgb[1];
        colorAttr.array[i + 2] = rgb[2];
      }
    } else if (colorMode === 'instance' && cloud.instanceIds) {
      // Hash-based hue per instance id. Stable across reloads of the same cloud.
      const ids = cloud.instanceIds;
      const tmp = new THREE.Color();
      for (let i = 0, p = 0; i < colorAttr.array.length; i += 3, p++) {
        const iid = ids[p];
        if (iid < 0) {
          colorAttr.array[i + 0] = 0.42;
          colorAttr.array[i + 1] = 0.44;
          colorAttr.array[i + 2] = 0.48;
        } else {
          // Golden-ratio hue spacing — visually well-separated even for adjacent ids.
          const hue = ((iid * 0.6180339887) % 1 + 1) % 1;
          tmp.setHSL(hue, 0.62, 0.58);
          colorAttr.array[i + 0] = tmp.r;
          colorAttr.array[i + 1] = tmp.g;
          colorAttr.array[i + 2] = tmp.b;
        }
      }
    } else if (colorMode === 'flat') {
      const c = new THREE.Color('#7c8088');
      for (let i = 0; i < orig.length; i += 3) {
        colorAttr.array[i + 0] = c.r;
        colorAttr.array[i + 1] = c.g;
        colorAttr.array[i + 2] = c.b;
      }
    } else {
      colorAttr.array.set(orig);
    }

    if (showDiff && diffMask) {
      const subsampleIdx = s.points.userData.subsampleIdx;
      if (subsampleIdx) {
        for (let sub = 0; sub < subsampleIdx.length; sub++) {
          const fullIdx = subsampleIdx[sub];
          if (fullIdx < diffMask.length && diffMask[fullIdx] === 1) {
            colorAttr.array[sub * 3]     = 1.0;
            colorAttr.array[sub * 3 + 1] = 0.18;
            colorAttr.array[sub * 3 + 2] = 0.18;
          }
        }
      } else {
        for (let i = 0; i < diffMask.length; i++) {
          if (diffMask[i] === 1) {
            colorAttr.array[i * 3]     = 1.0;
            colorAttr.array[i * 3 + 1] = 0.18;
            colorAttr.array[i * 3 + 2] = 0.18;
          }
        }
      }
    }

    if (selectionMask && selectionMask.length * 3 === colorAttr.array.length) {
      const arr = colorAttr.array;
      for (let p = 0; p < selectionMask.length; p++) {
        if (selectionMask[p]) continue;
        const i = p * 3;
        arr[i]     = 0.22;
        arr[i + 1] = 0.23;
        arr[i + 2] = 0.26;
      }
    }

    colorAttr.needsUpdate = true;
  }, [colorMode, cloud, showDiff, diffMask, segHulls, selectionMask]);

  // ── Selected-cuboid highlight overlay ───────────────────────────────────
  // Re-populates a separate Points buffer with the cloud points that fall
  // inside the selected cuboid (oriented AABB). Runs on every gizmo drag
  // since highlightCuboid is a fresh object whenever any of its fields
  // change. Inside-test is the same math as the IoU but in JS.
  useEffect(() => {
    const s = stateRef.current;
    if (!s.highlightGeom || !s.highlightPoints) return;
    const hAttr = s.highlightGeom.getAttribute('position');
    if (!hAttr) {
      s.highlightPoints.visible = false;
      return;
    }
    if (!highlightCuboid) {
      s.highlightGeom.setDrawRange(0, 0);
      s.highlightPoints.visible = false;
      return;
    }
    const posAttr = s.pointsGeom?.getAttribute('position');
    if (!posAttr) {
      s.highlightPoints.visible = false;
      return;
    }

    const { center, size, rotation } = highlightCuboid;
    const cx = center[0], cy = center[1], cz = center[2];
    const hx = size[0] / 2, hy = size[1] / 2, hz = size[2] / 2;
    const rx = rotation?.[0] || 0, ry = rotation?.[1] || 0, rz = rotation?.[2] || 0;
    const isAA = rx === 0 && ry === 0 && rz === 0;
    const pos = posAttr.array;
    const out = hAttr.array;
    const N = pos.length;
    let m = 0;

    if (isAA) {
      for (let p = 0; p < N; p += 3) {
        const dx = pos[p] - cx, dy = pos[p + 1] - cy, dz = pos[p + 2] - cz;
        if (dx > -hx && dx < hx && dy > -hy && dy < hy && dz > -hz && dz < hz) {
          out[m++] = pos[p];
          out[m++] = pos[p + 1];
          out[m++] = pos[p + 2];
        }
      }
    } else {
      // Inverse rotation matrix; XYZ Euler order matches cuboid line meshes.
      const mtx = new THREE.Matrix4().makeRotationFromEuler(
        new THREE.Euler(rx, ry, rz, 'XYZ')
      ).invert();
      const e = mtx.elements;
      const e0 = e[0], e1 = e[1], e2 = e[2];
      const e4 = e[4], e5 = e[5], e6 = e[6];
      const e8 = e[8], e9 = e[9], e10 = e[10];
      for (let p = 0; p < N; p += 3) {
        const dx = pos[p] - cx, dy = pos[p + 1] - cy, dz = pos[p + 2] - cz;
        const lx = e0 * dx + e4 * dy + e8  * dz;
        const ly = e1 * dx + e5 * dy + e9  * dz;
        const lz = e2 * dx + e6 * dy + e10 * dz;
        if (lx > -hx && lx < hx && ly > -hy && ly < hy && lz > -hz && lz < hz) {
          out[m++] = pos[p];
          out[m++] = pos[p + 1];
          out[m++] = pos[p + 2];
        }
      }
    }

    const count = m / 3;
    s.highlightGeom.setDrawRange(0, count);
    hAttr.needsUpdate = true;
    s.highlightPoints.visible = count > 0;
  }, [highlightCuboid, cloud]);

  // ── Dense overlay (full-density points around selected cuboid) ──────────
  // Replaces the geometry on each `denseOverlay` change. We allocate fresh
  // buffers each time (instead of a fixed cap + drawRange like the highlight
  // overlay) because the per-region count varies wildly with cuboid size and
  // scene density. Colors fall back to a flat tint when the LAZ has no RGB.
  useEffect(() => {
    const s = stateRef.current;
    if (!s.denseGeom || !s.densePoints) return;
    if (!denseOverlay || !denseOverlay.positions || denseOverlay.positions.length === 0) {
      s.denseGeom.setDrawRange(0, 0);
      s.densePoints.visible = false;
      return;
    }
    const positions = denseOverlay.positions;
    const n = positions.length / 3;
    let colors = denseOverlay.colors;
    if (!colors || colors.length !== n * 3) {
      // Flat warm tint so the region still reads as "extra detail" without RGB.
      colors = new Float32Array(n * 3);
      for (let i = 0; i < n; i++) {
        colors[i * 3]     = 0.85;
        colors[i * 3 + 1] = 0.78;
        colors[i * 3 + 2] = 0.55;
      }
    }
    s.denseGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    s.denseGeom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    s.denseGeom.computeBoundingSphere();
    s.denseGeom.setDrawRange(0, n);
    s.densePoints.visible = true;
  }, [denseOverlay]);

  // ── Instanced-box overlay (fallback) ────────────────────────────────────
  // One box per presegment sized to its axis-aligned bbox. InstancedMesh
  // renders all boxes in a single draw call. Selected segments are brighter.
  // Suppressed when ``segHulls`` is present — hulls give a tighter fit.
  useEffect(() => {
    const s = stateRef.current;
    if (!s.boxMesh) return;
    const hullsActive = !!(segHulls && segHulls.faces && segHulls.faces.length);
    if (hullsActive || !segBoxes || !segBoxes.segIds || segBoxes.segIds.length === 0) {
      s.boxMesh.count = 0;
      s.boxMesh.visible = false;
      s.boxSegIds = null;
      return;
    }

    const { segIds, segCenters, segSizes, selection } = segBoxes;
    const n = segIds.length;
    const GOLDEN = 0.6180339887;
    const matArr = s.boxMesh.instanceMatrix.array;
    const colArr = s.boxMesh.instanceColor.array;

    for (let i = 0; i < n; i++) {
      const sid = segIds[i];
      const cx = segCenters[i * 3], cy = segCenters[i * 3 + 1], cz = segCenters[i * 3 + 2];
      const sx = segSizes[i * 3],   sy = segSizes[i * 3 + 1],   sz = segSizes[i * 3 + 2];
      // Write scale+translate matrix directly (Three.js column-major Float32Array).
      const m = i * 16;
      matArr[m]      = sx; matArr[m+1]  = 0;  matArr[m+2]  = 0;  matArr[m+3]  = 0;
      matArr[m+4]    = 0;  matArr[m+5]  = sy; matArr[m+6]  = 0;  matArr[m+7]  = 0;
      matArr[m+8]    = 0;  matArr[m+9]  = 0;  matArr[m+10] = sz; matArr[m+11] = 0;
      matArr[m+12]   = cx; matArr[m+13] = cy; matArr[m+14] = cz; matArr[m+15] = 1;
      // Inline HSL→RGB (avoids per-iter THREE.Color allocation + method call overhead).
      const isSel = selection && selection.has(sid);
      const hue = ((sid * GOLDEN) % 1 + 1) % 1;
      const l = isSel ? 0.82 : 0.52;
      const p = l <= 0.5 ? l * 1.7 : l + 0.7 - l * 0.7;
      const q = 2 * l - p;
      const c = i * 3;
      colArr[c]   = _hue2rgb(q, p, hue + 1/3);
      colArr[c+1] = _hue2rgb(q, p, hue);
      colArr[c+2] = _hue2rgb(q, p, hue - 1/3);
    }

    s.boxMesh.count = n;
    s.boxMesh.instanceMatrix.needsUpdate = true;
    s.boxMesh.instanceColor.needsUpdate = true;
    s.boxSegIds = segIds;
    s.boxMesh.visible = true;
  }, [segBoxes, segHulls]);

  // ── Merged hull overlay ──────────────────────────────────────────────────
  // Per-segment convex hulls packed into a single BufferGeometry with vertex
  // colors. Mirrors 3d-labeler's `SupervoxelHulls` component: one mesh, one
  // draw call, translucent / double-sided / no depth-write so points stay
  // visible behind the hulls. Per-vertex hue cycles via golden-ratio HSL;
  // selected segments are brightened by lifting the lightness term.
  useEffect(() => {
    const s = stateRef.current;
    if (!s.hullMesh) return;
    if (!segHulls || !segHulls.faces || segHulls.faces.length === 0) {
      s.hullMesh.visible = false;
      s.hullFaceSeg = null;
      return;
    }

    const { vertices, faces, faceSeg, selection } = segHulls;
    const nVerts = vertices.length / 3;
    const colors = new Float32Array(nVerts * 3);
    const GOLDEN = 0.6180339887;

    // Per-face → per-vertex color: the same vertex can belong to multiple
    // faces of the same hull, so writing color per-vertex (using the first
    // face that touches it) is sufficient. We walk faces once and stamp.
    for (let f = 0; f < faceSeg.length; f++) {
      const sid = faceSeg[f];
      const isSel = selection && selection.has(sid);
      const hue = ((sid * GOLDEN) % 1 + 1) % 1;
      const l = isSel ? 0.82 : 0.52;
      const p = l <= 0.5 ? l * 1.7 : l + 0.7 - l * 0.7;
      const q = 2 * l - p;
      const r = _hue2rgb(q, p, hue + 1/3);
      const g = _hue2rgb(q, p, hue);
      const b = _hue2rgb(q, p, hue - 1/3);
      const v0 = faces[f * 3], v1 = faces[f * 3 + 1], v2 = faces[f * 3 + 2];
      colors[v0 * 3] = r; colors[v0 * 3 + 1] = g; colors[v0 * 3 + 2] = b;
      colors[v1 * 3] = r; colors[v1 * 3 + 1] = g; colors[v1 * 3 + 2] = b;
      colors[v2 * 3] = r; colors[v2 * 3 + 1] = g; colors[v2 * 3 + 2] = b;
    }

    const geom = s.hullMesh.geometry;
    geom.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
    geom.setIndex(new THREE.BufferAttribute(faces, 1));
    geom.computeVertexNormals();
    geom.computeBoundingSphere();

    s.hullFaceSeg = faceSeg;
    s.hullMesh.visible = !!showSegHulls;
  }, [segHulls, showSegHulls]);

  // ── Mesh load/show ──────────────────────────────────────────────────────
  // The GLB sits at meshUrl on the backend. Loading is gated by showMesh —
  // these files are huge (100MB+ for munich_water_pump). Once loaded for a
  // given URL we keep it around so toggling visibility off/on is free; when
  // the URL changes (different scene) we dispose and reload only on demand.
  useEffect(() => {
    const s = stateRef.current;
    if (!s.meshGroup) return;

    // Drop any stale mesh when the scene's mesh URL changes.
    if (s.meshUrlLoaded && s.meshUrlLoaded !== meshUrl) {
      while (s.meshGroup.children.length) {
        const c = s.meshGroup.children.pop();
        c.traverse?.((n) => {
          n.geometry?.dispose?.();
          if (n.material) {
            (Array.isArray(n.material) ? n.material : [n.material])
              .forEach((m) => { m.map?.dispose?.(); m.dispose?.(); });
          }
        });
      }
      s.meshUrlLoaded = null;
    }

    s.meshGroup.rotation.x = meshIsZUp ? -Math.PI / 2 : 0;
    s.meshGroup.visible = !!(showMesh && (meshUrl || s.meshUrlLoaded));
    // Mesh-or-points: when the mesh is being shown AND has finished loading,
    // hide the cloud so the user sees one or the other.
    s.points.visible = !(showMesh && s.meshUrlLoaded === meshUrl && !!meshUrl);

    // Don't fetch unless asked. This keeps default scene loads fast.
    if (!showMesh || !meshUrl || s.meshUrlLoaded === meshUrl) return;

    let cancelled = false;
    // The build pipeline writes EXT_meshopt_compression buffers (the
    // r05 / r05.small / .optimized GLB variants all use it). Without the
    // decoder registered, GLTFLoader throws "setMeshoptDecoder must be
    // called before loading compressed files" inside the loader Promise
    // and the mesh silently never appears.
    const loader = new GLTFLoader().setMeshoptDecoder(MeshoptDecoder);
    loader.load(
      meshUrl,
      (gltf) => {
        if (cancelled) return;
        // Swap PBR materials to fullbright Basic so vertex colors and
        // baked textures render at capture brightness — these meshes ARE
        // the data, we don't want lighting to multiply/darken them.
        gltf.scene.traverse((node) => {
          if (!node.isMesh || !node.material) return;
          const old = node.material;
          const m = new THREE.MeshBasicMaterial({
            vertexColors: true,
            map: old.map || null,
            side: THREE.DoubleSide,
          });
          node.material = m;
          old.dispose?.();
        });
        s.meshGroup.add(gltf.scene);
        s.meshUrlLoaded = meshUrl;
        s.meshGroup.visible = true;
        // Once the mesh has loaded and we're in showMesh mode, hide the
        // cloud so the user sees just the mesh.
        if (showMesh) s.points.visible = false;
      },
      (xhr) => {
        if (cancelled || !onMeshLoadProgress) return;
        onMeshLoadProgress({ loaded: xhr.loaded, total: xhr.total || 0 });
      },
      (err) => {
        if (!cancelled) console.error('GLB load failed:', err);
      },
    );
    return () => { cancelled = true; };
  }, [meshUrl, meshIsZUp, showMesh, onMeshLoadProgress]);

  // Mesh brightness: multiply every loaded mesh material's base color.
  // Linear tone mapping on the renderer lets values >1 actually brighten.
  useEffect(() => {
    const s = stateRef.current;
    if (!s.meshGroup) return;
    const b = Math.max(0, meshBrightness);
    s.meshGroup.traverse((node) => {
      if (!node.isMesh || !node.material) return;
      (Array.isArray(node.material) ? node.material : [node.material])
        .forEach((m) => { m.color?.setRGB(b, b, b); });
    });
  }, [meshBrightness, meshUrl, showMesh]);

  // Mesh-group offset, decoupled from cloud-load. The main viewer derives
  // the offset from cloud.recenterOffset inside the cloud-upload effect
  // (above), but that effect doesn't run in the mesh-companion window
  // (cloud is null). Apply meshOffset here so the popup can position its
  // mesh into the same recentered frame as the cloud.
  useEffect(() => {
    const s = stateRef.current;
    if (!s.meshGroup || !meshOffset) return;
    s.meshGroup.position.set(-meshOffset[0], -meshOffset[1], -meshOffset[2]);
  }, [meshOffset]);

  // ── Cuboid rebuild ──────────────────────────────────────────────────────
  useEffect(() => {
    const s = stateRef.current;
    if (!s.cuboidGroup) return;
    while (s.cuboidGroup.children.length) {
      const c = s.cuboidGroup.children.pop();
      c.geometry && c.geometry.dispose();
      c.material && c.material.dispose();
    }
    if (!showCuboids) return;
    const visible = visibleInstanceIds
      ? new Set(visibleInstanceIds)
      : new Set(instances.map((i) => i.id));
    instances.forEach((inst) => {
      if (!visible.has(inst.id)) return;
      // Pointset instances are display-only in the cuboid renderer, even when
      // they carry a persisted OBB (Box-tool selection volume now stores
      // center/size/rotation). Gate on kind, not on null center/size — the
      // mesh-companion window receives raw gtInstances and hits this path with
      // mixed kinds.
      if (inst.kind === 'pointset' || !inst.size || !inst.center) return;
      const isHi = inst.id === highlightedId || inst.id === selectedId;
      const box = new THREE.BoxGeometry(...inst.size);
      const edges = new THREE.EdgesGeometry(box);
      const lineMat = new THREE.LineBasicMaterial({
        color: inst.color, transparent: true,
        opacity: cuboidOpacity * (isHi ? 1 : 0.85),
      });
      const line = cuboidStyle === 'dashed'
        ? new THREE.LineSegments(edges, new THREE.LineDashedMaterial({
            color: inst.color, dashSize: 0.04, gapSize: 0.025,
            transparent: true, opacity: cuboidOpacity * (isHi ? 1 : 0.85),
          }))
        : new THREE.LineSegments(edges, lineMat);
      if (cuboidStyle === 'dashed') line.computeLineDistances();
      line.position.set(...inst.center);
      if (inst.rotation) line.rotation.set(...inst.rotation);
      s.cuboidGroup.add(line);

      if (isHi) {
        const fillMat = new THREE.MeshBasicMaterial({
          color: inst.color, transparent: true, opacity: 0.10, depthWrite: false,
        });
        const fill = new THREE.Mesh(box, fillMat);
        fill.position.set(...inst.center);
        if (inst.rotation) fill.rotation.set(...inst.rotation);
        s.cuboidGroup.add(fill);
      }
    });
  }, [instances, visibleInstanceIds, highlightedId, selectedId, showCuboids, cuboidStyle, cuboidOpacity]);

  // ── Gizmo: attach to selected cuboid, sync transform from props ─────────
  // - Skipped while the user is actively dragging (the anchor IS the truth
  //   then, and onCuboidTransform is feeding incoming props).
  // - Detaches the gizmo when nothing is selected, transformMode is null,
  //   or when cuboids are hidden.
  useEffect(() => {
    const s = stateRef.current;
    const tc = s.transformControls;
    const anchor = s.transformAnchor;
    if (!tc || !anchor) return;

    const selected = selectedId ? instances.find((i) => i.id === selectedId) : null;
    // Pointset instances have no cuboid → no gizmo, no anchor sync.
    const isCuboid = selected && selected.kind !== 'pointset' && selected.center && selected.size;
    const wantGizmo = !!(isCuboid && transformMode && showCuboids);

    if (!wantGizmo) {
      tc.detach();
      tc.visible = false;
      tc.enabled = false;
      gizmoTargetIdRef.current = null;
      return;
    }

    // Sync anchor only when the user is not currently dragging it. During
    // drag, anchor changes drive the props, not the other way around.
    if (!gizmoDraggingRef.current) {
      anchor.position.set(...selected.center);
      anchor.rotation.set(...(selected.rotation || [0, 0, 0]));
      const sz = selected.size || [1, 1, 1];
      anchor.scale.set(
        Math.max(0.005, sz[0]),
        Math.max(0.005, sz[1]),
        Math.max(0.005, sz[2]),
      );
      anchor.updateMatrixWorld(true);
    }

    gizmoTargetIdRef.current = selected.id;
    tc.setMode(transformMode);
    if (tc.object !== anchor) tc.attach(anchor);
    tc.visible = true;
    tc.enabled = true;
  }, [selectedId, instances, transformMode, showCuboids]);

  useImperativeHandle(ref, () => ({
    preset(name) {
      const s = stateRef.current;
      s.controller?.preset?.(name, s._lastCenter, s._lastRadius);
    },
    frame(center, radius) { stateRef.current.controller?.frame?.(center, radius); },
    setCameraState(s) { stateRef.current.controller?.setFromState?.(s); },
    getCameraState() { return stateRef.current.controller?.getState?.(); },
    domElement() {
      return stateRef.current.renderer?.domElement ?? null;
    },
    cameraForward() {
      const s = stateRef.current;
      if (!s.camera) return [0, 0, -1];
      const dir = new THREE.Vector3();
      s.camera.getWorldDirection(dir);
      return [dir.x, dir.y, dir.z];
    },
    firstHitUnderCursor(evt) {
      const s = stateRef.current;
      if (!s.camera || !s.points) return null;
      const rect = s.renderer.domElement.getBoundingClientRect();
      const ndc = evtToNdc(evt, rect);
      const raycaster = new THREE.Raycaster();
      raycaster.setFromCamera(ndc, s.camera);
      const subRow = pickPointSubRow(s.points, raycaster, s.camera, ndc, rect);
      if (subRow == null) return null;
      const subsampleIdx = s.points.userData.subsampleIdx;
      const fullIndex = subsampleIdx ? subsampleIdx[subRow] : subRow;
      // Recover the world position of the chosen sub-row from the geometry.
      const pos = s.points.geometry.attributes.position;
      const world = new THREE.Vector3(
        pos.getX(subRow), pos.getY(subRow), pos.getZ(subRow),
      );
      return { fullIndex, world };
    },
    onPointerPick(cb) {
      const s = stateRef.current;
      if (!s) return () => {};
      const entry = { cb };
      s.pickSubs.push(entry);
      return () => {
        const i = s.pickSubs.indexOf(entry);
        if (i !== -1) s.pickSubs.splice(i, 1);
      };
    },
    onPointerMove(cb) {
      const s = stateRef.current;
      if (!s) return () => {};
      const entry = { cb };
      s.moveSubs.push(entry);
      return () => {
        const i = s.moveSubs.indexOf(entry);
        if (i !== -1) s.moveSubs.splice(i, 1);
      };
    },
    onHullPick(cb) {
      const s = stateRef.current;
      if (!s) return () => {};
      const entry = { cb };
      s.hullPickSubs.push(entry);
      return () => {
        const i = s.hullPickSubs.indexOf(entry);
        if (i !== -1) s.hullPickSubs.splice(i, 1);
      };
    },
    attachBrushGizmo({ radius, color }) {
      const s = stateRef.current;
      if (!s.scene) return { remove: () => {}, mesh: null };
      const geom = new THREE.SphereGeometry(1, 16, 12);
      const mat = new THREE.MeshBasicMaterial({
        color: color || '#ffffff',
        transparent: true,
        opacity: 0.18,
        depthWrite: false,
        side: THREE.FrontSide,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.scale.setScalar(radius);
      mesh.visible = false;
      s.scene.add(mesh);
      return {
        mesh,
        remove() {
          s.scene.remove(mesh);
          geom.dispose();
          mat.dispose();
        },
      };
    },
    setBrushPosition(worldVec, mesh) {
      if (!mesh) return;
      if (worldVec == null) {
        mesh.visible = false;
      } else {
        mesh.position.copy(worldVec);
        mesh.visible = true;
      }
    },
    // Draw sub-mode hooks. setOrbitEnabled is the explicit seam the spec
    // calls out: pointer-drag of a control point and wheel-resize of a tube
    // must win over camera orbit/zoom, and both controllers already expose
    // an enabled flag for exactly this (the gizmo uses it too).
    setOrbitEnabled(on) {
      stateRef.current.controller?.setEnabled?.(!!on);
    },
    getCamera() {
      return stateRef.current.camera ?? null;
    },
    attachOverlayGroup() {
      const s = stateRef.current;
      if (!s.scene) return { group: null, remove: () => {} };
      const group = new THREE.Group();
      s.scene.add(group);
      return {
        group,
        remove() {
          s.scene.remove(group);
          group.traverse((n) => {
            n.geometry?.dispose?.();
            n.material?.dispose?.();
          });
        },
      };
    },
    /**
     * Highlight the subsampled points whose subRow has mask[subRow] !== 0.
     * Caller computes the mask from segState.selection + instanceFull
     * + cloud.subsampleIdx — the viewer just blits matching positions
     * into the yellow overlay buffer.
     */
    setSelectedSegmentMask(mask, color = 0xfacc15) {
      const s = stateRef.current;
      if (!s.segSelectionGeom || !s.pointsGeom) return;
      s.segSelectionPoints.material.color.setHex(color);
      const posAttr = s.pointsGeom.getAttribute('position');
      const outAttr = s.segSelectionGeom.getAttribute('position');
      if (!posAttr || !outAttr) return;
      const pos = posAttr.array;
      const out = outAttr.array;
      const subN = pos.length / 3;
      let m = 0;
      if (mask && mask.length > 0) {
        for (let p = 0; p < subN; p++) {
          if (!mask[p]) continue;
          const b = p * 3;
          out[m++] = pos[b];
          out[m++] = pos[b + 1];
          out[m++] = pos[b + 2];
        }
      }
      s.segSelectionGeom.setDrawRange(0, m / 3);
      outAttr.needsUpdate = true;
      s.segSelectionPoints.visible = m > 0;
    },
    recolorByEdit({ affectedFullIndices, classFull, instanceFull, colorMode, palette,
                    dimInstances = null }) {
      const s = stateRef.current;
      if (!s.pointsGeom) return;
      const colorAttr = s.pointsGeom.getAttribute('color');
      if (!colorAttr) return;

      // dimInstances: any Set-like. Points whose instance is in this set
      // paint as a dim grey instead of their class/instance colour. Used
      // for "hide confirmed" mode in the Presegment list.
      const dimSet = dimInstances && typeof dimInstances.has === 'function' ? dimInstances : null;
      const dim = [0.22, 0.23, 0.26];
      const subsampleIdx = s.points.userData.subsampleIdx;
      if (!subsampleIdx) {
        // No subsampling: sub row == full idx. Recolor directly.
        const paletteRgb = (colorMode === 'class' && palette) ? buildPaletteRGB(palette) : null;
        const grey = [0.42, 0.44, 0.48];
        const tmp = new THREE.Color();
        for (const fullIdx of affectedFullIndices) {
          const base = fullIdx * 3;
          if (dimSet && instanceFull && dimSet.has(instanceFull[fullIdx])) {
            colorAttr.array[base]     = dim[0];
            colorAttr.array[base + 1] = dim[1];
            colorAttr.array[base + 2] = dim[2];
          } else if (colorMode === 'class' && paletteRgb && classFull) {
            const cid = classFull[fullIdx];
            const rgb = cid >= 0 ? (paletteRgb[cid] || grey) : grey;
            colorAttr.array[base]     = rgb[0];
            colorAttr.array[base + 1] = rgb[1];
            colorAttr.array[base + 2] = rgb[2];
          } else if (colorMode === 'instance' && instanceFull) {
            const iid = instanceFull[fullIdx];
            if (iid < 0) {
              colorAttr.array[base]     = 0.42;
              colorAttr.array[base + 1] = 0.44;
              colorAttr.array[base + 2] = 0.48;
            } else {
              const hue = ((iid * 0.6180339887) % 1 + 1) % 1;
              tmp.setHSL(hue, 0.62, 0.58);
              colorAttr.array[base]     = tmp.r;
              colorAttr.array[base + 1] = tmp.g;
              colorAttr.array[base + 2] = tmp.b;
            }
          }
        }
        colorAttr.needsUpdate = true;
        return;
      }

      // Build inverse map on first call (cached on stateRef).
      if (!s._fullToSubMap || s._fullToSubMapFor !== subsampleIdx) {
        s._fullToSubMap = buildFullToSubMap(subsampleIdx, classFull?.length ?? instanceFull?.length ?? 0);
        s._fullToSubMapFor = subsampleIdx;
      }
      const fullToSub = s._fullToSubMap;

      const paletteRgb = (colorMode === 'class' && palette) ? buildPaletteRGB(palette) : null;
      const grey = [0.42, 0.44, 0.48];
      const tmp = new THREE.Color();

      for (const fullIdx of affectedFullIndices) {
        const subRow = fullToSub[fullIdx];
        if (subRow === -1) continue;
        const base = subRow * 3;
        if (dimSet && instanceFull && dimSet.has(instanceFull[fullIdx])) {
          colorAttr.array[base]     = dim[0];
          colorAttr.array[base + 1] = dim[1];
          colorAttr.array[base + 2] = dim[2];
        } else if (colorMode === 'class' && paletteRgb && classFull) {
          const cid = classFull[fullIdx];
          const rgb = cid >= 0 ? (paletteRgb[cid] || grey) : grey;
          colorAttr.array[base]     = rgb[0];
          colorAttr.array[base + 1] = rgb[1];
          colorAttr.array[base + 2] = rgb[2];
        } else if (colorMode === 'instance' && instanceFull) {
          const iid = instanceFull[fullIdx];
          if (iid < 0) {
            colorAttr.array[base]     = 0.42;
            colorAttr.array[base + 1] = 0.44;
            colorAttr.array[base + 2] = 0.48;
          } else {
            const hue = ((iid * 0.6180339887) % 1 + 1) % 1;
            tmp.setHSL(hue, 0.62, 0.58);
            colorAttr.array[base]     = tmp.r;
            colorAttr.array[base + 1] = tmp.g;
            colorAttr.array[base + 2] = tmp.b;
          }
        }
      }
      colorAttr.needsUpdate = true;
    },
  }));

  return <div ref={mountRef} style={{ width: '100%', height: '100%', position: 'relative' }} />;
});
