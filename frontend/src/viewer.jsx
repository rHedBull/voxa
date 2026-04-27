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

function attachOrbit(camera, dom, target, onChange) {
  const state = {
    dragging: false, mode: null, lx: 0, ly: 0,
    spherical: { r: 0, phi: 0, theta: 0 },
    target: target.clone(),
    silent: false,   // when true, apply() skips the onChange callback
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
  apply();

  const onDown = (e) => {
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
      // ping-pong each other into a stack overflow.
      state.silent = true;
      state.spherical = { ...s.spherical };
      state.target.copy(s.target);
      apply();
      state.silent = false;
    },
    getState() { return { spherical: { ...state.spherical }, target: state.target.clone() }; },
    frame(center, radius) {
      state.target.copy(center);
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
  apply();

  // Drag with any mouse button (left, middle, right) rotates look. Right
  // button needs the contextmenu listener to keep the OS menu from popping.
  const onDown = (e) => {
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
      state.silent = true;
      camera.position.copy(s.position);
      state.yaw = s.yaw; state.pitch = s.pitch;
      apply();
      state.silent = false;
    },
    getState() {
      return { position: camera.position.clone(), yaw: state.yaw, pitch: state.pitch };
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
    onCameraChange = null,
  } = props;

  const mountRef = useRef(null);
  const stateRef = useRef({});
  // Keep onCameraChange callable from inside the long-lived controller
  // closure without recreating the controller every render.
  const onCameraChangeRef = useRef(onCameraChange);
  onCameraChangeRef.current = onCameraChange;

  // ── Mount once ─────────────────────────────────────────────────────────
  useEffect(() => {
    const mount = mountRef.current;
    const w = mount.clientWidth, h = mount.clientHeight;
    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: false });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h);
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

    stateRef.current = {
      renderer, scene, camera, pointsGeom, pointsMat, points,
      controller: null,
      cuboidGroup, floor, grid, axes, mount,
    };

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      stateRef.current.controller?.dispose();
      pointsGeom.dispose();
      pointsMat.dispose();
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
    s.scene.background = new THREE.Color(background);
    s.floor.material.color = new THREE.Color(floorColor);
    s.floor.visible = showFloor;
    s.grid.visible = showFloor;
    s.axes.visible = showAxes;
  }, [pointSize, background, floorColor, showFloor, showAxes]);

  // ── Per-point color recompute ───────────────────────────────────────────
  useEffect(() => {
    const s = stateRef.current;
    if (!s.pointsGeom) return;
    const colorAttr = s.pointsGeom.getAttribute('color');
    const posAttr = s.pointsGeom.getAttribute('position');
    if (!colorAttr || !cloud) return;
    const orig = cloud.colors;

    if (colorMode === 'height' && cloud.bbox) {
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
    colorAttr.needsUpdate = true;
  }, [colorMode, cloud]);

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

  useImperativeHandle(ref, () => ({
    preset(name) {
      const s = stateRef.current;
      s.controller?.preset?.(name, s._lastCenter, s._lastRadius);
    },
    frame(center, radius) { stateRef.current.controller?.frame?.(center, radius); },
    setCameraState(s) { stateRef.current.controller?.setFromState?.(s); },
    getCameraState() { return stateRef.current.controller?.getState?.(); },
  }));

  return <div ref={mountRef} style={{ width: '100%', height: '100%', position: 'relative' }} />;
});
