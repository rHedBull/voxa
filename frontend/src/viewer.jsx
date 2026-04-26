// viewer.jsx
// Three.js viewport. Renders a point cloud, supports orbit/pan/zoom, draws
// cuboid overlays for instances, supports per-instance highlight, and exposes
// camera-state ref handles used by Compare mode for sync.

import { useEffect, useRef, useImperativeHandle, forwardRef } from 'react';
import * as THREE from 'three';

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
    overrideColor = null,
    onCameraChange = null,
  } = props;

  const mountRef = useRef(null);
  const stateRef = useRef({});

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

    const target = new THREE.Vector3(0, 0.2, 0);
    const orbit = attachOrbit(camera, renderer.domElement, target, (s) => {
      onCameraChange && onCameraChange(s);
    });

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
      orbit, cuboidGroup, floor, grid, axes, mount,
    };

    return () => {
      cancelAnimationFrame(raf);
      ro.disconnect();
      orbit.dispose();
      pointsGeom.dispose();
      pointsMat.dispose();
      renderer.dispose();
      mount.removeChild(renderer.domElement);
    };
    // eslint-disable-next-line
  }, []);

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
      s.orbit.frame(c, radius);
      s._lastCenter = c;
      s._lastRadius = radius;
    }
  }, [cloud]);

  // ── Reactive updates ────────────────────────────────────────────────────
  useEffect(() => {
    const s = stateRef.current;
    if (!s.pointsMat) return;
    s.pointsMat.size = pointSize;
    s.scene.background = new THREE.Color(background);
    s.floor.material.color = new THREE.Color(floorColor);
    s.floor.visible = showFloor;
    s.grid.visible = showFloor;
    s.axes.visible = showAxes;

    const colorAttr = s.pointsGeom.getAttribute('color');
    if (!colorAttr || !cloud) return;
    if (overrideColor) {
      const c = new THREE.Color(overrideColor);
      const orig = cloud.colors;
      for (let i = 0; i < orig.length; i += 3) {
        const lum = orig[i] * 0.299 + orig[i + 1] * 0.587 + orig[i + 2] * 0.114;
        colorAttr.array[i + 0] = c.r * (0.5 + lum * 0.7);
        colorAttr.array[i + 1] = c.g * (0.5 + lum * 0.7);
        colorAttr.array[i + 2] = c.b * (0.5 + lum * 0.7);
      }
    } else {
      colorAttr.array.set(cloud.colors);
    }
    colorAttr.needsUpdate = true;
  }, [pointSize, background, floorColor, showFloor, showAxes, overrideColor, cloud]);

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
      s.orbit?.preset(name, s._lastCenter, s._lastRadius);
    },
    frame(center, radius) { stateRef.current.orbit?.frame(center, radius); },
    setCameraState(s) { stateRef.current.orbit?.setFromState(s); },
    getCameraState() { return stateRef.current.orbit?.getState(); },
  }));

  return <div ref={mountRef} style={{ width: '100%', height: '100%', position: 'relative' }} />;
});
