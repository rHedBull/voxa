// Frontend mirror of backend labeling/shapes.py::prism_indices — keeps the
// live in-viewport preview identical to the applied label. Parity is locked by
// prism-geom.test.js sharing test_shapes.py's fixture. XZ world plane, Y-up.

// Even-odd ray-cast: is (x,z) inside polygon [[x,z],...] (>=3 verts)?
export function pointInPolygonXZ(x, z, polygon) {
  let inside = false;
  const n = polygon.length;
  for (let i = 0, j = n - 1; i < n; j = i++) {
    const xi = polygon[i][0], zi = polygon[i][1];
    const xj = polygon[j][0], zj = polygon[j][1];
    const straddles = (zi > z) !== (zj > z);
    if (straddles && x < ((xj - xi) * (z - zi)) / (zj - zi) + xi) inside = !inside;
  }
  return inside;
}

// Indices into `positions` (Float32 xyz triples) inside the prism. `pool` is an
// optional index array to restrict the scan (null = all points).
export function pointsInsidePrism(positions, pool, prism) {
  const { polygon, y0, height } = prism;
  if (!polygon || polygon.length < 3 || !(height > 0)) return [];
  const yTop = y0 + height;
  const out = [];
  const N = pool ? pool.length : positions.length / 3;
  for (let k = 0; k < N; k++) {
    const i = pool ? pool[k] : k;
    const y = positions[3 * i + 1];
    if (y < y0 || y > yTop) continue;
    if (pointInPolygonXZ(positions[3 * i], positions[3 * i + 2], polygon)) out.push(i);
  }
  return out;
}
