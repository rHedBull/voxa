// Single source of truth for the Label-mode tool rail. A tool is only a way to
// select points; downstream behavior is shared (see the unified-label-tools spec).
export const TOOLS = [
  { id: 'presegment', icon: '◱', label: 'Presegment' },
  { id: 'box',        icon: '▭', label: 'Box' },
  { id: 'prism',      icon: '⬠', label: 'Prism' },
  { id: 'draw',       icon: '✎', label: 'Draw' },
  { id: 'beam',       icon: '⌶', label: 'Beam' },
  { id: 'sam',        icon: '✦', label: 'SAM' },
  { id: 'region',     icon: '▦', label: 'Region' },
];

export function toolAvailable(id, { segState, isAnnotated, rawSourceAvailable }) {
  // SAM needs the raw-resolution source (it segments off a rendered capture
  // and projects masks back onto the full-density cloud).
  if (id === 'sam') return !!segState && !!isAnnotated && !!rawSourceAvailable;
  // Draw persists centerlines.json, Beam persists structure.json, Region
  // persists eval_regions.json — all three need a session/scan dir, which
  // only annotated-tier scans have.
  if (id === 'draw' || id === 'beam' || id === 'region') return !!segState && !!isAnnotated;
  // Every tool applies through the segment session (apply-shape / reassign),
  // which only exists on annotated-tier scans — so all seven need segState.
  // Box works without presegments, but still requires a session.
  return !!segState;
}

export function defaultTool(ctx) {
  return toolAvailable('presegment', ctx) ? 'presegment' : 'box';
}
