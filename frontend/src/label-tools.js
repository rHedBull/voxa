// Single source of truth for the Label-mode tool rail. A tool is only a way to
// select points; downstream behavior is shared (see the unified-label-tools spec).
export const TOOLS = [
  { id: 'presegment', icon: '◱', label: 'Presegment' },
  { id: 'box',        icon: '▭', label: 'Box' },
  { id: 'draw',       icon: '✎', label: 'Draw' },
  // { id: 'beam', ... } reserved — not built yet.
];

export function toolAvailable(id, { segState, isAnnotated }) {
  if (id === 'draw') return !!segState && !!isAnnotated;
  // Every tool applies through the segment session (apply-shape / reassign),
  // which only exists on annotated-tier scans — so all three need segState.
  // Box works without presegments, but still requires a session.
  return !!segState;
}

export function defaultTool(ctx) {
  return toolAvailable('presegment', ctx) ? 'presegment' : 'box';
}
