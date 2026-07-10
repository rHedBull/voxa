// Single source of truth for the Label-mode tool rail. A tool is only a way to
// select points; downstream behavior is shared (see the unified-label-tools spec).
export const TOOLS = [
  { id: 'presegment', icon: '◱', label: 'Presegment' },
  { id: 'box',        icon: '▭', label: 'Box' },
  { id: 'draw',       icon: '✎', label: 'Draw' },
  // { id: 'beam', ... } reserved — not built yet.
];

export function toolAvailable(id, { segState, isAnnotated }) {
  if (id === 'presegment') return !!segState;
  if (id === 'draw') return !!segState && !!isAnnotated;
  return true; // box always available (works on raw clouds)
}

export function defaultTool(ctx) {
  return toolAvailable('presegment', ctx) ? 'presegment' : 'box';
}
