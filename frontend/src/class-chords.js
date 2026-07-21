// Two-stroke class chords (phase-0 spec §3). Pure module: group table +
// chord state transition. Consumers hold `pendingGroup` (null | group) and
// feed every keydown through chordStep.

export const CLASS_GROUPS = [
  { id: 'pipe-network', key: '1', label: 'Pipe network' },
  { id: 'duct',         key: '2', label: 'Duct' },
  { id: 'electrical',   key: '3', label: 'Electrical' },
  { id: 'plant-units',  key: '4', label: 'Plant units' },
  { id: 'attachments',  key: '5', label: 'Attachments' },
  { id: 'structure',    key: '6', label: 'Structure' },
  { id: 'other',        key: '7', label: 'Other' },
  { id: 'stuff',        key: '8', label: 'Stuff' },
  { id: 'legacy',       key: null, label: 'Legacy (frozen)' },
];

export const assignable = (classes) => classes.filter((c) => !c.frozen);

export const groupMembers = (groupId, classes) =>
  assignable(classes).filter((c) => c.group === groupId);

// chordStep(pendingGroup, key, classes) →
//   {type:'group', group} | {type:'class', cls} | {type:'cancel'} | {type:'pass'}
export function chordStep(pendingGroup, key, classes) {
  if (pendingGroup == null) {
    const group = CLASS_GROUPS.find((g) => g.key === key);
    return group ? { type: 'group', group } : { type: 'pass' };
  }
  if (key === 'Escape') return { type: 'cancel' };
  const cls = groupMembers(pendingGroup.id, classes)
    .find((c) => String(c.hotkey) === key);
  return cls ? { type: 'class', cls } : { type: 'cancel' };
}
