// cut-eligibility.js — pure rules for whether the right-click "Edit
// selection…" menu item is enabled on a list row. Each of the three list
// surfaces (Presegment, SAM, Instances) only ever considers its OWN
// selection — the two selection Sets (segState.selection, segState.
// samSelection) and the Instances panel's single selectedId scalar are
// never combined across lists (the lists are never visible together and
// their selections aren't kept symmetric across tool switches).
//
// list: 'preseg' | 'sam' — eligible iff selectionSize > 0.
// list: 'instance'       — eligible iff isSelected && !confirmed
//   (confirmed instances are locked — see "Confirmed = locked" in CLAUDE.md).
export function cutEligibility(params) {
  const { list } = params;
  if (list === 'preseg' || list === 'sam') {
    const { selectionSize } = params;
    if (selectionSize > 0) return { eligible: true };
    return { eligible: false, reason: 'empty' };
  }
  if (list === 'instance') {
    const { isSelected, confirmed } = params;
    if (!isSelected) return { eligible: false, reason: 'not-selected' };
    if (confirmed) return { eligible: false, reason: 'confirmed' };
    return { eligible: true };
  }
  throw new Error(`cutEligibility: unknown list "${list}"`);
}
