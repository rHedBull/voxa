// cut-eligibility.js — pure rules for whether the right-click "Edit
// selection…" menu item is enabled on a list row. Each of the three list
// surfaces (Presegment, SAM, Instances) only ever considers its OWN
// selection — the two selection Sets (segState.selection, segState.
// samSelection) and the Instances panel's single selectedId scalar are
// never combined across lists (the lists are never visible together and
// their selections aren't kept symmetric across tool switches).
//
// list: 'preseg' | 'sam' — eligible iff selectionSize > 0.
// list: 'instance'       — eligible iff isSelected && !confirmed &&
//   !classFrozen (confirmed instances are locked — see "Confirmed = locked"
//   in CLAUDE.md; a frozen legacy class can't be newly assigned, and an
//   instance-cut inherits the source's class, so cutting a legacy-class
//   instance would 422 server-side — re-label it with a primitive first) —
//   and !reviewBlob (a review blob has no class to inherit; resolve it with
//   Relabel… first, eval-labeling phase 2).
export function cutEligibility(params) {
  const { list } = params;
  if (list === 'preseg' || list === 'sam') {
    const { selectionSize } = params;
    if (selectionSize > 0) return { eligible: true };
    return { eligible: false, reason: 'empty' };
  }
  if (list === 'instance') {
    const { isSelected, confirmed, classFrozen, reviewBlob } = params;
    if (!isSelected) return { eligible: false, reason: 'not-selected' };
    if (confirmed) return { eligible: false, reason: 'confirmed' };
    if (classFrozen) return { eligible: false, reason: 'frozen-class' };
    if (reviewBlob) return { eligible: false, reason: 'review-blob' };
    return { eligible: true };
  }
  throw new Error(`cutEligibility: unknown list "${list}"`);
}
