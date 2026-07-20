// outlier-eligibility.js — pure rule for enabling the right-click "Remove
// outliers" menu item. Mirrors cut-eligibility.js, with one difference: the
// SAM case requires EXACTLY ONE selected candidate, because
// /api/segment/denoise-selection targets a single { source, id } — there is
// no multi-selection form. Presegs are out of scope (immutable layer), so
// only 'sam' and 'instance' lists are handled.
export function removeOutliersEligibility(params) {
  const { list } = params;
  if (list === 'sam') {
    const { selectionSize } = params;
    if (selectionSize === 1) return { eligible: true };
    return { eligible: false, reason: selectionSize > 1 ? 'multi' : 'empty' };
  }
  if (list === 'instance') {
    const { isSelected, confirmed } = params;
    if (!isSelected) return { eligible: false, reason: 'not-selected' };
    if (confirmed) return { eligible: false, reason: 'confirmed' };
    return { eligible: true };
  }
  throw new Error(`removeOutliersEligibility: unknown list "${list}"`);
}
