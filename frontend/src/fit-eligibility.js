// Pure rule for the right-click "Fit box to selection…" menu item. Mirrors
// cut-eligibility.js, EXCEPT a confirmed instance is still fit-eligible: fitting
// only READS the source points to size a new, independent Box volume — it never
// relabels the confirmed instance (see the confirmed-instance note in
// docs/superpowers/specs/2026-07-20-fit-box-to-selection-design.md).
export function fitEligibility(params) {
  const { list } = params;
  if (list === 'preseg' || list === 'sam') {
    return params.selectionSize > 0
      ? { eligible: true } : { eligible: false, reason: 'empty' };
  }
  if (list === 'instance') {
    return params.isSelected
      ? { eligible: true } : { eligible: false, reason: 'not-selected' };
  }
  throw new Error(`fitEligibility: unknown list "${list}"`);
}
