// region-utils.js — pure helpers for the eval-regions feature (phase 1).
// Membership rule: an instance is "part of" a region iff >50% of its points
// are inside (settled 2026-07-21; boundary-crossers below the bar simply
// don't list). Confirmed-status lives frontend-side, so the majority filter
// runs here over the backend's raw {inside,total} counts.

export const REGION_COLORS = { draft: 0xf59e0b, eval_grade: 0x22c55e }; // amber-500 / green-500

export function regionCssColor(status) {
  return status === 'eval_grade' ? '#22c55e' : '#f59e0b';
}

export function majorityInstances(statRegion, instances) {
  if (!statRegion?.instances) return [];
  const out = [];
  for (const inst of instances) {
    if (!inst.confirmed || !Number.isFinite(inst.segId)) continue;
    const s = statRegion.instances[inst.segId];
    if (!s || !s.total) continue;
    const frac = s.inside / s.total;
    if (frac > 0.5) out.push({ inst, frac });
  }
  return out.sort((a, b) => b.frac - a.frac);
}

export function unlabeledPct(statRegion) {
  if (!statRegion || !statRegion.n_points) return null;
  return (100 * statRegion.n_unlabeled) / statRegion.n_points;
}
