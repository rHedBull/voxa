/**
 * Pure-function helpers for the export wizard. No React, no DOM — plain JS only.
 * These mirror the backend build_taxonomy semantics so the UI preview matches
 * what the server will produce.
 */

/**
 * Builds the export taxonomy preview.
 * A class = { class_id: number, label: string, color: string }.
 * A remap row = { from: number[], to: { id: number, label: string, color: string } }.
 *
 * Returns { [targetId]: { label, color } }.
 * Palette-driven (every kept class appears, even those with no points).
 * Excluded classes (not in includeSet) are omitted.
 * includeSet = null means "all classes included".
 *
 * A class consumed by a remap `from` set is not added under its own id
 * (it's merged into the target).
 */
export function remapToTaxonomy(classes, remapRows, includeSet) {
  // Build set of consumed class IDs (classes being merged away).
  const consumedIds = new Set();
  remapRows.forEach(row => {
    row.from.forEach(id => consumedIds.add(id));
  });

  // Start with palette classes that are included and not consumed.
  const taxonomy = {};
  classes.forEach(cls => {
    if (consumedIds.has(cls.class_id)) {
      // This class is being merged into a remap target, skip it.
      return;
    }
    if (includeSet !== null && !includeSet.has(cls.class_id)) {
      // This class is excluded, skip it.
      return;
    }
    taxonomy[cls.class_id] = { label: cls.label, color: cls.color };
  });

  // Add remap targets.
  remapRows.forEach(row => {
    taxonomy[row.to.id] = { label: row.to.label, color: row.to.color };
  });

  return taxonomy;
}

/**
 * Estimate point count for the Resolution step.
 * resolution = { kind: 'scan'|'subsample'|'raw', n?: number }
 *
 * Returns:
 * - 'scan' -> scanCount
 * - 'subsample' -> resolution.n
 * - 'raw' -> rawCount
 */
export function estimatePoints(resolution, scanCount, rawCount) {
  switch (resolution.kind) {
    case 'scan':
      return scanCount;
    case 'subsample':
      return resolution.n;
    case 'raw':
      return rawCount;
    default:
      return 0;
  }
}

/**
 * Estimate labeled points remaining after filters.
 * Used ONLY to disable Export when ~0.
 *
 * perClassCounts = { [class_id]: count } at scan density.
 * Scale by (targetPoints / scanCount).
 * Returns an integer estimate.
 *
 * includeSet = null means all classes are included.
 */
export function pointsAfterFilters(perClassCounts, includeSet, targetPoints, scanCount) {
  // Sum counts of included classes at scan density.
  let sumAtScanDensity = 0;
  Object.entries(perClassCounts).forEach(([classIdStr, count]) => {
    const classId = Number(classIdStr);
    if (includeSet === null || includeSet.has(classId)) {
      sumAtScanDensity += count;
    }
  });

  // Scale to target density.
  if (scanCount === 0) {
    return 0;
  }
  return Math.round(sumAtScanDensity * (targetPoints / scanCount));
}
