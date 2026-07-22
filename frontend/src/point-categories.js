// Point categories — the annotation-status axis (eval-labeling phase 2).
// Mirrors backend/labeling/categories.py (values pinned in sync by tests on
// both sides). Status never rides the class axis: a marked point carries a
// category here and class/instance -1 (a review blob keeps an instance id).

export const CATEGORY_NONE = 0;
export const CATEGORY_ARTIFACT = 1;
export const CATEGORY_TRANSIENT = 2;
export const CATEGORY_EXCLUDED_REVIEW = 3;

// Wire names (what the backend's parse_category accepts), UI labels, the
// picker hotkey, and the viewport color for marked points.
export const POINT_CATEGORIES = [
  { value: CATEGORY_ARTIFACT, name: 'artifact', label: 'Artifact', key: 'a',
    color: '#ff4dd2', hint: 'no real surface — ghost / multipath / edge smear' },
  { value: CATEGORY_TRANSIENT, name: 'transient', label: 'Transient', key: 't',
    color: '#ff9f1c', hint: 'person or self-mobile object' },
  { value: CATEGORY_EXCLUDED_REVIEW, name: 'excluded_review', label: 'Review', key: 'r',
    color: '#9aa0a6', hint: 'real, but identity uncommittable — budgeted to 3%' },
  { value: CATEGORY_NONE, name: 'none', label: 'Clear', key: 'x',
    color: null, hint: 'unmark: back to unlabeled' },
];

export const categoryByKey = (key) =>
  POINT_CATEGORIES.find((c) => c.key === String(key).toLowerCase()) || null;

export const categoryByValue = (value) =>
  POINT_CATEGORIES.find((c) => c.value === value) || null;

// Review blobs are the one category that owns instances, so the Instances
// panel needs a color/label for a class-less row.
export const REVIEW_COLOR = '#9aa0a6';
export const REVIEW_LABEL = 'Review';

// Normalized [r,g,b] per category value (index 0 = `none`, unused).
const CATEGORY_RGB = (() => {
  const hexToRGB = (hex) => [
    parseInt(hex.slice(1, 3), 16) / 255,
    parseInt(hex.slice(3, 5), 16) / 255,
    parseInt(hex.slice(5, 7), 16) / 255,
  ];
  const out = [null, null, null, null];
  for (const c of POINT_CATEGORIES) if (c.color) out[c.value] = hexToRGB(c.color);
  return out;
})();

// Viewport overlay for category-marked points: they carry class -1, so the
// base cloud draws them as plain unlabeled grey and the mark would be
// invisible. Returns {mask, colors} sized for the SUBSAMPLED cloud (subIdx
// maps rendered row -> full-res index; null when rendering full-res), or null
// when nothing is marked — callers then keep their untouched fast path.
export function buildCategoryOverlay(categories, subIdx, subN) {
  if (!categories) return null;
  const mask = new Uint8Array(subN);
  const colors = new Float32Array(subN * 3);
  let any = false;
  for (let p = 0; p < subN; p++) {
    const rgb = CATEGORY_RGB[categories[subIdx ? subIdx[p] : p]];
    if (!rgb) continue;
    mask[p] = 1;
    any = true;
    const o = p * 3;
    colors[o] = rgb[0]; colors[o + 1] = rgb[1]; colors[o + 2] = rgb[2];
  }
  return any ? { mask, colors } : null;
}

// { artifact, transient, excluded_review } counts over a category array.
// `none` is omitted deliberately — it is "everything else", not a mark.
export function categoryCounts(categories) {
  const out = { artifact: 0, transient: 0, excluded_review: 0 };
  if (!categories) return out;
  for (let i = 0; i < categories.length; i++) {
    const v = categories[i];
    if (v === CATEGORY_ARTIFACT) out.artifact++;
    else if (v === CATEGORY_TRANSIENT) out.transient++;
    else if (v === CATEGORY_EXCLUDED_REVIEW) out.excluded_review++;
  }
  return out;
}
