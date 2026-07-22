"""Point categories — the annotation-status axis (eval-labeling phase 2).

The non-object taxonomy from the upstream spec: a point is either a real
object point (confirmed thing/stuff — DERIVED from class+instance, never
stored) or one of three non-object categories. Class and status are never
conflated on the class axis again (that was `double` id 4, then `unknown`
id 6). See docs/superpowers/specs/2026-07-22-point-categories-components-design.md.

Mirrored by frontend/src/point-categories.js — the two are pinned in sync by
tests on both sides.
"""
from __future__ import annotations

CATEGORY_NONE = 0             # unlabeled, or a confirmed thing/stuff
CATEGORY_ARTIFACT = 1         # no real surface (ghost/multipath/mixed pixel)
CATEGORY_TRANSIENT = 2        # person or self-mobile object
CATEGORY_EXCLUDED_REVIEW = 3  # real, permanent, identity uncommittable

CATEGORY_NAMES: dict[int, str] = {
    CATEGORY_NONE: "none",
    CATEGORY_ARTIFACT: "artifact",
    CATEGORY_TRANSIENT: "transient",
    CATEGORY_EXCLUDED_REVIEW: "excluded_review",
}

CATEGORY_VALUES: dict[str, int] = {name: value for value, name in CATEGORY_NAMES.items()}


def parse_category(value) -> int:
    """Accept a name ('artifact') or an int/int-like value and return the
    canonical int. Raises ValueError on anything else — the wire format is
    small and closed, so an unknown category is a bug, not a default."""
    # bool is an int subclass, and str(None) == 'None' would otherwise match
    # the 'none' category — both are caller bugs, not the default category.
    if value is None or isinstance(value, bool):
        raise ValueError(f"unknown category: {value!r}")
    if isinstance(value, int):
        if value in CATEGORY_NAMES:
            return int(value)
        raise ValueError(f"unknown category: {value!r}")
    key = str(value).strip().lower()
    if key in CATEGORY_VALUES:
        return CATEGORY_VALUES[key]
    raise ValueError(f"unknown category: {value!r}")


def category_histogram(categories) -> dict[str, int]:
    """{name: n_points} over every category, zeros included — a stable shape
    for gt_segment_metadata.json and the review-budget check."""
    import numpy as np

    arr = np.asarray(categories)
    counts = np.bincount(arr.astype(np.int64, copy=False),
                         minlength=len(CATEGORY_NAMES))
    return {name: int(counts[value]) for value, name in CATEGORY_NAMES.items()}
