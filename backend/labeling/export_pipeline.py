"""Validation for the export wizard's ExportLabelsRequest (Phase B, spec §2).

Pure function: no `_state`, no I/O. The endpoint (later task) calls this and
raises 422 with the returned error list when it's non-empty.
"""
from __future__ import annotations

from app.schemas import ExportLabelsRequest


def validate_export_request(
    req: ExportLabelsRequest,
    n_scan: int,
    palette_ids: set[int],
    raw_available: bool,
) -> list[str]:
    errors: list[str] = []

    # from ids must exist in the source palette; from sets must not overlap.
    seen_from: dict[int, int] = {}   # source class id -> rule index
    to_by_id: dict[int, dict] = {}   # to.id -> {id,label,color} of first rule seen
    consumed: set[int] = set()       # source class ids consumed by some `from`

    for rule_idx, rule in enumerate(req.remap):
        for src_id in rule.from_:
            if src_id not in palette_ids:
                errors.append(f"remap rule {rule_idx}: 'from' id {src_id} is not a valid source class id")
            if src_id in seen_from:
                errors.append(
                    f"remap rules {seen_from[src_id]} and {rule_idx} both claim source class id {src_id} (overlapping 'from' sets)"
                )
            else:
                seen_from[src_id] = rule_idx
            consumed.add(src_id)

        to_id = rule.to.get("id")
        if to_id in to_by_id:
            prev = to_by_id[to_id]
            if prev.get("label") != rule.to.get("label") or prev.get("color") != rule.to.get("color"):
                errors.append(
                    f"multiple remap rules target class id {to_id} with different label/color"
                )
        else:
            to_by_id[to_id] = rule.to

    # kept-through classes: survive include_classes (or all classes if None)
    # and are not consumed by any 'from' set.
    included = palette_ids if req.include_classes is None else set(req.include_classes)
    kept_through = included - consumed

    for to_id in to_by_id:
        if to_id in kept_through:
            errors.append(
                f"remap target id {to_id} collides with a kept-through source class id"
            )

    if req.resolution.kind == "subsample":
        n = req.resolution.n
        if n is None or n < 1:
            errors.append(f"resolution.n must be >= 1 for kind=subsample, got {n}")
        elif n > n_scan:
            errors.append(f"resolution.n ({n}) exceeds n_scan ({n_scan}) for kind=subsample")

    if req.resolution.kind == "raw" and not raw_available:
        errors.append("resolution.kind=raw requested but no raw full-density source is available")

    return errors
