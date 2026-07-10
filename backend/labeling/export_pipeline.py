"""Validation + filter/remap pipeline for the export wizard (Phase B, spec §2).

Pure functions: no `_state`, no I/O. The endpoint (later task) calls these.
`apply_filters_remap` is chunk-safe (stateless, one array/chunk at a time) so
it can run per-array in regime A or per-chunk in regime B (raw); aggregate
work (`build_taxonomy`, `count_absent_instances`) is computed exactly once,
globally, by the caller.
"""
from __future__ import annotations

import numpy as np

from app.schemas import ExportLabelsRequest, RemapTarget


def validate_export_request(
    req: ExportLabelsRequest,
    n_scan: int,
    palette_ids: set[int],
    raw_available: bool,
) -> list[str]:
    errors: list[str] = []

    # from ids must exist in the source palette; from sets must not overlap.
    seen_from: dict[int, int] = {}   # source class id -> rule index
    to_by_id: dict[int, RemapTarget] = {}   # to.id -> RemapTarget of first rule seen
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

        to_id = rule.to.id
        if to_id in to_by_id:
            prev = to_by_id[to_id]
            if prev.label != rule.to.label or prev.color != rule.to.color:
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


def build_taxonomy(
    palette: list[dict],
    req: ExportLabelsRequest,
) -> tuple[dict[int, dict], dict[int, int]]:
    """Compute the exported taxonomy and the source->target class id map.

    PALETTE-DRIVEN: every kept class appears in `taxonomy` even if no point
    in the data currently carries it (never derived from np.unique).

    A class excluded by `include_classes` is left OUT of `taxonomy` (no
    exported point can carry it), but `src_to_tgt` still maps it to itself
    for simplicity — the include/exclude filter zeros those points in
    `apply_filters_remap` before `src_to_tgt` is consulted, so the excluded
    entry is never actually used.
    """
    included = (
        {c["class_id"] for c in palette}
        if req.include_classes is None
        else set(req.include_classes)
    )

    # Start with every palette class mapped to itself, then overlay remap
    # rules (from -> to.id).
    src_to_tgt: dict[int, int] = {c["class_id"]: c["class_id"] for c in palette}
    consumed: set[int] = set()
    taxonomy: dict[int, dict] = {}

    for rule in req.remap:
        for src_id in rule.from_:
            src_to_tgt[src_id] = rule.to.id
            consumed.add(src_id)
        taxonomy[rule.to.id] = {"label": rule.to.label, "color": rule.to.color}

    for c in palette:
        cid = c["class_id"]
        if cid in consumed:
            continue
        if cid not in included:
            continue
        taxonomy[cid] = {"label": c["label"], "color": c["color"]}

    return taxonomy, src_to_tgt


def apply_filters_remap(
    class_ids: np.ndarray,
    instance_ids: np.ndarray,
    confirmed_by_inst: dict[int, bool],
    req: ExportLabelsRequest,
    src_to_tgt: dict[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Apply confirmed-only -> include/exclude -> remap, in that order, to
    one array (or chunk). Stateless: no aggregate counts, no dropping of
    points (the caller applies `drop_unlabeled`). Instance ids are returned
    unchanged.
    """
    out_cls = class_ids.copy()

    if req.confirmed_only:
        # Instances absent from confirmed_by_inst are treated as confirmed.
        unconfirmed_mask = np.array(
            [confirmed_by_inst.get(int(iid), True) is False for iid in instance_ids],
            dtype=bool,
        )
        out_cls[unconfirmed_mask] = -1

    if req.include_classes is not None:
        include_set = set(req.include_classes)
        exclude_mask = np.array([int(c) not in include_set for c in out_cls], dtype=bool)
        out_cls[exclude_mask] = -1

    remap_mask = out_cls >= 0
    if remap_mask.any():
        out_cls[remap_mask] = np.array(
            [src_to_tgt.get(int(c), int(c)) for c in out_cls[remap_mask]],
            dtype=out_cls.dtype,
        )

    return out_cls, instance_ids.copy()


def count_absent_instances(
    work_inst: np.ndarray,
    confirmed_by_inst: dict[int, bool],
) -> int:
    """Count distinct instance ids in `work_inst` (excluding -1) that are
    not keys of `confirmed_by_inst`. Computed once from the full session
    array — never per chunk, or it would double-count.
    """
    distinct = set(int(iid) for iid in np.unique(work_inst) if iid != -1)
    return sum(1 for iid in distinct if iid not in confirmed_by_inst)
