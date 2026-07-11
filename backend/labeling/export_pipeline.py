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
        if not (0 <= to_id <= 65535):
            # Exported class_id is int32, but a negative id collides with the
            # unlabeled sentinel and a huge one would size the remap LUT.
            errors.append(f"remap rule {rule_idx}: remap target id {to_id} out of range [0, 65535]")
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

    if req.resolution.kind not in ("scan", "subsample", "raw"):
        errors.append(f"unknown resolution kind: {req.resolution.kind!r}")

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
    out_cls = class_ids.copy()  # mutated below

    if req.confirmed_only:
        # Instances absent from confirmed_by_inst are treated as confirmed:
        # they're not in `unconfirmed`, so np.isin leaves them untouched.
        unconfirmed = [iid for iid, ok in confirmed_by_inst.items() if not ok]
        if unconfirmed:
            out_cls[np.isin(instance_ids, unconfirmed)] = -1

    if req.include_classes is not None:
        # A -1 isn't in include_classes -> set to -1 again, harmless.
        out_cls[~np.isin(out_cls, list(req.include_classes))] = -1

    remap_mask = out_cls >= 0
    if src_to_tgt and remap_mask.any():
        # src_to_tgt covers every palette class as identity, so all real class
        # ids are <= hi; the np.where guard is just defensive.
        hi = max(src_to_tgt)
        # LUT (and, when a target exceeds the input dtype, the output) in
        # int32: class arrays arrive int8, and a remap target >= 128 would
        # silently wrap negative in the input dtype. Exported class_id is
        # int32 regardless (_ply_labeled_chunk_bytes).
        lut = np.arange(hi + 1, dtype=np.int32)
        for s, t in src_to_tgt.items():
            lut[s] = t
        if lut.max() > np.iinfo(out_cls.dtype).max:
            out_cls = out_cls.astype(np.int32)
        vals = out_cls[remap_mask]
        out_cls[remap_mask] = np.where(vals <= hi, lut[np.minimum(vals, hi)], vals)

    # instance_ids is read-only here; return it directly (no full-array memcpy).
    return out_cls, instance_ids


def drop_unlabeled_rows(class_ids, *arrays):
    """Keep only points with class_id >= 0. Returns (class_ids, *arrays) all
    masked by the same boolean. Used by both export regimes."""
    keep = class_ids >= 0
    return (class_ids[keep], *(a[keep] for a in arrays))


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


def build_manifest(
    taxonomy: dict[int, dict],
    p50: float,
    p90: float,
    scan: str,
    session: str,
    resolution: dict,
    points: int,
    confirmed_only: bool,
    include_classes: list[int] | None,
    drop_unlabeled: bool,
    absent_count: int,
    exported_at: str,
    labeling_points: int | None = None,
) -> dict:
    """Build the export zip's manifest.json (pure function; exported_at is passed in).

    Returns:
        A dict with classes, accuracy, source, resolution, and filters.
        All values are JSON-serializable (no numpy types).
    """
    return {
        "classes": {
            str(tid): {"label": t["label"], "color": t["color"]}
            for tid, t in taxonomy.items()
        },
        "accuracy": {
            "labeling_points": int(labeling_points) if labeling_points is not None else None,
            "sample_spacing_p50_m": float(round(p50, 4)),
            "sample_spacing_p90_m": float(round(p90, 4)),
            "semantic_boundary_uncertainty_m": float(round(p90, 4)),
            "note": (
                "Semantic (preseg/legacy) boundaries are accurate to ~one "
                "labeling-cloud sample spacing (reported as p90 to reflect non-"
                "uniform LiDAR sampling) and are set by labeling density, NOT the "
                "export resolution. Box/pipe (volumetric) boundaries are exact at "
                "any density."
            ),
        },
        "source": {"scan": scan, "session": session, "exported_at": exported_at},
        "resolution": {"kind": resolution["kind"], "points": int(points)},
        "filters": {
            "confirmed_only": confirmed_only,
            "include_classes": include_classes,
            "drop_unlabeled": drop_unlabeled,
            "absent_instances": int(absent_count),
        },
    }
