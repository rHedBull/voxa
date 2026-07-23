"""Density-agnostic label materialization: produce per-point (positions, colors,
class_ids, instance_ids) for a session at any target density. Regime A =
down-sample by index-transfer (exact); regime B (denser) replays labels onto
raw/upsampled points via the max-seq rule below."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from app.core import _subsample_indices  # seeded, ascending indices into range(N)
from labeling.shapes import obb_indices, prism_indices
from labeling.centerline import tube_indices


def materialize_downsample(positions, colors, class_ids, instance_ids, n):
    """Regime A: exact down-sample. n >= len -> identity; else index every array
    by the same seeded subsample so labels are transferred, never interpolated."""
    N = len(positions)
    if n >= N:
        return positions, colors, class_ids, instance_ids
    idx = _subsample_indices(N, n)
    return positions[idx], colors[idx], class_ids[idx], instance_ids[idx]


def collect_volumes(instances, centerlines):
    """Volumetric instances only (source in {'box','beam','draw','prism'}). Box/beam ->
    obb from center/size/rotation; draw -> tube from the instance's centerlines
    paths (grouped by instance_id == segId); prism -> prism from the instance's
    persisted polygon/y0/height. Each carries its apply-order `seq`."""
    paths_by_inst = {}
    for p in (centerlines or {}).get("paths", []):
        paths_by_inst.setdefault(int(p["instance_id"]), []).append(p)
    out = []
    for inst in instances:
        src = inst.get("source")
        if src not in ("box", "beam", "draw", "prism") or inst.get("segId") is None:
            continue
        seq = inst.get("seq")
        iid = int(inst["segId"])  # symmetric int key with paths_by_inst; the seq
        # map + KD-tree replay (Task 4) key off this same instance_id
        if src in ("box", "beam") and inst.get("center") and inst.get("size"):
            out.append({"kind": "obb", "instance_id": iid, "seq": seq,
                        "shape": {"center": inst["center"], "size": inst["size"],
                                  "rotation": inst.get("rotation", [0, 0, 0])}})
        elif src == "draw" and iid in paths_by_inst:
            out.append({"kind": "tube", "instance_id": iid, "seq": seq,
                        "paths": paths_by_inst[iid]})
        elif src == "prism" and inst.get("prism"):
            out.append({"kind": "prism", "instance_id": iid, "seq": seq,
                        "prism": inst["prism"]})
    return out


@dataclass
class ReplayIndex:
    """Precomputed structures for `replay_labels`, built once per session and
    reused across raw chunks (Task 5)."""
    work_inst: np.ndarray
    volumes: list
    vol_ids: set
    seq_by_inst: dict
    inst_class_id: dict
    tree_all: cKDTree
    tree_nonvol: "cKDTree | None"
    nonvol_idx: np.ndarray


def build_replay_index(scan_pos, work_inst, volumes, seq_by_inst, inst_class_id):
    """Build the KD-trees + volume-ownership mask needed by `replay_labels`.

    `inst_class_id` is passed in (not derived from working arrays) because a
    Box OBB can own zero scan.ply points yet still win raw points in regime B.
    """
    scan_pos = np.asarray(scan_pos, dtype=np.float32).reshape(-1, 3)
    work_inst = np.asarray(work_inst, dtype=np.int32)
    vol_ids = {v["instance_id"] for v in volumes}
    vol_owned = np.isin(work_inst, list(vol_ids)) if vol_ids else np.zeros(len(work_inst), dtype=bool)
    tree_all = cKDTree(scan_pos)
    nonvol_idx = np.where(~vol_owned)[0]
    tree_nonvol = cKDTree(scan_pos[nonvol_idx]) if nonvol_idx.size > 0 else None
    return ReplayIndex(
        work_inst=work_inst,
        volumes=volumes,
        vol_ids=vol_ids,
        seq_by_inst=seq_by_inst,
        inst_class_id=inst_class_id,
        tree_all=tree_all,
        tree_nonvol=tree_nonvol,
        nonvol_idx=nonvol_idx,
    )


def replay_labels(index: ReplayIndex, target_pos):
    """Regime B: replay labels onto `target_pos` (M,3) via the max-seq rule.

    Volumetric (box/tube) boundaries are exact and compete by apply-order
    `seq`; non-volumetric (preseg/legacy) labels transfer from the nearest
    scan.ply sample. A volume defends its own interior against lower-seq
    neighbors but loses to a higher-seq claimant that actually covers the
    point (regardless of source). Precedence everywhere is max-seq.
    """
    target_pos = np.asarray(target_pos, dtype=np.float32).reshape(-1, 3)
    n = len(target_pos)

    # One boolean mask per volume over the whole target cloud (vectorized),
    # keyed by instance_id for O(1) baseline lookup in the loop below.
    mask_by_inst = {}
    for v in index.volumes:
        mask = np.zeros(n, dtype=bool)
        if v["kind"] == "obb":
            idx = obb_indices(target_pos, v["shape"])
        elif v["kind"] == "tube":
            idx = tube_indices(target_pos, v["paths"])
        elif v["kind"] == "prism":
            idx = prism_indices(target_pos, v["prism"])
        else:
            raise ValueError(f"unknown volume kind: {v['kind']!r}")
        mask[idx] = True
        mask_by_inst[v["instance_id"]] = mask

    # Hoist every KD-tree query out of the per-point loop into one batched C
    # call each — regime B replays this over up to ~156M raw points per chunk.
    owner_all = index.work_inst[index.tree_all.query(target_pos)[1]]
    if index.tree_nonvol is not None:
        nonvol_owner_all = index.work_inst[
            index.nonvol_idx[index.tree_nonvol.query(target_pos)[1]]]
    else:
        nonvol_owner_all = None

    target_cls = np.full(n, -1, dtype=np.int8)
    target_inst = np.full(n, -1, dtype=np.int32)

    def seq_of(inst_id):
        # Missing seq -> -inf: pre-feature/imported labels with no recorded
        # apply-order lose every tie to an explicitly re-applied instance.
        s = index.seq_by_inst.get(inst_id)
        return s if s is not None else float("-inf")

    for p_idx in range(n):
        candidates = []  # (seq, instance_id)
        for iid, mask in mask_by_inst.items():
            if mask[p_idx]:
                candidates.append((seq_of(iid), iid))

        oi = int(owner_all[p_idx])
        if oi in index.vol_ids and not mask_by_inst[oi][p_idx]:
            # Leak: nearest sample belongs to a volume that doesn't actually
            # cover p -> fall back to the nearest non-volumetric sample.
            baseline_inst = int(nonvol_owner_all[p_idx]) if nonvol_owner_all is not None else -1
        else:
            baseline_inst = oi
        if baseline_inst != -1:
            candidates.append((seq_of(baseline_inst), baseline_inst))

        if not candidates:
            continue
        # inst_class_id must cover every winning instance_id (built by the
        # caller from the instance doc); a missing id is a caller bug -> KeyError.
        _, winner_inst = max(candidates, key=lambda c: c[0])
        target_inst[p_idx] = winner_inst
        try:
            target_cls[p_idx] = index.inst_class_id[winner_inst]
        except KeyError as e:
            raise KeyError(f"inst_class_id has no class for winning instance "
                           f"{winner_inst}; the caller must map every "
                           f"instance_id in work_inst/volumes") from e

    return target_cls, target_inst


def materialize_raw(index: ReplayIndex, raw_path, scene_is_z_up: bool, offset,
                     chunk: int = 1_000_000):
    """Regime B: stream the raw LAZ/LAS at native density, chunked, mapping
    each chunk into the display frame and replaying labels via `index` (built
    once by the caller so the scan.ply KD-trees are reused across chunks —
    never rebuilt here). Yields (display_xyz, rgb8, cls, inst) per chunk;
    never accumulates the whole cloud (raw clouds run up to 156M points).
    """
    from app.core import _to_display_frame
    from scenes.lidar_io import _laz_chunk_iter, _laz_rgb_to_uint8

    offset = np.asarray(offset)
    for _hdr, las_chunk in _laz_chunk_iter(raw_path, chunk_size=chunk):
        xyz_src = np.column_stack([
            np.asarray(las_chunk.x, dtype=np.float64),
            np.asarray(las_chunk.y, dtype=np.float64),
            np.asarray(las_chunk.z, dtype=np.float64),
        ])
        try:
            r = np.asarray(las_chunk.red, dtype=np.uint32)
            g = np.asarray(las_chunk.green, dtype=np.uint32)
            b = np.asarray(las_chunk.blue, dtype=np.uint32)
            rgb8 = _laz_rgb_to_uint8(np.column_stack([r, g, b]))
        except (AttributeError, ValueError):
            rgb8 = None

        display_xyz = _to_display_frame(xyz_src, scene_is_z_up, offset)
        cls, inst = replay_labels(index, display_xyz)
        yield (display_xyz, rgb8, cls, inst)


@dataclass
class MaterializeCtx:
    """Everything `materialize()` needs for one session, gathered by the
    caller (Phase B's export endpoint). Fields mirror the scan.ply working
    arrays (display frame) plus the regime-B replay inputs."""
    scan_pos: np.ndarray          # (N,3) float32, display frame
    colors: np.ndarray            # (N,3) uint8
    work_cls: np.ndarray          # (N,) int8
    work_inst: np.ndarray         # (N,) int32
    volumes: list                 # output of collect_volumes
    seq_by_inst: dict
    inst_class_id: dict
    raw_path: "str | None"
    scene_is_z_up: bool
    offset: np.ndarray


def materialize(ctx: MaterializeCtx, resolution: dict):
    """Top-level dispatcher: pick regime A (scan/subsample) or regime B (raw)
    and always attach the accuracy metric.

    `resolution` = {"kind": "scan"|"subsample"|"raw", "n": int|None}.

    NOTE: the "raw" branch here concatenates every `materialize_raw` chunk
    into single in-memory arrays. That is a convenience/small-cloud/TEST
    entry point ONLY. Phase B's real export endpoint must consume
    `materialize_raw(index, ...)`'s generator directly and stream each chunk
    to disk — it must NOT call `materialize(ctx, {"kind": "raw"})` against a
    real ~156M-point raw cloud, since that would hold the whole ~GB result
    in memory at once.
    """
    kind = resolution["kind"]

    if kind in ("scan", "subsample"):
        n = len(ctx.scan_pos) if kind == "scan" else resolution["n"]
        positions, colors, class_ids, instance_ids = materialize_downsample(
            ctx.scan_pos, ctx.colors, ctx.work_cls, ctx.work_inst, n)
        # materialize_downsample's identity branch (n >= N) returns the input
        # arrays BY REFERENCE. Phase B remaps class_ids in place after this
        # call, which would otherwise corrupt the session's live working
        # arrays. Copy defensively; cheap since scan.ply is <= a few million
        # points. positions/colors are read-only for the PLY, so left as-is.
        class_ids = class_ids.copy()
        instance_ids = instance_ids.copy()
        # NOTE for Phase B: only class_ids/instance_ids are copy-safe. In the
        # identity branch, positions/colors are still the live ctx arrays by
        # reference — a consumer must NOT mutate them in place.
    elif kind == "raw":
        if ctx.raw_path is None:
            # Defense-in-depth: Phase B gates this on `raw_source_available`,
            # but never trust the caller wired the gate (or that the file
            # survived between load and export). Fail loudly, not with a
            # confusing laspy "None" path error.
            raise ValueError("materialize: raw resolution requested but ctx.raw_path is None")
        index = build_replay_index(
            ctx.scan_pos, ctx.work_inst, ctx.volumes, ctx.seq_by_inst, ctx.inst_class_id)
        pos_chunks, rgb_chunks, cls_chunks, inst_chunks = [], [], [], []
        any_rgb_none = False
        for xyz, rgb8, cls, inst in materialize_raw(
                index, ctx.raw_path, ctx.scene_is_z_up, ctx.offset):
            pos_chunks.append(xyz)
            cls_chunks.append(cls)
            inst_chunks.append(inst)
            if rgb8 is None:
                any_rgb_none = True
            else:
                rgb_chunks.append(rgb8)
        positions = np.concatenate(pos_chunks, axis=0)
        class_ids = np.concatenate(cls_chunks, axis=0)
        instance_ids = np.concatenate(inst_chunks, axis=0)
        if any_rgb_none:
            # Colorless LAS: no per-point RGB to assemble. Return an
            # all-zeros uint8 array (not None) so the PLY writer always has
            # a colors array to write.
            colors = np.zeros((len(positions), 3), dtype=np.uint8)
        else:
            colors = np.concatenate(rgb_chunks, axis=0)
    else:
        raise ValueError(f"unknown resolution kind: {kind!r}")

    p50, p90 = raw_sample_spacing(ctx.scan_pos)
    meta = {"accuracy": {"p50": p50, "p90": p90, "loa": loa_band(p90)},
            "points": len(positions)}
    return positions, colors, class_ids, instance_ids, meta


def prism_aabb(prism: dict) -> tuple[np.ndarray, np.ndarray]:
    """Axis-aligned bounding box of a vertical prism (footprint polygon in
    XZ + a Y band), as (min, max) float64 (x, y, z) arrays. Used to pre-filter
    a raw point stream to a small region before the exact prism_indices test —
    no equivalent helper exists elsewhere (shapes.py only tests membership,
    never bounds)."""
    poly = np.asarray(prism["polygon"], dtype=np.float64)
    y0 = float(prism["y0"])
    height = float(prism["height"])
    x_min, x_max = float(poly[:, 0].min()), float(poly[:, 0].max())
    z_min, z_max = float(poly[:, 1].min()), float(poly[:, 1].max())
    return (np.array([x_min, y0, z_min]),
            np.array([x_max, y0 + height, z_max]))


def raw_region_sample_spacing(raw_path, prism: dict, scene_is_z_up: bool,
                               offset: np.ndarray) -> tuple[float, float]:
    """p50/p90 nearest-neighbour spacing of a scan's raw source, scoped to one
    eval region. Streams+filters the raw LAZ to the prism's AABB (cheap: only
    touches points local to one region, never the whole file in memory), then
    re-filters through the exact prism so points just outside a non-rectangular
    footprint (but inside its AABB) can't skew the measurement."""
    from scenes.lidar_io import load_laz_region

    aabb_min, aabb_max = prism_aabb(prism)
    positions, _colors = load_laz_region(
        raw_path, aabb_min.astype(np.float32), aabb_max.astype(np.float32),
        is_z_up=scene_is_z_up, offset=np.asarray(offset, dtype=np.float64))
    if len(positions) == 0:
        return 0.0, 0.0
    idx = prism_indices(positions, prism)
    return raw_sample_spacing(positions[idx])


def raw_sample_spacing(scan_pos, sample=100_000, seed=0):
    """Nearest-neighbor spacing of scan.ply (its true sampling pitch). Returns
    (p50, p90) over a bounded random subsample; p90 is the honest boundary bound
    under non-uniform LiDAR sampling. Built independent of any regime KD-tree."""
    scan_pos = np.asarray(scan_pos, dtype=np.float32).reshape(-1, 3)
    n = len(scan_pos)
    if n < 2:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    q = scan_pos if n <= sample else scan_pos[rng.choice(n, sample, replace=False)]
    d, _ = cKDTree(scan_pos).query(q, k=2)      # k=2: nearest non-self
    nn = d[:, 1]
    return float(np.percentile(nn, 50)), float(np.percentile(nn, 90))


# USIBD LOA Spec v3.0 accuracy bands: (name, upper tolerance in meters),
# finest first, boundaries inclusive.
LOA_BANDS = (("LOA50", 0.001), ("LOA40", 0.005), ("LOA30", 0.015), ("LOA20", 0.05))


def loa_band(spacing_m):
    """USIBD LOA band whose tolerance range contains `spacing_m`. Banded on the
    p90 sample spacing: a drawn boundary is off by at most ~one spacing, and
    p90 covers the sparse areas where boundaries are worst."""
    for name, upper in LOA_BANDS:
        if spacing_m <= upper:
            return name
    return "LOA10"
