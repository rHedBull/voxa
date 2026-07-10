"""Density-agnostic label materialization: produce per-point (positions, colors,
class_ids, instance_ids) for a session at any target density. Regime A =
down-sample by index-transfer (exact); regime B (denser) replays labels onto
raw/upsampled points via the max-seq rule below."""
from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from app.core import _subsample_indices  # seeded, ascending indices into range(N)
from labeling.shapes import obb_indices
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
    """Volumetric instances only (source in {'box','draw'}). Box -> obb from
    center/size/rotation; draw -> tube from the instance's centerlines paths
    (grouped by instance_id == segId). Each carries its apply-order `seq`."""
    paths_by_inst = {}
    for p in (centerlines or {}).get("paths", []):
        paths_by_inst.setdefault(int(p["instance_id"]), []).append(p)
    out = []
    for inst in instances:
        src = inst.get("source")
        if src not in ("box", "draw") or inst.get("segId") is None:
            continue
        seq = inst.get("seq")
        iid = int(inst["segId"])  # symmetric int key with paths_by_inst; the seq
        # map + KD-tree replay (Task 4) key off this same instance_id
        if src == "box" and inst.get("center") and inst.get("size"):
            out.append({"kind": "obb", "instance_id": iid, "seq": seq,
                        "shape": {"center": inst["center"], "size": inst["size"],
                                  "rotation": inst.get("rotation", [0, 0, 0])}})
        elif src == "draw" and iid in paths_by_inst:
            out.append({"kind": "tube", "instance_id": iid, "seq": seq,
                        "paths": paths_by_inst[iid]})
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
        target_cls[p_idx] = index.inst_class_id[winner_inst]

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
