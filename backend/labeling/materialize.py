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

    # One mask per volume, precomputed once for the whole call.
    vol_masks = []
    for v in index.volumes:
        mask = np.zeros(n, dtype=bool)
        if v["kind"] == "obb":
            idx = obb_indices(target_pos, v["shape"])
        elif v["kind"] == "tube":
            idx = tube_indices(target_pos, v["paths"])
        else:
            raise ValueError(f"unknown volume kind: {v['kind']!r}")
        mask[idx] = True
        vol_masks.append(mask)

    target_cls = np.full(n, -1, dtype=np.int8)
    target_inst = np.full(n, -1, dtype=np.int32)

    def seq_of(inst_id):
        return index.seq_by_inst.get(inst_id)

    for p_idx in range(n):
        p = target_pos[p_idx]
        candidates = []  # list of (seq_or_neg_inf, instance_id)

        for v, mask in zip(index.volumes, vol_masks):
            if mask[p_idx]:
                s = seq_of(v["instance_id"])
                candidates.append((s if s is not None else float("-inf"), v["instance_id"]))

        # Baseline via nearest scan.ply sample.
        _, s_idx = index.tree_all.query(p)
        oi = int(index.work_inst[s_idx])
        baseline_inst = None
        if oi in index.vol_ids:
            vol = next(v for v in index.volumes if v["instance_id"] == oi)
            vmask = vol_masks[index.volumes.index(vol)]
            if not vmask[p_idx]:
                # Leak: nearest sample belongs to a volume that doesn't
                # actually cover p -> re-query the non-volumetric tree.
                if index.tree_nonvol is not None:
                    _, nn_idx = index.tree_nonvol.query(p)
                    baseline_inst = int(index.work_inst[index.nonvol_idx[nn_idx]])
                # else: no baseline candidate.
            else:
                baseline_inst = oi
        else:
            baseline_inst = oi

        if baseline_inst is not None and baseline_inst != -1:
            s = seq_of(baseline_inst)
            candidates.append((s if s is not None else float("-inf"), baseline_inst))

        if not candidates:
            continue

        _, winner_inst = max(candidates, key=lambda c: c[0])
        target_inst[p_idx] = winner_inst
        target_cls[p_idx] = index.inst_class_id[winner_inst]

    return target_cls, target_inst
