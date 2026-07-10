"""Density-agnostic label materialization: produce per-point (positions, colors,
class_ids, instance_ids) for a session at any target density. Regime A =
down-sample by index-transfer (exact); regime B (denser) added in later tasks."""
import numpy as np
from app.core import _subsample_indices  # seeded, ascending indices into range(N)


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
        seq = inst.get("seq")
        iid = inst.get("segId")
        if src == "box" and inst.get("center") and inst.get("size"):
            out.append({"kind": "obb", "instance_id": iid, "seq": seq,
                        "shape": {"center": inst["center"], "size": inst["size"],
                                  "rotation": inst.get("rotation", [0, 0, 0])}})
        elif src == "draw" and iid in paths_by_inst:
            out.append({"kind": "tube", "instance_id": iid, "seq": seq,
                        "paths": paths_by_inst[iid]})
    return out
