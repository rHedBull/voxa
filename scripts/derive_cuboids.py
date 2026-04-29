"""Derive cuboid GT + predictions from per-point label arrays for one scene.

Compare mode wants cuboid annotations on disk under
`<voxa>/data/annotations/<safe-scene>/{ground_truth,predictions}.json`.
This script generates both from a SCHEMA-conformant `annotated/<scan>/`:

  ground_truth.json ← labels/gt_class_ids.npy + gt_segment_ids.npy
  predictions.json ← prelabel/ransac_instance_ids.npy + ransac_segment_summary.json

Cuboids are 2D-OBB around the vertical Y axis (XZ-plane PCA) — matches the
frontend `buildRecommendationCuboids` so on-screen recommendations and
saved predictions agree.

Run after RANSAC + merge inference have populated `prelabel/`. Skips
instances under 30 points or with any axis > 25 m (room-spanning planes).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from plyfile import PlyData

REC_MIN_POINTS = 30
REC_MAX_EXTENT = 25.0
REC_MIN_DIM = 0.05


def _load_classes(lidar_root: Path) -> dict[int, dict]:
    """Returns {class_id: {"label": ..., "color": ...}} from lidar/classes.json."""
    p = lidar_root / "classes.json"
    if not p.exists():
        return {}
    data = json.loads(p.read_text())
    out: dict[int, dict] = {}
    classes = data.get("classes") or data
    if isinstance(classes, dict):
        for name, meta in classes.items():
            cid = int(meta.get("id", -1))
            if cid >= 0:
                out[cid] = {"label": name, "color": meta.get("color", "#5b8def")}
    elif isinstance(classes, list):
        for entry in classes:
            cid = int(entry.get("id", -1))
            if cid >= 0:
                out[cid] = {"label": entry.get("label") or entry.get("name", "unknown"),
                            "color": entry.get("color", "#5b8def")}
    return out


def _instance_obb(points: np.ndarray, class_id: int,
                  classes_meta: dict[int, dict],
                  instance_id: int, source: str) -> Optional[dict]:
    """Compute one Cuboid dict for the given point cluster, or None if filtered.

    OBB is constrained to a single Y-axis rotation: PCA in the XZ plane gives
    the major-axis angle, and the box is AABB in (u, y, v) where u is along
    the major axis. Same convention as the frontend so on-screen and on-disk
    recommendations match.
    """
    n = len(points)
    if n < REC_MIN_POINTS:
        return None
    bb = np.ptp(points, axis=0)
    if bb.max() > REC_MAX_EXTENT:
        return None

    cx, _cy, cz = points.mean(axis=0)
    dxz = points[:, [0, 2]] - np.array([cx, cz])
    cxx = float((dxz[:, 0] ** 2).mean())
    czz = float((dxz[:, 1] ** 2).mean())
    cxz = float((dxz[:, 0] * dxz[:, 1]).mean())
    alpha = 0.5 * np.arctan2(2 * cxz, cxx - czz)
    ca, sa = float(np.cos(alpha)), float(np.sin(alpha))

    u = dxz[:, 0] * ca + dxz[:, 1] * sa
    v = -dxz[:, 0] * sa + dxz[:, 1] * ca
    u_min, u_max = float(u.min()), float(u.max())
    v_min, v_max = float(v.min()), float(v.max())
    y_min, y_max = float(points[:, 1].min()), float(points[:, 1].max())
    uC, vC = (u_min + u_max) / 2, (v_min + v_max) / 2

    center_x = cx + uC * ca - vC * sa
    center_z = cz + uC * sa + vC * ca
    center_y = (y_min + y_max) / 2
    size_u = max(u_max - u_min, REC_MIN_DIM)
    size_v = max(v_max - v_min, REC_MIN_DIM)
    size_y = max(y_max - y_min, REC_MIN_DIM)
    rot_y = -float(alpha)

    cls_meta = classes_meta.get(class_id, {})
    return {
        "id": f"{source}_{instance_id}",
        "cls": cls_meta.get("label", "unknown"),
        "label": "",
        "color": cls_meta.get("color", "#5b8def"),
        "center": [float(center_x), float(center_y), float(center_z)],
        "size": [size_u, size_y, size_v],
        "rotation": [0.0, rot_y, 0.0],
        "conf": 1.0,
        "source": source,
        "confirmed": False,
    }


def _build_cuboids(points: np.ndarray, class_ids: np.ndarray,
                   instance_ids: np.ndarray, classes_meta: dict[int, dict],
                   source: str) -> list[dict]:
    """One cuboid per unique instance id (>=0)."""
    cuboids: list[dict] = []
    for iid in np.unique(instance_ids):
        if iid < 0:
            continue
        m = instance_ids == iid
        pts = points[m]
        # Take the first point's class — instances are post-filtered to one
        # class on the backend so it's canonical.
        cid = int(class_ids[m][0]) if m.any() else -1
        if cid < 0:
            continue
        cb = _instance_obb(pts, cid, classes_meta, int(iid), source=source)
        if cb is not None:
            cuboids.append(cb)
    return cuboids


def main(scene: str = "munich_water_pump",
         lidar_root: Path = Path("/home/hendrik/coding/engine/data/lidar"),
         voxa_data: Path = Path("/home/hendrik/coding/engine/tools/labeling/voxa/data")) -> None:
    scan_dir = lidar_root / "annotated" / scene
    out_dir = voxa_data / "annotations" / f"annotated__{scene}"
    out_dir.mkdir(parents=True, exist_ok=True)

    ply = PlyData.read(str(scan_dir / "source" / "scan.ply"))
    v = ply["vertex"]
    points = np.column_stack([v["x"], v["y"], v["z"]]).astype(np.float64)
    n = len(points)
    classes_meta = _load_classes(lidar_root)

    # GT cuboids from labels/.
    gt_class = np.load(scan_dir / "labels" / "gt_class_ids.npy")
    gt_inst = np.load(scan_dir / "labels" / "gt_segment_ids.npy")
    if len(gt_class) == n == len(gt_inst):
        gt_cuboids = _build_cuboids(points, gt_class, gt_inst, classes_meta, source="gt")
    else:
        print(f"WARN: gt arrays size mismatch ({len(gt_class)}, {len(gt_inst)} vs {n}); skipping GT")
        gt_cuboids = []

    (out_dir / "ground_truth.json").write_text(json.dumps({
        "scene": f"annotated/{scene}",
        "kind": "gt",
        "instances": gt_cuboids,
        "meta": {"source": "derived from labels/", "n_points": n},
    }, indent=2))
    print(f"  GT cuboids:   {len(gt_cuboids)}")

    # Prediction cuboids from prelabel/.
    pred_path = scan_dir / "prelabel" / "ransac_instance_ids.npy"
    summary_path = scan_dir / "prelabel" / "ransac_segment_summary.json"
    if pred_path.exists() and summary_path.exists():
        pred_inst = np.load(pred_path)
        summary = json.loads(summary_path.read_text())
        seg_to_class = {int(s["id"]): int(s["class_id"])
                        for s in summary.get("segments", [])}
        pred_class = np.full(n, -1, dtype=np.int32)
        for sid, cid in seg_to_class.items():
            pred_class[pred_inst == sid] = cid
        if len(pred_inst) == n:
            pred_cuboids = _build_cuboids(points, pred_class, pred_inst, classes_meta, source="recommendation")
        else:
            print(f"WARN: prelabel size mismatch ({len(pred_inst)} vs {n}); skipping pred")
            pred_cuboids = []
    else:
        print("  no prelabel/ for this scene; skipping predictions")
        pred_cuboids = []

    (out_dir / "predictions.json").write_text(json.dumps({
        "scene": f"annotated/{scene}",
        "kind": "pred",
        "instances": pred_cuboids,
        "meta": {"source": "derived from prelabel/", "n_points": n},
    }, indent=2))
    print(f"  Pred cuboids: {len(pred_cuboids)}")


if __name__ == "__main__":
    scene = sys.argv[1] if len(sys.argv) > 1 else "munich_water_pump"
    print(f"=== {scene} ===")
    main(scene)
