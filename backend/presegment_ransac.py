"""Curvature + RANSAC presegmentation.

Ported from `industrial_point_labeler/segmentation/ransac.py` with the
file-I/O / argparse layer stripped out. The callable here returns
`(instance_ids, summary)` so callers can decide how to persist them
(CLI writes `prelabel/ransac_*`; `seg_inference.py` keeps its own
cache).

Pipeline (Strategy v4):
  1. Estimate normals + multi-scale principal curvatures (k1, k2)
  2. Classify points by (k1, k2) signature: flat / cylindrical /
     spherical / saddle / edge
  3. Iterative plane RANSAC over the flat points
  4. Region-grow cylindrical points by axis + radius compatibility,
     then fit a cylinder per cluster via algebraic + RANSAC circle fit
  4b. DBSCAN over spherical points (small eps to avoid catchall blobs)
  5. Post-assignment of stragglers via majority-vote with normal
      consistency
  6. Merge over-segmented co-axial cylinders

Public API:
    presegment(xyz, *, class_map=None, log=None) -> (instance_ids, summary)

Returns
-------
instance_ids : (N,) int32
    Per-point instance id, or -1 for unlabeled.
summary : list[dict]
    One entry per instance with keys
    ``id, class_id, label, n_points`` and primitive-specific fields
    (``normal/d`` for planes, ``center/axis/radius/length`` for cylinders).
    ``class_id`` is filled from ``class_map`` via the same keyword
    heuristic ``seg_inference._ransac_class_for_segment`` uses, so the
    output is directly consumable by ``segment_io.load_prelabel``.
"""
from __future__ import annotations

from collections import Counter
from typing import Callable, Optional

import numpy as np


# ── label → class_id mapping ──────────────────────────────────────────
# Mirrors ``seg_inference._LABEL_KEYWORD_MAP`` so prelabel summaries
# produced here and prelabel summaries produced by the merge model are
# interchangeable. Order matters: more specific keywords first.
_LABEL_KEYWORD_MAP: tuple[tuple[tuple[str, ...], str], ...] = (
    (("pipe",), "pipe"),
    (("tank", "vessel", "drum", "silo"), "tank"),
    (("flat_surface", "wall", "floor", "ceiling", "beam", "structural", "plane"), "structural"),
    (("fitting", "elbow", "joint", "flange"), "fitting"),
    (("equipment", "pump", "valve", "motor", "instrument"), "equipment"),
)


def _label_to_class_id(label: str, class_map: Optional[dict[str, int]]) -> int:
    """Map a RANSAC segment's free-form label to a voxa class id.

    Falls back to the "unknown" class (any of the conventional name
    variants) when the label doesn't match — keeps the SCHEMA invariant
    `class_id == -1 ⟺ instance_id == -1` from forcing perfectly good
    segment groupings to be erased just because the label was a generic
    "saddle"/"curved_unclassified"/"spherical".
    """
    if not class_map:
        return -1
    if label:
        key = label.lower().strip()
        for name, cid in class_map.items():
            if name.lower() == key:
                return int(cid)
        for keywords, target in _LABEL_KEYWORD_MAP:
            if any(kw in key for kw in keywords):
                for name, cid in class_map.items():
                    if name.lower() == target:
                        return int(cid)
    # No keyword match (or empty label) — default to the "unknown"-style
    # class so the segment is still saveable & visually distinct in the UI.
    for name, cid in class_map.items():
        n = name.lower()
        if n in ("unknown", "other", "uncategorized", "unlabeled"):
            return int(cid)
    return -1


# ── curvature estimation ──────────────────────────────────────────────

def _compute_curvatures_at_scale(points, normals, nn_idx):
    """Vectorized principal curvature computation for a single neighbor scale."""
    n = len(points)

    ref = np.where(np.abs(normals[:, 0:1]) < 0.9,
                   np.broadcast_to([1, 0, 0], (n, 3)),
                   np.broadcast_to([0, 1, 0], (n, 3)))
    t1 = np.cross(normals, ref)
    t1_norm = np.linalg.norm(t1, axis=1, keepdims=True) + 1e-10
    t1 = t1 / t1_norm
    t2 = np.cross(normals, t1)

    neighbors = points[nn_idx]
    diff = neighbors - points[:, np.newaxis, :]

    u = np.einsum('nkd,nd->nk', diff, t1)
    v = np.einsum('nkd,nd->nk', diff, t2)
    w = np.einsum('nkd,nd->nk', diff, normals)

    A = np.stack([u**2, u*v, v**2], axis=-1)
    AtA = np.einsum('nki,nkj->nij', A, A)
    Atw = np.einsum('nki,nk->ni', A, w)

    AtA[:, 0, 0] += 1e-8
    AtA[:, 1, 1] += 1e-8
    AtA[:, 2, 2] += 1e-8

    try:
        coeffs = np.linalg.solve(AtA, Atw[..., None])[..., 0]
    except np.linalg.LinAlgError:
        coeffs = np.zeros((n, 3))
        for i in range(n):
            try:
                coeffs[i] = np.linalg.solve(AtA[i], Atw[i])
            except np.linalg.LinAlgError:
                pass

    a = coeffs[:, 0]; b = coeffs[:, 1]; c = coeffs[:, 2]

    w_pred = np.einsum('nki,ni->nk', A, coeffs)
    residuals = np.mean((w - w_pred)**2, axis=1)

    trace = 2*a + 2*c
    det = 4*a*c - b*b
    disc = np.sqrt(np.maximum(trace**2 - 4*det, 0))
    lam1 = (trace + disc) / 2
    lam2 = (trace - disc) / 2

    abs1 = np.abs(lam1); abs2 = np.abs(lam2)
    swap = abs1 < abs2
    k1 = np.where(swap, lam2, lam1)
    k2 = np.where(swap, lam1, lam2)

    lam_max = np.where(swap, lam2, lam1)
    lam_min = np.where(swap, lam1, lam2)

    vmax_0 = -b
    vmax_1 = 2*a - lam_max
    vmax_norm = np.sqrt(vmax_0**2 + vmax_1**2) + 1e-10
    vmax_0 /= vmax_norm; vmax_1 /= vmax_norm

    vmin_0 = -b
    vmin_1 = 2*a - lam_min
    vmin_norm = np.sqrt(vmin_0**2 + vmin_1**2) + 1e-10
    vmin_0 /= vmin_norm; vmin_1 /= vmin_norm

    directions = vmax_0[:, np.newaxis] * t1 + vmax_1[:, np.newaxis] * t2
    axes = vmin_0[:, np.newaxis] * t1 + vmin_1[:, np.newaxis] * t2

    return k1, k2, directions, axes, residuals


def _principal_curvatures(points, normals, k=20, log=print):
    """Multi-scale (k=10, k, 40) curvature estimation, picking lowest-residual scale per point."""
    from scipy.spatial import cKDTree
    n = len(points)
    tree = cKDTree(points)

    scales = [10, k, 40]
    max_k = max(scales)
    _, nn_idx_all = tree.query(points, k=max_k + 1)
    nn_idx_all = nn_idx_all[:, 1:]

    all_k1, all_k2, all_dirs, all_axes, all_res = [], [], [], [], []
    for s in scales:
        nn_idx_s = nn_idx_all[:, :s]
        k1_s, k2_s, dir_s, ax_s, res_s = _compute_curvatures_at_scale(points, normals, nn_idx_s)
        all_k1.append(k1_s); all_k2.append(k2_s)
        all_dirs.append(dir_s); all_axes.append(ax_s)
        all_res.append(res_s)

    residuals = np.stack(all_res, axis=0)
    best_scale = np.argmin(residuals, axis=0)
    k1 = np.choose(best_scale, all_k1)
    k2 = np.choose(best_scale, all_k2)
    all_dirs = np.stack(all_dirs, axis=0)
    all_axes = np.stack(all_axes, axis=0)
    idx = np.arange(n)
    directions = all_dirs[best_scale, idx]
    axes = all_axes[best_scale, idx]

    for i, s in enumerate(scales):
        cnt = (best_scale == i).sum()
        log(f"    scale k={s}: {cnt} points ({100*cnt/n:.1f}%)")

    flip_mask = axes[:, 2] < 0
    flip_mask2 = (np.abs(axes[:, 2]) < 0.1) & (axes[:, 0] < 0)
    axes[flip_mask | flip_mask2] *= -1

    return k1, k2, directions, axes


RANSAC_DEFAULTS: dict[str, float] = {
    # plane RANSAC
    "plane_distance_threshold": 0.025,  # m — inlier band for plane fit
    "plane_min_inliers":        30.0,   # smallest plane that survives (low → small surfaces e.g. cover plates kept)
    "max_planes":               40.0,   # cap on iterative plane RANSAC (raised since planes are smaller now)
    "plane_cluster_eps":        0.15,   # m — DBSCAN eps to split spatially disjoint plane inliers (0 disables)
    "leftover_cluster_eps":     0.10,   # m — DBSCAN eps for unassigned-point catchall (0 disables, snap-to-nearest only)
    "leftover_min_pts":         30.0,   # smallest leftover blob that gets its own segment
    # curvature classification
    "flat_thresh":              0.9,    # k_max boundary: flat vs curved (raised → more points qualify as flat)
    "cylinder_ratio_thresh":    4.0,    # k_max/k_min to call a point cylindrical (raised → looser definition)
    # cylinder region-grow
    "cyl_search_radius":        0.12,   # m — neighborhood for region growing
    "cyl_axis_thresh":          0.92,   # |dot(n_a, n_b)| min to join
    "cyl_radius_ratio":         1.8,    # max(r1,r2)/min(r1,r2) to join
    "cyl_distance_threshold":   0.03,   # m — radial tolerance during fit
    # post-merge
    "merge_axis_dot":           0.95,   # |dot(axis_i, axis_j)| min to merge
    "merge_radius_ratio":       1.4,    # max(r1,r2)/min(r1,r2) to merge
}


def _classify_by_curvature(k1, k2, cylinder_ratio_thresh=3.0, flat_thresh=0.5):
    """Per-point surface label: 0=flat, 1=cyl, 2=sphere, 3=saddle, 4=edge."""
    n = len(k1)
    labels = np.full(n, 3, dtype=np.int32)
    abs_k1 = np.abs(k1); abs_k2 = np.abs(k2)
    k_max = np.maximum(abs_k1, abs_k2)
    k_min = np.minimum(abs_k1, abs_k2)

    flat = k_max < flat_thresh
    labels[flat] = 0

    ratio = k_max / (k_min + 1e-6)
    cylindrical = ~flat & (ratio > cylinder_ratio_thresh) & (k_max < 50)
    labels[cylindrical] = 1

    spherical = ~flat & ~cylindrical & (ratio < 2.0) & (k_max < 50)
    labels[spherical] = 2

    noisy = k_max > 50
    labels[noisy] = 4
    return labels


# ── plane RANSAC (open3d) ─────────────────────────────────────────────

def _iterative_plane_ransac(points, indices, *, distance_threshold=0.025,
                            min_inliers=100, max_planes=30,
                            cluster_eps=0.15, log=print):
    import open3d as o3d
    remaining = indices.copy()
    planes = []
    for _ in range(max_planes):
        if len(remaining) < min_inliers:
            break
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[remaining])
        model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold, ransac_n=3, num_iterations=500)
        if len(inliers) < min_inliers:
            break
        global_inliers = remaining[inliers]

        # Spatial connectivity: a plane has infinite extent so RANSAC pulls
        # in coplanar-but-distant slabs (e.g. floor + ceiling tile at the
        # same height). Keep only the largest connected component;
        # spatially separated inliers go back to the candidate pool so they
        # can form their own planes on later iterations.
        if cluster_eps > 0 and len(global_inliers) >= min_inliers:
            inlier_pcd = o3d.geometry.PointCloud()
            inlier_pcd.points = o3d.utility.Vector3dVector(points[global_inliers])
            cc_labels = np.asarray(
                inlier_pcd.cluster_dbscan(eps=cluster_eps, min_points=10, print_progress=False)
            )
            if cc_labels.size > 0 and (cc_labels >= 0).any():
                vals, counts = np.unique(cc_labels[cc_labels >= 0], return_counts=True)
                largest = vals[int(np.argmax(counts))]
                keep_mask = cc_labels == largest
                if int(keep_mask.sum()) < min_inliers:
                    break
                global_inliers = global_inliers[keep_mask]

        a, b, c, d = model
        extent_pts = points[global_inliers]
        extent = float(np.linalg.norm(extent_pts.max(axis=0) - extent_pts.min(axis=0)))
        planes.append({
            "type": "plane",
            "label": "flat_surface",
            "normal": [float(a), float(b), float(c)],
            "d": float(d),
            "n_points": int(len(global_inliers)),
            "extent": extent,
            "global_indices": global_inliers,
        })
        remaining = np.setdiff1d(remaining, global_inliers)
        log(f"    Plane: {len(global_inliers)} pts, "
            f"normal=[{a:.2f},{b:.2f},{c:.2f}], extent={extent:.2f}m")
    return planes, remaining


# ── per-cluster cylinder fit ──────────────────────────────────────────

def _fit_cylinder_to_cluster(points, normals, axis_hint=None,
                             distance_threshold=0.03,
                             min_radius=0.015, max_radius=2.0):
    n = len(points)
    if n < 20:
        return None

    if axis_hint is not None:
        axis = axis_hint / (np.linalg.norm(axis_hint) + 1e-10)
    else:
        mean_n = normals.mean(axis=0)
        centered_n = normals - mean_n
        try:
            _, _, Vt = np.linalg.svd(centered_n, full_matrices=False)
        except np.linalg.LinAlgError:
            return None
        axis = Vt[-1] / (np.linalg.norm(Vt[-1]) + 1e-10)

    centroid = points.mean(axis=0)
    vecs = points - centroid
    along = np.dot(vecs, axis)
    perp = vecs - along[:, np.newaxis] * axis

    e1 = np.array([1, 0, 0]) - np.dot([1, 0, 0], axis) * axis
    if np.linalg.norm(e1) < 0.1:
        e1 = np.array([0, 1, 0]) - np.dot([0, 1, 0], axis) * axis
    e1 = e1 / (np.linalg.norm(e1) + 1e-10)
    e2 = np.cross(axis, e1)
    e2 = e2 / (np.linalg.norm(e2) + 1e-10)

    coords_2d = np.column_stack([np.dot(perp, e1), np.dot(perp, e2)])

    A_mat = np.column_stack([coords_2d, np.ones(n)])
    b_vec = -(coords_2d[:, 0]**2 + coords_2d[:, 1]**2)
    try:
        D, E, F = np.linalg.lstsq(A_mat, b_vec, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    cx, cy = -D/2, -E/2
    r_sq = cx**2 + cy**2 - F
    if r_sq <= 0:
        return None
    radius = np.sqrt(r_sq)
    if radius < min_radius or radius > max_radius:
        return None

    dists = np.sqrt((coords_2d[:, 0] - cx)**2 + (coords_2d[:, 1] - cy)**2)
    best_mask = np.abs(dists - radius) < distance_threshold
    best_n = int(best_mask.sum())
    best_r, best_cx, best_cy = radius, cx, cy

    for _ in range(min(200, n * 3)):
        idx = np.random.choice(n, 3, replace=False)
        sub = coords_2d[idx]
        A_s = np.column_stack([sub, np.ones(3)])
        b_s = -(sub[:, 0]**2 + sub[:, 1]**2)
        try:
            D, E, F = np.linalg.lstsq(A_s, b_s, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        cx_t, cy_t = -D/2, -E/2
        r_sq_t = cx_t**2 + cy_t**2 - F
        if r_sq_t <= 0:
            continue
        r_t = np.sqrt(r_sq_t)
        if r_t < min_radius or r_t > max_radius:
            continue
        d_t = np.sqrt((coords_2d[:, 0] - cx_t)**2 + (coords_2d[:, 1] - cy_t)**2)
        mask_t = np.abs(d_t - r_t) < distance_threshold
        n_t = int(mask_t.sum())
        if n_t > best_n:
            best_n = n_t
            best_r, best_cx, best_cy = r_t, cx_t, cy_t
            best_mask = mask_t

    center_3d = centroid + best_cx * e1 + best_cy * e2
    inlier_along = along[best_mask]
    length = float(inlier_along.max() - inlier_along.min()) if best_mask.any() else 0.0

    return {
        "center": center_3d, "axis": axis, "radius": float(best_r),
        "length": length, "n_inliers": int(best_n), "inlier_ratio": best_n / n,
    }


def _classify_cylinder(radius, length):
    aspect = length / (2 * radius) if radius > 0 else 0
    if radius > 0.4:
        return "tank"
    if radius > 0.15:
        return "tank" if aspect < 3 else "large_pipe"
    if radius > 0.03:
        if aspect > 5:
            return "pipe"
        if aspect > 1.5:
            return "short_pipe"
        return "fitting"
    return "small_pipe"


# ── normal estimation (open3d) ────────────────────────────────────────

def _estimate_normals(points, *, knn=30, orient_k=15):
    """Open3D normals + tangent-plane orientation, with the same UTM-coord
    safety net main.py uses on load (qhull is unstable for far-from-origin
    coords). Returns normals in the original frame; the centering used here
    is internal and discarded."""
    import open3d as o3d
    pts = np.asarray(points, dtype=np.float64)
    centroid = pts.mean(axis=0)
    if np.abs(centroid).max() > 1000.0:
        pts = pts - centroid
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pts)
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_consistent_tangent_plane(k=orient_k)
    return np.asarray(pcd.normals).copy()


# ── main entrypoint ───────────────────────────────────────────────────

def _feature_split_instances(
    instance_ids: np.ndarray,
    features: np.ndarray,
    seen: np.ndarray,
    *,
    min_size: int,
    target_size: int,
    max_k: int,
    log: Callable[[str], None],
) -> np.ndarray:
    """Split large RANSAC instances into sub-clusters by feature k-means.

    Bias is toward over-segmentation: an instance is split when it is
    large AND most of its points carry features. Sub-clusters smaller
    than 5% of the parent are rejected and k is decremented once.
    """
    from sklearn.cluster import MiniBatchKMeans  # local import (heavy)

    out = instance_ids.copy()
    next_id = int(out.max()) + 1 if out.max() >= 0 else 0
    n_split_inst = 0
    n_added_clusters = 0
    for inst_id in range(int(out.max()) + 1):
        idx = np.where(out == inst_id)[0]
        if idx.size < min_size:
            continue
        has = seen[idx] > 0
        if has.sum() < idx.size * 0.5:
            continue
        k = int(np.clip(round(idx.size / max(target_size, 1)), 1, max_k))
        if k <= 1:
            continue
        X = features[idx][has].astype(np.float32)
        km = MiniBatchKMeans(n_clusters=k, random_state=0,
                              batch_size=2048, n_init=3)
        sub = km.fit_predict(X)
        counts = np.bincount(sub, minlength=k)
        if (counts < max(20, int(0.05 * has.sum()))).any():
            k = max(1, k - 1)
            if k == 1:
                continue
            km = MiniBatchKMeans(n_clusters=k, random_state=0,
                                  batch_size=2048, n_init=3)
            sub = km.fit_predict(X)
        # Points with features → their sub cluster; without features → 0.
        local = np.zeros(idx.size, dtype=np.int32)
        local[has] = sub
        # Leave first sub-cluster as the original id, allocate new ids for the rest.
        new_ids = np.array(
            [inst_id] + [next_id + j for j in range(k - 1)], dtype=np.int32,
        )
        next_id += k - 1
        out[idx] = new_ids[local]
        n_split_inst += 1
        n_added_clusters += k - 1
    log(f"  feature split: {n_split_inst} instances → "
        f"+{n_added_clusters} sub-clusters")
    return out


def presegment(
    xyz: np.ndarray,
    *,
    class_map: Optional[dict[str, int]] = None,
    log: Callable[[str], None] = print,
    params: Optional[dict[str, float]] = None,
    labeler_strict: bool = False,
    stage_callback: Optional[Callable[[str, np.ndarray], None]] = None,
    features: Optional[np.ndarray] = None,
    feature_seen: Optional[np.ndarray] = None,
    feature_split_min_size: int = 3000,
    feature_split_target_size: int = 5000,
    feature_split_max_k: int = 8,
) -> tuple[np.ndarray, list[dict]]:
    """Run RANSAC presegmentation.

    Parameters
    ----------
    xyz : (N, 3) float
        Point positions. Recentered internally if magnitudes exceed 1e3.
    class_map : dict[str, int] | None
        Voxa class-name → class-id mapping. Used to fill ``class_id`` in
        the summary; if None, all summary entries get ``class_id=-1``.
    log : callable
        Progress sink. Default ``print``; pass ``lambda *_: None`` to mute.
    params : dict | None
        Per-call overrides for ``RANSAC_DEFAULTS``. Unknown keys are
        ignored; missing keys fall back to the module-level defaults.
    labeler_strict : bool
        Run the bit-for-bit ``industrial_point_labeler`` pipeline:
        disables the connected-component plane split, skips the
        nearest-neighbour Step 7 catchall (so saddle/edge points stay
        at -1), and uses the labeler's stale ``global_indices`` in the
        cylinder merge. Use to A/B against the labeler; output will
        match modulo the unseeded RANSAC RNG noise both pipelines share.

    Returns
    -------
    instance_ids : (N,) int32
        Per-point instance id (-1 = unassigned).
    summary : list[dict]
        Per-instance metadata, ready to be serialized to
        ``prelabel/ransac_segment_summary.json`` under a ``segments`` key.
    """
    cfg = dict(RANSAC_DEFAULTS)
    if labeler_strict:
        cfg["plane_cluster_eps"] = 0.0
    if params:
        for k, v in params.items():
            if k in cfg and v is not None:
                cfg[k] = float(v)

    points = np.asarray(xyz, dtype=np.float64).copy()
    n_total = len(points)
    log(f"Presegmenting {n_total} points, extent: {np.ptp(points, axis=0).round(2)}")

    # Internal recenter for normal-estimation stability. Geometric outputs
    # (planes, cylinder centers) are reported in the original frame, so we
    # un-shift after fitting.
    recenter = np.zeros(3)
    centroid = points.mean(axis=0)
    if np.abs(centroid).max() > 1000.0:
        recenter = centroid.copy()
        points = points - recenter
        log(f"  Recentered: subtracted {recenter.round(2)}")

    log("Estimating normals…")
    normals = _estimate_normals(points)

    log("Computing principal curvatures…")
    k1, k2, _dirs, cyl_axes = _principal_curvatures(points, normals, k=20, log=log)
    log(f"  k1: median={np.median(np.abs(k1)):.3f}, p95={np.percentile(np.abs(k1), 95):.3f}")
    log(f"  k2: median={np.median(np.abs(k2)):.3f}, p95={np.percentile(np.abs(k2), 95):.3f}")

    surface_labels = _classify_by_curvature(
        k1, k2,
        cylinder_ratio_thresh=cfg["cylinder_ratio_thresh"],
        flat_thresh=cfg["flat_thresh"],
    )
    type_names = {0: "flat", 1: "cyl", 2: "sphere", 3: "saddle", 4: "edge"}
    for t, name in type_names.items():
        cnt = int((surface_labels == t).sum())
        log(f"  {name}: {cnt} ({100*cnt/n_total:.1f}%)")

    instance_ids = np.full(n_total, -1, dtype=np.int32)
    primitives: list[dict] = []
    next_id = 0

    # ── Step 3: planes from flat points ──
    log("Plane extraction…")
    flat_idx = np.where(surface_labels == 0)[0]
    planes, _ = _iterative_plane_ransac(
        points, flat_idx,
        distance_threshold=cfg["plane_distance_threshold"],
        min_inliers=int(cfg["plane_min_inliers"]),
        max_planes=int(cfg["max_planes"]),
        cluster_eps=cfg["plane_cluster_eps"],
        log=log,
    )
    for p in planes:
        p["id"] = next_id
        instance_ids[p["global_indices"]] = next_id
        next_id += 1
    primitives.extend(planes)
    log(f"  {len(planes)} planes")

    # ── Step 4: region-grow cylindrical points ──
    from scipy.spatial import cKDTree
    log("Cylinder region-growing…")
    cyl_idx = np.where(surface_labels == 1)[0]
    cyl_pts = points[cyl_idx]
    cyl_ax = cyl_axes[cyl_idx]
    cyl_k1 = np.abs(k1[cyl_idx])
    est_radius = np.where(cyl_k1 > 0.1, 1.0 / cyl_k1, 10.0)
    est_radius = np.clip(est_radius, 0.01, 5.0)

    if len(cyl_idx) > 0:
        cyl_tree = cKDTree(cyl_pts)
        search_radius = cfg["cyl_search_radius"]
        axis_thresh = cfg["cyl_axis_thresh"]
        radius_ratio = cfg["cyl_radius_ratio"]
        neighbor_lists = cyl_tree.query_ball_tree(cyl_tree, r=search_radius)

        cyl_cluster = np.full(len(cyl_idx), -1, dtype=np.int32)
        cluster_id = 0
        density = np.array([len(nl) for nl in neighbor_lists])
        seed_order = np.argsort(-density)

        for seed in seed_order:
            if cyl_cluster[seed] >= 0:
                continue
            cyl_cluster[seed] = cluster_id
            queue = [seed]
            seed_radius = est_radius[seed]
            axis_sum = cyl_ax[seed].copy()

            while queue:
                current = queue.pop(0)
                cur_axis = axis_sum / (np.linalg.norm(axis_sum) + 1e-10)
                for nb in neighbor_lists[current]:
                    if cyl_cluster[nb] >= 0:
                        continue
                    dot = abs(np.dot(cyl_ax[nb], cur_axis))
                    if dot < axis_thresh:
                        continue
                    r_ratio = max(est_radius[nb], seed_radius) / (min(est_radius[nb], seed_radius) + 1e-6)
                    if r_ratio > radius_ratio:
                        continue
                    cyl_cluster[nb] = cluster_id
                    queue.append(nb)
                    sign = 1.0 if np.dot(cyl_ax[nb], cur_axis) >= 0 else -1.0
                    axis_sum += sign * cyl_ax[nb]
            cluster_id += 1

        # Per-cluster cylinder fit
        min_cluster_pts = 20
        for c in range(cluster_id):
            c_mask = cyl_cluster == c
            n_pts = int(c_mask.sum())
            if n_pts < min_cluster_pts:
                continue
            c_global = cyl_idx[c_mask]
            c_pts = points[c_global]
            c_normals = normals[c_global]
            c_axis = cyl_ax[c_mask]
            signs = np.sign(np.dot(c_axis, c_axis.mean(axis=0)))
            signs[signs == 0] = 1
            mean_axis = (c_axis * signs[:, np.newaxis]).mean(axis=0)
            mean_axis = mean_axis / (np.linalg.norm(mean_axis) + 1e-10)

            cyl = _fit_cylinder_to_cluster(c_pts, c_normals,
                                           axis_hint=mean_axis,
                                           distance_threshold=cfg["cyl_distance_threshold"])
            if cyl and cyl["inlier_ratio"] > 0.25:
                lbl = _classify_cylinder(cyl["radius"], cyl["length"])
                primitives.append({
                    "id": next_id, "type": "cylinder", "label": lbl,
                    "radius": float(cyl["radius"]), "length": float(cyl["length"]),
                    "center": cyl["center"].tolist(),
                    "axis": cyl["axis"].tolist(),
                    "n_points": n_pts,
                    "inlier_ratio": float(cyl["inlier_ratio"]),
                    "global_indices": c_global,
                })
            else:
                primitives.append({
                    "id": next_id, "type": "unknown", "label": "curved_unclassified",
                    "n_points": n_pts, "global_indices": c_global,
                })
            instance_ids[c_global] = next_id
            next_id += 1

    # ── Step 4b: spherical clusters via DBSCAN ──
    log("Sphere clustering…")
    sph_idx = np.where(surface_labels == 2)[0]
    if len(sph_idx) > 50:
        from sklearn.cluster import DBSCAN
        sph_pts = points[sph_idx]
        db = DBSCAN(eps=0.05, min_samples=15).fit(sph_pts)
        max_sphere_pts = n_total // 20
        for c in range(int(db.labels_.max()) + 1):
            c_mask = db.labels_ == c
            n_pts = int(c_mask.sum())
            if n_pts < 30 or n_pts > max_sphere_pts:
                continue
            c_global = sph_idx[c_mask]
            primitives.append({
                "id": next_id, "type": "sphere", "label": "spherical",
                "n_points": n_pts, "global_indices": c_global,
            })
            instance_ids[c_global] = next_id
            next_id += 1

    # ── Step 5: post-assignment of stragglers ──
    log("Post-assignment…")
    full_tree = cKDTree(points)
    instance_mean_normals: dict[int, Optional[np.ndarray]] = {}
    instance_types: dict[int, str] = {}
    for prim in primitives:
        pid = prim["id"]
        idx = prim.get("global_indices")
        if idx is not None and len(idx) > 0:
            mn = normals[idx].mean(axis=0)
            mn_norm = float(np.linalg.norm(mn))
            instance_mean_normals[pid] = mn / (mn_norm + 1e-10) if mn_norm > 0.1 else None
            instance_types[pid] = prim["type"]
        else:
            instance_mean_normals[pid] = None
            instance_types[pid] = prim.get("type", "unknown")

    assign_radius = 0.10
    n_thresh_plane = 0.85
    n_thresh_other = 0.5

    for pass_num in range(3):
        unassigned = np.where(instance_ids == -1)[0]
        if len(unassigned) == 0:
            break
        un_pts = points[unassigned]
        un_neighbors = full_tree.query_ball_point(un_pts, r=assign_radius)
        changed = 0
        for i, un_idx in enumerate(unassigned):
            nbs = un_neighbors[i]
            if len(nbs) < 3:
                continue
            nb_labels = instance_ids[nbs]
            assigned_nbs = nb_labels[nb_labels >= 0]
            if len(assigned_nbs) < 3:
                continue
            counts = Counter(assigned_nbs.tolist())
            point_normal = normals[un_idx]
            for cand_label, cand_count in counts.most_common():
                if cand_count < max(3, len(assigned_nbs) * 0.4):
                    break
                mean_n = instance_mean_normals.get(cand_label)
                if mean_n is not None:
                    itype = instance_types.get(cand_label, "unknown")
                    thresh = n_thresh_plane if itype == "plane" else n_thresh_other
                    if abs(np.dot(point_normal, mean_n)) < thresh:
                        continue
                instance_ids[un_idx] = cand_label
                changed += 1
                break
        log(f"  pass {pass_num+1}: assigned {changed}")
        if changed == 0:
            break

    # ── Step 6: merge co-axial cylinders ──
    # Post-assignment may have added points to a primitive that aren't in
    # its stale ``global_indices`` — query ``instance_ids`` fresh so the
    # merge reassignment is total.
    log("Merging over-segmented cylinders…")
    cyl_prims = [p for p in primitives if p["type"] == "cylinder"]
    merged: set[int] = set()
    for i, p1 in enumerate(cyl_prims):
        if p1["id"] in merged:
            continue
        for j, p2 in enumerate(cyl_prims):
            if j <= i or p2["id"] in merged:
                continue
            a1 = np.array(p1["axis"]); a2 = np.array(p2["axis"])
            if abs(np.dot(a1, a2)) < cfg["merge_axis_dot"]:
                continue
            r_ratio = max(p1["radius"], p2["radius"]) / (min(p1["radius"], p2["radius"]) + 1e-6)
            if r_ratio > cfg["merge_radius_ratio"]:
                continue
            c1 = np.array(p1["center"]); c2 = np.array(p2["center"])
            diff = c2 - c1
            along = abs(np.dot(diff, a1))
            perp = float(np.linalg.norm(diff - np.dot(diff, a1) * a1))
            if perp > max(p1["radius"], p2["radius"]) * 0.5 + 0.05:
                continue
            half_len = (p1["length"] + p2["length"]) / 2
            if along > half_len + 0.3:
                continue
            if labeler_strict:
                instance_ids[p2["global_indices"]] = p1["id"]
            else:
                instance_ids[instance_ids == p2["id"]] = p1["id"]
            merged.add(p2["id"])

    primitives = [p for p in primitives if p["id"] not in merged]

    # Refresh n_points + re-fit cylinders post-merge
    for prim in primitives:
        mask = instance_ids == prim["id"]
        prim["n_points"] = int(mask.sum())
        prim["global_indices"] = np.where(mask)[0]
        if prim["type"] == "cylinder" and mask.sum() > 0:
            cyl = _fit_cylinder_to_cluster(
                points[prim["global_indices"]], normals[prim["global_indices"]],
                axis_hint=np.array(prim["axis"]),
                distance_threshold=cfg["cyl_distance_threshold"])
            if cyl:
                prim["radius"] = float(cyl["radius"])
                prim["length"] = float(cyl["length"])
                prim["center"] = cyl["center"].tolist()
                prim["axis"] = cyl["axis"].tolist()
                prim["inlier_ratio"] = float(cyl["inlier_ratio"])
                prim["label"] = _classify_cylinder(cyl["radius"], cyl["length"])

    # ── Step 7: leftover catchall ──
    # 7a. (default only) DBSCAN over unassigned points — each blob ≥
    #     leftover_min_pts becomes its own "leftover" segment so primitive
    #     segments don't get polluted with spatially distant outliers.
    #     Skipped under labeler_strict.
    # 7b. Snap-to-nearest for anything still at -1. Runs unconditionally so
    #     the labeler-strict path also achieves full coverage; the snap is
    #     the smallest possible deviation from bit-for-bit labeler output.
    if not labeler_strict:
        unassigned = np.where(instance_ids == -1)[0]
        if unassigned.size > 0:
            leftover_eps = float(cfg["leftover_cluster_eps"])
            leftover_min = int(cfg["leftover_min_pts"])
            if leftover_eps > 0 and unassigned.size >= leftover_min:
                import open3d as o3d
                un_pcd = o3d.geometry.PointCloud()
                un_pcd.points = o3d.utility.Vector3dVector(points[unassigned])
                cc_labels = np.asarray(
                    un_pcd.cluster_dbscan(eps=leftover_eps, min_points=10, print_progress=False)
                )
                next_id = int(instance_ids.max(initial=-1)) + 1
                for lbl in np.unique(cc_labels):
                    if lbl < 0:
                        continue
                    blob = unassigned[cc_labels == lbl]
                    if blob.size < leftover_min:
                        continue
                    instance_ids[blob] = next_id
                    primitives.append({
                        "id": next_id,
                        "type": "leftover",
                        "label": "curved_unclassified",
                        "n_points": int(blob.size),
                        "global_indices": blob.copy(),
                    })
                    next_id += 1
                n_blobs = int((cc_labels >= 0).sum())
                log(f"Catchall: clustered {n_blobs}/{unassigned.size} stragglers into leftover segments")
    unassigned = np.where(instance_ids == -1)[0]
    if unassigned.size > 0 and (instance_ids >= 0).any():
        assigned_idx = np.where(instance_ids >= 0)[0]
        nn_tree = cKDTree(points[assigned_idx])
        _, nn = nn_tree.query(points[unassigned], k=1)
        instance_ids[unassigned] = instance_ids[assigned_idx[nn]]
        log(f"Catchall: snapped {unassigned.size} stragglers to nearest segment")

    # If the entire pipeline produced nothing (degenerate cloud), fall back
    # to a single bucket so the UI still has something to show.
    if (instance_ids >= 0).sum() == 0:
        instance_ids[:] = 0
        primitives.append({
            "id": 0, "type": "unknown", "label": "curved_unclassified",
            "n_points": int(n_total), "global_indices": np.arange(n_total),
        })

    # Refresh n_points after catchall absorbed stragglers.
    for prim in primitives:
        mask = instance_ids == prim["id"]
        prim["n_points"] = int(mask.sum())
        prim["global_indices"] = np.where(mask)[0]

    # Un-shift any absolute positions back to the input frame.
    if np.any(recenter):
        for prim in primitives:
            if "center" in prim:
                prim["center"] = (np.array(prim["center"]) + recenter).tolist()
            if "d" in prim:
                # Plane equation a·x+b·y+c·z+d=0 in the recentered frame;
                # when shifting x→x+r, d_orig = d - n·r.
                n = np.array(prim["normal"])
                prim["d"] = float(prim["d"] - float(np.dot(n, recenter)))

    # Strip arrays + assign class_ids
    summary: list[dict] = []
    for prim in primitives:
        s = {k: v for k, v in prim.items() if k != "global_indices"}
        s["class_id"] = _label_to_class_id(s.get("label", ""), class_map)
        summary.append(s)

    log(f"Done: {len(summary)} segments, "
        f"{int((instance_ids >= 0).sum())}/{n_total} points assigned")

    if features is not None and feature_seen is not None:
        if features.shape[0] != n_total or feature_seen.shape[0] != n_total:
            log(f"  WARNING: feature array length mismatch "
                f"(features={features.shape[0]}, seen={feature_seen.shape[0]}, "
                f"N={n_total}) — skipping feature split")
        else:
            instance_ids = _feature_split_instances(
                instance_ids, features, feature_seen,
                min_size=int(feature_split_min_size),
                target_size=int(feature_split_target_size),
                max_k=int(feature_split_max_k),
                log=log,
            )
            # Rebuild summary stubs for new sub-cluster ids so n_segments
            # reflects the split. Existing primitive entries keep their id;
            # we append minimal stubs for the newly-allocated ones.
            existing_ids = {s.get("id") for s in summary if "id" in s}
            max_id = int(instance_ids.max())
            for new_id in range(max_id + 1):
                if new_id in existing_ids:
                    continue
                summary.append({
                    "id": int(new_id),
                    "class_id": -1,
                    "label": "sam3_subcluster",
                    "n_points": int((instance_ids == new_id).sum()),
                })

    return instance_ids, summary
