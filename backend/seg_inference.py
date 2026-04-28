"""On-load prelabel inference using the segmentation repo's tuned_merge_v4.

Voxa's lidar_io fallback chain is:

    labels/gt_*.npy         (authored ground truth, highest priority)
    prelabel/ransac_*       (cached prediction or model output)
    on-load inference       (this module — runs the merge model)
    all-(-1)                (empty editable state)

This module sits in the third slot. When a SCHEMA-conformant scene has
the 5 RANSAC artifacts under `fresh_run/segmentation/` but no `labels/`
or `prelabel/`, we load the trained `tuned_merge_v4` bundle from the
segmentation repo, run prediction, and write the result to `prelabel/`
as a side effect — so subsequent loads bypass inference entirely and
this module becomes a no-op for that scene.

Configuration (env vars, both optional):

    VOXA_SEGMENTATION_REPO  path to the feynman-research/segmentation
                            checkout (default: ~/feynman-research/segmentation).
                            sys.path.append-ed at import time so
                            `from models.tuned_merge_v4 import predict`
                            resolves.

    VOXA_MERGE_MODEL        path to the joblib bundle
                            (default: <segmentation_repo>/checkpoints/tuned_merge_v4.pkl).

If the segmentation repo isn't on disk, or the bundle is missing, this
module's `predict_for_scene` returns None and lidar_io falls through to
the all-(-1) tier — voxa stays usable, just without aided prelabels.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np

_SEGMENTATION_REPO = Path(os.environ.get(
    "VOXA_SEGMENTATION_REPO",
    str(Path.home() / "feynman-research" / "segmentation"),
))
_MODEL_PATH = Path(os.environ.get(
    "VOXA_MERGE_MODEL",
    str(_SEGMENTATION_REPO / "checkpoints" / "tuned_merge_v4.pkl"),
))


def _ensure_segmentation_on_path() -> bool:
    """Add the segmentation repo to sys.path so `models.tuned_merge_v4`
    resolves. Returns False if the repo isn't on disk."""
    if not _SEGMENTATION_REPO.exists():
        return False
    p = str(_SEGMENTATION_REPO)
    if p not in sys.path:
        sys.path.insert(0, p)
    return True


# Cache the loaded bundle across calls — joblib.load + xgboost ensemble
# unpacking is ~100ms; cheap to keep.
_bundle_cache: Optional[dict] = None
_bundle_path_cached: Optional[Path] = None


def _load_bundle_cached() -> Optional[dict]:
    global _bundle_cache, _bundle_path_cached
    if not _MODEL_PATH.exists():
        return None
    if _bundle_cache is not None and _bundle_path_cached == _MODEL_PATH:
        return _bundle_cache
    if not _ensure_segmentation_on_path():
        return None
    try:
        from models import io as model_io  # type: ignore[import-not-found]
        bundle = model_io.load(_MODEL_PATH)
    except Exception:
        return None
    _bundle_cache = bundle
    _bundle_path_cached = _MODEL_PATH
    return bundle


def _read_ransac_artifacts(scan_dir: Path) -> Optional[dict]:
    """Pull the 5 inputs `predict()` needs from `fresh_run/segmentation/`.
    Returns None if any are missing or malformed."""
    seg_dir = scan_dir / "fresh_run" / "segmentation"
    inst_p = seg_dir / "instance_labels.npy"
    surf_p = seg_dir / "surface_labels.npy"
    k1_p = seg_dir / "k1.npy"
    k2_p = seg_dir / "k2.npy"
    summary_p = seg_dir / "ransac_summary.json"
    if not all(p.exists() for p in (inst_p, surf_p, k1_p, k2_p, summary_p)):
        return None
    try:
        return {
            "inst_ids": np.load(inst_p),
            "surface_labels": np.load(surf_p),
            "k1": np.load(k1_p),
            "k2": np.load(k2_p),
            "ransac_summary": json.loads(summary_p.read_text()),
        }
    except (OSError, ValueError, json.JSONDecodeError):
        return None


_LABEL_KEYWORD_MAP: tuple[tuple[tuple[str, ...], str], ...] = (
    # (keywords-to-match, target-voxa-class-name). Order matters: more
    # specific keywords first. Voxa's class names ("Pipe", "Tank", etc.)
    # are matched case-insensitively against `class_map` keys.
    (("pipe",), "pipe"),
    (("tank", "vessel", "drum", "silo"), "tank"),
    (("flat_surface", "wall", "floor", "ceiling", "beam", "structural", "plane"), "structural"),
    (("fitting", "elbow", "joint", "flange"), "fitting"),
    (("equipment", "pump", "valve", "motor", "instrument"), "equipment"),
)


def _ransac_class_for_segment(label: str, class_map: dict[str, int]) -> int:
    """Map a RANSAC segment's free-form `label` field to a class id from
    `lidar/classes.json`. Tries: exact case-insensitive match against
    class names first, then a keyword heuristic for descriptive RANSAC
    labels like "flat_surface" or "large_pipe". Returns -1 on no match.

    Voxa's class names (per `config/classes.yaml`): Pipe, Tank, Equipment,
    Structural, Fitting, Unknown. RANSAC's label vocabulary is broader
    and more descriptive — e.g. ipl segment emits "flat_surface",
    "large_pipe", "short_pipe", "small_pipe", "tank", "fitting".
    """
    if not label:
        return -1
    key = label.lower().strip()

    # Exact case-insensitive match first.
    for name, cid in class_map.items():
        if name.lower() == key:
            return int(cid)

    # Keyword heuristic: substring containment, in priority order.
    for keywords, target in _LABEL_KEYWORD_MAP:
        if any(kw in key for kw in keywords):
            for name, cid in class_map.items():
                if name.lower() == target:
                    return int(cid)
    return -1


def predict_for_scene(scan_dir: Path, n_points: int,
                      class_map: dict[str, int],
                      *, write_cache: bool = True) -> Optional[tuple[np.ndarray, np.ndarray]]:
    """Run inference on a scene and return aligned (class_ids int8, instance_ids int32).

    `scan_dir` is the SCHEMA scan directory, e.g.
    `<lidar>/annotated/munich_water_pump/`. `n_points` must match the
    expected per-point array length. `class_map` is `{name: id}` from
    `lidar/classes.json`, used to assign per-segment class_ids from the
    RANSAC `label` field (the merge model itself only predicts instance
    boundaries, not classes).

    Returns None when:
      - the segmentation repo isn't on disk
      - the model bundle is missing or stale
      - the scene's RANSAC artifacts are missing or malformed
      - the artifacts don't have shape (n_points,)
      - prediction fails for any other reason

    On success, also writes `prelabel/ransac_instance_ids.npy` and
    `prelabel/ransac_segment_summary.json` so subsequent loads bypass
    inference (set `write_cache=False` to skip).
    """
    bundle = _load_bundle_cached()
    if bundle is None:
        return None

    artifacts = _read_ransac_artifacts(scan_dir)
    if artifacts is None:
        return None

    inst_ids = artifacts["inst_ids"].astype(np.int32, copy=False)
    if inst_ids.shape != (n_points,):
        return None

    try:
        from models import tuned_merge_v4  # type: ignore[import-not-found]
    except ImportError:
        return None

    # The merge model needs xyz too. Read it from the SCHEMA scan.
    from point_cloud import load_ply
    pc, _mesh = load_ply(scan_dir / "source" / "scan.ply")
    if len(pc) != n_points:
        return None
    xyz = pc.points

    try:
        parents = tuned_merge_v4.predict(
            xyz, inst_ids,
            artifacts["surface_labels"], artifacts["k1"], artifacts["k2"],
            artifacts["ransac_summary"],
            bundle["model"],
        )
    except Exception:
        return None

    instance_ids = tuned_merge_v4.parents_to_per_point(parents, inst_ids)

    # Per-point class: take the RANSAC segment's `label` and look up its
    # class id. Points whose RANSAC seg id isn't in the summary stay at -1.
    seg_to_label = {s["id"]: s.get("label", "") for s in artifacts["ransac_summary"]}
    class_ids = np.full(n_points, -1, dtype=np.int8)
    for seg_id, lbl in seg_to_label.items():
        cid = _ransac_class_for_segment(lbl, class_map)
        if cid >= 0:
            class_ids[inst_ids == seg_id] = cid

    # SCHEMA invariant 3: class==-1 ⟺ instance==-1. The merge model can
    # produce a positive instance for a point whose RANSAC label maps to
    # no class (e.g. "unknown") — drop those to -1 to keep invariants.
    unl = class_ids == -1
    instance_ids = instance_ids.copy()
    instance_ids[unl] = -1

    if write_cache:
        _write_prelabel_cache(scan_dir, instance_ids, artifacts["ransac_summary"], class_map)

    return class_ids, instance_ids


def _write_prelabel_cache(scan_dir: Path, instance_ids: np.ndarray,
                          ransac_summary: list[dict],
                          class_map: dict[str, int]) -> None:
    """Write SCHEMA-shaped prelabel/ artifacts so the next load skips inference."""
    pre = scan_dir / "prelabel"
    pre.mkdir(parents=True, exist_ok=True)
    np.save(pre / "ransac_instance_ids.npy", instance_ids.astype(np.int32))

    # The summary needs a class_id per segment so segment_io.load_prelabel
    # can reconstruct (class, instance) without consulting the model again.
    # We compute class_id from the original RANSAC `label` field.
    out_segments = []
    for s in ransac_summary:
        cid = _ransac_class_for_segment(s.get("label", ""), class_map)
        out_segments.append({"id": int(s["id"]), "class_id": int(cid),
                             "label": s.get("label", "")})
    (pre / "ransac_segment_summary.json").write_text(
        json.dumps({"segments": out_segments}, indent=2)
    )
