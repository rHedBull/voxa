"""Compare endpoint: IoU math, threshold gating, class isolation, multi-match.

These numbers are reported directly to the user as precision/recall/F1, so a
silent regression in the matching logic = wrong model evaluation.
"""

from __future__ import annotations


def _cuboid(id_, cls="pipe", center=(0.0, 0.0, 0.0), size=(1.0, 1.0, 1.0)):
    return {
        "id": id_, "cls": cls, "label": cls.title(), "color": "#22c55e",
        "center": list(center), "size": list(size),
        "rotation": [0.0, 0.0, 0.0], "conf": 1.0, "source": "manual",
    }


def _save(client, scene, kind, instances):
    client.put(
        f"/api/annotations/{scene}/{kind}",
        json={"scene": scene, "kind": kind, "instances": instances, "meta": {}},
    )


def test_partial_overlap_iou_is_one_third(client):
    """Two unit cubes shifted by 0.5 on x: inter=0.5, union=1.5, IoU=1/3."""
    scene = "cmp-partial"
    _save(client, scene, "gt",   [_cuboid("g", center=(0.0, 0, 0))])
    _save(client, scene, "pred", [_cuboid("p", center=(0.5, 0, 0))])

    body = client.post(f"/api/compare/{scene}", json={"scene": scene, "iou_threshold": 0.3}).json()
    assert body["tp"] == 1
    assert abs(body["iou_mean"] - 1 / 3) < 0.01


def test_threshold_gate_excludes_low_iou(client):
    """Same partial-overlap pair, but threshold=0.4 — should now be FN+FP."""
    scene = "cmp-thresh"
    _save(client, scene, "gt",   [_cuboid("g", center=(0.0, 0, 0))])
    _save(client, scene, "pred", [_cuboid("p", center=(0.5, 0, 0))])

    body = client.post(f"/api/compare/{scene}", json={"scene": scene, "iou_threshold": 0.4}).json()
    assert body["tp"] == 0
    assert body["fp"] == 1
    assert body["fn"] == 1


def test_class_mismatch_never_matches(client):
    """A pipe GT and a tank pred at the same pose must NOT count as a TP."""
    scene = "cmp-class"
    _save(client, scene, "gt",   [_cuboid("g", cls="pipe")])
    _save(client, scene, "pred", [_cuboid("p", cls="tank")])

    body = client.post(f"/api/compare/{scene}", json={"scene": scene, "iou_threshold": 0.3}).json()
    assert body["tp"] == 0
    assert body["fp"] == 1
    assert body["fn"] == 1


def test_greedy_match_two_gt_three_preds(client):
    """Two GT each with a perfect-match pred, plus one orphan pred far away.
    Locks in: each GT gets matched, no double-matching, orphan = FP."""
    scene = "cmp-multi"
    gt = [_cuboid("g1", center=(0, 0, 0)), _cuboid("g2", center=(5, 5, 5))]
    pr = [_cuboid("p1", center=(0, 0, 0)),
          _cuboid("p2", center=(5, 5, 5)),
          _cuboid("p3", center=(100, 100, 100))]
    _save(client, scene, "gt", gt)
    _save(client, scene, "pred", pr)

    body = client.post(f"/api/compare/{scene}", json={"scene": scene, "iou_threshold": 0.3}).json()
    assert body["tp"] == 2
    assert body["fp"] == 1
    assert body["fn"] == 0
    assert body["precision"] == round(2 / 3, 3)
    assert body["recall"] == 1.0


def test_empty_gt_and_pred_no_division_error(client):
    """Both sides empty: precision/recall/f1 should be 0.0, not NaN."""
    scene = "cmp-empty"
    _save(client, scene, "gt", [])
    _save(client, scene, "pred", [])

    body = client.post(f"/api/compare/{scene}", json={"scene": scene, "iou_threshold": 0.3}).json()
    assert body == {**body,
                    "tp": 0, "fp": 0, "fn": 0,
                    "precision": 0.0, "recall": 0.0, "f1": 0.0,
                    "iou_mean": 0.0, "rows": [], "gt": [], "pred": []}
