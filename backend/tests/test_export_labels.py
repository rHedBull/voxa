"""TDD for ExportLabelsRequest schema + validate_export_request (export-wizard Phase B, Task 1)."""
import numpy as np
import pytest
from pydantic import ValidationError

from app.schemas import ExportLabelsRequest
from labeling.export_pipeline import (
    apply_filters_remap,
    build_taxonomy,
    count_absent_instances,
    validate_export_request,
)


def _base_req(**overrides):
    payload = {
        "scene": "annotated/foo",
        "session_id": "s1",
        "resolution": {"kind": "scan"},
        "confirmed_only": False,
        "include_classes": None,
        "remap": [],
        "drop_unlabeled": False,
    }
    payload.update(overrides)
    return ExportLabelsRequest(**payload)


def test_parses_from_alias():
    req = ExportLabelsRequest(**{
        "scene": "annotated/foo",
        "session_id": "s1",
        "resolution": {"kind": "scan"},
        "remap": [{"from": [1, 2], "to": {"id": 9, "label": "merged", "color": "#fff"}}],
    })
    assert req.remap[0].from_ == [1, 2]
    assert req.remap[0].to.id == 9
    assert req.remap[0].to.label == "merged"
    assert req.remap[0].to.color == "#fff"


def test_remap_missing_to_id_rejected_at_parse():
    with pytest.raises(ValidationError):
        ExportLabelsRequest(**{
            "scene": "annotated/foo",
            "session_id": "s1",
            "resolution": {"kind": "scan"},
            "remap": [{"from": [1], "to": {"label": "a", "color": "#fff"}}],
        })


def test_valid_request_has_no_errors():
    req = _base_req(remap=[{"from": [1], "to": {"id": 9, "label": "merged", "color": "#fff"}}])
    errors = validate_export_request(req, n_scan=1000, palette_ids={0, 1, 2, 3}, raw_available=True)
    assert errors == []


def test_from_id_not_in_palette():
    req = _base_req(remap=[{"from": [99], "to": {"id": 9, "label": "merged", "color": "#fff"}}])
    errors = validate_export_request(req, n_scan=1000, palette_ids={0, 1, 2, 3}, raw_available=True)
    assert any("99" in e for e in errors)


def test_overlapping_from_sets():
    req = _base_req(remap=[
        {"from": [1, 2], "to": {"id": 9, "label": "a", "color": "#fff"}},
        {"from": [2, 3], "to": {"id": 10, "label": "b", "color": "#000"}},
    ])
    errors = validate_export_request(req, n_scan=1000, palette_ids={0, 1, 2, 3}, raw_available=True)
    assert any("overlap" in e.lower() for e in errors)


def test_same_to_id_different_label_or_color():
    req = _base_req(remap=[
        {"from": [1], "to": {"id": 9, "label": "a", "color": "#fff"}},
        {"from": [2], "to": {"id": 9, "label": "b", "color": "#fff"}},
    ])
    errors = validate_export_request(req, n_scan=1000, palette_ids={0, 1, 2, 3}, raw_available=True)
    assert any("9" in e for e in errors)


def test_to_id_collides_with_kept_through_class():
    # class 3 survives (not remapped, not excluded); remapping to id 3 collides.
    req = _base_req(remap=[{"from": [1], "to": {"id": 3, "label": "a", "color": "#fff"}}])
    errors = validate_export_request(req, n_scan=1000, palette_ids={0, 1, 2, 3}, raw_available=True)
    assert any("3" in e for e in errors)


def test_subsample_n_greater_than_n_scan():
    req = _base_req(resolution={"kind": "subsample", "n": 2000})
    errors = validate_export_request(req, n_scan=1000, palette_ids={0, 1, 2, 3}, raw_available=True)
    assert any("subsample" in e.lower() or "n_scan" in e.lower() for e in errors)


def test_subsample_n_less_than_one():
    req = _base_req(resolution={"kind": "subsample", "n": 0})
    errors = validate_export_request(req, n_scan=1000, palette_ids={0, 1, 2, 3}, raw_available=True)
    assert len(errors) == 1


def test_raw_unavailable():
    req = _base_req(resolution={"kind": "raw"})
    errors = validate_export_request(req, n_scan=1000, palette_ids={0, 1, 2, 3}, raw_available=False)
    assert any("raw" in e.lower() for e in errors)


# ---------------------------------------------------------------------------
# Task 2: build_taxonomy / apply_filters_remap / count_absent_instances
# ---------------------------------------------------------------------------

_PALETTE = [
    {"class_id": 0, "label": "unlabeled", "color": "#000"},
    {"class_id": 1, "label": "pipe", "color": "#111"},
    {"class_id": 2, "label": "tank", "color": "#222"},
]


def test_confirmed_only_zeros_unconfirmed_and_keeps_absent_as_confirmed():
    class_ids = np.array([1, 1, 2, 2], dtype=np.int64)
    instance_ids = np.array([10, 10, 20, 30], dtype=np.int64)
    # inst 10 unconfirmed -> zeroed; inst 20 confirmed -> kept;
    # inst 30 absent from the confirmed map -> treated as confirmed, kept.
    confirmed_by_inst = {10: False, 20: True}
    req = _base_req(confirmed_only=True)
    src_to_tgt = {0: 0, 1: 1, 2: 2}
    out_cls, out_inst = apply_filters_remap(class_ids, instance_ids, confirmed_by_inst, req, src_to_tgt)
    assert list(out_cls) == [-1, -1, 2, 2]
    assert list(out_inst) == list(instance_ids)


def test_exclude_zeros_class_not_in_include_classes():
    class_ids = np.array([0, 1, 2], dtype=np.int64)
    instance_ids = np.array([1, 2, 3], dtype=np.int64)
    req = _base_req(include_classes=[1, 2])
    src_to_tgt = {0: 0, 1: 1, 2: 2}
    out_cls, out_inst = apply_filters_remap(class_ids, instance_ids, {}, req, src_to_tgt)
    assert list(out_cls) == [-1, 1, 2]
    assert list(out_inst) == list(instance_ids)


def test_exclude_precedes_remap_excluded_class_never_remapped():
    # class 1 excluded via include_classes; also in a remap from-set targeting 9.
    class_ids = np.array([1, 2], dtype=np.int64)
    instance_ids = np.array([1, 2], dtype=np.int64)
    req = _base_req(
        include_classes=[2],
        remap=[{"from": [1], "to": {"id": 9, "label": "merged", "color": "#fff"}}],
    )
    src_to_tgt = {0: 0, 1: 9, 2: 2}
    out_cls, out_inst = apply_filters_remap(class_ids, instance_ids, {}, req, src_to_tgt)
    # class 1 excluded first -> -1, remap never sees it (would otherwise be 9).
    assert list(out_cls) == [-1, 2]


def test_remap_merges_two_source_classes_into_one_target():
    req = _base_req(
        remap=[{"from": [1, 2], "to": {"id": 9, "label": "merged", "color": "#fff"}}],
    )
    taxonomy, src_to_tgt = build_taxonomy(_PALETTE, req)
    assert src_to_tgt[1] == 9
    assert src_to_tgt[2] == 9
    assert taxonomy[9] == {"label": "merged", "color": "#fff"}

    class_ids = np.array([1, 2, 0], dtype=np.int64)
    instance_ids = np.array([1, 2, 3], dtype=np.int64)
    out_cls, out_inst = apply_filters_remap(class_ids, instance_ids, {}, req, src_to_tgt)
    assert list(out_cls) == [9, 9, 0]
    assert list(out_inst) == list(instance_ids)


def test_taxonomy_is_palette_driven_pass_through_class_present_even_if_absent_from_data():
    req = _base_req()
    taxonomy, src_to_tgt = build_taxonomy(_PALETTE, req)
    # class 2 is never remapped/excluded -> must appear in taxonomy regardless
    # of whether any point currently carries it.
    assert 2 in taxonomy
    assert taxonomy[2] == {"label": "tank", "color": "#222"}
    assert src_to_tgt[2] == 2


def test_excluded_class_absent_from_taxonomy():
    req = _base_req(include_classes=[0, 1])
    taxonomy, src_to_tgt = build_taxonomy(_PALETTE, req)
    assert 2 not in taxonomy


def test_count_absent_instances_counts_distinct_not_points():
    work_inst = np.array([10, 10, 10, 20, -1, -1, 30], dtype=np.int64)
    confirmed_by_inst = {20: True}
    # 10 and 30 are absent from confirmed_by_inst; -1 doesn't count.
    n = count_absent_instances(work_inst, confirmed_by_inst)
    assert n == 2


def test_empty_after_filters_yields_all_negative_one_no_error():
    class_ids = np.array([1, 1, 2], dtype=np.int64)
    instance_ids = np.array([1, 1, 2], dtype=np.int64)
    req = _base_req(include_classes=[])
    src_to_tgt = {0: 0, 1: 1, 2: 2}
    out_cls, out_inst = apply_filters_remap(class_ids, instance_ids, {}, req, src_to_tgt)
    assert list(out_cls) == [-1, -1, -1]
    assert list(out_inst) == list(instance_ids)


def test_apply_filters_remap_does_not_mutate_inputs():
    class_ids = np.array([1, 2, 0], dtype=np.int64)
    instance_ids = np.array([10, 20, 30], dtype=np.int64)
    cls_before = class_ids.copy()
    inst_before = instance_ids.copy()
    req = _base_req(
        confirmed_only=True,
        include_classes=[1, 2],
        remap=[{"from": [1, 2], "to": {"id": 9, "label": "merged", "color": "#fff"}}],
    )
    src_to_tgt = {0: 0, 1: 9, 2: 9}
    apply_filters_remap(class_ids, instance_ids, {10: False}, req, src_to_tgt)
    assert np.array_equal(class_ids, cls_before)
    assert np.array_equal(instance_ids, inst_before)


# ---------------------------------------------------------------------------
# Task 3: build_manifest
# ---------------------------------------------------------------------------

import json
from labeling.export_pipeline import build_manifest


def test_build_manifest_basic_structure():
    """Manifest from small taxonomy, verify structure and JSON serialization."""
    taxonomy = {
        0: {"label": "building", "color": "#8b5cf6"},
        1: {"label": "pipe", "color": "#22c55e"},
    }
    manifest = build_manifest(
        taxonomy=taxonomy,
        p50=0.021,
        p90=0.058,
        scan="annotated/test_scene",
        session="s_123",
        resolution={"kind": "scan"},
        points=50000,
        confirmed_only=False,
        include_classes=None,
        drop_unlabeled=False,
        absent_count=2,
        exported_at="2026-07-11T10:30:00Z",
        labeling_points=100000,
    )

    # Verify structure
    assert "classes" in manifest
    assert "accuracy" in manifest
    assert "source" in manifest
    assert "resolution" in manifest
    assert "filters" in manifest

    # Class keys are strings (JSON)
    assert "0" in manifest["classes"]
    assert manifest["classes"]["0"]["label"] == "building"
    assert manifest["classes"]["0"]["color"] == "#8b5cf6"
    assert "1" in manifest["classes"]
    assert manifest["classes"]["1"]["label"] == "pipe"

    # Accuracy: p90 is the headline uncertainty
    assert manifest["accuracy"]["sample_spacing_p50_m"] == 0.021
    assert manifest["accuracy"]["sample_spacing_p90_m"] == 0.058
    assert manifest["accuracy"]["semantic_boundary_uncertainty_m"] == 0.058
    assert manifest["accuracy"]["labeling_points"] == 100000

    # Note mentions labeling density and exact
    note = manifest["accuracy"]["note"]
    assert "labeling density" in note.lower()
    assert "exact" in note.lower()

    # Source matches passed values (not generated internally)
    assert manifest["source"]["scan"] == "annotated/test_scene"
    assert manifest["source"]["session"] == "s_123"
    assert manifest["source"]["exported_at"] == "2026-07-11T10:30:00Z"

    # Resolution and filters
    assert manifest["resolution"]["kind"] == "scan"
    assert manifest["resolution"]["points"] == 50000
    assert manifest["filters"]["confirmed_only"] is False
    assert manifest["filters"]["include_classes"] is None
    assert manifest["filters"]["drop_unlabeled"] is False
    assert manifest["filters"]["absent_instances"] == 2

    # JSON serializable (no numpy types)
    json.dumps(manifest)


def test_build_manifest_with_include_classes():
    """Manifest respects include_classes filter."""
    taxonomy = {0: {"label": "pipe", "color": "#22c55e"}}
    manifest = build_manifest(
        taxonomy=taxonomy,
        p50=0.05,
        p90=0.1,
        scan="scene_x",
        session="s_x",
        resolution={"kind": "subsample", "n": 100000},
        points=100000,
        confirmed_only=True,
        include_classes=[0, 1],
        drop_unlabeled=True,
        absent_count=0,
        exported_at="2026-07-11T11:00:00Z",
        labeling_points=50000,
    )

    assert manifest["filters"]["confirmed_only"] is True
    assert manifest["filters"]["include_classes"] == [0, 1]
    assert manifest["filters"]["drop_unlabeled"] is True
    assert manifest["resolution"]["kind"] == "subsample"


def test_build_manifest_labeling_points_optional():
    """Labeling_points can be None."""
    taxonomy = {0: {"label": "test", "color": "#fff"}}
    manifest = build_manifest(
        taxonomy=taxonomy,
        p50=0.01,
        p90=0.02,
        scan="s",
        session="x",
        resolution={"kind": "scan"},
        points=1000,
        confirmed_only=False,
        include_classes=None,
        drop_unlabeled=False,
        absent_count=0,
        exported_at="2026-07-11T00:00:00Z",
        labeling_points=None,
    )

    assert manifest["accuracy"]["labeling_points"] is None
    json.dumps(manifest)


def test_build_manifest_p50_p90_rounding():
    """Verify p50 and p90 are rounded to 4 decimals."""
    taxonomy = {0: {"label": "x", "color": "#000"}}
    manifest = build_manifest(
        taxonomy=taxonomy,
        p50=0.0123456,
        p90=0.9876543,
        scan="s",
        session="x",
        resolution={"kind": "scan"},
        points=1000,
        confirmed_only=False,
        include_classes=None,
        drop_unlabeled=False,
        absent_count=0,
        exported_at="2026-07-11T00:00:00Z",
    )

    # Rounding to 4 decimals
    assert manifest["accuracy"]["sample_spacing_p50_m"] == 0.0123
    assert manifest["accuracy"]["sample_spacing_p90_m"] == 0.9877  # rounds up
    assert manifest["accuracy"]["semantic_boundary_uncertainty_m"] == 0.9877
