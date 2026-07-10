"""TDD for ExportLabelsRequest schema + validate_export_request (export-wizard Phase B, Task 1)."""
import pytest
from pydantic import ValidationError

from app.schemas import ExportLabelsRequest
from labeling.export_pipeline import validate_export_request


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
