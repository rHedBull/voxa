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


def test_unknown_resolution_kind_rejected():
    req = _base_req(resolution={"kind": "bogus"})
    errors = validate_export_request(req, n_scan=1000, palette_ids={0, 1, 2, 3}, raw_available=True)
    assert any("unknown resolution kind" in e.lower() for e in errors)


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


def test_drop_unlabeled_rows_drops_negatives_keeps_alignment():
    from labeling.export_pipeline import drop_unlabeled_rows
    cls = np.array([-1, 0, 2, -1, 5], dtype=np.int64)
    pos = np.arange(5 * 3, dtype=np.float32).reshape(5, 3)
    inst = np.array([9, 10, 11, 12, 13], dtype=np.int64)
    out_cls, out_pos, out_inst = drop_unlabeled_rows(cls, pos, inst)
    assert list(out_cls) == [0, 2, 5]
    assert list(out_inst) == [10, 11, 13]
    # positions stay row-aligned with the kept class ids
    assert np.array_equal(out_pos, pos[[1, 2, 4]])
    assert len(out_cls) == len(out_pos) == len(out_inst) == 3


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


# ---------------------------------------------------------------------------
# Task 4: labeled binary PLY writer (decomposed header/body for streaming)
# ---------------------------------------------------------------------------


def test_ply_labeled_round_trip_and_streaming_assembly():
    import numpy as np
    from app.core import _ply_labeled_bytes, _ply_labeled_header, _ply_labeled_chunk_bytes
    n = 5
    xyz = np.arange(n*3, dtype=np.float32).reshape(n, 3)
    rgb = (np.arange(n*3, dtype=np.uint8) % 200).reshape(n, 3)
    cls = np.array([-1, 0, 1, 2, 7], dtype=np.int8)
    inst = np.array([-1, 10, 11, 12, 40], dtype=np.int32)
    blob = _ply_labeled_bytes(xyz, rgb, cls, inst)
    # header advertises the right count + label props
    head = blob.split(b"end_header\n")[0].decode("ascii")
    assert "element vertex 5" in head
    assert "class_id" in head and "instance_id" in head
    # streaming assembly is byte-identical to the one-shot
    assert blob == _ply_labeled_header(n, True) + _ply_labeled_chunk_bytes(xyz, rgb, cls, inst)
    # parse the binary body and read back class_id / instance_id (int32)
    body = blob.split(b"end_header\n", 1)[1]
    dt = np.dtype([("xyz","<f4",3),("rgb","u1",3),("class_id","<i4"),("instance_id","<i4")])
    rec = np.frombuffer(body, dtype=dt)
    assert np.array_equal(rec["class_id"], cls.astype(np.int32))
    assert np.array_equal(rec["instance_id"], inst.astype(np.int32))
    assert np.array_equal(rec["xyz"], xyz)

def test_ply_labeled_no_color():
    import numpy as np
    from app.core import _ply_labeled_bytes
    n = 3
    xyz = np.zeros((n,3), np.float32); cls = np.zeros(n, np.int8); inst = np.zeros(n, np.int32)
    blob = _ply_labeled_bytes(xyz, None, cls, inst)
    head = blob.split(b"end_header\n")[0].decode("ascii")
    assert "red" not in head  # no color props
    body = blob.split(b"end_header\n",1)[1]
    dt = np.dtype([("xyz","<f4",3),("class_id","<i4"),("instance_id","<i4")])
    rec = np.frombuffer(body, dtype=dt)
    assert len(rec) == n


# ---------------------------------------------------------------------------
# Task 5: _build_materialize_ctx
# ---------------------------------------------------------------------------


def test_build_materialize_ctx_shapes_and_invariant(client_with_annotated_scene):
    from routes.export import _build_materialize_ctx

    client, scene_id, session_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "session_id": session_id, "max_points": 100})
    assert r.status_code == 200

    ctx, instances, confirmed_by_inst, class_id_by_inst = _build_materialize_ctx(scene_id, session_id)

    assert ctx.scan_pos.shape[0] == ctx.work_cls.shape[0] == ctx.work_inst.shape[0]

    present_ids = {int(i) for i in np.unique(ctx.work_inst) if int(i) >= 0}
    assert present_ids.issubset(set(ctx.inst_class_id.keys()))


def test_build_materialize_ctx_scene_mismatch_raises_409(client_with_annotated_scene):
    from fastapi import HTTPException
    from routes.export import _build_materialize_ctx

    client, scene_id, session_id = client_with_annotated_scene
    r = client.post("/api/load", json={"name": scene_id, "session_id": session_id, "max_points": 100})
    assert r.status_code == 200

    with pytest.raises(HTTPException) as exc_info:
        _build_materialize_ctx("wrong-scene", session_id)
    assert exc_info.value.status_code == 409


# ---------------------------------------------------------------------------
# Task 6: POST /api/labels/export
# ---------------------------------------------------------------------------

import io
import zipfile

_PLY_LABELED_DTYPE = np.dtype([
    ("xyz", "<f4", 3), ("rgb", "u1", 3), ("class_id", "<i4"), ("instance_id", "<i4"),
])


def _unzip_export(content: bytes):
    zf = zipfile.ZipFile(io.BytesIO(content))
    assert set(zf.namelist()) == {"scan_labeled.ply", "manifest.json"}
    ply_bytes = zf.read("scan_labeled.ply")
    manifest = json.loads(zf.read("manifest.json"))
    return ply_bytes, manifest


def _parse_labeled_ply(ply_bytes: bytes):
    head, body = ply_bytes.split(b"end_header\n", 1)
    head = head.decode("ascii")
    n = int([ln for ln in head.splitlines() if ln.startswith("element vertex")][0].split()[-1])
    rec = np.frombuffer(body, dtype=_PLY_LABELED_DTYPE)
    assert len(rec) == n
    return n, rec


def _load_demo(client, scene_id, session_id):
    r = client.post("/api/load", json={"name": scene_id, "session_id": session_id, "max_points": 100})
    assert r.status_code == 200


def test_export_labels_scan_resolution_roundtrip(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    _load_demo(client, scene_id, session_id)

    r = client.post("/api/labels/export", json={
        "scene": scene_id, "session_id": session_id,
        "resolution": {"kind": "scan"},
    })
    assert r.status_code == 200, r.text
    ply_bytes, manifest = _unzip_export(r.content)
    n, rec = _parse_labeled_ply(ply_bytes)
    assert n == 8  # demo scan.ply has 8 points
    assert manifest["accuracy"]["sample_spacing_p90_m"] >= manifest["accuracy"]["sample_spacing_p50_m"]
    assert manifest["resolution"]["kind"] == "scan"
    assert manifest["resolution"]["points"] == 8


def test_export_labels_include_classes_excludes(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    _load_demo(client, scene_id, session_id)

    # demo working classes seeded from preseg summary: seg0->0 (pipe), seg1->1
    # (tank), seg2/seg3->2 (equipment). Keep only class 0.
    r = client.post("/api/labels/export", json={
        "scene": scene_id, "session_id": session_id,
        "resolution": {"kind": "scan"},
        "include_classes": [0],
    })
    assert r.status_code == 200, r.text
    ply_bytes, manifest = _unzip_export(r.content)
    _, rec = _parse_labeled_ply(ply_bytes)
    present = set(int(c) for c in rec["class_id"])
    assert present.issubset({-1, 0})
    assert 1 not in present and 2 not in present
    assert set(manifest["classes"].keys()) == {"0"}


def test_export_labels_remap_merges_classes(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    _load_demo(client, scene_id, session_id)

    r = client.post("/api/labels/export", json={
        "scene": scene_id, "session_id": session_id,
        "resolution": {"kind": "scan"},
        "remap": [{"from": [1, 2], "to": {"id": 20, "label": "merged", "color": "#abcabc"}}],
    })
    assert r.status_code == 200, r.text
    ply_bytes, manifest = _unzip_export(r.content)
    _, rec = _parse_labeled_ply(ply_bytes)
    present = set(int(c) for c in rec["class_id"])
    assert 1 not in present and 2 not in present
    assert 20 in present
    assert manifest["classes"]["20"] == {"label": "merged", "color": "#abcabc"}


def test_export_labels_confirmed_only_zeros_unconfirmed(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    _load_demo(client, scene_id, session_id)

    instances = [
        {"id": "i0", "cls": "pipe", "kind": "pointset", "segId": 0, "confirmed": True},
        {"id": "i1", "cls": "tank", "kind": "pointset", "segId": 1, "confirmed": False},
        {"id": "i2", "cls": "equipment", "kind": "pointset", "segId": 2, "confirmed": True},
        {"id": "i3", "cls": "equipment", "kind": "pointset", "segId": 3, "confirmed": False},
    ]
    r = client.put(
        f"/api/annotations/gt/{scene_id}?session_id={session_id}",
        json={"scene": scene_id, "kind": "gt", "instances": instances, "meta": {}},
    )
    assert r.status_code == 200, r.text

    r = client.post("/api/labels/export", json={
        "scene": scene_id, "session_id": session_id,
        "resolution": {"kind": "scan"},
        "confirmed_only": True,
    })
    assert r.status_code == 200, r.text
    ply_bytes, manifest = _unzip_export(r.content)
    _, rec = _parse_labeled_ply(ply_bytes)
    # instance 1 (all its points) should be zeroed (unconfirmed).
    inst_arr = rec["instance_id"]
    cls_arr = rec["class_id"]
    unconfirmed_mask = np.isin(inst_arr, [1, 3])
    assert np.all(cls_arr[unconfirmed_mask] == -1)
    confirmed_mask = np.isin(inst_arr, [0, 2])
    assert np.all(cls_arr[confirmed_mask] != -1)
    assert manifest["filters"]["confirmed_only"] is True


def test_export_labels_scene_session_mismatch_409(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    _load_demo(client, scene_id, session_id)

    r = client.post("/api/labels/export", json={
        "scene": "annotated/wrong", "session_id": session_id,
        "resolution": {"kind": "scan"},
    })
    assert r.status_code == 409


def test_export_labels_empty_after_filters_no_500(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    _load_demo(client, scene_id, session_id)

    r = client.post("/api/labels/export", json={
        "scene": scene_id, "session_id": session_id,
        "resolution": {"kind": "scan"},
        "include_classes": [],
    })
    assert r.status_code == 200, r.text
    ply_bytes, manifest = _unzip_export(r.content)
    n, rec = _parse_labeled_ply(ply_bytes)
    assert n == 8
    assert np.all(rec["class_id"] == -1)


def test_export_labels_empty_after_filters_drop_unlabeled_zero_vertex(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    _load_demo(client, scene_id, session_id)

    r = client.post("/api/labels/export", json={
        "scene": scene_id, "session_id": session_id,
        "resolution": {"kind": "scan"},
        "include_classes": [],
        "drop_unlabeled": True,
    })
    assert r.status_code == 200, r.text
    ply_bytes, manifest = _unzip_export(r.content)
    n, rec = _parse_labeled_ply(ply_bytes)
    assert n == 0
    assert manifest["resolution"]["points"] == 0


def test_export_labels_raw_unavailable_422(client_with_annotated_scene):
    # The synthesized demo fixture has no linked raw LAZ/LAS source
    # (scene_registry never sets extras["source_laz_path"] for it), so
    # resolution.kind="raw" must be rejected by validate_export_request
    # rather than exercised end-to-end here. Regime B streaming itself is
    # covered by materialize.py's own tests.
    client, scene_id, session_id = client_with_annotated_scene
    _load_demo(client, scene_id, session_id)

    r = client.post("/api/labels/export", json={
        "scene": scene_id, "session_id": session_id,
        "resolution": {"kind": "raw"},
    })
    assert r.status_code == 422


def test_export_labels_unknown_kind_422_error_shape(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    _load_demo(client, scene_id, session_id)

    r = client.post("/api/labels/export", json={
        "scene": scene_id, "session_id": session_id,
        "resolution": {"kind": "bogus"},
    })
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert isinstance(detail["errors"], list)
    assert any("unknown resolution kind" in e.lower() for e in detail["errors"])


# ---------------------------------------------------------------------------
# Task 6b: GET /api/labels/accuracy
# ---------------------------------------------------------------------------

def test_labels_accuracy_returns_p50_p90(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    client.post("/api/load", json={"name": scene_id, "session_id": session_id, "max_points": 100000})
    r = client.get(f"/api/labels/accuracy?scene={scene_id}&session_id={session_id}")
    assert r.status_code == 200
    d = r.json()
    assert d["p90"] >= d["p50"] >= 0


def test_labels_accuracy_scene_mismatch_409(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    client.post("/api/load", json={"name": scene_id, "session_id": session_id, "max_points": 100000})
    r = client.get(f"/api/labels/accuracy?scene=wrong/scene&session_id={session_id}")
    assert r.status_code == 409
