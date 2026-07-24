import json
from labeling.instances_doc import load_instances_for_invariants
from app.schemas import Cuboid


def test_load_instances_for_invariants(tmp_path):
    session_dir = tmp_path / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    doc = {
        "scene": "x", "kind": "gt",
        "instances": [
            {"id": "a", "kind": "pointset", "segId": 5, "cls": "pipe", "confirmed": True},
            {"id": "b", "kind": "pointset", "segId": 7, "cls": None, "confirmed": False},
        ],
    }
    (session_dir / "instances_gt.json").write_text(json.dumps(doc))
    result = load_instances_for_invariants(session_dir)
    assert result == {
        5: {"class_id": "pipe", "confirmed": True},
        7: {"class_id": None, "confirmed": False},
    }


def test_load_instances_for_invariants_missing_file(tmp_path):
    session_dir = tmp_path / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    assert load_instances_for_invariants(session_dir) == {}


def test_load_instances_for_invariants_skips_non_pointset_kind(tmp_path):
    """A legacy kind:'cuboid' row (display-only, no gizmo) must not be
    surfaced to the invariant gate even though it carries a segId."""
    session_dir = tmp_path / "sessions" / "s1"
    session_dir.mkdir(parents=True)
    doc = {
        "scene": "x", "kind": "gt",
        "instances": [
            {"id": "a", "kind": "pointset", "segId": 5, "cls": "pipe", "confirmed": True},
            {"id": "b", "kind": "cuboid", "segId": 7, "cls": "tank", "confirmed": True},
        ],
    }
    (session_dir / "instances_gt.json").write_text(json.dumps(doc))
    result = load_instances_for_invariants(session_dir)
    assert result == {5: {"class_id": "pipe", "confirmed": True}}


def test_classless_row_confirmed_is_normalized_to_false(tmp_path):
    doc = {"instances": [
        {"segId": 5, "kind": "pointset", "cls": None, "confirmed": True},   # blob
        {"segId": 6, "kind": "pointset", "cls": 2, "confirmed": True},      # real
    ]}
    (tmp_path / "instances_gt.json").write_text(json.dumps(doc))
    got = load_instances_for_invariants(tmp_path)
    assert got[5]["confirmed"] is False   # class-less → normalized off
    assert got[6]["confirmed"] is True    # classed → untouched
    assert got[5]["class_id"] is None


def test_expected_keys_are_cuboid_fields():
    """instances_doc hand-parses instances_gt.json rows instead of importing
    Cuboid (see module docstring), so nothing pins its field expectations
    (kind/segId/cls/confirmed) to Cuboid's actual fields — except this test.
    If Cuboid ever renames one of these, this test is the tripwire."""
    expected_keys = {"kind", "segId", "cls", "confirmed"}
    assert expected_keys <= set(Cuboid.model_fields)
