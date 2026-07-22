import json
from labeling.instances_doc import load_instances_for_invariants


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
