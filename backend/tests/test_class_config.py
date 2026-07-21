"""classes.yaml ↔ canonical classes.json consistency (phase 0, spec §1)."""
import json
import os
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
YAML_PATH = REPO_ROOT / "config" / "classes.yaml"
CANON_PATH = Path(os.environ.get(
    "VOXA_CANON_CLASSES",
    "/home/hendrik/coding/engine/data/lidar/classes.json"))

GROUPS = {"pipe-network", "duct", "electrical", "plant-units",
          "attachments", "structure", "other", "stuff", "legacy"}


def _yaml_classes():
    raw = yaml.safe_load(YAML_PATH.read_text())
    return raw["classes"]


def test_yaml_ids_unique_and_grouped():
    classes = _yaml_classes()
    ids = [b["id"] for b in classes.values()]
    assert len(ids) == len(set(ids)), "duplicate class ids in classes.yaml"
    for name, body in classes.items():
        assert body.get("group") in GROUPS, f"{name}: missing/unknown group"
        if body.get("group") == "legacy":
            assert body.get("frozen") is True, f"{name}: legacy must be frozen"
        else:
            assert not body.get("frozen"), f"{name}: frozen class outside legacy group"


def test_yaml_assignable_count_and_chord_keys():
    classes = _yaml_classes()
    assignable = {n: b for n, b in classes.items() if not b.get("frozen")}
    assert len(assignable) == 34
    # Within a group, chord second-keys must be unique.
    by_group = {}
    for name, body in assignable.items():
        by_group.setdefault(body["group"], []).append(str(body["key"]))
    for group, keys in by_group.items():
        assert len(keys) == len(set(keys)), f"duplicate chord keys in {group}"


@pytest.mark.skipif(not CANON_PATH.exists(),
                    reason="canonical classes.json absent (non-archive checkout)")
def test_yaml_matches_canonical_classes_json():
    canon = json.loads(CANON_PATH.read_text())
    assert canon["version"] == 7
    canon_by_id = {c["id"]: c["name"] for c in canon["classes"]}
    for name, body in _yaml_classes().items():
        assert body["id"] in canon_by_id, f"{name}: id {body['id']} not canonical"
        assert canon_by_id[body["id"]] == name, (
            f"id {body['id']}: yaml name {name!r} != canonical {canon_by_id[body['id']]!r}")
    # Every canonical id is represented in the yaml (nothing silently dropped).
    yaml_ids = {b["id"] for b in _yaml_classes().values()}
    assert yaml_ids == set(canon_by_id)


def test_api_config_carries_group_and_frozen(client):
    r = client.get("/api/config")
    assert r.status_code == 200
    classes = r.json()["classes"]
    by_id = {c["id"]: c for c in classes}
    assert by_id["elbow"]["group"] == "pipe-network"
    assert by_id["elbow"]["frozen"] is False
    assert by_id["pipe"]["group"] == "legacy"
    assert by_id["pipe"]["frozen"] is True
