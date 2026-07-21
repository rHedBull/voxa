"""Cuboid metadata fields (phase 0 spec §4): inert pass-through + validation."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.schemas import Cuboid


def _base(**kw):
    return Cuboid(id="i1", cls="elbow", **kw)


def test_metadata_defaults():
    c = _base()
    assert c.flags == [] and c.subtype is None
    assert c.insulated is None and c.note == ""


def test_metadata_roundtrip():
    c = _base(flags=["boundary_uncertain", "incomplete"],
              subtype="ball valve", insulated=True, note="jacket transition W side")
    d = c.model_dump()
    assert Cuboid(**d) == c


def test_unknown_flag_rejected():
    with pytest.raises(ValidationError):
        _base(flags=["completely_made_up"])


def test_metadata_survives_annotations_store(client):
    # End-to-end through PUT/GET /api/annotations — catches a save path that
    # drops unknown fields (the metadata must ride instances_gt.json).
    scene = "meta-roundtrip"
    inst = {
        "id": "i1", "cls": "valve", "label": "Valve 3", "color": "#34d399",
        "kind": "pointset", "segId": 7,
        "flags": ["incomplete"], "subtype": "gate valve",
        "insulated": True, "note": "far-field, sparse",
    }
    r = client.put(
        "/api/annotations/gt/" + scene,
        json={"scene": scene, "kind": "gt", "instances": [inst], "meta": {}},
    )
    assert r.status_code == 200, r.text

    out = client.get("/api/annotations/gt/" + scene).json()
    got = out["instances"][0]
    assert got["flags"] == ["incomplete"]
    assert got["subtype"] == "gate valve"
    assert got["insulated"] is True
    assert got["note"] == "far-field, sparse"
