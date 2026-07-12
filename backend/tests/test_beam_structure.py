"""Tests for labeling/beams.py — structure.json persistence."""
from __future__ import annotations

import json

import pytest

from labeling.beams import STRUCTURE_FILENAME, load_structure, save_structure


DOC = {
    "nodes": [{"id": 1, "pos": [0.0, 0.0, 0.0]}, {"id": 2, "pos": [2.0, 0.0, 0.0]}],
    "edges": [{"id": 3, "a": 1, "b": 2, "width": 0.2, "class_id": 10,
               "instance_id": None, "dirty": True}],
    "committed_beams": [{"a": [0.0, 0.0, 0.0], "b": [0.0, 2.0, 0.0], "width": 0.3,
                         "class_id": 11, "instance_id": 42}],
}


def test_load_missing_returns_empty(tmp_path):
    assert load_structure(tmp_path) == {"nodes": [], "edges": [], "committed_beams": []}


def test_save_load_round_trip(tmp_path):
    save_structure(tmp_path, DOC)
    assert (tmp_path / STRUCTURE_FILENAME).exists()
    assert load_structure(tmp_path) == DOC


def test_load_malformed_raises(tmp_path):
    (tmp_path / STRUCTURE_FILENAME).write_text(json.dumps({"nodes": []}))
    with pytest.raises(ValueError, match="malformed structure.json"):
        load_structure(tmp_path)
