"""classes_from_yaml (scripts/preseg/_common.py) must honor explicit ids.

The preseg scripts stamp these ids into prelabel/<id>/segment_summary.json;
if they diverge from the backend's canonical mapping, sessions seeded from a
preseg render and export the wrong classes.
"""
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts" / "preseg"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from _common import classes_from_yaml  # noqa: E402


def test_explicit_ids_win_over_position(tmp_path):
    cfg = tmp_path / "classes.yaml"
    cfg.write_text(
        "classes:\n"
        "  pipe:\n    id: 0\n"
        "  fitting:\n    id: 5\n"
        "  tank:\n    id: 1\n"
    )
    assert classes_from_yaml(cfg) == {"pipe": 0, "fitting": 5, "tank": 1}


def test_positional_fallback_without_explicit_ids(tmp_path):
    cfg = tmp_path / "classes.yaml"
    cfg.write_text("classes:\n  a: {label: A}\n  b: {label: B}\n")
    assert classes_from_yaml(cfg) == {"a": 0, "b": 1}


def test_matches_backend_mapping_on_repo_config(monkeypatch):
    repo_cfg = Path(__file__).resolve().parents[2] / "config" / "classes.yaml"
    from app import core
    monkeypatch.setattr(core, "CONFIG_PATH", repo_cfg)
    assert classes_from_yaml(repo_cfg) == core._voxa_class_name_to_id()
