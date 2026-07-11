"""Task 1 (materialize core): raw-cloud resolver fallback via
derivation -> raw/sources.json, and the raw_source_available flag on
/api/load.

meta.source_laz is null on regenerated scans; when that's the case we
fall back to meta.derivation.root.source_id -> raw/sources.json -> path.
"""
import json
from pathlib import Path

from tests.conftest import build_annotated_root


def _add_raw_lineage(root: Path, scan_name: str, source_id: str, raw_rel: str,
                      write_raw_file: bool = True):
    """Register a raw source and point the scan's meta.json derivation at it."""
    (root / "raw").mkdir(parents=True, exist_ok=True)
    (root / "raw" / "sources.json").write_text(json.dumps({
        "sources": [{"source_id": source_id, "path": raw_rel, "format": "laz"}],
    }))
    if write_raw_file:
        raw_path = root / raw_rel
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(b"fake-laz-bytes")

    meta_path = root / "annotated" / scan_name / "meta.json"
    meta = json.loads(meta_path.read_text())
    meta["source_laz"] = None
    meta["derivation"] = {"root": {"source_id": source_id}}
    meta_path.write_text(json.dumps(meta))


def test_raw_source_available_true_via_derivation_lineage(monkeypatch, tmp_path):
    import main
    from fastapi.testclient import TestClient

    root, session_id = build_annotated_root(tmp_path)
    _add_raw_lineage(root, "demo", "demo-source", "raw/demo.laz")
    monkeypatch.setattr("app.constants.LIDAR_ROOT", root, raising=False)

    client = TestClient(main.app)
    r = client.post("/api/load", json={"name": "annotated/demo", "max_points": 100})
    assert r.status_code == 200
    assert r.json()["raw_source_available"] is True


def test_raw_source_available_false_without_lineage(monkeypatch, tmp_path):
    import main
    from fastapi.testclient import TestClient

    root, session_id = build_annotated_root(tmp_path)
    # No raw/sources.json, no derivation in meta.json -> nothing to resolve.
    monkeypatch.setattr("app.constants.LIDAR_ROOT", root, raising=False)

    client = TestClient(main.app)
    r = client.post("/api/load", json={"name": "annotated/demo", "max_points": 100})
    assert r.status_code == 200
    assert r.json()["raw_source_available"] is False


def test_raw_source_available_false_when_registered_path_missing_on_disk(monkeypatch, tmp_path):
    import main
    from fastapi.testclient import TestClient

    root, session_id = build_annotated_root(tmp_path)
    _add_raw_lineage(root, "demo", "demo-source", "raw/missing.laz", write_raw_file=False)
    monkeypatch.setattr("app.constants.LIDAR_ROOT", root, raising=False)

    client = TestClient(main.app)
    r = client.post("/api/load", json={"name": "annotated/demo", "max_points": 100})
    assert r.status_code == 200
    assert r.json()["raw_source_available"] is False
