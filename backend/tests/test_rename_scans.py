# backend/tests/test_rename_scans.py
import json
from pathlib import Path

from scripts.scan.rename_scans import run_migration
import scan_schema


def _ply(path: Path, n: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    pts = "\n".join("0 0 0" for _ in range(n))
    path.write_text(
        f"ply\nformat ascii 1.0\nelement vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\nend_header\n" + pts + "\n"
    )


def _build_fixture(root: Path):
    """Two-scan archive: one exercises the full blast radius (factory_large);
    one exercises the prefix-substring trap (construction_site, whose old name is
    a prefix of its new name construction_site_navvis — spec §8 regression guard)."""
    raw = root / "raw"; raw.mkdir(parents=True)
    (raw / "sources.json").write_text(json.dumps({"sources": [
        {"source_id": "factory_large", "path": "raw/Factory.laz", "format": "laz",
         "fingerprint": "sha256:aa", "n_points": 5, "origin_url": None,
         "registered_at": "2026-01-01T00:00:00+00:00"},
        {"source_id": "construction_site_sample_data", "path": "raw/Construction.laz",
         "format": "laz", "fingerprint": "sha256:cc", "n_points": 5, "origin_url": None,
         "registered_at": "2026-01-01T00:00:00+00:00"},
    ]}))

    # --- second scan: prefix-substring trap (minimal — meta + ply only) ---
    c = root / "annotated" / "construction_site"
    _ply(c / "source" / "scan.ply", 5)
    (c / "meta.json").write_text(json.dumps({
        "schema_version": "3.0", "scan_name": "construction_site", "n_points": 5,
        "units": "meters", "class_map_version": 1,
        "frame": {"canonical_id": "construction_site#local",
                  "transform_to_canonical": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
        "derivation": {"scan_id": "construction_site", "variant_id": "construction_site",
                       "varies": ["density"], "role": "labeling",
                       "root": {"source_id": "construction_site_sample_data",
                                "fingerprint": "sha256:cc"},
                       "parent": {"ref": "construction_site_sample_data",
                                  "fingerprint": "sha256:cc"}},
    }))

    s = root / "annotated" / "factory_large"
    _ply(s / "source" / "scan.ply", 5)
    # NOTE: variant_id / labeling_variant are deliberately set to the OLD scan_name
    # ("factory_large") — the real-world case for 6/9 scans. The migration MUST leave
    # them untouched and the residual gate MUST NOT trip on them (deferred namespace).
    (s / "meta.json").write_text(json.dumps({
        "schema_version": "3.0", "scan_name": "factory_large", "n_points": 5,
        "units": "meters", "class_map_version": 1,
        "frame": {"canonical_id": "factory_large#local",
                  "transform_to_canonical": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]},
        "derivation": {"scan_id": "factory_large", "variant_id": "factory_large",
                       "varies": ["density"], "role": "labeling",
                       "root": {"source_id": "factory_large", "fingerprint": "sha256:aa"},
                       "parent": {"ref": "factory_large", "fingerprint": "sha256:aa"}},
    }))
    (s / "variants.json").write_text(json.dumps({
        "scan_id": "factory_large", "canonical_id": "factory_large#local",
        "labeling_variant": "factory_large",
        "variants": [{"variant_id": "factory_large", "varies": ["density"], "role": "labeling",
                      "path": str(s), "root_source_id": "factory_large",
                      "root_fingerprint": "sha256:aa", "source_fingerprint": None,
                      "source": "potree:factory_large (scan_15M.las)",
                      "transform_to_canonical": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}],
    }))
    r = s / "renders" / "upper"; r.mkdir(parents=True)
    (r / "meta.json").write_text(json.dumps({
        "run_id": "upper",
        "frame": {"canonical_id": "factory_large#local"},
        "generated_from": {"scan_id": "factory_large", "variant_id": "factory_large",
                           "source": "potree:factory_large (scan_15M.las)",
                           "source_fingerprint": "sha256:bb", "n_points": 5},
    }))
    (r / "manifest.json").write_text(json.dumps({"scene": "factory_large", "frames": []}))
    sess = s / "sessions" / "20260101-000000_blank"; sess.mkdir(parents=True)
    (sess / "session.json").write_text(json.dumps({"source_fingerprint": "x",
                                                    "preseg_fingerprint": None}))
    (sess / "instances_gt.json").write_text(json.dumps({
        "scene": "annotated/factory_large", "instances": []}))
    # stray .bak — must be left untouched
    (sess / "instances_gt.json.bak-inst1").write_text(json.dumps({
        "scene": "annotated/factory_large"}))


RENAME = {
    "scans": {"factory_large": "factory_navvis",
              "construction_site": "construction_site_navvis"},
    "sources": {"factory_large": "factory_navvis",
                "construction_site_sample_data": "construction_site_navvis"},
}


def test_dry_run_changes_nothing(tmp_path):
    _build_fixture(tmp_path)
    run_migration(tmp_path, RENAME, apply=False)
    assert (tmp_path / "annotated" / "factory_large").exists()
    assert not (tmp_path / "annotated" / "factory_navvis").exists()


def test_apply_renames_and_rewrites_all_refs(tmp_path):
    _build_fixture(tmp_path)
    run_migration(tmp_path, RENAME, apply=True)
    new = tmp_path / "annotated" / "factory_navvis"
    assert new.exists() and not (tmp_path / "annotated" / "factory_large").exists()

    meta = json.loads((new / "meta.json").read_text())
    assert meta["scan_name"] == "factory_navvis"
    assert meta["frame"]["canonical_id"] == "factory_navvis#local"
    assert meta["derivation"]["scan_id"] == "factory_navvis"
    assert meta["derivation"]["root"]["source_id"] == "factory_navvis"
    assert meta["derivation"]["parent"]["ref"] == "factory_navvis"
    # deferred namespace: retained even though it equals the OLD scan_name
    assert meta["derivation"]["variant_id"] == "factory_large"

    var = json.loads((new / "variants.json").read_text())
    assert var["scan_id"] == "factory_navvis"
    assert var["canonical_id"] == "factory_navvis#local"
    assert var["variants"][0]["path"].endswith("factory_navvis")
    assert var["variants"][0]["root_source_id"] == "factory_navvis"
    assert var["variants"][0]["source"] == "potree:factory_navvis (scan_15M.las)"
    assert var["variants"][0]["variant_id"] == "factory_large"   # untouched
    assert var["labeling_variant"] == "factory_large"            # untouched

    rmeta = json.loads((new / "renders" / "upper" / "meta.json").read_text())
    assert rmeta["frame"]["canonical_id"] == "factory_navvis#local"
    assert rmeta["generated_from"]["scan_id"] == "factory_navvis"
    assert rmeta["generated_from"]["variant_id"] == "factory_large"  # untouched
    assert rmeta["generated_from"]["source"] == "potree:factory_navvis (scan_15M.las)"
    rman = json.loads((new / "renders" / "upper" / "manifest.json").read_text())
    assert rman["scene"] == "factory_navvis"

    ig = json.loads((new / "sessions" / "20260101-000000_blank" /
                     "instances_gt.json").read_text())
    assert ig["scene"] == "annotated/factory_navvis"

    # stray .bak untouched
    bak = json.loads((new / "sessions" / "20260101-000000_blank" /
                      "instances_gt.json.bak-inst1").read_text())
    assert bak["scene"] == "annotated/factory_large"

    sources = json.loads((tmp_path / "raw" / "sources.json").read_text())
    ids = {e["source_id"] for e in sources["sources"]}
    assert ids == {"factory_navvis", "construction_site_navvis"}

    # prefix-substring trap: construction_site renamed cleanly, no self-trip,
    # and its already-correct prefix wasn't double-appended
    cnew = tmp_path / "annotated" / "construction_site_navvis"
    assert cnew.exists() and not (tmp_path / "annotated" / "construction_site").exists()
    cmeta = json.loads((cnew / "meta.json").read_text())
    assert cmeta["scan_name"] == "construction_site_navvis"
    assert cmeta["frame"]["canonical_id"] == "construction_site_navvis#local"
    assert cmeta["derivation"]["root"]["source_id"] == "construction_site_navvis"
    assert cmeta["derivation"]["variant_id"] == "construction_site"   # retained

    # postconditions: validate_archive error-free + no naming warnings
    report = scan_schema.validate_archive(tmp_path)
    assert all(r["errors"] == [] for r in report.values())
    assert not any("does not match" in w
                   for r in report.values() for w in r["warnings"])
