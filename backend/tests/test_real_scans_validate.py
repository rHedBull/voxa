"""Gate: every v1.3 scan in the lidar archive must pass the §7 invariants.

Skips cleanly when the archive isn't present (CI without data), so it acts as a
real conformance gate on machines/pipelines that do have the scans.
"""
import json
import os
import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "scripts" / "scan"))
from validate_scan import validate_scan_dir  # noqa: E402

_LIDAR_ROOT = pathlib.Path(
    os.environ.get("VOXA_LIDAR_ROOT", "/home/hendrik/coding/engine/data/lidar"))
_ANNOTATED = _LIDAR_ROOT / "annotated"


def _v13_scans():
    if not _ANNOTATED.exists():
        return []
    out = []
    for sd in sorted(_ANNOTATED.iterdir()):
        meta = sd / "meta.json"
        if not meta.exists():
            continue
        try:
            v = str(json.loads(meta.read_text()).get("schema_version", ""))
            # validate_scan_dir implements the v1.3 §7 invariants — apply it
            # only to v1.x scans. The old `>= "1.3"` string compare scooped
            # up migrated "2.0" scans and failed them against v1.3 rules.
            # A v2 conformance gate is a separate follow-up.
            if v.startswith("1.") and v >= "1.3":
                out.append(sd)
        except (json.JSONDecodeError, OSError):
            continue
    return out


_SCANS = _v13_scans()


@pytest.mark.skipif(not _SCANS, reason="no v1.3 scans in the lidar archive")
@pytest.mark.parametrize("scan_dir", _SCANS, ids=lambda p: p.name)
def test_v13_scan_passes_invariants(scan_dir):
    violations = validate_scan_dir(scan_dir)
    assert violations == [], f"{scan_dir.name}: " + "; ".join(violations)
