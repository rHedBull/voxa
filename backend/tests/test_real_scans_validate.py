"""Conformance gate: every scan in the lidar archive must pass scan_schema.

Delegates to the package's whole-archive validator (the single schema
definition). 2.x scans are grandfathered (frame/derivation are warnings), so
this asserts only that no scan has hard ERRORS. Skips cleanly when the archive
isn't present (CI without data).
"""
import os
import pathlib

import pytest

from scan_schema.validate import validate_archive

_LIDAR_ROOT = pathlib.Path(
    os.environ.get("VOXA_LIDAR_ROOT", "/home/hendrik/coding/engine/data/lidar"))
_ANNOTATED = _LIDAR_ROOT / "annotated"

_REPORT = validate_archive(_LIDAR_ROOT) if _ANNOTATED.exists() else {}


@pytest.mark.skipif(not _REPORT, reason="no scans in the lidar archive")
@pytest.mark.parametrize("scan_name", sorted(_REPORT), ids=lambda n: n)
def test_real_scan_has_no_errors(scan_name):
    errors = _REPORT[scan_name]["errors"]
    assert errors == [], f"{scan_name}: " + "; ".join(errors)
