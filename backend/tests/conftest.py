"""Shared pytest fixtures.

VOXA_DATA_DIR must be set before main.py is imported because main.py reads
it at module load time. Setting it here in conftest.py is safe — pytest
imports conftest before any test module.
"""

from __future__ import annotations

import os
import tempfile

import pytest

os.environ["VOXA_DATA_DIR"] = tempfile.mkdtemp(prefix="voxa-test-")


@pytest.fixture
def client():
    from fastapi.testclient import TestClient

    import main

    return TestClient(main.app)
