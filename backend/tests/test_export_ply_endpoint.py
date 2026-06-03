"""Integration tests for /api/edit/export-ply at native source density.

Voxa caps the in-memory cloud at VOXA_MAX_POINTS for the viewer, but the
export endpoint must work from the *source file* so the saved PLY has
every point inside the slice — not just what was subsampled for display.
"""
from __future__ import annotations

import struct
import io
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from plyfile import PlyData


def _decode_ply_bytes(data: bytes) -> np.ndarray:
    pd = PlyData.read(io.BytesIO(data))
    v = pd['vertex']
    return np.column_stack([v['x'], v['y'], v['z']]).astype(np.float64)


def test_export_ply_returns_full_source_density(client_with_annotated_scene, monkeypatch):
    """Load with max_points=2 (heavy subsample). Source has 8 points; the
    full-density export must return all 8, not just 2."""
    client, scene_id, session_id = client_with_annotated_scene
    r = client.post('/api/load', json={'name': scene_id, 'max_points': 2})
    assert r.status_code == 200
    payload = r.json()
    assert payload['num_subsampled'] == 2
    assert payload['num_points'] == 8

    # No ops → keep everything.
    r = client.post('/api/edit/export-ply', json={'scene': scene_id, 'ops': []})
    assert r.status_code == 200, r.text
    xyz = _decode_ply_bytes(r.content)
    assert len(xyz) == 8


def test_export_ply_obb_delete_drops_points(client_with_annotated_scene):
    """An OBB delete op in display frame must reduce the kept count."""
    client, scene_id, session_id = client_with_annotated_scene
    r = client.post('/api/load', json={'name': scene_id, 'max_points': 8})
    assert r.status_code == 200

    # A box covering the whole world deletes everything.
    big_box = {
        'op': 'delete',
        'center': [0.0, 0.0, 0.0],
        'size': [1000.0, 1000.0, 1000.0],
        'rotation': [0.0, 0.0, 0.0],
    }
    r = client.post('/api/edit/export-ply',
                    json={'scene': scene_id, 'ops': [big_box]})
    assert r.status_code == 200
    xyz = _decode_ply_bytes(r.content)
    assert len(xyz) == 0


def test_export_ply_scene_mismatch_409(client_with_annotated_scene):
    client, scene_id, session_id = client_with_annotated_scene
    r = client.post('/api/load', json={'name': scene_id, 'max_points': 2})
    assert r.status_code == 200

    r = client.post('/api/edit/export-ply',
                    json={'scene': 'annotated/wrong', 'ops': []})
    assert r.status_code == 409
