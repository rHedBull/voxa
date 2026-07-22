# backend/tests/test_components.py
import numpy as np
import pytest

from labeling.components import component_ids


def _grid(n, spacing, origin=(0.0, 0.0, 0.0)):
    """n^1 points along X at `spacing`, so consecutive points are linked iff
    spacing <= the link radius."""
    ox, oy, oz = origin
    return np.array([[ox + i * spacing, oy, oz] for i in range(n)], dtype=np.float32)


def test_points_outside_instances_are_minus_one():
    pos = _grid(5, 0.01)
    inst = np.full(5, -1, dtype=np.int32)
    out = component_ids(pos, inst)
    assert out.dtype == np.int16
    assert bool((out == -1).all())


def test_contiguous_instance_is_one_component():
    pos = _grid(20, 0.01)
    inst = np.zeros(20, dtype=np.int32)
    out = component_ids(pos, inst, link_radius=0.05)
    assert bool((out == 0).all())


def test_separated_clusters_split_into_two_components():
    near = _grid(10, 0.01)
    far = _grid(10, 0.01, origin=(5.0, 0.0, 0.0))
    pos = np.vstack([near, far])
    inst = np.zeros(20, dtype=np.int32)
    out = component_ids(pos, inst, link_radius=0.05)
    assert set(np.unique(out).tolist()) == {0, 1}
    assert len(set(out[:10].tolist())) == 1
    assert len(set(out[10:].tolist())) == 1
    assert out[0] != out[10]


def test_components_are_numbered_per_instance():
    # Two instances, each split into two far-apart pieces: both must use 0/1.
    pos = np.vstack([_grid(4, 0.01), _grid(4, 0.01, origin=(5, 0, 0)),
                     _grid(4, 0.01, origin=(0, 20, 0)), _grid(4, 0.01, origin=(5, 20, 0))])
    inst = np.array([0] * 8 + [7] * 8, dtype=np.int32)
    out = component_ids(pos, inst, link_radius=0.05)
    assert set(out[:8].tolist()) == {0, 1}
    assert set(out[8:].tolist()) == {0, 1}


def test_first_appearance_ordering():
    pos = np.vstack([_grid(2, 0.01, origin=(5, 0, 0)), _grid(2, 0.01)])
    inst = np.zeros(4, dtype=np.int32)
    out = component_ids(pos, inst, link_radius=0.05)
    # the component of the FIRST point is always 0
    assert out[0] == 0 and out[1] == 0
    assert out[2] == 1 and out[3] == 1


def test_diagonal_neighbour_cells_are_linked():
    # Two points in diagonally adjacent cells (26-connectivity), farther apart
    # than the radius in Euclidean terms but within one cell step.
    r = 0.05
    pos = np.array([[0.049, 0.049, 0.049], [0.051, 0.051, 0.051]], dtype=np.float32)
    inst = np.zeros(2, dtype=np.int32)
    out = component_ids(pos, inst, link_radius=r)
    assert out[0] == out[1]


def test_unlabeled_points_do_not_bridge_components():
    pos = np.vstack([_grid(3, 0.01), _grid(3, 0.01, origin=(0.03, 0, 0)),
                     _grid(3, 0.01, origin=(0.06, 0, 0))])
    inst = np.array([0, 0, 0, -1, -1, -1, 0, 0, 0], dtype=np.int32)
    out = component_ids(pos, inst, link_radius=0.02)
    assert bool((out[3:6] == -1).all())
    assert out[0] != out[6]


def test_too_many_components_raises():
    # 40k isolated points in one instance -> 40k components, past int16.
    pos = (np.arange(40000, dtype=np.float32)[:, None] * np.array([[1.0, 0, 0]],
                                                                 dtype=np.float32))
    inst = np.zeros(40000, dtype=np.int32)
    with pytest.raises(ValueError, match="component"):
        component_ids(pos, inst, link_radius=0.05)


def test_invalid_radius_raises():
    with pytest.raises(ValueError):
        component_ids(_grid(3, 0.01), np.zeros(3, dtype=np.int32), link_radius=0.0)
