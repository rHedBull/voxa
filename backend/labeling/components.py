"""Fragment components within an instance (eval-labeling phase 2).

The instance is the physical object, never the point-cloud connected
component: a pipe run occluded by a column is ONE instance in two pieces.
Instance metrics need both readings — object-level (did the model link the
pieces?) and fragment-level (did it segment each piece?) — so component
membership is stored per point.

Derived, never labeled: labelers link fragments by merging instances;
geometry decides what is one connected piece. See
docs/superpowers/specs/2026-07-22-point-categories-components-design.md §4.
"""
from __future__ import annotations

import numpy as np

# 5 cm — the upstream spec's instance size floor. Recorded in
# gt_segment_metadata.json on save so a change is visible, not silent.
LINK_RADIUS_M = 0.05

_INT16_MAX = 32767
_KEY_LIMIT = 1 << 62      # int64 headroom guard for the cell-key encoding

# The forward half of the 26-neighbourhood: every neighbouring pair is found
# once, and the graph is symmetrized by connected_components(directed=False).
_OFFSETS = np.asarray(
    [(dx, dy, dz)
     for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
     if (dx, dy, dz) > (0, 0, 0)],
    dtype=np.int64,
)


def component_ids(positions, instance_ids, link_radius: float = LINK_RADIUS_M) -> np.ndarray:
    """Per-point component index WITHIN its instance (0-based per instance),
    -1 for points with instance_id < 0.

    Connectivity rule: two points of the same instance are linked iff they
    share a cell or occupy neighbouring cells of a `link_radius` grid
    (26-connectivity) — a coarser, deterministic stand-in for a metric ball.
    A KD-tree pair query at 5 cm over a million-point instance would generate
    pair counts in the 10^8+ range; this is O(n log n) and density-independent.

    Component numbering follows first appearance in point order, so a given
    pair of arrays always yields the same numbering.
    """
    if not float(link_radius) > 0:
        raise ValueError(f"link_radius must be > 0 (got {link_radius})")
    positions = np.asarray(positions, dtype=np.float64).reshape(-1, 3)
    instance_ids = np.asarray(instance_ids, dtype=np.int32)
    out = np.full(instance_ids.shape[0], -1, dtype=np.int16)
    labeled = np.flatnonzero(instance_ids >= 0)
    if labeled.size == 0:
        return out

    inst_vals, inst_idx = np.unique(instance_ids[labeled], return_inverse=True)
    inst_idx = inst_idx.ravel().astype(np.int64)

    # Cell grid, rebased so coordinates sit in [1, R+1] inside padded [0, R+2]
    # dims: a ±1 neighbour step can then never wrap into another row/instance.
    cells = np.floor(positions[labeled] / float(link_radius)).astype(np.int64)
    cells -= cells.min(axis=0) - 1
    dims = cells.max(axis=0) + 2
    stride_y, stride_x = int(dims[2]), int(dims[2] * dims[1])
    stride_inst = stride_x * int(dims[0])
    if inst_vals.size * stride_inst >= _KEY_LIMIT:
        raise ValueError(
            "cell-key space overflows int64 — the cloud extent divided by "
            f"link_radius={link_radius} m is too large")
    keys = (inst_idx * stride_inst + cells[:, 0] * stride_x
            + cells[:, 1] * stride_y + cells[:, 2])

    uniq_keys, cell_of_point = np.unique(keys, return_inverse=True)
    cell_of_point = cell_of_point.ravel()
    rows, cols = [], []
    for dx, dy, dz in _OFFSETS.tolist():
        neighbour = uniq_keys + dx * stride_x + dy * stride_y + dz
        pos = np.clip(np.searchsorted(uniq_keys, neighbour), 0, uniq_keys.size - 1)
        hit = uniq_keys[pos] == neighbour
        if hit.any():
            rows.append(np.flatnonzero(hit))
            cols.append(pos[hit])

    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    if rows:
        r = np.concatenate(rows); c = np.concatenate(cols)
    else:
        r = c = np.empty(0, dtype=np.int64)
    graph = coo_matrix((np.ones(r.size, dtype=np.int8), (r, c)),
                       shape=(uniq_keys.size, uniq_keys.size))
    n_cell_comp, cell_comp = connected_components(graph, directed=False)
    comp_of_point = cell_comp[cell_of_point].astype(np.int64)

    # Renumber per instance, by first appearance in point order.
    combined = inst_idx * n_cell_comp + comp_of_point
    _, first_idx, inv = np.unique(combined, return_index=True, return_inverse=True)
    inv = inv.ravel()
    group_inst = inst_idx[first_idx]
    order = np.lexsort((first_idx, group_inst))
    sorted_inst = group_inst[order]
    starts = np.flatnonzero(np.concatenate(([True], sorted_inst[1:] != sorted_inst[:-1])))
    sizes = np.diff(np.append(starts, sorted_inst.size))
    within = np.arange(sorted_inst.size) - np.repeat(starts, sizes)
    if sizes.max(initial=0) > _INT16_MAX:
        worst = int(inst_vals[sorted_inst[starts[int(np.argmax(sizes))]]])
        raise ValueError(
            f"instance {worst} resolves to {int(sizes.max())} components at "
            f"link_radius={link_radius} m — past the int16 limit; the radius "
            f"is wrong for this cloud")
    rank = np.empty(sorted_inst.size, dtype=np.int64)
    rank[order] = within
    out[labeled] = rank[inv].astype(np.int16)
    return out
