import numpy as np
from labeling.materialize import (
    materialize_downsample,
    collect_volumes,
    build_replay_index,
    replay_labels,
    materialize_raw,
    raw_sample_spacing,
    materialize,
    MaterializeCtx,
)


def _cloud(n):
    rng = np.random.default_rng(0)
    pos = rng.random((n, 3)).astype(np.float32)
    col = (rng.random((n, 3)) * 255).astype(np.uint8)
    cls = rng.integers(-1, 4, n).astype(np.int8)
    inst = rng.integers(-1, 10, n).astype(np.int32)
    return pos, col, cls, inst


def test_downsample_identity():
    pos, col, cls, inst = _cloud(100)
    o = materialize_downsample(pos, col, cls, inst, n=100)
    assert np.array_equal(o[2], cls) and np.array_equal(o[3], inst)
    assert len(o[0]) == 100


def test_downsample_indexes_labels():
    pos, col, cls, inst = _cloud(100)
    p2, c2, cl2, in2 = materialize_downsample(pos, col, cls, inst, n=30)
    assert len(p2) == 30
    from app.core import _subsample_indices
    idx = _subsample_indices(100, 30)
    assert np.array_equal(cl2, cls[idx]) and np.array_equal(in2, inst[idx])


def test_collect_volumes_box_and_tube():
    instances = [
        {"id": "a", "source": "box", "kind": "pointset", "segId": 5, "seq": 2,
         "center": [0, 0, 0], "size": [1, 1, 1], "rotation": [0, 0, 0]},
        {"id": "b", "source": "draw", "kind": "pointset", "segId": 9, "seq": 3},
        {"id": "c", "source": "preseg", "kind": "pointset", "segId": 1, "seq": 1},
        {"id": "d", "source": "manual", "kind": "cuboid", "seq": 0,
         "center": [9, 9, 9], "size": [1, 1, 1], "rotation": [0, 0, 0]},
    ]
    centerlines = {"paths": [
        {"instance_id": 9, "points": [[0, 0, 0], [1, 0, 0]], "radius": 0.2, "smooth": False},
        {"instance_id": 9, "points": [[1, 0, 0], [1, 1, 0]], "radius": 0.2, "smooth": True},
    ]}
    vols = collect_volumes(instances, centerlines)
    kinds = {v["instance_id"]: v["kind"] for v in vols}
    assert kinds == {5: "obb", 9: "tube"}  # preseg + legacy excluded
    tube = next(v for v in vols if v["instance_id"] == 9)
    assert len(tube["paths"]) == 2 and tube["seq"] == 3  # both paths unioned
    obb = next(v for v in vols if v["instance_id"] == 5)
    assert obb["seq"] == 2 and obb["shape"]["size"] == [1, 1, 1]


def test_collect_volumes_beam_source_is_obb():
    instances = [
        {"source": "beam", "segId": 7, "seq": 3,
         "center": [1.0, 0.0, 0.0], "size": [2.0, 0.2, 0.2],
         "rotation": [0.0, 0.0, 0.5]},
        # Beam row without a persisted OBB (defensive): skipped, not crashed.
        {"source": "beam", "segId": 8, "seq": 4},
    ]
    vols = collect_volumes(instances, {"paths": []})
    assert len(vols) == 1
    assert vols[0] == {"kind": "obb", "instance_id": 7, "seq": 3,
                       "shape": {"center": [1.0, 0.0, 0.0], "size": [2.0, 0.2, 0.2],
                                 "rotation": [0.0, 0.0, 0.5]}}


# ---------------------------------------------------------------------------
# Regime-B max-seq replay rule
# ---------------------------------------------------------------------------

def _index(scan_pos, work_inst, volumes, seq_by_inst, inst_class_id):
    scan_pos = np.asarray(scan_pos, dtype=np.float32)
    work_inst = np.asarray(work_inst, dtype=np.int32)
    return build_replay_index(scan_pos, work_inst, volumes, seq_by_inst, inst_class_id)


def _obb_vol(instance_id, seq, center, size):
    return {"kind": "obb", "instance_id": instance_id, "seq": seq,
            "shape": {"center": center, "size": size, "rotation": [0, 0, 0]}}


def _tube_vol(instance_id, seq, paths):
    return {"kind": "tube", "instance_id": instance_id, "seq": seq, "paths": paths}


def test_replay_blocker_higher_seq_box_wins_over_nearer_sample():
    # A. Two overlapping boxes; target is nearest to A's sample but inside B
    # too, and B has higher seq -> B wins.
    volA = _obb_vol(10, 1, [0, 0, 0], [2, 2, 2])
    volB = _obb_vol(11, 2, [0.5, 0, 0], [2, 2, 2])
    scan_pos = [[-0.8, 0, 0], [1.2, 0, 0], [0, 0, 0], [5, 5, 5]]
    work_inst = [10, 11, 11, 20]
    seq_by_inst = {10: 1, 11: 2, 20: 0}
    inst_class_id = {10: 0, 11: 1, 20: 2}
    idx = _index(scan_pos, work_inst, [volA, volB], seq_by_inst, inst_class_id)
    cls, inst = replay_labels(idx, np.array([[-0.45, 0, 0]], dtype=np.float32))
    assert inst[0] == 11
    assert cls[0] == 1


def test_replay_box_defends_interior_not_covered_by_higher_seq_preseg():
    # B. P doesn't cover V's interior -> V defended even though seq_P > seq_V.
    volV = _obb_vol(10, 1, [0, 0, 0], [2, 2, 2])
    scan_pos = [[0, 0, 0], [1.5, 0, 0]]
    work_inst = [10, 20]
    seq_by_inst = {10: 1, 20: 5}
    inst_class_id = {10: 0, 20: 1}
    idx = _index(scan_pos, work_inst, [volV], seq_by_inst, inst_class_id)
    cls, inst = replay_labels(idx, np.array([[0.2, 0, 0]], dtype=np.float32))
    assert inst[0] == 10


def test_replay_reassigned_point_inside_box_wins_by_seq():
    # C. P DOES cover the interior sample point with higher seq -> P wins,
    # distinct from case B (box interiors don't ALWAYS win).
    volV = _obb_vol(10, 1, [0, 0, 0], [2, 2, 2])
    scan_pos = [[0, 0, 0], [0.8, 0, 0]]
    work_inst = [20, 10]
    seq_by_inst = {10: 1, 20: 5}
    inst_class_id = {10: 0, 20: 1}
    idx = _index(scan_pos, work_inst, [volV], seq_by_inst, inst_class_id)
    cls, inst = replay_labels(idx, np.array([[0.1, 0, 0]], dtype=np.float32))
    assert inst[0] == 20


def test_replay_exclusion_leak_reroutes_to_wall_not_background():
    # D. Point outside V, nearest sample is V's edge -> re-query non-vol tree
    # for the surrounding wall preseg, not -1 and not V's instance.
    volV = _obb_vol(10, 1, [0, 0, 0], [2, 2, 2])
    scan_pos = [[0.95, 0, 0], [1.3, 0, 0]]
    work_inst = [10, 20]
    seq_by_inst = {10: 1, 20: 0}
    inst_class_id = {10: 0, 20: 1}
    idx = _index(scan_pos, work_inst, [volV], seq_by_inst, inst_class_id)
    cls, inst = replay_labels(idx, np.array([[1.05, 0, 0]], dtype=np.float32))
    assert inst[0] == 20


def test_replay_legacy_cuboid_is_not_a_volume_uses_nn():
    # E. Legacy cuboid -> collect_volumes yields [] for it; transfer by NN,
    # ignoring its box geometry.
    scan_pos = [[0, 0, 0]]
    work_inst = [30]
    seq_by_inst = {30: 2}
    inst_class_id = {30: 3}
    idx = _index(scan_pos, work_inst, [], seq_by_inst, inst_class_id)
    cls, inst = replay_labels(idx, np.array([[1.5, 0, 0]], dtype=np.float32))
    assert inst[0] == 30
    assert cls[0] == 3


def test_replay_multi_path_tube_second_segment():
    # F. Draw instance with two path segments; target is near the second
    # segment only -> still resolves to the tube's instance.
    paths = [
        {"points": [[0, 0, 0], [1, 0, 0]], "radius": 0.2},
        {"points": [[1, 0, 0], [1, 1, 0]], "radius": 0.2},
    ]
    volT = _tube_vol(40, 3, paths)
    scan_pos = [[0.5, 0, 0], [9, 9, 9]]
    work_inst = [40, 50]
    seq_by_inst = {40: 3, 50: 0}
    inst_class_id = {40: 4, 50: 5}
    idx = _index(scan_pos, work_inst, [volT], seq_by_inst, inst_class_id)
    cls, inst = replay_labels(idx, np.array([[1, 0.5, 0]], dtype=np.float32))
    assert inst[0] == 40


def test_replay_all_scan_points_volume_owned_leak_resolves_to_background():
    # Guard: nonvol tree is None when every scan.ply sample is volume-owned;
    # a leak case (target outside the box, nearest sample is edge-inside)
    # must resolve to -1/-1 without raising.
    volV = _obb_vol(10, 1, [0, 0, 0], [2, 2, 2])
    scan_pos = [[0.95, 0, 0]]
    work_inst = [10]
    seq_by_inst = {10: 1}
    inst_class_id = {10: 0}
    idx = _index(scan_pos, work_inst, [volV], seq_by_inst, inst_class_id)
    assert idx.tree_nonvol is None
    cls, inst = replay_labels(idx, np.array([[1.05, 0, 0]], dtype=np.float32))
    assert inst[0] == -1
    assert cls[0] == -1


# ---------------------------------------------------------------------------
# Regime-B raw-LAZ streaming (Task 5)
# ---------------------------------------------------------------------------

def _write_tiny_las_at(path, points):
    """A minimal LAS 1.2 / point-format-3 (RGB-carrying) file with explicit
    point coordinates, so inside/outside-the-OBB membership is unambiguous."""
    import laspy

    n = len(points)
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    pts = np.asarray(points, dtype=np.float64)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    rng = np.random.default_rng(1)
    las.red = rng.integers(0, 65535, n).astype(np.uint16)
    las.green = rng.integers(0, 65535, n).astype(np.uint16)
    las.blue = rng.integers(0, 65535, n).astype(np.uint16)
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(path))


def test_materialize_raw_streams_chunks_through_replay(tmp_path):
    inside = [[0.0, 0.0, 0.0], [0.4, -0.3, 0.2], [-0.5, 0.5, 0.5]]
    outside = [[5.0, 5.0, 5.0], [6.0, 6.0, 6.0], [10.0, 0.0, 0.0]]
    points = inside + outside
    las_path = tmp_path / "raw.las"
    _write_tiny_las_at(las_path, points)

    volV = _obb_vol(10, 1, [0, 0, 0], [2, 2, 2])
    scan_pos = [[0, 0, 0], [8, 8, 8]]
    work_inst = [10, 20]
    seq_by_inst = {10: 1, 20: 0}
    inst_class_id = {10: 7, 20: 3}
    index = _index(scan_pos, work_inst, [volV], seq_by_inst, inst_class_id)

    chunks = list(materialize_raw(
        index, las_path, scene_is_z_up=False, offset=np.array([0.0, 0.0, 0.0]),
        chunk=2,
    ))

    assert len(chunks) > 1  # multiple chunks actually streamed (6 pts / chunk=2)

    all_xyz = np.concatenate([c[0] for c in chunks], axis=0)
    all_rgb = np.concatenate([c[1] for c in chunks], axis=0)
    all_cls = np.concatenate([c[2] for c in chunks], axis=0)
    all_inst = np.concatenate([c[3] for c in chunks], axis=0)

    assert len(all_xyz) == len(points)
    assert all_rgb.shape == (len(points), 3)
    assert all_rgb.dtype == np.uint8

    # Display frame == source frame here (scene_is_z_up=False, zero offset),
    # so xyz should match the LAS points directly (within LAS quantization).
    np.testing.assert_allclose(all_xyz, np.asarray(points), atol=1e-2)

    for i in range(len(inside)):
        assert all_inst[i] == 10
        assert all_cls[i] == 7
    for i in range(len(inside), len(points)):
        assert all_inst[i] == 20
        assert all_cls[i] == 3


def _write_tiny_las_no_rgb(path, points):
    """A minimal LAS 1.2 / point-format-0 file (NO color channels)."""
    import laspy

    header = laspy.LasHeader(point_format=0, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    header.offsets = np.array([0.0, 0.0, 0.0])
    las = laspy.LasData(header)
    pts = np.asarray(points, dtype=np.float64)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    path.parent.mkdir(parents=True, exist_ok=True)
    las.write(str(path))


def test_materialize_raw_colorless_las_yields_none_rgb(tmp_path):
    # A LAS without RGB must stream fine, with rgb8 None on every chunk.
    points = [[0.0, 0.0, 0.0], [0.4, 0.0, 0.0], [5.0, 5.0, 5.0]]
    las_path = tmp_path / "raw_nocolor.las"
    _write_tiny_las_no_rgb(las_path, points)

    index = _index([[0, 0, 0], [8, 8, 8]], [10, 20],
                   [_obb_vol(10, 1, [0, 0, 0], [2, 2, 2])],
                   {10: 1, 20: 0}, {10: 7, 20: 3})

    chunks = list(materialize_raw(
        index, las_path, scene_is_z_up=False, offset=np.array([0.0, 0.0, 0.0]),
        chunk=2,
    ))
    assert sum(len(c[0]) for c in chunks) == len(points)
    assert all(c[1] is None for c in chunks)   # rgb-absent path
    all_inst = np.concatenate([c[3] for c in chunks], axis=0)
    assert all_inst[0] == 10 and all_inst[2] == 20   # replay still runs


# ---------------------------------------------------------------------------
# Sample-spacing accuracy metric (Task 6)
# ---------------------------------------------------------------------------

def test_sample_spacing_p90_ge_p50_positive():
    rng = np.random.default_rng(0)
    pos = rng.random((2000, 3)).astype(np.float32)
    p50, p90 = raw_sample_spacing(pos)
    assert p50 > 0 and p90 >= p50


def test_sample_spacing_matches_direct_nn():
    # A regular grid has ~uniform spacing 1.0; p50 and p90 both ~1.0.
    xs = np.arange(10)
    ys = np.arange(10)
    zs = np.arange(10)
    grid = np.array([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32)
    p50, p90 = raw_sample_spacing(grid)
    assert abs(p50 - 1.0) < 1e-5
    assert abs(p90 - 1.0) < 0.5   # boundary points push p90 slightly, still ~1


def test_sample_spacing_tiny_cloud():
    p50, p90 = raw_sample_spacing(np.zeros((1, 3), dtype=np.float32))
    assert p50 == 0.0 and p90 == 0.0


# ---------------------------------------------------------------------------
# Top-level materialize() dispatcher (Task 7)
# ---------------------------------------------------------------------------

def _simple_ctx():
    pos, col, cls, inst = _cloud(50)
    return MaterializeCtx(
        scan_pos=pos, colors=col, work_cls=cls, work_inst=inst,
        volumes=[], seq_by_inst={}, inst_class_id={},
        raw_path=None, scene_is_z_up=False, offset=np.array([0.0, 0.0, 0.0]),
    )


def test_materialize_scan_regime_matches_working_arrays():
    ctx = _simple_ctx()
    positions, colors, class_ids, instance_ids, meta = materialize(ctx, {"kind": "scan"})
    assert len(positions) == len(ctx.scan_pos)
    assert np.array_equal(class_ids, ctx.work_cls)
    assert np.array_equal(instance_ids, ctx.work_inst)
    assert meta["accuracy"]["p90"] >= meta["accuracy"]["p50"] >= 0
    assert meta["points"] == len(ctx.scan_pos)


def test_materialize_scan_regime_returns_defensive_copy():
    ctx = _simple_ctx()
    _, _, class_ids, instance_ids, _ = materialize(ctx, {"kind": "scan"})
    class_ids[0] = 99
    instance_ids[0] = 999
    assert ctx.work_cls[0] != 99
    assert ctx.work_inst[0] != 999


def test_materialize_raw_without_raw_path_raises_clearly():
    import pytest
    ctx = _simple_ctx()  # raw_path=None
    with pytest.raises(ValueError, match="raw_path is None"):
        materialize(ctx, {"kind": "raw"})


def test_materialize_subsample_returns_exact_count():
    ctx = _simple_ctx()
    k = 10
    positions, colors, class_ids, instance_ids, meta = materialize(
        ctx, {"kind": "subsample", "n": k})
    assert len(positions) == k
    assert len(colors) == k
    assert len(class_ids) == k
    assert len(instance_ids) == k
    assert meta["points"] == k


def test_materialize_raw_regime_replays_onto_las(tmp_path):
    inside = [[0.0, 0.0, 0.0], [0.4, -0.3, 0.2]]
    outside = [[5.0, 5.0, 5.0], [6.0, 6.0, 6.0]]
    points = inside + outside
    las_path = tmp_path / "raw.las"
    _write_tiny_las_at(las_path, points)

    volV = _obb_vol(10, 1, [0, 0, 0], [2, 2, 2])
    scan_pos = np.array([[0, 0, 0], [8, 8, 8]], dtype=np.float32)
    col = np.zeros((2, 3), dtype=np.uint8)
    work_cls = np.array([7, 3], dtype=np.int8)
    work_inst = np.array([10, 20], dtype=np.int32)
    ctx = MaterializeCtx(
        scan_pos=scan_pos, colors=col, work_cls=work_cls, work_inst=work_inst,
        volumes=[volV], seq_by_inst={10: 1, 20: 0}, inst_class_id={10: 7, 20: 3},
        raw_path=las_path, scene_is_z_up=False, offset=np.array([0.0, 0.0, 0.0]),
    )

    positions, colors, class_ids, instance_ids, meta = materialize(ctx, {"kind": "raw"})

    assert len(positions) == len(points)
    assert colors.shape == (len(points), 3)
    assert "accuracy" in meta
    for i in range(len(inside)):
        assert instance_ids[i] == 10
        assert class_ids[i] == 7
    for i in range(len(inside), len(points)):
        assert instance_ids[i] == 20
        assert class_ids[i] == 3


def test_replay_multi_axis_rotated_box_labels_exactly_the_drawn_volume():
    # The confirmed-label guarantee end-to-end: a box the user drew (Three.js
    # Euler XYZ = Rx@Ry@Rz, pinned in test_reproject) must label exactly its
    # interior when replayed at a different density — apply-time selection
    # (shape_indices) and export-time replay (build_replay_index/replay_labels)
    # go through the same obb_indices, and both must match the drawn volume.
    from scenes.reproject import euler_xyz_matrix
    from labeling.shapes import shape_indices

    rot = [0.4, -0.7, 0.3]
    shape = {"type": "obb", "center": [0.0, 0.0, 0.0],
             "size": [1.2, 0.7, 0.4], "rotation": rot}

    rng = np.random.default_rng(1)
    scan_pos = rng.uniform(-1, 1, (500, 3)).astype(np.float32)
    raw_pos = rng.uniform(-1, 1, (5000, 3)).astype(np.float32)

    # Apply at scan resolution exactly as POST /api/segment/apply-shape does.
    inside_scan = shape_indices(scan_pos.reshape(-1), shape)
    work_inst = np.full(500, -1, dtype=np.int32)
    work_inst[inside_scan] = 7

    vol = {"kind": "obb", "instance_id": 7, "seq": 1,
           "shape": {k: shape[k] for k in ("center", "size", "rotation")}}
    idx = _index(scan_pos, work_inst, [vol], {7: 1}, {7: 3})
    cls, inst = replay_labels(idx, raw_pos)

    # Independent reference: containment under the pinned Rx@Ry@Rz basis.
    R = euler_xyz_matrix(*rot)
    local = raw_pos.astype(np.float64) @ R
    expected_inside = np.all(np.abs(local) <= np.array([0.6, 0.35, 0.2]), axis=1)
    assert expected_inside.any() and not expected_inside.all()  # non-trivial
    np.testing.assert_array_equal(inst == 7, expected_inside)
    assert (cls[expected_inside] == 3).all()


def test_replay_skew_beam_labels_exactly_the_swept_volume():
    # Same end-to-end guarantee as the box test above, for a beam: the
    # frame->euler construction (frontend beam-graph.js, mirrored in
    # test_shapes._beam_frame / _euler_xyz_from_basis) must select exactly
    # its swept box on apply (shape_indices) and on replay at a denser,
    # foreign density (build_replay_index/replay_labels) — both go through
    # the same obb_indices as the box path, since a beam is just an OBB.
    from tests.test_shapes import _beam_frame, _euler_xyz_from_basis
    from labeling.shapes import shape_indices

    a = np.array([0.2, 0.1, -0.3])
    b = np.array([1.8, 1.2, 0.9])
    width = 0.4
    u, v, w = _beam_frame(a, b)
    length = np.linalg.norm(b - a)

    shape = {
        "type": "obb",
        "center": ((a + b) / 2).tolist(),
        "size": [float(length), width, width],
        "rotation": _euler_xyz_from_basis(u, v, w),
    }

    # collect_volumes picks up the beam instance row as an "obb" volume.
    instances = [{"id": "beam-1", "source": "beam", "kind": "pointset",
                  "segId": 11, "seq": 4,
                  "center": shape["center"], "size": shape["size"],
                  "rotation": shape["rotation"]}]
    vols = collect_volumes(instances, {"paths": []})
    assert len(vols) == 1
    assert vols[0] == {"kind": "obb", "instance_id": 11, "seq": 4,
                        "shape": {"center": shape["center"], "size": shape["size"],
                                  "rotation": shape["rotation"]}}

    rng = np.random.default_rng(2)
    scan_pos = rng.uniform(-1, 3, (500, 3)).astype(np.float32)
    raw_pos = rng.uniform(-1, 3, (8000, 3)).astype(np.float32)

    # Apply at scan resolution exactly as POST /api/segment/apply-shape does.
    inside_scan = shape_indices(scan_pos.reshape(-1), shape)
    work_inst = np.full(500, -1, dtype=np.int32)
    work_inst[inside_scan] = 11

    idx = _index(scan_pos, work_inst, vols, {11: 4}, {11: 6})
    cls, inst = replay_labels(idx, raw_pos)

    # Independent reference: containment straight from the beam frame, the
    # spec's own containment definition (mirrors test_shapes' ground truth).
    rel = raw_pos.astype(np.float64) - a
    du, dv, dw = rel @ u, rel @ v, rel @ w
    # float32 points on float64 boundaries: the ground truth above uses the raw
    # frame directly, while the code path round-trips frame -> euler -> matrix in
    # float, so boundary points may legitimately flip between the two. Only
    # strict inside/outside (outside this epsilon shell) is asserted (mirrors
    # test_shapes.py).
    margin = 1e-5
    strict_inside = (du >= margin) & (du <= length - margin) \
        & (np.abs(dv) <= width / 2 - margin) & (np.abs(dw) <= width / 2 - margin)
    strict_outside = (du < -margin) | (du > length + margin) \
        | (np.abs(dv) > width / 2 + margin) | (np.abs(dw) > width / 2 + margin)

    assert strict_inside.any() and strict_outside.any()  # non-trivial split
    assert (inst[strict_inside] == 11).all()
    assert (cls[strict_inside] == 6).all()
    assert (inst[strict_outside] != 11).all()
