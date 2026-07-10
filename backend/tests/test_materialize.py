import numpy as np
from labeling.materialize import materialize_downsample, collect_volumes


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
