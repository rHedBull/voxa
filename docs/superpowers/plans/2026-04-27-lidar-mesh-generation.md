# LiDAR Mesh Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an offline tool that reconstructs `source/mesh.glb` from each LAZ-sourced annotated lidar scene, so Voxa Inspect renders all six annotated scenes with the same mesh-overlay UX as `munich_water_pump`.

**Architecture:** A standalone Python package at `engine/data/lidar/scripts/` (sibling of the per-scan tooling, but at archive scope). Pure Python — no Voxa code changes. Pipeline: laspy → voxel-downsample → Open3D BPA reconstruction → per-vertex color from nearest input point → GLB via trimesh. GPU is opportunistic on prep stages (Open3D tensor API), CPU on BPA. Voxa's existing `scene_registry.py` discovers `source/mesh.glb` automatically once written.

**Tech Stack:** Python 3.11+, Open3D ≥ 0.18 (BPA, tensor ops), laspy[lazrs] ≥ 2.5 (LAZ reader), trimesh ≥ 4 (GLB writer), numpy, tqdm, pytest.

**Spec:** `docs/superpowers/specs/2026-04-27-lidar-mesh-generation-design.md`

---

## File Structure

All new code lives outside the Voxa repo, in the lidar archive:

```
engine/data/lidar/scripts/
├── __init__.py
├── build_meshes.py            ← CLI entry point
├── discovery.py               ← scene eligibility scan
├── pipeline.py                ← LAZ → GLB reconstruction
├── requirements.txt           ← script-local deps
├── README.md                  ← usage
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_discovery.py
    ├── test_pipeline.py
    └── test_cli.py
```

Each module has one responsibility:

- **`discovery.py`** — pure metadata logic. Walks `annotated/<scene>/meta.json`, applies the eligibility rule (`source_laz` set ∧ `mesh.glb` absent OR `force`), returns a list of `SceneTask` records. No I/O beyond reading meta.json.
- **`pipeline.py`** — the reconstruction pipeline. One public function `build_mesh_for_scene(task) → BuildResult`, plus a `build_mesh_from_points(points, colors_or_intensity)` core that's testable on synthetic data without touching disk.
- **`build_meshes.py`** — argparse CLI, orchestrates discovery + pipeline + summary log. Thin.
- **Tests** — `test_discovery.py` covers eligibility against a fake archive layout in `tmp_path`. `test_pipeline.py` runs the core on a synthetic sphere. `test_cli.py` runs the CLI as a subprocess against a fake archive.

The `engine/data/lidar/` repo currently has no `pyproject.toml` or `pytest.ini` — we add a minimal `pyproject.toml` in `scripts/` so the test suite runs in isolation from Voxa.

---

## Task 1: Scaffold scripts/ package, dependencies, and a smoke test

**Files:**
- Create: `engine/data/lidar/scripts/__init__.py`
- Create: `engine/data/lidar/scripts/requirements.txt`
- Create: `engine/data/lidar/scripts/pyproject.toml`
- Create: `engine/data/lidar/scripts/README.md`
- Create: `engine/data/lidar/scripts/build_meshes.py`
- Create: `engine/data/lidar/scripts/discovery.py`
- Create: `engine/data/lidar/scripts/pipeline.py`
- Create: `engine/data/lidar/scripts/tests/__init__.py`
- Create: `engine/data/lidar/scripts/tests/conftest.py`
- Create: `engine/data/lidar/scripts/tests/test_smoke.py`

- [ ] **Step 1: Create the package skeleton**

```bash
mkdir -p /home/hendrik/coding/engine/data/lidar/scripts/tests
cd /home/hendrik/coding/engine/data/lidar/scripts
touch __init__.py tests/__init__.py
```

- [ ] **Step 2: Write `requirements.txt`**

```
open3d>=0.18
laspy[lazrs]>=2.5
trimesh>=4
numpy>=1.24
tqdm>=4
```

Plus a dev-side requirement for tests; put pytest in `pyproject.toml` extras.

- [ ] **Step 3: Write `pyproject.toml`**

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "lidar-archive-scripts"
version = "0.1.0"
requires-python = ">=3.11"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["."]
markers = [
    "integration: marks integration tests against real LAZ files (deselect with -m 'not integration')",
]
```

- [ ] **Step 4: Write empty stubs for `discovery.py`, `pipeline.py`, `build_meshes.py`**

`discovery.py`:
```python
"""Scene eligibility for the mesh-build pipeline."""
from __future__ import annotations
```

`pipeline.py`:
```python
"""LAZ → reconstructed mesh GLB pipeline."""
from __future__ import annotations
```

`build_meshes.py` (executable shebang, argparse skeleton, exits 0):
```python
#!/usr/bin/env python3
"""Reconstruct source/mesh.glb for every LAZ-sourced annotated scene.

Usage:
  python build_meshes.py                    # all eligible scenes
  python build_meshes.py factory_large      # one scene
  python build_meshes.py --dry-run          # list what would run
  python build_meshes.py --force            # overwrite existing mesh.glb

See README.md for the full CLI surface.
"""
from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenes", nargs="*", help="Scene names to build (default: all eligible)")
    parser.add_argument("--archive-root", default="/home/hendrik/coding/engine/data/lidar")
    parser.add_argument("--force", action="store_true", help="Overwrite existing mesh.glb")
    parser.add_argument("--voxel", type=float, default=None, help="Override voxel size (meters)")
    parser.add_argument("--dry-run", action="store_true", help="List scenes, do not build")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU prep")
    args = parser.parse_args(argv)
    print(f"args: {args}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

Make it executable:
```bash
chmod +x build_meshes.py
```

- [ ] **Step 5: Write `tests/conftest.py`** (pytest layout)

```python
"""Shared fixtures for the mesh-build tests."""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _write_meta(scene_dir: Path, *, source_laz: str | None, source_mesh: str | None) -> None:
    """Write a minimal meta.json mirroring the lidar archive's schema."""
    scene_dir.mkdir(parents=True, exist_ok=True)
    (scene_dir / "source").mkdir(parents=True, exist_ok=True)
    (scene_dir / "meta.json").write_text(json.dumps({
        "scan_name": scene_dir.name,
        "source_laz": source_laz,
        "source_mesh": source_mesh,
        "n_points": 1000,
        "coords": "world",
        "units": "meters",
    }))


@pytest.fixture
def fake_archive(tmp_path: Path) -> Path:
    """An archive with one LAZ-sourced scene, one mesh-sourced scene,
    and one already-meshed LAZ scene (mesh.glb present)."""
    root = tmp_path / "lidar"
    (root / "laz").mkdir(parents=True)
    (root / "laz" / "alpha.laz").write_bytes(b"")  # empty placeholder
    (root / "laz" / "gamma.laz").write_bytes(b"")

    _write_meta(root / "annotated" / "alpha", source_laz="lidar/laz/alpha.laz", source_mesh=None)
    _write_meta(root / "annotated" / "beta",  source_laz=None, source_mesh="source/mesh.glb")
    _write_meta(root / "annotated" / "gamma", source_laz="lidar/laz/gamma.laz", source_mesh=None)
    (root / "annotated" / "beta"  / "source" / "mesh.glb").write_bytes(b"GLB!")
    (root / "annotated" / "gamma" / "source" / "mesh.glb").write_bytes(b"GLB!")
    return root
```

- [ ] **Step 6: Write `tests/test_smoke.py`** (TDD step: failing test)

```python
"""Smoke: package imports and CLI returns 0 on --help."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent.parent


def test_imports() -> None:
    import discovery  # noqa: F401
    import pipeline   # noqa: F401


def test_cli_help_exits_zero() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "build_meshes.py"), "--help"],
        capture_output=True, text=True,
    )
    assert result.returncode == 0
    assert "Reconstruct source/mesh.glb" in result.stdout
```

- [ ] **Step 7: Write `README.md`**

```markdown
# Lidar archive scripts

Archive-wide tooling. One script per use case.

## build_meshes.py

Reconstructs `source/mesh.glb` for annotated scenes whose `meta.json::source_laz`
is set and whose `source/mesh.glb` does not yet exist. Munich (which has
`source_mesh` instead of `source_laz`) is automatically skipped.

```bash
# Install once
pip install -r requirements.txt

# Build all eligible scenes
python build_meshes.py

# Build one
python build_meshes.py factory_large

# Override voxel size, force overwrite, list-only
python build_meshes.py --voxel 0.04 --force factory_large
python build_meshes.py --dry-run
```

Outputs `annotated/<scene>/source/mesh.glb` and a sidecar
`annotated/<scene>/source/mesh.meta.json` with the parameters used.

Voxa picks up the new GLBs automatically — no Voxa code change required.
```

- [ ] **Step 8: Run the smoke tests; verify all pass**

```bash
cd /home/hendrik/coding/engine/data/lidar/scripts
python -m pytest tests/test_smoke.py -v
```

Expected: 2 passed.

If pytest isn't installed in the active venv:
```bash
pip install pytest
```

- [ ] **Step 9: Commit**

```bash
cd /home/hendrik/coding/engine/data/lidar
git add scripts/
git commit -m "Scaffold lidar/scripts/build_meshes pipeline package"
```

(Note: `engine/data/lidar/` is its own git repo — independent of Voxa. Confirm with `git rev-parse --show-toplevel` before committing.)

---

## Task 2: Discovery module — eligibility scan (TDD)

**Files:**
- Create: `engine/data/lidar/scripts/tests/test_discovery.py`
- Modify: `engine/data/lidar/scripts/discovery.py`

The discovery module is pure: read meta.json files, apply the eligibility rule, return work items. No reconstruction, no LAZ reading.

- [ ] **Step 1: Write the failing tests** (`tests/test_discovery.py`)

```python
"""Tests for scene eligibility logic."""
from __future__ import annotations

from pathlib import Path

import pytest

from discovery import SceneTask, find_eligible_scenes


def test_finds_laz_sourced_scene_without_mesh(fake_archive: Path) -> None:
    tasks = find_eligible_scenes(fake_archive)
    names = [t.name for t in tasks]
    assert "alpha" in names


def test_skips_mesh_sourced_scene(fake_archive: Path) -> None:
    tasks = find_eligible_scenes(fake_archive)
    names = [t.name for t in tasks]
    assert "beta" not in names, "munich-style scene should be skipped"


def test_skips_scene_with_existing_mesh(fake_archive: Path) -> None:
    tasks = find_eligible_scenes(fake_archive)
    names = [t.name for t in tasks]
    assert "gamma" not in names, "scene with mesh.glb already present should be skipped"


def test_force_overrides_existing_mesh(fake_archive: Path) -> None:
    tasks = find_eligible_scenes(fake_archive, force=True)
    names = [t.name for t in tasks]
    assert "gamma" in names
    assert "beta" not in names, "force does NOT override the source_laz/source_mesh distinction"


def test_filter_by_name(fake_archive: Path) -> None:
    tasks = find_eligible_scenes(fake_archive, names=["alpha"])
    assert [t.name for t in tasks] == ["alpha"]


def test_filter_by_name_unknown_raises(fake_archive: Path) -> None:
    with pytest.raises(KeyError, match="zzz"):
        find_eligible_scenes(fake_archive, names=["zzz"])


def test_scene_task_paths(fake_archive: Path) -> None:
    [task] = find_eligible_scenes(fake_archive, names=["alpha"])
    assert task.laz_path == fake_archive / "laz" / "alpha.laz"
    assert task.out_glb_path == fake_archive / "annotated" / "alpha" / "source" / "mesh.glb"
    assert task.meta_path == fake_archive / "annotated" / "alpha" / "meta.json"


def test_skip_scene_with_missing_meta(tmp_path: Path) -> None:
    (tmp_path / "annotated" / "lonely").mkdir(parents=True)
    assert find_eligible_scenes(tmp_path) == []
```

- [ ] **Step 2: Run the tests to verify they fail**

```bash
cd /home/hendrik/coding/engine/data/lidar/scripts
python -m pytest tests/test_discovery.py -v
```

Expected: ImportError / "cannot import find_eligible_scenes from discovery".

- [ ] **Step 3: Implement `discovery.py`**

```python
"""Scene eligibility for the mesh-build pipeline.

Applies the rule: a scene is eligible if `meta.json::source_laz` is set
AND `source/mesh.glb` does not exist (unless `force=True`). Scenes whose
meta sets `source_mesh` (munich-style authored meshes) are always skipped.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class SceneTask:
    name: str
    archive_root: Path
    meta_path: Path
    laz_path: Path
    out_glb_path: Path
    out_sidecar_path: Path


def _resolve_laz(archive_root: Path, source_laz: str) -> Path:
    """`source_laz` in meta.json is recorded as 'lidar/laz/foo.laz' — a
    relative path from the parent of the archive root. We accept that form,
    plus a few looser variants, and fall back to `archive_root / Path(source_laz).name`.
    """
    p = Path(source_laz)
    if p.is_absolute() and p.exists():
        return p
    parent_relative = archive_root.parent / p
    if parent_relative.exists():
        return parent_relative
    archive_relative = archive_root / p
    if archive_relative.exists():
        return archive_relative
    name_only = archive_root / "laz" / p.name
    return name_only


def find_eligible_scenes(
    archive_root: Path,
    *,
    force: bool = False,
    names: Optional[list[str]] = None,
) -> list[SceneTask]:
    """Walk archive_root/annotated/* and return one SceneTask per eligible scene.

    Eligibility:
      * meta.json must exist and parse
      * meta.source_laz must be a non-empty string
      * meta.source_mesh must be falsy (None / empty / missing)
      * source/mesh.glb must not exist, unless force=True

    `names`: if provided, restrict to scenes with these names. Raises KeyError
    if any requested name is unknown.
    """
    archive_root = Path(archive_root)
    annotated = archive_root / "annotated"
    if not annotated.is_dir():
        return []

    requested = set(names) if names else None
    seen: set[str] = set()
    tasks: list[SceneTask] = []

    for scene_dir in sorted(annotated.iterdir()):
        if not scene_dir.is_dir():
            continue
        seen.add(scene_dir.name)
        if requested is not None and scene_dir.name not in requested:
            continue

        meta_path = scene_dir / "meta.json"
        if not meta_path.exists():
            continue
        try:
            meta = json.loads(meta_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue

        source_laz = meta.get("source_laz")
        source_mesh = meta.get("source_mesh")
        if not source_laz:
            continue
        if source_mesh:
            continue  # authored mesh wins; never overwrite

        out_glb = scene_dir / "source" / "mesh.glb"
        if out_glb.exists() and not force:
            continue

        tasks.append(SceneTask(
            name=scene_dir.name,
            archive_root=archive_root,
            meta_path=meta_path,
            laz_path=_resolve_laz(archive_root, source_laz),
            out_glb_path=out_glb,
            out_sidecar_path=scene_dir / "source" / "mesh.meta.json",
        ))

    if requested is not None:
        unknown = requested - seen
        if unknown:
            raise KeyError(f"unknown scene(s): {sorted(unknown)}")

    return tasks
```

- [ ] **Step 4: Run tests; verify all pass**

```bash
python -m pytest tests/test_discovery.py -v
```

Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/discovery.py scripts/tests/test_discovery.py
git commit -m "Discovery: walk annotated/ and return scenes eligible for mesh build"
```

---

## Task 3: Pipeline core — synthetic-sphere TDD

**Files:**
- Create: `engine/data/lidar/scripts/tests/test_pipeline.py`
- Modify: `engine/data/lidar/scripts/pipeline.py`

We test the pipeline as a black box on a fabricated 10K-point sphere. The synthetic test cleanly avoids the LAZ-reading dependency and lets us assert hard properties: non-empty mesh, vertex colors present, GLB round-trips.

- [ ] **Step 1: Write the failing test** (`tests/test_pipeline.py`)

```python
"""Tests for the reconstruction pipeline core.

Synthetic-sphere driven: we manufacture a colored point cloud on the unit
sphere and assert the pipeline produces a non-empty colored mesh.
"""
from __future__ import annotations

import numpy as np
import pytest
import trimesh

from pipeline import build_mesh_from_points


def _sphere_points(n: int = 10_000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.normal(size=(n, 3))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    return v.astype(np.float32)


def _sphere_colors(points: np.ndarray) -> np.ndarray:
    """RGB encodes the unit-sphere position (each channel in [0,1])."""
    return ((points + 1.0) / 2.0).astype(np.float32)


def test_synthetic_sphere_produces_mesh() -> None:
    pts = _sphere_points()
    rgb = _sphere_colors(pts)
    mesh = build_mesh_from_points(pts, rgb=rgb, voxel_size=0.05)
    assert mesh.vertices.shape[0] > 1000
    assert mesh.faces.shape[0] > 1000


def test_synthetic_sphere_carries_vertex_colors() -> None:
    pts = _sphere_points()
    rgb = _sphere_colors(pts)
    mesh = build_mesh_from_points(pts, rgb=rgb, voxel_size=0.05)
    assert mesh.visual.kind == "vertex"
    colors = mesh.visual.vertex_colors  # (N, 4) uint8
    assert colors.shape[0] == mesh.vertices.shape[0]
    assert colors[:, :3].std() > 5.0, "vertex colors must vary, not be a constant"


def test_intensity_fallback_produces_grayscale() -> None:
    pts = _sphere_points()
    intensity = np.linspace(0.0, 1.0, len(pts), dtype=np.float32)
    mesh = build_mesh_from_points(pts, rgb=None, intensity=intensity, voxel_size=0.05)
    rgba = mesh.visual.vertex_colors
    # Grayscale: R == G == B per vertex
    np.testing.assert_array_equal(rgba[:, 0], rgba[:, 1])
    np.testing.assert_array_equal(rgba[:, 1], rgba[:, 2])


def test_glb_roundtrips(tmp_path) -> None:
    pts = _sphere_points()
    rgb = _sphere_colors(pts)
    mesh = build_mesh_from_points(pts, rgb=rgb, voxel_size=0.05)
    out = tmp_path / "sphere.glb"
    mesh.export(out)
    assert out.stat().st_size > 1024
    loaded = trimesh.load(out)
    if isinstance(loaded, trimesh.Scene):
        loaded = trimesh.util.concatenate(tuple(loaded.dump()))
    assert loaded.vertices.shape[0] > 1000


def test_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        build_mesh_from_points(np.zeros((0, 3), dtype=np.float32), rgb=None)
```

- [ ] **Step 2: Run tests; verify they fail**

```bash
python -m pytest tests/test_pipeline.py -v
```

Expected: ImportError on `build_mesh_from_points`.

- [ ] **Step 3: Implement `pipeline.py`** (core only — LAZ reading + scene driver come in Task 4)

```python
"""LAZ → reconstructed mesh GLB pipeline.

Public API:
  build_mesh_from_points(points, rgb=None, intensity=None, voxel_size=None,
                          gpu=True) -> trimesh.Trimesh

  build_mesh_for_scene(task, *, voxel_size=None, gpu=True) -> BuildResult
    (added in Task 4)

Pipeline stages:
  1. Voxel-downsample (Open3D, GPU when available)
  2. Estimate normals (kNN, GPU when available)
  3. Ball-Pivoting reconstruction (CPU; multi-radius)
  4. Transfer per-vertex color from nearest input point (KDTree)
  5. Return a trimesh.Trimesh with vertex colors set
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d
import trimesh


# --- Tunables -----------------------------------------------------------

DEFAULT_VOXEL_FRAC = 1.5    # voxel_size = this * mean_spacing
VOXEL_MIN = 0.02            # meters
VOXEL_MAX = 0.10            # meters
RADII_MULTIPLIERS = (1.5, 2.5, 4.0)
NORMAL_KNN = 30
INTENSITY_PCTILES = (5.0, 95.0)


# --- Helpers ------------------------------------------------------------

def _estimate_mean_spacing(points: np.ndarray, sample: int = 10_000) -> float:
    """Mean nearest-neighbor distance over a random sample."""
    if len(points) <= sample:
        sub = points
    else:
        idx = np.random.default_rng(0).choice(len(points), sample, replace=False)
        sub = points[idx]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(sub.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(pcd)
    dists = []
    for i in range(len(sub)):
        _, _, d2 = tree.search_knn_vector_3d(sub[i], 2)
        if len(d2) >= 2:
            dists.append(np.sqrt(d2[1]))
    return float(np.median(dists)) if dists else 0.05


def _intensity_to_rgb(intensity: np.ndarray) -> np.ndarray:
    """Percentile-clipped linear stretch → grayscale [0,1] floats."""
    if intensity.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    lo, hi = np.percentile(intensity, INTENSITY_PCTILES)
    if hi <= lo:
        v = np.zeros_like(intensity, dtype=np.float32)
    else:
        v = np.clip((intensity - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)
    return np.stack([v, v, v], axis=1)


def _voxel_downsample(pcd: o3d.geometry.PointCloud, voxel: float) -> o3d.geometry.PointCloud:
    return pcd.voxel_down_sample(voxel)


def _estimate_normals(pcd: o3d.geometry.PointCloud, knn: int = NORMAL_KNN) -> None:
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    pcd.orient_normals_consistent_tangent_plane(knn)


def _run_bpa(pcd: o3d.geometry.PointCloud, radii: list[float]) -> o3d.geometry.TriangleMesh:
    return o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii)
    )


def _transfer_colors(mesh_verts: np.ndarray, src_points: np.ndarray, src_rgb: np.ndarray) -> np.ndarray:
    """For each mesh vertex, find the nearest source point and copy its RGB."""
    src_pcd = o3d.geometry.PointCloud()
    src_pcd.points = o3d.utility.Vector3dVector(src_points.astype(np.float64))
    tree = o3d.geometry.KDTreeFlann(src_pcd)
    out = np.zeros((len(mesh_verts), 3), dtype=np.float32)
    for i, v in enumerate(mesh_verts):
        _, idx, _ = tree.search_knn_vector_3d(v.astype(np.float64), 1)
        out[i] = src_rgb[idx[0]] if idx else 0.0
    return out


def _to_trimesh(o3d_mesh: o3d.geometry.TriangleMesh, vertex_rgb: np.ndarray) -> trimesh.Trimesh:
    verts = np.asarray(o3d_mesh.vertices, dtype=np.float32)
    faces = np.asarray(o3d_mesh.triangles, dtype=np.int32)
    rgba = np.empty((len(verts), 4), dtype=np.uint8)
    rgba[:, :3] = np.clip(vertex_rgb * 255.0, 0, 255).astype(np.uint8)
    rgba[:, 3] = 255
    return trimesh.Trimesh(vertices=verts, faces=faces, vertex_colors=rgba, process=False)


# --- Public ------------------------------------------------------------

@dataclass
class BuildResult:
    n_input_points: int
    n_downsampled: int
    voxel_size: float
    radii: list[float]
    n_vertices: int
    n_triangles: int
    color_source: str          # "rgb" | "intensity"


def build_mesh_from_points(
    points: np.ndarray,
    *,
    rgb: Optional[np.ndarray] = None,
    intensity: Optional[np.ndarray] = None,
    voxel_size: Optional[float] = None,
    gpu: bool = True,           # currently unused; reserved for tensor-API switch
) -> trimesh.Trimesh:
    """Reconstruct a mesh from points with per-vertex color.

    `rgb` (N,3) floats in [0,1] takes precedence over `intensity` (N,).
    If both are None, the mesh has flat-gray vertex colors.
    """
    if len(points) == 0:
        raise ValueError("build_mesh_from_points: empty input")

    points = np.asarray(points, dtype=np.float32)
    if rgb is not None:
        src_rgb = np.asarray(rgb, dtype=np.float32)
    elif intensity is not None:
        src_rgb = _intensity_to_rgb(np.asarray(intensity, dtype=np.float32))
    else:
        src_rgb = np.full((len(points), 3), 0.5, dtype=np.float32)

    if voxel_size is None:
        spacing = _estimate_mean_spacing(points)
        voxel_size = float(np.clip(DEFAULT_VOXEL_FRAC * spacing, VOXEL_MIN, VOXEL_MAX))

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    pcd = _voxel_downsample(pcd, voxel_size)
    if len(pcd.points) < 100:
        raise ValueError(f"voxel downsample left {len(pcd.points)} points; voxel too coarse")

    _estimate_normals(pcd)
    spacing = _estimate_mean_spacing(np.asarray(pcd.points, dtype=np.float32))
    radii = [m * spacing for m in RADII_MULTIPLIERS]

    o3d_mesh = _run_bpa(pcd, radii)
    if len(o3d_mesh.vertices) == 0:
        raise RuntimeError("BPA produced an empty mesh")

    vertex_rgb = _transfer_colors(np.asarray(o3d_mesh.vertices, dtype=np.float32), points, src_rgb)
    return _to_trimesh(o3d_mesh, vertex_rgb)
```

- [ ] **Step 4: Run tests; verify all pass**

```bash
python -m pytest tests/test_pipeline.py -v
```

Expected: 5 passed.

(Open3D ships its own threading; if `test_synthetic_sphere_produces_mesh` is unstable on your machine, increase `n=20_000` in `_sphere_points` — denser sampling makes BPA more reliable.)

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline.py scripts/tests/test_pipeline.py
git commit -m "Pipeline: BPA mesh reconstruction with per-vertex color"
```

---

## Task 4: LAZ ingestion + scene driver + decimation gate

**Files:**
- Modify: `engine/data/lidar/scripts/pipeline.py` (add `read_laz_points`, `build_mesh_for_scene`, `_decimate_if_oversized`)
- Modify: `engine/data/lidar/scripts/tests/test_pipeline.py` (LAZ + decimation tests)

- [ ] **Step 1: Add the failing tests**

Append to `tests/test_pipeline.py`:

```python
import laspy

from pipeline import (
    BuildResult,
    build_mesh_for_scene,
    read_laz_points,
)
from discovery import SceneTask


def _write_synthetic_laz(path, n=5_000, with_rgb=True, seed=0):
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3)).astype(np.float64)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= 5.0  # ~5m sphere
    fmt = 7 if with_rgb else 1
    header = laspy.LasHeader(point_format=fmt, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = pts.min(axis=0)
    las = laspy.LasData(header)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    las.intensity = (rng.random(n) * 65535).astype(np.uint16)
    if with_rgb:
        las.red   = (rng.random(n) * 65535).astype(np.uint16)
        las.green = (rng.random(n) * 65535).astype(np.uint16)
        las.blue  = (rng.random(n) * 65535).astype(np.uint16)
    las.write(path)


def test_read_laz_with_rgb(tmp_path) -> None:
    p = tmp_path / "rgb.laz"
    _write_synthetic_laz(p, with_rgb=True)
    points, rgb, intensity, color_source = read_laz_points(p)
    assert points.shape == (5_000, 3)
    assert rgb is not None and rgb.shape == (5_000, 3)
    assert color_source == "rgb"


def test_read_laz_without_rgb(tmp_path) -> None:
    p = tmp_path / "no_rgb.laz"
    _write_synthetic_laz(p, with_rgb=False)
    points, rgb, intensity, color_source = read_laz_points(p)
    assert rgb is None
    assert intensity is not None
    assert color_source == "intensity"


def test_build_mesh_for_scene_writes_glb(tmp_path, fake_archive) -> None:
    laz_path = fake_archive / "laz" / "alpha.laz"
    _write_synthetic_laz(laz_path, n=10_000, with_rgb=True)

    [task] = [t for t in __import__("discovery").find_eligible_scenes(fake_archive) if t.name == "alpha"]
    result = build_mesh_for_scene(task, voxel_size=0.1)

    assert task.out_glb_path.exists()
    assert task.out_sidecar_path.exists()
    assert result.n_vertices > 100
    assert result.color_source == "rgb"


def test_decimation_keeps_under_threshold(tmp_path) -> None:
    from pipeline import _decimate_if_oversized
    pts = _sphere_points(50_000)
    rgb = _sphere_colors(pts)
    mesh = build_mesh_from_points(pts, rgb=rgb, voxel_size=0.02)
    out = tmp_path / "big.glb"
    mesh.export(out)
    raw_size = out.stat().st_size
    decimated = _decimate_if_oversized(mesh, max_bytes=raw_size // 2)
    decimated.export(out)
    assert out.stat().st_size <= raw_size, "decimation must shrink output"
    assert decimated.vertices.shape[0] < mesh.vertices.shape[0]
```

- [ ] **Step 2: Run tests; verify they fail**

```bash
python -m pytest tests/test_pipeline.py -v
```

Expected: ImportError on `read_laz_points` / `build_mesh_for_scene`.

- [ ] **Step 3: Add `read_laz_points`, `_decimate_if_oversized`, and `build_mesh_for_scene` to `pipeline.py`**

```python
# Append to pipeline.py

import json
import time
from pathlib import Path

import laspy

# ── LAZ reader ──────────────────────────────────────────────────────────

LAZ_RGB_FORMATS = {2, 3, 5, 7, 8, 10}


def read_laz_points(
    path: Path,
) -> tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray], str]:
    """Read a LAZ file. Returns (points, rgb_or_None, intensity_or_None, color_source).

    Streaming via laspy[lazrs]. Coordinates land in float32 after subtracting
    the file's offset for numerical stability. Color preference: RGB if the
    point format carries it, else intensity-grayscale, else flat gray.
    """
    with laspy.open(path) as fh:
        pts_chunks: list[np.ndarray] = []
        rgb_chunks: list[np.ndarray] = []
        i_chunks: list[np.ndarray] = []
        fmt = fh.header.point_format.id
        has_rgb = fmt in LAZ_RGB_FORMATS

        for chunk in fh.chunk_iterator(2_000_000):
            xyz = np.column_stack([chunk.x, chunk.y, chunk.z]).astype(np.float64)
            pts_chunks.append(xyz)
            if has_rgb:
                rgb_chunks.append(np.column_stack([chunk.red, chunk.green, chunk.blue]).astype(np.float32))
            i_chunks.append(np.asarray(chunk.intensity, dtype=np.float32))

    points = np.concatenate(pts_chunks).astype(np.float64)
    intensity = np.concatenate(i_chunks).astype(np.float32)
    if has_rgb:
        rgb16 = np.concatenate(rgb_chunks).astype(np.float32)
        rgb = (rgb16 / 65535.0).clip(0.0, 1.0)
        color_source = "rgb"
    else:
        rgb = None
        color_source = "intensity"

    # Recenter to keep float32 stable
    centroid = points.mean(axis=0)
    points = (points - centroid).astype(np.float32)
    return points, rgb, intensity, color_source


# ── Decimation gate ─────────────────────────────────────────────────────

def _decimate_if_oversized(mesh: trimesh.Trimesh, max_bytes: int) -> trimesh.Trimesh:
    """Quadric-edge-collapse decimate via Open3D until the GLB fits under max_bytes.

    Returns the original mesh if it already fits.
    """
    import io

    buf = io.BytesIO()
    mesh.export(buf, file_type="glb")
    if buf.tell() <= max_bytes:
        return mesh

    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh.vertices.astype(np.float64))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh.faces.astype(np.int32))

    target = max(int(len(mesh.faces) * 0.6), 5_000)
    for _ in range(5):
        o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=target)
        verts = np.asarray(o3d_mesh.vertices, dtype=np.float32)
        faces = np.asarray(o3d_mesh.triangles, dtype=np.int32)
        # Reuse nearest-point color transfer from the original mesh
        vertex_rgb = _transfer_colors(verts, mesh.vertices.astype(np.float32),
                                      mesh.visual.vertex_colors[:, :3].astype(np.float32) / 255.0)
        out = _to_trimesh(o3d_mesh, vertex_rgb)
        buf = io.BytesIO()
        out.export(buf, file_type="glb")
        if buf.tell() <= max_bytes:
            return out
        target = int(target * 0.6)

    return out  # best effort


# ── Scene driver ────────────────────────────────────────────────────────

DEFAULT_MAX_BYTES = 400 * 1024 * 1024


def build_mesh_for_scene(
    task,                                # discovery.SceneTask
    *,
    voxel_size: Optional[float] = None,
    max_bytes: int = DEFAULT_MAX_BYTES,
    gpu: bool = True,
) -> BuildResult:
    """End-to-end build for one scene. Writes mesh.glb and mesh.meta.json."""
    t0 = time.time()
    points, rgb, intensity, color_source = read_laz_points(task.laz_path)

    mesh = build_mesh_from_points(
        points, rgb=rgb, intensity=intensity, voxel_size=voxel_size, gpu=gpu,
    )
    mesh = _decimate_if_oversized(mesh, max_bytes=max_bytes)

    task.out_glb_path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(task.out_glb_path)

    spacing = _estimate_mean_spacing(points)
    actual_voxel = voxel_size if voxel_size else float(np.clip(DEFAULT_VOXEL_FRAC * spacing, VOXEL_MIN, VOXEL_MAX))
    radii = [m * spacing for m in RADII_MULTIPLIERS]

    sidecar = {
        "scene": task.name,
        "input_laz": str(task.laz_path),
        "n_input_points": int(len(points)),
        "voxel_size_m": actual_voxel,
        "bpa_radii_m": radii,
        "color_source": color_source,
        "n_vertices": int(len(mesh.vertices)),
        "n_triangles": int(len(mesh.faces)),
        "wall_clock_s": round(time.time() - t0, 2),
    }
    task.out_sidecar_path.write_text(json.dumps(sidecar, indent=2))

    return BuildResult(
        n_input_points=len(points),
        n_downsampled=int(len(mesh.vertices)),  # post-mesh, not post-voxel — fine for log
        voxel_size=actual_voxel,
        radii=radii,
        n_vertices=int(len(mesh.vertices)),
        n_triangles=int(len(mesh.faces)),
        color_source=color_source,
    )
```

- [ ] **Step 4: Run tests; verify all pass**

```bash
python -m pytest tests/test_pipeline.py -v
```

Expected: 9 passed.

- [ ] **Step 5: Commit**

```bash
git add scripts/pipeline.py scripts/tests/test_pipeline.py
git commit -m "Pipeline: LAZ reader, scene driver, decimation gate"
```

---

## Task 5: CLI driver wiring (TDD via subprocess)

**Files:**
- Create: `engine/data/lidar/scripts/tests/test_cli.py`
- Modify: `engine/data/lidar/scripts/build_meshes.py`

- [ ] **Step 1: Write the failing tests** (`tests/test_cli.py`)

```python
"""End-to-end CLI test on a fake archive layout."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPTS = Path(__file__).resolve().parent.parent


def _write_synthetic_laz(path, n=5_000, with_rgb=True, seed=0):
    import laspy
    rng = np.random.default_rng(seed)
    pts = rng.normal(size=(n, 3)).astype(np.float64)
    pts /= np.linalg.norm(pts, axis=1, keepdims=True)
    pts *= 5.0
    fmt = 7 if with_rgb else 1
    header = laspy.LasHeader(point_format=fmt, version="1.4")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = pts.min(axis=0)
    las = laspy.LasData(header)
    las.x, las.y, las.z = pts[:, 0], pts[:, 1], pts[:, 2]
    las.intensity = (rng.random(n) * 65535).astype(np.uint16)
    if with_rgb:
        las.red   = (rng.random(n) * 65535).astype(np.uint16)
        las.green = (rng.random(n) * 65535).astype(np.uint16)
        las.blue  = (rng.random(n) * 65535).astype(np.uint16)
    las.write(path)


def _run_cli(*args, archive: Path, expect_zero=True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "build_meshes.py"),
         "--archive-root", str(archive), *args],
        capture_output=True, text=True,
    )
    if expect_zero:
        assert result.returncode == 0, f"stdout: {result.stdout}\nstderr: {result.stderr}"
    return result


def test_dry_run_lists_eligible_scenes(fake_archive: Path) -> None:
    result = _run_cli("--dry-run", archive=fake_archive)
    assert "alpha" in result.stdout
    assert "beta" not in result.stdout, "munich-style scene should not be listed"
    assert "gamma" not in result.stdout, "scene with mesh.glb already present should not be listed"


def test_dry_run_with_force_lists_gamma(fake_archive: Path) -> None:
    result = _run_cli("--dry-run", "--force", archive=fake_archive)
    assert "gamma" in result.stdout


def test_unknown_scene_errors_nonzero(fake_archive: Path) -> None:
    result = _run_cli("zzz", archive=fake_archive, expect_zero=False)
    assert result.returncode != 0


@pytest.mark.slow
def test_full_run_on_synthetic(fake_archive: Path) -> None:
    """Run the real pipeline on a synthetic LAZ. Slow because Open3D + BPA."""
    laz = fake_archive / "laz" / "alpha.laz"
    _write_synthetic_laz(laz, n=10_000)
    result = _run_cli("alpha", "--voxel", "0.1", archive=fake_archive)
    out_glb = fake_archive / "annotated" / "alpha" / "source" / "mesh.glb"
    out_meta = fake_archive / "annotated" / "alpha" / "source" / "mesh.meta.json"
    assert out_glb.exists()
    assert out_meta.exists()
    assert "alpha" in result.stdout
    assert "wrote" in result.stdout.lower() or "✅" in result.stdout
```

- [ ] **Step 2: Run tests; verify they fail**

```bash
python -m pytest tests/test_cli.py -v
```

Expected: failures (current `build_meshes.py` is just a print stub).

- [ ] **Step 3: Wire `build_meshes.py`** (replace the existing `main`)

```python
#!/usr/bin/env python3
"""Reconstruct source/mesh.glb for every LAZ-sourced annotated scene.

Usage:
  python build_meshes.py                    # all eligible scenes
  python build_meshes.py factory_large      # one scene
  python build_meshes.py --dry-run          # list what would run
  python build_meshes.py --force            # overwrite existing mesh.glb

See README.md for the full CLI surface.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from discovery import find_eligible_scenes
from pipeline import build_mesh_for_scene


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f} {unit}"
        n /= 1024


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("scenes", nargs="*", help="Scene names to build (default: all eligible)")
    parser.add_argument("--archive-root", default="/home/hendrik/coding/engine/data/lidar")
    parser.add_argument("--force", action="store_true", help="Overwrite existing mesh.glb")
    parser.add_argument("--voxel", type=float, default=None, help="Override voxel size (meters)")
    parser.add_argument("--dry-run", action="store_true", help="List scenes, do not build")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU prep")
    args = parser.parse_args(argv)

    archive = Path(args.archive_root).resolve()
    try:
        tasks = find_eligible_scenes(
            archive,
            force=args.force,
            names=args.scenes or None,
        )
    except KeyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    if not tasks:
        print("no eligible scenes")
        return 0

    if args.dry_run:
        print(f"would build {len(tasks)} scene(s):")
        for t in tasks:
            print(f"  - {t.name}  (laz: {t.laz_path.name})")
        return 0

    ok = skipped = failed = 0
    for i, task in enumerate(tasks, 1):
        print(f"[{i}/{len(tasks)}] {task.name}")
        if not task.laz_path.exists():
            print(f"  ⏭ skipped: laz not found ({task.laz_path})")
            skipped += 1
            continue
        try:
            r = build_mesh_for_scene(task, voxel_size=args.voxel, gpu=not args.no_gpu)
            size = _human_bytes(task.out_glb_path.stat().st_size)
            print(f"  ✅ wrote {task.out_glb_path.name}: {r.n_vertices:,} verts / "
                  f"{r.n_triangles:,} tris, {size}, color={r.color_source}")
            ok += 1
        except Exception as exc:  # noqa: BLE001 — batch-level resilience
            print(f"  ❌ failed: {exc}")
            failed += 1

    print(f"\nsummary: {ok} ok, {skipped} skipped, {failed} failed")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 4: Run tests; verify all pass**

```bash
python -m pytest tests/test_cli.py -v -m "not slow"
python -m pytest tests/test_cli.py::test_full_run_on_synthetic -v
```

First command expected: 3 passed. Second: 1 passed (a few minutes).

- [ ] **Step 5: Commit**

```bash
git add scripts/build_meshes.py scripts/tests/test_cli.py
git commit -m "CLI: argparse driver, batch-resilient summary log"
```

---

## Task 6: Run on real scenes; verify in Voxa

This task is mostly verification — running the script against the real lidar archive and confirming Voxa's Inspect mode picks the meshes up.

- [ ] **Step 1: Install the script's runtime dependencies**

```bash
cd /home/hendrik/coding/engine/data/lidar/scripts
pip install -r requirements.txt
```

- [ ] **Step 2: Dry-run to confirm eligibility**

```bash
python build_meshes.py --dry-run
```

Expected output:
```
would build 5 scene(s):
  - construction_site  (laz: Construction-site-sample-data.laz)
  - factory_large  (laz: Factory-large.laz)
  - navvis_mlx  (laz: NavVis-MLX-Sample-Data.laz)
  - navvis_vlx3_water_treatment  (laz: NavVis-VLX-3-data-water-treatment-facility.laz)
  - smart_ais  (laz: Sample-Data-VLX3-ProcessIndustry-SMART-AIS.laz)
```

- [ ] **Step 3: Build the smallest scene first** (sanity check)

```bash
python build_meshes.py construction_site
```

Confirm `annotated/construction_site/source/mesh.glb` and `mesh.meta.json` exist.

- [ ] **Step 4: Verify Voxa picks it up**

In a separate terminal:
```bash
cd /home/hendrik/coding/engine/tools/labeling/voxa
npm run dev
```

Open `http://127.0.0.1:5173`, select `annotated/construction_site` from the picker, and click the mesh-overlay toggle. Expect: the mesh appears, replaces the point cloud, and is colored.

If alignment looks off, check `mesh_is_z_up` in the network response — it should be `True`. If colors look gray on a scene that should have RGB, check the sidecar `mesh.meta.json::color_source`.

- [ ] **Step 5: Build the rest**

```bash
python build_meshes.py
```

Several minutes per scene; SMART-AIS is the longest. The batch is resilient — one failure doesn't abort the rest.

- [ ] **Step 6: Spot-check the remaining four scenes in Voxa**

Cycle through each annotated scene with the mesh toggle on. Note any obvious reconstruction defects in the sidecar (which scene, what's wrong) for follow-up tuning with `--voxel`.

- [ ] **Step 7: Commit the results**

The mesh GLBs themselves are gitignored (large binary). What gets committed is the sidecar metadata for reproducibility:

```bash
cd /home/hendrik/coding/engine/data/lidar
# If sidecars are gitignored too, skip; otherwise:
git add annotated/*/source/mesh.meta.json
git commit -m "Build: mesh.glb for the five LAZ-sourced annotated scenes"
```

(Check the archive's `.gitignore` first — `mesh.glb` is almost certainly ignored, sidecar might or might not be.)

---

## Final verification

- [ ] All 5 scenes have `source/mesh.glb` on disk
- [ ] `npm test` in the Voxa repo still passes (no Voxa code touched, but verify nothing regressed)
- [ ] All 5 scenes show the mesh overlay correctly in Voxa Inspect
