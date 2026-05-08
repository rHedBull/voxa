# RANSAC presegmentation param auto-tuning — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an "Optimize" button to the RANSAC presegment popover that runs a 20-trial Optuna TPE study on a 200k-point subsample, scores each trial by per-segment primitive-fit RMS, and auto-applies the best params to the full cloud.

**Architecture:** New `backend/preseg_optimize.py` (scoring + study runner). Three new endpoints (`/api/segment/presegment/optimize` + `/status` + `/abort`) with a single global background-thread job stored in `_state["preseg_opt_job"]`. Frontend polls status every 1.5 s, replaces Run/Cancel with a progress line + Abort during the run, populates the 11 knob inputs from `best_params` on completion.

**Tech Stack:** Python 3.12 / FastAPI / numpy / scipy (already vendored) / **optuna (new dep)**, frontend React 18 + Vite (no TS), pytest.

**Spec:** `docs/superpowers/specs/2026-05-08-ransac-param-autotune-design.md`.

---

## File map

- **Create** `backend/preseg_optimize.py` — `score_segmentation`, `run_study`, search-space definition.
- **Create** `backend/tests/test_preseg_optimize.py` — unit tests for scorer, smoke test for study.
- **Create** `backend/tests/test_preseg_optimize_endpoint.py` — endpoint lifecycle test.
- **Modify** `backend/requirements.txt` — add `optuna>=3.5,<5`.
- **Modify** `backend/main.py` — `_state["preseg_opt_job"]`, 3 endpoints, response models. Also extend the existing preseg apply path so the worker thread can reuse it (extract `_apply_ransac_to_session(positions, params, preserve_labeled) -> SegmentSession` if it isn't already factored out).
- **Modify** `frontend/src/api.js` — 3 wrappers: `segPresegOptimizeStart`, `segPresegOptimizeStatus`, `segPresegOptimizeAbort`.
- **Modify** `frontend/src/segment-tools.jsx` — Optimize button next to Run, polling state, abort UI, post-run knob hydration.

---

## Task 1: Add optuna dependency

**Files:**
- Modify: `backend/requirements.txt`

- [ ] **Step 1:** Append `optuna>=3.5,<5` to `backend/requirements.txt`.
- [ ] **Step 2:** Install into the venv:

  Run: `.venv/bin/pip install -r backend/requirements.txt`
  Expected: ends with `Successfully installed optuna-…` (or `Requirement already satisfied`).

- [ ] **Step 3:** Sanity-check import.

  Run: `.venv/bin/python -c "import optuna; print(optuna.__version__)"`
  Expected: a version ≥ 3.5 prints, no traceback.

- [ ] **Step 4:** Commit.

  ```bash
  git add backend/requirements.txt
  git commit -m "deps(backend): add optuna for RANSAC param auto-tuning"
  ```

---

## Task 2: `score_segmentation` — TDD

Per-segment best-primitive-fit RMS, point-weighted, with `-λ·log(n_segments)` bonus. λ = 1e-3 m. Excludes segments with < 30 points from the average. Returns `1e6` if `n_segments < 5` or `n_segments > 5000`.

**Files:**
- Create: `backend/preseg_optimize.py`
- Create: `backend/tests/test_preseg_optimize.py`

- [ ] **Step 1: Write failing tests.**

  ```python
  # backend/tests/test_preseg_optimize.py
  import numpy as np
  import pytest
  from preseg_optimize import score_segmentation


  def _plane_xyz(n=2000, rng=None):
      """A horizontal plane patch at z=0, side 2 m."""
      rng = rng or np.random.default_rng(0)
      xy = rng.uniform(-1.0, 1.0, size=(n, 2))
      z = rng.normal(0.0, 0.001, size=(n, 1))  # 1 mm noise
      return np.hstack([xy, z]).astype(np.float64)


  def _cylinder_xyz(n=2000, radius=0.3, height=1.5, rng=None):
      """Vertical cylinder, axis = z, centered at (3, 0)."""
      rng = rng or np.random.default_rng(1)
      theta = rng.uniform(0, 2 * np.pi, size=n)
      z = rng.uniform(0, height, size=n)
      r = radius + rng.normal(0.0, 0.001, size=n)
      x = 3.0 + r * np.cos(theta)
      y = r * np.sin(theta)
      return np.column_stack([x, y, z]).astype(np.float64)


  def _two_primitive_cloud():
      plane = _plane_xyz()
      cyl = _cylinder_xyz()
      xyz = np.vstack([plane, cyl])
      ids = np.concatenate([np.zeros(len(plane), dtype=np.int32),
                            np.ones(len(cyl), dtype=np.int32)])
      return xyz, ids


  def test_score_low_for_pure_primitives():
      xyz, ids = _two_primitive_cloud()
      # 2 segments < 5 → would normally hit the degenerate penalty. Pad with
      # 4 extra trivial segments (4 points each, ignored by the < 30 filter
      # but still counted in n_segments).
      pad_xyz = np.zeros((16, 3))
      pad_ids = np.repeat(np.arange(2, 6, dtype=np.int32), 4)
      xyz_full = np.vstack([xyz, pad_xyz])
      ids_full = np.concatenate([ids, pad_ids])
      score = score_segmentation(xyz_full, ids_full)
      assert score < 0.01, f"pure primitives should score low, got {score}"


  def test_score_high_for_mixed_segment():
      xyz, _ = _two_primitive_cloud()
      ids = np.zeros(len(xyz), dtype=np.int32)
      pad_xyz = np.zeros((16, 3))
      pad_ids = np.repeat(np.arange(1, 5, dtype=np.int32), 4)
      xyz_full = np.vstack([xyz, pad_xyz])
      ids_full = np.concatenate([ids, pad_ids])
      score = score_segmentation(xyz_full, ids_full)
      assert score > 0.05, f"mixed segment should score high, got {score}"


  def test_score_penalises_too_few_segments():
      xyz, _ = _two_primitive_cloud()
      ids = np.zeros(len(xyz), dtype=np.int32)
      score = score_segmentation(xyz, ids)
      assert score >= 1e5, f"degenerate (1 segment) should hit penalty, got {score}"


  def test_score_penalises_too_many_segments():
      rng = np.random.default_rng(2)
      xyz = rng.uniform(-1, 1, size=(20000, 3)).astype(np.float64)
      # one point per segment → 20000 segments, all > 5000 cap
      ids = np.arange(len(xyz), dtype=np.int32)
      score = score_segmentation(xyz, ids)
      assert score >= 1e5
  ```

- [ ] **Step 2: Run tests, expect ImportError / module-not-found.**

  Run: `.venv/bin/pytest backend/tests/test_preseg_optimize.py -v`
  Expected: collection fails because `preseg_optimize` doesn't exist.

- [ ] **Step 3: Implement `score_segmentation`.**

  Create `backend/preseg_optimize.py` with this content (and a module docstring):

  ```python
  """RANSAC presegmentation param auto-tuning.

  Two public entry points:

  - ``score_segmentation(xyz, instance_ids)`` — heuristic objective used by
    the Optuna study. Lower is better.
  - ``run_study(xyz_sub, …)`` — runs the TPE search and returns best params.
  """
  from __future__ import annotations

  import math
  import threading
  from typing import Callable, Optional

  import numpy as np

  PENALTY = 1e6
  MIN_SEG_PTS = 30
  LAMBDA_SEG = 1e-3
  MIN_SEGMENTS = 5
  MAX_SEGMENTS = 5000


  def _plane_rms(pts: np.ndarray) -> float:
      """RMS distance to the best-fit plane (PCA: smallest eigenvector)."""
      c = pts.mean(axis=0)
      d = pts - c
      cov = d.T @ d / len(pts)
      eig = np.linalg.eigvalsh(cov)
      # eigvalsh returns ascending; smallest = variance along plane normal
      return float(math.sqrt(max(eig[0], 0.0)))


  def _sphere_rms(pts: np.ndarray) -> float:
      """Algebraic sphere fit, RMS of |pi - c| - r."""
      n = len(pts)
      A = np.hstack([2 * pts, np.ones((n, 1))])
      b = (pts ** 2).sum(axis=1)
      try:
          x, *_ = np.linalg.lstsq(A, b, rcond=None)
      except np.linalg.LinAlgError:
          return float("inf")
      c = x[:3]
      r2 = x[3] + (c ** 2).sum()
      if r2 <= 0:
          return float("inf")
      r = math.sqrt(r2)
      d = np.linalg.norm(pts - c, axis=1) - r
      return float(math.sqrt((d ** 2).mean()))


  def _cylinder_rms(pts: np.ndarray) -> float:
      """Cylinder RMS: PCA largest eigenvector = axis. Project onto plane
      perpendicular to axis, fit a 2D circle, RMS radial residual."""
      c = pts.mean(axis=0)
      d = pts - c
      cov = d.T @ d / len(pts)
      eigvals, eigvecs = np.linalg.eigh(cov)
      axis = eigvecs[:, -1]  # largest eigenvalue → axis direction
      # project onto plane ⊥ axis
      proj = d - np.outer(d @ axis, axis)
      # 2D basis in that plane
      _, _, vh = np.linalg.svd(proj, full_matrices=False)
      basis = vh[:2]              # (2, 3)
      pts2d = proj @ basis.T      # (n, 2)
      # algebraic circle fit
      n = len(pts2d)
      A = np.hstack([2 * pts2d, np.ones((n, 1))])
      b = (pts2d ** 2).sum(axis=1)
      try:
          x, *_ = np.linalg.lstsq(A, b, rcond=None)
      except np.linalg.LinAlgError:
          return float("inf")
      cc = x[:2]
      r2 = x[2] + (cc ** 2).sum()
      if r2 <= 0:
          return float("inf")
      r = math.sqrt(r2)
      resid = np.linalg.norm(pts2d - cc, axis=1) - r
      return float(math.sqrt((resid ** 2).mean()))


  def _seg_score(pts: np.ndarray) -> float:
      """Best-primitive RMS. Plane / cylinder / sphere."""
      candidates = [_plane_rms(pts), _cylinder_rms(pts), _sphere_rms(pts)]
      return min(candidates)


  def score_segmentation(
      xyz: np.ndarray,
      instance_ids: np.ndarray,
      *,
      min_seg_pts: int = MIN_SEG_PTS,
      lambda_seg: float = LAMBDA_SEG,
  ) -> float:
      """Heuristic: weighted-mean per-segment primitive-fit RMS minus
      ``λ · log(n_segments)``. Lower is better. Degenerate runs return
      ``PENALTY``."""
      ids = np.asarray(instance_ids)
      mask = ids >= 0
      if not mask.any():
          return PENALTY
      uniq, counts = np.unique(ids[mask], return_counts=True)
      n_segments = len(uniq)
      if n_segments < MIN_SEGMENTS or n_segments > MAX_SEGMENTS:
          return PENALTY

      total_pts = 0
      total = 0.0
      for sid, n in zip(uniq, counts):
          if n < min_seg_pts:
              continue
          pts = xyz[ids == sid]
          rms = _seg_score(pts)
          if not math.isfinite(rms):
              continue
          total += n * rms
          total_pts += n
      if total_pts == 0:
          return PENALTY
      mean_rms = total / total_pts
      return mean_rms - lambda_seg * math.log(n_segments)
  ```

- [ ] **Step 4: Run tests.**

  Run: `.venv/bin/pytest backend/tests/test_preseg_optimize.py -v`
  Expected: 4 PASSED.

- [ ] **Step 5: Commit.**

  ```bash
  git add backend/preseg_optimize.py backend/tests/test_preseg_optimize.py
  git commit -m "feat(preseg): score_segmentation primitive-fit objective"
  ```

---

## Task 3: `run_study` — Optuna TPE driver

20 trials, TPE sampler. Search space mirrors `RANSAC_DEFAULTS` in `backend/presegment_ransac.py:201`. Distance/radius knobs use `log=True`. Per-trial exceptions caught → return `PENALTY`. A callback reports progress and stops the study when `cancel_event` is set.

**Files:**
- Modify: `backend/preseg_optimize.py`
- Modify: `backend/tests/test_preseg_optimize.py`

- [ ] **Step 1: Write failing smoke test.**

  Append to `backend/tests/test_preseg_optimize.py`:

  ```python
  import threading
  from preseg_optimize import run_study, SEARCH_SPACE


  def test_search_space_mirrors_ransac_defaults():
      from presegment_ransac import RANSAC_DEFAULTS
      assert set(SEARCH_SPACE.keys()) == set(RANSAC_DEFAULTS.keys())


  def test_run_study_smoke():
      rng = np.random.default_rng(3)
      # tiny mixed cloud big enough that RANSAC produces > MIN_SEGMENTS
      plane = rng.uniform(-1, 1, (1500, 2))
      plane = np.hstack([plane, rng.normal(0, 1e-3, (1500, 1))])
      cyl_theta = rng.uniform(0, 2 * np.pi, 1500)
      cyl_z = rng.uniform(0, 1, 1500)
      cyl = np.column_stack([
          3 + 0.3 * np.cos(cyl_theta),
          0.3 * np.sin(cyl_theta),
          cyl_z,
      ])
      xyz = np.vstack([plane, cyl]).astype(np.float64)

      progress = []
      cancel = threading.Event()

      def cb(info):
          progress.append(info["trial"])

      result = run_study(
          xyz,
          n_trials=3,
          cancel_event=cancel,
          progress_cb=cb,
          class_map={},
      )
      assert result["n_trials_run"] == 3
      assert set(result["best_params"].keys()) == set(SEARCH_SPACE.keys())
      assert isinstance(result["best_score"], float)
      assert progress, "progress_cb should have been called at least once"


  def test_run_study_cancel():
      rng = np.random.default_rng(4)
      xyz = rng.uniform(-1, 1, (2000, 3)).astype(np.float64)
      cancel = threading.Event()
      cancel.set()  # pre-cancelled

      def cb(_info):
          pass

      result = run_study(
          xyz,
          n_trials=20,
          cancel_event=cancel,
          progress_cb=cb,
          class_map={},
      )
      assert result["n_trials_run"] < 20
  ```

- [ ] **Step 2: Run tests, expect failure.**

  Run: `.venv/bin/pytest backend/tests/test_preseg_optimize.py::test_search_space_mirrors_ransac_defaults -v`
  Expected: ImportError on `SEARCH_SPACE`.

- [ ] **Step 3: Add `SEARCH_SPACE` and `run_study`.**

  Append to `backend/preseg_optimize.py`:

  ```python
  # ── search space ────────────────────────────────────────────────────────
  # (low, high, kind)  kind ∈ {"float", "int", "logfloat"}
  SEARCH_SPACE: dict[str, tuple[float, float, str]] = {
      "plane_distance_threshold": (0.005, 0.10,  "logfloat"),
      "plane_min_inliers":        (20,    300,   "int"),
      "max_planes":               (5,     50,    "int"),
      "flat_thresh":              (0.1,   2.0,   "float"),
      "cylinder_ratio_thresh":    (1.5,   8.0,   "float"),
      "cyl_search_radius":        (0.05,  0.30,  "logfloat"),
      "cyl_axis_thresh":          (0.85,  0.99,  "float"),
      "cyl_radius_ratio":         (1.2,   3.0,   "float"),
      "cyl_distance_threshold":   (0.005, 0.10,  "logfloat"),
      "merge_axis_dot":           (0.85,  0.99,  "float"),
      "merge_radius_ratio":       (1.1,   3.0,   "float"),
  }


  def _suggest_params(trial) -> dict[str, float]:
      out: dict[str, float] = {}
      for name, (lo, hi, kind) in SEARCH_SPACE.items():
          if kind == "int":
              out[name] = float(trial.suggest_int(name, int(lo), int(hi)))
          elif kind == "logfloat":
              out[name] = trial.suggest_float(name, lo, hi, log=True)
          else:
              out[name] = trial.suggest_float(name, lo, hi)
      return out


  def run_study(
      xyz_sub: np.ndarray,
      *,
      n_trials: int,
      cancel_event: threading.Event,
      progress_cb: Callable[[dict], None],
      class_map: Optional[dict[str, int]] = None,
  ) -> dict:
      """Optuna TPE study over SEARCH_SPACE. Returns
      {best_params, best_score, n_trials_run, trials}."""
      import optuna
      from presegment_ransac import presegment as _ransac

      optuna.logging.set_verbosity(optuna.logging.WARNING)
      sampler = optuna.samplers.TPESampler(seed=0)
      study = optuna.create_study(direction="minimize", sampler=sampler)
      trials_log: list[dict] = []
      ran = 0

      def objective(trial):
          nonlocal ran
          if cancel_event.is_set():
              raise optuna.TrialPruned()
          params = _suggest_params(trial)
          try:
              instance_ids, _summary = _ransac(
                  xyz_sub,
                  class_map=class_map,
                  log=lambda *_: None,
                  params=params,
              )
              score = score_segmentation(xyz_sub, instance_ids)
          except Exception:
              score = PENALTY
          ran += 1
          trials_log.append({"params": params, "score": score})
          return score

      def cb(study_, _trial):
          if cancel_event.is_set():
              study_.stop()
          best = study_.best_trial if study_.trials else None
          progress_cb({
              "trial": ran,
              "total": n_trials,
              "best_score": float(best.value) if best else None,
              "best_params": dict(best.params) if best else None,
          })

      study.optimize(objective, n_trials=n_trials, callbacks=[cb])

      best = study.best_trial
      return {
          "best_params": dict(best.params),
          "best_score": float(best.value),
          "n_trials_run": ran,
          "trials": trials_log,
      }
  ```

- [ ] **Step 4: Run tests.**

  Run: `.venv/bin/pytest backend/tests/test_preseg_optimize.py -v`
  Expected: all PASSED. The smoke test may take ~10 s.

- [ ] **Step 5: Commit.**

  ```bash
  git add backend/preseg_optimize.py backend/tests/test_preseg_optimize.py
  git commit -m "feat(preseg): Optuna TPE study driver over RANSAC knobs"
  ```

---

## Task 4: Backend endpoints + background job

Three endpoints, single global job in `_state["preseg_opt_job"]`. Worker thread runs the study on a 200k-point random subsample, then re-runs RANSAC with `best_params` on the full cloud and writes the result into `_state["seg"]` exactly like the existing `/api/segment/presegment` route.

**Files:**
- Modify: `backend/main.py`
- Create: `backend/tests/test_preseg_optimize_endpoint.py`

- [ ] **Step 1: Read the existing `/api/segment/presegment` body** at `backend/main.py:1052-1156` to understand the apply-result-to-session flow. Identify the steps that build `instance_ids`, `class_ids`, the renumber, the `SegmentSession` write, and the box/hull computation.

- [ ] **Step 2: Extract a helper.** In `backend/main.py`, just above the existing `segment_presegment` route, add:

  ```python
  def _apply_ransac_result_to_session(
      *,
      sub_inst: np.ndarray,
      sub_summary: list[dict],
      positions: np.ndarray,
      existing_class: np.ndarray,
      existing_inst: np.ndarray,
      keep_mask: np.ndarray,
      redo_mask: np.ndarray,
      resolution: float,
  ) -> tuple[SegmentSession, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
      """Renumber sub-instance ids above the existing max, write the new
      SegmentSession, and return (session, instance_ids, box_ids,
      box_centers, box_sizes, hull_v, hull_f, hull_seg)."""
      # body lifted verbatim from the existing route, lines ~1112–1143
      ...
  ```

  Then refactor `segment_presegment` to call this helper. Add the import
  for `SegmentSession` at module scope if it isn't already there.

- [ ] **Step 3: Run existing preseg tests to confirm refactor is green.**

  Run: `.venv/bin/pytest backend/tests/test_segment_presegment_endpoint.py -v`
  Expected: same number of tests pass as before.

- [ ] **Step 4: Commit the refactor on its own.**

  ```bash
  git add backend/main.py
  git commit -m "refactor(main): extract _apply_ransac_result_to_session helper"
  ```

- [ ] **Step 5: Write failing endpoint test.**

  Create `backend/tests/test_preseg_optimize_endpoint.py`:

  ```python
  import time
  import numpy as np
  import pytest
  from fastapi.testclient import TestClient


  @pytest.fixture
  def client(monkeypatch, tmp_path):
      monkeypatch.setenv("VOXA_DATA_DIR", str(tmp_path))
      import importlib, main as _m
      importlib.reload(_m)
      return TestClient(_m.app), _m


  def _load_synthetic(client, m):
      """Inject a small synthetic point cloud into _state."""
      from point_cloud import PointCloud
      rng = np.random.default_rng(42)
      plane = rng.uniform(-1, 1, (3000, 2))
      plane = np.hstack([plane, rng.normal(0, 1e-3, (3000, 1))])
      cyl_theta = rng.uniform(0, 2 * np.pi, 3000)
      cyl = np.column_stack([
          3 + 0.3 * np.cos(cyl_theta),
          0.3 * np.sin(cyl_theta),
          rng.uniform(0, 1, 3000),
      ])
      xyz = np.vstack([plane, cyl]).astype(np.float32)
      colors = np.tile(np.array([0.5, 0.5, 0.5], dtype=np.float32), (len(xyz), 1))
      m._state["pc"] = PointCloud(points=xyz, colors=colors)


  def test_optimize_lifecycle_abort(client):
      c, m = client
      _load_synthetic(c, m)
      r = c.post("/api/segment/presegment/optimize",
                 json={"n_trials": 20, "subsample_n": 2000})
      assert r.status_code == 200
      job_id = r.json()["job_id"]

      # poll once
      r = c.get(f"/api/segment/presegment/optimize/status?job_id={job_id}")
      assert r.status_code == 200
      assert r.json()["status"] in ("running", "done")

      # abort
      r = c.post(f"/api/segment/presegment/optimize/abort?job_id={job_id}")
      assert r.status_code == 200

      # wait briefly for the worker to wind down
      for _ in range(50):
          r = c.get(f"/api/segment/presegment/optimize/status?job_id={job_id}")
          if r.json()["status"] in ("aborted", "done", "error"):
              break
          time.sleep(0.1)
      assert r.json()["status"] in ("aborted", "done")


  def test_optimize_rejects_concurrent(client):
      c, m = client
      _load_synthetic(c, m)
      r1 = c.post("/api/segment/presegment/optimize",
                  json={"n_trials": 20, "subsample_n": 2000})
      assert r1.status_code == 200
      r2 = c.post("/api/segment/presegment/optimize",
                  json={"n_trials": 20, "subsample_n": 2000})
      assert r2.status_code == 409
      # tidy up
      c.post(f"/api/segment/presegment/optimize/abort?job_id={r1.json()['job_id']}")


  def test_optimize_requires_scene(client):
      c, _m = client
      r = c.post("/api/segment/presegment/optimize", json={})
      assert r.status_code == 409
  ```

- [ ] **Step 6: Run, expect failure (404 on the new endpoints).**

  Run: `.venv/bin/pytest backend/tests/test_preseg_optimize_endpoint.py -v`
  Expected: 404s.

- [ ] **Step 7: Implement the endpoints.** Append to `backend/main.py` (after the existing preseg route, before `SegmentStateResponse`):

  ```python
  import threading as _threading
  import uuid as _uuid


  class PresegOptimizeRequest(BaseModel):
      n_trials: int = 20
      subsample_n: int = 200_000
      preserve_labeled: bool = True


  class PresegOptimizeStartResponse(BaseModel):
      job_id: str
      total: int


  class PresegOptimizeStatusResponse(BaseModel):
      status: Literal["running", "done", "aborted", "error"]
      trial: int
      total: int
      best_score: Optional[float] = None
      best_params: Optional[dict] = None
      error: Optional[str] = None


  def _new_job_state(total: int) -> dict:
      return {
          "id": str(_uuid.uuid4()),
          "thread": None,
          "cancel": _threading.Event(),
          "status": "running",
          "trial": 0,
          "total": total,
          "best_score": None,
          "best_params": None,
          "error": None,
          "started_at": _time.time(),
      }


  def _preseg_optimize_worker(
      *,
      job: dict,
      positions: np.ndarray,
      existing_class: np.ndarray,
      existing_inst: np.ndarray,
      keep_mask: np.ndarray,
      redo_mask: np.ndarray,
      subsample_n: int,
      n_trials: int,
      class_map: dict[str, int],
  ) -> None:
      from preseg_optimize import run_study
      from presegment_ransac import presegment as _ransac
      try:
          # subsample for the study
          n = int(positions.shape[0])
          if n > subsample_n:
              idx = np.random.default_rng(0).choice(n, size=subsample_n, replace=False)
              xyz_sub = np.asarray(positions[idx], dtype=np.float64)
          else:
              xyz_sub = np.asarray(positions, dtype=np.float64)

          def cb(info: dict) -> None:
              job["trial"] = info["trial"]
              job["best_score"] = info["best_score"]
              job["best_params"] = info["best_params"]

          result = run_study(
              xyz_sub,
              n_trials=n_trials,
              cancel_event=job["cancel"],
              progress_cb=cb,
              class_map=class_map,
          )
          job["best_score"] = result["best_score"]
          job["best_params"] = result["best_params"]

          if job["cancel"].is_set():
              job["status"] = "aborted"
              return

          # final apply on the full cloud (only the redo-mask region)
          sub_positions = np.asarray(positions[redo_mask], dtype=np.float64)
          sub_inst, sub_summary = _ransac(
              sub_positions,
              class_map=class_map,
              log=lambda *_: None,
              params=result["best_params"],
          )
          (sess, *_unused) = _apply_ransac_result_to_session(
              sub_inst=sub_inst,
              sub_summary=sub_summary,
              positions=positions,
              existing_class=existing_class,
              existing_inst=existing_inst,
              keep_mask=keep_mask,
              redo_mask=redo_mask,
              resolution=0.05,
          )
          _state["seg"] = sess
          _state["seg"].dirty = True
          job["status"] = "done"
      except Exception as e:  # noqa: BLE001
          job["status"] = "error"
          job["error"] = str(e)


  @app.post("/api/segment/presegment/optimize", response_model=PresegOptimizeStartResponse)
  def segment_presegment_optimize(req: PresegOptimizeRequest = PresegOptimizeRequest()):
      existing = _state.get("preseg_opt_job")
      if existing and existing["status"] == "running":
          raise HTTPException(409, "An optimization is already running")

      seg = _state.get("seg")
      if seg is not None:
          positions = seg.positions
          existing_class = seg.class_ids.copy()
          existing_inst = seg.instance_ids.copy()
      else:
          pc = _state.get("pc")
          if pc is None:
              raise HTTPException(409, "No scene loaded — call /api/load first")
          positions = pc.points
          n_init = int(positions.shape[0])
          existing_class = np.full(n_init, -1, dtype=np.int8)
          existing_inst = np.full(n_init, -1, dtype=np.int32)

      n_points = int(positions.shape[0])
      keep_mask = (existing_class >= 0) if req.preserve_labeled else np.zeros(n_points, dtype=bool)
      redo_mask = ~keep_mask

      job = _new_job_state(req.n_trials)
      _state["preseg_opt_job"] = job
      t = _threading.Thread(
          target=_preseg_optimize_worker,
          kwargs=dict(
              job=job,
              positions=positions,
              existing_class=existing_class,
              existing_inst=existing_inst,
              keep_mask=keep_mask,
              redo_mask=redo_mask,
              subsample_n=req.subsample_n,
              n_trials=req.n_trials,
              class_map=_voxa_class_name_to_id(),
          ),
          daemon=True,
      )
      job["thread"] = t
      t.start()
      return PresegOptimizeStartResponse(job_id=job["id"], total=req.n_trials)


  @app.get("/api/segment/presegment/optimize/status", response_model=PresegOptimizeStatusResponse)
  def segment_presegment_optimize_status(job_id: str):
      job = _state.get("preseg_opt_job")
      if not job or job["id"] != job_id:
          raise HTTPException(404, "Unknown job_id")
      return PresegOptimizeStatusResponse(
          status=job["status"],
          trial=job["trial"],
          total=job["total"],
          best_score=job["best_score"],
          best_params=job["best_params"],
          error=job["error"],
      )


  @app.post("/api/segment/presegment/optimize/abort")
  def segment_presegment_optimize_abort(job_id: str):
      job = _state.get("preseg_opt_job")
      if not job or job["id"] != job_id:
          raise HTTPException(404, "Unknown job_id")
      job["cancel"].set()
      return {"status": "aborting"}
  ```

  Add `import time as _time` at the top of `main.py` if it isn't already imported.

- [ ] **Step 8: Run endpoint tests.**

  Run: `.venv/bin/pytest backend/tests/test_preseg_optimize_endpoint.py -v`
  Expected: all 3 PASSED.

- [ ] **Step 9: Run the full backend suite to confirm nothing else regressed.**

  Run: `.venv/bin/pytest backend/tests/ -q`
  Expected: all green.

- [ ] **Step 10: Commit.**

  ```bash
  git add backend/main.py backend/tests/test_preseg_optimize_endpoint.py
  git commit -m "feat(preseg): /optimize endpoints with background job"
  ```

---

## Task 5: Frontend API wrappers

**Files:**
- Modify: `frontend/src/api.js`

- [ ] **Step 1: Add three wrappers** just below `segPresegment` (around `frontend/src/api.js:200`):

  ```js
    async segPresegOptimizeStart({ nTrials = 20, subsampleN = 200_000, preserveLabeled = true } = {}) {
      const r = await fetch('/api/segment/presegment/optimize', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ n_trials: nTrials, subsample_n: subsampleN, preserve_labeled: preserveLabeled }),
      });
      if (!r.ok) throw new Error(`segPresegOptimizeStart failed: ${r.status} ${await r.text()}`);
      const j = await r.json();
      return { jobId: j.job_id, total: j.total };
    },
    async segPresegOptimizeStatus(jobId) {
      const r = await fetch(`/api/segment/presegment/optimize/status?job_id=${encodeURIComponent(jobId)}`);
      if (!r.ok) throw new Error(`segPresegOptimizeStatus failed: ${r.status} ${await r.text()}`);
      const j = await r.json();
      return {
        status: j.status,
        trial: j.trial,
        total: j.total,
        bestScore: j.best_score,
        bestParams: j.best_params,
        error: j.error,
      };
    },
    async segPresegOptimizeAbort(jobId) {
      const r = await fetch(`/api/segment/presegment/optimize/abort?job_id=${encodeURIComponent(jobId)}`, { method: 'POST' });
      if (!r.ok) throw new Error(`segPresegOptimizeAbort failed: ${r.status} ${await r.text()}`);
      return r.json();
    },
  ```

- [ ] **Step 2: Sanity check via dev server.**

  Run: `npm run dev` (in another shell, if not already running) and confirm `curl -s http://127.0.0.1:5173/api/segment/presegment/optimize/status?job_id=foo` returns `{"detail":"Unknown job_id"}` (404).

- [ ] **Step 3: Commit.**

  ```bash
  git add frontend/src/api.js
  git commit -m "feat(api): segPresegOptimize {Start,Status,Abort} wrappers"
  ```

---

## Task 6: Optimize button + polling UI

**Files:**
- Modify: `frontend/src/segment-tools.jsx`

- [ ] **Step 1: Add state.** Inside `PresegmentButton`, after the existing `useState` declarations:

  ```js
  const [optStatus, setOptStatus] = useState('idle'); // idle|running|done|aborted|error
  const [optInfo, setOptInfo]     = useState(null);   // {jobId, trial, total, bestScore, bestParams}
  const optTimerRef = useRef(null);
  ```

- [ ] **Step 2: Add `runOptimize`** alongside `run`:

  ```js
  async function runOptimize() {
    if (!cloud) return;
    setError(null);
    setStats(null);
    try {
      const { jobId, total } = await VoxaAPI.segPresegOptimizeStart({
        nTrials: 20, subsampleN: 200_000, preserveLabeled,
      });
      setOptInfo({ jobId, trial: 0, total, bestScore: null, bestParams: null });
      setOptStatus('running');
      setBusy(true);
      // polling loop
      const tick = async () => {
        try {
          const s = await VoxaAPI.segPresegOptimizeStatus(jobId);
          setOptInfo({ jobId, trial: s.trial, total: s.total,
                       bestScore: s.bestScore, bestParams: s.bestParams });
          if (s.status === 'running') return;
          // terminal
          clearInterval(optTimerRef.current);
          optTimerRef.current = null;
          setOptStatus(s.status);
          setBusy(false);
          if (s.status === 'done' && s.bestParams) {
            // hydrate knob inputs
            setRansacKnobs((prev) => ({ ...prev, ...s.bestParams }));
            // pull the freshly-applied session and update the viewer
            const state = await VoxaAPI.segState();
            if (state?.hasState) {
              if (prelabelRef) {
                prelabelRef.current = {
                  classFull: state.fullClassIds.slice(),
                  instanceFull: state.fullInstanceIds.slice(),
                };
              }
              setSegState(initSegState({
                classFull: state.fullClassIds,
                instanceFull: state.fullInstanceIds,
                isFromPrelabel: true,
                segBoxes: (state.segIds && state.segCenters && state.segSizes)
                  ? { segIds: state.segIds, segCenters: state.segCenters, segSizes: state.segSizes }
                  : null,
                segHulls: (state.hullVertices && state.hullFaces && state.hullFaceSeg)
                  ? { vertices: state.hullVertices, faces: state.hullFaces, faceSeg: state.hullFaceSeg }
                  : null,
              }));
              if (setCloud) {
                const subIdx = cloud.subsampleIdx;
                const subN = (cloud.positions?.length || 0) / 3;
                const subClass = new Int8Array(subN);
                const subInst = new Int32Array(subN);
                for (let p = 0; p < subN; p++) {
                  const f = subIdx ? subIdx[p] : p;
                  subClass[p] = state.fullClassIds[f];
                  subInst[p] = state.fullInstanceIds[f];
                }
                setCloud({ ...cloud, classIds: subClass, instanceIds: subInst, isFromPrelabel: true });
              }
              setStats({ nSegments: state.nSegments, meanSize: state.nAssigned / Math.max(state.nSegments, 1) });
            }
          } else if (s.status === 'error') {
            setError(s.error || 'optimization failed');
          }
        } catch (e) {
          setError(String(e.message || e));
          clearInterval(optTimerRef.current);
          optTimerRef.current = null;
          setOptStatus('error');
          setBusy(false);
        }
      };
      optTimerRef.current = setInterval(tick, 1500);
      tick();  // first poll immediately
    } catch (e) {
      setError(String(e.message || e));
      setOptStatus('error');
      setBusy(false);
    }
  }

  async function abortOptimize() {
    if (!optInfo?.jobId) return;
    try { await VoxaAPI.segPresegOptimizeAbort(optInfo.jobId); } catch { /* ignore */ }
  }

  // cleanup polling on unmount
  useEffect(() => () => {
    if (optTimerRef.current) clearInterval(optTimerRef.current);
  }, []);
  ```

  Confirm `VoxaAPI.segState` exists in `frontend/src/api.js` (used elsewhere — search to verify the shape returned matches the field names referenced above; adjust on the fly if any field name differs).

- [ ] **Step 3: Render the Optimize button + progress UI.** Inside the popover JSX, replace the existing footer block:

  ```jsx
  <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end', marginTop: 2 }}>
    <button
      type="button"
      className="tool-btn mini"
      onClick={() => setOpen(false)}
    >Cancel</button>
    <button
      type="button"
      className="tool-btn mini active"
      onClick={run}
    >Run</button>
  </div>
  ```

  with:

  ```jsx
  {optStatus === 'running' ? (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <div style={{ fontSize: 11, opacity: 0.85 }}>
        Trial {optInfo?.trial ?? 0}/{optInfo?.total ?? 20}
        {optInfo?.bestScore != null && ` · best ${optInfo.bestScore.toFixed(4)}`}
      </div>
      <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end' }}>
        <button type="button" className="tool-btn mini" onClick={abortOptimize}>Abort</button>
      </div>
    </div>
  ) : (
    <div style={{ display: 'flex', gap: 6, justifyContent: 'flex-end', marginTop: 2 }}>
      <button type="button" className="tool-btn mini" onClick={() => setOpen(false)}>Cancel</button>
      <button
        type="button"
        className="tool-btn mini"
        disabled={mode !== 'ransac'}
        title={mode !== 'ransac' ? 'Optimization is only available in RANSAC mode' : 'Run a 20-trial param search'}
        onClick={runOptimize}
      >Optimize</button>
      <button type="button" className="tool-btn mini active" onClick={run}>Run</button>
    </div>
  )}
  ```

- [ ] **Step 4: Manual smoke test.**

  Run: `npm run dev` (if not already), open a small annotated scene, switch to Label mode, open the ⚙ popover, switch mode to RANSAC, click Optimize. Watch the trial counter advance and confirm the Abort button stops it. Click Optimize again and let it finish — the 11 knob inputs should populate with `best_params`, the stats line should show `N segments · mean size X pts`, and the viewport should show the new segmentation.

- [ ] **Step 5: Commit.**

  ```bash
  git add frontend/src/segment-tools.jsx
  git commit -m "feat(preseg): Optimize button with polling, abort, and result hydration"
  ```

---

## Task 7: Verification + memory update

- [ ] **Step 1: Full backend test pass.**

  Run: `.venv/bin/pytest backend/tests/ -q`
  Expected: all green; new tests in `test_preseg_optimize.py` and `test_preseg_optimize_endpoint.py` included.

- [ ] **Step 2: Update auto-memory.** Edit `~/.claude/projects/-home-hendrik-coding-engine-tools-labeling-voxa/memory/MEMORY.md` to drop the WIP marker if any remains, and add a one-liner pointing to the new spec + plan.

- [ ] **Step 3: Final commit if memory was updated, then offer the user a manual run with a real scene.**

---

## Skills referenced
- @superpowers:test-driven-development for tasks 2, 3, 4
- @superpowers:verification-before-completion at task 7
- @superpowers:systematic-debugging if any task fails

