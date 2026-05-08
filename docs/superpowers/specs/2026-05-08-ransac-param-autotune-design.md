# RANSAC presegmentation param auto-tuning

**Status:** approved 2026-05-08
**Scope:** voxa Label-mode preseg popover

## Goal

Let the user click "Optimize" in the RANSAC presegment popover and have an
algorithm search the 11 RANSAC knobs (already exposed in the popover) for a
configuration that produces high-quality per-primitive segments — preferring
many small, primitive-pure segments over few large mixed ones. Best params are
auto-applied to the full cloud at the end.

## Non-goals

- Cross-scene parameter learning. Each scene is optimized independently.
- Replacing manual tuning. The button augments the existing knob inputs; the
  user can still tweak by hand and re-run.
- Multi-run gallery / visual diff. The user explicitly chose unattended
  numeric optimization.

## Objective function

Per-segment primitive-fit residual, weighted by point count:

```
for each segment with n_pts >= 30:
    fit best plane (PCA), cylinder (least-squares with axis from PCA),
        and sphere (algebraic), pick the one with smallest RMS
    seg_score = best_primitive_rms      # in scene units (≈ metres)
score = sum(n_pts * seg_score) / sum(n_pts) - λ * log(max(n_segments, 1))
```

- λ = 1e-3 m. Small enough that primitive-fit dominates; large enough to
  break ties in favour of more segments.
- Segments with fewer than 30 points are excluded from the average (too
  noisy to fit) but still counted in `n_segments`.
- Degenerate runs (`n_segments < 5` or `n_segments > 5000`) return penalty
  `1e6` so Optuna avoids that region.
- Trial exceptions (e.g. a knob combination that crashes RANSAC) are caught
  and also return `1e6`.

## Search space

11 knobs (mirrors `RANSAC_DEFAULTS` in `presegment_ransac.py`):

| key                        | type  | range          | default |
| -------------------------- | ----- | -------------- | ------- |
| plane_distance_threshold   | float | 0.005 – 0.10   | 0.025   |
| plane_min_inliers          | int   | 20 – 300       | 80      |
| max_planes                 | int   | 5 – 50         | 25      |
| flat_thresh                | float | 0.1 – 2.0      | 0.5     |
| cylinder_ratio_thresh      | float | 1.5 – 8.0      | 3.0     |
| cyl_search_radius          | float | 0.05 – 0.30    | 0.12    |
| cyl_axis_thresh            | float | 0.85 – 0.99    | 0.92    |
| cyl_radius_ratio           | float | 1.2 – 3.0      | 1.8     |
| cyl_distance_threshold     | float | 0.005 – 0.10   | 0.03    |
| merge_axis_dot             | float | 0.85 – 0.99    | 0.95    |
| merge_radius_ratio         | float | 1.1 – 3.0      | 1.4     |

Sampler: Optuna TPE (default). 20 trials. The first 5 are random (Optuna's
n_startup_trials default). Distance/radius knobs use `log=True` to spread
samples across orders of magnitude.

## Speed strategy

Each trial runs on a fixed random subsample of 200k points (or all points
if the cloud is smaller). The same subsample is reused across all trials so
scores are comparable. After the study finishes, the winner is re-run on the
full cloud and stored in `_state["seg"]` like any normal preseg.

Wall-time target: ~2 min for 20 trials at 200k points (~5s/trial), plus
~30s for the final full-cloud apply.

## Backend

### New module: `backend/preseg_optimize.py`

```python
def score_segmentation(xyz: np.ndarray, instance_ids: np.ndarray,
                       *, min_seg_pts: int = 30,
                       lambda_seg: float = 1e-3) -> float: ...

def run_study(xyz_sub: np.ndarray, *, n_trials: int = 20,
              cancel_event: threading.Event,
              progress_cb: Callable[[dict], None],
              class_map: dict[str, int]) -> dict:
    """Returns {best_params: dict, best_score: float, n_trials_run: int,
                 trials: list[{params, score}]}."""
```

`run_study` constructs the `optuna.Study`, adds a callback that:

1. Calls `progress_cb({trial: i, total: N, best_score, best_params})`.
2. Calls `study.stop()` if `cancel_event.is_set()`.

### State

```python
# in main.py _state
_state["preseg_opt_job"] = {
    "id": "<uuid>",
    "thread": Thread,
    "cancel": threading.Event(),
    "status": "running" | "done" | "aborted" | "error",
    "trial": int, "total": int,
    "best_score": float | None,
    "best_params": dict | None,
    "error": str | None,
    "started_at": float,
}
```

Single global job. Starting a second one while one runs returns `409`.

### Endpoints

```
POST  /api/segment/presegment/optimize
  body: {n_trials?: int (default 20), subsample_n?: int (default 200000),
         preserve_labeled?: bool (default true)}
  -> {job_id: str, total: int}
  errors: 409 if a job is already running, 409 if no scene loaded

GET   /api/segment/presegment/optimize/status?job_id=<id>
  -> {status, trial, total, best_score, best_params, error?}
  errors: 404 if job_id unknown

POST  /api/segment/presegment/optimize/abort?job_id=<id>
  -> {status: "aborting"}
```

When a job hits `status="done"` the worker has already updated `_state["seg"]`
with the full-cloud RANSAC result for `best_params`. Frontend then calls the
existing `/api/segment/state` to hydrate.

### Tests (`backend/tests/test_preseg_optimize.py`)

1. `test_score_low_for_pure_primitives` — synthesize a 5k-pt cloud: one plane
   + one cylinder, instance ids assigning each to its own segment. Assert
   `score_segmentation` < 0.01.
2. `test_score_high_for_mixed_segment` — same cloud but assign all points to
   one segment. Assert score is much larger (≥ 5× the pure case).
3. `test_score_penalises_too_few_segments` — 1 segment in the cloud →
   `n_segments < 5` triggers the `1e6` penalty.
4. `test_run_study_smoke` — tiny synthetic cloud, `n_trials=3`, no
   cancellation; returns `n_trials_run == 3` and a `best_params` dict with
   all 11 keys.
5. `test_endpoint_lifecycle` — start → poll a few times → abort; final
   status is `aborted` and `_state["seg"]` is unchanged.

## Frontend

### `segment-tools.jsx` PresegmentButton

State additions:

```js
const [optJob, setOptJob]    = useState(null);   // {jobId, trial, total, bestScore}
const [optStatus, setStatus] = useState('idle'); // idle|running|done|aborted|error
```

UI changes:

- New "Optimize" button next to Run, enabled only when `mode === 'ransac'`
  and not busy.
- While `optStatus === 'running'`:
  - Replace Run/Cancel with a status line:
    `Trial {trial}/{total} · best {bestScore.toFixed(4)} · {bestNSegs} segs`
    and an "Abort" button.
  - Poll `/api/segment/presegment/optimize/status` every 1500 ms via
    `setInterval`. Clean up on unmount or status flip.
- On `done`:
  - Set the 11 RANSAC knob inputs to `best_params` so the user can see what
    won and tweak.
  - Show `Optimized: {nSegments} segs · mean size {meanSize} pts · score
    {bestScore.toFixed(4)}` (reuse the existing stats line).
  - Reload segment state via existing `VoxaAPI.segState()` and hydrate
    `setSegState` / `setCloud` (same path as the post-run preseg flow).

### `api.js`

Three thin wrappers: `segPresegOptimizeStart`, `segPresegOptimizeStatus`,
`segPresegOptimizeAbort`.

## Dependencies

- `optuna` added to `backend/requirements.txt`. Pure Python, no compiled
  deps. ~3 MB install.

## Risks and mitigations

- **Optuna study state bloat** — kept in-process for one scene; discarded
  when the next job starts. No persistence.
- **Subsample bias** — best params on 200k may differ slightly from full.
  Acceptable: the final apply runs on full cloud and the user can still
  tweak knobs after. If it becomes an issue, surface `subsample_n` as a UI
  knob.
- **Long full-cloud apply blocks the request** — the final apply runs in
  the worker thread, not the request handler, so polling continues to
  return progress until the apply finishes (status flips to `done` only
  after the session is updated).
- **Concurrent presegmentation** — disable the regular Run button while
  optimization is running; the existing `disabled = !cloud || busy` gate
  is extended.

## Out of scope (future)

- Persisting study history per scene to resume / warm-start.
- Multi-objective optimization (Pareto front of n_segments vs RMS).
- Re-using primitive fits across trials (cache plane/cylinder fits keyed by
  point subset hash).
