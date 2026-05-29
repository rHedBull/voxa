"""Multi-run collections for ``labels/`` and ``prelabel/`` (scan-schema v1.3 §4.6).

A collection dir holds ``runs/<run_id>/`` (per-point ``*.npy`` + ``run.json``) and a
generated ``runs.json`` index with a ``default_run`` (the legacy single-slot alias
target). Generic over the array names so it serves both labels (``gt_class_ids`` /
``gt_segment_ids``) and prelabel (``instance_ids``) collections.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _runs_dir(collection: Path) -> Path:
    return Path(collection) / "runs"


def write_run(collection: Path, run_id: str, arrays: dict, run_meta: dict) -> Path:
    d = _runs_dir(collection) / run_id
    d.mkdir(parents=True, exist_ok=True)
    for name, arr in arrays.items():
        np.save(d / f"{name}.npy", np.asarray(arr))
    meta = dict(run_meta)
    meta["run_id"] = run_id
    (d / "run.json").write_text(json.dumps(meta, indent=2))
    return d


def read_run(collection: Path, run_id: str):
    d = _runs_dir(collection) / run_id
    run_meta = json.loads((d / "run.json").read_text())
    arrays = {p.stem: np.load(p) for p in sorted(d.glob("*.npy"))}
    return arrays, run_meta


def list_runs(collection: Path) -> list[dict]:
    rd = _runs_dir(collection)
    if not rd.exists():
        return []
    out = []
    for d in sorted(rd.iterdir()):
        rj = d / "run.json"
        if rj.exists():
            out.append(json.loads(rj.read_text()))
    return out


def write_runs_index(collection: Path, default_run: str | None = None) -> dict:
    idx = {"default_run": default_run, "runs": list_runs(collection)}
    (Path(collection) / "runs.json").write_text(json.dumps(idx, indent=2))
    return idx


def read_runs_index(collection: Path) -> dict:
    p = Path(collection) / "runs.json"
    return json.loads(p.read_text()) if p.exists() else {"default_run": None, "runs": []}
