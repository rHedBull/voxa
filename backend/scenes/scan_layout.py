"""Canonical on-disk layout of an annotated scan (scan-schema v1.3 and v2).

Single executable encoding of the directory contract documented in
``docs/scan-schema.md`` (and the cross-tool ``lidar/SCHEMA.md``). Every voxa
module that needs a path *inside* an annotated scan dir resolves it here, so the
layout is defined once instead of being re-derived (``parent.parent`` +
hard-coded subpaths) across the loader, the registry, the save route, and
segment_io. Sibling tools that share the ``lidar/`` archive follow the same
SCHEMA.md; this class is the piece you would lift into a shared package if that
ever becomes worth it.

All properties are pure path joins — nothing is required to exist on disk.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SessionPaths:
    """Paths inside sessions/<session_id>/ (scan-schema v2)."""
    dir: Path

    @property
    def session_json(self) -> Path:
        return self.dir / "session.json"

    @property
    def working_class_ids(self) -> Path:
        return self.dir / "working_class_ids.npy"

    @property
    def working_segment_ids(self) -> Path:
        return self.dir / "working_segment_ids.npy"

    @property
    def output_dir(self) -> Path:
        return self.dir / "output"

    @property
    def output_gt_class_ids(self) -> Path:
        return self.output_dir / "gt_class_ids.npy"

    @property
    def output_gt_segment_ids(self) -> Path:
        return self.output_dir / "gt_segment_ids.npy"

    @property
    def output_gt_segment_metadata(self) -> Path:
        return self.output_dir / "gt_segment_metadata.json"

    @property
    def history_dir(self) -> Path:
        return self.dir / "history"


@dataclass(frozen=True)
class ScanLayout:
    """Paths under ``annotated/<scan_name>/``. ``scan_dir`` is that directory."""

    scan_dir: Path

    # source/
    @property
    def source_dir(self) -> Path:
        return self.scan_dir / "source"

    @property
    def scan_ply(self) -> Path:
        return self.source_dir / "scan.ply"

    @property
    def mesh_glb(self) -> Path:
        return self.source_dir / "mesh.glb"

    # labels/
    @property
    def labels_dir(self) -> Path:
        return self.scan_dir / "labels"

    @property
    def gt_class_ids(self) -> Path:
        return self.labels_dir / "gt_class_ids.npy"

    @property
    def gt_segment_ids(self) -> Path:
        return self.labels_dir / "gt_segment_ids.npy"

    @property
    def gt_segment_metadata(self) -> Path:
        return self.labels_dir / "gt_segment_metadata.json"

    # prelabel/
    @property
    def prelabel_dir(self) -> Path:
        return self.scan_dir / "prelabel"

    @property
    def ransac_instance_ids(self) -> Path:
        return self.prelabel_dir / "ransac_instance_ids.npy"

    @property
    def ransac_segment_summary(self) -> Path:
        return self.prelabel_dir / "ransac_segment_summary.json"

    # session/ renders/ sam3/ annotation_history/
    @property
    def session_dir(self) -> Path:
        return self.scan_dir / "session"

    @property
    def renders_dir(self) -> Path:
        return self.scan_dir / "renders"

    @property
    def sam3_dir(self) -> Path:
        return self.scan_dir / "sam3"

    @property
    def annotation_history_dir(self) -> Path:
        return self.scan_dir / "annotation_history"

    # top-level
    @property
    def meta_json(self) -> Path:
        return self.scan_dir / "meta.json"

    # archive-level: classes.json lives at the lidar root (scan_dir's grandparent:
    # <lidar_root>/annotated/<scan>/ -> <lidar_root>/classes.json) and is shared by
    # every scan, so all scans reference the same class IDs.
    @property
    def classes_json(self) -> Path:
        return self.scan_dir.parent.parent / "classes.json"

    # v2: prelabel/<preseg_id>/ + sessions/<session_id>/
    @property
    def presegs_root(self) -> Path:
        return self.scan_dir / "prelabel"

    def preseg_dir(self, preseg_id: str) -> Path:
        return self.presegs_root / preseg_id

    @property
    def sessions_root(self) -> Path:
        return self.scan_dir / "sessions"

    def session(self, session_id: str) -> SessionPaths:
        return SessionPaths(self.sessions_root / session_id)
