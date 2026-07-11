"""Per-scene editing-session state.

One SegmentSession instance is held in main._state per loaded cloud.
All mutations are recorded as inverse-deltas on a bounded undo stack so
undo/redo can replay without re-deriving anything.
"""
from __future__ import annotations

import threading
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class _Delta:
    op: str
    indices: np.ndarray            # int32
    before_cls: np.ndarray         # int8
    before_inst: np.ndarray        # int32
    after_cls: np.ndarray          # int8
    after_inst: np.ndarray         # int32


class SegmentSession:
    """In-memory editor state for one scene.

    Class arrays are int8 (voxa class count is small); instance arrays are
    int32. Both are full-resolution, length == positions.shape[0].
    """

    def __init__(
        self,
        class_ids: np.ndarray,
        instance_ids: np.ndarray,
        positions: np.ndarray,
        *,
        is_from_prelabel: bool = False,
        session_dir: Optional[Path] = None,
        autosave_debounce_s: float = 0.25,
    ) -> None:
        if class_ids.dtype != np.int8:
            class_ids = class_ids.astype(np.int8)
        if instance_ids.dtype != np.int32:
            instance_ids = instance_ids.astype(np.int32)
        n = positions.shape[0]
        if class_ids.shape != (n,) or instance_ids.shape != (n,):
            raise ValueError("class/instance/position lengths disagree")
        self.class_ids = class_ids
        self.instance_ids = instance_ids
        # Fresh-instance ids are session-monotonic: this floor never
        # decreases, so an id freed by undo is never re-issued to a later
        # apply (the frontend's instance doc may still reference it).
        self._next_fresh_inst = int(instance_ids.max(initial=-1)) + 1
        self.positions = positions.astype(np.float32, copy=False)
        self.preseg_ids: np.ndarray = np.full(n, -1, dtype=np.int32)
        self.preseg_id: Optional[str] = None
        self.preseg_fingerprint: Optional[str] = None
        self.source_fingerprint: Optional[str] = None
        self.hidden_inst_ids: set[int] = set()
        self.is_from_prelabel: bool = bool(is_from_prelabel)
        self.name: str = ""
        self.created_at: Optional[str] = None
        self._undo: deque[_Delta] = deque()
        self._redo: deque[_Delta] = deque()
        self.history_cap: int = 100
        self.dirty: bool = False
        self.session_dir: Optional[Path] = Path(session_dir) if session_dir else None
        self._autosave_debounce_s = float(autosave_debounce_s)
        self._autosave_timer: Optional[threading.Timer] = None
        self._autosave_lock = threading.Lock()

    # ── Public ops ──

    def apply_set_class(self, indices: np.ndarray, class_id: int) -> dict:
        return self._apply("set_class", indices, dict(class_id=int(class_id)))

    def apply_merge(self, source_inst: int, target_inst: int) -> dict:
        # Merge: every point with instance==source becomes target. Class is
        # taken from the target's existing class. Source must exist.
        if source_inst == target_inst:
            return {"op": "merge", "n_affected": 0}
        idx = np.flatnonzero(self.instance_ids == source_inst).astype(np.int32)
        if idx.size == 0:
            return {"op": "merge", "n_affected": 0}
        # Resolve target class id (might be -1 if target has no points; rare).
        tgt_mask = self.instance_ids == target_inst
        if tgt_mask.any():
            target_class = int(self.class_ids[tgt_mask][0])
        else:
            target_class = int(self.class_ids[idx][0])
        return self._apply("merge", idx, dict(target_inst=int(target_inst),
                                              target_class=target_class))

    def apply_reassign(
        self, indices: np.ndarray,
        target_inst: Optional[int],
        target_class: Optional[int],
    ) -> dict:
        """Brush op. target_inst=None + target_class=None → erase to (-1, -1).
        target_inst<0 + target_class>=0 → allocate a new instance id."""
        return self._apply("reassign", indices, dict(
            target_inst=None if target_inst is None else int(target_inst),
            target_class=None if target_class is None else int(target_class),
        ))

    def undo(self) -> Optional[dict]:
        if not self._undo:
            return None
        d = self._undo.pop()
        # Apply 'before' to current
        self.class_ids[d.indices] = d.before_cls
        self.instance_ids[d.indices] = d.before_inst
        self._redo.append(d)
        self.dirty = True
        self.schedule_autosave(write_arrays=True)
        return self._delta_payload(d, direction="undo")

    def redo(self) -> Optional[dict]:
        if not self._redo:
            return None
        d = self._redo.pop()
        self.class_ids[d.indices] = d.after_cls
        self.instance_ids[d.indices] = d.after_inst
        self._undo.append(d)
        self.dirty = True
        self.schedule_autosave(write_arrays=True)
        return self._delta_payload(d, direction="redo")

    # ── Preseg layer ──
    # The immutable preseg layer (`preseg_ids` + pins) is populated by
    # app.core._resume_session from the on-disk session/preseg stores; no
    # in-memory mutator exists in v2 (sessions pin their preseg at creation).

    def current_inst_ids_for_preseg(self, preseg_id: int) -> set[int]:
        """Which live instance ids does preseg cluster `preseg_id` currently
        cover? Resolves through any merges/reassigns since the session was
        seeded."""
        mask = self.preseg_ids == int(preseg_id)
        if not mask.any():
            return set()
        return set(int(v) for v in np.unique(self.instance_ids[mask]) if v >= 0)

    def hide_instance(self, inst_id: int) -> None:
        self.hidden_inst_ids.add(int(inst_id))
        self.schedule_autosave(write_arrays=False)

    def unhide_instance(self, inst_id: int) -> None:
        self.hidden_inst_ids.discard(int(inst_id))
        self.schedule_autosave(write_arrays=False)

    def snap_to_preseg(self, inst_ids: list[int]) -> dict:
        """For every point whose live instance is in inst_ids, reset its
        instance to its preseg id (class unchanged). On the undo stack."""
        live = np.asarray(inst_ids, dtype=np.int32)
        mask = np.isin(self.instance_ids, live) & (self.preseg_ids >= 0)
        indices = np.flatnonzero(mask).astype(np.int32)
        if indices.size == 0:
            return {"op": "snap_to_preseg", "n_affected": 0}
        return self._apply(
            "snap_to_preseg", indices,
            dict(after_inst=self.preseg_ids[indices].copy()),
        )

    # ── KD-tree query ──

    def _ensure_tree(self):
        if not hasattr(self, "_tree"):
            from scipy.spatial import cKDTree
            self._tree = cKDTree(self.positions)
        return self._tree

    def brush_query(
        self,
        center: np.ndarray,
        radius: float,
        *,
        camera_ray: Optional[np.ndarray] = None,
        depth_cull: Optional[float] = None,
    ) -> np.ndarray:
        tree = self._ensure_tree()
        idx = np.array(tree.query_ball_point(center, r=float(radius)), dtype=np.int32)
        if idx.size == 0:
            return idx
        if camera_ray is not None and depth_cull is not None:
            ray = camera_ray.astype(np.float32)
            ray = ray / (np.linalg.norm(ray) + 1e-9)
            disp = self.positions[idx] - center.astype(np.float32)
            along = disp @ ray
            keep = along <= float(depth_cull)
            idx = idx[keep]
        return idx

    # ── Internal ──

    def _apply(self, op: str, indices: np.ndarray, payload: dict) -> dict:
        indices = indices.astype(np.int32, copy=False)
        before_cls = self.class_ids[indices].copy()
        before_inst = self.instance_ids[indices].copy()
        new_inst_id: Optional[int] = None

        if op == "set_class":
            self.class_ids[indices] = np.int8(payload["class_id"])
            # instance_ids unchanged
        elif op == "merge":
            self.instance_ids[indices] = np.int32(payload["target_inst"])
            self.class_ids[indices] = np.int8(payload["target_class"])
        elif op == "reassign":
            ti, tc = payload["target_inst"], payload["target_class"]
            if ti is None and tc is None:
                self.instance_ids[indices] = np.int32(-1)
                self.class_ids[indices] = np.int8(-1)
            else:
                # Non-erase reassign must specify a class — otherwise points
                # would carry a fresh/foreign instance id with their old class,
                # which can break SCHEMA invariant 4 at save.
                if tc is None:
                    raise ValueError(
                        "apply_reassign: target_class is required unless target_inst is None too (erase)",
                    )
                if ti is None or ti < 0:
                    new_inst_id = max(self._next_fresh_inst,
                                      int(self.instance_ids.max(initial=-1)) + 1)
                    self._next_fresh_inst = new_inst_id + 1
                    self.instance_ids[indices] = np.int32(new_inst_id)
                else:
                    self.instance_ids[indices] = np.int32(ti)
                self.class_ids[indices] = np.int8(tc)
        elif op == "snap_to_preseg":
            self.instance_ids[indices] = payload["after_inst"].astype(np.int32, copy=False)
            # class_ids unchanged
        else:
            raise ValueError(f"unknown op: {op}")

        delta = _Delta(
            op=op, indices=indices,
            before_cls=before_cls, before_inst=before_inst,
            after_cls=self.class_ids[indices].copy(),
            after_inst=self.instance_ids[indices].copy(),
        )
        self._undo.append(delta)
        if len(self._undo) > self.history_cap:
            self._undo.popleft()
        self._redo.clear()
        self.dirty = True

        out = {
            "op": op,
            "n_affected": int(indices.size),
            "indices": indices,
            "after_class": delta.after_cls,
            "after_instance": delta.after_inst,
        }
        if new_inst_id is not None:
            out["new_instance_id"] = new_inst_id
        self.schedule_autosave(write_arrays=True)
        return out

    # ── Autosave ──

    def _aux_payload(self) -> dict:
        return {
            "preseg_id": self.preseg_id,
            "preseg_fingerprint": self.preseg_fingerprint,
            "source_fingerprint": self.source_fingerprint,
            "hidden_inst_ids": sorted(int(x) for x in self.hidden_inst_ids),
            "dirty": bool(self.dirty),
            "name": self.name,
            "created_at": self.created_at,
        }

    @classmethod
    def from_aux(cls, aux: dict, *, class_ids, instance_ids, positions,
                 session_dir) -> "SegmentSession":
        """Inverse of _aux_payload(): rebuild in-memory session state from a
        parsed session.json. Every field written there is restored here —
        keep the two methods in lockstep."""
        seg = cls(class_ids=class_ids, instance_ids=instance_ids,
                  positions=positions,
                  is_from_prelabel=aux.get("preseg_id") is not None,
                  session_dir=session_dir)
        seg.preseg_id = aux.get("preseg_id")
        seg.preseg_fingerprint = aux.get("preseg_fingerprint")
        seg.source_fingerprint = aux.get("source_fingerprint")
        seg.name = aux.get("name") or ""
        seg.created_at = aux.get("created_at")
        seg.hidden_inst_ids = set(int(x) for x in aux.get("hidden_inst_ids", []))
        seg.dirty = bool(aux.get("dirty", False))
        return seg

    def _do_autosave(self, write_arrays: bool) -> None:
        if self.session_dir is None:
            return
        from labeling.segment_io import save_session_aux
        with self._autosave_lock:
            save_session_aux(
                self.session_dir,
                self._aux_payload(),
                class_ids=self.class_ids if write_arrays else None,
                instance_ids=self.instance_ids if write_arrays else None,
            )

    def schedule_autosave(self, *, write_arrays: bool = True) -> None:
        if self.session_dir is None:
            return
        if self._autosave_debounce_s <= 0.0:
            self._do_autosave(write_arrays)
            return
        if self._autosave_timer is not None:
            self._autosave_timer.cancel()
        self._autosave_timer = threading.Timer(
            self._autosave_debounce_s,
            self._do_autosave, args=(write_arrays,),
        )
        self._autosave_timer.daemon = True
        self._autosave_timer.start()

    def flush_autosave(self) -> None:
        if self._autosave_timer is not None:
            self._autosave_timer.cancel()
            self._autosave_timer = None
        self._do_autosave(write_arrays=True)

    def persist_aux(self) -> None:
        """Rewrite session.json (aux only, no working arrays) so a flag change
        like `dirty` reaches disk. Needed after an export clears `dirty`, since
        the preceding autosave persisted the still-True flag."""
        self._do_autosave(write_arrays=False)

    def _delta_payload(self, d: _Delta, direction: str) -> dict:
        # Frontend reapplies these by index → (class, instance).
        cls = d.before_cls if direction == "undo" else d.after_cls
        inst = d.before_inst if direction == "undo" else d.after_inst
        return {
            "op": d.op, "direction": direction,
            "indices": d.indices,
            "after_class": cls,
            "after_instance": inst,
            "n_affected": int(d.indices.size),
        }
