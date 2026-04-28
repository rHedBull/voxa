"""Per-scene editing-session state.

One SegmentSession instance is held in main._state per loaded cloud.
All mutations are recorded as inverse-deltas on a bounded undo stack so
undo/redo can replay without re-deriving anything.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
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
        self.positions = positions.astype(np.float32, copy=False)
        self._undo: deque[_Delta] = deque()
        self._redo: deque[_Delta] = deque()
        self.history_cap: int = 100
        self.dirty: bool = False

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
        return self._delta_payload(d, direction="undo")

    def redo(self) -> Optional[dict]:
        if not self._redo:
            return None
        d = self._redo.pop()
        self.class_ids[d.indices] = d.after_cls
        self.instance_ids[d.indices] = d.after_inst
        self._undo.append(d)
        self.dirty = True
        return self._delta_payload(d, direction="redo")

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
                    new_inst_id = int(self.instance_ids.max(initial=-1)) + 1
                    self.instance_ids[indices] = np.int32(new_inst_id)
                else:
                    self.instance_ids[indices] = np.int32(ti)
                self.class_ids[indices] = np.int8(tc)
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
        return out

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
