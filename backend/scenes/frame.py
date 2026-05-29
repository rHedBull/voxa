"""Coordinate-frame model for scan-schema v1.3 (§3.1, §4.1)."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class Frame:
    transform_to_canonical: np.ndarray   # 4x4, maps this variant's coords -> canonical
    canonical_id: str                    # "<scan_id>#local"
    units: str = "meters"
    georef: Optional[dict] = None        # {"crs": ..., "offset_m": [x,y,z]} canonical-local -> world
    frame_uncertain: bool = False        # synthesized from legacy coords -> force the §6 check

    def to_dict(self) -> dict:
        d = {
            "canonical_id": self.canonical_id,
            "transform_to_canonical": np.asarray(self.transform_to_canonical).tolist(),
            "units": self.units,
            "frame_uncertain": self.frame_uncertain,
        }
        if self.georef is not None:
            d["georef"] = self.georef
        return d


def is_rigid(M, atol: float = 1e-6) -> bool:
    M = np.asarray(M, dtype=np.float64)
    if M.shape != (4, 4):
        return False
    R = M[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=atol):
        return False
    if not np.isclose(abs(np.linalg.det(R)), 1.0, atol=atol):
        return False
    return np.allclose(M[3], [0, 0, 0, 1], atol=atol)


def frame_from_dict(d: dict) -> Frame:
    M = np.asarray(d["transform_to_canonical"], dtype=np.float64)
    if M.shape != (4, 4):
        raise ValueError(f"transform_to_canonical must be 4x4, got {M.shape}")
    return Frame(M, d["canonical_id"], d.get("units", "meters"),
                 d.get("georef"), bool(d.get("frame_uncertain", False)))


def compose_a_to_v(a: Frame, v: Frame) -> np.ndarray:
    """4x4 mapping a point/pose expressed in frame ``a`` into frame ``v``:
    apply a->canonical, then canonical->v  ==  inv(v.T_can) @ a.T_can."""
    return np.linalg.inv(v.transform_to_canonical) @ a.transform_to_canonical


def apply_transform(T: np.ndarray, pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=np.float64)
    homo = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    return (T @ homo.T).T[:, :3]
