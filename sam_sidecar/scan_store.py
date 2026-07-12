"""One loaded scan at a time, keyed by source fingerprint. The cross-scan
identity guard: same scan_id + different fingerprint is a hard error so
/project can never return indices for the wrong cloud."""
from __future__ import annotations
from typing import Callable

class FingerprintMismatch(Exception):
    pass

class ScanStore:
    def __init__(self, loader: Callable[[str], tuple]):
        self._loader = loader
        self.scan_id = None; self.fingerprint = None
        self.raw_xyz = None; self.raw_rgb = None; self.scan_xyz = None

    def ensure(self, scan_id: str, fingerprint: str, raw_laz_path: str = None,
               scan_ply_path: str = None) -> None:
        if self.scan_id == scan_id:
            if self.fingerprint != fingerprint:
                raise FingerprintMismatch(
                    f"scan '{scan_id}' loaded at fingerprint {self.fingerprint}, "
                    f"request carried {fingerprint}")
            return
        self.raw_xyz, self.raw_rgb, self.scan_xyz = self._loader(scan_id, raw_laz_path, scan_ply_path)
        self.scan_id = scan_id; self.fingerprint = fingerprint
