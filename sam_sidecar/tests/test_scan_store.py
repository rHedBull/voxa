import numpy as np, pytest
from scan_store import ScanStore, FingerprintMismatch

def _fake_loader(scan_id):
    xyz = np.zeros((10, 3), np.float32); rgb = np.zeros((10, 3), np.uint8)
    return xyz, rgb, xyz.copy()   # raw_xyz, raw_rgb, scan_xyz

def test_loads_and_serves_matching_scan():
    s = ScanStore(loader=_fake_loader)
    s.ensure("scanA", "fp1")
    assert s.scan_id == "scanA" and s.fingerprint == "fp1"
    assert s.scan_xyz.shape == (10, 3)

def test_different_scan_id_reloads():
    calls = []
    def loader(sid): calls.append(sid); return _fake_loader(sid)
    s = ScanStore(loader=loader)
    s.ensure("scanA", "fp1"); s.ensure("scanB", "fp2")
    assert s.scan_id == "scanB" and calls == ["scanA", "scanB"]

def test_same_id_different_fingerprint_raises():
    s = ScanStore(loader=_fake_loader)
    s.ensure("scanA", "fp1")
    with pytest.raises(FingerprintMismatch):
        s.ensure("scanA", "fp-DIFFERENT")
