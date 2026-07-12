import base64, numpy as np
from fastapi.testclient import TestClient

def _make_client(monkeypatch):
    import main
    g = np.linspace(-1, 1, 60); xx, zz = np.meshgrid(g, g)
    wall = np.column_stack([xx.ravel(), np.zeros(xx.size), zz.ravel()]).astype(np.float32)
    rgb = np.tile(np.array([180,180,180], np.uint8), (wall.shape[0],1))
    main.STORE._loader = lambda sid, *_: (wall, rgb, wall.copy())   # ignores paths
    monkeypatch.setattr(main, "_sam_box", lambda img, box, text: (np.ones(img.size[::-1], bool), 0.9))
    return TestClient(main.app)

_PATHS = {"raw_laz_path": "x.laz", "scan_ply_path": "y.ply"}

def test_capture_then_project(monkeypatch):
    c = _make_client(monkeypatch)
    cam = {"pos": [0,-5,0], "target": [0,0,0], "fov": 60.0, "W": 128, "H": 128}
    r = c.post("/capture", json={"scan_id":"A","source_fingerprint":"fp",**_PATHS,"camera":cam,
                                 "mode":"box","box":[0.5,0.5,0.6,0.6]})
    assert r.status_code == 200
    cap = r.json(); assert cap["masks"] and "capture_id" in cap
    r2 = c.post("/project", json={"scan_id":"A","source_fingerprint":"fp",
                                  "capture_id":cap["capture_id"],"mask_ids":[cap["masks"][0]["mask_id"]]})
    assert r2.status_code == 200
    inst = r2.json()["instances"][0]
    sel = np.frombuffer(base64.b64decode(inst["scan_indices_b64"]), np.int32)
    assert sel.size > 0

def test_stale_capture_id_409(monkeypatch):
    c = _make_client(monkeypatch)
    cam = {"pos":[0,-5,0],"target":[0,0,0],"fov":60.0,"W":128,"H":128}
    c.post("/capture", json={"scan_id":"A","source_fingerprint":"fp",**_PATHS,"camera":cam,"mode":"box","box":[0.5,0.5,0.6,0.6]})
    r = c.post("/project", json={"scan_id":"A","source_fingerprint":"fp","capture_id":"stale","mask_ids":[0]})
    assert r.status_code == 409

def test_fingerprint_mismatch_409(monkeypatch):
    c = _make_client(monkeypatch)
    cam = {"pos":[0,-5,0],"target":[0,0,0],"fov":60.0,"W":128,"H":128}
    c.post("/capture", json={"scan_id":"A","source_fingerprint":"fp",**_PATHS,"camera":cam,"mode":"box","box":[0.5,0.5,0.6,0.6]})
    r = c.post("/capture", json={"scan_id":"A","source_fingerprint":"DIFF",**_PATHS,"camera":cam,"mode":"box","box":[0.5,0.5,0.6,0.6]})
    assert r.status_code == 409 and r.json()["detail"].get("diverged") == "source"
