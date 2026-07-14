import base64
import httpx
import numpy as np

class _Src:                       # stub of the resolved SceneSource
    extras = {"source_laz_path": "/x.laz"}
    source_path = "/y.ply"

def _fake_post(url, json, **kw):  # stub sidecar: /capture then /project
    class R:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            if url.endswith("/capture"):
                return {"capture_id": "c1", "overlay_png_b64": "x",
                        "masks": [{"mask_id": 0, "score": 0.9}]}
            idx = np.array([0, 1, 2], np.int32)
            return {"instances": [{"mask_id": 0,
                    "scan_indices_b64": base64.b64encode(idx.tobytes()).decode()}]}
    return R()

def test_capture_then_project(client_with_loaded_annotated_scene, monkeypatch):
    client = client_with_loaded_annotated_scene
    monkeypatch.setenv("VOXA_SAM_SIDECAR_URL", "http://side")
    monkeypatch.setattr("routes.sam.httpx.post", _fake_post)
    monkeypatch.setattr("routes.sam._resolve", lambda scene: _Src())
    cam = {"pos": [0,0,0], "target": [0,0,1], "fov": 60, "W": 128, "H": 128}
    r = client.post("/api/sam/capture", json={"camera": cam, "mode": "box", "box": [0.5,0.5,0.4,0.4]})
    assert r.status_code == 200 and r.json()["capture_id"] == "c1"
    r2 = client.post("/api/sam/project",
                     json={"capture_id": "c1", "mask_ids": [0], "protect_instances": []})
    assert r2.status_code == 200
    body = r2.json()
    assert "segments" in body and "instances" not in body
    seg = body["segments"][0]
    assert seg["mask_id"] == 0
    assert seg["sam_seg_id"] == 0
    assert seg["n_affected"] == 3
    assert "scan_indices_b64" in seg

def test_project_materializes_candidates_with_source_sam(client_with_loaded_annotated_scene, monkeypatch):
    import main
    client = client_with_loaded_annotated_scene
    monkeypatch.setenv("VOXA_SAM_SIDECAR_URL", "http://side")
    monkeypatch.setattr("routes.sam.httpx.post", _fake_post)
    monkeypatch.setattr("routes.sam._resolve", lambda scene: _Src())
    cam = {"pos": [0,0,0], "target": [0,0,1], "fov": 60, "W": 128, "H": 128}
    client.post("/api/sam/capture", json={"camera": cam, "mode": "box", "box": [0.5,0.5,0.4,0.4]})
    r2 = client.post("/api/sam/project",
                     json={"capture_id": "c1", "mask_ids": [0], "protect_instances": []})
    sam_seg_id = r2.json()["segments"][0]["sam_seg_id"]
    seg = main._state["seg"]
    assert seg.sam_segments[sam_seg_id]["source"] == "sam"

def test_project_does_not_call_apply_reassign(client_with_loaded_annotated_scene, monkeypatch):
    import main
    client = client_with_loaded_annotated_scene
    monkeypatch.setenv("VOXA_SAM_SIDECAR_URL", "http://side")
    monkeypatch.setattr("routes.sam.httpx.post", _fake_post)
    monkeypatch.setattr("routes.sam._resolve", lambda scene: _Src())
    cam = {"pos": [0,0,0], "target": [0,0,1], "fov": 60, "W": 128, "H": 128}
    client.post("/api/sam/capture", json={"camera": cam, "mode": "box", "box": [0.5,0.5,0.4,0.4]})
    seg = main._state["seg"]
    before = seg.instance_ids.copy()
    client.post("/api/sam/project", json={"capture_id": "c1", "mask_ids": [0], "protect_instances": []})
    assert (seg.instance_ids == before).all()   # untouched — only sam_ids changed
    assert int(seg.sam_ids[0]) >= 0

def test_capture_applies_georef_offset_zup(client_with_loaded_annotated_scene, monkeypatch):
    client = client_with_loaded_annotated_scene
    monkeypatch.setenv("VOXA_SAM_SIDECAR_URL", "http://side")
    class _SrcZUp(_Src):
        extras = {"source_laz_path": "/x.laz", "is_z_up": True}
    monkeypatch.setattr("routes.sam._resolve", lambda scene: _SrcZUp())
    import routes.sam as sam_route
    monkeypatch.setitem(sam_route._state, "raw_georef_offset_m", [100.0, 200.0, 5.0])  # x,y,z zup
    monkeypatch.setitem(sam_route._state, "recenter_offset", [1.0, 1.0, 1.0])           # x,y,z yup
    sent = {}
    def _capture_post(url, json, **kw):
        sent.update(json)
        return _fake_post(url, json, **kw)
    monkeypatch.setattr("routes.sam.httpx.post", _capture_post)
    cam = {"pos": [0,0,0], "target": [0,0,1], "fov": 60, "W": 128, "H": 128}
    client.post("/api/sam/capture", json={"camera": cam, "mode": "box", "box": [0.5,0.5,0.4,0.4]})
    # off = recenter[1,1,1]_yup + georef[100,200,5]_zup rotated to yup [100,5,-200] = [101,6,-199]_yup
    # pos [0,0,0] + off = [101,6,-199]_yup -> rotated back to native Z-up (x,-z,y) -> [101,199,6]
    assert sent["camera"]["pos"] == [101.0, 199.0, 6.0]
    # target [0,0,1] + off = [101,6,-198]_yup -> (x,-z,y) -> [101,198,6]
    assert sent["camera"]["target"] == [101.0, 198.0, 6.0]
    assert sent["scan_ply_offset_m"] == [100.0, 200.0, 5.0]   # sidecar wants raw/native order

def test_missing_sidecar_url_503(client_with_loaded_annotated_scene, monkeypatch):
    client = client_with_loaded_annotated_scene
    monkeypatch.delenv("VOXA_SAM_SIDECAR_URL", raising=False)
    cam = {"pos": [0,0,0], "target": [0,0,1], "fov": 60, "W": 128, "H": 128}
    r = client.post("/api/sam/capture", json={"camera": cam, "mode": "box", "box": [0.5,0.5,0.4,0.4]})
    assert r.status_code == 503

def test_sidecar_connection_error_502(client_with_loaded_annotated_scene, monkeypatch):
    client = client_with_loaded_annotated_scene
    monkeypatch.setenv("VOXA_SAM_SIDECAR_URL", "http://side")
    monkeypatch.setattr("routes.sam._resolve", lambda scene: _Src())
    def _boom(url, json, **kw):
        raise httpx.ConnectError("boom")
    monkeypatch.setattr("routes.sam.httpx.post", _boom)
    cam = {"pos": [0,0,0], "target": [0,0,1], "fov": 60, "W": 128, "H": 128}
    r = client.post("/api/sam/capture", json={"camera": cam, "mode": "box", "box": [0.5,0.5,0.4,0.4]})
    assert r.status_code == 502

def test_sidecar_409_passthrough(client_with_loaded_annotated_scene, monkeypatch):
    client = client_with_loaded_annotated_scene
    monkeypatch.setenv("VOXA_SAM_SIDECAR_URL", "http://side")
    monkeypatch.setattr("routes.sam._resolve", lambda scene: _Src())
    def _post_409(url, json, **kw):
        class Resp:
            status_code = 409
            text = "diverged"
            def json(self): return {"detail": {"diverged": "source"}}
        class R:
            status_code = 409
            def raise_for_status(self):
                raise httpx.HTTPStatusError("409", request=None, response=Resp())
        return R()
    monkeypatch.setattr("routes.sam.httpx.post", _post_409)
    cam = {"pos": [0,0,0], "target": [0,0,1], "fov": 60, "W": 128, "H": 128}
    r = client.post("/api/sam/capture", json={"camera": cam, "mode": "box", "box": [0.5,0.5,0.4,0.4]})
    assert r.status_code == 409
