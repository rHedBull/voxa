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
                     json={"capture_id": "c1", "mask_ids": [0], "target_class": "pipe", "protect_instances": []})
    assert r2.status_code == 200
    assert r2.json()["instances"][0]["n_affected"] >= 0

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

def test_bad_class_id_400(client_with_loaded_annotated_scene, monkeypatch):
    client = client_with_loaded_annotated_scene
    monkeypatch.setenv("VOXA_SAM_SIDECAR_URL", "http://side")
    monkeypatch.setattr("routes.sam.httpx.post", _fake_post)
    monkeypatch.setattr("routes.sam._resolve", lambda scene: _Src())
    cam = {"pos": [0,0,0], "target": [0,0,1], "fov": 60, "W": 128, "H": 128}
    client.post("/api/sam/capture", json={"camera": cam, "mode": "box", "box": [0.5,0.5,0.4,0.4]})
    r = client.post("/api/sam/project",
                    json={"capture_id": "c1", "mask_ids": [0], "target_class": "not_a_class", "protect_instances": []})
    assert r.status_code == 400

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
