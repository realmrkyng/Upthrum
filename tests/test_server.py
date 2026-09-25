
import numpy as np
import pytest

fastapi = pytest.importorskip("fastapi", reason="server extra not installed")
pytest.importorskip("httpx", reason="TestClient needs httpx")

from fastapi.testclient import TestClient  # noqa: E402

from pixelboost import imageio  # noqa: E402
from pixelboost.config import Config  # noqa: E402
from pixelboost.server.app import create_app  # noqa: E402


def png_bytes(h=24, w=32):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    img = np.stack([x / w, y / h, np.full((h, w), 0.5)], -1)
    return imageio.encode(img.astype(np.float32), fmt="PNG")


def make_client(**server_overrides):
    cfg = Config()
    cfg.warmup = False
    cfg.backend = "classical"
    for key, value in server_overrides.items():
        setattr(cfg.server, key, value)
    return TestClient(create_app(cfg)), cfg


def test_healthz_needs_no_auth():
    client, _ = make_client()
    with client:
        response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_capabilities_reports_backends():
    client, _ = make_client()
    with client:
        data = client.get("/v1/capabilities").json()
    assert "classical" in data["backends"]
    assert any(m["name"] == "realesrgan-x4plus" for m in data["models"])
    assert data["version"]


def test_api_key_is_enforced():
    client, _ = make_client(api_keys=["sk-test"])
    with client:
        assert client.get("/v1/capabilities").status_code == 401
        assert client.get("/v1/capabilities", headers={"X-API-Key": "wrong"}).status_code == 401
        ok = client.get("/v1/capabilities", headers={"X-API-Key": "sk-test"})
    assert ok.status_code == 200


def test_api_key_via_query_string():
    client, _ = make_client(api_keys=["sk-test"])
    with client:
        response = client.get("/v1/capabilities?api_key=sk-test")
    assert response.status_code == 200


def test_enhance_multipart_returns_image():
    client, _ = make_client()
    with client:
        response = client.post(
            "/v1/enhance",
            files={"file": ("in.png", png_bytes(), "image/png")},
            data={"scale": "2", "backend": "classical"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.headers["X-PixelBoost-Backend"] == "classical"
    assert int(response.headers["X-PixelBoost-Tiles"]) >= 1

    loaded = imageio.load(response.content)
    assert loaded.meta.width == 64
    assert loaded.meta.height == 48


def test_enhance_json_envelope():
    client, _ = make_client()
    with client:
        response = client.post(
            "/v1/enhance?response=json&scale=2",
            content=png_bytes(),
            headers={"Content-Type": "image/png"},
        )
    assert response.status_code == 200
    payload = response.json()
    assert payload["format"] == "png"
    assert payload["meta"]["src_size"] == [32, 24]
    assert payload["meta"]["input"]["format"] == "PNG"
    assert len(payload["image_base64"]) > 100


def test_enhance_rejects_garbage():
    client, _ = make_client()
    with client:
        response = client.post(
            "/v1/enhance",
            files={"file": ("x.png", b"definitely not a png", "image/png")},
        )
    assert response.status_code == 415


def test_enhance_rejects_empty_body():
    client, _ = make_client()
    with client:
        response = client.post("/v1/enhance", content=b"")
    assert response.status_code == 400


def test_upload_limit_is_enforced():
    client, _ = make_client(max_upload_mb=0)
    with client:
        response = client.post(
            "/v1/enhance",
            files={"file": ("in.png", png_bytes(), "image/png")},
        )
    assert response.status_code == 413
    assert "limit" in response.json()["detail"]


def test_missing_file_field():
    client, _ = make_client()
    with client:
        response = client.post("/v1/enhance", files={"notfile": ("a.png", b"x", "image/png")})
    assert response.status_code == 400


def test_job_lifecycle():
    client, _ = make_client()
    with client:
        submitted = client.post(
            "/v1/jobs",
            files={"file": ("in.png", png_bytes(), "image/png")},
            data={"scale": "2", "backend": "classical", "output_format": "png"},
        )
        assert submitted.status_code == 202
        job_id = submitted.json()["id"]

        state = client.get(f"/v1/jobs/{job_id}").json()
        assert state["status"] in ("queued", "running", "succeeded")
        assert "progress" in state

        for _ in range(200):
            state = client.get(f"/v1/jobs/{job_id}").json()
            if state["status"] in ("succeeded", "failed"):
                break
        assert state["status"] == "succeeded", state.get("error")

        result = client.get(f"/v1/jobs/{job_id}/result")
        assert result.status_code == 200
        assert result.headers["content-type"] == "image/png"
        assert len(result.content) > 100


def test_unknown_job_is_404():
    client, _ = make_client()
    with client:
        assert client.get("/v1/jobs/deadbeef").status_code == 404
        assert client.get("/v1/jobs/deadbeef/result").status_code == 404


def test_delete_unknown_job_is_409():
    client, _ = make_client()
    with client:
        assert client.delete("/v1/jobs/deadbeef").status_code == 409


def test_stats_reports_queue():
    client, _ = make_client(workers=1)
    with client:
        data = client.get("/v1/stats").json()
    assert data["jobs"]["workers"] == 1


def test_unknown_parameters_are_ignored():
    """Forward compatibility: a client sending a future flag must not 4xx."""
    client, _ = make_client()
    with client:
        response = client.post(
            "/v1/enhance",
            files={"file": ("in.png", png_bytes(), "image/png")},
            data={"scale": "2", "backend": "classical", "some_future_flag": "yes"},
        )
    assert response.status_code == 200
