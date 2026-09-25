import json
import os

import pytest

from pixelboost.backends.registry import (
    available_providers,
    normalize_provider,
    resolve_backend_name,
    resolve_providers,
)
from pixelboost.config import MODEL_REGISTRY, Config, load_config
from pixelboost.errors import BackendUnavailable, ConfigError, ProviderUnavailable
from pixelboost.types import EnhanceOptions

needs_ort = pytest.mark.skipif(
    not available_providers(), reason="onnxruntime is not installed"
)


def test_registry_entries_are_well_formed():
    for name, spec in MODEL_REGISTRY.items():
        assert spec.name == name
        assert spec.scale in (1, 2, 3, 4, 8)
        assert spec.arch in ("rrdb", "srvgg")
        assert spec.filename.endswith((".pth", ".onnx"))
        assert spec.url and spec.url.startswith("https://")


def test_model_path_join():
    cfg = Config(models_dir="/tmp/models")
    spec = cfg.resolve_model("realesrgan-x4plus")
    assert cfg.model_path(spec) == os.path.join("/tmp/models", "RealESRGAN_x4plus.pth")


def test_resolve_classical_pseudo_model():
    cfg = Config()
    spec = cfg.resolve_model("classical")
    assert spec.kind == "none"
    assert spec.url is None


def test_resolve_unknown_model_raises():
    with pytest.raises(ConfigError):
        Config().resolve_model("definitely-not-a-model")


def test_resolve_local_file_path(tmp_path):
    p = tmp_path / "mine.pth"
    p.write_bytes(b"x")
    spec = Config().resolve_model(str(p))
    assert spec.kind == "pth"
    assert spec.name == "mine"


def test_config_rejects_unknown_keys():
    with pytest.raises(ConfigError):
        Config.from_dict({"nope": 1})


def test_config_merges_nested_defaults():
    cfg = Config.from_dict(
        {"tile": 256, "defaults": {"scale": 2.0, "detail": 0.8}, "server": {"port": 9001}}
    )
    assert cfg.tile == 256
    assert cfg.defaults.scale == 2.0
    assert cfg.defaults.detail == 0.8
    assert cfg.server.port == 9001


def test_load_config_yaml(tmp_path):
    path = tmp_path / "pixelboost.yaml"
    path.write_text("backend: classical\ntile: 128\ndefaults:\n  scale: 3\n", encoding="utf-8")
    cfg = load_config(str(path))
    assert cfg.backend == "classical"
    assert cfg.tile == 128
    assert cfg.defaults.scale == 3


def test_load_config_json(tmp_path):
    path = tmp_path / "pixelboost.json"
    path.write_text(json.dumps({"model": "realesr-general-x4v3"}), encoding="utf-8")
    cfg = load_config(str(path))
    assert cfg.model == "realesr-general-x4v3"


def test_load_config_missing_file(tmp_path):
    with pytest.raises(ConfigError):
        load_config(str(tmp_path / "nope.yaml"))


def test_discover_config_env(tmp_path, monkeypatch):
    path = tmp_path / "custom.yaml"
    path.write_text("tile: 64\n", encoding="utf-8")
    monkeypatch.setenv("PIXELBOOST_CONFIG", str(path))
    cfg = load_config()
    assert cfg.tile == 64
    monkeypatch.delenv("PIXELBOOST_CONFIG")


def test_provider_alias_normalisation():
    assert normalize_provider("gpu") == "CUDAExecutionProvider"
    assert normalize_provider("dml") == "DmlExecutionProvider"
    assert normalize_provider("CPU") == "CPUExecutionProvider"
    assert normalize_provider("auto") is None
    with pytest.raises(ProviderUnavailable):
        normalize_provider("quantum")


@needs_ort
def test_resolve_providers_cpu_always_last():
    providers = resolve_providers("cpu")
    names = [p[0] if isinstance(p, tuple) else p for p in providers]
    assert "CPUExecutionProvider" in names


@needs_ort
def test_resolve_providers_gpu_alias_maps_to_cuda():
    if "CUDAExecutionProvider" not in available_providers():
        pytest.skip("CUDA provider not present in this onnxruntime build")
    providers = resolve_providers("gpu")
    names = [p[0] if isinstance(p, tuple) else p for p in providers]
    assert names[0] == "CUDAExecutionProvider"
    assert names[-1] == "CPUExecutionProvider"


def test_resolve_backend_name_auto_without_runtime():
    cfg = Config(models_dir="./models")
    opts = EnhanceOptions(backend="auto")
    name = resolve_backend_name(cfg, opts)
    assert name in ("onnx", "torch", "classical")


def test_resolve_backend_name_rejects_unknown():
    with pytest.raises(BackendUnavailable):
        resolve_backend_name(Config(), EnhanceOptions(backend="tpu"))


def test_enhance_options_roundtrip():
    opts = EnhanceOptions(scale=2.0, detail=0.5, tile=256)
    again = EnhanceOptions.from_dict(opts.to_dict())
    assert again.scale == 2.0
    assert again.detail == 0.5
    assert again.tile == 256


def test_enhance_options_unknown_keys_go_to_extra():
    opts = EnhanceOptions.from_dict({"scale": 2.0, "future_flag": True})
    assert opts.scale == 2.0
    assert opts.extra["future_flag"] is True


def test_enhance_options_from_none():
    assert EnhanceOptions.from_dict(None).scale == 4.0
