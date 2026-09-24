"""Configuration loading and the built-in model registry.

Configuration is resolved in three layers, later wins:

1. dataclass defaults
2. a YAML / JSON / TOML file (``--config``, ``$PIXELBOOST_CONFIG``, or an
   auto-discovered ``pixelboost.yaml`` next to the working directory)
3. explicit command-line flags

The registry below is the single source of truth for model filenames, download
URLs and architecture parameters. Adding a model means adding one entry -- the
download script, the CLI, the HTTP capability endpoint and the backend factory
all read from here.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from pixelboost.errors import ConfigError
from pixelboost.types import EnhanceOptions

CONFIG_ENV = "PIXELBOOST_CONFIG"
HOME_ENV = "PIXELBOOST_HOME"

CONFIG_FILENAMES = (
    "pixelboost.yaml",
    "pixelboost.yml",
    "pixelboost.json",
    "pixelboost.toml",
)


@dataclass(frozen=True)
class ModelSpec:
    """One downloadable model."""

    name: str
    kind: str
    scale: int
    filename: str
    arch: str = "rrdb"
    url: Optional[str] = None
    sha256: Optional[str] = None
    num_block: int = 23
    num_feat: int = 64
    num_conv: int = 32
    license: str = "BSD-3-Clause"
    description: str = ""
    tags: tuple = ()


MODEL_REGISTRY: Dict[str, ModelSpec] = {
    "realesrgan-x4plus": ModelSpec(
        name="realesrgan-x4plus",
        kind="pth",
        scale=4,
        filename="RealESRGAN_x4plus.pth",
        arch="rrdb",
        num_block=23,
        num_feat=64,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        license="BSD-3-Clause",
        description="General photography. The best all-round starting point.",
        tags=("photo", "general", "default"),
    ),
    "realesrgan-x4plus-anime": ModelSpec(
        name="realesrgan-x4plus-anime",
        kind="pth",
        scale=4,
        filename="RealESRGAN_x4plus_anime_6B.pth",
        arch="rrdb",
        num_block=6,
        num_feat=64,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth",
        license="BSD-3-Clause",
        description="Anime, illustration and line art. One third the compute of x4plus.",
        tags=("anime", "illustration", "fast"),
    ),
    "realesr-general-x4v3": ModelSpec(
        name="realesr-general-x4v3",
        kind="pth",
        scale=4,
        filename="realesr-general-x4v3.pth",
        arch="srvgg",
        num_conv=32,
        num_feat=64,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        license="BSD-3-Clause",
        description="Compact 1.2M-param net. ~8x faster than x4plus, ideal for CPU tiers.",
        tags=("fast", "cpu", "general"),
    ),
    "realesr-general-wdn-x4v3": ModelSpec(
        name="realesr-general-wdn-x4v3",
        kind="pth",
        scale=4,
        filename="realesr-general-wdn-x4v3.pth",
        arch="srvgg",
        num_conv=32,
        num_feat=64,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-wdn-x4v3.pth",
        license="BSD-3-Clause",
        description="Denoise variant of x4v3. Use on JPEG-compressed or noisy sources.",
        tags=("fast", "denoise", "jpeg"),
    ),
    "realesr-animevideov3": ModelSpec(
        name="realesr-animevideov3",
        kind="pth",
        scale=4,
        filename="realesr-animevideov3.pth",
        arch="srvgg",
        num_conv=16,
        num_feat=64,
        url="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
        license="BSD-3-Clause",
        description="Tiny 2.4M anime model tuned for video frames; lowest latency.",
        tags=("anime", "video", "fastest"),
    ),
}

CLASSICAL_MODEL = "classical"


def default_models_dir() -> str:
    home = os.environ.get(HOME_ENV)
    if home:
        return os.path.join(home, "models")
    return os.path.join(os.path.expanduser("~"), ".cache", "pixelboost", "models")


@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 2
    queue_size: int = 64
    api_keys: List[str] = field(default_factory=list)
    cors_origins: List[str] = field(default_factory=lambda: ["*"])
    max_upload_mb: int = 32
    job_ttl_seconds: int = 3600
    keep_alive: int = 30
    timeout: int = 300
    result_format: str = "png"


@dataclass
class Config:
    """Fully resolved configuration."""

    models_dir: str = field(default_factory=default_models_dir)
    backend: str = "auto"
    model: str = "realesrgan-x4plus"
    provider: Optional[str] = None
    fallback_model: str = "realesr-general-x4v3"
    fp16: bool = False
    threads: int = 0
    tile: int = 512
    tile_overlap: int = 16
    tile_pad: int = 16
    align: int = 8
    max_pixels: int = 64_000_000
    warmup: bool = True
    defaults: EnhanceOptions = field(default_factory=EnhanceOptions)
    server: ServerConfig = field(default_factory=ServerConfig)

    def model_path(self, spec: ModelSpec) -> str:
        return os.path.join(self.models_dir, spec.filename)

    def resolve_model(self, name: Optional[str]) -> ModelSpec:
        key = (name or self.model or "realesrgan-x4plus").strip()
        if key == CLASSICAL_MODEL:
            return ModelSpec(
                name=CLASSICAL_MODEL,
                kind="none",
                scale=4,
                filename="",
                arch="classical",
                url=None,
                description="Model-free resampler. No download required.",
                tags=("fallback",),
            )
        spec = MODEL_REGISTRY.get(key)
        if spec is None:
            local = Path(key)
            if local.is_file():
                return ModelSpec(
                    name=local.stem,
                    kind=local.suffix.lstrip("."),
                    scale=4,
                    filename=local.name,
                    url=None,
                    description="User-supplied model file.",
                )
            raise ConfigError(
                f"unknown model {key!r}. Known: {', '.join(sorted(MODEL_REGISTRY))}, "
                f"or pass an existing .pth/.onnx path."
            )
        return spec

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Optional[Dict[str, Any]]) -> "Config":
        data = dict(data or {})
        defaults = data.pop("defaults", None) or data.pop("enhance", None)
        server = data.pop("server", None)
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        unknown = {k: v for k, v in data.items() if k not in known}
        if unknown:
            raise ConfigError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(**{k: v for k, v in data.items() if k in known and k != "defaults"})
        cfg.defaults = EnhanceOptions.from_dict(defaults)
        if server:
            cfg.server = ServerConfig(**server)
        return cfg


def _read_file(path: str) -> Dict[str, Any]:
    suffix = Path(path).suffix.lower()
    text = Path(path).read_text(encoding="utf-8")
    if suffix in (".yaml", ".yml"):
        try:
            import yaml  # noqa: WPS433
        except ImportError as exc:
            raise ConfigError(
                f"{path} is YAML but PyYAML is not installed; "
                f"`pip install pyyaml` or convert the file to JSON."
            ) from exc
        return yaml.safe_load(text) or {}
    if suffix == ".toml":
        import tomllib  # noqa: WPS433

        return tomllib.loads(text)
    return json.loads(text)


def discover_config(explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        if not os.path.isfile(explicit):
            raise ConfigError(f"config file not found: {explicit}")
        return explicit

    env = os.environ.get(CONFIG_ENV)
    if env:
        if not os.path.isfile(env):
            raise ConfigError(f"{CONFIG_ENV} points at a missing file: {env}")
        return env

    for name in CONFIG_FILENAMES:
        if os.path.isfile(name):
            return name

    for name in CONFIG_FILENAMES:
        candidate = os.path.join(os.path.expanduser("~"), ".config", "pixelboost", name)
        if os.path.isfile(candidate):
            return candidate
    return None


def load_config(path: Optional[str] = None) -> Config:
    resolved = discover_config(path)
    if not resolved:
        return Config()
    data = _read_file(resolved)
    if not isinstance(data, dict):
        raise ConfigError(f"{resolved} must contain a mapping at the top level")
    cfg = Config.from_dict(data)
    cfg.extra_source = resolved  # type: ignore[attr-defined]
    return cfg


def ensure_models_dir(cfg: Config) -> str:
    os.makedirs(cfg.models_dir, exist_ok=True)
    return cfg.models_dir
