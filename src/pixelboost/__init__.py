"""PixelBoost -- CPU/GPU accelerated image quality enhancement.

    from pixelboost import enhance
    result, path = enhance("photo.jpg", "photo_4x.png", scale=4, backend="onnx")
"""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"
__author__ = "PixelBoost contributors"
__license__ = "MIT"

_LAZY = {
    "Engine": ("pixelboost.engine", "Engine"),
    "enhance": ("pixelboost.engine", "enhance"),
    "open_image": ("pixelboost.engine", "open_image"),
    "ops": ("pixelboost.ops", None),
    "imageio": ("pixelboost.imageio", None),
    "Pipeline": ("pixelboost.pipeline", "Pipeline"),
    "Tiler": ("pixelboost.tiling", "Tiler"),
    "EnhanceOptions": ("pixelboost.types", "EnhanceOptions"),
    "EnhanceResult": ("pixelboost.types", "EnhanceResult"),
    "ImageMeta": ("pixelboost.types", "ImageMeta"),
    "resolve_target": ("pixelboost.types", "resolve_target"),
    "Config": ("pixelboost.config", "Config"),
    "load_config": ("pixelboost.config", "load_config"),
    "MODEL_REGISTRY": ("pixelboost.config", "MODEL_REGISTRY"),
    "create_backend": ("pixelboost.backends.registry", "create_backend"),
    "backend_capabilities": ("pixelboost.backends.registry", "backend_capabilities"),
    "PixelBoostError": ("pixelboost.errors", "PixelBoostError"),
    "BackendUnavailable": ("pixelboost.errors", "BackendUnavailable"),
    "ModelNotFound": ("pixelboost.errors", "ModelNotFound"),
}

__all__ = sorted(_LAZY)


def __getattr__(name: str) -> Any:
    """Lazy attribute access.

    ``import pixelboost`` must stay cheap: the CLI prints ``--help`` and the
    server reports capabilities before any backend is needed, and pulling in
    numpy plus a 60 MB graph at import time makes both feel broken.

    Note the ``None`` target for submodules -- resolving those with ``getattr``
    on the package itself would re-enter this function and recurse forever.
    """
    target = _LAZY.get(name)
    if target is None:
        raise AttributeError(f"module 'pixelboost' has no attribute {name!r}")

    import importlib

    module = importlib.import_module(target[0])
    value = module if target[1] is None else getattr(module, target[1])
    globals()[name] = value
    return value


def __dir__() -> list:
    return sorted(set(globals()) | set(_LAZY))
