"""Backend package.

Imports are lazy so that ``import pixelboost`` never pulls in onnxruntime or
torch. Everything here resolves on first attribute access.
"""

from pixelboost.backends.base import Backend
from pixelboost.backends.classical import ClassicalBackend, NearestBackend

__all__ = [
    "Backend",
    "ClassicalBackend",
    "NearestBackend",
    "OnnxBackend",
    "TorchBackend",
    "backend_capabilities",
    "create_backend",
    "onnx_available",
    "resolve_backend_name",
]


def __getattr__(name: str):
    import importlib

    if name == "registry":
        return importlib.import_module("pixelboost.backends.registry")
    if name in (
        "create_backend",
        "onnx_available",
        "torch_available",
        "backend_capabilities",
        "resolve_backend_name",
        "describe_environment",
        "available_providers",
    ):
        return getattr(importlib.import_module("pixelboost.backends.registry"), name)
    if name == "OnnxBackend":
        return getattr(importlib.import_module("pixelboost.backends.onnx_backend"), "OnnxBackend")
    if name == "TorchBackend":
        return getattr(
            importlib.import_module("pixelboost.backends.torch_backend"), "TorchBackend"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list:
    return sorted(set(globals()) | set(__all__))
