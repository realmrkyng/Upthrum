"""Backend selection.

``auto`` resolution is deliberately ordered by *robustness*, not raw speed:

1. **onnx** -- if ``onnxruntime`` is importable and an ``.onnx`` file exists.
   Smallest dependency, works CPU and GPU from one artifact, GIL-free.
2. **torch** -- if torch is importable and the ``.pth`` exists. Slower to ship,
   but it is what the published weights are.
3. **classical** -- always available, no download. Used as the silent fallback
   so that a broken install degrades in quality instead of returning a 500.

Overriding to a specific backend is respected literally and raises on failure;
that is what you want when you are deliberately benchmarking CUDA.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pixelboost.backends.base import Backend
from pixelboost.backends.classical import ClassicalBackend, NearestBackend
from pixelboost.backends.onnx_backend import (
    OnnxBackend,
    available_providers,
    has_gpu,
    normalize_provider,
    provider_name,
    resolve_providers,
)
from pixelboost.backends.torch_backend import TorchBackend
from pixelboost.config import CLASSICAL_MODEL, Config, ModelSpec
from pixelboost.errors import BackendUnavailable, ModelNotFound
from pixelboost.types import EnhanceOptions

BACKEND_NAMES = ("auto", "onnx", "torch", "classical", "nearest")

__all__ = [
    "BACKEND_NAMES",
    "Backend",
    "ClassicalBackend",
    "NearestBackend",
    "OnnxBackend",
    "TorchBackend",
    "available_providers",
    "backend_capabilities",
    "create_backend",
    "describe_environment",
    "has_gpu",
    "normalize_provider",
    "onnx_available",
    "onnx_path_for",
    "resolve_backend_name",
    "resolve_providers",
    "torch_available",
]


def _has_module(name: str) -> bool:
    import importlib.util

    try:
        return importlib.util.find_spec(name) is not None
    except (ImportError, ValueError):
        return False


def onnx_available() -> bool:
    return _has_module("onnxruntime")


def torch_available() -> bool:
    return _has_module("torch")


def onnx_path_for(cfg: Config, spec: ModelSpec) -> Optional[str]:
    """Locate an ``.onnx`` sibling for a registry entry."""
    if spec.kind == "onnx":
        candidate = os.path.join(cfg.models_dir, spec.filename)
        return candidate if os.path.isfile(candidate) else None
    if not spec.filename:
        return None
    stem = Path(spec.filename).stem
    for candidate in (
        os.path.join(cfg.models_dir, stem + ".onnx"),
        os.path.join(cfg.models_dir, stem + "_fp16.onnx"),
    ):
        if os.path.isfile(candidate):
            return candidate
    return None


def torch_path_for(cfg: Config, spec: ModelSpec) -> Optional[str]:
    if spec.kind not in ("pth", "pt", "ckpt"):
        return None
    candidate = os.path.join(cfg.models_dir, spec.filename)
    return candidate if os.path.isfile(candidate) else None


def resolve_backend_name(cfg: Config, opts: EnhanceOptions) -> str:
    requested = (opts.backend or cfg.backend or "auto").strip().lower()
    if requested != "auto":
        if requested not in BACKEND_NAMES:
            raise BackendUnavailable(
                f"unknown backend {requested!r}; expected one of {BACKEND_NAMES}"
            )
        return requested

    spec = cfg.resolve_model(opts.model or cfg.model)
    if opts.model_path:
        suffix = Path(opts.model_path).suffix.lower()
        return "onnx" if suffix == ".onnx" else "torch"
    if spec.name == CLASSICAL_MODEL:
        return "classical"
    if onnx_available() and onnx_path_for(cfg, spec):
        return "onnx"
    if torch_available() and torch_path_for(cfg, spec):
        return "torch"
    if onnx_available() or torch_available():
        return "classical"
    return "classical"


def create_backend(cfg: Config, opts: EnhanceOptions) -> Backend:
    """Build the backend implied by ``cfg`` + ``opts``."""
    name = resolve_backend_name(cfg, opts)
    spec = cfg.resolve_model(opts.model or cfg.model)

    if name == "classical":
        return ClassicalBackend(
            detail=opts.detail if opts.detail else 0.35,
            detail_radius=opts.detail_radius,
            detail_eps=opts.detail_eps,
            sharpen=opts.sharpen,
            sharpen_radius=opts.sharpen_radius,
            sharpen_threshold=opts.sharpen_threshold,
        )

    if name == "nearest":
        return NearestBackend()

    if name == "onnx":
        path = opts.model_path or onnx_path_for(cfg, spec)
        if not path or not os.path.isfile(path):
            raise ModelNotFound(
                f"no ONNX model found for {spec.name!r}. Either run "
                f"`python scripts/download_models.py --export-onnx --model {spec.name}` "
                f"or point --model-path at an existing .onnx file. Searched: "
                f"{cfg.models_dir}/*.onnx"
            )
        return OnnxBackend(
            path,
            provider=opts.provider or cfg.provider,
            native_scale=spec.scale,
            fp16=opts.fp16 or cfg.fp16,
            threads=opts.threads or cfg.threads,
            model_name=spec.name,
        )

    if name == "torch":
        path = opts.model_path or torch_path_for(cfg, spec)
        if not path or not os.path.isfile(path):
            raise ModelNotFound(
                f"no PyTorch checkpoint found for {spec.name!r}. Run "
                f"`python scripts/download_models.py --model {spec.name}` first."
            )
        return TorchBackend(
            path,
            provider=opts.provider or cfg.provider,
            native_scale=spec.scale,
            num_block=spec.num_block,
            num_feat=spec.num_feat,
            num_conv=spec.num_conv,
            arch=spec.arch,
            fp16=opts.fp16 or cfg.fp16,
            model_name=spec.name,
        )

    raise BackendUnavailable(f"unhandled backend {name!r}")


def backend_capabilities(cfg: Optional[Config] = None) -> Dict[str, Any]:
    """Everything the ``/v1/capabilities`` endpoint reports."""
    providers = available_providers()
    from pixelboost.config import MODEL_REGISTRY

    models: List[Dict[str, Any]] = []
    for spec in MODEL_REGISTRY.values():
        entry = {
            "name": spec.name,
            "scale": spec.scale,
            "arch": spec.arch,
            "license": spec.license,
            "description": spec.description,
            "tags": list(spec.tags),
            "url": spec.url,
        }
        if cfg is not None:
            entry["pth_present"] = bool(torch_path_for(cfg, spec))
            entry["onnx_present"] = bool(onnx_path_for(cfg, spec))
        models.append(entry)

    backends = ["classical", "nearest"]
    if onnx_available():
        backends.append("onnx")
    if torch_available():
        backends.append("torch")

    return {
        "backends": backends,
        "onnxruntime": onnx_available(),
        "torch": torch_available(),
        "providers": providers,
        "gpu": has_gpu(),
        "models": models,
    }


def describe_environment() -> str:
    lines = [
        f"onnxruntime : {'yes' if onnx_available() else 'no'}",
        f"torch       : {'yes' if torch_available() else 'no'}",
        f"providers   : {', '.join(available_providers()) or '(none)'}",
        f"gpu         : {'yes' if has_gpu() else 'no'}",
    ]
    for name in ("onnxruntime", "torch"):
        if _has_module(name):
            mod = __import__(name)
            lines.append(f"{name:<12}: {getattr(mod, '__version__', '?')}")
    return "\n".join(lines)
