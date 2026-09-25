"""Engine -- the public entry point.

Holds warm backends so that a server does not pay model-loading cost per
request. Loading RealESRGAN_x4plus is ~1.5 s and building a TensorRT engine can
be 30-90 s; doing that inside a request handler is the difference between a
service and a demo.

Thread safety: backends are cached behind a lock and shared. ONNX Runtime's
``InferenceSession.run`` is documented as thread-safe, and PyTorch modules used
under ``inference_mode`` with no parameter mutation are safe to share as well.
The engine therefore supports a thread-pool web server without process
forking -- which matters because forking after CUDA initialisation is invalid.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from collections.abc import Iterable
from typing import Any, Callable, Union

import numpy as np

from pixelboost import imageio, ops
from pixelboost.backends.base import Backend
from pixelboost.backends.registry import backend_capabilities, create_backend, resolve_backend_name
from pixelboost.config import Config, ModelSpec, load_config
from pixelboost.errors import PixelBoostError
from pixelboost.pipeline import Pipeline
from pixelboost.types import EnhanceOptions, EnhanceResult, ImageMeta

log = logging.getLogger("pixelboost.engine")

ImageInput = Union[str, bytes, "os.PathLike", Any, np.ndarray]


class Engine:
    """Loads configuration, caches backends, runs pipelines."""

    def __init__(self, config: Config | None = None, warmup: bool | None = None) -> None:
        self.config = config or load_config()
        self._lock = threading.RLock()
        self._backends: dict[tuple[str, str, str, bool, str], Backend] = {}
        self._warmup_done: set = set()
        self.warmup_enabled = self.config.warmup if warmup is None else bool(warmup)

    @property
    def models_dir(self) -> str:
        return self.config.models_dir

    def _cache_key(
        self, name: str, spec: ModelSpec, opts: EnhanceOptions
    ) -> tuple[str, str, str, bool, str]:
        return (
            name,
            opts.model_path or spec.name,
            str(opts.provider or self.config.provider or "auto"),
            bool(opts.fp16 or self.config.fp16),
            json.dumps(opts.extra or {}, sort_keys=True, default=str),
        )

    def backend_for(self, opts: EnhanceOptions, force_new: bool = False) -> Backend:
        """Return a shared backend instance for these options."""
        name = resolve_backend_name(self.config, opts)
        spec = self.config.resolve_model(opts.model or self.config.model)
        key = self._cache_key(name, spec, opts)

        with self._lock:
            if not force_new and key in self._backends:
                return self._backends[key]

            log.info("loading backend=%s model=%s provider=%s", name, spec.name, opts.provider or "auto")
            backend = create_backend(self.config, opts)
            self._backends[key] = backend

        if self.warmup_enabled and key not in self._warmup_done:
            started = time.perf_counter()
            try:
                backend.warmup()
                self._warmup_done.add(key)
                log.info("warmup %s in %.0f ms", key[0], (time.perf_counter() - started) * 1000)
            except Exception as exc:  # pragma: no cover - warmup must never be fatal
                log.warning("warmup failed for %s: %s", key[0], exc)
        return backend

    def close(self) -> None:
        with self._lock:
            for backend in self._backends.values():
                with contextlib.suppress(Exception):
                    backend.close()
            self._backends.clear()
            self._warmup_done.clear()

    def __enter__(self) -> Engine:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def enhance(
        self,
        src: ImageInput,
        opts: EnhanceOptions | None = None,
        on_progress: Callable[[int, int], None] | None = None,
        backend: Backend | None = None,
    ) -> EnhanceResult:
        """Enhance one image. ``src`` may be a path, bytes, PIL image or RGB array."""
        options = opts or self.config.defaults
        started = time.perf_counter()

        if isinstance(src, np.ndarray):
            if src.ndim == 2:
                # Greyscale input is promoted to a 3-channel luma stack; the
                # grey-ness is not tracked further, so output is RGB.
                rgb = np.repeat(src[..., None].astype(np.float32), 3, axis=2)
                alpha = None
            else:
                rgb = src.astype(np.float32)
                if rgb.dtype == np.uint8:
                    rgb = rgb / 255.0
                alpha = None
            meta = ImageMeta(width=rgb.shape[1], height=rgb.shape[0], mode="RGB")
        else:
            loaded = imageio.load(src)
            rgb, alpha, meta, _gray = loaded.rgb, loaded.alpha, loaded.meta, loaded.gray

        be = backend or self.backend_for(options)

        pipeline = Pipeline(be, options, rgb.shape[1], rgb.shape[0])
        out_rgb, out_alpha = pipeline.run(rgb, alpha, on_progress)

        elapsed = (time.perf_counter() - started) * 1000.0
        return EnhanceResult(
            image=out_rgb,
            alpha=out_alpha,
            meta=meta,
            backend=be.name,
            model=be.model,
            provider=be.provider,
            src_size=(rgb.shape[1], rgb.shape[0]),
            dst_size=(out_rgb.shape[1], out_rgb.shape[0]),
            tiles=pipeline.tile_count,
            elapsed_ms=elapsed,
            notes=list(pipeline.notes),
        )

    def enhance_file(
        self,
        src: ImageInput,
        dst: str | None = None,
        opts: EnhanceOptions | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> tuple[EnhanceResult, str | None]:
        """Enhance and write to ``dst``. Returns ``(result, written_path)``."""
        options = opts or self.config.defaults
        result = self.enhance(src, options, on_progress)
        if dst is None:
            return result, None

        fmt = options.output_format
        if not fmt:
            src_fmt = result.meta.format if result.meta else "PNG"
            base, ext = os.path.splitext(dst)
            fmt = ext.lstrip(".") if ext else src_fmt
        gray = bool(result.meta and result.meta.mode in imageio.GRAY_MODES)
        written = imageio.save(
            dst,
            result.image,
            result.alpha,
            fmt=fmt,
            quality=options.quality,
            meta=result.meta,
            gray=gray,
            preserve_metadata=options.preserve_metadata,
        )
        return result, written

    def enhance_bytes(self, data: bytes, opts: EnhanceOptions | None = None) -> tuple[bytes, EnhanceResult]:
        options = opts or self.config.defaults
        result = self.enhance(data, options)
        fmt = options.output_format or (result.meta.format if result.meta else "PNG")
        gray = bool(result.meta and result.meta.mode in imageio.GRAY_MODES)
        blob = imageio.encode(
            result.image,
            result.alpha,
            fmt=fmt,
            quality=options.quality,
            meta=result.meta,
            gray=gray,
            preserve_metadata=options.preserve_metadata,
        )
        return blob, result

    def batch(
        self,
        paths: Iterable[str],
        out_dir: str,
        opts: EnhanceOptions | None = None,
        suffix: str = "_upscaled",
        skip_existing: bool = True,
    ) -> list[dict[str, Any]]:
        options = opts or self.config.defaults
        os.makedirs(out_dir, exist_ok=True)
        # Load and cache the backend up front so the first file does not pay
        # for model loading; enhance() would resolve the same instance anyway.
        self.backend_for(options)
        reports: list[dict[str, Any]] = []

        for path in paths:
            stem, ext = os.path.splitext(os.path.basename(path))
            target = os.path.join(out_dir, f"{stem}{suffix}{ext}")
            entry: dict[str, Any] = {"input": path, "output": target}
            if skip_existing and os.path.exists(target):
                entry["status"] = "skipped"
                reports.append(entry)
                continue
            try:
                result, written = self.enhance_file(path, target, options)
                entry.update({"status": "ok", "output": written, **result.summary()})
            except PixelBoostError as exc:
                entry.update({"status": "error", "error": str(exc)})
            except Exception as exc:  # pragma: no cover - defensive
                entry.update({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
            reports.append(entry)
            log.info(
                "%s -> %s [%s]",
                os.path.basename(path),
                entry.get("status"),
                entry.get("error", ""),
            )
        return reports

    def capabilities(self) -> dict[str, Any]:
        data = backend_capabilities(self.config)
        data["config"] = {
            "models_dir": self.config.models_dir,
            "loaded_backends": [
                {"key": list(k), "info": v.info()} for k, v in self._backends.items()
            ],
        }
        return data


def open_image(path: str) -> dict[str, Any]:
    return imageio.probe(path)


def enhance(
    src: ImageInput,
    dst: str | None = None,
    config: Config | None = None,
    **option_overrides: Any,
) -> tuple[EnhanceResult, str | None]:
    """One-shot functional API::

        from pixelboost import enhance
        result, path = enhance("in.jpg", "out.png", scale=4, backend="onnx")
    """
    opts = EnhanceOptions(**option_overrides) if option_overrides else None
    with Engine(config) as engine:
        return engine.enhance_file(src, dst, opts)


__all__ = ["Engine", "enhance", "open_image", "ops"]
