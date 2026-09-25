"""ONNX Runtime backend -- the recommended way to run neural upscalers in production.

Why ONNX Runtime rather than raw PyTorch:

* **One artifact, two execution paths.** The same ``.onnx`` file runs on CPU
  (``CPUExecutionProvider``) and GPU (``CUDAExecutionProvider`` /
  ``TensorrtExecutionProvider`` / ``DmlExecutionProvider``). No code fork, no
  ``torch.cuda`` conditional anywhere.
* **Much smaller deployment.** ``onnxruntime`` CPU wheel is ~15 MB against
  PyTorch's ~800 MB, so a cheap VPS can host the CPU tier.
* **Real CPU parallelism.** ORT releases the GIL during ``run()``, so a thread
  pool gives genuine multi-core scaling rather than serialised numpy calls. This
  is what makes the sync HTTP endpoint viable without a process pool.

TensorRT additionally needs a timing cache to avoid 30-90 s of engine building
on every cold start; the cache directory is configured here for that reason.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from typing import Any

import numpy as np

from pixelboost.backends.base import Backend
from pixelboost.errors import BackendUnavailable, ModelNotFound, ProviderUnavailable

PROVIDER_ALIASES = {
    "tensorrt": "TensorrtExecutionProvider",
    "trt": "TensorrtExecutionProvider",
    "cuda": "CUDAExecutionProvider",
    "gpu": "CUDAExecutionProvider",
    "dml": "DmlExecutionProvider",
    "directml": "DmlExecutionProvider",
    "rocm": "ROCMExecutionProvider",
    "coreml": "CoreMLExecutionProvider",
    "openvino": "OpenVINOExecutionProvider",
    "openvino-cpu": "OpenVINOExecutionProvider",
    "cpu": "CPUExecutionProvider",
}

AUTO_PROVIDER_ORDER: Sequence[str] = (
    "TensorrtExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "ROCMExecutionProvider",
    "CoreMLExecutionProvider",
    "OpenVINOExecutionProvider",
    "CPUExecutionProvider",
)

_SHORT = {
    "TensorrtExecutionProvider": "tensorrt",
    "CUDAExecutionProvider": "cuda",
    "DmlExecutionProvider": "directml",
    "ROCMExecutionProvider": "rocm",
    "CoreMLExecutionProvider": "coreml",
    "OpenVINOExecutionProvider": "openvino",
    "CPUExecutionProvider": "cpu",
}


def _ort():
    try:
        import onnxruntime as ort  # noqa: WPS433
    except ImportError as exc:  # pragma: no cover - depends on install extra
        raise BackendUnavailable(
            "onnxruntime is not installed. Install the CPU runtime with "
            "`pip install onnxruntime`, or the GPU runtime with "
            "`pip install onnxruntime-gpu` (they are mutually exclusive)."
        ) from exc
    return ort


def available_providers() -> list[str]:
    try:
        ort = _ort()
    except BackendUnavailable:
        return []
    return list(ort.get_available_providers())


def has_gpu() -> bool:
    providers = available_providers()
    return any(p != "CPUExecutionProvider" for p in providers)


def normalize_provider(requested: str | None) -> str | None:
    if not requested or requested == "auto":
        return None
    key = requested.strip().lower()
    if key in PROVIDER_ALIASES:
        return PROVIDER_ALIASES[key]
    if requested in AUTO_PROVIDER_ORDER:
        return requested
    raise ProviderUnavailable(
        f"unknown provider {requested!r}; expected one of "
        f"{sorted(set(PROVIDER_ALIASES) | set(AUTO_PROVIDER_ORDER))}"
    )


def resolve_providers(requested: str | None, fp16: bool = False) -> list[Any]:
    """Build the ONNX Runtime provider list, with a CPU fallback always present.

    TensorRT is listed *before* CUDA on purpose: ORT tries providers in order and
    falls through when a node is unsupported, so the pair gives you TensorRT
    speed with CUDA as the safety net.
    """
    installed = available_providers()
    if not installed:
        raise BackendUnavailable(
            "onnxruntime is not installed; run `pip install onnxruntime` (CPU) "
            "or `pip install onnxruntime-gpu` (GPU)"
        )

    target = normalize_provider(requested)
    chosen: list[str]
    if target is None:
        chosen = [p for p in AUTO_PROVIDER_ORDER if p in installed]
        if not chosen:
            chosen = ["CPUExecutionProvider"]
    else:
        if target not in installed:
            raise ProviderUnavailable(
                f"{target} is not available in this onnxruntime build. "
                f"Installed providers: {installed}. The CPU wheel only ships "
                f"CPUExecutionProvider -- install onnxruntime-gpu and matching "
                f"CUDA/cuDNN libraries for GPU execution."
            )
        chosen = [target]
        if target == "TensorrtExecutionProvider" and "CUDAExecutionProvider" in installed:
            chosen.append("CUDAExecutionProvider")

    providers: list[Any] = []
    for name in chosen:
        if name == "TensorrtExecutionProvider":
            cache = os.environ.get(
                "PIXELBOOST_TRT_CACHE",
                os.path.join(os.path.expanduser("~"), ".cache", "pixelboost", "trt"),
            )
            os.makedirs(cache, exist_ok=True)
            providers.append(
                (
                    name,
                    {
                        "trt_fp16_enable": bool(fp16),
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": cache,
                        "trt_timing_cache_enable": True,
                        "trt_max_workspace_size": int(
                            os.environ.get("PIXELBOOST_TRT_WORKSPACE", 4 << 30)
                        ),
                    },
                )
            )
        elif name == "CUDAExecutionProvider":
            providers.append(
                (
                    name,
                    {
                        "device_id": int(os.environ.get("PIXELBOOST_CUDA_DEVICE", 0)),
                        "cudnn_conv_algo_search": os.environ.get(
                            "PIXELBOOST_CUDNN_SEARCH", "HEURISTIC"
                        ),
                        "arena_extend_strategy": "kSameAsRequested",
                        "do_copy_in_default_stream": True,
                    },
                )
            )
        else:
            providers.append(name)

    if not any(p == "CPUExecutionProvider" for p in providers):
        providers.append("CPUExecutionProvider")
    return providers


def provider_name(providers: Sequence[Any]) -> str:
    first = providers[0]
    raw = first[0] if isinstance(first, (tuple, list)) else first
    return _SHORT.get(raw, raw)


class OnnxBackend(Backend):
    name = "onnx"

    def __init__(
        self,
        model_path: str,
        provider: str | None = None,
        *,
        native_scale: int = 4,
        fp16: bool = False,
        threads: int = 0,
        model_name: str | None = None,
        normalize_output: str = "auto",
        device_id: int = 0,
    ) -> None:
        super().__init__(model=model_name or os.path.basename(model_path), provider=provider)
        if not os.path.isfile(model_path):
            raise ModelNotFound(
                f"model file not found: {model_path}. Run "
                f"`python scripts/download_models.py` to fetch the default set."
            )

        ort = _ort()
        self.model_path = model_path
        self.native_scale = int(native_scale)
        self.requires_tiling = True
        self.fp16 = bool(fp16)
        self.normalize_output = normalize_output
        self.device_id = int(device_id)

        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads and threads > 0:
            so.intra_op_num_threads = int(threads)
            so.inter_op_num_threads = 1

        providers = resolve_providers(provider, fp16)
        self.providers = providers
        self.device = provider_name(providers)

        if self.device == "cpu":
            so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        else:
            so.execution_mode = ort.ExecutionMode.ORT_PARALLEL
            if self.device == "directml":
                self.device = "directml"

        try:
            self.sess = ort.InferenceSession(model_path, so, providers=providers)
        except Exception as exc:
            raise BackendUnavailable(f"failed to load {model_path}: {exc}") from exc

        meta = self.sess.get_inputs()[0]
        self.input_name = meta.name
        self.input_type = meta.type
        self.input_dtype = np.float16 if "float16" in meta.type else np.float32
        self.output_names = [o.name for o in self.sess.get_outputs()]
        self.fixed_size = self._read_fixed_size(meta.shape)

    @staticmethod
    def _read_fixed_size(shape: Sequence[Any]) -> tuple[int, int] | None:
        if len(shape) != 4:
            return None
        h, w = shape[2], shape[3]
        if isinstance(h, int) and isinstance(w, int) and h > 0 and w > 0:
            return int(h), int(w)
        return None

    @property
    def provider(self) -> str:
        return self.device

    def _to_input(self, tile: np.ndarray) -> np.ndarray:
        x = tile
        if self.fixed_size:
            th, tw = self.fixed_size
            if tile.shape[0] > th or tile.shape[1] > tw:
                raise BackendUnavailable(
                    f"tile {tile.shape[:2]} exceeds the model's fixed input "
                    f"{self.fixed_size}; lower --tile"
                )
            canvas = np.zeros((th, tw, tile.shape[2]), np.float32)
            canvas[: tile.shape[0], : tile.shape[1]] = tile
            x = canvas
        arr = np.ascontiguousarray(np.transpose(x, (2, 0, 1))[None])
        if self.input_dtype == np.float16:
            arr = arr.astype(np.float16)
        return arr

    def _from_output(self, raw: np.ndarray, h: int, w: int) -> np.ndarray:
        out = np.asarray(raw)
        if out.ndim == 4:
            out = out[0]
        out = np.transpose(out, (1, 2, 0)).astype(np.float32)

        if self.normalize_output == "auto":
            peak = float(out.max()) if out.size else 0.0
            floor = float(out.min()) if out.size else 0.0
            if peak > 1.5:
                out = out / 255.0
            elif floor < -0.05:
                out = (out + 1.0) * 0.5
        elif self.normalize_output == "unit":
            pass
        elif self.normalize_output == "signed":
            out = (out + 1.0) * 0.5
        elif self.normalize_output == "byte":
            out = out / 255.0

        if self.fixed_size:
            out = out[: h * self.native_scale, : w * self.native_scale]
        return out

    def process(self, tile: np.ndarray, scale: float) -> np.ndarray:
        h, w = tile.shape[0], tile.shape[1]
        feeds = {self.input_name: self._to_input(tile)}
        raw = self.sess.run(self.output_names, feeds)[0]
        out = self._from_output(raw, h, w)
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    def warmup(self, tile: int = 64) -> None:
        side = tile
        if self.fixed_size:
            side = min(tile, self.fixed_size[0], self.fixed_size[1])
        side = max(8, int(side))
        self.process(np.zeros((side, side, 3), np.float32), float(self.native_scale))

    def info(self) -> dict[str, Any]:
        data = super().info()
        data.update(
            {
                "model_path": self.model_path,
                "input": {"name": self.input_name, "type": self.input_type},
                "fixed_input": list(self.fixed_size) if self.fixed_size else None,
                "outputs": self.output_names,
                "fp16": self.fp16,
                "available_providers": available_providers(),
                "active_providers": [
                    p[0] if isinstance(p, (tuple, list)) else p for p in self.providers
                ],
            }
        )
        return data

    def close(self) -> None:
        self.sess = None  # type: ignore[assignment]
