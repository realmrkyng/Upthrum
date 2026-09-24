"""Backend contract.

A backend owns exactly one job: turn an ``(h, w, 3)`` float32 tile into an
``(h * scale, w * scale, 3)`` float32 tile. It knows nothing about tiling,
colour management or file formats -- those belong to :mod:`pixelboost.pipeline`.

Keeping that boundary tight is what lets the same pipeline drive a numpy Lanczos
resizer, an ONNX Runtime graph and a PyTorch module without branching.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


class Backend:
    """Abstract accelerator.

    Subclasses set :attr:`name`, :attr:`native_scale` and implement
    :meth:`process`. They advertise their own state through :meth:`info`, which
    the HTTP layer exposes at ``/v1/capabilities``.
    """

    name: str = "base"
    native_scale: int = 1
    requires_tiling: bool = False
    device: str = "cpu"

    def __init__(self, model: Optional[str] = None, provider: Optional[str] = None) -> None:
        self.model = model
        self.provider_requested = provider

    @property
    def provider(self) -> str:
        return self.device

    def process(self, tile: np.ndarray, scale: float) -> np.ndarray:
        raise NotImplementedError

    def warmup(self, tile: int = 64) -> None:
        """Run one throwaway inference.

        First-call latency for CUDA and TensorRT includes context creation and
        kernel autotuning -- 3-10 seconds is normal. Doing it at start-up keeps
        the first real user request fast.
        """

    def close(self) -> None:
        """Release device memory and file handles."""

    def info(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "model": self.model,
            "provider": self.provider,
            "device": self.device,
            "native_scale": self.native_scale,
            "requires_tiling": self.requires_tiling,
        }

    def __enter__(self) -> "Backend":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name} provider={self.provider}>"
