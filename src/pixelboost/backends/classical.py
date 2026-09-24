"""Classical (model-free) backend.

This is the safety net. It is always constructible, needs no download, no CUDA
and no 60 MB of weights, and on clean photographs it lands somewhere between
bicubic and a small neural net. On screenshots, line art and text it is often
*preferable* to a GAN model, because GANs invent texture that is not there and
turn 12pt glyphs into mush.

The pipeline is a compressed version of what production resamplers do:

1.  Multi-step Lanczos-3 (2x per step) -- the resampling backbone.
2.  Guided-filter detail reinjection -- recovers the micro-contrast that any
    low-pass step removes, without ringing.
3.  Optional soft-threshold unsharp -- final acutance pass, off by default.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from pixelboost import ops
from pixelboost.backends.base import Backend


class ClassicalBackend(Backend):
    name = "classical"
    requires_tiling = False
    device = "cpu"

    def __init__(
        self,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        *,
        max_step_ratio: float = 2.0,
        detail: float = 0.35,
        detail_radius: int = 4,
        detail_eps: float = 1e-3,
        sharpen: float = 0.0,
        sharpen_radius: float = 1.0,
        sharpen_threshold: float = 0.004,
        anti_alias_downscale: bool = True,
    ) -> None:
        super().__init__(model=None, provider="cpu")
        self.max_step_ratio = float(max_step_ratio)
        self.detail = float(detail)
        self.detail_radius = int(detail_radius)
        self.detail_eps = float(detail_eps)
        self.sharpen = float(sharpen)
        self.sharpen_radius = float(sharpen_radius)
        self.sharpen_threshold = float(sharpen_threshold)
        self.anti_alias_downscale = anti_alias_downscale

    def process(self, tile: np.ndarray, scale: float) -> np.ndarray:
        h, w = tile.shape[0], tile.shape[1]
        out_w = max(1, int(round(w * scale)))
        out_h = max(1, int(round(h * scale)))

        out = ops.resize_multi(tile, out_w, out_h, max_step_ratio=self.max_step_ratio)

        if self.detail > 0:
            radius = max(1, int(round(self.detail_radius * max(1.0, scale / 4.0))))
            out = ops.detail_enhance(out, radius, self.detail_eps, self.detail)

        if self.sharpen > 0:
            out = ops.unsharp(
                out,
                self.sharpen,
                max(0.5, self.sharpen_radius * max(1.0, scale / 4.0)),
                self.sharpen_threshold,
            )
        return out

    def info(self) -> Dict[str, Any]:
        data = super().info()
        data.update(
            {
                "max_step_ratio": self.max_step_ratio,
                "detail": self.detail,
                "sharpen": self.sharpen,
                "note": "model-free; albedo-safe on text and line art",
            }
        )
        return data


class NearestBackend(Backend):
    """Nearest-neighbour. Only useful for pixel-art, kept for completeness."""

    name = "nearest"
    requires_tiling = False
    device = "cpu"

    def process(self, tile: np.ndarray, scale: float) -> np.ndarray:
        h, w = tile.shape[0], tile.shape[1]
        out_w = max(1, int(round(w * scale)))
        out_h = max(1, int(round(h * scale)))
        yi = np.minimum((np.arange(out_h) * h // out_h), h - 1)
        xi = np.minimum((np.arange(out_w) * w // out_w), w - 1)
        return tile[yi][:, xi]

    def info(self) -> Dict[str, Any]:
        data = super().info()
        data["note"] = "pixel-art only; no interpolation"
        return data
