"""The enhancement pipeline.

Stage order is not arbitrary and is worth stating once, because getting it wrong
is the difference between a good result and a mushy one:

``decode -> [denoise] -> [auto-levels] -> [pre-downscale] -> upscale -> exact-fit -> [chroma denoise] -> [detail] -> [unsharp] -> [colour] -> encode``

* Denoise goes **before** upscaling. A neural net asked to upscale noise will
  faithfully upscale the noise, and it is far cheaper to remove it at 1x.
* Auto-levels goes before upscaling too: a flat, low-contrast input gives the
  network little gradient to work with, and stretching first measurably helps on
  re-compressed sources.
* Chroma denoise goes **after** upscaling because that is where the speckle
  appears; doing it before would leave the net free to re-invent it.
* Detail and unsharp go last, on the final-resolution buffer, so their radii are
  expressed in output pixels and mean the same thing at every scale factor.
  A backend that already runs the detail pass internally sets
  ``handles_detail``, which suppresses this one -- otherwise the reinforcement
  is applied twice.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from pixelboost import ops
from pixelboost.backends.base import Backend
from pixelboost.errors import OutOfMemory
from pixelboost.tiling import Tiler, plan_tiles
from pixelboost.types import EnhanceOptions, resolve_target

log = logging.getLogger("pixelboost.pipeline")


class Pipeline:
    """Stateless per-image workhorse. One instance handles one image."""

    def __init__(
        self,
        backend: Backend,
        opts: EnhanceOptions,
        src_w: int,
        src_h: int,
    ) -> None:
        self.backend = backend
        self.opts = opts
        self.src_w = int(src_w)
        self.src_h = int(src_h)
        self.notes: list[str] = []

        self.target_w, self.target_h = resolve_target(
            self.src_w,
            self.src_h,
            scale=opts.scale,
            width=opts.width,
            height=opts.height,
            longest_side=opts.longest_side,
            keep_aspect=opts.keep_aspect,
        )
        if opts.max_pixels > 0 and self.target_w * self.target_h > opts.max_pixels:
            clamped_w, clamped_h = ops.fit_within(self.target_w, self.target_h, opts.max_pixels)
            self.notes.append(
                f"target {self.target_w}x{self.target_h} exceeds max_pixels="
                f"{opts.max_pixels}; clamped to {clamped_w}x{clamped_h}"
            )
            self.target_w, self.target_h = clamped_w, clamped_h

        self.tiler: Tiler | None = None
        self.tile_count = 0

    @property
    def requested_scale(self) -> float:
        return max(self.target_w / self.src_w, self.target_h / self.src_h)

    def _uniform_scale(self, w: int, h: int) -> float | None:
        sx = self.target_w / w
        sy = self.target_h / h
        if abs(sx - sy) <= 0.01 * max(sx, sy):
            return (sx + sy) / 2.0
        return None

    def _fit(self, img: np.ndarray, tw: int, th: int) -> np.ndarray:
        if img.shape[1] == tw and img.shape[0] == th:
            return img
        return ops.resize_multi(img, tw, th)

    def _upscale_tiled(
        self,
        work: np.ndarray,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        s = int(self.backend.native_scale)
        opts = self.opts
        self.tiler = Tiler(
            scale=s,
            tile=opts.tile,
            overlap=opts.tile_overlap,
            pad=opts.tile_pad,
            align=opts.align,
            max_output_pixels=opts.max_pixels if opts.max_pixels > 0 else 0,
        )

        def factory(tile_size: int) -> Callable[[np.ndarray], np.ndarray]:
            return lambda t: self.backend.process(t, float(s))

        if self.tiler.should_tile(work.shape[0], work.shape[1]):
            planned = len(plan_tiles(work.shape[0], work.shape[1], opts.tile, opts.tile_overlap))
            log.debug(
                "tiled inference: %d tiles of %dpx (scale %dx)",
                planned,
                opts.tile,
                s,
            )
            out = self.tiler.run_resilient(work, factory, on_progress)
            if self.tiler.stats.retries:
                self.notes.append(
                    f"tile auto-shrunk {self.tiler.stats.retries}x to "
                    f"{self.tiler.tile}px after an allocation failure"
                )
        else:
            out = self.backend.process(work, float(s))
            if on_progress:
                on_progress(1, 1)

        self.tile_count = self.tiler.stats.count
        return out

    def run(
        self,
        rgb: np.ndarray,
        alpha: np.ndarray | None = None,
        on_progress: Callable[[int, int], None] | None = None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        opts = self.opts
        work = rgb.astype(np.float32, copy=False)
        src_w, src_h = work.shape[1], work.shape[0]

        if opts.denoise > 0:
            work = ops.denoise(work, opts.denoise_radius, opts.denoise)
        if opts.auto_levels > 0:
            work = ops.percentile_stretch(work, strength=opts.auto_levels)

        native = int(getattr(self.backend, "native_scale", 1)) or 1
        tiled = bool(getattr(self.backend, "requires_tiling", False))

        if tiled:
            need = self.requested_scale
            pre = opts.pre_downscale
            if pre is None:
                pre = need < native / 2.0 and src_w * src_h > 4_000_000
            if pre and need < native:
                factor = need / native
                pw = max(1, int(round(work.shape[1] * factor)))
                ph = max(1, int(round(work.shape[0] * factor)))
                log.debug("pre-downscale %dx%d -> %dx%d before %dx inference", src_w, src_h, pw, ph, native)
                work = ops.resize_multi(work, pw, ph)
                self.notes.append(f"pre-downscaled to {pw}x{ph} before {native}x inference")
            out = self._upscale_tiled(work, on_progress)
            out = self._fit(out, self.target_w, self.target_h)
        else:
            uniform = self._uniform_scale(work.shape[1], work.shape[0])
            if uniform is not None:
                out = self.backend.process(work, uniform)
                out = self._fit(out, self.target_w, self.target_h)
            else:
                area = (self.target_w / work.shape[1]) * (self.target_h / work.shape[0])
                out = self.backend.process(work, float(np.sqrt(area)))
                out = self._fit(out, self.target_w, self.target_h)
            self.tile_count = 1

        out = out.astype(np.float32, copy=False)
        gain = max(1.0, out.shape[1] / max(1, src_w))

        if opts.chroma_denoise > 0:
            out = ops.chroma_denoise(out, opts.chroma_denoise * gain)
        if opts.detail > 0 and not getattr(self.backend, "handles_detail", False):
            out = ops.detail_enhance(
                out,
                max(1, int(round(opts.detail_radius * gain / 4.0))) or 1,
                opts.detail_eps,
                opts.detail,
            )
        if opts.sharpen > 0:
            out = ops.unsharp(
                out,
                opts.sharpen,
                max(0.5, opts.sharpen_radius * gain / 4.0),
                opts.sharpen_threshold,
            )
        if opts.gamma != 1.0 or opts.contrast or opts.saturation != 1.0:
            out = ops.adjust_color(out, opts.contrast, opts.saturation, opts.gamma)
        out = np.clip(out, 0.0, 1.0)

        out_alpha = None
        if alpha is not None:
            out_alpha = self._upscale_alpha(alpha, src_w, src_h)
        return out, out_alpha

    def _upscale_alpha(self, alpha: np.ndarray, src_w: int, src_h: int) -> np.ndarray:
        mode = (self.opts.alpha_mode or "lanczos").lower()
        plane = alpha.astype(np.float32)
        if plane.ndim == 2:
            plane = plane[..., None]
        if mode == "nearest":
            yi = np.minimum((np.arange(self.target_h) * src_h // self.target_h), src_h - 1)
            xi = np.minimum((np.arange(self.target_w) * src_w // self.target_w), src_w - 1)
            scaled = plane[yi][:, xi]
        else:
            scaled = ops.resize_multi(plane, self.target_w, self.target_h)
        return np.clip(scaled[..., 0], 0.0, 1.0)

    def guard_memory(self, limit_bytes: int) -> None:
        """Refuse a job whose final float buffer would not fit."""
        need = self.target_w * self.target_h * 3 * 4
        if limit_bytes > 0 and need > limit_bytes:
            raise OutOfMemory(
                f"final buffer needs {need / 1e6:.0f} MB but the budget is "
                f"{limit_bytes / 1e6:.0f} MB; reduce --scale/--longest-side"
            )
