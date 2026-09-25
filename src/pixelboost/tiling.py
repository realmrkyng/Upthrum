"""Tiled inference with feather blending.

A 4x neural upscaler on a 6000x4000 photo needs a 96000x64000x3 float32 output
buffer -- about 73 GB. Nobody has that. The standard remedy is to cut the image
into overlapping tiles, upscale each independently, and stitch.

Stitching is where naive implementations fail: hard seams are visible even when
every tile is perfect, because convolution padding differs per tile. This module
fixes that with a cosine feather over the overlap, and it streams the result so
peak memory stays proportional to *one tile* rather than the whole output.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from pixelboost.errors import OutOfMemory

TileRect = tuple[int, int, int, int]


class _ReflectPad(np.ndarray):
    """Marker subclass so callers can tell a padded array from a real one."""


@dataclass
class TileStats:
    count: int = 0
    max_tile_px: int = 0
    retries: int = 0


def axis_ranges(total: int, tile: int, overlap: int) -> list[tuple[int, int]]:
    """Split one axis into overlapping windows covering ``[0, total)``.

    The last window is clamped to the far edge, so the stride is uneven by at
    most ``overlap``. Every pixel is covered at least once.
    """
    if tile <= 0 or tile >= total:
        return [(0, total)]

    overlap = max(0, min(overlap, tile // 2))
    stride = max(1, tile - 2 * overlap)

    ranges: list[tuple[int, int]] = []
    pos = 0
    while True:
        end = min(pos + tile, total)
        start = max(0, end - tile)
        if ranges and ranges[-1][0] == start:
            break
        ranges.append((start, end))
        if end >= total:
            break
        pos += stride
    return ranges


def plan_tiles(h: int, w: int, tile: int, overlap: int) -> list[TileRect]:
    """Row-major list of ``(y0, y1, x0, x1)`` source rectangles."""
    return [
        (y0, y1, x0, x1)
        for y0, y1 in axis_ranges(h, tile, overlap)
        for x0, x1 in axis_ranges(w, tile, overlap)
    ]


def _ramp(length: int) -> np.ndarray:
    """Half-cosine rise from ~0 to ~1, sampled at pixel centres.

    Sampling at centres instead of endpoints avoids a zero weight on the very
    first pixel, which would otherwise leave a one-pixel hole when a tile's
    neighbour turns out not to cover it.
    """
    if length <= 1:
        return np.ones(max(1, length), np.float32)
    t = (np.arange(length, dtype=np.float32) + 0.5) / length
    return (0.5 - 0.5 * np.cos(np.pi * t)).astype(np.float32)


def _axis_weights(length: int, ramp_px: int, rises: bool, falls: bool) -> np.ndarray:
    w = np.ones(length, np.float32)
    r = int(min(ramp_px, max(0, length // 2)))
    if r > 0:
        rise = _ramp(r)
        if rises:
            w[:r] = rise
        if falls:
            w[length - r :] = rise[::-1]
    return w


class Tiler:
    """Streaming tiled upscaler.

    Usage::

        tiler = Tiler(tile=512, overlap=16, scale=4, pad=16, align=8)
        image = tiler.run(reference_image, backend.process)

    ``backend.process`` receives an ``(h, w, 3)`` float32 tile and must return
    ``(h * scale, w * scale, 3)``.
    """

    def __init__(
        self,
        scale: int,
        tile: int = 512,
        overlap: int = 16,
        pad: int = 16,
        align: int = 8,
        max_output_pixels: int = 0,
    ) -> None:
        self.scale = max(1, int(scale))
        self.tile = max(0, int(tile))
        self.overlap = max(0, int(overlap))
        self.pad = max(0, int(pad))
        self.align = max(1, int(align))
        self.max_output_pixels = int(max_output_pixels)
        self.stats = TileStats()

    def should_tile(self, h: int, w: int) -> bool:
        return 0 < self.tile < max(h, w)

    def _run_tile(self, proc: Callable[[np.ndarray], np.ndarray], tile: np.ndarray) -> np.ndarray:
        h, w = tile.shape[0], tile.shape[1]
        aligned = tile
        pad_h = (-h) % self.align
        pad_w = (-w) % self.align
        if pad_h or pad_w:
            aligned = np.pad(tile, ((0, pad_h), (0, pad_w), (0, 0)), mode="reflect")

        out = proc(aligned)
        out = np.asarray(out, dtype=np.float32)
        if out.ndim == 2:
            out = out[..., None]

        if pad_h or pad_w:
            out = out[: h * self.scale, : w * self.scale]

        want = (h * self.scale, w * self.scale)
        if out.shape[:2] != want:
            raise OutOfMemory(
                f"backend returned {out.shape[:2]} for a {want} tile; "
                f"native scale {self.scale} does not match the model"
            )
        self.stats.count += 1
        self.stats.max_tile_px = max(self.stats.max_tile_px, h * w)
        return out

    def run(
        self,
        img: np.ndarray,
        proc: Callable[[np.ndarray], np.ndarray],
        on_progress: Callable[[int, int], None] | None = None,
    ) -> np.ndarray:
        h, w = img.shape[0], img.shape[1]
        s = self.scale
        out_h, out_w = h * s, w * s

        if self.max_output_pixels and out_h * out_w > self.max_output_pixels:
            raise OutOfMemory(
                f"output would be {out_w}x{out_h} = {out_h * out_w / 1e6:.1f} MP, "
                f"over the {self.max_output_pixels / 1e6:.1f} MP budget"
            )

        src = img
        if self.pad:
            p = min(self.pad, max(1, min(h, w) - 1))
            src = np.pad(img, ((p, p), (p, p), (0, 0)), mode="reflect")
        else:
            p = 0

        sh, sw = src.shape[0], src.shape[1]
        ys = axis_ranges(sh, self.tile, self.overlap)
        xs = axis_ranges(sw, self.tile, self.overlap)
        total = len(ys) * len(xs)

        out = np.empty((sh * s, sw * s, img.shape[2]), np.float32)
        ramp_px = self.overlap * s
        overlaps_y = len(ys) > 1
        overlaps_x = len(xs) > 1

        band_h = max(y1 - y0 for y0, y1 in ys) * s
        acc = np.zeros((band_h, sw * s, img.shape[2]), np.float32)
        wsum = np.zeros((band_h, sw * s, 1), np.float32)
        carry_acc: np.ndarray | None = None
        carry_wsum: np.ndarray | None = None

        band_base = 0
        done = 0
        for bi, (y0, y1) in enumerate(ys):
            acc[:] = 0.0
            wsum[:] = 0.0
            if carry_acc is not None:
                rows = carry_acc.shape[0]
                acc[:rows] = carry_acc
                wsum[:rows] = carry_wsum

            wy = _axis_weights(
                (y1 - y0) * s,
                ramp_px,
                rises=overlaps_y and y0 > 0,
                falls=overlaps_y and y1 < sh,
            )

            for x0, x1 in xs:
                tile = src[y0:y1, x0:x1, :]
                up = self._run_tile(proc, tile)
                wx = _axis_weights(
                    (x1 - x0) * s,
                    ramp_px,
                    rises=overlaps_x and x0 > 0,
                    falls=overlaps_x and x1 < sw,
                )
                weight = (wy[:, None] * wx[None, :])[:, :, None]
                co = x0 * s
                acc[: up.shape[0], co : co + up.shape[1]] += up * weight
                wsum[: up.shape[0], co : co + up.shape[1]] += weight

                done += 1
                if on_progress:
                    on_progress(done, total)

            stride = (ys[bi + 1][0] - y0) if bi + 1 < len(ys) else (y1 - y0)
            flush = min(stride * s, band_h)
            out[band_base * s : band_base * s + flush] = acc[:flush] / np.maximum(
                wsum[:flush], 1e-6
            )

            keep = band_h - flush
            if keep > 0:
                carry_acc = np.ascontiguousarray(acc[band_h - keep :])
                carry_wsum = np.ascontiguousarray(wsum[band_h - keep :])
            else:
                carry_acc = carry_wsum = None

            band_base += stride

        if p:
            y0 = p * s
            x0 = p * s
            out = out[y0 : y0 + out_h, x0 : x0 + out_w]

        return out

    def run_resilient(
        self,
        img: np.ndarray,
        proc_factory: Callable[[int], Callable[[np.ndarray], np.ndarray]],
        on_progress: Callable[[int, int], None] | None = None,
        min_tile: int = 64,
    ) -> np.ndarray:
        """Retry with progressively smaller tiles when the GPU runs out of memory.

        ``proc_factory(tile_size)`` must return a fresh ``proc`` closure -- CUDA
        contexts generally need to be rebuilt after an OOM, so reusing the old
        closure is not safe.
        """
        attempt = 0
        while True:
            try:
                return self.run(img, proc_factory(self.tile), on_progress)
            except (MemoryError, OutOfMemory) as exc:
                if isinstance(exc, OutOfMemory) and "budget" in str(exc):
                    raise
                attempt += 1
                self.stats.retries += 1
                next_tile = max(min_tile, self.tile // 2)
                if next_tile >= self.tile:
                    raise
                self.tile = next_tile
                self.overlap = min(self.overlap, max(4, next_tile // 32))


def estimate_tile_memory(tile: int, scale: int, channels: int = 3) -> int:
    """Bytes of packed float32 activations for one tile at the given scale."""
    return int(tile) * int(tile) * channels * 4 * (scale**2)


def suggest_tile(h: int, w: int, budget_bytes: int, scale: int, channels: int = 3) -> int:
    """Largest power-of-two-ish tile whose output fits a byte budget."""
    if budget_bytes <= 0:
        return 0
    per_px = channels * 4 * (scale**2)
    max_px = budget_bytes // max(1, per_px)
    side = int(math.sqrt(max(1, max_px)))
    side = max(64, min(side, 2048))
    return side
