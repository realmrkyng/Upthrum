"""Value objects shared across the engine.

The library passes images around as ``float32`` numpy arrays in ``[0, 1]`` with
shape ``(H, W, 3)``. Alpha travel separately as ``(H, W)`` float arrays. Keeping
a single internal representation means every stage can be written and tested
without mode branching.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

BACKEND_AUTO = "auto"
ALPHA_MODES = ("lanczos", "nearest", "backend")


def resolve_target(
    src_w: int,
    src_h: int,
    *,
    scale: float | None = None,
    width: int | None = None,
    height: int | None = None,
    longest_side: int | None = None,
    keep_aspect: bool = True,
) -> tuple[int, int]:
    """Resolve an output size from whichever sizing hint the caller supplied.

    Precedence is explicit size first (``width``/``height``), then
    ``longest_side``, then the plain ``scale`` factor.

    Aspect handling: with ``keep_aspect`` on (the default) the missing dimension
    is derived from the source ratio. With it off, the missing dimension is left
    at its source value -- so ``width=320, keep_aspect=False`` means "320 wide,
    height untouched" rather than "320 square".
    """
    if width and height:
        return max(1, int(width)), max(1, int(height))

    if width:
        w = max(1, int(width))
        if keep_aspect:
            h = max(1, int(round(src_h * w / src_w)))
        else:
            h = max(1, int(height or src_h))
        return w, h

    if height:
        h = max(1, int(height))
        if keep_aspect:
            w = max(1, int(round(src_w * h / src_h)))
        else:
            w = max(1, int(width or src_w))
        return w, h

    if longest_side:
        target = max(1, int(longest_side))
        if src_w >= src_h:
            return target, max(1, int(round(src_h * target / src_w)))
        return max(1, int(round(src_w * target / src_h))), target

    factor = float(scale or 4.0)
    return max(1, int(round(src_w * factor))), max(1, int(round(src_h * factor)))


@dataclass
class ImageMeta:
    """Everything about the source file that is worth preserving."""

    width: int
    height: int
    format: str = "PNG"
    mode: str = "RGB"
    icc_profile: bytes | None = None
    exif: bytes | None = None
    icc_bytes: int = 0
    has_alpha: bool = False

    @property
    def megapixels(self) -> float:
        return self.width * self.height / 1_000_000.0


@dataclass
class EnhanceOptions:
    """User-facing knobs.

    Defaults are deliberately conservative: a mild guided-filter detail boost and
    nothing else, because over-sharpened output is the most common way an
    automatic pipeline ruins an image.
    """

    scale: float | None = 4.0
    width: int | None = None
    height: int | None = None
    longest_side: int | None = None
    keep_aspect: bool = True

    backend: str = BACKEND_AUTO
    model: str | None = None
    model_path: str | None = None
    provider: str | None = None
    fp16: bool = False
    threads: int = 0

    tile: int = 512
    tile_overlap: int = 16
    tile_pad: int = 16
    align: int = 8
    max_pixels: int = 64_000_000
    pre_downscale: bool = False

    denoise: float = 0.0
    denoise_radius: float = 3.0
    chroma_denoise: float = 0.0
    detail: float = 0.35
    detail_radius: int = 4
    detail_eps: float = 1e-3
    sharpen: float = 0.0
    sharpen_radius: float = 1.0
    sharpen_threshold: float = 0.004

    gamma: float = 1.0
    contrast: float = 0.0
    saturation: float = 1.0
    auto_levels: float = 0.0

    alpha_mode: str = "lanczos"
    output_format: str | None = None
    quality: int = 95
    preserve_metadata: bool = True

    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> EnhanceOptions:
        if not data:
            return cls()
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        unknown = {k: v for k, v in data.items() if k not in known}
        clean = {k: v for k, v in data.items() if k in known and k != "extra"}
        opts = cls(**clean)
        if unknown:
            opts.extra.update(unknown)
        return opts


@dataclass
class EnhanceResult:
    """The engine's return value, including enough telemetry to debug a deployment."""

    image: np.ndarray
    alpha: np.ndarray | None = None
    meta: ImageMeta | None = None
    backend: str = ""
    model: str | None = None
    provider: str = ""
    src_size: tuple[int, int] = (0, 0)
    dst_size: tuple[int, int] = (0, 0)
    tiles: int = 0
    elapsed_ms: float = 0.0
    notes: list = field(default_factory=list)

    @property
    def scale_factor(self) -> float:
        if not self.src_size[0]:
            return 0.0
        return self.dst_size[0] / self.src_size[0]

    def summary(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "model": self.model,
            "provider": self.provider,
            "src_size": list(self.src_size),
            "dst_size": list(self.dst_size),
            "scale": round(self.scale_factor, 4),
            "tiles": self.tiles,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "notes": list(self.notes),
        }
