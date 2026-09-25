"""Image decoding and encoding.

Pillow is the only hard image dependency, which keeps the CPU install to
numpy + Pillow + (optionally) onnxruntime. The module hides the awkward parts of
that choice:

* 16-bit and palette sources are normalised to float32 RGB for the engine, and
  the *original* mode is remembered so a greyscale input comes back greyscale
  instead of silently tripling in size.
* Alpha is lifted out into its own plane. Running a GAN over RGBA is a classic
  mistake -- it will happily paint opaque colour into the transparent halo
  around a cut-out subject.
* Output is written atomically (temp file + ``os.replace``), so a crashed
  process never leaves a truncated JPEG where a good one used to be.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import numpy as np

from pixelboost import ops
from pixelboost.errors import ImageReadError, ImageWriteError, PixelBoostError
from pixelboost.types import ImageMeta

GRAY_MODES = ("L", "LA", "I", "I;16", "I;16B", "I;16L", "F")

FORMAT_ALIASES = {
    "jpg": "JPEG",
    "jpeg": "JPEG",
    "png": "PNG",
    "webp": "WEBP",
    "tif": "TIFF",
    "tiff": "TIFF",
    "bmp": "BMP",
    "avif": "AVIF",
}

LOSSLESS = ("PNG", "BMP", "TIFF", "WEBP")


@dataclass
class LoadedImage:
    rgb: np.ndarray
    alpha: np.ndarray | None
    meta: ImageMeta
    gray: bool = False


def normalize_format(name: str | None, fallback: str = "PNG") -> str:
    if not name:
        return fallback
    key = name.strip().lstrip(".").lower()
    return FORMAT_ALIASES.get(key, key.upper())


def _to_pil(src: str | bytes | os.PathLike | Any) -> Any:
    from PIL import Image

    if hasattr(src, "convert") and hasattr(src, "size"):
        return src
    if isinstance(src, (bytes, bytearray)):
        return Image.open(io.BytesIO(bytes(src)))
    return Image.open(os.fspath(src))


def load(src: str | bytes | os.PathLike | Any) -> LoadedImage:
    """Decode to ``(float32 RGB in [0,1], optional alpha, metadata)``."""
    try:
        im = _to_pil(src)
        im.load()
    except Exception as exc:
        raise ImageReadError(f"cannot decode image: {exc}") from exc

    original_mode = im.mode
    info: dict[str, Any] = dict(getattr(im, "info", {}) or {})
    icc = info.get("icc_profile")
    exif = info.get("exif")

    has_alpha = original_mode in ("RGBA", "LA", "PA") or (
        original_mode == "P" and "transparency" in info
    )
    gray = original_mode in GRAY_MODES or original_mode == "LA"

    try:
        if has_alpha:
            rgba = im.convert("RGBA")
            arr = np.asarray(rgba, dtype=np.float32)
            rgb = arr[..., :3] / 255.0
            alpha = arr[..., 3] / 255.0
        elif gray:
            lum = im.convert("L")
            base = np.asarray(lum, dtype=np.float32) / 255.0
            rgb = np.repeat(base[..., None], 3, axis=2)
            alpha = None
        else:
            conv = im.convert("RGB")
            rgb = np.asarray(conv, dtype=np.float32) / 255.0
            alpha = None
    except Exception as exc:
        raise ImageReadError(f"unsupported image mode {original_mode!r}: {exc}") from exc

    w, h = im.size
    meta = ImageMeta(
        width=w,
        height=h,
        format=(im.format or "PNG").upper(),
        mode=original_mode,
        icc_profile=icc,
        exif=exif,
        icc_bytes=len(icc) if icc else 0,
        has_alpha=has_alpha,
    )
    return LoadedImage(rgb=np.ascontiguousarray(rgb), alpha=alpha, meta=meta, gray=gray)


def _build_pil(
    rgb: np.ndarray,
    alpha: np.ndarray | None,
    gray: bool,
) -> Any:
    from PIL import Image

    u8 = ops.float_to_u8(rgb)
    if gray:
        lum = ops.float_to_u8(ops.luminance(rgb)) if rgb.shape[2] == 3 else u8[..., 0]
        if alpha is not None:
            a = ops.float_to_u8(alpha)
            return Image.fromarray(np.stack([lum, a], axis=-1), mode="LA")
        return Image.fromarray(lum, mode="L")

    if alpha is not None:
        a = ops.float_to_u8(alpha)
        return Image.fromarray(np.concatenate([u8, a[..., None]], axis=2), mode="RGBA")
    return Image.fromarray(u8, mode="RGB")


def save_kwargs(fmt: str, quality: int) -> dict[str, Any]:
    fmt = fmt.upper()
    if fmt == "JPEG":
        return {
            "quality": int(quality),
            "subsampling": 0,
            "optimize": True,
            "progressive": True,
        }
    if fmt == "PNG":
        return {"compress_level": 6, "optimize": True}
    if fmt == "WEBP":
        return {"quality": int(quality), "method": 6}
    if fmt == "TIFF":
        return {"compression": "tiff_lzw"}
    if fmt == "AVIF":
        return {"quality": int(quality)}
    return {}


def encode(
    rgb: np.ndarray,
    alpha: np.ndarray | None = None,
    *,
    fmt: str = "PNG",
    quality: int = 95,
    meta: ImageMeta | None = None,
    gray: bool = False,
    preserve_metadata: bool = True,
) -> bytes:
    """Serialise to bytes for an HTTP response."""
    fmt = normalize_format(fmt)
    im = _build_pil(rgb, alpha, gray)
    if fmt == "JPEG" and im.mode == "RGBA":
        im = im.convert("RGB")

    params = save_kwargs(fmt, quality)
    if preserve_metadata and meta is not None:
        if meta.icc_profile and fmt in ("JPEG", "PNG", "TIFF", "WEBP"):
            params["icc_profile"] = meta.icc_profile
        if meta.exif and fmt in ("JPEG", "TIFF", "WEBP"):
            params["exif"] = meta.exif

    buf = io.BytesIO()
    try:
        im.save(buf, format=fmt, **params)
    except Exception as exc:
        raise ImageWriteError(f"cannot encode {fmt}: {exc}") from exc
    return buf.getvalue()


def save(
    path: str,
    rgb: np.ndarray,
    alpha: np.ndarray | None = None,
    *,
    fmt: str | None = None,
    quality: int = 95,
    meta: ImageMeta | None = None,
    gray: bool = False,
    preserve_metadata: bool = True,
) -> str:
    """Atomically write an image to disk."""
    target = os.fspath(path)
    fmt = normalize_format(fmt or os.path.splitext(target)[1], fallback="PNG")
    blob = encode(
        rgb,
        alpha,
        fmt=fmt,
        quality=quality,
        meta=meta,
        gray=gray,
        preserve_metadata=preserve_metadata,
    )

    directory = os.path.dirname(os.path.abspath(target)) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".pixelboost-", dir=directory)
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(blob)
        os.replace(tmp, target)
    except Exception as exc:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise ImageWriteError(f"cannot write {target}: {exc}") from exc
    return target


def probe(src: str | bytes | Any) -> dict[str, Any]:
    """Cheap header-only read, used to validate an upload before queueing it.

    Must never leak a Pillow exception: the HTTP layer maps
    :class:`~pixelboost.errors.ImageReadError` to a 415 and anything else to a
    500, so an unwrapped ``UnidentifiedImageError`` turns a bad upload into a
    server error.
    """
    try:
        im = _to_pil(src)
        info = dict(getattr(im, "info", {}) or {})
        return {
            "width": im.size[0],
            "height": im.size[1],
            "mode": im.mode,
            "format": (im.format or "").upper(),
            "has_alpha": im.mode in ("RGBA", "LA", "PA") or "transparency" in info,
            "animated": bool(getattr(im, "n_frames", 1) > 1),
            "icc_bytes": len(info.get("icc_profile") or b""),
        }
    except PixelBoostError:
        raise
    except Exception as exc:
        raise ImageReadError(f"cannot identify image data: {exc}") from exc
