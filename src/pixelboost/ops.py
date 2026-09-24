"""Dependency-light image operators.

Everything in this module is pure numpy. That matters for two reasons:

1. The ``classical`` backend must work on a bare server with no OpenCV, no
   PyTorch and no model files. It is the always-available fallback.
2. Every operator here is also used as *post-processing* on top of the neural
   backends, so it runs on 4x-sized buffers where OpenCV's float64 paths would
   blow the memory budget.

All functions take and return ``float32`` arrays in ``[0, 1]``. Shapes are
``(H, W, 3)`` for colour images and ``(H, W)`` or ``(H, W, C)`` for the helpers
that are channel-agnostic.
"""

from __future__ import annotations

import math

import numpy as np

RGB_WEIGHTS = np.array([0.299, 0.587, 0.114], dtype=np.float32)


def luminance(img: np.ndarray) -> np.ndarray:
    """Rec.601 luma of an RGB image in ``[0, 1]``."""
    return np.asarray(img, dtype=np.float32) @ RGB_WEIGHTS


def _axis_slice(axis: int, ndim: int, start: int, stop: int) -> tuple:
    sl = [slice(None)] * ndim
    sl[axis] = slice(start, stop)
    return tuple(sl)


def _box_blur_axis(a: np.ndarray, radius: int, axis: int) -> np.ndarray:
    """Single-pass moving-average along one axis, O(n) via a cumsum."""
    n = a.shape[axis]
    r = int(min(radius, max(0, n - 1)))
    if r < 1:
        return a

    k = 2 * r + 1
    pad = [(0, 0)] * a.ndim
    pad[axis] = (r, r)
    padded = np.pad(a, pad, mode="edge")

    cum = np.cumsum(padded, axis=axis, dtype=np.float32)
    lead_shape = list(cum.shape)
    lead_shape[axis] = 1
    cum = np.concatenate([np.zeros(lead_shape, np.float32), cum], axis=axis)

    hi = cum[_axis_slice(axis, cum.ndim, k, k + n)]
    lo = cum[_axis_slice(axis, cum.ndim, 0, n)]
    return ((hi - lo) / np.float32(k)).astype(np.float32, copy=False)


def box_blur(img: np.ndarray, radius: int, passes: int = 1) -> np.ndarray:
    """Separable box filter. Three passes approximate a Gaussian closely."""
    if radius < 1:
        return img
    out = img
    for _ in range(max(1, passes)):
        out = _box_blur_axis(out, radius, 0)
        out = _box_blur_axis(out, radius, 1)
    return out


def gaussian_sigma_to_box_radius(sigma: float) -> int:
    """Match a 3-pass box filter cascade to a target Gaussian sigma.

    Each box of width ``w = 2r + 1`` contributes variance ``(w^2 - 1) / 12``;
    three of them give ``(w^2 - 1) / 4``, so inverting yields the expression
    below. This is the classic central-limit Gaussian approximation.
    """
    return max(1, int(round((math.sqrt(4.0 * float(sigma) ** 2 + 1.0) - 1.0) / 2.0)))


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img
    return box_blur(img, gaussian_sigma_to_box_radius(sigma), passes=3)


def guided_filter(guide: np.ndarray, src: np.ndarray, radius: int, eps: float) -> np.ndarray:
    """Fast guided filter (He et al., 2013).

    This is the workhorse of the classical backend. Unlike a Gaussian blur it
    smooths flat regions while leaving edges intact, which is exactly the
    property needed to separate *structure* from *texture*:

    ``detail = src - guided_filter(src, src)`` is a high-pass signal with almost
    no edge bleed, so amplifying it sharpens without producing halos.
    """
    r = max(1, int(radius))
    guide = np.asarray(guide, dtype=np.float32)
    src = np.asarray(src, dtype=np.float32)

    mean_i = box_blur(guide, r)
    mean_p = box_blur(src, r)
    corr_i = box_blur(guide * guide, r)
    corr_ip = box_blur(guide * src, r)

    var_i = corr_i - mean_i * mean_i
    cov_ip = corr_ip - mean_i * mean_p

    a = cov_ip / (var_i + np.float32(eps))
    b = mean_p - a * mean_i

    mean_a = box_blur(a, r)
    mean_b = box_blur(b, r)
    return (mean_a * guide + mean_b).astype(np.float32, copy=False)


def detail_enhance(
    img: np.ndarray,
    radius: int = 4,
    eps: float = 1e-3,
    strength: float = 0.35,
) -> np.ndarray:
    """Edge-preserving local contrast / texture boost.

    ``strength`` scales the extracted detail band. 0.3-0.6 reads as "sharper"
    without the crunch of a plain unsharp mask; above ~1.0 texture starts to
    look like watercolour.
    """
    if strength <= 0:
        return img
    base = guided_filter(img, img, radius, eps)
    detail = img - base
    return np.clip(base + detail * np.float32(1.0 + strength), 0.0, 1.0).astype(np.float32)


def unsharp(
    img: np.ndarray,
    amount: float,
    radius: float = 1.0,
    threshold: float = 0.004,
) -> np.ndarray:
    """Unsharp mask with a soft threshold so sensor noise is not amplified."""
    if amount <= 0:
        return img
    high = img - gaussian_blur(img, radius)
    if threshold > 0:
        soft = np.clip((np.abs(high) - threshold) / threshold, 0.0, 1.0)
        high = high * soft
    return np.clip(img + np.float32(amount * 1.5) * high, 0.0, 1.0).astype(np.float32)


def denoise(img: np.ndarray, radius: float = 3.0, sigma: float = 0.04) -> np.ndarray:
    """Edge-preserving smoothing. ``sigma`` is a noise level in ``[0, 1]``."""
    if sigma <= 0:
        return img
    r = max(1, int(round(radius)))
    out = guided_filter(img, img, r, float(sigma) ** 2 + 1e-6)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def rgb_to_ycbcr(img: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = luminance(img)
    r = img[..., 0]
    b = img[..., 2]
    cb = (b - y) * np.float32(0.564) + np.float32(0.5)
    cr = (r - y) * np.float32(0.713) + np.float32(0.5)
    return y, cb, cr


def ycbcr_to_rgb(y: np.ndarray, cb: np.ndarray, cr: np.ndarray) -> np.ndarray:
    crs = (cr - np.float32(0.5)) / np.float32(0.713)
    cbs = (cb - np.float32(0.5)) / np.float32(0.564)
    r = y + crs
    b = y + cbs
    g = (y - np.float32(0.299) * r - np.float32(0.114) * b) / np.float32(0.587)
    return np.stack([r, g, b], axis=-1)


def chroma_denoise(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Blur only the chroma planes.

    Neural upscalers often hallucinate colour speckle in smooth gradients. The
    eye is far more tolerant of blurred chroma than of blurred luma, so this
    removes the artefact at almost zero perceptual cost.
    """
    if sigma <= 0:
        return img
    y, cb, cr = rgb_to_ycbcr(img)
    return np.clip(
        ycbcr_to_rgb(y, gaussian_blur(cb, sigma), gaussian_blur(cr, sigma)),
        0.0,
        1.0,
    ).astype(np.float32)


def adjust_color(
    img: np.ndarray,
    contrast: float = 0.0,
    saturation: float = 1.0,
    gamma: float = 1.0,
) -> np.ndarray:
    """Luma-preserving contrast, saturation scaling and gamma in one pass."""
    out = img
    if gamma != 1.0:
        out = np.power(np.clip(out, 0.0, 1.0), np.float32(1.0 / float(gamma)))
    if contrast:
        y = luminance(out)[..., None]
        out = out + (y - np.float32(0.5)) * np.float32(contrast)
    if saturation != 1.0:
        y = luminance(out)[..., None]
        out = y + (out - y) * np.float32(saturation)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def percentile_stretch(
    img: np.ndarray,
    low: float = 0.5,
    high: float = 99.5,
    strength: float = 1.0,
) -> np.ndarray:
    """Auto-levels driven by luma percentiles.

    A JPEG that has been through three social networks usually occupies only the
    30-200 range. Stretching first makes the upscaler's job much easier.
    ``strength`` blends between the original and the fully stretched result.
    """
    if strength <= 0:
        return img
    y = luminance(img)
    lo = float(np.percentile(y, low))
    hi = float(np.percentile(y, high))
    if hi - lo < 1e-4:
        return img

    gain = 1.0 / (hi - lo)
    g = 1.0 + (gain - 1.0) * strength
    b = -lo * gain * strength
    return np.clip(img * np.float32(g) + np.float32(b), 0.0, 1.0).astype(np.float32)


def lanczos_kernel(x: np.ndarray, a: float = 3.0) -> np.ndarray:
    """Normalised sinc window. ``numpy.sinc`` already returns 1 at x = 0."""
    x = np.asarray(x, dtype=np.float32)
    w = np.sinc(x) * np.sinc(x / np.float32(a))
    return np.where(np.abs(x) < a, w, 0.0).astype(np.float32)


def lanczos_taps(n_in: int, n_out: int, a: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Precompute gather indices and weights for one axis of a Lanczos resample.

    Returns ``(idx, weights)`` of shape ``(n_out, n_taps)``. Indices are clamped
    to the valid range and the weights of the clamped taps are zeroed, then the
    row is renormalised. That is the standard way to keep a resample
    DC-preserving at the borders without special-casing edge pixels.
    """
    scale = n_in / n_out
    support = a * max(1.0, scale)
    lo = int(math.floor(support))
    n_taps = 2 * lo + 2

    centers = (np.arange(n_out, dtype=np.float64) + 0.5) * scale - 0.5
    base = np.floor(centers).astype(np.int64) - lo
    idx = base[:, None] + np.arange(n_taps, dtype=np.int64)[None, :]

    dist = centers[:, None] - idx.astype(np.float64)
    weights = np.sinc(dist) * np.sinc(dist / a)
    weights = np.where(np.abs(dist) < a, weights, 0.0)

    valid = (idx >= 0) & (idx < n_in)
    idx = np.clip(idx, 0, n_in - 1)
    weights = (weights * valid).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-8)
    return idx, weights


def _resample_axis(
    img: np.ndarray,
    out_n: int,
    axis: int,
    a: float,
    max_elems: int,
) -> np.ndarray:
    """Apply one Lanczos pass along ``axis`` with a bounded working set.

    Naively gathering every tap at once costs ``H * n_out * n_taps * C`` floats,
    which for a 4000px source upscaled 4x is several gigabytes. Instead we
    accumulate tap by tap over a chunk of rows, capping the peak buffer at
    ``max_elems`` floats.
    """
    n_in = img.shape[axis]
    idx, weights = lanczos_taps(n_in, out_n, a)
    n_taps = idx.shape[1]
    ndim = img.ndim
    tail = int(np.prod(img.shape[2:])) if ndim > 2 else 1

    if axis == 1:
        out = np.empty((img.shape[0], out_n) + img.shape[2:], np.float32)
        chunk = max(1, max_elems // max(1, out_n * tail))
        for y0 in range(0, img.shape[0], chunk):
            y1 = min(y0 + chunk, img.shape[0])
            acc = np.zeros((y1 - y0, out_n) + img.shape[2:], np.float32)
            band = img[y0:y1]
            for t in range(n_taps):
                acc += band[:, idx[:, t]] * weights[:, t].reshape(
                    (1, out_n) + (1,) * (ndim - 2)
                )
            out[y0:y1] = acc
        return out

    out = np.empty((out_n,) + img.shape[1:], np.float32)
    chunk = max(1, max_elems // max(1, tail))
    for y0 in range(0, out_n, chunk):
        y1 = min(y0 + chunk, out_n)
        acc = np.zeros((y1 - y0,) + img.shape[1:], np.float32)
        for t in range(n_taps):
            acc += img[idx[y0:y1, t]] * weights[y0:y1, t].reshape(
                (y1 - y0,) + (1,) * (ndim - 1)
            )
        out[y0:y1] = acc
    return out


def resize_lanczos(
    img: np.ndarray,
    out_w: int,
    out_h: int,
    a: float = 3.0,
    max_elems: int = 1 << 22,
) -> np.ndarray:
    """High-quality separable Lanczos resample, any ratio, bounded memory."""
    h, w = img.shape[0], img.shape[1]
    out_w = max(1, int(out_w))
    out_h = max(1, int(out_h))
    if out_w == w and out_h == h:
        return img.astype(np.float32, copy=False)

    work = img.astype(np.float32, copy=False)
    if out_h != h:
        work = _resample_axis(work, out_h, 0, a, max_elems)
    if out_w != w:
        work = _resample_axis(work, out_w, 1, a, max_elems)
    return work


def resize_multi(
    img: np.ndarray,
    out_w: int,
    out_h: int,
    max_step_ratio: float = 2.0,
    a: float = 3.0,
    max_elems: int = 1 << 22,
) -> np.ndarray:
    """Lanczos in <=``max_step_ratio`` increments.

    A single 8x Lanczos pass samples a 6-tap kernel that is far too narrow for
    the job and softens badly. Chaining 2x passes lets each step use the full
    kernel, which is why every good resizer works this way.
    """
    h, w = img.shape[0], img.shape[1]
    out_w = max(1, int(out_w))
    out_h = max(1, int(out_h))
    if (out_w, out_h) == (w, h):
        return img.astype(np.float32, copy=False)

    cur_w, cur_h = w, h
    work = img.astype(np.float32, copy=False)
    while True:
        ratio = max(out_w / cur_w, out_h / cur_h)
        if ratio <= max_step_ratio + 1e-6:
            return resize_lanczos(work, out_w, out_h, a, max_elems)
        next_w = min(out_w, max(cur_w + 1, int(math.ceil(cur_w * max_step_ratio))))
        next_h = min(out_h, max(cur_h + 1, int(math.ceil(cur_h * max_step_ratio))))
        work = resize_lanczos(work, next_w, next_h, a, max_elems)
        cur_w, cur_h = next_w, next_h


def fit_within(w: int, h: int, max_pixels: int) -> tuple[int, int]:
    """Scale a size down proportionally until it fits a pixel budget."""
    total = w * h
    if max_pixels <= 0 or total <= max_pixels:
        return w, h
    factor = math.sqrt(max_pixels / total)
    return max(1, int(w * factor)), max(1, int(h * factor))


def alpha_to_float(alpha: np.ndarray) -> np.ndarray:
    if alpha.dtype == np.uint8:
        return alpha.astype(np.float32) / 255.0
    if alpha.dtype == np.uint16:
        return alpha.astype(np.float32) / 65535.0
    return np.clip(alpha.astype(np.float32), 0.0, 1.0)


def float_to_u8(arr: np.ndarray) -> np.ndarray:
    return np.clip(arr * 255.0 + 0.5, 0.0, 255.0).astype(np.uint8)
