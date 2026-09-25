"""Frequency-domain analysis operators for UPTHRUM.

Everything here is a linear operator applied to the half-spectrum produced by
``numpy.fft.rfft2``. Working in the frequency domain is not an optimisation
trick for this algorithm -- it is what makes two of its defining properties
exact:

* **Partition of unity.** The band filters are normalised so that
  ``sum_k G_k + LP == 1`` holds at every frequency. That is the reason UPTHRUM is
  an exact identity at scale 1, and it is why the low-pass path can be added back
  to the transported bands without double counting or leaving a spectral hole.
* **Exact Riesz transform.** The monogenic signal needs the Riesz transform,
  whose kernel is non-local and non-separable. In the frequency domain it is a
  pointwise complex multiplier, which is both cheaper and more accurate than any
  spatial approximation.

The Riesz transform has a property that the whole method rests on: for a signal
that is locally constant along some direction (a ridge or an edge profile), the
Riesz transform evaluated along the profile direction reduces *exactly* to the
1-D Hilbert transform. It is, in effect, the 1-D Hilbert transform taken
simultaneously along every direction, which is why a single transform can supply
a quadrature component for structures of any orientation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EPS = 1e-6


def shape_of(spectrum: np.ndarray) -> tuple[int, int]:
    """Recover the real-space shape from an ``rfft2`` half-spectrum.

    Only valid when the real width is even. ``rfft2`` keeps ``w // 2 + 1`` columns,
    so inverting that as ``2 * (ncols - 1)`` recovers ``w`` for even ``w`` but
    ``w - 1`` for odd ``w`` -- a silent off-by-one that only odd-sized inputs
    expose. Callers that know the true shape should pass it explicitly instead of
    inferring it; that is why :func:`split_band` and :func:`lowpass_component` both
    take an optional ``shape``.
    """
    h = spectrum.shape[0]
    w = (spectrum.shape[1] - 1) * 2
    return h, w


def frequency_grid(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Signed frequency coordinates, in cycles per pixel, on the ``rfft2`` grid."""
    h, w = shape
    fy = np.fft.fftfreq(h, d=1.0).astype(np.float64)[:, None]
    fx = np.fft.rfftfreq(w, d=1.0).astype(np.float64)[None, :]
    return fx, fy


def radius_grid(shape: tuple[int, int]) -> np.ndarray:
    """Radial frequency magnitude, floored away from zero so logs stay finite."""
    fx, fy = frequency_grid(shape)
    r = np.sqrt(fx * fx + fy * fy)
    r[0, 0] = EPS
    return r


def riesz_multipliers(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Fourier multipliers of the 2-D Riesz transform.

    ``R = -i * f / |f|``. The DC entry is set to zero, which is what makes the
    transform zero-mean -- the Riesz components carry no DC and therefore cannot
    shift the image's brightness.
    """
    fx, fy = frequency_grid(shape)
    r = np.sqrt(fx * fx + fy * fy)
    r[0, 0] = np.inf
    rx = (-1j * fx / r).astype(np.complex128)
    ry = (-1j * fy / r).astype(np.complex128)
    rx[0, 0] = 0.0
    ry[0, 0] = 0.0
    return rx, ry


def derivative_multipliers(shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Fourier multipliers for ``d/dx`` and ``d/dy``, in cycles per pixel.

    Spectral differentiation is exact for band-limited data. The phase gradient
    ``grad(phi)`` is the single most sensitivity-critical quantity in the whole
    algorithm -- a one-pixel error in it becomes a whole-period phase error in
    the transport -- so this is deliberately not a finite difference.
    """
    fx, fy = frequency_grid(shape)
    dx = (2j * np.pi * fx).astype(np.complex128)
    dy = (2j * np.pi * fy).astype(np.complex128)
    return dx, dy


def log_gabor(shape: tuple[int, int], centre: float, sigma: float) -> np.ndarray:
    """Log-Gabor transfer function centred at ``centre`` cycles per pixel.

    Log-Gabor rather than ordinary Gabor because the DC response is zero in the
    limit and the transfer stays symmetric under octave scaling: bands placed one
    octave apart are then self-similar, which is what lets a single ``sigma``
    describe the whole bank.
    """
    r = radius_grid(shape)
    ratio = np.log(r / float(centre)) / np.log(float(sigma))
    return np.exp(-0.5 * ratio * ratio).astype(np.float64)


@dataclass
class FilterBank:
    """A partition-of-unity band hierarchy plus its low-pass residual.

    ``bands`` tile the spectrum; ``lowpass`` is whatever is left over:

        ``lowpass = 1 - sum(bands)``

    By construction ``sum(bands) + lowpass == 1`` at every frequency, so
    ``IFFT(FFT(x) * (sum(bands) + lowpass)) == x`` to machine precision. Every
    reconstruction path in UPTHRUM depends on this identity holding.
    """

    bands: list[np.ndarray] = field(default_factory=list)
    lowpass: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    centres: list[float] = field(default_factory=list)
    coverage: float = 1.0

    def __len__(self) -> int:
        return len(self.bands)

    def summary(self) -> dict:
        return {
            "bands": len(self.bands),
            "centres": [round(c, 4) for c in self.centres],
            "peak_band_coverage": round(self.coverage, 4),
        }


def build_filter_bank(
    shape: tuple[int, int],
    bands: int = 3,
    top_frequency: float = 0.22,
    sigma: float = 1.6,
) -> FilterBank:
    """Construct the octave-spaced log-Gabor hierarchy.

    Band centres are ``top_frequency * 2**-k``. After building them the whole set
    is scaled by ``1 / max(sum)`` if that sum ever exceeds one, which is the
    condition that keeps the residual low-pass non-negative. Scaling the bands
    down rather than clipping the low-pass up is deliberate: a negative low-pass
    would invert contrast in the residual, which is far worse than a slightly
    under-weighted band.
    """
    n = max(1, int(bands))
    centres = [float(top_frequency) * (2.0**-k) for k in range(n)]
    filters = [log_gabor(shape, c, sigma) for c in centres]

    total = np.zeros_like(filters[0])
    for f in filters:
        total += f
    peak = float(total.max())
    if peak > 1.0:
        scale = 1.0 / peak
        filters = [f * scale for f in filters]
        total = total * scale

    lowpass = (1.0 - total).astype(np.float64)
    np.clip(lowpass, 0.0, None, out=lowpass)
    return FilterBank(bands=filters, lowpass=lowpass, centres=centres, coverage=peak)


def split_band(
    spectrum: np.ndarray,
    transfer: np.ndarray,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Inverse-transform one band of an already-computed half-spectrum."""
    return np.fft.irfft2(spectrum * transfer, s=shape or shape_of(spectrum)).astype(np.float32)


def lowpass_component(
    spectrum: np.ndarray,
    bank: FilterBank,
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    return np.fft.irfft2(spectrum * bank.lowpass, s=shape or shape_of(spectrum)).astype(np.float32)
