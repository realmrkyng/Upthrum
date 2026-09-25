"""Monogenic decomposition -- the observation stage of UPTHRUM.

For every band produced by the filter bank we form the *monogenic signal* (Felsberg
& Sommer): the band-pass component ``B`` together with its two Riesz transforms
``Rx``, ``Ry``. From them we take the four quantities UPTHRUM actually transports:

    A        = sqrt(B^2 + Rdir^2)          local amplitude
    phi      = atan2(Rdir, B)              local phase
    theta                                  local orientation
    grad(phi)                              phase gradient

The monogenic phase is folded, and that matters
-----------------------------------------------

The textbook monogenic phase is ``atan2(sqrt(Rx^2 + Ry^2), B)``, which lands in
``[0, pi]``. It is a perfectly good description of *where* a structure is, and it is
what phase-congruency work uses. It is the wrong variable to *differentiate*.

The reason is a fold. ``|sin|`` is not differentiable at zero, so that phase has a
kink wherever the oscillation crosses zero, and its gradient flips sign there --
twice per period. Extrapolating a kinked field linearly is exactly what the
transport must not do, and the failure is not subtle: measured on a pure sinusoid
at 0.125 cycles/pixel, samples landing on the input lattice were reconstructed to
1e-8 while samples landing between lattice points were off by up to 0.028 on a 0.4
amplitude, a 7% error appearing in a periodic pattern with the sub-pixel offset.

The fix is to keep the sign. Take the component of the Riesz vector *along the
local orientation* rather than its magnitude:

    Rdir = Rx * cos(theta) + Ry * sin(theta)
    phi  = atan2(Rdir, B)

``theta`` comes from the doubled-angle mean of the per-band orientations, which is
smooth where ``atan2(Ry, Rx)`` is not: the raw orientation jumps by pi at every
zero crossing, but ``2 * theta`` does not, so averaging on the doubled map removes
the jump while the average is still a genuine orientation field. With a smooth
``theta``, ``Rdir`` is a *signed* directional Hilbert component and ``phi`` is the
true analytic phase -- unfolded, smooth, and with the correct gradient everywhere.

The global sign of ``phi`` is arbitrary (flipping ``theta`` by pi flips it, and
``cos`` cannot tell the difference) but it is *consistent* wherever ``theta`` varies
continuously, and a globally sign-flipped phase satisfies
``phi_t + grad(phi) . delta = phi(q)`` just as well. The sign is therefore not a
degree of freedom that needs resolving; it cancels.

Nothing downstream changes shape. ``A * cos(phi) == B`` holds for *any* choice of
odd component, since ``A`` is defined as the hypotenuse and ``phi`` as its angle, so
the band is still recovered exactly and the partition-of-unity identity is
untouched. What changes is only that the phase is now safe to extrapolate.

Why the phase gradient is computed the way it is
------------------------------------------------

``phi`` lives on a circle and cannot be fed to a spectral derivative -- every wrap
would produce a spike of height ``2*pi``, which is a whole-period error in the
transport. The fix is to differentiate the *phasor* instead of the angle. Writing
``z = exp(i*phi)``,

    grad(z) = i * z * grad(phi)   =>   grad(phi) = Im( conj(z) * grad(z) )

``z`` is continuous and differentiable everywhere it is defined -- including across
the branch cut, where ``z`` passes smoothly through 1 -- so differentiating it
spectrally is valid. This gives the exact phase gradient with no unwrapping step and
no heuristic. The alternative, branch-cut-aware unwrapping on a 2-D lattice, is
ill-posed precisely in the low-amplitude regions where the answer matters least.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from .filters import (
    FilterBank,
    derivative_multipliers,
    riesz_multipliers,
    split_band,
)

EPS = 1e-8

_Components = tuple[np.ndarray, np.ndarray, np.ndarray]


@dataclass
class BandAnalysis:
    """Everything the transport stage needs from one band, and nothing else."""

    index: int = 0
    centre: float = 0.0
    band: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.float32))
    amplitude: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.float32))
    phase: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.float32))
    orientation: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.float32))
    grad_x: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.float32))
    grad_y: np.ndarray = field(default_factory=lambda: np.zeros((1, 1), np.float32))
    energy: float = 0.0

    @property
    def shape(self) -> tuple[int, int]:
        return self.band.shape  # type: ignore[return-value]

    def phasor(self) -> np.ndarray:
        """``exp(i*phi)`` as complex64 -- rebuilt on demand to save a whole array."""
        return (np.cos(self.phase) + 1j * np.sin(self.phase)).astype(np.complex64)

    def diagnostics(self) -> dict:
        return {
            "index": self.index,
            "centre": round(self.centre, 4),
            "energy": round(self.energy, 6),
            "phase_span": round(float(np.ptp(self.phase)), 4),
        }


def _phase_gradient(z: np.ndarray, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Exact ``grad(phi)`` from the unit phasor field, without unwrapping.

    Differentiates the real and imaginary parts in one batched transform, so the
    cost is one batch forward and two batch inverses rather than four separate
    pairs.
    """
    zc = np.stack([z.real, z.imag], axis=0).astype(np.float64)
    spec = np.fft.rfft2(zc, axes=(-2, -1))
    dx, dy = derivative_multipliers(shape)

    gx = np.fft.irfft2(spec * dx, s=shape, axes=(-2, -1))
    gy = np.fft.irfft2(spec * dy, s=shape, axes=(-2, -1))

    conj = np.conj(z).astype(np.complex128)
    dphi_dx = np.imag(conj * (gx[0] + 1j * gx[1]))
    dphi_dy = np.imag(conj * (gy[0] + 1j * gy[1]))
    return dphi_dx.astype(np.float32), dphi_dy.astype(np.float32)


def _riesz(band: np.ndarray, rx_mult: np.ndarray, ry_mult: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    spectrum = np.fft.rfft2(band.astype(np.float64))
    shape = band.shape
    rx = np.fft.irfft2(spectrum * rx_mult, s=shape)
    ry = np.fft.irfft2(spectrum * ry_mult, s=shape)
    return rx, ry


def smooth_orientation(components: Sequence[_Components]) -> np.ndarray:
    """Amplitude-weighted orientation across bands, on the doubled-angle map.

    Orientation is a *direction*, not a vector: ``theta`` and ``theta + pi`` name
    the same ridge, and ``atan2(Ry, Rx)`` therefore jumps by pi at every zero
    crossing of the band. Averaging the raw angles would cancel a contour against
    itself; averaging on the doubled map does not, because a pi jump becomes a 2pi
    jump, which is no jump at all:

        theta = 0.5 * atan2( sum_k w_k sin 2theta_k, sum_k w_k cos 2theta_k )

    The result is a smooth orientation field, which is what makes the signed
    directional component in :func:`analyse` meaningful. Weighting by amplitude
    means a band that is locally silent does not vote.
    """
    if not components:
        return np.zeros((1, 1), np.float32)

    acc_sin = np.zeros_like(components[0][0], dtype=np.float64)
    acc_cos = np.zeros_like(components[0][0], dtype=np.float64)
    for band, rx, ry in components:
        weight = np.sqrt(band.astype(np.float64) ** 2 + rx * rx + ry * ry)
        theta = np.arctan2(ry, rx)
        acc_sin += weight * np.sin(2.0 * theta)
        acc_cos += weight * np.cos(2.0 * theta)
    return (0.5 * np.arctan2(acc_sin, acc_cos)).astype(np.float32)


def analyse_band(
    band: np.ndarray,
    index: int = 0,
    centre: float = 0.0,
    orientation: np.ndarray | None = None,
) -> BandAnalysis:
    """Decompose one band into its monogenic quantities.

    ``orientation`` may be supplied to share a smooth field across the hierarchy
    (the path :func:`analyse` takes). When omitted it is derived from this band
    alone, which is correct but has the pi jumps that :func:`smooth_orientation`
    removes -- usable for a single band, not for transport.
    """
    shape = band.shape
    rx_mult, ry_mult = riesz_multipliers(shape)
    rx, ry = _riesz(band, rx_mult, ry_mult)

    if orientation is None:
        theta = np.arctan2(ry, rx).astype(np.float32)
    else:
        theta = orientation

    cos_t = np.cos(theta).astype(np.float64)
    sin_t = np.sin(theta).astype(np.float64)
    directional = rx * cos_t + ry * sin_t

    b = band.astype(np.float64)
    amplitude = np.sqrt(b * b + directional * directional)
    safe = np.maximum(amplitude, EPS)
    phase = np.arctan2(directional, b)

    z = (b + 1j * directional) / safe
    grad_x, grad_y = _phase_gradient(z, shape)

    return BandAnalysis(
        index=index,
        centre=centre,
        band=band.astype(np.float32),
        amplitude=amplitude.astype(np.float32),
        phase=phase.astype(np.float32),
        orientation=np.asarray(theta, dtype=np.float32),
        grad_x=grad_x,
        grad_y=grad_y,
        energy=float(np.mean(amplitude)),
    )


def decompose(luminance: np.ndarray, bank: FilterBank) -> list[_Components]:
    """Band-pass component plus Riesz pair, per band.

    The real shape is passed down explicitly rather than inferred from the
    half-spectrum, because ``rfft2`` makes the inference lossy for odd widths.
    """
    shape = (int(luminance.shape[0]), int(luminance.shape[1]))
    spectrum = np.fft.rfft2(luminance.astype(np.float64))
    rx_mult, ry_mult = riesz_multipliers(shape)

    components: list[_Components] = []
    for transfer in bank.bands:
        band = split_band(spectrum, transfer, shape)
        rx, ry = _riesz(band, rx_mult, ry_mult)
        components.append((band, rx, ry))
    del spectrum
    return components


def analyse(luminance: np.ndarray, bank: FilterBank) -> list[BandAnalysis]:
    """Decompose every band of a luminance image.

    Two passes over the bands, which is the price of a smooth orientation: the
    orientation cannot be known until every band has been filtered, and the
    directional component cannot be formed until the orientation is known. Holding
    the Riesz pairs of all bands at once is the memory this costs, and it is worth
    it -- see the module docstring for what the alternative costs in accuracy.
    """
    components = decompose(luminance, bank)
    theta = smooth_orientation(components)
    return [
        analyse_band(band, index=i, centre=float(bank.centres[i]), orientation=theta)
        for i, (band, _, _) in enumerate(components)
    ]


def dominant_orientation(
    analyses: Sequence[BandAnalysis],
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Amplitude-weighted orientation across an already-analysed hierarchy.

    Kept as a separate entry point for callers holding ``BandAnalysis`` objects
    (the tests, the CUDA path). When the analyses came from :func:`analyse` they
    already share one orientation field and this returns it unchanged.
    """
    if not analyses:
        return np.zeros(shape or (1, 1), np.float32)

    acc_sin = np.zeros_like(analyses[0].amplitude, dtype=np.float64)
    acc_cos = np.zeros_like(analyses[0].amplitude, dtype=np.float64)
    for a in analyses:
        weight = a.amplitude.astype(np.float64)
        acc_sin += weight * np.sin(2.0 * a.orientation)
        acc_cos += weight * np.cos(2.0 * a.orientation)

    return (0.5 * np.arctan2(acc_sin, acc_cos)).astype(np.float32)


def coherence(analyses: Sequence[BandAnalysis], shape: tuple[int, int] | None = None) -> np.ndarray:
    """Cross-band phase coherence in [0, 1] -- the algorithm's local confidence.

    For each pixel the per-band unit phasors are averaged with amplitude weights
    and the resultant length measured:

        kappa = | sum_k w_k z_k | / sum_k w_k

    ``kappa = 1`` means every scale agrees on where the local oscillation sits: a
    genuine, well-resolved structure. ``kappa = 0`` means the scales disagree: the
    "detail" there is noise or aliasing, and it is exactly where a naive sharpening
    method manufactures texture that is not in the source. Weighting each band by
    its own amplitude is what keeps a silent band from voting.
    """
    if not analyses:
        return np.ones(shape or (1, 1), np.float32)

    acc = np.zeros_like(analyses[0].amplitude, dtype=np.complex128)
    mass = np.zeros_like(analyses[0].amplitude, dtype=np.float64)
    for a in analyses:
        weight = a.amplitude.astype(np.float64)
        acc += weight * np.exp(1j * a.phase.astype(np.float64))
        mass += weight

    mass = np.maximum(mass, EPS)
    return (np.abs(acc) / mass).astype(np.float32)
