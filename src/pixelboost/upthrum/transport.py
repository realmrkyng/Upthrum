"""Sub-pixel phase transport -- the core of UPTHRUM.

Everything else in this package exists to support the three lines of arithmetic
below, so this is the file worth reading first.

The idea in one paragraph
-------------------------

Interpolation is linear in intensity, and that is exactly why it blurs. Consider
a fine sinusoid sampled at ``t`` and ``t+1``; the value halfway between them is
``cos(phi)`` where the true signal wants ``cos(phi + g/2)``. Averaging the two
endpoint *values* gives ``cos(phi + g/2) cos(g/2)`` -- the correct phase, but
multiplied by ``cos(g/2)`` < 1. The detail is not lost, it is *attenuated*, and at
the finest scales ``cos(g/2)`` approaches zero. That attenuation is the entire
mechanism of interpolation blur, whether the kernel is bilinear, bicubic, Lanczos
or a windowed sinc.

UPTHRUM does not average values. It averages *phasors*, after first rolling each
one back to where the query point sits. Writing the local phase gradient
``g = grad(phi)`` and the offset from tap to query ``Delta = q - t``:

    phi_out(q) = arg( sum_t W_t A_t^gamma * exp( i * ( phi_t + phase_gain * g_t . Delta_t ) ) )

Each tap's phasor is advanced or retarded by exactly the amount the phase field
says it should be, so taps that share a wavefront arrive *in step* instead of
partially cancelling. For a locally plane wave the alignment is exact, and the
coherent mean reproduces the underlying signal with no attenuation at all -- not
"less blur than bicubic", but no attenuation term to begin with.

Why the phase is the right variable
-----------------------------------

Amplitude is what a sensor measures and it is a poor thing to interpolate: it is
bounded, biased by exposure, and its local variation carries no information about
*where* a structure is. Phase is unbounded, lives on a circle, and its gradient
points at the structure. Translation along a wavefront is a pure phase shift, so
the *shape* of the oscillation survives sub-pixel sampling intact -- which is
precisely the information a linear interpolant throws away. Reconstructing phase
and re-deriving intensity from it is not a sharpening trick applied after the
resize; it is a different reconstruction problem, and it does not have the
attenuation term.

The estimate is self-diagnosing
-------------------------------

The modulus of the coherent sum measures how much the taps actually agreed:

    kappa = | sum_t W_t A_t^gamma * exp(i * (phi_t + ...)) | / sum_t W_t A_t^gamma

``kappa = 1``: every tap landed on one wavefront, so the local model (plane wave)
fits and the reconstruction is trustworthy. ``kappa ~ 0``: the taps disagree, the
plane-wave model does not apply there, and any detail invented at that pixel would
be fiction. So instead of a heuristic blur/heuristics mix, the same sum that
produces the phase also reports its own confidence, and the amplitude is scaled by
``kappa ** coherence_power``. Incoherent regions are damped toward the interpolant
automatically; nothing has to detect them.

The tap set
-----------

Taps are the four corners of the bilinear cell containing ``q``, i.e. the exact
set of input samples that carry any information about that output sample. This is
not a cost-saving choice, it is the choice that makes the scale-1 identity exact:
at ``scale == 1`` every ``q`` is an integer, three of the four bilinear weights are
exactly zero, and the sum collapses to the coincident sample -- so the transport
reproduces the input to machine precision, with no special case anywhere.

Within that cell the weights are additionally shaped by a structure-aligned
Gaussian: narrow across the local orientation, wide along it. This matters because
the taps straddle the structure they are meant to preserve, and isotropic
averaging across an edge softens the amplitude step that the phase transport just
placed exactly. Shaping the cell weights keeps the amplitude sharp across the edge
while leaving it free along the contour. The kernel widths are expressed in input
pixels so that a single pair of numbers means the same thing at every scale.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .params import UpthrumParams
from .transform import BandAnalysis

EPS = 1e-12
_BLOCK_BYTES = 32 << 20


def _axis_coords(out_len: int, scale: float, in_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Query positions in input-pixel units, with their cell index and fraction.

    The mapping is the pixel-centre one, ``q = (u + 0.5) / scale - 0.5``, not the
    corner one ``q = u / scale``. The two differ by ``0.5 * (1 - 1/scale)`` input
    pixels, which is exactly the kind of half-pixel offset that shows up as a soft
    double edge when the low-pass path (handled by an ordinary Lanczos resampler,
    which uses pixel centres) is added back to the transported bands. Both agree at
    ``scale == 1``, where the mapping is the identity on integers.

    ``q`` is clamped to the input range, which is border replication: the outermost
    output sample refers to the edge input sample rather than to a phantom one
    outside the image.
    """
    q = (np.arange(out_len, dtype=np.float64) + 0.5) / float(scale) - 0.5
    np.clip(q, 0.0, max(float(in_len) - 1.0, 0.0), out=q)
    base = np.minimum(np.floor(q).astype(np.intp), max(in_len - 1, 0))
    base = np.maximum(base, 0)
    return q, base, q - base


def query_coordinates(out_shape: tuple[int, int], scale: float, in_shape: tuple[int, int]):
    """Per-output-pixel query point, expressed in input-pixel coordinates."""
    qy, by, fy = _axis_coords(out_shape[0], scale, in_shape[0])
    qx, bx, fx = _axis_coords(out_shape[1], scale, in_shape[1])
    return qy, qx, by, bx, fy, fx


def _auto_rows(width: int, height: int) -> int:
    per_row = max(width, 1) * 8 * 8
    return int(max(1, min(height, max(1, _BLOCK_BYTES // per_row))))


def _transport_block(
    analysis: BandAnalysis,
    theta: np.ndarray,
    scale: float,
    params: UpthrumParams,
    y0: int,
    y1: int,
) -> np.ndarray:
    """Transport one horizontal strip of the output."""
    h_in, w_in = analysis.band.shape
    w_out = int(round(w_in * scale))
    rows = y1 - y0

    _, _, by, bx, fy_all, fx = query_coordinates((int(round(h_in * scale)), w_out), scale, (h_in, w_in))
    by = by[y0:y1]
    fy = fy_all[y0:y1]

    amp = analysis.amplitude.ravel().astype(np.float64)
    pha = analysis.phase.ravel().astype(np.float64)
    gx = analysis.grad_x.ravel().astype(np.float64)
    gy = analysis.grad_y.ravel().astype(np.float64)
    th = theta.ravel().astype(np.float64)

    gamma = float(params.amplitude_gamma)
    amp_g = np.power(np.maximum(amp, EPS), gamma)
    amp_gm1 = np.power(np.maximum(amp, EPS), gamma - 1.0)

    sigma_n = max(float(params.sigma_normal), 1e-3)
    aniso = float(np.clip(params.anisotropy, 0.0, 1.0))
    sigma_t = sigma_n + aniso * (float(params.sigma_tangent) - sigma_n)
    inv_n = 1.0 / (sigma_n * sigma_n)
    inv_t = 1.0 / (sigma_t * sigma_t)
    gain = float(params.phase_gain)

    acc = np.zeros((rows, w_out), dtype=np.complex128)
    mass = np.zeros((rows, w_out), dtype=np.float64)
    flat = np.zeros((rows, w_out), dtype=np.float64)

    for oy in (0, 1):
        wy = (fy if oy else 1.0 - fy)[:, None]
        if not np.any(wy):
            continue
        for ox in (0, 1):
            wx = (fx if ox else 1.0 - fx)[None, :]
            cell = wy * wx
            if not np.any(cell):
                continue

            ty = np.minimum(np.maximum(by + oy, 0), h_in - 1)
            tx = np.minimum(np.maximum(bx + ox, 0), w_in - 1)
            idx = ty[:, None] * w_in + tx[None, :]
            inside = (
                (by[:, None] + oy >= 0)
                & (by[:, None] + oy < h_in)
                & (bx[None, :] + ox >= 0)
                & (bx[None, :] + ox < w_in)
            )

            d_y = (oy - fy)[:, None]
            d_x = (ox - fx)[None, :]
            c = np.cos(th[idx])
            s = np.sin(th[idx])
            along = d_x * c + d_y * s
            across = -d_x * s + d_y * c
            kern = np.exp(-0.5 * (across * across * inv_n + along * along * inv_t))

            weight = cell * kern * inside
            advance = pha[idx] + gain * (
                gx[idx] * (fx[None, :] - ox) + gy[idx] * (fy[:, None] - oy)
            )

            step = weight * amp_g[idx]
            acc += step * np.exp(1j * advance)
            mass += step
            flat += weight * amp_gm1[idx]

    safe_mass = np.maximum(mass, EPS)
    kappa = np.abs(acc) / safe_mass
    smooth = mass / np.maximum(flat, EPS)
    amplitude = smooth * np.power(kappa, float(params.coherence_power))
    return (amplitude * np.cos(np.angle(acc))).astype(np.float32)


def transport_band(
    analysis: BandAnalysis,
    theta: np.ndarray,
    out_shape: tuple[int, int],
    scale: float,
    params: UpthrumParams,
) -> np.ndarray:
    """Transport one band to the output lattice.

    The result is forced to zero mean, which is not a cosmetic correction but the
    restoration of an exact invariant: the log-Gabor transfer is *identically* zero
    at DC, so every source band has zero mean to machine precision. The transport
    can violate that, because the tap weights are not symmetric about the query
    point -- the bilinear cell splits as ``frac`` and ``1 - frac``, and the
    anisotropic kernel is evaluated with each tap's own orientation. The resulting
    drift is small (about 1e-4 of the dynamic range on a blob field) but it lands
    directly on overall brightness, which is meant to be carried untouched by the
    low-pass path.

    Removing it costs nothing and cannot break the scale-1 identity: the source band
    already has a mean of ~1e-17, so the subtraction moves the output by that much.
    """
    rows = params.band_rows if params.band_rows > 0 else _auto_rows(out_shape[1], out_shape[0])
    rows = int(max(1, min(rows, out_shape[0])))
    out = np.empty(out_shape, dtype=np.float32)
    for y0 in range(0, out_shape[0], rows):
        y1 = min(y0 + rows, out_shape[0])
        out[y0:y1] = _transport_block(analysis, theta, scale, params, y0, y1)
    out -= np.float32(out.mean(dtype=np.float64))
    return out


def transport(
    analyses: Sequence[BandAnalysis],
    theta: np.ndarray,
    out_shape: tuple[int, int],
    scale: float,
    params: UpthrumParams,
) -> list[np.ndarray]:
    """Transport every band of the hierarchy."""
    return [transport_band(a, theta, out_shape, scale, params) for a in analyses]
