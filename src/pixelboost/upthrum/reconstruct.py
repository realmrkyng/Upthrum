"""Assembly -- turning transported bands back into an image.

Three things happen here, and the order matters.

1. **Spectral reassembly.** The transported bands are added to a Lanczos-upscaled
   low-pass residual. Because the filter bank satisfies ``sum(G_k) + LP == 1``
   exactly, this is not a blend or a mix: it is the unique decomposition of the
   source, with each band replaced by its transported counterpart. At ``scale == 1``
   both halves reproduce the input bit for bit, so the sum does too.

2. **Topological constraint.** Applied to the *residual*, ``transported - smooth``,
   not to the finished luminance. That choice is deliberate. The residual is the
   structure the algorithm synthesised; the smooth path is structure that came from
   the source. Constraining the residual means UPTHRUM polices its own synthesis and
   never overwrites information the camera actually recorded. It also keeps the
   scale-1 identity intact for free -- the residual is identically zero there, so
   the relaxation has nothing to act on.

3. **Chroma.** Reconstructed on the smooth path and sharpened *only* by detail
   derived from the luma, gated by local luma-chroma gradient correlation. A colour
   edge therefore cannot exist unless a luminance edge exists underneath it. That is
   the structural reason UPTHRUM's chroma is crisp without the invented colour
   speckle that generative upscalers are known for.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np

from ..ops import resize_multi
from . import topology as topo
from .filters import FilterBank, lowpass_component
from .params import UpthrumParams
from .transform import BandAnalysis

EPS = 1e-8


@dataclass
class Reconstruction:
    """Bundled form of :func:`reconstruct_luminance`'s three return values."""

    luma: np.ndarray
    residual: np.ndarray
    diagnostics: dict[str, object] = field(default_factory=dict)

    @classmethod
    def of(cls, triple: tuple[np.ndarray, np.ndarray, dict[str, object]]) -> Reconstruction:
        luma, residual, info = triple
        return cls(luma=luma, residual=residual, diagnostics=info)


def _grad_magnitude(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    gy, gx = np.gradient(field.astype(np.float64))
    return gx, gy, np.sqrt(gx * gx + gy * gy)


def _orient_smooth(field: np.ndarray, theta: np.ndarray, radius: int) -> np.ndarray:
    """Smooth ``field`` along the orientation field ``theta``.

    A one-dimensional average taken along the local structure direction, which
    leaves structure across the direction untouched. This is what makes the chroma
    gate follow contours instead of spraying colour where the luma has no edge.
    """
    if radius <= 0:
        return field.astype(np.float64, copy=False)

    h, w = field.shape
    work = field.astype(np.float64)
    c = np.cos(theta).astype(np.float64)
    s = np.sin(theta).astype(np.float64)
    ys, xs = np.mgrid[0:h, 0:w]

    acc = work.copy()
    norm = np.ones((h, w), dtype=np.float64)
    for step in range(1, radius + 1):
        for sign in (-1.0, 1.0):
            sy = np.clip(np.round(ys + sign * step * s).astype(np.intp), 0, h - 1)
            sx = np.clip(np.round(xs + sign * step * c).astype(np.intp), 0, w - 1)
            weight = np.exp(-0.5 * (step * step) / max(radius * radius, 1e-6))
            acc += weight * work[sy, sx]
            norm += weight
    return acc / norm


def reconstruct_luminance(
    luminance: np.ndarray,
    analyses: Sequence[BandAnalysis],
    bank: FilterBank,
    transported: Sequence[np.ndarray],
    out_shape: tuple[int, int],
    scale: float,
    params: UpthrumParams,
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    """Reassemble the luminance channel and return ``(luma, residual, info)``.

    ``residual`` is the synthesised detail, i.e. the part of the output that is not
    simply a resample of the source. It is returned because the chroma stage needs
    it and because it is the only part of the output the topology stage is allowed
    to touch.
    """
    out_h, out_w = out_shape
    in_shape = (int(luminance.shape[0]), int(luminance.shape[1]))
    smooth = resize_multi(luminance.astype(np.float32), out_w, out_h)

    spectrum = np.fft.rfft2(luminance.astype(np.float64))
    low = lowpass_component(spectrum, bank, in_shape)
    del spectrum
    low_up = resize_multi(low.astype(np.float32), out_w, out_h)

    assembled = low_up.astype(np.float64)
    for band in transported:
        assembled += band.astype(np.float64)

    residual = assembled - smooth.astype(np.float64)
    info: dict[str, object] = {
        "scale": float(scale),
        "bands": len(transported),
        "residual_energy": float(np.mean(np.abs(residual))),
    }

    if not params.topology:
        info["topology"] = "disabled"
        return assembled.astype(np.float32), residual.astype(np.float32), info

    reference_tau = topo.persistence_threshold(luminance, params.persistence_relative)
    baseline = topo.signature(smooth, reference_tau, params.topology_max_pixels)

    luma, suppression = topo.suppress(
        assembled, smooth, reference_tau, max_pixels=params.topology_max_pixels
    )

    restored = 0
    if params.topology_repair:
        significant = topo.peaks(luminance, params.topology_max_pixels).significant(reference_tau)
        if len(significant) > 0:
            span = max(2, significant.pooled_factor // 2 + 2)
            mapped = significant.points.copy()
            mapped[:, 0] = (mapped[:, 0] + 0.5) * scale - 0.5
            mapped[:, 1] = (mapped[:, 1] + 0.5) * scale - 0.5
            luma, restored = topo.restore(
                luma, mapped, reference_tau, params.topology_max_pixels, radius=span
            )

    candidate = topo.signature(luma, reference_tau, params.topology_max_pixels)
    constrained = luma - smooth.astype(np.float64)

    info["topology"] = {
        "tau": round(reference_tau, 8),
        "baseline": baseline,
        "candidate": candidate,
        "matches": topo.matches(baseline, candidate),
        "suppressed_peaks": suppression.get("suppressed", 0),
        "restored_peaks": restored,
        "suppressed_energy": float(np.mean(np.abs(residual - constrained))),
    }
    return luma.astype(np.float32), constrained.astype(np.float32), info


def inject_chroma(
    chroma: tuple[np.ndarray, np.ndarray],
    luminance: np.ndarray,
    residual: np.ndarray,
    theta: np.ndarray,
    out_shape: tuple[int, int],
    params: UpthrumParams,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct the two chroma planes with correlation-gated luma detail.

    The gate is the cosine between the local luminance and chroma gradients, floored
    at zero, scaled by how much chroma structure there is relative to luma. Detail
    is therefore injected where the two channels demonstrably describe the same edge,
    and nowhere else -- a saturated but flat colour region receives none, so it
    cannot develop speckle.
    """
    out_h, out_w = out_shape
    cb, cr = chroma
    cb_up = resize_multi(cb.astype(np.float32), out_w, out_h).astype(np.float64)
    cr_up = resize_multi(cr.astype(np.float32), out_w, out_h).astype(np.float64)

    if not params.chroma or params.chroma_detail <= 0.0:
        return cb_up.astype(np.float32), cr_up.astype(np.float32)

    _, _, luma_mag = _grad_magnitude(luminance)
    luma_mag = np.maximum(luma_mag, EPS)

    detail = residual.astype(np.float64)

    planes = []
    for plane in (cb, cr):
        gx, gy, mag = _grad_magnitude(plane)
        lgy, lgx = np.gradient(luminance.astype(np.float64))
        dot = lgx * gx + lgy * gy
        corr = dot / (luma_mag * np.maximum(mag, EPS))
        gate = np.clip(corr, 0.0, 1.0) * np.clip(mag / luma_mag, 0.0, 1.0)

        if params.chroma_guided:
            gate = _orient_smooth(gate, theta, radius=2)
        else:
            gate = np.abs(gate)

        gate_up = resize_multi(gate.astype(np.float32), out_w, out_h).astype(np.float64)
        planes.append(gate_up)

    amount = float(params.chroma_detail)
    cb_out = cb_up + amount * detail * planes[0]
    cr_out = cr_up + amount * detail * planes[1]
    return cb_out.astype(np.float32), cr_out.astype(np.float32)


__all__ = ["Reconstruction", "inject_chroma", "reconstruct_luminance"]
