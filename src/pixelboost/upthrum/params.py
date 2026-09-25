"""UPTHRUM parameters.

Every knob here corresponds to one term in the algorithm's definition, not to a
tuning hack. If you cannot point at the equation a parameter belongs to, it does
not belong in this file.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass
class UpthrumParams:
    """Configuration of the UPTHRUM reconstruction.

    The defaults are the ones used in the reference results and are the ones the
    invariants in ``tests/test_upthrum.py`` are stated against. Changing them
    changes which invariants hold, so read the docstrings before you do.
    """

    bands: int = 3
    """Number of log-Gabor bands in the hierarchy.

    Each band is analysed and transported independently at its own scale, which
    is what makes the method *hierarchical* rather than broadband. Three covers
    the useful range for photographic content; 4-5 helps on images that mix very
    fine texture with large flat regions, at a roughly linear cost in FFT time.
    """

    top_frequency: float = 0.22
    """Centre frequency of the highest band, in cycles per input pixel.

    Nyquist is 0.5. Sitting a little below it keeps the highest band away from
    the sampling limit, where phase is numerically fragile because the signal is
    barely resolved. Raising this to 0.3 sharpens fine texture and increases
    band-to-band aliasing in the transport.
    """

    band_sigma: float = 1.6
    """Log-Gabor bandwidth as a frequency ratio.

    A value of 1.6 gives roughly one octave, so adjacent bands overlap by about
    half. Band centres are placed one octave apart, so the filter bank tiles the
    spectrum with no gaps. Below about 1.2 the bands stop overlapping and the
    partition of unity breaks, which destroys the identity property.
    """

    phase_gain: float = 1.0
    """Multiplier on the phase transport term ``grad(phi) . delta``.

    This is the algorithm's core control. 1.0 is the mathematically correct
    linear extrapolation of the phase field. Below 1 the reconstruction
    under-shoots and edges soften toward the interpolant; above 1 it overshoots
    and edges develop a phase halo. Do not treat this as a sharpness slider --
    use ``amplitude_gamma`` and the post detail stage for that.
    """

    amplitude_gamma: float = 1.0
    """Exponent on the amplitude weight inside the phasor mean.

    The amplitude-weighted circular mean of unit phasors is the maximum
    likelihood phase estimate under additive Gaussian noise, so 1.0 is the
    principled value. Lowering it toward 0 lets low-amplitude (noisy) taps vote,
    which is occasionally useful for very low contrast detail.
    """

    coherence_power: float = 0.5
    """Exponent applied to the phase coherence gate ``kappa``.

    ``kappa`` is the resultant length of the local phasor cloud, in [0, 1]. 1.0
    applies the gate as-is, which is the strict reading; 0.5 relaxes it and
    preserves more of the reconstructed amplitude in partially coherent regions.
    Above roughly 2.0 the gate becomes nearly binary and produces visible
    switching artefacts at the boundary of coherent regions.
    """

    anisotropy: float = 0.55
    """Strength of the structure-aligned amplitude kernel, in [0, 1].

    Phase transport is intrinsically anisotropic, but the *amplitude* estimate is
    not: it needs an explicit orientation term to be smooth along an edge while
    staying sharp across it. 0 disables the term (isotropic interpolation of
    amplitude); 1 applies the full ridge kernel. This affects smoothness along
    contours, not edge sharpness, so it is safe to raise on noisy sources.
    """

    sigma_normal: float = 0.6
    """Kernel width across the structure, in input pixels.

    Keep this below 1.0. It is what preserves the amplitude step at an edge; a
    larger value softens the very thing the phase transport just sharpened.
    """

    sigma_tangent: float = 1.8
    """Kernel width along the structure, in input pixels.

    This is the axis that should be generous: amplitude varies slowly along a
    contour, so smoothing there removes interpolation staircase without
    destroying anything. Values of 1.5-3.0 are all reasonable; the upper end
    helps on synthetic gradients and hurts on fine texture.
    """

    topology: bool = True
    """Enable the persistent-homology stage.

    This is what distinguishes UPTHRUM from every interpolation or regression
    method: the synthesised detail is constrained to carry no significant
    structure that the source did not already have, so noise-scale critical points
    cannot be amplified into apparent texture.

    The constraint is one-sided, and deliberately so. Equality of signatures is the
    wrong target -- any resampling changes the pixel count, and the number of
    critical points above a fixed threshold therefore changes with it (measured:
    a factor of 1.46 at scale 2 for both UPTHRUM and Lanczos). What is meaningful,
    and what is actually enforced, is that the output does not carry *more*
    topological energy than a plain high-quality resample of the same source.
    That is an apples-to-apples comparison at identical resolution, and it is the
    direction in which the invariant can actually be violated.
    """

    persistence_relative: float = 0.18
    """Persistence threshold, as a fraction of a percentile range.

    Dense-sample critical points are then classified as noise and cancelled. The
    threshold is ``tau = persistence_relative * (p99 - p1)``, on percentiles so
    that a few blown highlights cannot inflate it.

    The value is calibrated, not chosen. For a roughly Gaussian field
    ``p99 - p1 ~ 4.6 * sigma``, so ``tau ~ sigma`` -- the persistence scale at
    which a peak is no more prominent than the noise it sits on -- requires
    ``persistence_relative ~ 0.22``. Measuring the actual separation on a synthetic
    target (79 real features, additive Gaussian noise) gives:

    ==========  ============  =============  =============
    value       clean image   sigma = 0.008  sigma = 0.030
    ==========  ============  =============  =============
    0.02        79            91             2575
    0.15        79            79             80
    0.20        79            79             79
    0.25        74            74             74
    0.30        14            44             55
    ==========  ============  =============  =============

    Below 0.10 the threshold sits under the noise floor and cancels nothing -- the
    stage is inert. Above 0.25 it starts eating real structure, and by 0.30 the
    image is gutted. 0.18 sits in the middle of the flat region where the separation
    is complete and the clean-image signature is untouched. Raise toward 0.22 for
    heavily compressed sources, where blocking artefacts are *coherent* and need a
    higher bar to be judged insignificant.
    """

    topology_max_pixels: int = 65536
    """Resolution cap for the topology analysis, in pixels.

    The merge tree is built by a union-find whose inner loop is inherently
    sequential, so it is capped and run on a block-pooled copy of the field.
    Block *max* pooling preserves maxima exactly, so no significant bright
    structure is lost by this; only sub-block-scale detail is invisible to the
    analysis, which is precisely the scale the filter is meant to ignore.

    Note the cap interacts with :func:`topology.restore`: extrema are located at
    block centres, so the caller must widen its "already present" window by
    ``factor // 2`` or the restoration will inject duplicate peaks a few pixels off
    the true apex. The callers do this from the reported ``pooled_factor``.
    """

    topology_repair: bool = True
    """Restore significant peaks that the transport erased.

    The counterpart to cancellation. Cancellation can only remove structure; it
    cannot notice that a real peak went missing, and numerical drift at the finest
    band is exactly where that happens.

    A source peak counts as erased when its height is not attained within
    ``0.25 * tau`` of its transported position. That tolerance is a separation
    argument, not a fudge: sampling shortfall (the output lattice has no sample
    exactly at the peak) measures 0.3-0.8% of peak height, while genuine erasure
    produces a shortfall of the order of the peak's persistence, which is at least
    ``tau``. There is more than an order of magnitude of clear space between them.
    Set the tolerance too low and this stage restores peaks that were never missing
    -- measured at 33 of 35 peaks on smooth Gaussian blobs with an exact test.

    Restoration raises a bounded Gaussian bump whose apex is the source's own peak
    height, so the reference sets the ceiling and the algorithm cannot exaggerate.
    Unlike the cancellation path, this operation is not contractive with respect to
    the transport result; it is bounded by the *source* instead. That is the weaker
    of the two guarantees, which is why the tolerance is set to keep it from firing
    outside genuine erasure.
    """

    chroma: bool = True
    """Reconstruct chroma at all. Off leaves chroma on the smooth path."""

    chroma_detail: float = 0.25
    """Fraction of the reconstructed luma detail injected into chroma.

    Gated by the local luma-chroma gradient correlation, so it only fires where
    the two channels genuinely share structure. This is the mechanism that gives
    colour crispness *without* the invented colour speckle that GAN upscalers
    produce, because every chroma edge here is derived from a luma edge.
    """

    chroma_guided: bool = True
    """Use the luma orientation field to steer the chroma reconstruction."""

    detail: float = 0.0
    """Post-transport micro-contrast, delegated to ``ops.detail_enhance``.

    Off by default. UPTHRUM's output is already sharp; stacking a detail boost
    on top is the most common way to ruin it. When enabled, this is the guided
    filter's *strength*, not its radius: the radius is fixed at four output
    pixels scaled by the transport factor, which keeps the pass expressing
    "one input pixel of local contrast" at every scale. 0.3-0.5 reads as
    sharper; above ~1.0 texture starts to look like watercolour.
    """

    band_rows: int = 0
    """Output rows processed per streaming block. 0 selects automatically."""

    max_pixels: int = 0
    """Refuse outputs above this pixel count. 0 means no limit."""

    guard_identity: bool = True
    """Assert the identity property when ``scale == 1``.

    At scale 1 every output sample lands exactly on an input sample, so the
    coherent mean must reproduce the input band exactly. This is a cheap,
    decisive self-check on the whole pipeline; it is enabled by default because
    it catches coordinate-convention bugs that are otherwise invisible.
    """

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> UpthrumParams:
        if not data:
            return cls()
        known = set(cls.__dataclass_fields__)  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in data.items() if k in known})
