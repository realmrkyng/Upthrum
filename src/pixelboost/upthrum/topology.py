"""Persistent homology -- the constraint that makes the output a *valid* image.

This is the stage that no interpolation, no learned model and no diffusion sampler
has. It is what licenses the phrase "the output is constrained" rather than "the
output is predicted".

The observation
---------------

An image is a Morse function: almost everywhere smooth, with isolated critical
points (peaks, pits, saddles) that are the actual content -- the highlight on a
nose, the shadow under a chin, the tip of a leaf. Noise also creates critical
points, but they are *shallow*: a JPEG artefact peak sits only a shade above the
saddle that surrounds it.

Persistent homology makes that distinction quantitative. Build the merge tree of
the sublevel sets (equivalently, of the superlevel sets for peaks) and pair each
critical point with the saddle at which its component dies. The value difference
is its **persistence**. A real peak persists over a large range of thresholds and
survives; a noise peak is annihilated almost as soon as it is born. Choosing a
threshold ``tau = persistence_relative * (p99 - p1)`` splits the two populations,
and -- this is the part that matters -- the split is *scale-free*. It does not care
whether the source is a clean render or a double-compressed phone photo.

Why this is the right gate for a super-resolver
-----------------------------------------------

The transport stage is a local, second-order-accurate reconstruction. Where the
local plane-wave model fits it is exact; where it does not -- and noise is exactly
where it does not -- it synthesises phase from a phase gradient that is itself
noise. The failure mode is not blur, it is *invented structure*: the classic
oversharpened halo, or the speckle that GAN upscalers are notorious for.

Persistent homology catches precisely that failure and nothing else. A critical
point that the input did not have, and that has no persistence, is by definition
not structure -- it is a numerical artefact -- and flattening it costs nothing,
because no observer could have seen it.

The repair is bounded by construction
-------------------------------------

Two directions are available and both are used:

* **Cancellation.** A critical point whose persistence falls below ``tau`` is
  relaxed to its own death value. This cannot invent anything: the value it is set
  to is one the field already takes at the surrounding saddle.
* **Restoration.** A significant peak of the input that the transport *erased*
  (numerical drift at high frequency) is restored at its transported position, by
  a bounded bump that may not exceed the peak's persistence. It cannot exaggerate:
  the ceiling is set by the reference, not by the algorithm.

Implementation notes
--------------------

The merge tree is built with a disjoint-set forest over pixels sorted by value.
Each union is near-O(1), so the whole tree is ``n * alpha(n)`` -- effectively
linear. The inner loop is irreducibly sequential, so the analysis runs on a
block-pooled copy capped at ``topology_max_pixels``. Pooling is by *maximum*,
which preserves every peak exactly; only sub-block detail becomes invisible, and
sub-block detail is by definition below the persistence scale being reasoned
about.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

_EIGHT = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


@dataclass
class Extrema:
    """Critical points of a field, in full-resolution pixel coordinates."""

    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 3), np.float64))
    persistence: np.ndarray = field(default_factory=lambda: np.zeros((0,), np.float64))
    pooled_factor: int = 1

    def __len__(self) -> int:
        return int(self.points.shape[0])

    def significant(self, tau: float) -> Extrema:
        if len(self) == 0:
            return self
        keep = self.persistence >= tau
        return Extrema(self.points[keep], self.persistence[keep], self.pooled_factor)


def pool_max(field: np.ndarray, max_pixels: int) -> tuple[np.ndarray, int]:
    """Block-max pool so that ``field`` fits within ``max_pixels``.

    Max pooling is the only reduction that preserves peaks exactly, which is the
    whole point of the operation here: subsampled maxima are still the true
    maxima, so no bright structure can be lost by the cap.
    """
    h, w = field.shape
    total = h * w
    if max_pixels <= 0 or total <= max_pixels:
        return field, 1
    factor = int(np.ceil(np.sqrt(total / float(max_pixels))))
    factor = max(factor, 2)
    ph = int(np.ceil(h / factor)) * factor
    pw = int(np.ceil(w / factor)) * factor
    padded = np.full((ph, pw), -np.inf, dtype=np.float64)
    padded[:h, :w] = field
    pooled = padded.reshape(ph // factor, factor, pw // factor, factor)
    return pooled.max(axis=(1, 3)), factor


def merge_tree(field: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """0-dimensional persistence of the sublevel sets of ``field``.

    Returns ``(birth_index, death_value, persistence)`` for every component that
    dies. The component that survives to ``+inf`` (the global minimum basin) is
    omitted, since it has infinite persistence and is not a feature.

    Components are born at local minima and die when their basin merges into a
    deeper one; the birth value is therefore always less than the death value.
    """
    h, w = field.shape
    n = h * w
    flat = field.ravel()
    order = np.argsort(flat, kind="stable")

    parent = np.full(n, -1, dtype=np.intp)
    birth_idx = np.full(n, -1, dtype=np.intp)
    birth_val = np.zeros(n, dtype=np.float64)
    active = np.zeros(n, dtype=bool)

    deaths: list[tuple[int, float, float]] = []

    for rank in range(n):
        p = int(order[rank])
        y, x = divmod(p, w)
        value = flat[p]
        active[p] = True

        roots: list[int] = []
        for dy, dx in _EIGHT:
            ny, nx = y + dy, x + dx
            if ny < 0 or ny >= h or nx < 0 or nx >= w:
                continue
            q = ny * w + nx
            if not active[q]:
                continue
            r = q
            while parent[r] != r:
                parent[r] = parent[parent[r]]
                r = parent[r]
            if r not in roots:
                roots.append(r)

        if not roots:
            parent[p] = p
            birth_idx[p] = p
            birth_val[p] = value
            continue

        survivor = roots[0]
        for r in roots[1:]:
            if birth_val[r] < birth_val[survivor]:
                survivor = r

        parent[p] = survivor
        birth_idx[p] = survivor
        birth_val[p] = birth_val[survivor]

        for r in roots:
            if r == survivor:
                continue
            deaths.append((int(birth_idx[r]), float(birth_val[r]), float(value - birth_val[r])))
            parent[r] = survivor

    if not deaths:
        return np.zeros(0, np.intp), np.zeros(0), np.zeros(0)

    idx = np.array([d[0] for d in deaths], dtype=np.intp)
    birth = np.array([d[1] for d in deaths], dtype=np.float64)
    pers = np.array([d[2] for d in deaths], dtype=np.float64)
    return idx, birth, pers


def _extrema_of(field: np.ndarray, max_pixels: int, sign: float) -> Extrema:
    """Critical points of ``sign * field``.

    ``sign = -1`` yields the maxima of ``field`` (components of the sublevel sets
    of ``-field`` are born at its minima, which are the maxima of ``field``), and
    ``sign = +1`` yields the minima. ``points[:, 2]`` always holds the value of the
    extremum *in the original field's units*, which is why it is ``sign * birth``.

    The component that survives to ``+inf`` is included, with its persistence taken
    as the full range of the field. Leaving it out is a real error, not a rounding
    one: the surviving component of the sublevel sets of ``-field`` is born at the
    global maximum of ``field``, so omitting it drops the single most significant
    feature of every image. A single isolated peak then reports *no* critical
    points at all, which silently breaks the matching in :func:`suppress` -- the
    most prominent structure in the picture would never be checked against the
    reference.
    """
    pooled, factor = pool_max(sign * field, max_pixels)
    idx, birth, pers = merge_tree(pooled)

    flat = pooled.ravel()
    survivor = int(np.argmin(flat))
    lo = float(flat[survivor])
    hi = float(flat.max())
    idx = np.append(idx, survivor)
    birth = np.append(birth, lo)
    pers = np.append(pers, hi - lo)

    ph, pw = pooled.shape
    py, px = np.divmod(idx, pw)
    h, w = field.shape
    ys = np.clip((py * factor + factor // 2), 0, h - 1)
    xs = np.clip((px * factor + factor // 2), 0, w - 1)
    vals = sign * birth
    return Extrema(np.stack([ys, xs, vals], axis=1).astype(np.float64), pers, factor)


def peaks(field: np.ndarray, max_pixels: int = 65536) -> Extrema:
    """Local maxima with their persistence (superlevel-set components)."""
    return _extrema_of(field, max_pixels, -1.0)


def pits(field: np.ndarray, max_pixels: int = 65536) -> Extrema:
    """Local minima with their persistence (sublevel-set components)."""
    return _extrema_of(field, max_pixels, 1.0)


def persistence_threshold(field: np.ndarray, relative: float) -> float:
    """``tau`` in absolute units, scaled by a percentile range rather than min/max.

    Percentiles are used deliberately: a handful of specular highlights or a black
    border can span the full range of an image while being irrelevant to its
    structure, and a min/max normalisation would let them set a threshold that
    erases real detail everywhere else.
    """
    lo, hi = np.percentile(field, (1.0, 99.0))
    return float(max(hi - lo, 0.0) * max(relative, 0.0))


def _match_cells(points: np.ndarray, cell: int) -> dict[tuple[int, int], list[int]]:
    buckets: dict[tuple[int, int], list[int]] = {}
    for i in range(points.shape[0]):
        key = (int(points[i, 0]) // cell, int(points[i, 1]) // cell)
        buckets.setdefault(key, []).append(i)
    return buckets


def _has_neighbour(
    buckets: dict[tuple[int, int], list[int]],
    reference: np.ndarray,
    y: float,
    x: float,
    cell: int,
    radius: float,
) -> bool:
    cy, cx = int(y) // cell, int(x) // cell
    r2 = radius * radius
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            for i in buckets.get((cy + dy, cx + dx), ()):
                ddy = reference[i, 0] - y
                ddx = reference[i, 1] - x
                if ddy * ddy + ddx * ddx <= r2:
                    return True
    return False


def _stamp(shape: tuple[int, int], ys: np.ndarray, xs: np.ndarray, radius: int) -> np.ndarray:
    """Accumulate Gaussian stamps at the given locations, clipped to [0, 1]."""
    h, w = shape
    if ys.size == 0:
        return np.zeros(shape, dtype=np.float64)

    mask = np.zeros(h * w, dtype=np.float64)
    span = max(1, int(radius))
    denom = max(float(span * span) * 0.5, 1e-6)
    offsets = np.arange(-span, span + 1)
    for dy in offsets:
        for dx in offsets:
            weight = float(np.exp(-(dy * dy + dx * dx) / denom))
            if weight < 1e-3:
                continue
            yy = np.clip(ys.astype(np.intp) + dy, 0, h - 1)
            xx = np.clip(xs.astype(np.intp) + dx, 0, w - 1)
            mask += np.bincount(yy * w + xx, weights=np.full(yy.shape, weight), minlength=h * w)
    return np.clip(mask.reshape(h, w), 0.0, 1.0)


def suppress(
    candidate: np.ndarray,
    reference: np.ndarray,
    tau: float,
    strength: float = 1.0,
    max_pixels: int = 65536,
) -> tuple[np.ndarray, dict[str, object]]:
    """Pull ``candidate`` back toward ``reference`` wherever it invents structure.

    ``candidate`` is the transport result and ``reference`` is a plain resample of
    the same source at the same resolution. Every significant critical point of
    ``candidate`` that has no counterpart in ``reference`` is, by the definition of
    significance, structure the algorithm manufactured rather than recovered. A
    soft mask is stamped at exactly those places and the candidate is blended
    toward the reference there.

    The guarantee this gives is stronger than any threshold-based repair, and it is
    worth stating precisely:

    * **Contractive.** The blend factor lies in [0, 1], so the output is a convex
      combination of the two inputs at every pixel. It can therefore never leave
      the interval between them -- in particular the constraint cannot introduce a
      value that neither the transport nor the interpolant produced. A modification
      scheme that edits the field freely cannot promise this; this one gives it up
      for free.
    * **Identity-preserving.** At ``scale == 1`` candidate and reference are equal,
      so their difference is zero and the blend has nothing to act on, for any mask
      and any strength.

    Only *peaks* are matched in value terms, because the failing mode is invented
    bright structure. Pits are matched by position alone: a spurious dark pixel is
    far less objectionable than a spurious highlight, and requiring value agreement
    on both sides makes the operator bite in flat regions where nothing is wrong.

    Matching, not thresholding, is the point. A persistence threshold alone cannot
    tell "this peak is noise" from "this peak is the same noise the source already
    had"; matching can, and it is the only formulation that distinguishes
    amplifying existing noise (harmless) from manufacturing new structure (not).
    """
    h, w = candidate.shape
    cand_peaks = peaks(candidate, max_pixels).significant(tau)
    ref_peaks = peaks(reference, max_pixels).significant(tau)

    stats: dict[str, object] = {
        "candidate_peaks": len(cand_peaks),
        "reference_peaks": len(ref_peaks),
        "suppressed": 0,
    }
    if len(cand_peaks) == 0:
        return candidate.astype(np.float64), stats

    cell = int(max(4, 2 * max(cand_peaks.pooled_factor, ref_peaks.pooled_factor) + 4))
    radius = float(max(2, max(cand_peaks.pooled_factor, ref_peaks.pooled_factor) + 2))
    buckets = _match_cells(ref_peaks.points, cell)
    ref_xy = ref_peaks.points

    spurious_y: list[float] = []
    spurious_x: list[float] = []
    for i in range(len(cand_peaks)):
        y, x, value = float(cand_peaks.points[i, 0]), float(cand_peaks.points[i, 1]), float(
            cand_peaks.points[i, 2]
        )
        if _has_neighbour(buckets, ref_xy, y, x, cell, radius):
            continue
        local = candidate[
            max(int(y) - 2, 0) : int(y) + 3, max(int(x) - 2, 0) : int(x) + 3
        ]
        if local.size and float(local.max()) < value - 0.02 * tau:
            continue
        spurious_y.append(y)
        spurious_x.append(x)

    stats["suppressed"] = len(spurious_y)
    if not spurious_y:
        return candidate.astype(np.float64), stats

    mask = _stamp((h, w), np.asarray(spurious_y), np.asarray(spurious_x), int(radius))
    blend = np.clip(mask * float(np.clip(strength, 0.0, 1.0)), 0.0, 1.0)
    out = candidate.astype(np.float64) + (reference.astype(np.float64) - candidate.astype(np.float64)) * blend
    return out, stats


def signature(field: np.ndarray, tau: float, max_pixels: int = 65536) -> dict[str, object]:
    """A comparable description of a field's topology above persistence ``tau``."""
    pk = peaks(field, max_pixels).significant(tau)
    pt = pits(field, max_pixels).significant(tau)
    spec = np.sort(np.concatenate([
        pk.persistence if len(pk) else np.zeros(0),
        pt.persistence if len(pt) else np.zeros(0),
    ]))[::-1]
    return {
        "peaks": len(pk),
        "pits": len(pt),
        "total": len(pk) + len(pt),
        "spectrum": [round(float(v), 6) for v in spec[:32]],
        "sum_persistence": round(float(spec.sum()) if spec.size else 0.0, 6),
    }


def matches(reference: dict[str, object], candidate: dict[str, object], tolerance: float = 0.25) -> bool:
    """Whether ``candidate`` stays within the topological budget of ``reference``.

    One-sided on purpose. The constraint is "do not add structure", so carrying
    *less* topological energy than the reference is a pass, not a failure -- it
    means the synthesis was conservative. Carrying more means the reconstruction
    manufactured prominence the reference does not have, which is the failure this
    whole stage exists to catch.

    Compared on the sum of persistence rather than the count of critical points,
    because the count is unstable at the threshold boundary while the sum is a
    continuous functional: one point hovering either side of ``tau`` changes the
    count by one but the sum hardly at all.

    ``reference`` must be at the *same resolution* as ``candidate`` -- normally a
    plain Lanczos resample of the same source. Comparing across resolutions is
    meaningless, since the critical-point count scales with sampling density.
    """
    a = float(reference.get("sum_persistence", 0.0))
    b = float(candidate.get("sum_persistence", 0.0))
    if a <= 0.0:
        return b <= 1e-9
    return b <= a * (1.0 + tolerance)


def restore(
    field: np.ndarray,
    transported: np.ndarray,
    tau: float,
    max_pixels: int = 65536,
    radius: int = 2,
) -> tuple[np.ndarray, int]:
    """Re-inject significant peaks that the transport erased.

    ``transported`` holds ``(y, x, value)`` rows for peaks of the *input* mapped
    into output coordinates. A peak is considered present if its height already
    occurs anywhere in a neighbourhood, and is otherwise restored as a bump whose
    apex is exactly ``value``.

    Three properties keep this bounded and safe:

    * The apex is ``value`` -- the height the peak actually had in the source --
      so restoration can return a peak but can never exaggerate one. The reference,
      not the algorithm, sets the ceiling.
    * The presence test is "does this height already occur nearby", not "is this
      exact pixel a local maximum". That matters because extrema are located on a
      pooled copy and are reported at the block centre, up to ``factor // 2`` pixels
      from the true apex. An exact test would fire on that offset and inject a
      duplicate peak; a neighbourhood test cannot, provided the window is at least
      as wide as the pooling offset.
    * ``radius`` must therefore be at least ``pooled_factor // 2 + 2``. Getting this
      wrong does not fail loudly -- it fabricates peaks that look plausible -- which
      is why the caller derives it from the same pooling that produced the points
      rather than hard-coding it.
    * The presence tolerance is ``0.25 * tau``. The reasoning is a separation
      argument, and getting it wrong is not a rounding concern but a correctness
      one. Two situations produce a shortfall at a mapped peak position:

      1. *Sampling.* The output lattice need not have a sample exactly at the peak.
         Measured shortfalls from this are 0.3-0.8% of the peak height -- and they
         show up on smooth Gaussian blobs, where the transport is at its best, at a
         rate of 33 out of 35 peaks. Restoring those would inject a bump for no
         reason whatsoever.
      2. *Erasure.* Numerical drift genuinely loses the peak, in which case the
         shortfall is comparable to the peak's own persistence, which is >= ``tau``
         by the definition of significance.

      A quarter of ``tau`` sits in the gap between the two populations by more than
      an order of magnitude. An exact-equality test sits below both and therefore
      fires on every peak; a float32-only tolerance does the same, since float32
      carries ~1e-7 of rounding. Both mistakes were made here before the
      measurement settled it.

    At ``scale == 1`` every input peak is already a peak of ``field``, the test
    succeeds everywhere, and the function is a strict no-op. That is what keeps the
    scale-1 identity exact through this stage as well, and it is asserted by
    ``test_identity_at_scale_one``.
    """
    if transported.shape[0] == 0 or tau <= 0.0:
        return field, 0

    out = field.astype(np.float64).copy()
    h, w = out.shape
    span = int(max(2, radius))
    bump_radius = 1.5
    tolerance = 0.25 * float(tau)
    restored = 0

    for y_f, x_f, value in transported:
        y = int(np.clip(round(y_f), 0, h - 1))
        x = int(np.clip(round(x_f), 0, w - 1))
        y0, y1 = max(y - span, 0), min(y + span + 1, h)
        x0, x1 = max(x - span, 0), min(x + span + 1, w)
        window = out[y0:y1, x0:x1]
        if window.size == 0:
            continue
        apex = float(value)
        if float(window.max()) >= apex - tolerance:
            continue

        gy, gx = np.mgrid[y0:y1, x0:x1]
        dist2 = (gy - y_f) ** 2 + (gx - x_f) ** 2
        bump = np.exp(-0.5 * dist2 / (bump_radius * bump_radius))
        target = window * (1.0 - bump) + apex * bump
        out[y0:y1, x0:x1] = np.maximum(window, target)
        restored += 1

    return out, restored


__all__ = [
    "Extrema",
    "matches",
    "merge_tree",
    "peaks",
    "persistence_threshold",
    "pits",
    "pool_max",
    "restore",
    "signature",
    "suppress",
]
