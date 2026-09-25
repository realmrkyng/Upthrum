# UPTHRUM — derivation and design notes

**U**nified **P**hase-**T**opology **H**ierarchical **R**econstruction **U**nder
the **M**onogenic manifold.

This document is the reasoning, not the usage. For how to run it, see
[README.md § UPTHRUM](../README.md#upthrum); for what each parameter does, see
[docs/TUNING.md](TUNING.md).

---

## 1. Why intensity is the wrong variable

Super-resolution is usually stated as: given samples `I(x)` on a lattice of
spacing `1`, estimate `I(x/s)` on a lattice of spacing `1/s`. Whether the
estimator is Lanczos, a CNN or a diffusion model, the target is the same — a
number per output pixel.

The problem is that the map from a fine continuous image to its coarse samples
is many-to-one. Infinitely many fine images produce the same coarse samples, and
they differ precisely in the high-frequency structure the method is being asked
to recover. Any intensity-domain estimator must therefore pick one preimage, and
the only defensible choices are:

* the smoothest preimage — which is interpolation blur, or
* a plausible preimage sampled from a learned prior — which is hallucination.

Both failures are structural, not deficiencies of a particular network. Adding
parameters moves you between them; it cannot remove the fork.

## 2. Phase is the right variable

There is a domain in which the answer is determined rather than guessed. For a
locally plane wave `I(x) ≈ A cos(k·x + φ₀)`, translating the wave by any `δ`
changes exactly one thing:

```
φ₀  →  φ₀ + k·δ
```

Translation is a phase shift, and `k` — the local wavevector — is measurable
from the coarse samples. So if you represent the image as, per frequency band, a
local amplitude and a local phase, the sub-pixel arrangement of structure is
*determined* by the phase field and its gradient. Nothing has to be invented;
the only estimate is `k`, which is a first-derivative quantity and therefore
stable where the intensity itself is ambiguous.

That is the whole idea. Everything below is the mechanics of doing it without
breaking the properties that make it worth doing.

## 3. The monogenic signal

A 1-D analytic signal gives phase but has no notion of orientation. In 2-D the
monogenic signal (Felsberg & Sommer, 2001) extends this with the two Riesz
components — the 2-D analogue of the Hilbert transform — giving, per band:

```
B   band-pass of the image        (log-Gabor, radial)
Rx  Riesz component along x
Ry  Riesz component along y
```

The Riesz transform is isotropic, so `(B, Rx, Ry)` is a 3-vector that rotates
covariantly with the signal. From it:

```
amplitude     A      = hypot(B, R)
phase         φ      = atan2(R, B)          on the circle S¹
orientation   θ      = ½ · atan2(⟨sin 2θ⟩, ⟨cos 2θ⟩)
```

where `R` is a *directional* Riesz component (see §4 — this is the part that
went wrong first) and `⟨·⟩` is a weighted local mean over bands of the doubled
angle, which is how orientation is averaged without a branch cut at ±π/2.

### 4.1 The folded-phase bug

The natural-looking choice is `φ = atan2(|R|, B)` using the Riesz magnitude.
This is **wrong**, and the wrongness is invisible to casual inspection.

`|R| ≥ 0`, so `atan2(|R|, B) ∈ [0, π]`. But a real wave's phase must traverse a
full `2π` per period. Constrained to `[0, π]`, the phase is folded: at every
zero crossing of the wave the *sign of the phase gradient flips*. A linear
extrapolation `φ + ∇φ·δ` then extrapolates the wave backwards on every second
half-period, and the result is not a shifted wave at all.

Measured on a plane wave at a half-pixel offset, lattice-aligned samples are
exact (the error is at the float32 floor) and the sub-pixel error from folding
was `2.8e-2`. The fix is to use a signed directional component along the local
orientation:

```
Rdir = Rx·cos θ + Ry·sin θ
φ    = atan2(Rdir, B)
```

`Rdir` changes sign exactly when the wave does, so `φ` is now unfolded and its
gradient is constant across a period. Plane-wave error after the fix:
`1.08e-5` — a factor of 2600. `tests/test_upthrum.py::test_transport_is_exact_for_a_plane_wave`
pins it, because this bug produces output that looks *plausible* on real
images. It is the most dangerous class of bug a reconstruction method can have.

## 4. Transport

### 4.1 The core formula

For each output pixel `q` and each band, the phase is transported by a
coherent — not arithmetic — mean over the input taps that surround `q`'s
preimage:

```
            Σ_t  W_t · A_t^γ · exp( i ( φ_t + g · ∇φ_t · Δ_t ) )
φ_out(q) = arg ────────────────────────────────────────────────────
                        Σ_t  W_t · A_t^γ

Δ_t = tap_t − q        g = phase_gain        W_t = anisotropic kernel · cell area
```

Read the exponent. It is *not* `i·φ_t` alone. The term `∇φ_t·Δ_t` is the linear
extrapolation of the phase from the tap to the query point — that is where the
sub-pixel information lives, and it is the entire reason the method works.

### 4.2 The missing-phase bug

The first implementation had `advance = g·∇φ_t·Δ_t` and omitted `φ_t`. The
result was catastrophic and *silent*: for a locally plane wave the argument of
the sum collapses toward zero (the `φ_t` that would make the sum complex is
gone, so the residual sum is real and positive), `arg` returns `0`, and the
output becomes `amplitude · cos(0)` — i.e. **the amplitude envelope**. The
program produced a plausible-looking brightened version of the input, with no
error and no NaN.

It was caught only because the scale-1 identity test failed with error `0.837`:
at unit scale the transport must reproduce the input exactly, and an envelope
does not. The lesson is recorded here because it generalises: **in a method
whose output is an `arg`, the failure mode is a plausible real number, not an
exception.** Every subsequent invariant test was written on the assumption that
silence proves nothing.

With `φ_t` restored, identity error is `5.96e-8`.

### 4.3 Why a coherent mean, and not interpolation

An arithmetic mean of intensities is what interpolation does, and it is exactly
where blur comes from: samples of a wave that are out of phase partially cancel,
so the mean is smaller than the samples. The phasor mean cannot do this, because
each tap contributes a unit-length vector at its own phase; the resultant keeps
the phase information and loses nothing to cancellation. What it *does* lose —
when the phasor cloud genuinely disagrees — is measured by the resultant length

```
κ = |Σ W exp(iφ)| / Σ W    ∈ [0, 1]
```

and that is used as a gate: `amplitude · κ^coherence_power`. Where the phase
field is coherent the reconstruction is full strength; where it is not, the
output is damped toward the low-pass rather than inventing structure.

## 5. The topological constraint

Phase transport is sharp. Sharp is not automatically *right*: a method that
reconstructs texture can also reconstruct texture that was never there, and on
noisy input it will. The constraint is a persistent-homology budget.

### 5.1 What is measured

Build the merge tree of the 0-dimensional sublevel sets of the reconstructed
luma (union-find over pixels sorted by value). Every critical point dies at a
persistence — the height of the saddle that merges it into a more prominent
feature. The signature of an image is the multiset of those persistences.

### 5.2 What is enforced, and why it is one-sided

The tempting invariant — "the output has the same topological signature as the
input" — is false on its face, because resampling changes the pixel count and
the number of critical points above a fixed absolute threshold scales with it.
Measured: a factor of 1.46 at scale 2, for UPTHRUM *and* for plain Lanczos.

The invariant that can actually hold is a comparison at identical resolution:

> the output carries no more topological energy than a plain high-quality
> resample of the same source.

That is apples-to-apples, and it is the direction in which a hallucinating
method violates the contract. Implementation is two operations:

* **`suppress`** — cancel critical points of the output below
  `τ = persistence_relative · (p99 − p1)` by convex-blending toward the smooth
  reference in a stamped neighbourhood. Convexity is the guarantee: the result
  is provably between the candidate and the reference, so suppression cannot
  overshoot.
* **`restore`** — re-add source peaks the transport erased, as bounded bumps
  whose apex is the *source's own* peak height. This path is bounded by the
  source rather than contractive, which is the weaker guarantee — hence the
  deliberately conservative firing tolerance (§5.4).

### 5.3 Calibrating `persistence_relative`

`τ` must sit above the noise floor and below the real structure. For a Gaussian
field `p99 − p1 ≈ 4.6σ`, so `τ ≈ persistence_relative · 4.6σ`; the persistence
at which a peak is indistinguishable from noise is `≈ σ`, i.e. a relative value
of `≈ 0.22`. Measured separation on a synthetic target (79 real features,
additive Gaussian noise):

| value | clean | σ = 0.008 | σ = 0.030 |
|---|---|---|---|
| 0.02 | 79 | 91 | 2575 |
| 0.15 | 79 | 79 | 80 |
| 0.20 | 79 | 79 | 79 |
| 0.25 | 74 | 74 | 74 |
| 0.30 | 14 | 44 | 55 |

Below 0.10 the stage is inert — the default was originally 0.02 and removed
0% of noise features. Above 0.25 it eats real structure; by 0.30 the image is
gutted. The default is 0.18, in the middle of the flat region where separation
is complete and the clean signature is untouched.

### 5.4 Calibrating the restore tolerance

A source peak counts as "erased" when its height is not attained within a
tolerance of its transported position. Two regimes:

* **Sampling shortfall** — the output lattice simply has no sample at the apex.
  Measured 0.3–0.8% of peak height.
* **Genuine erasure** — shortfall of the order of the peak's persistence, i.e.
  at least `τ`.

There is more than an order of magnitude of clear space. The tolerance is
`0.25·τ`. At `1e-9` (a previous value, chosen as "any float error") the stage
fired on 14 smooth peaks that were never erased, injecting bumps and moving the
DC by `1.4e-1`; float32 noise floor is `≈3e-8`, so "any error" was never the
right threshold. At `0.02·τ` it still misfired. `0.25·τ` misfires on 0 of 35
smooth-blob peaks.

## 6. Colour

Luma is reconstructed by the mechanism above. Chroma is *not* transported: it is
carried on the smooth path and given detail only where the local luma–chroma
gradient correlation says the two channels genuinely share structure
(`chroma_detail`, default 0.25). Every chroma edge is therefore derived from a
luma edge, which is why the output does not show the invented colour speckle
that GAN upscalers produce on text.

## 7. Complexity and memory

| Stage | Cost | Scales with |
|---|---|---|
| filter bank + monogenic analysis | a fixed number of FFTs per band | input area × bands |
| orientation, coherence | pointwise + small separable convolutions | input area |
| transport | one gather per output pixel | **output** area |
| topology | union-find, sequential, Python | `min(input area, topology_max_pixels)` |

Measured split at scale 2 on a 512² source: topology 52%, everything else 48%.
At scale 4 the transport's output-area cost grows and the topology share falls.
The topology stage is capped by `topology_max_pixels` (65536) with block-*max*
pooling, which preserves maxima exactly — no significant bright structure is
lost, only sub-block detail the threshold is meant to ignore.

Tiling is deliberately **not** used for the input: the log-Gabor analysis is
non-local (it is defined on the whole 2-D spectrum), so cutting the image into
tiles would put a seam through every band. Output rows are streamed instead.

## 8. What would falsify this

A method is only as good as the tests that could kill it. Each of these is in
`tests/test_upthrum.py`, and each was written to fail on a bug that was actually
made:

* identity at scale 1 → caught the missing phase term and two coordinate bugs;
* exact plane-wave transport → caught the folded phase;
* no attenuation at the band centre → catches any regression to interpolation;
* edge sharpness vs Lanczos → catches any reintroduction of a low-pass;
* topological budget vs Lanczos → catches hallucinated critical points;
* DC preservation → catches drift from the repair stages.

If someone proposes a change, the question is not whether it looks better on a
photo. It is whether these still hold.
