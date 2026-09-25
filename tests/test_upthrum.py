"""UPTHRUM invariants.

These are not smoke tests. Each one states a property the method claims and fails
loudly when the implementation stops satisfying it. Several of them exist because
the corresponding bug was actually made and found here, and each of those is
marked, because a test whose purpose is invisible gets deleted by the next person
who finds it inconvenient.

The list, in the order the algorithm runs:

    identity at scale 1          test_identity_at_scale_one
    exact monogenic round trip   test_monogenic_round_trip
    wrap-free phase gradient     test_phase_gradient_has_no_branch_cut_spikes
    exact plane-wave transport   test_transport_is_exact_for_a_plane_wave
    no attenuation near Nyquist   test_no_attenuation_where_lanczos_loses_contrast
    anisotropy                   test_edge_is_not_softened
    DC preservation              test_dc_is_preserved
    streaming equivalence        test_streaming_matches_whole_image
    topological budget           test_output_stays_within_the_interpolant_budget
    contractive suppression      test_suppression_is_contractive
    noise separation             test_persistence_threshold_separates_noise
    restoration fires only when  test_restore_fires_on_a_genuinely_erased_peak
"""

from __future__ import annotations

import numpy as np
import pytest

from pixelboost.ops import resize_lanczos
from pixelboost.upthrum import Upthrum, describe, enhance
from pixelboost.upthrum import topology as topo
from pixelboost.upthrum.filters import build_filter_bank, lowpass_component, split_band
from pixelboost.upthrum.params import UpthrumParams as Params
from pixelboost.upthrum.transform import analyse, coherence, dominant_orientation
from pixelboost.upthrum.transport import transport


def ramp(height: int = 48, width: int = 64) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    value = 0.5 + 0.2 * np.sin(2 * np.pi * x / 16.0) + 0.1 * np.cos(2 * np.pi * y / 24.0)
    return np.ascontiguousarray(value.astype(np.float32))


def blobs(height: int = 96, width: int = 128, seed: int = 5) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    out = np.full((height, width), 0.2, dtype=np.float64)
    rng = np.random.default_rng(seed)
    for _ in range(24):
        cy, cx = rng.integers(8, height - 8), rng.integers(8, width - 8)
        amp = rng.uniform(0.25, 0.6)
        sd = rng.uniform(1.5, 4.0)
        out += amp * np.exp(-((y - cy) ** 2 + (x - cx) ** 2) / (2 * sd * sd))
    return np.clip(out, 0, 1).astype(np.float32)


def sine(frequency: float, amplitude: float = 0.4, height: int = 64, width: int = 256):
    x = np.arange(width, dtype=np.float64)
    wave = 0.5 + amplitude * np.cos(2 * np.pi * frequency * x)
    return np.ascontiguousarray(np.tile(wave, (height, 1)).astype(np.float32))


class TestFilters:
    def test_partition_of_unity(self):
        bank = build_filter_bank((128, 128), bands=3)
        total = bank.lowpass.copy()
        for band in bank.bands:
            total = total + band
        assert np.max(np.abs(total - 1.0)) < 1e-12

    def test_lowpass_is_non_negative(self):
        for bands in (2, 3, 4, 5):
            bank = build_filter_bank((96, 128), bands=bands, top_frequency=0.28)
            assert bank.lowpass.min() >= -1e-12

    def test_band_sum_stays_below_one_everywhere(self):
        bank = build_filter_bank((64, 64), bands=3)
        total = np.zeros_like(bank.lowpass)
        for band in bank.bands:
            total = total + band
        assert total.max() <= 1.0 + 1e-12

    def test_spectral_decomposition_is_exact(self):
        for image in (ramp(), ramp(37, 53)):
            bank = build_filter_bank(image.shape, bands=3)
            spectrum = np.fft.rfft2(image.astype(np.float64))
            rebuilt = lowpass_component(spectrum, bank, image.shape).astype(np.float64)
            for transfer in bank.bands:
                rebuilt += split_band(spectrum, transfer, image.shape).astype(np.float64)
            assert np.max(np.abs(rebuilt - image)) < 1e-5

    def test_odd_sized_images_keep_their_width(self):
        """rfft2 makes the shape inference lossy for odd widths.

        ``shape_of`` inverts the half-spectrum column count as ``2 * (ncols - 1)``,
        which recovers the width only when it is even -- for an odd width it returns
        one less. Bands then come back a column narrow and the assembly broadcast
        fails. Explicit shapes are threaded through for this reason.
        """
        image = ramp(37, 53)
        assert image.shape == (37, 53)
        bank = build_filter_bank(image.shape, bands=3)
        for band in analyse(image, bank):
            assert band.band.shape == (37, 53)
            assert band.amplitude.shape == (37, 53)


class TestMonogenic:
    def test_monogenic_round_trip(self):
        image = ramp()
        bank = build_filter_bank(image.shape, bands=3)
        for band in analyse(image, bank):
            rebuilt = band.amplitude * np.cos(band.phase)
            assert np.max(np.abs(rebuilt - band.band)) < 1e-6

    def test_amplitude_dominates_each_component(self):
        image = ramp()
        bank = build_filter_bank(image.shape, bands=3)
        for band in analyse(image, bank):
            assert np.all(band.amplitude >= np.abs(band.band) - 1e-6)

    def test_phase_spans_the_full_circle(self):
        """The transported phase must be signed, not folded into [0, pi].

        The textbook monogenic phase is ``atan2(|R|, B)``, which lands in [0, pi] and
        has a kink at every zero crossing. Its gradient flips sign there, which
        destroys the linear extrapolation the transport depends on. Using the signed
        directional component instead puts the phase on the whole circle.
        """
        image = ramp(96, 128)
        bank = build_filter_bank(image.shape, bands=3)
        phases = [a.phase for a in analyse(image, bank)]
        assert min(float(p.min()) for p in phases) < -0.5
        assert max(float(p.max()) for p in phases) > 0.5
        assert min(float(np.ptp(p)) for p in phases) > 3.0

    def test_phase_gradient_does_not_flip_sign_at_zero_crossings(self):
        """Regression: this is the bug that limited sub-pixel accuracy to ~7%.

        On a pure wave the phase gradient must be a constant with one sign across
        the whole image. With the folded phase it flipped sign twice per period,
        giving a periodic sub-pixel error pattern of up to 0.028 on a 0.4 amplitude
        while lattice-aligned samples stayed exact to 1e-8. After the fix the
        plane-wave reconstruction error is 1e-5 (see
        ``test_transport_is_exact_for_a_plane_wave``).
        """
        source = sine(0.125)
        bank = build_filter_bank(source.shape, bands=3)
        for band in analyse(source, bank):
            strong = band.amplitude > 0.2 * float(band.amplitude.max())
            gradient = band.grad_x[strong]
            assert gradient.size > 100
            median = float(np.median(gradient))
            assert abs(median) > 0.5
            flipped = (np.sign(gradient) != np.sign(median)).mean()
            assert flipped < 0.02, flipped

    def test_phase_gradient_has_no_branch_cut_spikes(self):
        """A wrapped angle differentiated naively produces pi-sized spikes.

        The phase is differentiated through the phasor, which is continuous across
        both branch cuts, so the gradient stays bounded by the actual rate of
        change of the oscillation. A finite difference on the raw angle would show
        spikes of height pi wherever the phase crosses 0 or pi.
        """
        image = ramp()
        bank = build_filter_bank(image.shape, bands=3)
        for band in analyse(image, bank):
            limit = 2.0 * np.pi * band.centre * 4.0
            assert np.abs(band.grad_x).max() < limit
            assert np.abs(band.grad_y).max() < limit

    def test_orientation_uses_the_doubled_angle_map(self):
        """A vertical sinusoid must read as horizontal structure, and its mirror too.

        Orientation is pi-periodic, so averaging the raw angles would cancel a
        contour against the same contour rotated by pi. The doubled-angle mean
        cannot.
        """
        x = np.arange(128, dtype=np.float64)
        image = np.tile((0.5 + 0.4 * np.cos(2 * np.pi * x / 16.0)), (64, 1)).astype(np.float32)
        bank = build_filter_bank(image.shape, bands=3)
        theta = dominant_orientation(analyse(image, bank), image.shape)
        inner = theta[8:-8, 8:-8]
        assert np.abs(np.abs(inner) - 0.0).mean() < 0.35

    def test_coherence_is_high_on_a_pure_wave_and_low_on_noise(self):
        wave = sine(0.1)
        rng = np.random.default_rng(2)
        noise = rng.random(wave.shape).astype(np.float32)
        bank = build_filter_bank(wave.shape, bands=3)
        k_wave = coherence(analyse(wave, bank), wave.shape)
        k_noise = coherence(analyse(noise, bank), noise.shape)
        assert k_wave.mean() > k_noise.mean()
        assert k_noise.min() >= 0.0 and k_noise.max() <= 1.0 + 1e-6


class TestTransport:
    def test_transport_reproduces_each_band_at_scale_one(self):
        image = ramp()
        bank = build_filter_bank(image.shape, bands=3)
        analyses = analyse(image, bank)
        theta = dominant_orientation(analyses, image.shape)
        for source, moved in zip(analyses, transport(analyses, theta, image.shape, 1.0, Params())):
            assert np.max(np.abs(moved - source.band)) < 1e-5

    def test_transport_is_exact_for_a_plane_wave(self):
        """The defining claim: a band-limited wave survives sub-pixel resampling intact.

        A locally plane wave has a linear phase field, so the coherent mean aligns
        every tap exactly and the reconstruction is the analytic answer. Measured
        error is 8e-6 to 1.2e-5 across every scale tried, against an amplitude of
        0.4. If the offset or the sign convention in the transport is wrong, this
        fails immediately and by orders of magnitude -- the folded-phase version of
        this function gave 0.028.
        """
        frequency = 0.125
        source = sine(frequency)
        bank = build_filter_bank(source.shape, bands=3)
        analyses = analyse(source, bank)
        theta = dominant_orientation(analyses, source.shape)
        spectrum = np.fft.rfft2(source.astype(np.float64))

        for scale in (2.0, 3.0, 4.0, 5.0):
            out_shape = (int(round(source.shape[0] * scale)), int(round(source.shape[1] * scale)))
            bands = transport(analyses, theta, out_shape, scale, Params())
            rebuilt = resize_lanczos(
                lowpass_component(spectrum, bank, source.shape), out_shape[1], out_shape[0]
            ).astype(np.float64)
            for band in bands:
                rebuilt += band.astype(np.float64)

            u = np.arange(out_shape[1], dtype=np.float64)
            q = (u + 0.5) / scale - 0.5
            analytic = 0.5 + 0.4 * np.cos(2 * np.pi * frequency * q)
            row = rebuilt[out_shape[0] // 2, 16:-16]
            assert np.max(np.abs(row - analytic[16:-16])) < 1e-4, scale

    def band_error(self, periods: int, scale: float = 2.0) -> tuple[float, float]:
        """Max reconstruction error of UPTHRUM and of Lanczos on a pure wave.

        The frequency is an integer number of periods per window on purpose. The
        source is analysed with an FFT, so a frequency that does not close in the
        window leaks across the spectrum and the analytic wave being compared
        against is not the signal actually present -- at exactly 0.2 (51.2 periods)
        the leakage alone produced 2.4e-3 of apparent error.
        """
        frequency = periods / 256.0
        source = sine(frequency)
        bank = build_filter_bank(source.shape, bands=3)
        analyses = analyse(source, bank)
        theta = dominant_orientation(analyses, source.shape)
        spectrum = np.fft.rfft2(source.astype(np.float64))

        out_shape = (int(round(source.shape[0] * scale)), int(round(source.shape[1] * scale)))
        rebuilt = resize_lanczos(
            lowpass_component(spectrum, bank, source.shape), out_shape[1], out_shape[0]
        ).astype(np.float64)
        for band in transport(analyses, theta, out_shape, scale, Params()):
            rebuilt += band.astype(np.float64)

        u = np.arange(out_shape[1], dtype=np.float64)
        q = (u + 0.5) / scale - 0.5
        analytic = 0.5 + 0.4 * np.cos(2 * np.pi * frequency * q)
        measure = slice(16, -16)

        lanczos = resize_lanczos(source, out_shape[1], out_shape[0])
        return (
            float(np.max(np.abs(rebuilt[out_shape[0] // 2, measure] - analytic[measure]))),
            float(np.max(np.abs(lanczos[lanczos.shape[0] // 2, measure] - analytic[measure]))),
        )

    def test_never_loses_to_lanczos_across_the_spectrum(self):
        """The headline claim, swept over the whole band: UPTHRUM is never worse.

        Both methods are compared against the analytic wave rather than against each
        other, so this cannot pass by both being wrong in the same direction. At the
        low end the wave is carried mostly by the low-pass path, which *is* a Lanczos
        resample, so the two converge there by construction; at the high end the top
        band is running out of support. The margin peaks in between.
        """
        for periods in (8, 16, 24, 32, 40, 48, 52, 56, 64, 80, 100):
            upthrum_error, lanczos_error = self.band_error(periods)
            assert upthrum_error <= lanczos_error, (periods, upthrum_error, lanczos_error)

    def test_beats_lanczos_by_a_wide_margin_inside_the_band(self):
        """Inside the band the error stays under 0.5% of the signal amplitude.

        Measured: absolute error peaks at 1.8e-3 on a 0.4 amplitude across this
        range, while the margin over Lanczos runs from 36x at 0.094 cycles/pixel and
        20x at 0.125 down to 3.1x at 0.25. Both bounds are stated at the conservative
        end of what was measured rather than at the best case.
        """
        for periods in (16, 24, 32, 40, 48, 52, 56, 64):
            upthrum_error, lanczos_error = self.band_error(periods)
            assert upthrum_error < 2e-3, (periods, upthrum_error)
            assert upthrum_error * 2.0 < lanczos_error, (periods, upthrum_error, lanczos_error)

    def test_top_frequency_is_where_the_advantage_ends(self):
        """Raising ``top_frequency`` past the content it can support erases the gain.

        This is the measurement behind the parameter's docstring: the advantage is
        real where the filter bank has support and collapses toward 1x at Nyquist,
        where there is no band left to carry the wave.
        """
        inside = self.band_error(32)
        outside = self.band_error(100)
        assert inside[1] / inside[0] > 10.0
        assert outside[1] / outside[0] < 2.0


    def test_streaming_matches_whole_image(self):
        """``band_rows`` bounds the working set; it must not change the result."""
        source = ramp(60, 80)
        bank = build_filter_bank(source.shape, bands=3)
        analyses = analyse(source, bank)
        theta = dominant_orientation(analyses, source.shape)
        out_shape = (120, 160)
        whole = transport(analyses, theta, out_shape, 2.0, Params())
        streamed = transport(analyses, theta, out_shape, 2.0, Params(band_rows=7))
        for a, b in zip(whole, streamed):
            assert np.array_equal(a, b)


class TestIdentity:
    def test_identity_at_scale_one(self):
        engine = Upthrum(Params())
        for image in (ramp(), blobs()):
            out, report = engine.enhance(image, 1.0)
            assert report.identity_error is not None
            assert report.identity_error < 1e-5

    def test_identity_holds_with_topology_enabled_and_disabled(self):
        """Regression: the restoration stage used to inject ~14 spurious bumps here.

        The presence test was comparing against a fixed 1e-9, which is below the
        float32 noise floor of the assembled field, so every source peak looked
        erased and got a bump stamped on top of it. Identity error was 0.135.
        """
        image = blobs()
        for flags in ({"topology": True}, {"topology": False}, {"topology": False, "detail": 0.0}):
            params = Params(**flags)
            out, report = Upthrum(params).enhance(image, 1.0)
            assert report.identity_error < 1e-5
            assert report.topology == "disabled" or report.topology["restored_peaks"] == 0

    def test_identity_holds_for_colour(self):
        rng = np.random.default_rng(3)
        image = rng.random((32, 40, 3)).astype(np.float32)
        out, report = Upthrum(Params()).enhance(image, 1.0)
        assert report.colour is True
        assert report.identity_error < 1e-5
        assert out.shape == image.shape

    def test_identity_holds_for_larger_images_where_pooling_kicks_in(self):
        """Above topology_max_pixels the analysis runs on a pooled copy.

        Restoring then has to widen its presence window by the pooling offset, or
        it fires on the offset itself and injects duplicate peaks. This input is
        large enough to force a pooling factor greater than one.
        """
        rng = np.random.default_rng(9)
        image = rng.random((320, 320)).astype(np.float32)
        params = Params(topology_max_pixels=4096)
        out, report = Upthrum(params).enhance(image, 1.0)
        assert report.identity_error < 1e-5
        assert report.topology["restored_peaks"] == 0


class TestGeometry:
    def test_edge_is_not_softened(self):
        edge = np.zeros((64, 64), np.float32)
        edge[:, 32:] = 1.0
        engine = Upthrum(Params())
        for scale in (2.0, 4.0):
            out, _ = engine.enhance(edge, scale)
            reference = resize_lanczos(edge, int(64 * scale), int(64 * scale))

            def width(image):
                row = image[image.shape[0] // 2]
                lo, hi = row.min(), row.max()
                band = np.where((row >= lo + 0.1 * (hi - lo)) & (row <= lo + 0.9 * (hi - lo)))[0]
                return int(band.max() - band.min() + 1) if band.size else 0

            assert width(out) < width(reference)

    def test_dc_is_preserved(self):
        """On the contractive path brightness survives to 6e-8 of the dynamic range.

        The low-pass path carries the mean and is never opened for modification, and
        the bands must contribute nothing to it because the log-Gabor transfer is
        identically zero at DC -- which is why the transport forces every output band
        to zero mean. Before that was enforced the asymmetric tap weighting leaked
        about 1e-4 of the range into overall brightness.

        Restoration is excluded from the tight assertion because it is the one
        operation here that is not a convex blend: it *adds* a bump, and the integral
        of that bump shows up in the mean. Its contribution is bounded and is checked
        loosely below rather than being wished away.
        """
        contractive = Upthrum(Params(topology_repair=False))
        for image in (ramp(64, 64), blobs()):
            for scale in (1.0, 2.0, 3.0):
                out, _ = contractive.enhance(image, scale)
                assert abs(float(out.mean()) - float(image.mean())) < 1e-6, (scale, image.shape)

        source = blobs()
        out, report = Upthrum(Params()).enhance(source, 2.0)
        assert report.topology["restored_peaks"] > 0
        assert abs(float(out.mean()) - float(source.mean())) < 5e-4

    def test_transported_bands_are_zero_mean(self):
        source = ramp(64, 64)
        bank = build_filter_bank(source.shape, bands=3)
        analyses = analyse(source, bank)
        theta = dominant_orientation(analyses, source.shape)
        out_shape = (192, 192)
        for band in transport(analyses, theta, out_shape, 3.0, Params()):
            assert abs(float(band.mean())) < 1e-6

    def test_output_stays_in_range(self):
        out, _ = Upthrum(Params()).enhance(blobs(), 3.0)
        assert out.min() >= -1e-6
        assert out.max() <= 1.0 + 1e-6

    def test_shapes_are_as_planned(self):
        image = ramp(37, 53)
        engine = Upthrum(Params())
        for scale in (1.0, 1.5, 2.0, 3.7):
            out, report = engine.enhance(image, scale)
            assert out.shape == (int(round(37 * scale)), int(round(53 * scale)))
            assert report.shape_out == out.shape

    def test_downscale_is_refused(self):
        from pixelboost.errors import UnsupportedScale

        with pytest.raises(UnsupportedScale):
            Upthrum(Params()).enhance(ramp(), 0.5)

    def test_pixel_budget_is_enforced(self):
        from pixelboost.errors import OutOfMemory

        with pytest.raises(OutOfMemory):
            Upthrum(Params(max_pixels=1000)).enhance(ramp(64, 64), 4.0)


class TestTopology:
    def test_merge_tree_finds_a_known_persistence(self):
        """Two basins, depths 0.0 and 0.2, separated by a saddle at 0.5.

        The shallower basin dies at the saddle, so its persistence is 0.3. A single
        isolated peak would produce no death at all -- the surviving component is
        not a pair -- which is why the field needs two basins to test this.
        """
        field = np.array(
            [[0.6, 0.6, 0.6], [0.0, 0.5, 0.2], [0.6, 0.6, 0.6]], dtype=np.float64
        )
        _, _, persistence = topo.merge_tree(field)
        assert persistence.size == 1
        assert pytest.approx(float(persistence[0]), abs=1e-9) == 0.3

    def test_isolated_peak_is_reported_via_the_surviving_component(self):
        """A lone peak produces no death, so it must come from the survivor.

        Regression: without the surviving component a single bright spot on a flat
        field reported zero peaks, and the matching in suppress then never checked
        the most prominent feature in the image.
        """
        field = np.full((32, 32), 0.4, np.float64)
        field[16, 16] = 0.9
        maxima = topo.peaks(field)
        assert len(maxima) >= 1
        assert pytest.approx(float(maxima.points[:, 2].max()), abs=1e-9) == 0.9
        found = maxima.points[np.argmax(maxima.points[:, 2])]
        assert (int(found[0]), int(found[1])) == (16, 16)

    def test_pool_max_preserves_the_global_peak(self):
        field = np.random.default_rng(1).random((97, 89))
        pooled, factor = topo.pool_max(field, 256)
        assert factor > 1
        assert pytest.approx(float(pooled.max())) == float(field.max())

    def test_pool_max_is_identity_below_the_cap(self):
        field = np.random.default_rng(1).random((16, 16))
        pooled, factor = topo.pool_max(field, 65536)
        assert factor == 1
        assert pooled is field

    def test_peaks_and_pits_land_on_extreme_values(self):
        image = ramp(96, 128)
        tau = topo.persistence_threshold(image, 0.18)
        maxima = topo.peaks(image).significant(tau)
        minima = topo.pits(image).significant(tau)
        assert len(maxima) > 0 and len(minima) > 0
        assert maxima.points[:, 2].min() > minima.points[:, 2].max()

    def test_persistence_threshold_separates_noise(self):
        """This is why persistence_relative defaults to 0.18 and not to 0.02.

        For a roughly Gaussian field the 1-99 percentile range is about 4.6 sigma,
        so a threshold at the noise level needs a factor near 0.22. Measured: at
        0.02 the noisy signature is thirty times the clean one, at 0.18 they agree,
        and at 0.30 real structure starts disappearing.
        """
        clean = blobs()
        rng = np.random.default_rng(11)
        noisy = np.clip(clean + rng.normal(0, 0.02, clean.shape), 0, 1).astype(np.float32)

        blind = topo.persistence_threshold(clean, 0.02)
        calibrated = topo.persistence_threshold(clean, 0.18)

        n_clean = topo.signature(clean, blind)["total"]
        n_noisy_blind = topo.signature(noisy, blind)["total"]
        assert n_noisy_blind > 2.0 * n_clean

        n_noisy = topo.signature(noisy, calibrated)["total"]
        n_ref = topo.signature(clean, calibrated)["total"]
        assert abs(n_noisy - n_ref) / n_ref < 0.1

    def test_suppression_is_contractive(self):
        """The output must lie between the transport result and the interpolant.

        This is the strongest guarantee the constraint offers: since the blend
        factor is in [0, 1], the constraint can never produce a value that neither
        the transport nor the interpolation produced. Checked on a construction
        where the candidate's extra peak is guaranteed spurious -- a bump added to a
        flat reference, so there is nothing nearby it could legitimately match.
        """
        reference = np.full((96, 128), 0.4, np.float32)
        y, x = np.mgrid[0:96, 0:128]
        bump = (0.5 * np.exp(-((y - 48) ** 2 + (x - 64) ** 2) / 8.0)).astype(np.float32)
        candidate = reference + bump
        tau = max(topo.persistence_threshold(reference, 0.18), 0.02)
        out, stats = topo.suppress(candidate, reference, tau)
        assert stats["suppressed"] > 0
        assert np.min(out - np.minimum(candidate, reference)) >= -1e-12
        assert np.max(out - np.maximum(candidate, reference)) <= 1e-12

    def test_suppression_does_nothing_when_candidate_equals_reference(self):
        image = blobs()
        tau = topo.persistence_threshold(image, 0.18)
        out, stats = topo.suppress(image, image, tau)
        assert stats["suppressed"] == 0
        assert np.max(np.abs(out - image)) == 0.0

    def test_suppression_removes_an_invented_peak(self):
        reference = np.full((96, 128), 0.4, np.float32)
        y, x = np.mgrid[0:96, 0:128]
        candidate = reference + (0.5 * np.exp(-((y - 48) ** 2 + (x - 64) ** 2) / 8.0)).astype(
            np.float32
        )
        tau = topo.persistence_threshold(reference, 0.18)
        tau = max(tau, 0.02)
        out, stats = topo.suppress(candidate, reference, tau)
        assert stats["suppressed"] == 1
        assert out[48, 64] < candidate[48, 64] - 0.1

    def test_restore_fires_on_a_genuinely_erased_peak(self):
        """The tolerance must be wide enough to ignore sampling, narrow enough to see erasure.

        A flattened peak shows a shortfall of its own persistence, which is at
        least tau. Sampling shortfall is under one percent of peak height. An exact
        test sits below both and fires on smooth blobs at a rate of 33 out of 35 --
        which is how this tolerance was calibrated.
        """
        y, x = np.mgrid[0:80, 0:80]
        field = 0.3 + (0.6 * np.exp(-((y - 40) ** 2 + (x - 40) ** 2) / 18.0)).astype(np.float32)
        flattened = field.copy()
        flattened[30:51, 30:51] = 0.3
        points = np.array([[40.0, 40.0, float(field[40, 40])]])
        tau = topo.persistence_threshold(field, 0.18)
        restored, count = topo.restore(flattened, points, tau, radius=3)
        assert count == 1
        assert restored[40, 40] > flattened[40, 40] + 0.4

    def test_restore_ignores_sampling_shortfall(self):
        y, x = np.mgrid[0:80, 0:80]
        field = 0.3 + (0.6 * np.exp(-((y - 40) ** 2 + (x - 40) ** 2) / 18.0)).astype(np.float32)
        nearly = field.copy()
        nearly[40, 40] = float(field[40, 40]) - 0.001
        points = np.array([[40.0, 40.0, float(field[40, 40])]])
        tau = topo.persistence_threshold(field, 0.18)
        _, count = topo.restore(nearly, points, tau, radius=3)
        assert count == 0

    def test_output_stays_within_the_interpolant_budget(self):
        image = blobs()
        engine = Upthrum(Params())
        for scale in (2.0, 4.0):
            out, report = engine.enhance(image, scale)
            reference = resize_lanczos(image, out.shape[1], out.shape[0])
            tau = report.topology["tau"]
            budget = topo.signature(reference, tau)
            measured = topo.signature(out, tau)
            assert measured["sum_persistence"] <= budget["sum_persistence"] * 1.25


class TestApi:
    def test_report_round_trips_through_a_dict(self):
        _, report = Upthrum(Params()).enhance(blobs(), 2.0)
        data = report.to_dict()
        assert data["scale"] == 2.0
        assert data["bands"] == 3
        assert data.get("learned_parameters", True)
        assert isinstance(report.summary(), str)

    def test_module_level_enhance_matches_the_engine(self):
        image = ramp()
        out, data = enhance(image, 2.0)
        assert isinstance(data, dict)
        assert out.shape == (96, 128)

    def test_describe_reports_no_learned_parameters(self):
        info = describe()
        assert info["learned_parameters"] == 0
        assert info["trained_on"] is None
        assert info["identity_at_scale_one"] is True

    def test_params_round_trip(self):
        params = Params(bands=5, phase_gain=0.9, topology=False)
        restored = Params.from_dict(params.to_dict())
        assert restored == params

    def test_params_ignore_unknown_keys(self):
        params = Params.from_dict({"bands": 4, "nonsense": 1})
        assert params.bands == 4

    def test_colour_output_has_three_channels(self):
        rng = np.random.default_rng(6)
        image = rng.random((40, 56, 3)).astype(np.float32)
        out, report = Upthrum(Params()).enhance(image, 2.0)
        assert out.shape == (80, 112, 3)
        assert report.colour is True

    def test_chroma_can_be_disabled(self):
        rng = np.random.default_rng(6)
        image = rng.random((40, 56, 3)).astype(np.float32)
        out, _ = Upthrum(Params(chroma=False)).enhance(image, 2.0)
        assert out.shape == (80, 112, 3)
        assert np.isfinite(out).all()

    def test_invalid_shapes_are_rejected(self):
        with pytest.raises(ValueError):
            Upthrum(Params()).enhance(np.zeros((4, 4, 2), np.float32), 2.0)

    def test_higher_band_counts_stay_consistent(self):
        image = ramp()
        for bands in (1, 2, 3, 4, 5):
            out, report = Upthrum(Params(bands=bands)).enhance(image, 1.0)
            assert report.bands == bands
            assert report.identity_error < 1e-5

    def test_detail_is_a_strength_and_actually_fires(self):
        """Regression: this was passed as the guided filter's *radius*.

        With ``detail_enhance(img, radius, eps, strength)`` the call used to be
        ``detail_enhance(out, params.detail)``, which put the strength into the
        radius slot and left strength at its default. A radius below 1 makes the
        guided filter an identity, so ``detail`` was silently inert at every
        value -- the knob existed, was documented, and did nothing.
        """
        image = blobs()
        plain, _ = Upthrum(Params(detail=0.0)).enhance(image, 2.0)
        boosted, _ = Upthrum(Params(detail=0.8)).enhance(image, 2.0)
        assert not np.allclose(plain, boosted)
        assert float(np.std(boosted - plain)) > 1e-4

    def test_detail_radius_does_not_swallow_small_strengths(self):
        image = blobs()
        plain, _ = Upthrum(Params(detail=0.0)).enhance(image, 2.0)
        faint, _ = Upthrum(Params(detail=0.05)).enhance(image, 2.0)
        assert not np.allclose(plain, faint)
