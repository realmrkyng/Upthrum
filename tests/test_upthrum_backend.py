"""UPTHRUM as a PixelBoost backend.

The algorithm's own invariants live in ``test_upthrum.py``. This file is about
the wiring: that the registry offers the backend, that the options layer passes
parameters through without leaking them, that a model-free backend does not
claim to have loaded a model, and -- the one that is easy to get wrong -- that
the pipeline does not apply its own detail pass on top of a backend that
already applies one.

That last test uses a stub backend rather than UPTHRUM on purpose. The bug it
guards against is a property of the pipeline's contract, not of UPTHRUM, and a
stub makes the assertion exact instead of "the two images differ slightly".
"""

from __future__ import annotations

import numpy as np
import pytest

from pixelboost.backends.base import Backend
from pixelboost.backends.classical import ClassicalBackend, NearestBackend
from pixelboost.backends.registry import (
    BACKEND_NAMES,
    backend_capabilities,
    create_backend,
    resolve_backend_name,
    upthrum_available,
)
from pixelboost.backends.upthrum_backend import UpthrumBackend
from pixelboost.cli import _build_options, build_parser
from pixelboost.config import Config
from pixelboost.engine import Engine
from pixelboost.errors import BackendUnavailable, UnsupportedScale
from pixelboost.pipeline import Pipeline
from pixelboost.types import EnhanceOptions


def photo(height: int = 32, width: int = 40, seed: int = 3) -> np.ndarray:
    y, x = np.mgrid[0:height, 0:width]
    r = 0.5 + 0.25 * np.sin(2 * np.pi * x / 11.0)
    g = 0.5 + 0.25 * np.cos(2 * np.pi * y / 13.0)
    b = 0.4 + 0.2 * np.sin(2 * np.pi * (x + y) / 17.0)
    return np.stack([r, g, b], -1).astype(np.float32)


class _Stub(Backend):
    """A backend that returns a fixed high-frequency pattern and no detail pass."""

    name = "stub"
    handles_detail = False

    def process(self, tile: np.ndarray, scale: float) -> np.ndarray:
        h, w = tile.shape[0], tile.shape[1]
        oh, ow = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
        y, x = np.mgrid[0:oh, 0:ow]
        return np.stack(
            [
                (0.5 + 0.35 * np.sin(x / 4.0)),
                (0.5 + 0.35 * np.cos(y / 6.0)),
                np.full((oh, ow), 0.5),
            ],
            -1,
        ).astype(np.float32)


class _StubSelfDetail(_Stub):
    handles_detail = True


def _run(cls, detail: float) -> np.ndarray:
    opts = EnhanceOptions(scale=2.0, detail=detail)
    out, _alpha = Pipeline(cls(), opts, 32, 32).run(np.full((32, 32, 3), 0.5, np.float32))
    return out


class TestRegistry:
    def test_upthrum_is_a_known_backend(self):
        assert "upthrum" in BACKEND_NAMES

    def test_explicit_request_resolves(self):
        name = resolve_backend_name(Config(), EnhanceOptions(backend="upthrum"))
        assert name == "upthrum"

    def test_auto_never_selects_upthrum(self):
        """`auto` is about robustness; UPTHRUM is an editorial choice."""
        assert resolve_backend_name(Config(), EnhanceOptions(backend="auto")) != "upthrum"
        assert resolve_backend_name(Config(), EnhanceOptions(backend=None)) != "upthrum"

    def test_unknown_backend_still_raises(self):
        with pytest.raises(BackendUnavailable):
            resolve_backend_name(Config(), EnhanceOptions(backend="upthrum2"))

    def test_capabilities_offer_upthrum_and_describe_it(self):
        caps = backend_capabilities(Config())
        assert "upthrum" in caps["backends"]
        assert "upthrum" in [m["backend"] for m in caps["methods"]]
        entry = next(m for m in caps["methods"] if m["backend"] == "upthrum")
        assert entry["family"] == "phase reconstruction"
        assert entry["learned_parameters"] == 0
        assert entry["needs_model_file"] is False

    def test_upthrum_is_always_available(self):
        """numpy is a hard dependency, so this must hold on a bare server."""
        assert upthrum_available() is True


class TestBackend:
    def test_process_returns_the_scaled_shape(self):
        out = UpthrumBackend().process(photo(), 2.0)
        assert out.shape == (64, 80, 3)
        assert out.dtype == np.float32
        assert float(out.min()) >= 0.0 and float(out.max()) <= 1.0

    def test_identity_at_scale_one(self):
        image = photo()
        out = UpthrumBackend().process(image, 1.0)
        assert float(np.max(np.abs(out - image))) < 1e-5

    def test_fractional_scale_is_supported(self):
        """native_scale is 1, so --width/--longest-side need no extra resample."""
        out = UpthrumBackend().process(photo(height=16, width=16), 2.5)
        assert out.shape == (40, 40, 3)

    def test_downscale_is_refused(self):
        with pytest.raises(UnsupportedScale):
            UpthrumBackend().process(photo(), 0.5)

    def test_no_model_no_tiling_and_a_detail_pass_of_its_own(self):
        backend = UpthrumBackend()
        assert backend.model is None
        assert backend.native_scale == 1
        assert backend.requires_tiling is False
        assert backend.handles_detail is True

    def test_info_carries_the_params_and_the_method(self):
        data = UpthrumBackend().info()
        assert data["method"]["name"] == "UPTHRUM"
        assert data["params"]["bands"] == 3
        assert data["requires_tiling"] is False

    def test_warmup_runs_and_stays_consistent(self):
        backend = UpthrumBackend()
        backend.warmup(tile=16)
        assert backend.last_report["scale"] == 1.0
        assert backend.last_report["identity_error"] < 1e-5

    def test_report_is_recorded_after_a_call(self):
        backend = UpthrumBackend()
        assert backend.last_report is None
        backend.process(photo(), 2.0)
        assert backend.last_report["scale"] == 2.0
        assert backend.last_report["shape_out"] == [64, 80]
        assert backend.last_report["bands"] == 3
        assert backend.info()["last_report"]["scale"] == 2.0

    def test_close_is_a_no_op_that_still_works(self):
        backend = UpthrumBackend()
        backend.close()
        assert backend.process(photo(), 2.0).shape == (64, 80, 3)


class TestOptionsPlumbing:
    def test_overrides_reach_the_params(self):
        opts = EnhanceOptions(
            backend="upthrum",
            extra={"upthrum": {"bands": 5, "topology": False, "persistence_relative": 0.2}},
        )
        backend = create_backend(Config(), opts)
        assert isinstance(backend, UpthrumBackend)
        assert backend.params.bands == 5
        assert backend.params.topology is False
        assert backend.params.persistence_relative == 0.2

    def test_device_is_not_swallowed_by_the_params(self):
        opts = EnhanceOptions(backend="upthrum", extra={"upthrum": {"device": "cpu"}})
        backend = create_backend(Config(), opts)
        assert backend.device == "cpu"
        assert backend.device_request == "cpu"
        assert "device" not in backend.params.to_dict()

    def test_unknown_parameter_names_are_ignored(self):
        opts = EnhanceOptions(backend="upthrum", extra={"upthrum": {"nonsense": 1, "bands": 2}})
        backend = create_backend(Config(), opts)
        assert backend.params.bands == 2

    def test_provider_selects_the_device_when_no_explicit_one_is_given(self):
        opts = EnhanceOptions(backend="upthrum", provider="cpu")
        assert create_backend(Config(), opts).device == "cpu"

    def test_auto_is_not_affected_by_the_upthrum_block(self):
        opts = EnhanceOptions(backend="auto", extra={"upthrum": {"bands": 5}})
        assert resolve_backend_name(Config(), opts) != "upthrum"


class TestPipelineContract:
    def test_handles_detail_suppresses_the_pipeline_pass(self):
        baseline = _run(_Stub, detail=0.0)
        boosted = _run(_Stub, detail=0.6)
        suppressed = _run(_StubSelfDetail, detail=0.6)
        assert not np.allclose(baseline, boosted), "the pipeline pass is inert; test is vacuous"
        np.testing.assert_allclose(suppressed, baseline, atol=1e-6)

    def test_classical_still_suppresses_the_pipeline_pass(self):
        """The flag replaced a `name != 'classical'` check; behaviour must match."""
        assert ClassicalBackend.handles_detail is True
        assert NearestBackend.handles_detail is False

    def test_upthrum_runs_end_to_end_through_the_pipeline(self):
        opts = EnhanceOptions(scale=2.0, backend="upthrum", detail=0.35)
        pipeline = Pipeline(UpthrumBackend(), opts, 40, 32)
        out, _alpha = pipeline.run(photo())
        assert out.shape == (64, 80, 3)
        assert pipeline.tile_count == 1


class TestEngine:
    def test_backend_for_returns_an_upthrum_backend(self):
        engine = Engine(Config(), warmup=False)
        try:
            backend = engine.backend_for(EnhanceOptions(backend="upthrum"))
            assert isinstance(backend, UpthrumBackend)
        finally:
            engine.close()

    def test_engine_caches_by_parameters_not_just_by_name(self):
        engine = Engine(Config(), warmup=False)
        try:
            first = engine.backend_for(EnhanceOptions(backend="upthrum"))
            same = engine.backend_for(EnhanceOptions(backend="upthrum"))
            other = engine.backend_for(
                EnhanceOptions(backend="upthrum", extra={"upthrum": {"bands": 5}})
            )
            assert first is same
            assert other is not first
            assert other.params.bands == 5
        finally:
            engine.close()

    def test_enhance_reports_no_model_for_a_model_free_backend(self):
        engine = Engine(Config(), warmup=False)
        try:
            result = engine.enhance(photo(), EnhanceOptions(scale=2.0, backend="upthrum"))
            assert result.backend == "upthrum"
            assert result.model is None
            assert result.provider == "cpu"
            assert result.dst_size == (80, 64)
        finally:
            engine.close()


class TestCli:
    def test_flags_land_in_the_extra_block(self):
        parser = build_parser()
        ns = parser.parse_args(
            [
                "upscale",
                "in.png",
                "-o",
                "out.png",
                "--backend",
                "upthrum",
                "--upthrum-bands",
                "4",
                "--upthrum-persistence",
                "0.2",
                "--no-upthrum-topology",
            ]
        )
        cfg = Config()
        opts = _build_options(cfg, ns)
        assert opts.backend == "upthrum"
        assert opts.extra["upthrum"]["bands"] == 4
        assert opts.extra["upthrum"]["persistence_relative"] == 0.2
        assert opts.extra["upthrum"]["topology"] is False
        # `store_false` flags are SUPPRESSed like every other flag, so an
        # unpassed one must be absent rather than re-stated as True: absence is
        # what lets UpthrumParams supply its own default.
        assert "chroma" not in opts.extra["upthrum"]

    def test_defaults_are_not_mutated_by_the_cli(self):
        """`dataclasses.replace` copies the reference, so the dict is shared."""
        parser = build_parser()
        ns = parser.parse_args(
            ["upscale", "in.png", "-o", "out.png", "--upthrum-bands", "4"]
        )
        cfg = Config()
        _build_options(cfg, ns)
        assert cfg.defaults.extra == {}

    def test_omitted_flags_put_nothing_in_extra(self):
        parser = build_parser()
        ns = parser.parse_args(["upscale", "in.png", "-o", "out.png", "--scale", "2"])
        opts = _build_options(Config(), ns)
        assert "upthrum" not in (opts.extra or {})

    def test_the_backend_choice_matches_the_registry(self):
        parser = build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["upscale", "in.png", "-o", "out.png", "--backend", "upthrum2"])
        for name in BACKEND_NAMES:
            ns = parser.parse_args(["upscale", "in.png", "-o", "out.png", "--backend", name])
            assert ns.backend == name
