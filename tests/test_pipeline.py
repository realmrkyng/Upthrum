import numpy as np
import pytest

from pixelboost import imageio
from pixelboost.backends.classical import ClassicalBackend, NearestBackend
from pixelboost.config import Config
from pixelboost.engine import Engine
from pixelboost.errors import ImageReadError, OutOfMemory
from pixelboost.pipeline import Pipeline
from pixelboost.types import EnhanceOptions, resolve_target


def photo(h=48, w=64):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    r = x / max(1, w - 1)
    g = y / max(1, h - 1)
    b = 0.5 + 0.25 * np.sin(x / 5.0)
    return np.clip(np.stack([r, g, b], -1), 0, 1).astype(np.float32)


@pytest.mark.parametrize(
    "kwargs,expect",
    [
        ({"scale": 4}, (256, 192)),
        ({"scale": 2}, (128, 96)),
        ({"scale": 0.5}, (32, 24)),
        ({"width": 128}, (128, 96)),
        ({"height": 96}, (128, 96)),
        ({"longest_side": 128}, (128, 96)),
        ({"width": 100, "height": 100}, (100, 100)),
    ],
)
def test_resolve_target(kwargs, expect):
    assert resolve_target(64, 48, **kwargs) == expect


def test_resolve_target_default_scale():
    assert resolve_target(64, 48) == (256, 192)


def test_resolve_target_no_keep_aspect_leaves_other_axis_alone():
    assert resolve_target(64, 48, width=128, keep_aspect=False) == (128, 48)
    assert resolve_target(64, 48, height=96, keep_aspect=False) == (64, 96)
    assert resolve_target(64, 48, width=128, height=96, keep_aspect=False) == (128, 96)


def test_pipeline_classical_4x():
    backend = ClassicalBackend()
    opts = EnhanceOptions(scale=4.0)
    out, alpha = Pipeline(backend, opts, 64, 48).run(photo())
    assert out.shape == (192, 256, 3)
    assert alpha is None
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_pipeline_non_uniform_target():
    backend = ClassicalBackend()
    opts = EnhanceOptions(scale=None, width=200, height=100)
    out, _ = Pipeline(backend, opts, 64, 48).run(photo())
    assert out.shape == (100, 200, 3)


def test_pipeline_alpha_is_scaled_too():
    backend = ClassicalBackend()
    opts = EnhanceOptions(scale=2.0)
    rgb = photo(20, 20)
    alpha = np.linspace(0, 1, 20 * 20, dtype=np.float32).reshape(20, 20)
    out, out_alpha = Pipeline(backend, opts, 20, 20).run(rgb, alpha)
    assert out.shape == (40, 40, 3)
    assert out_alpha.shape == (40, 40)
    assert out_alpha.min() >= 0.0 and out_alpha.max() <= 1.0


class QuadBackend:
    """Fixed-4x tiling backend used to exercise the tiled code paths."""

    name = "quad"
    native_scale = 4
    requires_tiling = True
    device = "cpu"
    model = "fake"
    provider = "cpu"

    def process(self, tile, scale=4.0):
        return np.repeat(np.repeat(tile, 4, axis=0), 4, axis=1)


def test_pipeline_pre_downscale_is_recorded():
    opts = EnhanceOptions(scale=2.0, pre_downscale=True, tile=32, tile_overlap=8, tile_pad=0)
    pipeline = Pipeline(QuadBackend(), opts, 128, 128)
    out, _ = pipeline.run(photo(64, 64))
    assert out.shape == (256, 256, 3)
    assert any("pre-downscale" in n for n in pipeline.notes)


def test_pipeline_tiled_reports_tile_count():
    opts = EnhanceOptions(scale=4.0, tile=32, tile_overlap=8, tile_pad=0, pre_downscale=False)
    pipeline = Pipeline(QuadBackend(), opts, 64, 64)
    out, _ = pipeline.run(photo(64, 64))
    assert out.shape == (256, 256, 3)
    assert pipeline.tile_count > 1


def test_pipeline_clamps_to_max_pixels():
    backend = ClassicalBackend()
    opts = EnhanceOptions(scale=4.0, max_pixels=100_000)
    pipeline = Pipeline(backend, opts, 400, 400)
    assert pipeline.target_w * pipeline.target_h <= 100_000
    assert pipeline.notes


def test_pipeline_memory_guard():
    backend = ClassicalBackend()
    pipeline = Pipeline(backend, EnhanceOptions(scale=4.0), 1000, 1000)
    with pytest.raises(OutOfMemory):
        pipeline.guard_memory(1_000_000)


def test_engine_enhance_array_roundtrip():
    cfg = Config()
    cfg.models_dir = "."
    with Engine(cfg, warmup=False) as engine:
        opts = EnhanceOptions(scale=2.0, backend="classical", detail=0.3)
        result = engine.enhance(photo(32, 32), opts)
    assert result.backend == "classical"
    assert result.dst_size == (64, 64)
    assert result.src_size == (32, 32)
    assert result.scale_factor == pytest.approx(2.0, abs=0.01)
    assert result.elapsed_ms > 0
    assert result.summary()["tiles"] == 1


def test_engine_gray_input_stays_gray(tmp_path):
    from PIL import Image

    path = tmp_path / "gray.png"
    Image.fromarray((np.arange(32 * 32).reshape(32, 32) % 255).astype(np.uint8), "L").save(path)

    cfg = Config()
    with Engine(cfg, warmup=False) as engine:
        opts = EnhanceOptions(scale=2.0, backend="classical")
        _result, written = engine.enhance_file(str(path), str(tmp_path / "out.png"), opts)

    with Image.open(written) as im:
        assert im.mode == "L"
        assert im.size == (64, 64)


def test_engine_preserves_alpha(tmp_path):
    from PIL import Image

    path = tmp_path / "rgba.png"
    arr = np.zeros((16, 16, 4), np.uint8)
    arr[..., :3] = 120
    arr[..., 3] = np.arange(16)[None, :] * 16
    Image.fromarray(arr, "RGBA").save(path)

    cfg = Config()
    with Engine(cfg, warmup=False) as engine:
        opts = EnhanceOptions(scale=2.0, backend="classical")
        _result, written = engine.enhance_file(str(path), str(tmp_path / "out.png"), opts)

    with Image.open(written) as im:
        assert im.mode == "RGBA"
        assert im.size == (32, 32)


def test_engine_reuses_backend_instance():
    cfg = Config()
    with Engine(cfg, warmup=False) as engine:
        opts = EnhanceOptions(scale=2.0, backend="classical")
        first = engine.backend_for(opts)
        second = engine.backend_for(opts)
    assert first is second


def test_engine_batch_reports(tmp_path):
    from PIL import Image

    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        Image.fromarray((np.random.rand(24, 24, 3) * 255).astype(np.uint8)).save(src / f"{i}.png")

    cfg = Config()
    out = tmp_path / "out"
    with Engine(cfg, warmup=False) as engine:
        opts = EnhanceOptions(scale=2.0, backend="classical")
        reports = engine.batch([str(src / f"{i}.png") for i in range(3)], str(out), opts)
    assert all(r["status"] == "ok" for r in reports)
    assert len(list(out.glob("*.png"))) == 3


def test_nearest_backend_blocks():
    img = np.zeros((4, 4, 3), np.float32)
    img[1, 1] = 1.0
    out = NearestBackend().process(img, 2.0)
    assert out.shape == (8, 8, 3)
    np.testing.assert_allclose(out[2:4, 2:4], 1.0)


def test_imageio_roundtrip_png(tmp_path):
    rgb = photo(20, 20)
    path = tmp_path / "a.png"
    imageio.save(str(path), rgb, fmt="PNG")
    loaded = imageio.load(str(path))
    assert loaded.rgb.shape == (20, 20, 3)
    np.testing.assert_allclose(loaded.rgb, rgb, atol=2e-2)


def test_imageio_encode_jpeg_bytes():
    blob = imageio.encode(photo(16, 16), fmt="JPEG", quality=90)
    assert blob[:2] == b"\xff\xd8"
    loaded = imageio.load(blob)
    assert loaded.meta.format == "JPEG"


def test_imageio_probe():
    blob = imageio.encode(photo(16, 24), fmt="PNG")
    info = imageio.probe(blob)
    assert info["width"] == 24 and info["height"] == 16
    assert info["format"] == "PNG"
    assert info["animated"] is False


def test_imageio_rejects_garbage():
    with pytest.raises(ImageReadError):
        imageio.load(b"not an image at all")


def test_alpha_plane_roundtrip(tmp_path):
    rgb = photo(16, 16)
    alpha = np.linspace(0, 1, 256, dtype=np.float32).reshape(16, 16)
    path = tmp_path / "a.png"
    imageio.save(str(path), rgb, alpha, fmt="PNG")
    loaded = imageio.load(str(path))
    assert loaded.alpha is not None
    assert loaded.meta.has_alpha
    np.testing.assert_allclose(loaded.alpha, alpha, atol=3e-2)
