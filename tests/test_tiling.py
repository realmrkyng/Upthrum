import numpy as np
import pytest

from pixelboost.errors import OutOfMemory
from pixelboost.tiling import Tiler, axis_ranges, plan_tiles, suggest_tile


def ramp(h, w, c=3):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = ((x * 7 + y * 13) % 53) / 53.0
    return np.repeat(base[..., None], c, axis=2).astype(np.float32)


class DoubleBackend:
    """Position-independent 2x nearest replicate.

    Because every overlapping tile computes the *same* value for a shared pixel,
    the feather-weighted average must reproduce the untiled result exactly. Any
    disagreement is a bug in the tiling geometry or the weight normalisation.
    """

    name = "double"
    native_scale = 2
    requires_tiling = True

    def process(self, tile, scale=2.0):
        return np.repeat(np.repeat(tile, 2, axis=0), 2, axis=1)


def test_axis_ranges_cover_every_pixel():
    for total in (1, 7, 64, 100, 511, 512, 513, 1000):
        for tile in (0, 16, 64, 128):
            for overlap in (0, 4, 16):
                if tile == 0:
                    assert axis_ranges(total, tile, overlap) == [(0, total)]
                    continue
                ranges = axis_ranges(total, tile, overlap)
                covered = np.zeros(total, bool)
                for a, b in ranges:
                    assert 0 <= a < b <= total
                    assert b - a <= tile
                    covered[a:b] = True
                assert covered.all(), (total, tile, overlap, ranges)


def test_axis_ranges_single_tile_when_big_enough():
    assert axis_ranges(100, 200, 8) == [(0, 100)]


def test_plan_tiles_is_row_major():
    tiles = plan_tiles(100, 100, 64, 8)
    assert len(tiles) == 4
    assert tiles[0][0] == 0 and tiles[0][2] == 0
    assert tiles[1][2] > tiles[0][2]


def test_tiled_matches_whole_image():
    img = ramp(200, 300)
    backend = DoubleBackend()
    whole = backend.process(img, 2.0)

    for tile, overlap in ((64, 8), (128, 16), (37, 5)):
        tiler = Tiler(scale=2, tile=tile, overlap=overlap, pad=0, align=1)
        tiled = tiler.run(img, lambda t: backend.process(t, 2.0))
        assert tiled.shape == whole.shape
        np.testing.assert_allclose(tiled, whole, atol=1e-5)
        assert tiler.stats.count == len(plan_tiles(200, 300, tile, overlap))


def test_no_tiling_path_when_tile_exceeds_image():
    img = ramp(40, 40)
    tiler = Tiler(scale=2, tile=512, overlap=16, pad=0, align=1)
    assert not tiler.should_tile(40, 40)
    out = tiler.run(img, lambda t: t.repeat(2, 0).repeat(2, 1))
    assert out.shape == (80, 80, 3)


def test_pad_is_cropped_back():
    img = ramp(50, 70)
    tiler = Tiler(scale=2, tile=64, overlap=8, pad=16, align=1)
    out = tiler.run(img, lambda t: t.repeat(2, 0).repeat(2, 1))
    assert out.shape == (100, 140, 3)


def test_align_padding_is_cropped():
    img = ramp(30, 30)
    seen = []

    def proc(tile):
        seen.append(tile.shape[:2])
        return tile.repeat(2, 0).repeat(2, 1)

    tiler = Tiler(scale=2, tile=20, overlap=4, pad=0, align=8)
    out = tiler.run(img, proc)
    assert out.shape == (60, 60, 3)
    assert seen
    assert all(h % 8 == 0 and w % 8 == 0 for h, w in seen)
    assert any(h > 20 for h, _ in seen)


def test_weight_never_leaves_holes():
    img = ramp(96, 96)
    tiler = Tiler(scale=1, tile=32, overlap=8, pad=0, align=1)
    out = tiler.run(img, lambda t: t)
    np.testing.assert_allclose(out, img, atol=1e-5)


def test_progress_callback_is_exhaustive():
    calls = []
    tiler = Tiler(scale=2, tile=32, overlap=8, pad=0, align=1)
    tiler.run(ramp(64, 64), lambda t: t.repeat(2, 0).repeat(2, 1), lambda d, n: calls.append((d, n)))
    assert calls
    assert calls[-1][0] == calls[-1][1]


def test_output_budget_guard():
    tiler = Tiler(scale=4, tile=64, overlap=8, max_output_pixels=10_000)
    with pytest.raises(OutOfMemory):
        tiler.run(ramp(200, 200), lambda t: t.repeat(4, 0).repeat(4, 1))


def test_tile_shape_mismatch_is_reported():
    tiler = Tiler(scale=4, tile=32, overlap=4, pad=0, align=1)
    with pytest.raises(OutOfMemory):
        tiler.run(ramp(64, 64), lambda t: t.repeat(2, 0).repeat(2, 1))


def test_suggest_tile_scales_with_budget():
    small = suggest_tile(4000, 4000, 64 << 20, 4)
    large = suggest_tile(4000, 4000, 1024 << 20, 4)
    assert 64 <= small <= large <= 2048
