import numpy as np
import pytest

from pixelboost import ops


def ramp(h=32, w=48, c=3):
    y, x = np.mgrid[0:h, 0:w].astype(np.float32)
    base = (x / max(1, w - 1) + y / max(1, h - 1)) / 2.0
    return np.repeat(base[..., None], c, axis=2).astype(np.float32)


def test_lanczos_weights_normalised():
    idx, w = ops.lanczos_taps(37, 91)
    assert idx.shape == w.shape
    np.testing.assert_allclose(w.sum(axis=1), 1.0, atol=1e-5)
    assert (idx >= 0).all() and (idx < 37).all()


def test_lanczos_weights_downscale_support_widens():
    _, up = ops.lanczos_taps(64, 256)
    _, down = ops.lanczos_taps(256, 64)
    assert down.shape[1] > up.shape[1]


def test_resize_preserves_constant_image():
    flat = np.full((20, 30, 3), 0.42, np.float32)
    out = ops.resize_lanczos(flat, 61, 41)
    assert out.shape == (41, 61, 3)
    np.testing.assert_allclose(out, 0.42, atol=2e-3)


def test_resize_identity_short_circuits():
    img = ramp()
    out = ops.resize_lanczos(img, img.shape[1], img.shape[0])
    assert out is not None
    np.testing.assert_array_equal(out, img)


def test_resize_multi_splits_large_ratio():
    img = ramp(16, 16)
    out = ops.resize_multi(img, 256, 256, max_step_ratio=2.0)
    assert out.shape == (256, 256, 3)
    assert out.min() > -0.02 and out.max() < 1.02


def test_lanczos_ringing_is_bounded():
    """Lanczos has negative lobes, so a hard step edge overshoots.

    Measured overshoot on a synthetic 0 -> 1 step is ~11 %. That is inherent to
    the kernel, not a defect: a real photograph almost never contains a
    one-pixel step, and the pipeline clamps to [0, 1] at the very end.
    """
    img = np.zeros((32, 32, 3), np.float32)
    img[:, 16:] = 1.0
    out = ops.resize_multi(img, 64, 64)
    assert out.max() < 1.13
    assert out.min() > -0.13
    assert not np.isnan(out).any()


def test_chaining_steps_does_not_worsen_ringing():
    """Multi-step resampling must not compound the overshoot.

    Each 2x step sees a smoother input than the previous one, so the ringing
    stays flat instead of accumulating. If this ever regresses, the step planner
    is wrong.
    """
    img = np.zeros((32, 32, 3), np.float32)
    img[:, 16:] = 1.0
    chained = ops.resize_multi(img, 128, 128, max_step_ratio=2.0)
    single = ops.resize_lanczos(img, 128, 128)
    assert chained.max() <= single.max() + 1e-3
    assert chained.min() >= single.min() - 1e-3


def test_resize_handles_scale_below_one():
    img = ramp(128, 128)
    out = ops.resize_multi(img, 32, 32)
    assert out.shape == (32, 32, 3)
    np.testing.assert_allclose(out.mean(), img.mean(), atol=0.02)


def test_guided_filter_preserves_edges_better_than_gaussian():
    img = np.zeros((64, 64, 3), np.float32)
    img[:, 32:] = 1.0
    guide = ops.guided_filter(img, img, 4, 1e-4)
    blur = ops.gaussian_blur(img, 2.0)
    edge_guide = abs(guide[32, 31, 0] - guide[32, 32, 0])
    edge_blur = abs(blur[32, 31, 0] - blur[32, 32, 0])
    assert edge_guide > edge_blur


def test_detail_enhance_stays_in_range():
    img = ramp(24, 24)
    out = ops.detail_enhance(img, radius=3, eps=1e-3, strength=0.8)
    assert out.min() >= 0.0 and out.max() <= 1.0
    assert out.std() >= img.std()


def test_detail_enhance_zero_strength_is_noop():
    img = ramp()
    np.testing.assert_array_equal(ops.detail_enhance(img, strength=0.0), img)


def test_unsharp_increases_contrast():
    img = np.full((16, 16, 3), 0.5, np.float32)
    img[8, :, :] = 0.9
    out = ops.unsharp(img, 0.6, 1.0, 0.0)
    assert out[8, 8, 0] >= img[8, 8, 0]
    assert out.min() >= 0.0 and out.max() <= 1.0


def test_ycbcr_roundtrip():
    img = ramp(16, 16)
    y, cb, cr = ops.rgb_to_ycbcr(img)
    back = ops.ycbcr_to_rgb(y, cb, cr)
    np.testing.assert_allclose(back, img, atol=1e-4)


def test_chroma_denoise_keeps_luma():
    img = ramp(32, 32)
    noisy = img.copy()
    noisy[..., 0] += 0.03 * np.sin(np.arange(32))[None, :]
    out = ops.chroma_denoise(noisy, 1.5)
    before = ops.luminance(noisy).std()
    after = ops.luminance(out).std()
    assert abs(before - after) < 0.05


def test_percentile_stretch_expands_range():
    img = np.full((32, 32, 3), 0.4, np.float32)
    img[:16] = 0.45
    img[16:] = 0.55
    out = ops.percentile_stretch(img, strength=1.0)
    assert out.max() - out.min() > img.max() - img.min()


def test_adjust_color_saturation_zero_is_gray():
    img = ramp()
    out = ops.adjust_color(img, saturation=0.0)
    np.testing.assert_allclose(out[..., 0], out[..., 2], atol=1e-5)


def test_fit_within_respects_budget():
    w, h = ops.fit_within(8000, 6000, 4_000_000)
    assert w * h <= 4_000_100
    assert abs((w / h) - (8000 / 6000)) < 0.01


def test_fit_within_noop_when_small():
    assert ops.fit_within(100, 100, 1_000_000) == (100, 100)


def test_float_to_u8_clamps():
    arr = np.array([[[-1.0, 0.5, 2.0]]], np.float32)
    out = ops.float_to_u8(arr)
    assert out.dtype == np.uint8
    assert out.tolist() == [[[0, 128, 255]]]


def test_box_blur_of_constant_is_constant():
    img = np.full((20, 20, 3), 0.3, np.float32)
    out = ops.box_blur(img, 3, passes=3)
    np.testing.assert_allclose(out, 0.3, atol=1e-5)


@pytest.mark.parametrize("sigma", [0.5, 1.0, 3.0])
def test_gaussian_radius_mapping_monotonic(sigma):
    assert ops.gaussian_sigma_to_box_radius(sigma) >= 1
    assert ops.gaussian_sigma_to_box_radius(sigma) <= ops.gaussian_sigma_to_box_radius(sigma + 1)
